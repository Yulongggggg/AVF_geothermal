"""Physical input encoding, permeability resampling and point readouts.

Sources: ``temperature_revision/inference.py`` (``PhysicalInputEncoder``, ``point_readout``),
``det_sampler.py`` (``field_on_grid``), ``history_matching_revision/inputs.py``
(``realized_inputs``) and ``direct_operator/inference.py`` (``SamplingAdapter``).

Grids
-----
* Output/branch grid: 160 x 128 cells, ``y`` in [-500, 500] m (rows), ``x`` in [-400, 400] m.
  Rows 16..143 (``|y| <= 400`` m) are the reservoir; the rest are cap and bed layers.
* Permeability grid: 128 x 128 over the reservoir square [-400, 400]^2 m.
* Legacy 13-channel input (the stored "base" arrays): channel 0 is standardized
  log10 permeability, channels 5-6 are producer/injector distances divided by 400 m and
  channels 7-12 hold standardized scalar controls broadcast over the grid.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

H, W = 160, 128


def field_on_grid(t, span_y=400.0):
    """Resample ``[B, C, n, n]`` fields on [-400, 400]^2 to the ``[B, C, 160, 128]`` grid.

    Cap and bed rows (``|y| > 400`` m) copy the nearest reservoir row.
    """
    yq = torch.linspace(-500, 500, H, device=t.device).clamp(-span_y, span_y) / span_y
    xq = torch.linspace(-400, 400, W, device=t.device) / 400.0
    gy, gx = torch.meshgrid(yq, xq, indexing="ij")
    grid = torch.stack([gx, gy], -1)[None].expand(t.shape[0], -1, -1, -1).to(t.dtype)
    return F.grid_sample(t, grid, mode="bilinear", align_corners=True)


def realized_inputs(base, logk128, mean, std):
    """Replace only log-permeability in a legacy base input with ``logk128`` (``[B,128,128]``)."""
    fields = torch.as_tensor(logk128, device=base.device, dtype=torch.float32)
    if fields.ndim == 2:
        fields = fields[None]
    result = base.float().expand(len(fields), -1, -1, -1).clone()
    result[:, 0] = (field_on_grid(fields[:, None])[:, 0] - mean) / std
    return result


def point_readout(fields, positions_m):
    """Bilinear point values on x=[-400,400], y=[-500,500], preserving gradients."""
    if positions_m.shape != (len(fields), 2):
        raise ValueError("positions_m must have shape (batch,2), in x/y metres")
    grid = positions_m.to(device=fields.device, dtype=fields.dtype) / fields.new_tensor([400., 500.])
    if (grid.abs() > 1).any():
        raise ValueError("Point lies outside the output grid")
    return torch.nn.functional.grid_sample(fields, grid[:, None, None, :], mode="bilinear",
                                           align_corners=True)[:, :, 0, 0]


class PhysicalInputEncoder(nn.Module):
    """Convert a legacy 13-channel input into GeoUNO's six fields and five controls.

    Fields: ``log10 k + 13.5``, reservoir mask, ``x/400``, ``y/500``, producer distance / 400,
    injector distance / 400. Controls: ``q/3.5``, ``(T_inj - 303)/10``, ``(p_prod - 20)/10``
    (absolute producer bottom-hole pressure in MPa), ``(log10 t - 6)/2`` (t in s),
    ``(depth - 3.06)/0.6`` (depth in km).
    """

    def __init__(self, stats):
        super().__init__()
        self.stats = stats
        coeff = np.polyfit(stats["tau_norm"], np.log10(stats["times_s"]), 1)
        if np.max(np.abs(np.polyval(coeff, stats["tau_norm"]) - np.log10(stats["times_s"]))) > 1e-6:
            raise ValueError("Legacy tau must be an affine normalization of log time")
        self.register_buffer("tau_to_log10", torch.tensor(coeff, dtype=torch.float32))

    def forward(self, x):
        if x.shape[1] != 13:
            raise ValueError("This revision requires the 13-channel depth-aware input")
        st = self.stats
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(torch.linspace(-500, 500, h, device=x.device),
                                torch.linspace(-400, 400, w, device=x.device), indexing="ij")
        coords = torch.stack([(yy.abs() <= 400).float(), xx / 400, yy / 500])[None].expand(b, -1, -1, -1)
        m = x[:, 0:1].float() * st["M_std"] + st["M_mean"]
        field = torch.cat([m + 13.5, coords, x[:, 5:7].float()], dim=1)
        scalar = x[:, :, 0, 0].float()
        q = scalar[:, 7] * st["scal_std"][0] + st["scal_mean"][0]
        tinj = scalar[:, 8] * st["scal_std"][1] + st["scal_mean"][1]
        pp = scalar[:, 10] * st["pp_norm"][1] + st["pp_norm"][0]
        depth = scalar[:, 12] * st["depth_norm"][1] + st["depth_norm"][0]
        logt = scalar[:, 11] * self.tau_to_log10[0] + self.tau_to_log10[1]
        cond = torch.stack([q / 3.5, (tinj - 303) / 10, (pp - 20) / 10, (logt - 6) / 2, (depth - 3.06) / .6], dim=1)
        return field, cond


class SamplingAdapter(nn.Module):
    """Differentiable legacy-input interface used by the inversion backend.

    ``mean_modes(x)[0]`` returns fields standardized by the training mean/std; the backend
    converts them back to physical units. This round trip is kept because it is the exact
    computation behind every reported inversion.
    """

    def __init__(self, operator, stats):
        super().__init__()
        self.bundle = _Bundle(operator)
        self.encoder = PhysicalInputEncoder(stats)
        self.register_buffer('scale', torch.tensor(stats['fields_std'])[None, :, None, None])
        self.register_buffer('offset', torch.tensor(stats['fields_mean'])[None, :, None, None])

    def physical_fields(self, x):
        return self.bundle(*self.encoder(x)).float()

    def mean_modes(self, x):
        fields = self.physical_fields(x)
        empty = fields.new_zeros((len(fields), 0, *fields.shape[-2:]))
        return (fields - self.offset) / self.scale, empty, empty

    def forward(self, x):
        return self.mean_modes(x)[0]


class _Bundle(nn.Module):
    """Holds the operator under the attribute name ``joint`` (as in the original bundle)."""

    def __init__(self, operator):
        super().__init__()
        self.joint = operator

    def forward(self, field, scalar):
        return self.joint(field, scalar)
