"""Physical-unit GeoUNO evaluation used by forecasting and inversion.

Sources: ``operator_final_20260930/backend.py`` (portable bundle version) and
``operator_final_20260930/inverse_backend.py`` (fixed-basis observation contractions).

Two evaluation paths are provided, exactly as in the reported experiments:

* :meth:`Backend.predict_real` returns dense ``[B, 2, 160, 128]`` fields (``Delta T`` [K],
  ``Delta p`` [MPa]) in FP32. It is used for ten-year forecasts and well-block scores.
* :meth:`Backend.blocks` returns block-averaged observations directly from the basis
  coefficients, contracting the fixed basis with the observation masks in FP64. It is
  mathematically identical to averaging the dense output and is used in every likelihood.

The learned coordinate functions do not depend on the input, so the basis is evaluated
once on the sensor grid and cached.
"""
import json
from pathlib import Path

import numpy as np
import torch

from .inputs import SamplingAdapter, realized_inputs
from .model import CHECKPOINT_SHA256, configure_inference, load_geouno


def read_json(path):
    return json.loads(Path(path).read_text())


class Backend:
    """Frozen GeoUNO in physical units plus the inverse-prior factor.

    Parameters
    ----------
    model_path : checkpoint file (``ed043489...``).
    stats : dict or path of the training normalization statistics.
    prior_factor_path : optional ``prior_factor.npy`` (4096 x 4096, float32) for inversion.
    device : torch device string.
    threads : CPU threads (``torch.set_num_threads``); ``None`` leaves the setting unchanged.
    """

    def __init__(self, model_path, stats, prior_factor_path=None, device='cpu', threads=None,
                 expected_sha256=CHECKPOINT_SHA256):
        configure_inference(threads)
        self.device = torch.device(device)
        self.stats = read_json(stats) if isinstance(stats, (str, Path)) else stats
        operator = load_geouno(model_path, expected_sha256=expected_sha256)
        self.net = SamplingAdapter(operator, self.stats).to(self.device).eval().requires_grad_(False)
        self.operator = self.net.bundle.joint
        self.M_mean = float(self.stats['M_mean'])
        self.M_std = float(self.stats['M_std'])
        self.factor = None
        if prior_factor_path is not None:
            self.factor = torch.tensor(np.load(prior_factor_path), device=self.device)
        self.mean = torch.tensor(self.stats['fields_mean'], device=self.device).float()[None, :, None, None]
        self.std = torch.tensor(self.stats['fields_std'], device=self.device).float()[None, :, None, None]
        self.evaluations = 0
        with torch.no_grad():
            self.fixed_basis = tuple(v.detach() for v in self.operator.basis())
        original_basis = self.operator.basis
        self.operator.basis = lambda query=None: self.fixed_basis if query is None else original_basis(query)

    def base_tensor(self, base):
        """Move a stored ``[11, 13, 160, 128]`` base-input array to the backend device."""
        return torch.as_tensor(np.asarray(base, dtype=np.float32), device=self.device)

    def predict_real(self, base, ti, m):
        """Dense physical fields at stored time index ``ti`` for log10-permeability ``m``.

        ``base`` is the device tensor from :meth:`base_tensor`; ``m`` is ``[128, 128]`` or
        ``[B, 128, 128]`` log10 permeability (m^2) on the reservoir grid.
        """
        x = realized_inputs(base[ti][None], m, self.M_mean, self.M_std)
        self.evaluations += len(x)
        return self.net.mean_modes(x)[0].float() * self.std + self.mean

    def observation_basis(self, weights):
        """Contract the cached basis and mean shape with ``[P, 160, 128]`` block weights (FP64)."""
        basis, mean = self.fixed_basis
        w = weights.reshape(len(weights), -1).double()
        return torch.einsum('crn,pn->crp', basis.double(), w), torch.einsum('cn,pn->cp', mean.double(), w)

    def blocks(self, base, times, m, weights, projected=None):
        """Block-averaged ``[B, time, channel, point]`` predictions for ``m`` at ``times``."""
        if m.ndim == 2:
            m = m[None]
        fields = []
        scalars = []
        for ti in times:
            x = realized_inputs(base[ti][None], m, self.M_mean, self.M_std)
            f, s = self.net.encoder(x)
            fields.append(f)
            scalars.append(s)
        f, s = torch.cat(fields), torch.cat(scalars)
        coefficients = []
        levels = []
        for first in range(0, len(f), 4):
            c, l = self.operator.branch(f[first:first + 4], s[first:first + 4])
            coefficients.append(c)
            levels.append(l)
        c, l = torch.cat(coefficients).double(), torch.cat(levels).double()
        basis, mean = self.observation_basis(weights) if projected is None else projected
        v = torch.einsum('bcr,crp->bcp', c, basis) + mean[None]
        t = (v[:, 0] + l[:, 0:1]) * self.operator.scales[0].double()
        p = v[:, 1] * self.operator.pressure_residual_scale + l[:, 1:2] * self.operator.scales[1].double()
        self.evaluations += len(f)
        return torch.stack((t, p), 1).reshape(len(times), len(m), 2, len(weights)).permute(1, 0, 2, 3)
