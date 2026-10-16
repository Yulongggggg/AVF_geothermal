"""Simplified GeoUNO candidates (see common.py). Inner products use <f,g> = N^-1 sum_i f_i g_i on the 160x128 grid;
the POD functions are orthonormal under it, the learned functions are centred and unit-normalised but not orthogonal.
Output: u = b + s [ mu + sum_r a_r psi_r + sum_k c_k phi_k ]. No correction MLP; FiLM kept; P0 removed.
This file is a superset of simple_geouno_20261015/model.py: resid64 / joint64 / chan64 behave exactly as there.
  *_p1   the 12 parameter-free near-well permeability statistics (P1) are appended to the pressure-level input
  resid64g*  the learned-function coefficients are multiplied by a gain (2 x 64) initialised at zero
"""

import torch
from torch import nn
from torch.nn import functional as F
from .geouno import GeoUNOBasisHead
from .direct import DirectFieldOperator

RIDGE = 1e-6
VARIANTS = ("resid64", "joint64", "chan64", "resid64_p1", "resid64g_p1")


def spd_solve(G, rhs):
    """Solve (G + ridge I) x = rhs in float64. G: (2, k, k); rhs: (B, 2, k)."""
    k = G.shape[-1]
    A = (G + RIDGE * torch.eye(k, device=G.device, dtype=G.dtype)).double()
    return (
        torch.linalg.solve(A[None].expand(len(rhs), -1, -1, -1), rhs.double().unsqueeze(-1)).squeeze(-1).to(rhs.dtype)
    )


class FinalGeoUNO(GeoUNOBasisHead):
    def __init__(self, scales, pressure_residual_scale, variant="resid64_p1", pod_rank=1024, learned_rank=64, width=32):
        assert variant in VARIANTS
        super().__init__(scales, pressure_residual_scale, pod_rank=pod_rank, learned_rank=learned_rank, width=width)
        self.variant = variant
        self.method = variant.split("_")[0].replace("resid64g", "resid64")  # resid64 / joint64 / chan64
        self.well_inputs = variant.endswith("_p1")
        self.config.update(architecture="geouno_simple", variant=variant)
        del self.coefficient_delta, self.geology_readout  # no correction MLP, no P0
        self.level_head = nn.Sequential(
            nn.Linear(192 + 192 + 256 + 5 + (12 if self.well_inputs else 0), 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        if variant.startswith("resid64g"):
            self.learned_gain = nn.Parameter(torch.zeros(2, learned_rank))
        if variant == "chan64":
            self.temperature_head = nn.Conv2d(self.temperature_head.in_channels, 1 + learned_rank, 1)
            self.pressure_head = nn.Conv2d(self.pressure_head.in_channels, 1 + learned_rank, 1)

    def forward(self, field, scalar, query=None, return_parts=False):
        self.branch_chunk = 10**9  # no gradient checkpointing
        if not self.training:
            return super().forward(field, scalar, query, return_parts)
        coeff, level = self.branch(field, scalar)
        with torch.autocast(field.device.type, enabled=False):
            basis, mean = self.basis(None)
            values = torch.einsum("bcr,crn->bcn", coeff.float(), basis) + mean[None]
            t = (values[:, 0] + level[:, 0:1]) * self.scales[0]
            p = values[:, 1] * self.pressure_residual_scale + level[:, 1:2] * self.scales[1]
            return torch.stack((t, p), 1).reshape(-1, 2, 160, 128)

    def branch(self, field, scalar):
        if field.shape[-2:] != (160, 128):
            raise ValueError("Sample the input function on the fixed 160 x 128 branch sensors.")
        condition = self.condition(scalar)
        h, skips = field, []
        for i, block in enumerate(self.enc):
            h = block(F.avg_pool2d(h, 2) if i else h, condition)
            skips.append(h)
        pos = F.interpolate(field[:, 2:4], size=h.shape[-2:], mode="bilinear", align_corners=True)
        shape = h.shape
        h = self.global_mix((h + self.position(pos)).flatten(2).transpose(1, 2)).transpose(1, 2).reshape(shape)
        features = torch.cat((h.mean((-2, -1)), h.amax((-2, -1)), scalar, self.pressure_grid(h).flatten(1)), 1)
        ht = self.decode(h, skips, condition, self.temperature_decoder)
        hp = self.decode(h, skips, condition, self.pressure_decoder)
        with torch.autocast(field.device.type, enabled=False):
            if self.well_inputs:  # P1: 12 near-well statistics
                features = torch.cat((features.float(), DirectFieldOperator.well_features(field)), 1)
            mt, mp = self.temperature_head(ht.float()), self.pressure_head(hp.float())  # B, 1(+L), H, W
            raw = torch.cat((mt[:, :1], mp[:, :1]), 1).flatten(2)  # B, 2, N
            raw_mean = raw.mean(-1, keepdim=True)
            hc = raw - raw_mean - self.mean_function.flatten(2).squeeze(1)[None]  # centred decoder maps
            pod = self.pod.flatten(2)
            n = raw.shape[-1]
            a = torch.einsum("bcn,crn->bcr", hc, pod) / n  # POD coefficients
            phi = self.basis(None)[0][:, -self.config["learned_rank"] :]  # 2, L, N
            if self.method == "chan64":  # C: dedicated decoder channels
                g = torch.stack((mt[:, 1:], mp[:, 1:]), 1).flatten(3)  # B, 2, L, N
                c = (g * phi[None]).sum(-1) / n
            else:
                Gff = torch.einsum("ckn,cjn->ckj", phi, phi) / n  # 2, L, L
                if self.method == "resid64":  # A: residual projection
                    r = hc - torch.einsum("bcr,crn->bcn", a, pod)
                    c = spd_solve(Gff, torch.einsum("bcn,ckn->bck", r, phi) / n)
                else:  # B: joint least squares
                    Gpf = torch.einsum("crn,ckn->crk", pod, phi) / n  # 2, R, L
                    S = Gff - torch.einsum("crk,crj->ckj", Gpf, Gpf)  # Schur complement
                    rhs = torch.einsum("bcn,ckn->bck", hc, phi) / n - torch.einsum("bcr,crk->bck", a, Gpf)
                    c = spd_solve(S, rhs)
                    a = a - torch.einsum("crk,bck->bcr", Gpf, c)
            if hasattr(self, "learned_gain"):
                c = c * self.learned_gain[None]
            levels = torch.cat((raw_mean[:, 0], self.level_head(features.float())), 1)
        return torch.cat((a, c), -1), levels


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["model_config"].copy()
    assert cfg.pop("architecture") == "geouno_simple"
    net = FinalGeoUNO(**cfg)
    net.load_state_dict(ck["model"], strict=True)
    return net.eval()
