"""Whitened permeability posterior shared by every inference algorithm (Methods Eq. 5).

Sources: ``inference_comparison/geothermal.py`` (``Problem``) and
``operator_final_20260930/inverse_backend.py`` (``Problem`` subclass with configurable
channels, times, noise and static conditioning). The two constructors are merged here;
the final state and every arithmetic operation are unchanged.

The negative log posterior in whitened coordinates ``y`` is::

    J(y) = 0.5 ||y||^2 + 0.5 || C^{-1} (g(y) - d) ||^2,     R = C C^T,

where ``g`` stacks predicted block temperatures/pressures at the observation times and
``R`` is block diagonal over (time, channel). Each block is
``(s_local^2 + sigma^2) I`` plus, for pressure, a rank-one common component
``s_common^2 1 1^T``; ``s_local`` and ``s_common`` are validation-fitted surrogate-error
variances (``calibration.json``) and ``sigma`` is 0.5 K or 0.25 MPa.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .prior import condition, transform


class Problem:
    """Posterior for one test reservoir and one observation configuration.

    Parameters
    ----------
    backend : :class:`geouno.backend.Backend` with a prior factor loaded.
    case : mapping with the case arrays (``np.load`` of a case file).
    base : ``[11, 13, 160, 128]`` stored base inputs of the reservoir.
    calibration : dict loaded from ``calibration.json``.
    channels : dynamic channels (0 temperature, 1 pressure); empty for static-only.
    static : apply analytic static permeability conditioning.
    times : stored time indices of the dynamic observations (0..6 reach 4.02 years).
    sigma_t, sigma_p : instrument noise SD for temperature [K] and pressure [MPa].
    """

    def __init__(self, backend, case, base, calibration, channels=(0, 1), static=True,
                 times=tuple(range(7)), sigma_t=.5, sigma_p=.25):
        self.backend = backend
        self.device = backend.device
        self.case = case
        self.base = backend.base_tensor(base)
        self.mean = self.tensor(case['prior_mean']).float()
        self.weights = self.tensor(case['weights'])
        self.dim = 4096
        self.conditional = None
        if static:
            conditional = condition(self.mean, backend.factor, self.tensor(case['k_weights']),
                                    self.tensor(case['k_observed']), sigma=.1)
            self.conditional = {k: (v.double() if isinstance(v, torch.Tensor) else v)
                                for k, v in conditional.items()}
        self.cal = calibration
        self.channels = list(channels)
        self.times = list(times)
        self.sigma_t = sigma_t
        self.sigma_p = sigma_p
        self.projected = backend.observation_basis(self.weights)
        n = len(self.weights)
        covs = []
        for ti in self.times:
            for ch in self.channels:
                variance = self.cal['block_local_variance'][ti][ch] + (sigma_t if ch == 0 else sigma_p) ** 2
                cov = variance * torch.eye(n, dtype=torch.float64, device=self.device)
                if ch == 1:
                    cov = cov + self.cal['pressure_common_variance'][ti] * torch.ones_like(cov)
                covs.append(cov)
        self.cholesky = torch.linalg.cholesky(torch.block_diag(*covs)) if covs else self.tensor(np.empty((0, 0)))
        observed = self.tensor(case['observed'])[self.times][:, self.channels]
        self.data = self.whiten(observed.flatten()[None])[0].detach() if covs else self.tensor([])
        self.nobs = len(self.data)
        self.qoi_weights = self.tensor(case['qoi_weights'])
        self.calls = 0

    def tensor(self, x):
        return torch.as_tensor(x, device=self.device, dtype=torch.float64)

    def whiten(self, predictions):
        return torch.linalg.solve_triangular(self.cholesky, predictions.T, upper=False).T

    def permeability(self, y):
        """Whitened coordinates ``[B, 4096]`` -> log10 permeability ``[B, 128, 128]``."""
        z = transform(y, self.conditional)
        dev = (z.float() @ self.backend.factor.T).reshape(-1, 1, 64, 64)
        return self.mean + F.interpolate(dev, (128, 128), mode='bilinear', align_corners=True)[:, 0]

    def forward(self, y):
        """Whitened predicted observations ``C^{-1} g(y)`` for ``y`` of shape ``[B, 4096]``."""
        self.calls += 1
        values = self.backend.blocks(self.base, self.times, self.permeability(y), self.weights, self.projected)
        return self.whiten(values[:, :, self.channels].flatten(1))

    def forward_numpy(self, y, chunk=4):
        return np.concatenate([self.forward(self.tensor(batch)).detach().cpu().numpy()
                               for batch in np.array_split(y, max(1, int(np.ceil(len(y) / chunk))))])

    def potential(self, y):
        """``J(y)`` for a single ``[4096]`` vector (differentiable)."""
        residual = self.forward(y[None])[0] - self.data
        return .5 * (y.square().sum() + residual.square().sum())
