"""The permeability posterior in the manuscript's coordinates ``xi``.

``d`` and ``forward(xi) = g(xi)`` contain physical observations (K, MPa),
ordered by time, then channel, then observation location. ``R`` is their
fixed measurement-plus-surrogate error covariance. All methods use::

    J(xi) = 0.5 * xi.T @ xi + 0.5 * (g(xi) - d).T @ R^{-1} @ (g(xi) - d)

Static permeability data are assimilated once in the conditional prior:
``x = mu_{x|y} + S_k xi``, ``m = mu_k + B_k xi``, ``xi ~ N(0, I)``.
The likelihood is evaluated directly as ``(g-d).T @ solve(R, g-d)`` using
PyTorch. Observations and predictions are never pre-whitened.
See ``docs/INVERSE.md`` for the equation-to-code mapping.
"""
import hashlib
import json
from pathlib import Path

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
        covariance_blocks = []
        for ti in self.times:
            for ch in self.channels:
                # R_{T,t} = (sigma_T^2 + v_{T,t}) I
                # R_{p,t} = (sigma_p^2 + v_{p,t}) I + q_t 1 1^T
                sigma_c = sigma_t if ch == 0 else sigma_p
                v_c_t = self.cal['block_local_variance'][ti][ch]
                R_c_t = (sigma_c ** 2 + v_c_t) * torch.eye(n, dtype=torch.float64, device=self.device)
                if ch == 1:
                    q_t = self.cal['pressure_common_variance'][ti]
                    R_c_t = R_c_t + q_t * torch.ones_like(R_c_t)
                covariance_blocks.append(R_c_t)
        self.R = torch.block_diag(*covariance_blocks) if covariance_blocks else self.tensor(np.empty((0, 0)))
        observed = self.tensor(case['observed'])[self.times][:, self.channels]
        self.d = observed.flatten().detach()
        self.nobs = len(self.d)
        self.qoi_weights = self.tensor(case['qoi_weights'])
        self.calls = 0

    def cache_key(self):
        """Identity of the numerical posterior, used to prevent reuse of unrelated MAP starts."""
        digest = hashlib.sha256()

        def add(value):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().contiguous().numpy()
            if isinstance(value, np.ndarray):
                digest.update(str((value.shape, value.dtype.str)).encode())
                digest.update(value.tobytes())
            elif isinstance(value, dict):
                for key in sorted(value):
                    add(key)
                    add(value[key])
            else:
                digest.update(json.dumps(value, sort_keys=True).encode())
            digest.update(b'\0')

        add(dict(channels=self.channels, times=self.times, device=str(self.device),
                 torch=str(torch.__version__), stats=self.backend.stats))
        for value in (self.base, self.mean, self.weights, self.d, self.R,
                      self.conditional, self.backend.factor, self.backend.operator.state_dict()):
            add(value)
        package = Path(__file__).resolve().parents[1]
        for path in sorted(package.rglob('*.py')):
            add(str(path.relative_to(package)))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def tensor(self, x):
        return torch.as_tensor(x, device=self.device, dtype=torch.float64)

    def data_misfit(self, predictions):
        """Paper's quadratic form ``(g-d).T R^{-1} (g-d)`` for each member.

        ``solve(R, residual)`` evaluates ``R^{-1} residual`` using PyTorch's
        linear algebra. The extra last axis supports single vectors and batches.
        """
        residual = predictions - self.d
        weighted_residual = torch.linalg.solve(self.R, residual.unsqueeze(-1)).squeeze(-1)
        return (residual * weighted_residual).sum(-1)

    def permeability(self, xi):
        """``xi [B, 4096]`` -> ``m = mu_k + B_k xi [B, 128, 128]`` in log10(m²).

        ``transform`` applies ``x = mu_{x|y} + S_k xi`` without forming the
        dense conditional covariance root. Without static data, ``x = xi``.
        """
        x = transform(xi, self.conditional)
        deviation = (x.float() @ self.backend.factor.T).reshape(-1, 1, 64, 64)
        return self.mean + F.interpolate(deviation, (128, 128), mode='bilinear', align_corners=True)[:, 0]

    def forward(self, xi):
        """Physical predicted observations ``g(xi) [B, nobs]`` for ``xi [B, 4096]``.

        Temperature is in K and pressure in MPa; ordering matches ``d`` and ``R``.
        """
        if self.nobs == 0:
            return xi.new_empty((len(xi), 0))
        self.calls += 1
        values = self.backend.blocks(self.base, self.times, self.permeability(xi), self.weights, self.projected)
        return values[:, :, self.channels].flatten(1)

    def forward_numpy(self, xi, chunk=4):
        """Batched physical ``g(xi)`` as a NumPy array, with members in rows."""
        return np.concatenate([self.forward(self.tensor(batch)).detach().cpu().numpy()
                               for batch in np.array_split(xi, max(1, int(np.ceil(len(xi) / chunk))))])

    def potential(self, xi):
        """Differentiable ``J(xi)`` for a single latent vector; shared by MAP and NUTS.

        The log determinant is constant because ``R`` is fixed.
        """
        g = self.forward(xi[None])[0]
        return .5 * (xi.square().sum() + self.data_misfit(g))
