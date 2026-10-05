"""Latent Gaussian permeability prior and exact static conditioning (Methods Eq. 4).

Source: ``history_matching_dense/conditioning.py`` (``condition``, ``transform``,
``common_mode_quadratic``).

Permeability is ``m(z) = M0 + I(L z)`` with ``z`` standard normal on a 64 x 64 latent grid,
``L`` the training-fitted prior factor (``prior_factor.npy``, 4096 x 4096) and ``I``
bilinear interpolation to the 128 x 128 reservoir grid. Local log-permeability data
``d_k = A_k m + e``, ``e ~ N(0, sigma_k^2 I)``, are assimilated once, analytically::

    z_bar = H^T (H H^T + sigma^2 I)^{-1} (d_k - A_k M0),     H = A_k I L
    S S^T = I - H^T (H H^T + sigma^2 I)^{-1} H

and ``z = z_bar + S y`` with ``y ~ N(0, I)``. The symmetric root ``S = I - R R^T`` uses
``R = H^T Q diag(c)^{1/2}``, ``H H^T = Q diag(lambda) Q^T`` and
``c = 1 / (s (s + sigma))``, ``s = sqrt(lambda + sigma^2)``; one checks that
``2c - c^2 lambda = 1 / (lambda + sigma^2)``, so ``S S^T`` equals the posterior covariance.
"""
import torch
import torch.nn.functional as F


def interpolation_matrix(device=None, dtype=torch.float64):
    """``[128, 64]`` matrix of 1-D linear interpolation with aligned corners."""
    return F.interpolate(torch.eye(64, device=device, dtype=dtype).reshape(64, 1, 64), size=128,
                         mode='linear', align_corners=True)[:, 0].T


def condition(mean128, factor, masks, observed, sigma=.1, amplitude=1.):
    """Return latent conditional mean and a symmetric covariance-root factor.

    ``mean128`` is the prior mean ``M0`` (128 x 128), ``factor`` the 4096 x 4096 prior factor,
    ``masks`` the ``[P, 128, 128]`` averaging blocks of the ``P`` local permeability data and
    ``observed`` their values. Sampling: ``z = mean + y - (y @ root) @ root.T``,
    ``y ~ N(0, I)``. Each static datum is used once; dynamic inversion then works with the
    conditional prior and does not repeat this likelihood.
    """
    device = factor.device
    dtype = torch.float64
    masks = masks.to(device=device, dtype=dtype)
    weights = masks / masks.sum((-2, -1), keepdim=True)
    B = interpolation_matrix(device, dtype)
    A = torch.einsum('pxy,xi,yj->pij', weights, B, B).reshape(len(masks), 4096)
    H = amplitude * (A @ factor.double())
    offset = (weights * mean128.to(device=device, dtype=dtype)).sum((-2, -1))
    gram = H @ H.T
    ev, Q = torch.linalg.eigh(gram)
    ev = ev.clamp_min(0)
    rhs = observed.to(device=device, dtype=dtype) - offset
    latent_mean = H.T @ torch.linalg.solve(gram + sigma ** 2 * torch.eye(len(masks), device=device, dtype=dtype), rhs)
    s = torch.sqrt(ev + sigma ** 2)
    coefficient = 1 / (s * (s + sigma))
    root = (H.T @ Q) * torch.sqrt(coefficient)[None]
    return dict(mean=latent_mean.float(), root=root.float(), H=H.float(), offset=offset.float(), sigma=float(sigma))


def transform(y, posterior):
    """Map whitened coordinates ``y`` (rows) to prior latent coordinates ``z``."""
    if posterior is None:
        return y
    root = posterior['root']
    return posterior['mean'][None] + y - (y @ root) @ root.T


def common_mode_quadratic(error, diagonal_sigma, common_sigma):
    """Quadratic form for diagonal-plus-common-mode covariance (Sherman-Morrison).

    ``error`` has shape ``(..., time, point)``, ``diagonal_sigma`` ``(time, point)`` and
    ``common_sigma`` ``(time,)``; times are independent.
    """
    # Dense arrays can otherwise lose precision when subtracting a common mode.
    error = error.double()
    inv = diagonal_sigma.double().square().reciprocal()
    common_variance = common_sigma.double().square()
    diagonal = (error.square() * inv).sum(-1)
    correction = (error * inv).sum(-1).square() * common_variance / (1 + common_variance * inv.sum(-1))
    return (diagonal - correction).sum(-1)
