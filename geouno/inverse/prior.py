"""Gaussian permeability prior and analytic static conditioning.

The manuscript first defines ``m(z) = M0 + I L_* z`` with ``z ~ N(0, I)``.
Static measurements ``y = A_k m + e`` condition these latent coordinates,
denoted ``x`` in the Gaussian conditioning equations. With ``A = A_k I L_*``::

    mu_{x|y} = A.T (A A.T + sigma_k^2 I)^{-1} (y - A_k M0)
    Sigma_{x|y} = I - A.T (A A.T + sigma_k^2 I)^{-1} A
    x = mu_{x|y} + S_k xi,  xi ~ N(0, I),  S_k S_k.T = Sigma_{x|y}

We apply the symmetric root as ``S_k = I - U U.T`` without storing a dense
4096 x 4096 matrix. For ``A A.T = Q diag(lambda) Q.T``,
``U = A.T Q diag(sqrt(c))``, where ``s = sqrt(lambda + sigma_k^2)`` and
``c = 1 / (s (s + sigma_k))``. The identity
``2c - c^2 lambda = 1 / (lambda + sigma_k^2)`` gives the required covariance.
The returned dictionary retains its existing storage keys (``mean``, ``root``, ``H``).
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
    ``observed`` their values. Sampling: ``x = mean + xi - (xi @ root) @ root.T``,
    ``xi ~ N(0, I)``; ``mean = mu_{x|y}``, ``root = U``, ``H = A``.
    Each static datum is used once; dynamic inversion then works with the
    conditional prior and does not repeat this likelihood.
    """
    device = factor.device
    dtype = torch.float64
    masks = masks.to(device=device, dtype=dtype)
    weights = masks / masks.sum((-2, -1), keepdim=True)
    interpolation = interpolation_matrix(device, dtype)
    projected_blocks = torch.einsum('pxy,xi,yj->pij', weights, interpolation, interpolation).reshape(len(masks), 4096)
    A = amplitude * (projected_blocks @ factor.double())
    offset = (weights * mean128.to(device=device, dtype=dtype)).sum((-2, -1))
    gram = A @ A.T
    ev, Q = torch.linalg.eigh(gram)
    ev = ev.clamp_min(0)
    rhs = observed.to(device=device, dtype=dtype) - offset
    latent_mean = A.T @ torch.linalg.solve(gram + sigma ** 2 * torch.eye(len(masks), device=device, dtype=dtype), rhs)
    s = torch.sqrt(ev + sigma ** 2)
    coefficient = 1 / (s * (s + sigma))
    root = (A.T @ Q) * torch.sqrt(coefficient)[None]
    return dict(mean=latent_mean.float(), root=root.float(), H=A.float(), offset=offset.float(), sigma=float(sigma))


def transform(xi, posterior):
    """Apply ``x = mu_{x|y} + S_k xi`` to row vectors; without static data, ``x = xi``."""
    if posterior is None:
        return xi
    root = posterior['root']
    return posterior['mean'][None] + xi - (xi @ root) @ root.T


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
