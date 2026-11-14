"""ES-MDA in the manuscript's physical observation space.

Equinor's ``iterative_ensemble_smoother`` 1.1.0 implements the update::

    xi_j <- xi_j + C_xi_g (C_gg + alpha_l R)^{-1} (d + epsilon_j - g_j)
    epsilon_j ~ N(0, alpha_l R)

``C_xi_g`` and ``C_gg`` are sample covariances with divisor ``N_e - 1``.
The paper uses 256 members, eight updates with ``alpha_l = 8``, all singular
values retained and no localization. Both the gain and the perturbations use
the full ``R``, including the shared pressure-error component.
``PhysicalESMDA`` corrects the pinned library's dense-covariance perturbation
factor orientation; the gain calculation is unchanged (see ``_esmda_reference``).
"""
import time

import numpy as np
import torch


def esmda(problem, samples=256, steps=8, seed=1, progress=None):
    """Return ``(xi [samples, dim], info)`` after ``steps`` assimilations.

    The initial ensemble uses ``default_rng(seed)`` and the observation
    perturbations use the independent stream ``seed + 1``. Passing integer
    ``steps`` to the reference solver sets every inflation factor to ``steps``,
    so ``sum(1 / alpha_l) = 1``. Internally the library stores members in columns.
    """
    from ._esmda_reference import PhysicalESMDA
    if not isinstance(samples, (int, np.integer)) or samples < 2:
        raise ValueError('ES-MDA requires at least two ensemble members')
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError('ES-MDA requires a positive integer number of updates')
    if problem.nobs == 0:
        raise ValueError('ES-MDA requires dynamic observations; use static_members for static-only inference')
    start = time.monotonic()
    rng = np.random.default_rng(seed)
    xi = rng.normal(size=(problem.dim, samples))
    solver = PhysicalESMDA(
        covariance=problem.R.detach().cpu().numpy(),
        observations=problem.d.detach().cpu().numpy(),
        alpha=steps,
        seed=seed + 1,
    )
    history = []
    for iteration in range(steps):
        with torch.no_grad():
            g = problem.forward_numpy(xi.T)
            misfit = problem.data_misfit(problem.tensor(g)).mean() / problem.nobs
        # Y contains physical g(xi), with the same time/channel/location order as d and R.
        solver.prepare_assimilation(Y=g.T, truncation=1.)
        xi = solver.assimilate_batch(X=xi)
        # Retain the archived diagnostic key; its value is mean[(g-d)^T R^{-1}(g-d)] / nobs.
        history.append({'iteration': iteration + 1,
                        'mean_whitened_misfit': float(misfit)})
        if progress:
            progress(history[-1])
    return xi.T, {'seconds': time.monotonic() - start, 'history': history,
                  'inflation': solver.alpha.tolist(), 'observation_space': 'physical',
                  'latent_coordinate': 'xi', 'misfit_evaluated_at': 'before_update'}


def static_members(problem, samples=256, seed=1):
    """Draw ``xi`` for the static-only configuration, with ES-MDA's draw order.

    Applying ``problem.permeability(xi)`` supplies the analytic conditioning.
    Draw parameters x members, then transpose, so the static-only forecast and
    ES-MDA start from identical latent vectors (the 2 October 2026 correction).
    """
    return np.random.default_rng(seed).normal(size=(problem.dim, samples)).T
