"""Ensemble smoother with multiple data assimilation (ES-MDA).

Source: ``inference_comparison/algorithms.py`` (``esmda``). The update itself is the
reference implementation in Equinor's ``iterative_ensemble_smoother`` 1.1.0 (GPL-3.0,
installed separately): 256 members, eight updates with inflation factors alpha_i = 8
(sum of reciprocals = 1), all singular values retained, no localization. Observations are
pre-whitened, so the observation covariance passed to the smoother is the identity.
"""
import time

import numpy as np
import torch


def esmda(problem, samples=256, steps=8, seed=1, progress=None):
    """Run ES-MDA on ``problem``; return ``(y [samples, 4096], info)``.

    The initial ensemble is ``default_rng(seed).normal(size=(4096, samples))``; the
    smoother's own perturbation stream uses ``seed + 1``.
    """
    from iterative_ensemble_smoother import ESMDA
    start = time.monotonic()
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(problem.dim, samples))
    solver = ESMDA(np.ones(problem.nobs), problem.data.cpu().numpy(), alpha=steps, seed=seed + 1)
    history = []
    for iteration in range(steps):
        with torch.no_grad():
            pred = problem.forward_numpy(x.T)
        solver.prepare_assimilation(Y=pred.T, truncation=1.)
        x = solver.assimilate_batch(X=x)
        history.append({'iteration': iteration + 1,
                        'mean_whitened_misfit': float(np.mean((pred - problem.data.cpu().numpy()) ** 2))})
        if progress:
            progress(history[-1])
    return x.T, {'seconds': time.monotonic() - start, 'history': history, 'inflation': solver.alpha.tolist()}


def static_members(problem, samples=256, seed=1):
    """Initial members for the static-only (``k``) configuration.

    Draws exactly the ES-MDA initial ensemble (parameters x members, then transposed), so
    the static-only forecast and ES-MDA start from identical latent vectors. This is the
    2 October 2026 correction of the historical ``(members, parameters)`` draw order.
    """
    return np.random.default_rng(seed).normal(size=(problem.dim, samples)).T
