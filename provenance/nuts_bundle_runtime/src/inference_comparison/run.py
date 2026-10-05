"""Verbatim scientific definitions extracted from pinned original source."""
import json
import numpy as np
import torch

def finite_json(value):
    if isinstance(value, dict): return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [finite_json(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)): return finite_json(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
    return value

def gradient_check(problem):
    rng = np.random.default_rng(8712)
    y = problem.tensor(rng.normal(size=problem.dim)).requires_grad_(True)
    direction = problem.tensor(rng.normal(size=problem.dim))
    direction /= torch.linalg.norm(direction)
    residual = problem.forward(y[None])[0]-problem.data
    potential = .5*(y.square().sum()+residual.square().sum())
    grad, = torch.autograd.grad(potential, y)
    analytic = (grad*direction).sum().item()
    values = []
    with torch.no_grad():
        for eps in (.1, .03, .01):
            potentials = []
            for sign in (-1, 1):
                yy = y+sign*eps*direction
                error = problem.forward(yy[None])[0]-problem.data
                potentials.append(.5*(yy.square().sum()+error.square().sum()).item())
            fd = (potentials[1]-potentials[0])/(2*eps)
            values.append(dict(step=eps, finite_difference=fd, relative_error=abs(fd-analytic)/max(1., abs(analytic))))
        # Whitening must equal the same dense physical covariance quadratic.
        unwhitened = problem.data@problem.cholesky.T
        q_dense = unwhitened@torch.linalg.solve(problem.cholesky@problem.cholesky.T, unwhitened)
        np.testing.assert_allclose(q_dense.item(), problem.data.square().sum().item(), rtol=1e-9)
    result = dict(analytic=analytic, comparisons=values, potential=float(potential.detach()),
                  gradient_norm=float(torch.linalg.norm(grad)))
    if not torch.isfinite(grad).all() or min(v['relative_error'] for v in values) > .05:
        raise RuntimeError('FP32 surrogate gradient check failed: '+json.dumps(result))
    return result
