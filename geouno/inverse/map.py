"""Maximum a posteriori estimate with automatic-differentiation gradients and L-BFGS-B.

Source: ``operator_final_20260930/inverse_run.py`` (``map_optimize``). Three starts (the
origin and two standard-normal draws from ``default_rng(seed)``); the start with the
smallest posterior objective is retained. Options used for the paper:
``maxiter=1000, maxfun=20000, maxls=40, gtol=1e-4, ftol=1e-10, maxcor=10``.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

PAPER_OPTIONS = dict(maxiter=1000, maxfun=20000, maxls=40, gtol=.0001, ftol=1e-10, maxcor=10)


def map_optimize(problem, seed, options=None, dest=None):
    """Return ``(y [1, 4096], info)``.

    If ``dest`` is given, each finished start is stored there (``start_<i>.npz`` and
    ``start_<i>.json``) and reused only when the posterior, seed and optimizer options match.
    Custom problems must implement ``cache_key()`` to enable persistent restart files.
    """
    options = dict(PAPER_OPTIONS if options is None else options)
    if dest is not None:
        dest = Path(dest)
        identity = dict(posterior=problem.cache_key(), seed=int(seed), options=options)
        protocol = dest / 'protocol.json'
        if protocol.exists():
            if json.loads(protocol.read_text()) != identity:
                raise ValueError('MAP restart protocol differs; use a new output directory')
        else:
            if dest.exists() and any(dest.glob('start_*')):
                raise ValueError('MAP restart files lack a protocol identity; use a new output directory')
            dest.mkdir(parents=True, exist_ok=True)
            temporary = protocol.with_suffix('.tmp')
            temporary.write_text(json.dumps(identity, indent=2) + '\n')
            os.replace(temporary, protocol)
    starts = [np.zeros(4096)] + list(np.random.default_rng(seed).normal(size=(2, 4096)))
    summaries = []
    solutions = []
    for i, initial in enumerate(starts):
        if dest is not None:
            file, meta = Path(dest) / f'start_{i}.npz', Path(dest) / f'start_{i}.json'
            if file.exists() and meta.exists():
                with np.load(file, allow_pickle=False) as saved:
                    y = saved['y'].copy()
                info = json.loads(meta.read_text())
                if y.shape != (4096,) or not np.isfinite(y).all() or not np.isfinite(info['objective']):
                    raise ValueError(f'Invalid cached MAP start: {file}')
                solutions.append(y)
                summaries.append(info)
                continue
        begin = time.monotonic()

        def objective(z):
            y = problem.tensor(z).requires_grad_(True)
            r = problem.forward(y[None])[0] - problem.data
            u = .5 * (y.square().sum() + r.square().sum())
            g, = torch.autograd.grad(u, y)
            assert torch.isfinite(u) and torch.isfinite(g).all()
            return u.item(), g.detach().cpu().numpy()

        r = minimize(objective, initial, jac=True, method='L-BFGS-B', options=options)
        info = dict(start=i, success=bool(r.success), message=str(r.message), objective=float(r.fun),
                    gradient_inf=float(abs(r.jac).max()), iterations=int(r.nit), evaluations=int(r.nfev),
                    seconds=time.monotonic() - begin)
        if dest is not None:
            Path(dest).mkdir(parents=True, exist_ok=True)
            tmp = file.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, y=r.x)
            os.replace(tmp, file)
            meta.write_text(json.dumps(info, indent=2) + '\n')
        solutions.append(r.x)
        summaries.append(info)
    best = int(np.argmin([s['objective'] for s in summaries]))
    return solutions[best][None], dict(starts=summaries, selected_start=best, success=summaries[best]['success'],
                                       seconds=sum(s['seconds'] for s in summaries),
                                       evaluations=sum(s['evaluations'] for s in summaries), point_estimate=True)
