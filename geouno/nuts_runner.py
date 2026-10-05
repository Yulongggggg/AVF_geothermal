"""NUTS reference runs for the six prespecified reservoirs (``test_01``-``test_06``).

The paper ran four chains per reservoir on one NVIDIA RTX A4000 (about 3.6 h per chain;
every retained transition reached the depth-7 cap). The released chains can be checked
without rerunning: :func:`verify_released` recomputes the convergence diagnostics from all
4 x 1,024 draws and rescores the balanced 256-member forecast.
"""
import json
import time
from pathlib import Path

import numpy as np

from . import artifacts
from .inverse import balanced_draws, convergence, forecast, headline, nuts, write_forecast
from .inverse.nuts import PAPER_PROTOCOL
from .monitoring.protocol import ENSEMBLE_SEED_OFFSET
from .tasks import load_backend, problem_for, _finite

NUTS_CASES = tuple(f'test_{i:02d}' for i in range(1, 7))


def run_chain(label, chain, out, device='cuda:0', threads=None, root=None, backend=None, log=print):
    """Run formal chain ``chain`` (0-3) for ``label``; writes ``latent_samples.npz`` and ``diagnostics.json``."""
    if label not in NUTS_CASES or chain not in range(4):
        raise ValueError('NUTS was registered for test_01..test_06, chains 0..3')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    reservoirs = artifacts.Reservoirs(root)
    backend = backend or load_backend(device, threads, root)
    problem, _ = problem_for(backend, reservoirs, label, 'multiwell_opt17')
    seed = reservoirs.entry(label)['observation_seed'] + ENSEMBLE_SEED_OFFSET + chain
    began = time.monotonic()
    calls = problem.calls

    def progress(record):
        if record['iteration'] % 64 == 0:
            log(json.dumps(record | dict(seconds=time.monotonic() - began)))
    protocol = {k: PAPER_PROTOCOL[k] for k in ('samples', 'warmup', 'max_depth', 'target_accept')}
    y, info = nuts(problem, **protocol, chains=1, seed=seed, progress=progress)
    assert y.shape == (1, 1024, 4096) and np.isfinite(y).all()
    np.savez_compressed(out / 'latent_samples.npz', y=y)
    info.update(seed=seed, potential_evaluations=problem.calls - calls, device=str(backend.device),
                checkpoint_sha256=artifacts.manifest()['checkpoint_sha256'])
    (out / 'diagnostics.json').write_text(json.dumps(_finite(info), indent=2) + '\n')
    return info


def merge(label, run_dir, out=None, device='cpu', threads=None, root=None, backend=None):
    """Diagnostics and balanced forecast from ``run_dir/chain_{0..3}``; returns a summary dict."""
    run_dir = Path(run_dir)
    ys, infos = [], []
    for c in range(4):
        with np.load(run_dir / f'chain_{c}' / 'latent_samples.npz') as z:
            ys.append(z['y'].copy())
        infos.append(json.loads((run_dir / f'chain_{c}' / 'diagnostics.json').read_text()))
    y = np.concatenate(ys)
    diagnostics, _ = convergence(y, infos)
    backend = backend or load_backend(device, threads, root)
    problem, _ = problem_for(backend, artifacts.Reservoirs(root), label, 'multiwell_opt17')
    selected, index = balanced_draws(y)
    result, arrays = forecast(problem, selected, save_members=True)
    if out is not None:
        write_forecast(out, result, arrays)
        (Path(out) / 'diagnostics.json').write_text(json.dumps(_finite(diagnostics), indent=2) + '\n')
    return dict(case=label, diagnostics=diagnostics, scores=result, selected_draw_indices=index.tolist())


def verify_released(label, device='cpu', threads=None, root=None, backend=None):
    """Recompute diagnostics and scores from the released chains; compare with the archived merge."""
    root = Path(root) if root else artifacts.data_dir()
    run_dir = root / 'nuts' / label
    if not run_dir.exists():
        raise FileNotFoundError(f'{run_dir} is missing; run `geouno download --group nuts`')
    summary = merge(label, run_dir, device=device, threads=threads, root=root, backend=backend)
    archived_diag = json.loads((run_dir / 'diagnostics.json').read_text())
    archived_scores = json.loads((run_dir / 'scores.json').read_text())
    keys = ('rank_rhat_max', 'bulk_ess_min', 'tail_ess_min', 'divergences', 'convergence_pass')
    comparison = dict(diagnostics={k: dict(recomputed=summary['diagnostics'][k], archived=archived_diag[k])
                                   for k in keys})
    new, old = headline(summary['scores']), headline(archived_scores)
    comparison['scores'] = {k: dict(recomputed=new[k], archived=old[k],
                                    rel_diff=abs(new[k] - old[k]) / max(abs(old[k]), 1e-12)) for k in old}
    return comparison
