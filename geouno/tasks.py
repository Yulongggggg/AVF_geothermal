"""Run the registered inference tasks of the paper and compare them with the archived results.

A task is ``(configuration, reservoir, method)``:

* ``esmda``         ES-MDA, 256 members, 8 updates (all dynamic configurations);
* ``static``        analytic static conditioning only (``multiwell_opt9_k``);
* ``map``           three-start L-BFGS-B MAP (``multiwell_opt17``);
* ``prior``         prior ensemble reference (``multiwell_opt17``);
* ``static_prior``  statically conditioned prior reference (``multiwell_opt17``).

Seeds follow the registered protocol: ensembles use ``observation_seed + 19000``, MAP starts
``observation_seed + 29000`` and the reference ensembles ``default_rng(81071)``.
NUTS has its own entry points (:mod:`geouno.nuts_runner`).
"""
import json
import time
from pathlib import Path

import numpy as np

from . import artifacts
from .backend import Backend
from .inverse import Problem, esmda, forecast, headline, map_optimize, static_members, write_forecast
from .monitoring import configuration, prepare_case
from .monitoring.protocol import (ENSEMBLE_SEED_OFFSET, ESMDA, MAP_SEED_OFFSET, PRIOR_REFERENCE_SAMPLES,
                                  PRIOR_REFERENCE_SEED)

METHODS = ('esmda', 'static', 'map', 'prior', 'static_prior')
TASK_FILES = ('scores.json', 'forecast.npz', 'latent_samples.npz', 'diagnostics.json')


def task_complete(out, config_key, label, method):
    """Require the matching completion record and all four intact task outputs."""
    out = Path(out)
    try:
        record = artifacts.read_json(out / 'complete.json')
        if not isinstance(record, dict) or record.get('format_version') != 1:
            return False
        if record.get('task') != dict(config=config_key, case=label, method=method):
            return False
        files = record.get('files')
        return (isinstance(files, dict) and set(files) == set(TASK_FILES)
                and all((out / name).is_file() and artifacts.sha256(out / name) == files[name]
                        for name in TASK_FILES))
    except (OSError, ValueError, TypeError):
        return False


def load_backend(device='cpu', threads=None, root=None):
    """Backend with the released checkpoint, statistics and inverse-prior factor."""
    info = artifacts.manifest()['assets']
    model = artifacts.path(info['geouno_pod1024_ed043489.pt']['path'], root)
    factor = artifacts.path(info['prior_factor_64eb46fb.npy']['path'], root)
    return Backend(model, artifacts.resource('stats.json'), factor, device=device, threads=threads)


def calibration():
    return artifacts.read_json(artifacts.resource('calibration.json'))


def design():
    return artifacts.read_json(artifacts.resource('design.json'))


def case_for(reservoirs, label, config_key):
    """Case arrays of ``config_key`` for reservoir ``label`` (regenerated from the opt17 case)."""
    base_case = reservoirs.case(label)
    if config_key == 'multiwell_opt17':
        return base_case
    entry = reservoirs.entry(label)
    return prepare_case(base_case, configuration(config_key), design(), entry['observation_seed'], label)


def problem_for(backend, reservoirs, label, config_key):
    config = configuration(config_key)
    case = case_for(reservoirs, label, config_key)
    problem = Problem(backend, case, reservoirs.base(label), calibration(),
                      **{k: config[k] for k in ('channels', 'static', 'times', 'sigma_t', 'sigma_p')})
    return problem, config


def check_method(config, method):
    if method not in METHODS:
        raise ValueError(f'method must be one of {METHODS}')
    if method == 'static' and config['channels']:
        raise ValueError('static-only inference applies to the multiwell_opt9_k configuration')
    if method == 'esmda' and not config['channels']:
        raise ValueError('multiwell_opt9_k has no dynamic data; use --method static')
    if method in ('map', 'prior', 'static_prior') and config['key'] != 'multiwell_opt17':
        raise ValueError(f'{method} was registered for multiwell_opt17 only')


def run(config_key, label, method, out=None, device='cpu', threads=None, root=None, backend=None, log=None):
    """Run one task; return ``(scores, info)``. Writes ``scores.json``/``forecast.npz`` to ``out``."""
    log = log or flush_print
    reservoirs = artifacts.Reservoirs(root)
    entry = reservoirs.entry(label)
    backend = backend or load_backend(device, threads, root)
    problem, config = problem_for(backend, reservoirs, label, config_key)
    check_method(config, method)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'complete.json').unlink(missing_ok=True)
    began = time.monotonic()
    seed = entry['observation_seed'] + ENSEMBLE_SEED_OFFSET
    initial_evaluations = backend.evaluations
    point = False
    if method == 'esmda':
        y, info = esmda(problem, **ESMDA, seed=seed, progress=lambda r: log(json.dumps(r)))
    elif method == 'static':
        y, info = static_members(problem, ESMDA['samples'], seed), dict(method='analytic static conditioning')
    elif method == 'map':
        y, info = map_optimize(problem, entry['observation_seed'] + MAP_SEED_OFFSET,
                               dest=None if out is None else Path(out) / 'map_starts')
        point = True
    else:
        y = np.random.default_rng(PRIOR_REFERENCE_SEED).normal(size=(PRIOR_REFERENCE_SAMPLES, 4096))
        if method == 'prior':
            problem.conditional = None
        info = dict(method=f'{method} reference ensemble')
    inference_elapsed = time.monotonic() - began
    inference_evaluations = backend.evaluations - initial_evaluations
    forecast_started = time.monotonic()
    result, arrays = forecast(problem, y, save_members=False, point_estimate=point)
    # Keep algorithm timing (including all MAP starts) separate from forecast generation.
    info.setdefault('seconds', inference_elapsed)
    info.update(inference_wall_seconds=inference_elapsed, forecast_seconds=time.monotonic() - forecast_started,
                total_seconds=time.monotonic() - began, config=config_key, case=label, method=method,
                device=str(backend.device), forward_member_times=inference_evaluations,
                forecast_member_times=backend.evaluations - initial_evaluations - inference_evaluations)
    if out is not None:
        out = Path(out)
        write_forecast(out, result, arrays)
        np.savez_compressed(out / 'latent_samples.npz', y=np.asarray(y))
        (out / 'diagnostics.json').write_text(json.dumps(_finite(info), indent=2) + '\n')
        record = dict(format_version=1, task=dict(config=config_key, case=label, method=method),
                      files={name: artifacts.sha256(out / name) for name in TASK_FILES})
        marker = out / 'complete.json.tmp'
        marker.write_text(json.dumps(record, indent=2) + '\n')
        marker.replace(out / 'complete.json')
    return result, info


def flush_print(*args):
    print(*args, flush=True)


def reference_key(config_key, label, method):
    if method in ('prior', 'static_prior'):
        return f'{method}/{label}'
    return f'{config_key}/{label}/{method}'


def reference_scores():
    return artifacts.read_json(artifacts.resource('paper_task_scores.json'))


def compare(result, reference):
    """Relative differences of the manuscript metrics (``headline``) against the archived run."""
    new, old = headline(result), headline(reference)
    rows = {}
    for key in old:
        a, b = new[key], old[key]
        rows[key] = dict(new=a, paper=b, abs_diff=abs(a - b), rel_diff=abs(a - b) / max(abs(b), 1e-12))
    return rows


def _finite(value):
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value
