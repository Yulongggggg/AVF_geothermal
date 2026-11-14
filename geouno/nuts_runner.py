"""NUTS runs and archived-chain verification for all 24 test reservoirs.

Four chains per reservoir use H100/H200 GPUs, two concurrent chains per GPU.
The saved full-cohort runs have zero divergences; 23 of 24 pass the diagnostic criteria.
The verify_released command recomputes diagnostics and forecast scores from saved draws.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

from . import artifacts
from .inverse import balanced_draws, convergence, forecast, headline, nuts, write_forecast
from .inverse.nuts import PAPER_PROTOCOL
from .monitoring.protocol import ENSEMBLE_SEED_OFFSET
from .tasks import _finite, flush_print, load_backend, problem_for

NUTS_CASES = tuple(f'test_{i:02d}' for i in range(1, 25))
RELEASED_NUTS_CASES = NUTS_CASES


def run_chain(label, chain, out, device='cuda:0', threads=None, root=None, backend=None, log=None):
    """Run formal chain ``chain`` (0-3) for ``label``; writes ``latent_samples.npz`` and ``diagnostics.json``."""
    if label not in NUTS_CASES or chain not in range(4):
        raise ValueError('NUTS was registered for test_01..test_24, chains 0..3')
    log = log or flush_print
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
    xi, info = nuts(problem, **protocol, chains=1, seed=seed, progress=progress)
    if xi.shape != (1, PAPER_PROTOCOL['samples'], 4096) or not np.isfinite(xi).all():
        raise ValueError('NUTS returned incomplete or non-finite samples')
    samples = out / 'latent_samples.npz'
    temporary = out / f'latent_samples.{os.getpid()}.tmp'
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, y=xi)  # Archive key y stores xi.
    temporary.replace(samples)
    info.update(format_version=1, case=label, chain=chain, seed=seed, protocol=protocol,
                potential_evaluations=problem.calls - calls, device=str(backend.device),
                checkpoint_sha256=artifacts.manifest()['checkpoint_sha256'],
                case_sha256=artifacts.sha256(reservoirs.case_path(label)),
                calibration_sha256=artifacts.sha256(artifacts.resource('calibration.json')),
                samples_sha256=artifacts.sha256(samples))
    temporary = out / f'diagnostics.{os.getpid()}.tmp'
    temporary.write_text(json.dumps(_finite(info), indent=2) + '\n')
    temporary.replace(out / 'diagnostics.json')
    return info


def _read_chain(label, chain, folder, reservoirs):
    """Read one complete chain and check its reservoir, protocol and saved sample identity."""
    sample_path, info_path = folder / 'latent_samples.npz', folder / 'diagnostics.json'
    info = json.loads(info_path.read_text())
    entry = reservoirs.entry(label)
    expected = dict(checkpoint_sha256=artifacts.manifest()['checkpoint_sha256'],
                    case_sha256=entry['multiwell_opt17_case_sha256'],
                    calibration_sha256=artifacts.sha256(artifacts.resource('calibration.json')))
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(f'{folder}: {key} does not match {label}')
    digest = artifacts.sha256(sample_path)
    if info.get('format_version') == 1:
        expected.update(case=label, chain=chain,
                        seed=entry['observation_seed'] + ENSEMBLE_SEED_OFFSET + chain,
                        samples_sha256=digest,
                        protocol={k: PAPER_PROTOCOL[k] for k in ('samples', 'warmup', 'max_depth', 'target_accept')})
        for key, value in expected.items():
            if info.get(key) != value:
                raise ValueError(f'{folder}: {key} does not match chain {chain} of {label}')
    else:
        # The original released chains predate the portable metadata schema. Their
        # manifest hashes identify both the chain position and the original diagnostics.
        files = {str(Path(record['extract']) / name): value
                 for record in artifacts.manifest()['assets'].values() if record.get('group') == 'nuts'
                 for name, value in record['files'].items()}
        for name, actual in [('latent_samples.npz', digest), ('diagnostics.json', artifacts.sha256(info_path))]:
            if files.get(f'nuts/{label}/chain_{chain}/{name}') != actual:
                raise ValueError(f'{folder}: legacy chain identity is not in the release manifest')
    with np.load(sample_path, allow_pickle=False) as archive:
        xi = archive['y'].copy()
    if xi.shape != (1, PAPER_PROTOCOL['samples'], 4096) or not np.isfinite(xi).all():
        raise ValueError(f'{folder}: incomplete or non-finite NUTS samples')
    return xi, info, digest


def merge(label, run_dir, out=None, device='cpu', threads=None, root=None, backend=None):
    """Diagnostics and balanced forecast from ``run_dir/chain_{0..3}``; returns a summary dict."""
    if label not in NUTS_CASES:
        raise ValueError('NUTS was registered for test_01..test_24')
    run_dir = Path(run_dir)
    reservoirs = artifacts.Reservoirs(root)
    chain_samples, infos, digests = [], [], set()
    for c in range(4):
        xi, info, digest = _read_chain(label, c, run_dir / f'chain_{c}', reservoirs)
        if digest in digests:
            raise ValueError('The four NUTS chains must contain independently generated samples')
        chain_samples.append(xi)
        infos.append(info)
        digests.add(digest)
    xi = np.concatenate(chain_samples)
    diagnostics, _ = convergence(xi, infos)
    backend = backend or load_backend(device, threads, root)
    problem, _ = problem_for(backend, reservoirs, label, 'multiwell_opt17')
    selected, index = balanced_draws(xi)
    result, arrays = forecast(problem, selected, save_members=True)
    if out is not None:
        write_forecast(out, result, arrays)
        (Path(out) / 'diagnostics.json').write_text(json.dumps(_finite(diagnostics), indent=2) + '\n')
    return dict(case=label, diagnostics=diagnostics, scores=result, selected_draw_indices=index.tolist())


def verify_released(label, device='cpu', threads=None, root=None, backend=None, rtol=1e-4, atol=1e-8):
    """Recompute diagnostics and scores from the released chains; compare with the archived merge."""
    if label not in RELEASED_NUTS_CASES:
        raise ValueError('Released NUTS cases are test_01..test_24')
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) < 0:
        raise ValueError('Verification tolerances must be finite and nonnegative')
    root = Path(root) if root else artifacts.data_dir()
    run_dir = root / 'nuts' / label
    if not run_dir.exists():
        raise FileNotFoundError(f'{run_dir} is missing; run `geouno download --group nuts`')
    files = ['diagnostics.json', 'scores.json']
    files += [f'chain_{c}/{name}' for c in range(4) for name in ('latent_samples.npz', 'diagnostics.json')]
    for name in files:
        artifacts.path(f'nuts/{label}/{name}', root)
    summary = merge(label, run_dir, device=device, threads=threads, root=root, backend=backend)
    archived_diag = json.loads((run_dir / 'diagnostics.json').read_text())
    archived_scores = json.loads((run_dir / 'scores.json').read_text())
    keys = ('rank_rhat_max', 'bulk_ess_min', 'tail_ess_min', 'divergences', 'convergence_pass')
    comparison = dict(diagnostics={k: dict(recomputed=summary['diagnostics'][k], archived=archived_diag[k])
                                   for k in keys})
    new, old = headline(summary['scores']), headline(archived_scores)
    comparison['scores'] = {k: dict(recomputed=new[k], archived=old[k],
                                    rel_diff=abs(new[k] - old[k]) / max(abs(old[k]), 1e-12)) for k in old}
    for key, values in comparison['diagnostics'].items():
        if key in ('divergences', 'convergence_pass'):
            values['passed'] = values['recomputed'] == values['archived']
        else:
            values['passed'] = bool(np.isclose(values['recomputed'], values['archived'], rtol=1e-10, atol=1e-12))
    for values in comparison['scores'].values():
        values['passed'] = bool(np.isclose(values['recomputed'], values['archived'], rtol=rtol, atol=atol))
    comparison['tolerances'] = dict(score_rtol=rtol, score_atol=atol, diagnostic_rtol=1e-10,
                                    diagnostic_atol=1e-12)
    comparison['convergence_pass'] = bool(summary['diagnostics']['convergence_pass'])
    comparison['passed'] = all(v['passed'] for group in ('diagnostics', 'scores')
                               for v in comparison[group].values())
    return comparison
