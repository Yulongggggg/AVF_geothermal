"""Command-line interface: ``geouno <command> --help`` for details."""
import argparse
import json
import sys
from pathlib import Path


def _print(value):
    print(json.dumps(value, indent=2, default=float))


def cmd_download(a):
    from . import artifacts
    names = artifacts.asset_names(a.group) if a.asset is None else a.asset
    artifacts.fetch(names, root=a.data, force=a.force, source_dir=a.from_dir, keep_archives=a.keep_archives)


def cmd_verify(a):
    """Artifact hashes, then the archived pilot value/gradient check through the full posterior."""
    import numpy as np
    import torch
    from . import artifacts
    from .tasks import load_backend, problem_for
    groups = tuple(a.group) if a.group else ('core',)
    results = artifacts.verify(a.data, groups)
    bad = [p for p, ok in results.items() if not ok]
    print(f'{len(results) - len(bad)}/{len(results)} artifact files verified')
    if bad:
        print('missing or corrupt:', *bad, sep='\n  ')
        return 1
    backend = load_backend(a.device, a.threads, a.data)
    problem, _ = problem_for(backend, artifacts.Reservoirs(a.data), 'pilot_01', 'multiwell_opt17')
    with np.load(artifacts.path('reservoirs/validation/pilot_01_value_gradient.npz', a.data)) as z:
        ref = {k: z[k].copy() for k in z.files}
    y = problem.tensor(ref['y']).requires_grad_(True)
    prediction = problem.forward(y[None])[0]
    loss = .5 * (y.square().sum() + (prediction - problem.data).square().sum())
    gradient, = torch.autograd.grad(loss, y)
    got = dict(prediction=prediction.detach().cpu().numpy(), data=problem.data.cpu().numpy(),
               loss=np.array(loss.item()), gradient=gradient.cpu().numpy(),
               permeability=problem.permeability(y.detach()[None]).cpu().numpy())
    report = {}
    for key, value in got.items():
        diff = float(np.max(np.abs(value - ref[key])))
        scale = float(np.max(np.abs(ref[key])))
        report[key] = dict(max_abs_diff=diff, max_rel_diff=diff / max(scale, 1e-30))
    # The archived reference was computed on the original cluster CPU. Different hardware and
    # thread counts change FP32 round-off; agreement to ~1e-4 relative is expected.
    worst = max(r['max_rel_diff'] for r in report.values())
    _print(dict(reference='pilot_01 value and gradient (238 observations, 4096 latent coordinates)',
                comparisons=report, worst_relative_difference=worst, passed=worst < a.rtol))
    return 0 if worst < a.rtol else 1


def cmd_predict(a):
    """Forward prediction for the true permeability of a released reservoir, against FE truth."""
    import numpy as np
    import torch
    from . import artifacts
    from .tasks import load_backend
    reservoirs = artifacts.Reservoirs(a.data)
    backend = load_backend(a.device, a.threads, a.data)
    case, base = reservoirs.case(a.case), backend.base_tensor(reservoirs.base(a.case))
    m = torch.as_tensor(case['logk_truth'], device=backend.device)
    times = range(11) if a.times is None else a.times
    fields, rows = [], []
    with torch.no_grad():
        for ti in times:
            f = backend.predict_real(base, ti, m)[0].cpu().numpy()
            fields.append(f)
            err = f[:, 16:144] - case['truth'][ti, :, 16:144]
            rows.append(dict(time_index=ti, years=float(backend.stats['times_s'][ti]) / 3.15576e7,
                             temperature_rmse_K=float(np.sqrt(np.mean(err[0] ** 2))),
                             pressure_rmse_MPa=float(np.sqrt(np.mean(err[1] ** 2)))))
    _print(dict(case=a.case, note='Transferred application model on one FE truth. '
                                  'Table 1 uses three independent models and 8,621 members of 618 test scenarios.',
                errors=rows))
    if a.out:
        np.savez_compressed(a.out, prediction=np.stack(fields), truth=case['truth'][list(times)],
                            times=np.array(list(times)))


def cmd_prepare_cases(a):
    """Regenerate monitoring case files; ``--check`` compares multiwell_opt17 with the released files."""
    import numpy as np
    from . import artifacts
    from .monitoring import configurations, prepare_case
    from .tasks import design
    reservoirs = artifacts.Reservoirs(a.data)
    frozen = design()
    labels = reservoirs.labels('test') + ['pilot_01']
    configs = [c for c in configurations() if a.config in (None, c['key'])]
    if not configs:
        raise ValueError(f'Unknown monitoring configuration: {a.config}')
    mismatches = 0
    for label in labels:
        base_case = reservoirs.case(label)
        seed = reservoirs.entry(label)['observation_seed']
        for c in configs:
            if label.startswith('test') and c.get('test_limit') and int(label[5:]) > c['test_limit']:
                continue
            values = prepare_case(base_case, c, frozen, seed, label)
            if a.check and c['key'] == 'multiwell_opt17':
                same = all(np.array_equal(values[k], base_case[k]) for k in base_case)
                mismatches += not same
                print(f'{label} multiwell_opt17 regenerated == released: {same}')
            if a.out:
                dest = Path(a.out) / c['key'] / f'{label}.npz'
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, **values)
    return 1 if mismatches else 0


def cmd_run(a):
    from .tasks import compare, reference_key, reference_scores, run
    result, info = run(a.config, a.case, a.method, out=a.out, device=a.device, threads=a.threads, root=a.data)
    reference = reference_scores().get(reference_key(a.config, a.case, a.method))
    summary = dict(task=f'{a.config}/{a.case}/{a.method}', seconds=info['seconds'])
    if reference is not None:
        summary['comparison_with_paper_run'] = compare(result, reference)
    _print(summary)


def cmd_nuts_chain(a):
    from .nuts_runner import run_chain
    run_chain(a.case, a.chain, Path(a.out) / a.case / f'chain_{a.chain}', a.device, a.threads, a.data)


def cmd_nuts_merge(a):
    from .nuts_runner import merge
    summary = merge(a.case, Path(a.dir) / a.case, out=Path(a.dir) / a.case, device=a.device,
                    threads=a.threads, root=a.data)
    _print(dict(case=a.case, diagnostics=summary['diagnostics']))


def cmd_nuts_verify(a):
    from .nuts_runner import NUTS_CASES, verify_released
    from .tasks import load_backend
    backend = load_backend(a.device, a.threads, a.data)
    passed = True
    for case in (a.case or NUTS_CASES):
        report = verify_released(case, root=a.data, backend=backend, rtol=a.rtol, atol=a.atol)
        _print(dict(case=case, **report))
        passed = passed and report['passed']
    return 0 if passed else 1


def cmd_paper_numbers(a):
    from .paper import main
    return main()


def cmd_refit_prior(a):
    from . import artifacts
    from .inverse.fit_prior import refit_from_release
    _print(refit_from_release(artifacts.path('prior_fit', a.data), a.out))


def build_parser():
    ap = argparse.ArgumentParser(prog='geouno', description='GeoUNO reproduction tools.')
    sub = ap.add_subparsers(dest='command', required=True)

    def add(name, func, help_text, device=True):
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument('--data', type=Path, default=None,
                       help='Artifact directory (default: $GEOUNO_DATA or ./artifacts).')
        if device:
            p.add_argument('--device', default='cpu', help="Torch device, e.g. 'cpu' or 'cuda:0'.")
            p.add_argument('--threads', type=int, default=None, help='CPU threads (paper MAP/ES-MDA: 4).')
        p.set_defaults(func=func)
        return p

    p = add('download', cmd_download, 'Download and verify release assets.', device=False)
    p.add_argument('--group', default='core', choices=('core', 'nuts', 'prior', 'training', 'all'))
    p.add_argument('--asset', nargs='+', default=None, help='Specific asset names.')
    p.add_argument('--from-dir', type=Path, default=None, help='Install from a local copy of the assets.')
    p.add_argument('--force', action='store_true')
    p.add_argument('--keep-archives', action='store_true', help='Keep downloaded .tar files after extraction.')

    p = add('verify', cmd_verify, 'Verify artifacts and reproduce the archived value/gradient check.')
    p.add_argument('--group', nargs='+', default=None, choices=('core', 'nuts', 'prior', 'training', 'all'))
    p.add_argument('--rtol', type=float, default=1e-3)

    p = add('predict', cmd_predict, 'Forward prediction with the true permeability of a released reservoir.')
    p.add_argument('--case', default='test_01')
    p.add_argument('--times', type=int, nargs='+', default=None)
    p.add_argument('--out', type=Path, default=None)

    p = add('prepare-cases', cmd_prepare_cases, 'Regenerate monitoring case files from released reservoirs.',
            device=False)
    p.add_argument('--config', default=None, help='One configuration key (default: all 49).')
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--check', action='store_true')

    p = add('run', cmd_run, 'Run one registered inference task and compare with the paper run.')
    p.add_argument('--config', default='multiwell_opt17')
    p.add_argument('--case', default='test_01')
    p.add_argument('--method', default='esmda', choices=('esmda', 'static', 'map', 'prior', 'static_prior'))
    p.add_argument('--out', type=Path, default=None)

    p = add('nuts-chain', cmd_nuts_chain, 'Run one formal NUTS chain (hours on a GPU).')
    p.add_argument('--case', required=True)
    p.add_argument('--chain', type=int, required=True, choices=range(4))
    p.add_argument('--out', type=Path, required=True)

    p = add('nuts-merge', cmd_nuts_merge, 'Diagnostics and forecast from four chains in --dir/<case>.')
    p.add_argument('--case', required=True)
    p.add_argument('--dir', type=Path, required=True)

    p = add('nuts-verify', cmd_nuts_verify, 'Recompute diagnostics and scores from the released NUTS chains.')
    p.add_argument('--case', nargs='+', default=None, choices=tuple(f'test_{i:02d}' for i in range(1, 7)))
    p.add_argument('--rtol', type=float, default=1e-4, help='Relative tolerance for forecast scores.')
    p.add_argument('--atol', type=float, default=1e-8, help='Absolute tolerance for forecast scores.')

    add('paper-numbers', cmd_paper_numbers, 'Recompute Tables 1, 2, S1 and abstract numbers from Source Data.',
        device=False)

    p = add('refit-prior', cmd_refit_prior, 'Refit the inverse prior from the released training fields.',
            device=False)
    p.add_argument('--out', type=Path, required=True)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    code = a.func(a)
    sys.exit(code or 0)


if __name__ == '__main__':
    main()
