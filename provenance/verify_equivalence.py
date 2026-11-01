# Historical 7 October checker: use the artifacts-v1 code revision and artifacts.
"""Check that the ``geouno`` package computes exactly what the archived research code computed.

Two references are used:

* ``provenance/nuts_bundle_runtime`` - the portable runtime of the frozen NUTS bundle
  (OL-geothermal-final-scheme @ 137d61a). Its scientific
  definitions are verbatim copies of the cluster sources (hashes in its ``bundle.json``),
  with data access adapted to stored base inputs. It is the reference for the model,
  backend, posterior, conditioning and forecast.
* ``provenance/original_sources`` - the original modules. Modules that import only
  standard scientific packages are imported directly; functions inside modules whose
  imports reference the original cluster layout are extracted from the source text with
  ``ast`` and executed unchanged in a namespace that supplies their dependencies.

All comparisons run on the CPU with identical thread counts and require bitwise equality
(``torch.equal`` / ``np.array_equal`` / ``==``) unless stated. Requires the core release
artifacts (``geouno download``). Usage::

    python provenance/verify_equivalence.py [--data DIR] [--threads 2]
"""
import argparse
import ast
import copy
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / 'original_sources'
RUNTIME = HERE / 'nuts_bundle_runtime'
sys.path.insert(0, str(HERE.parent))

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append(dict(check=name, passed=bool(ok), detail=detail))
    print(f'{"PASS" if ok else "FAIL"}  {name}' + (f'  ({detail})' if detail else ''), flush=True)


def same_tree(a, b):
    """Recursive exact equality for nested dict/list/tensor/array/float structures."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a):
        return math.isnan(b)
    return a == b


def extract(path, names, namespace):
    """Execute the named top-level functions/classes of ``path`` unchanged in ``namespace``."""
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    missing = set(names) - {n.name for n in nodes}
    assert not missing, missing
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec')
    exec(code, namespace)
    return [namespace[n] for n in names]


def fresh_import(root, module):
    """Import ``module`` with ``root`` first on ``sys.path`` (purging same-named modules first)."""
    top = module.split('.')[0]
    for name in list(sys.modules):
        if name == top or name.startswith(top + '.'):
            del sys.modules[name]
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.remove(str(root))


def runtime_bundle(data, tmp):
    """Lay out a bundle directory for the archived portable runtime using released files."""
    from geouno import artifacts
    bundle = json.loads((RUNTIME / 'bundle.json').read_text())
    root = Path(tmp) / 'bundle'
    (root / 'data').mkdir(parents=True)
    shutil.copytree(RUNTIME / 'src', root / 'src')
    info = artifacts.manifest()['assets']
    os.symlink(artifacts.path(info['geouno_pod1024_ed043489.pt']['path'], data), root / 'data/model.pt')
    os.symlink(artifacts.path(info['prior_factor_64eb46fb.npy']['path'], data), root / 'data/prior_factor.npy')
    for split in ('test', 'val'):
        shutil.copyfile(artifacts.resource('stats.json'), root / f'data/stats_{split}.json')
    shutil.copyfile(artifacts.resource('calibration.json'), root / 'data/calibration.json')
    (root / 'data/cases').mkdir()
    (root / 'data/base_inputs').mkdir()
    for e in bundle['cases']:
        os.symlink(artifacts.path(f"reservoirs/cases/{e['label']}.npz", data), root / e['case_path'])
        os.symlink(artifacts.path(f"reservoirs/base_inputs/{e['label']}.npz", data), root / e['base_path'])
    (root / 'bundle.json').write_text(json.dumps(bundle))
    return root


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, default=None)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--low-memory', action='store_true',
                    help='recompute activations for memory-limited CPU verification')
    ap.add_argument('--report', type=Path, default=HERE / 'equivalence_report.json')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    started = time.monotonic()
    from check_memory import recompute_activations, release_unused

    from geouno import artifacts
    from geouno.backend import Backend
    from geouno.inverse import Problem, condition, transform, common_mode_quadratic, esmda, forecast
    from geouno.inverse import map_optimize, nuts as new_nuts, static_members
    from geouno.metrics import score
    from geouno import monitoring as M
    from geouno.training import loss as new_loss, metrics as new_metrics
    from geouno.training.train import ShardedBatches, phase_rates, bucket
    reservoirs = artifacts.Reservoirs(a.data)
    calibration = json.loads(artifacts.resource('calibration.json').read_text())

    with tempfile.TemporaryDirectory() as tmp:
        root = runtime_bundle(a.data, tmp)
        sys.path.insert(0, str(root / 'src'))
        from operator_final_20260930.inverse_backend import Backend as OldBackend, Problem as OldProblem
        from operator_final_20260930.inverse_run import forecast as old_forecast
        from operator_campaign_20260925.model import load as old_load
        from operator_parallel_20260927_v2.runtime import fixed_pressure_pool as old_pool
        from history_matching_dense.conditioning import (condition as old_condition, transform as old_transform,
                                                         common_mode_quadratic as old_cmq)
        from temperature_revision.inference import PhysicalInputEncoder as OldEncoder
        sys.path.remove(str(root / 'src'))

        # 1. Model forward and basis queries.
        from geouno.model import load_geouno
        stats = json.loads(artifacts.resource('stats.json').read_text())
        model_path = root / 'data/model.pt'
        new_net = load_geouno(model_path)
        old_net = old_pool(old_load(model_path)).eval()
        torch.backends.mha.set_fastpath_enabled(False)
        enc = OldEncoder(stats)
        for label in ('test_01', 'test_13', 'pilot_01'):
            base = torch.from_numpy(reservoirs.base(label))
            f, c = enc(base[[0, 4, 6, 10]].float())
            with torch.no_grad():
                check(f'model forward identical ({label}, 4 times)', torch.equal(new_net(f, c), old_net(f, c)))
        with torch.no_grad():
            q = torch.rand(37, 2) * 2 - 1
            check('model off-grid query identical', torch.equal(new_net(f, c, query=q), old_net(f, c, query=q)))

        # Release the two standalone models before allocating posterior backends.
        del new_net, old_net, enc, base, f, c, q
        release_unused()

        # 2. Backend and posterior on the validation pilot and two test reservoirs.
        # Compare lossless temporary snapshots, so only one full backend/posterior
        # is resident at a time. Neither implementation shares model state or data.
        rng = np.random.default_rng(2026)
        obs_cfg = dict(channels=(0, 1), static=True, times=tuple(range(7)), sigma_t=.5, sigma_p=.25)
        for label in ('pilot_01', 'test_01', 'test_06'):
            latent_y = rng.normal(size=(2, 4096))
            if label == 'test_01':
                latent = rng.normal(size=(8, 4096))
                one = rng.normal(size=(1, 4096))
            snapshot = Path(tmp) / label
            snapshot.mkdir()
            for original in (False, True):
                def compare_saved(name, value):
                    file = snapshot / (name.replace(':', '_').replace(' ', '_') + '.pt')
                    if original:
                        # Only this process's own temporary file is deserialized.
                        with file.open('rb') as handle:
                            reference = torch.load(handle, map_location='cpu', weights_only=False)
                            if hasattr(os, 'posix_fadvise'):
                                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                        check(name, same_tree(value, reference))
                    else:
                        with file.open('wb') as handle:
                            torch.save(value, handle)
                            handle.flush()
                            os.fsync(handle.fileno())
                            if hasattr(os, 'posix_fadvise'):
                                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

                backend = (OldBackend('val' if label == 'pilot_01' else 'test', 'cpu') if original else
                           Backend(model_path, stats, root / 'data/prior_factor.npy', threads=a.threads))
                torch.set_num_threads(a.threads)
                if a.low_memory:
                    recompute_activations(backend.operator)
                if label == 'pilot_01':
                    compare_saved('cached basis identical', backend.fixed_basis)
                case = reservoirs.case(label)
                if original:
                    p = OldProblem(backend, root / f'data/cases/{label}.npz', root / 'data/calibration.json',
                                   **{k: list(v) if isinstance(v, tuple) else v for k, v in obs_cfg.items()})
                else:
                    p = Problem(backend, case, reservoirs.base(label), calibration, **obs_cfg)
                compare_saved(f'{label}: whitening factor and data identical', (p.cholesky, p.data))
                compare_saved(f'{label}: static conditioning identical',
                              tuple(p.conditional[k] for k in ('mean', 'root', 'H', 'offset')))
                y = p.tensor(latent_y)
                with torch.no_grad():
                    compare_saved(f'{label}: permeability map identical', p.permeability(y))
                    compare_saved(f'{label}: whitened predictions identical', p.forward(y))
                release_unused()
                v = y[0].clone().requires_grad_(True)
                r = p.forward(v[None])[0] - p.data
                u = .5 * (v.square().sum() + r.square().sum())
                gradient, = torch.autograd.grad(u, v)
                compare_saved(f'{label}: posterior value and gradient identical', (u.detach(), gradient))
                del r, u, v, gradient, y
                release_unused()
                if label == 'test_01':
                    if original:
                        out = Path(tmp) / 'old_forecast'
                        out.mkdir()
                        scores = old_forecast(p, latent, out, save_members=True)
                        with np.load(out / 'forecast.npz') as z:
                            arrays = {k: z[k] for k in z.files}
                        point_scores = old_forecast(p, one, out, point_estimate=True)
                    else:
                        scores, arrays = forecast(p, latent, save_members=True)
                        point_scores = forecast(p, one, point_estimate=True)[0]
                    compare_saved('forecast scores identical', scores)
                    compare_saved('forecast arrays identical', arrays)
                    compare_saved('point-estimate scores identical', point_scores)
                    del scores, arrays, point_scores
                    if not original:
                        new_k = Problem(backend, case, reservoirs.base(label), calibration,
                                        channels=(), static=True, times=tuple(range(7)))
                        check('static-only problem has no dynamic data', new_k.nobs == 0)
                        del new_k
                del p, backend, case
                release_unused()
            if label == 'test_01':
                del latent, one

        new_b = Backend(model_path, stats, root / 'data/prior_factor.npy', threads=a.threads)
        if a.low_memory:
            recompute_activations(new_b.operator)
        release_unused()

        # 3. Conditioning primitives on random problems.
        factor = new_b.factor
        masks = torch.from_numpy(M.masks([('a', 10., 20.), ('b', -100., 50.), ('c', 200., -300.)], True))
        observed = torch.tensor([.3, -.2, .1], dtype=torch.float64)
        mean = torch.from_numpy(reservoirs.case('test_02')['prior_mean'])
        new_c, old_c = condition(mean, factor, masks, observed), old_condition(mean, factor, masks, observed)
        check('condition() identical', all(torch.equal(new_c[k], old_c[k]) for k in ('mean', 'root', 'H', 'offset')))
        yy = torch.randn(3, 4096, dtype=torch.float64)
        cd = {k: (v.double() if isinstance(v, torch.Tensor) else v) for k, v in new_c.items()}
        check('transform() identical', torch.equal(transform(yy, cd), old_transform(yy, cd)))
        err, ds, cs = torch.randn(4, 7, 9), torch.rand(7, 9) + .1, torch.rand(7)
        check('common_mode_quadratic() identical',
              torch.equal(common_mode_quadratic(err, ds, cs), old_cmq(err, ds, cs)))
        del new_c, old_c, cd, yy, err, ds, cs, masks, observed, mean
        release_unused()

        # 4. Scores.
        old_metrics = fresh_import(ORIGINAL, 'inference_comparison.metrics')
        draws, truth = rng.normal(size=(64,)), np.array(.3)
        check('score() raw identical', same_tree(score(draws, truth), old_metrics.score(draws, truth)))
        check('score() mixture identical', same_tree(score(draws, truth, .4), old_metrics.score(draws, truth, .4)))

        # 5. ES-MDA, static members, MAP and NUTS against the original algorithms.
        old_alg = fresh_import(ORIGINAL, 'inference_comparison.algorithms')
        problem = Problem(new_b, reservoirs.case('pilot_01'), reservoirs.base('pilot_01'), calibration, **obs_cfg)
        x_new, info_new = esmda(problem, samples=8, steps=2, seed=610000 + 19000)
        x_old, info_old = old_alg.esmda(problem, samples=8, steps=2, seed=610000 + 19000)
        check('ES-MDA (8 members, 2 updates) identical', np.array_equal(x_new, x_old)
              and info_new['history'] == info_old['history'])
        corrected = ORIGINAL / 'operator_final_20260930/inverse_run.py'
        text = corrected.read_text()
        assert 'size=(problem.dim,p[\'esmda\'][\'samples\'])).T' in text
        expected = np.random.default_rng(629000).normal(size=(problem.dim, 256)).T
        check('static-only members equal the corrected original draw',
              np.array_equal(static_members(problem, 256, 629000), expected))
        first = np.random.default_rng(629000).normal(size=(4096, 256))
        check('static-only members equal the ES-MDA initial ensemble',
              np.array_equal(static_members(problem, 256, 629000), first.T))
        namespace = dict(np=np, torch=torch, minimize=__import__('scipy.optimize', fromlist=['minimize']).minimize,
                         time=time, write=lambda *a, **k: None, read=lambda p: json.loads(Path(p).read_text()),
                         atomic_npz=lambda path, **v: np.savez(path, **v))
        old_map, = extract(ORIGINAL / 'operator_final_20260930/inverse_run_registered_20260930.py', ['map_optimize'],
                           namespace)
        options = dict(maxiter=3, maxfun=50, maxls=40, gtol=.0001, ftol=1e-10, maxcor=10)
        d1 = Path(tmp) / 'map_old'
        d1.mkdir()
        y_old, i_old = old_map(problem, d1, 639000, options)
        y_new, i_new = map_optimize(problem, 639000, options)
        check('MAP (3 starts, 3 iterations) identical', np.array_equal(y_old, y_new)
              and [s['objective'] for s in i_old['starts']] == [s['objective'] for s in i_new['starts']]
              and i_old['selected_start'] == i_new['selected_start'])

        class Toy:
            dim = 6
            data = torch.zeros(6, dtype=torch.float64)

            def tensor(self, x):
                return torch.as_tensor(x, dtype=torch.float64)

            def forward(self, x):
                return torch.tanh(x) * 2
        a_new, _ = new_nuts(Toy(), samples=20, warmup=20, chains=2, max_depth=4, seed=7)
        a_old, _ = old_alg.nuts(Toy(), samples=20, warmup=20, chains=2, max_depth=4, seed=7)
        check('NUTS kernel identical (toy posterior, 2 chains)', np.array_equal(a_new, a_old))
        # Pyro's built-in chain diagnostics need at least four retained draws.
        s_new, _ = new_nuts(problem, samples=4, warmup=2, chains=1, max_depth=2, seed=629000)
        s_old, _ = old_alg.nuts(problem, samples=4, warmup=2, chains=1, max_depth=2, seed=629000)
        check('NUTS on the GeoUNO posterior (2+4 steps, depth 2) identical', np.array_equal(s_new, s_old))

        # 6. Monitoring geometry, noise, design and configurations.
        hmn = fresh_import(ORIGINAL, 'history_matching_networks.networks')
        hmd = fresh_import(ORIGINAL, 'history_matching_dense.networks')
        geo_old = fresh_import(ORIGINAL, 'operator_final_20260930.monitoring_geometry')
        protocol = json.loads((HERE.parent / 'protocols/monitoring_protocol.json').read_text())
        wells = {e['label']: np.asarray(protocol['geometries'][e['label']]['multiwell']['points'], dtype=object)[:2, 1:]
                 .astype(float) for e in protocol['cases']['pilot'] + protocol['cases']['test']}
        ok_geo, exact, worst = True, 0, 0.
        from geouno.monitoring.protocol import registered_geometries
        packaged = registered_geometries()
        ok_pack = sorted(packaged) == sorted(protocol['geometries'])
        for label, w in wells.items():
            for family in M.FAMILIES:
                new_pts, new_info = M.geometry(w, family)
                old_pts, old_info = geo_old.geometry(w, family)
                ok_geo &= same_tree(new_pts, old_pts) and same_tree(new_info, old_info)
                stored = [tuple([p[0], float(p[1]), float(p[2])])
                          for p in protocol['geometries'][label][family]['points']]
                exact += same_tree(new_pts, stored)
                worst = max(worst, float(np.max(np.abs(np.array([p[1:] for p in new_pts])
                                                       - np.array([p[1:] for p in stored])))))
                ok_pack &= packaged[label][family] == protocol['geometries'][label][family]['points']
        check('monitoring geometries identical to original (28 reservoirs x 3 families)', ok_geo)
        check('recomputed geometries match the frozen protocol to 1e-9 m', worst < 1e-9,
              f'{exact}/84 bitwise; max difference {worst:.1e} m (platform hypot)')
        check('packaged geometries identical to the frozen protocol', ok_pack)
        pts = M.geometry(wells['test_03'], 'multiwell')[0][:40]
        check('point_noise() identical', np.array_equal(M.point_noise(pts, (11,), 610002, '0', 'monitoring'),
                                                        hmn.point_noise(pts, (11,), 610002, '0', 'monitoring')))
        old_masks, = extract(ORIGINAL / 'inference_comparison/geothermal.py', ['masks'], dict(np=np))
        check('masks() identical', np.array_equal(M.masks(pts), old_masks(pts))
              and np.array_equal(M.masks(pts, True), old_masks(pts, True)))
        check('candidate bank identical', same_tree(M.candidate_points(120., 200., 'bank'),
                                                    hmd.network_points(120., 200., 'bank')))
        old_cfg, = extract(ORIGINAL / 'operator_final_20260930/monitoring_prepare.py', ['configurations'],
                           dict(FAMILIES=('multiwell', 'vertical', 'inclined'),
                                BUDGETS=(2, 3, 5, 7, 9, 13, 17, 25, 33)))
        check('configurations() identical to original and to the frozen protocol',
              M.configurations() == old_cfg() == protocol['configs'])
        design_old = fresh_import(ORIGINAL, 'history_matching_sparse.design')
        ns = dict(np=np)
        old_state, old_choose = extract(ORIGINAL / 'final_model/monitoring_budget.py', ['state', 'choose'], ns)
        sys.modules.setdefault('history_matching_sparse.design', design_old)
        ok = True
        for trial in range(2):
            caches = []
            for r in range(4):
                g = np.random.default_rng(100 * trial + r)
                path = Path(tmp) / f'design_cache_{trial}_{r}.npz'
                np.savez_compressed(path, dynamic=g.normal(size=(11, 64, 2, 99)), geology=g.normal(size=(64, 99)))
                caches.append(path)
            geoms = [M.geometry(w, 'vertical')[0] for w in list(wells.values())[:6]]
            for rule in ('optimized', 'space_filling'):
                h_new = M.choose([M.state(c, calibration) for c in caches], geoms, rule, maximum=12)
                h_old = old_choose([old_state(c, calibration) for c in caches], geoms, rule, maximum=12)
                ok &= same_tree(h_new, h_old)
        check('design state()/choose() identical (synthetic validation priors)', ok)

        # 7. Training components.
        old_mmd = fresh_import(ORIGINAL, 'joint_mmd.loss')
        p = torch.randn(7, 2, 160, 128)
        t = torch.randn(7, 2, 160, 128)
        lengths = [3, 1, 3]
        a1 = new_loss.objective(p, t, lengths, [11.3, 8.0], 1.6, weight=.2, kind='rmse')
        a2 = old_mmd.objective(p, t, lengths, [11.3, 8.0], 1.6, weight=.2, kind='rmse')
        check('training objective identical', torch.equal(a1[0], a2[0]) and same_tree(a1[1], a2[1]))
        old_bv, = extract(ORIGINAL / 'direct_operator/model.py', ['block_values'], dict(torch=torch))
        old_rec, old_sum = extract(ORIGINAL / 'joint_precision/metrics.py', ['records', 'summarize'],
                                   dict(torch=torch, np=np, block_values=old_bv))
        field = torch.cat([torch.randn(7, 4, 160, 128), torch.rand(7, 2, 160, 128) * .05], 1)
        r_new, r_old = new_metrics.records(p, t, field, lengths), old_rec(p, t, field, lengths)
        check('validation records identical', same_tree(r_new, r_old))
        rows = [dict(scen=f's{i}', time_index=ti, **r) for i, r in enumerate(r_new) for ti in (4, 10)]
        check('validation summary identical', same_tree(new_metrics.summarize(rows), old_sum(copy.deepcopy(rows))))
        ns = dict(np=np, math=math, EpochScenarios=None)
        old_phase, old_bucket, old_quality = extract(ORIGINAL / 'operator_campaign_20260925/train.py',
                                                     ['phase_rates', 'bucket', 'quality'], ns)
        check('learning-rate schedule identical', all(phase_rates(e, 132) == old_phase(e, 132) for e in range(132)))
        names = [n for n, _ in new_b.operator.named_parameters()]
        check('parameter groups identical', [bucket(n) for n in names] == [old_bucket(n) for n in names])
        counts = [sum(phase_rates(e, 132)[0] == k for e in range(132))
                  for k in ('head_warmup', 'joint_transfer', 'fine_tune')]
        check('schedule is 11 warm-up, 55 transfer, 66 fine-tuning epochs', counts == [11, 55, 66], str(counts))
        _training_data_checks(tmp, ShardedBatches)

    report = dict(passed=all(r['passed'] for r in RESULTS), checks=RESULTS, threads=a.threads,
                  low_memory=a.low_memory,
                  torch=torch.__version__, numpy=np.__version__, seconds=time.monotonic() - started)
    a.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}))
    return 0 if report['passed'] else 1


def _training_data_checks(tmp, ShardedBatches):
    """Dataset items, scenario order and sampler: package vs original classes on a synthetic pack."""
    from geouno.training import data as new_data
    from geouno.training.synthetic import make_synthetic_pack
    pack = Path(tmp) / 'pack'
    make_synthetic_pack(pack, train=5, val=2, members=3)
    det = fresh_import(ORIGINAL, 'det_sampler')
    sys.modules['det_sampler'] = det
    old_trd = fresh_import(ORIGINAL, 'temperature_revision.data')
    sys.modules['temperature_revision.data'] = old_trd
    old_on = fresh_import(ORIGINAL, 'overnight_bench.data')
    sys.modules['overnight_bench.data'] = old_on
    old_dd = fresh_import(ORIGINAL, 'direct_operator.data')
    new_ds, old_ds = new_data.ScenarioPairs(pack, 'train'), old_dd.ScenarioPairs(pack, 'train')
    check('scenario order and members identical', new_ds.pairs.names == old_ds.pairs.names
          and new_ds.indices == old_ds.indices)
    ok = all(same_tree(new_ds[(g, ti)], old_ds[(g, ti)]) for g in range(len(new_ds)) for ti in (0, 5, 10))
    check('training samples identical (inputs, controls, targets)', ok)
    e_new, e_old = new_data.EpochScenarios(new_ds, 9), old_dd.EpochScenarios(old_ds, 9)
    for epoch in range(3):
        e_new.epoch = e_old.epoch = epoch
        ok &= list(e_new) == list(e_old)
    check('epoch traversal identical', ok)
    seen = []
    for rank in range(2):
        s = ShardedBatches(new_ds, 3, rank, 2, batch=2)
        seen += [i for b in s for i, _ in b]
    check('sharded batches cover every scenario once', sorted(seen) == list(range(len(new_ds))))


if __name__ == '__main__':
    sys.exit(main())
