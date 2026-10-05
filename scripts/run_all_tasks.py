"""Run every registered inference task of the paper (resumable) and summarize agreement.

1,104 monitoring evaluations (ES-MDA, or analytic conditioning for `multiwell_opt9_k`), 24 MAP runs and the
48 prior/static-prior reference ensembles. Each task writes `<out>/<config>/<case>/<method>/scores.json`;
finished tasks are skipped. Distribute the work by running several copies with different `--shard i/n`.

    python scripts/run_all_tasks.py --out runs/paper --threads 4 --shard 0/8
    python scripts/run_all_tasks.py --out runs/paper --summary
"""
import argparse
import json
from pathlib import Path

from geouno import artifacts
from geouno.monitoring import configurations
from geouno.tasks import compare, load_backend, reference_key, reference_scores, run


def tasks():
    test = artifacts.Reservoirs().labels('test')
    out = [('multiwell_opt17', case, 'map') for case in test]
    for c in configurations():
        cases = test[:c['test_limit']] if c.get('test_limit') else test
        method = 'esmda' if c['channels'] else 'static'
        out += [(c['key'], case, method) for case in cases]
    out += [('multiwell_opt17', case, m) for case in test for m in ('prior', 'static_prior')]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()
    todo = tasks()
    assert len(todo) == 1104 + 24 + 48
    if a.summary:
        ref, worst, done = reference_scores(), {}, 0
        for config, case, method in todo:
            path = a.out / config / case / method / 'scores.json'
            if not path.exists():
                continue
            done += 1
            rows = compare(json.loads(path.read_text()), ref[reference_key(config, case, method)])
            for metric, r in rows.items():
                worst[metric] = max(worst.get(metric, 0.), r['rel_diff'])
        print(json.dumps(dict(completed=done, total=len(todo), max_relative_difference=worst), indent=2))
        return
    index, count = map(int, a.shard.split('/'))
    backend = load_backend(a.device, a.threads)
    for i, (config, case, method) in enumerate(todo):
        dest = a.out / config / case / method
        if i % count != index or (dest / 'scores.json').exists():
            continue
        backend.evaluations = 0
        run(config, case, method, out=dest, backend=backend, log=lambda *_: None)
        print(f'{config}/{case}/{method} done', flush=True)


if __name__ == '__main__':
    main()
