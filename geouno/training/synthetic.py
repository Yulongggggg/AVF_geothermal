"""Small random packs with the corpus layout, for tests and pipeline checks (values are not physical)."""
import csv
import json

import numpy as np


def make_synthetic_pack(pack, train=5, val=2, members=3, seed=0):
    """Small random pack with the corpus layout (for tests; values are not physical)."""
    g = np.random.default_rng(seed)
    pack.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, count in (('train', train), ('val', val)):
        for s in range(count):
            for r in range(members):
                rows.append(dict(case_id=f'{split}{s:03d}_r{r:03d}', scen=f'{split}{s:03d}', split=split,
                                 row=len(rows), real=f'r{r:03d}'))
            rows.append(dict(case_id=f'{split}{s:03d}_rnom', scen=f'{split}{s:03d}', split=split, row=len(rows),
                             real='nom'))
    with (pack / 'meta.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    np.save(pack / 'real.npy', (g.normal(size=(n, 128, 128)) * .5 - 13.4).astype(np.float32))
    np.save(pack / 'fields.npy', g.normal(size=(n, 11, 2, 160, 128)).astype(np.float32))
    scal = np.column_stack([g.uniform(1, 3.5, n), g.uniform(293, 313, n), np.zeros(n), g.uniform(-180, 180, n),
                            g.uniform(0, 360, n), np.zeros(n), g.uniform(4, 14, n), np.full(n, 3.06), np.full(n, 30.)])
    np.save(pack / 'scal.npy', scal.astype(np.float32))
    np.save(pack / '_ref_times.npy', np.array([7e3, 6.3e4, 5.11e5, 4.095e6, 3.2183e7, 7.9583e7, 1.26983e8,
                                               1.74383e8, 2.21783e8, 2.69183e8, 3.16e8]))
    (pack / 'stats.json').write_text(json.dumps(dict(fields_std=[11.32, 7.99], fields_mean=[2.7, .86],
                                                     p_res_std_mpa=1.6, M_mean=-13.4, M_std=.79)))
    return pack
