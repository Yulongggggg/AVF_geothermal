"""Training-only POD bank for the GeoUNO output basis.

Source: ``operator_campaign_20260925/prepare_basis.py``. The original ran the two
channels on two GPUs; here they run sequentially with the same per-channel seeds
(``20260925 + channel``) and the same randomized SVD, which yields the same result on
the same hardware.

Snapshots: for each of the 6,582 training scenarios with actual members, two
(member, time) pairs chosen with ``default_rng(20260925)`` and a scenario-hashed time
offset (times ``o`` and ``o + 5`` mod 11). Each snapshot is divided by the channel scale
(temperature: ``fields_std[0]``; pressure: ``p_res_std_mpa``) and its spatial mean is
removed; the snapshot mean shape is then removed. Modes are orthonormal under the
pixel-mean inner product: ``modes @ modes.T / 20480 = I``.

Usage::

    python -m geouno.training.pod --pack /path/to/pack --out runs/basis_r1024 --rank 1024
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..model import sha256


def members(pack, split):
    result = {}
    for row in csv.DictReader((Path(pack) / 'meta.csv').open()):
        if row['split'] == split and row['real'] != 'nom':
            result.setdefault(row['scen'], []).append(int(row['row']))
    return result


def fit_channel(pack, channel, rank, device='cuda', expected_train_scenarios=6582):
    pack = Path(pack)
    stats = json.loads((pack / 'stats.json').read_text())
    scale = stats['fields_std'][0] if channel == 0 else stats['p_res_std_mpa']
    source = np.load(pack / 'fields.npy', mmap_mode='r')
    train = members(pack, 'train')
    if expected_train_scenarios is not None:
        assert len(train) == expected_train_scenarios, len(train)
    rng = np.random.default_rng(20260925)
    selection = []
    for name in sorted(train):
        offset = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little') % 11
        for j in range(2):
            selection.append((name, int(rng.choice(train[name])), (offset + 5 * j) % 11))
    started = time.monotonic()
    x = np.empty((len(selection), 160 * 128), dtype=np.float32)
    for i, (_, row, ti) in enumerate(selection):
        x[i] = np.asarray(source[row, ti, channel], dtype=np.float32).reshape(-1) / scale
        x[i] -= x[i].mean()
    x = torch.from_numpy(x).to(device)
    mean = x.mean(0)
    mean -= mean.mean()
    x -= mean
    torch.manual_seed(20260925 + channel)
    # Randomized SVD with oversampling; no validation or test data enters this fit.
    _, singular, v = torch.pca_lowrank(x, q=rank + 32, center=False, niter=4)
    q, _ = torch.linalg.qr(v[:, :rank] - v[:, :rank].mean(0), mode='reduced')
    modes = q.T * (x.shape[1] ** .5)
    orth_error = float((modes @ modes.T / x.shape[1] - torch.eye(rank, device=device)).abs().max())
    assert orth_error < 2e-5
    diagnostic = {}
    val = members(pack, 'val')
    with torch.no_grad():
        for r in (128, 256, 512, 1024):
            if r > rank:
                continue
            errors = []
            for ti in (4, 7, 10):
                for offset in range(0, len(val), 32):
                    names = sorted(val)[offset:offset + 32]
                    y = torch.tensor(np.stack([source[val[n][0], ti, channel] for n in names]),
                                     device=device).flatten(1) / scale
                    ym = y.mean(-1, keepdim=True)
                    c = (y - ym - mean) @ modes[:r].T / y.shape[1]
                    e = (c @ modes[:r] + ym + mean - y).reshape(-1, 160, 128)[:, 16:144] * scale
                    errors.extend(e.square().mean((-2, -1)).sqrt().cpu().tolist())
            diagnostic[str(r)] = float(np.mean(errors))
    report = dict(channel=channel, physical_scale=scale, fit_split='train', snapshots=len(selection),
                  train_scenarios=len(train), selection=selection, orthogonality_error=orth_error,
                  validation_projection_rmse=diagnostic, seconds=time.monotonic() - started,
                  projection_diagnostic_scope='First actual member of each validation scenario, times 4/7/10; '
                                              'not a network evaluation.')
    return dict(basis=modes.cpu(), mean=mean.cpu()), report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rank', type=int, default=1024)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--expected-train-scenarios', type=int, default=6582,
                        help='Guard on the number of training scenarios (use 0 to disable).')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    channels, reports = [], []
    for channel in (0, 1):
        result, report = fit_channel(args.pack, channel, args.rank, args.device,
                                     args.expected_train_scenarios or None)
        channels.append(result)
        reports.append(report)
        (args.out / f'channel_{channel}.json').write_text(json.dumps(report, indent=2) + '\n')
        print(dict(channel=channel, validation_projection_rmse=report['validation_projection_rmse']), flush=True)
    bank = args.out / 'bank.pt'
    torch.save(dict(split='train', basis=torch.stack([c['basis'] for c in channels]),
                    mean=torch.stack([c['mean'] for c in channels]), rank=args.rank,
                    meta_sha256=sha256(args.pack / 'meta.csv')), bank)
    (args.out / 'complete.json').write_text(json.dumps(dict(status='complete', bank_sha256=sha256(bank),
                                                            projection=[r['validation_projection_rmse']
                                                                        for r in reports]), indent=2) + '\n')


if __name__ == '__main__':
    main()
