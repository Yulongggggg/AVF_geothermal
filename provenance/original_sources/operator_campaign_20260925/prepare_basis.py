"""Training-only POD initialization; two ranks independently fit T and p modes."""
import argparse
import csv
import hashlib
import os
import time
import numpy as np
import torch
import torch.distributed as dist
from .common import PACK, OUT, PARENT, PARENT_SHA, read, save, sha, write


def members(split):
    result = {}
    for row in csv.DictReader((PACK/'meta.csv').open()):
        if row['split'] == split and row['real'] != 'nom':
            result.setdefault(row['scen'], []).append(int(row['row']))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=512)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    assert dist.get_world_size() == 2
    assert sha(PARENT) == PARENT_SHA
    out = OUT/f'basis_r{args.rank}'; out.mkdir(parents=True, exist_ok=True)
    stats = read(PACK/'stats.json')
    scale = stats['fields_std'][0] if rank == 0 else stats['p_res_std_mpa']
    source = np.load(PACK/'fields.npy', mmap_mode='r')
    train = members('train')
    assert len(train) == 6582
    rng = np.random.default_rng(20260925)
    selection = []
    for name in sorted(train):
        offset = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little') % 11
        for j in range(2):
            selection.append((name, int(rng.choice(train[name])), (offset+5*j) % 11))
    started = time.monotonic()
    x = np.empty((len(selection), 160*128), dtype=np.float32)
    for i, (_, row, ti) in enumerate(selection):
        x[i] = np.asarray(source[row, ti, rank], dtype=np.float32).reshape(-1)/scale
        x[i] -= x[i].mean()
        if (i+1) % 2000 == 0:
            print(f'channel={rank} loaded={i+1}/{len(x)}', flush=True)
    x = torch.from_numpy(x).cuda()
    mean = x.mean(0)
    mean -= mean.mean()
    x -= mean
    torch.manual_seed(20260925+rank)
    # Randomized SVD with oversampling; no validation or test data enters this fit.
    _, singular, v = torch.pca_lowrank(x, q=args.rank+32, center=False, niter=4)
    q, _ = torch.linalg.qr(v[:, :args.rank]-v[:, :args.rank].mean(0), mode='reduced')
    modes = q.T * (x.shape[1]**.5)
    orth_error = float((modes @ modes.T/x.shape[1]-torch.eye(args.rank, device='cuda')).abs().max())
    assert orth_error < 2e-5
    save(out/f'channel_{rank}.pt', dict(basis=modes.cpu(), mean=mean.cpu()))
    diagnostic = {}
    val = members('val')
    with torch.no_grad():
        for r in (128, 256, 512, 1024):
            if r > args.rank:
                continue
            errors = []
            for ti in (4, 7, 10):
                for offset in range(0, len(val), 32):
                    names = sorted(val)[offset:offset+32]
                    y = torch.tensor(np.stack([source[val[n][0], ti, rank] for n in names]), device='cuda').flatten(1)/scale
                    ym = y.mean(-1, keepdim=True)
                    c = (y-ym-mean) @ modes[:r].T/y.shape[1]
                    e = (c @ modes[:r]+ym+mean-y).reshape(-1, 160, 128)[:, 16:144]*scale
                    errors.extend(e.square().mean((-2, -1)).sqrt().cpu().tolist())
            diagnostic[str(r)] = float(np.mean(errors))
    write(out/f'channel_{rank}.json', dict(channel=rank, physical_scale=scale,
          fit_split='train', snapshots=len(selection), train_scenarios=len(train),
          selection=selection, orthogonality_error=orth_error,
          validation_projection_rmse=diagnostic,
          projection_diagnostic_scope='First actual member of each of 542 validation scenarios, times 4/7/10; not a network evaluation.',
          seconds=time.monotonic()-started, gpu=torch.cuda.get_device_name()))
    dist.barrier()
    if rank == 0:
        channels = [torch.load(out/f'channel_{c}.pt', map_location='cpu', weights_only=False) for c in (0, 1)]
        save(out/'bank.pt', dict(split='train', basis=torch.stack([x['basis'] for x in channels]),
             mean=torch.stack([x['mean'] for x in channels]), rank=args.rank,
             meta_sha256=sha(PACK/'meta.csv'), parent_sha256=PARENT_SHA))
        write(out/'complete.json', dict(status='complete', bank_sha256=sha(out/'bank.pt'),
             projection=[read(out/f'channel_{c}.json')['validation_projection_rmse'] for c in (0, 1)]))
        print(read(out/'complete.json'), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
