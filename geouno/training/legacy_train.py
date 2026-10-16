"""Distributed, equal-scenario transfer training of GeoUNO from the grid-output parent.

Source: ``operator_campaign_20260925/train.py``. Scientific content is unchanged:
the network inherits every tensor of the parent predictor (``cf278e64...``), its POD
basis comes from :mod:`geouno.training.pod`, and 132 epochs follow the schedule

* epochs 0-10   head warm-up      (only new coefficient head and trunk, lr ramp to 1e-4)
* epochs 11-65  joint transfer    (new 3e-5, shared 1e-6, T 1e-5, p 3e-6, level 1e-5)
* epochs 66-131 fine tuning        (from the best validation checkpoint, cosine to 10 %)

with AdamW (weight decay 1e-4), gradient clipping at 5, a global batch of 16 complete
scenarios, BF16 autocast and loss = paired RMSE + 0.2 MMD^2. Validation on all 542
validation scenarios at time indices 4, 7, 10 selects ``best.pt`` (relative to the parent);
a final evaluation covers all eleven times. No test data are used.

Changes for portability: paths are arguments, provenance is written to ``protocol.json``
in the output directory, and ``--device cpu`` (gloo backend, no autocast) exists for smoke
tests only.

Usage (four GPUs, as in the paper)::

    torchrun --nproc_per_node 4 -m geouno.training.train --pack PACK --parent parent.pt \\
        --bank runs/basis_r1024/bank.pt --out runs/transfer --pod-rank 1024
"""
import argparse
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from ..model import PARENT_SHA256, sha256
from ..model.geothermal import load_parent
from ..model.geouno import GeoUNOBasisHead
from .data import EpochScenarios, ScenarioPairs, collate
from .loss import objective
from .metrics import records, summarize

PAPER_COUNTS = (6582, 72939, 542, 6275)  # train scenarios/members, validation scenarios/members


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


class ShardedBatches:
    """Global batches of whole scenarios, split across ranks without duplication."""

    def __init__(self, data, seed, rank, world, batch=16):
        self.base = EpochScenarios(data, seed)
        self.rank, self.world, self.batch = rank, world, batch
        self.start = 0

    def batches(self):
        all_items = list(self.base)
        batches = [all_items[j:j + self.batch] for j in range(0, len(all_items), self.batch)]
        # Merge a small tail so that every rank always has a real scenario.
        if len(batches) > 1 and len(batches[-1]) < self.world:
            tail = batches.pop()
            batches[-1].extend(tail)
        return batches

    def __len__(self):
        return len(self.batches()) - self.start

    def __iter__(self):
        for batch in self.batches()[self.start:]:
            idx = np.array_split(np.arange(len(batch)), self.world)[self.rank]
            assert len(idx)
            yield [batch[int(j)] for j in idx]


@torch.no_grad()
def evaluate(net, data, rank, world, times, device, workers=2):
    net.eval()
    sampler = [(i, ti) for i in range(len(data)) for ti in times][rank::world]
    loader = DataLoader(data, batch_size=1, sampler=sampler, collate_fn=collate,
                        num_workers=workers, pin_memory=device.type == 'cuda')
    rows = []
    for (i, ti), (f, c, y, lengths) in zip(sampler, loader):
        f, c, y = [v.to(device, non_blocking=True) for v in (f, c, y)]
        outputs = []
        for j in range(0, len(f), 8):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                outputs.append(net(f[j:j + 8], c[j:j + 8]).float())
        pred = torch.cat(outputs)
        assert torch.isfinite(pred).all()
        rows.append(dict(scen=data.pairs.names[i], time_index=ti, **records(pred, y, f, lengths)[0]))
    gathered = [None] * world
    dist.all_gather_object(gathered, rows)
    rows = sorted([r for part in gathered for r in part], key=lambda r: (r['scen'], r['time_index']))
    assert len(rows) == len(data) * len(times)
    assert len({(r['scen'], r['time_index']) for r in rows}) == len(rows)
    return dict(scenario_count=len(data), actual_members=len(data.pairs), times=list(times),
                summary=summarize(rows), records=rows)


def quality(result, reference):
    """Mean T/p RMSE ratio to the parent (final time and all evaluated times) and eligibility."""
    a, b = result['summary'], reference['summary']
    ratios = [a[t][k] / b[t][k] for t in ('10', 'all_evaluated_times') for k in ('T_rmse', 'P_rmse')]
    tails = ('inj_P90', 'prod_P90', 'P_scenario_P90', 'P_shape_mse', 'inj_rmse', 'prod_rmse')
    eligible = max(ratios) <= 1.0 and all(a['10'][k] <= 1.02 * b['10'][k] for k in tails if b['10'][k] is not None)
    return float(np.mean(ratios)), bool(eligible)


def phase_rates(epoch, epochs, smoke=False):
    warm, joint = (1, 2) if smoke else (11, 66)
    if epoch < warm:
        scale = min(1., (epoch + 1) / 3)
        return 'head_warmup', dict(new=1e-4 * scale, shared=0., temperature=0., pressure=0., level=0.)
    if epoch < joint:
        return 'joint_transfer', dict(new=3e-5, shared=1e-6, temperature=1e-5, pressure=3e-6, level=1e-5)
    scale = .1 + .9 * (1 + math.cos(math.pi * (epoch - joint) / max(1, epochs - joint))) / 2
    return 'fine_tune', {k: v * scale for k, v in dict(new=1e-5, shared=3e-7, temperature=3e-6,
                                                      pressure=1e-6, level=3e-6).items()}


def bucket(name):
    if name.startswith(('trunk.', 'coefficient_delta.')):
        return 'new'
    if name.startswith('temperature_'):
        return 'temperature'
    if name.startswith('level_head.'):
        return 'level'
    if name.startswith(('pressure_', 'geology_')):
        return 'pressure'
    return 'shared'


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pack', type=Path, required=True, help='Packed FE corpus (see docs/DATA.md).')
    ap.add_argument('--parent', type=Path, required=True, help='Grid-output parent checkpoint.')
    ap.add_argument('--bank', type=Path, required=True, help='POD bank from geouno.training.pod.')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--pod-rank', type=int, choices=(512, 1024), required=True)
    ap.add_argument('--seed', type=int, default=20260925)
    ap.add_argument('--epochs', type=int, default=132)
    ap.add_argument('--max-hours', type=float, default=22.5, help='Checkpoint and stop after this wall time.')
    ap.add_argument('--smoke', action='store_true', help='Tiny subset and 3 epochs (pipeline test only).')
    ap.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--skip-parent-hash-check', action='store_true',
                    help='Allow a parent other than cf278e64... (not the paper lineage).')
    ap.add_argument('--skip-count-check', action='store_true',
                    help='Allow a pack whose split sizes differ from the paper corpus.')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(1)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29517')
    if args.device == 'cuda':
        torch.cuda.set_device(local)
        device = torch.device('cuda', local)
        dist.init_process_group('nccl', timeout=timedelta(minutes=60), rank=rank, world_size=world)
    else:
        device = torch.device('cpu')
        dist.init_process_group('gloo', timeout=timedelta(minutes=60), rank=rank, world_size=world)
    began = time.monotonic()
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    parent_sha = sha256(args.parent)
    if not args.skip_parent_hash_check:
        assert parent_sha == PARENT_SHA256, 'Parent checkpoint differs from the paper lineage'
    out = args.out / ('smoke' if args.smoke else 'formal') / f'r{args.pod_rank}_s{seed}'
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(pack_meta_sha256=sha256(args.pack / 'meta.csv'), pack_stats_sha256=sha256(args.pack / 'stats.json'),
                    parent_sha256=parent_sha, basis_sha256=sha256(args.bank), pod_rank=args.pod_rank, seed=seed,
                    epochs=args.epochs, smoke=args.smoke, world_size=world,
                    device_type=device.type, torch_version=str(torch.__version__),
                    numpy_version=np.__version__, cuda_version=torch.version.cuda,
                    tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                    tf32_cudnn=torch.backends.cudnn.allow_tf32,
                    sources_sha256={str(p.relative_to(Path(__file__).parents[1])): sha256(p)
                                    for p in sorted(Path(__file__).parents[1].rglob('*.py'))})
    if rank == 0 and not (out / 'protocol.json').exists():
        write(out / 'protocol.json', protocol)
    dist.barrier()
    assert read(out / 'protocol.json') == protocol, 'Existing run used different inputs; use a new --out'
    protocol_sha = sha256(out / 'protocol.json')
    if (out / 'complete.json').exists():
        dist.destroy_process_group()
        return
    stats = read(args.pack / 'stats.json')
    net = GeoUNOBasisHead(stats['fields_std'], stats['p_res_std_mpa'], pod_rank=args.pod_rank)
    copied = net.initialize_parent(args.parent)
    net.initialize_pod(args.bank)
    net.to(device)
    groups = {k: [] for k in ('new', 'shared', 'temperature', 'pressure', 'level')}
    for name, p in net.named_parameters():
        groups[bucket(name)].append(p)
    opt = torch.optim.AdamW([dict(params=p, name=k, lr=1e-5) for k, p in groups.items()], weight_decay=1e-4)
    epochs = 3 if args.smoke else args.epochs
    tr = ScenarioPairs(args.pack, 'train', 13 if args.smoke else 0, 2 if args.smoke else 0)
    va = ScenarioPairs(args.pack, 'val', 4 if args.smoke else 0, 2 if args.smoke else 0)
    assert not set(tr.pairs.names) & set(va.pairs.names)
    if not args.smoke and not args.skip_count_check:
        assert (len(tr), len(tr.pairs), len(va), len(va.pairs)) == PAPER_COUNTS
    workers = args.workers if args.workers is not None else (0 if args.smoke else 2)
    sampler = ShardedBatches(tr, seed + 3000, rank, world, batch=8 if args.smoke else 16)
    loader = DataLoader(tr, batch_sampler=sampler, collate_fn=collate, num_workers=workers,
                        persistent_workers=workers > 0, pin_memory=device.type == 'cuda')
    times = (10,) if args.smoke else (4, 7, 10)
    history, best, eligible_best, start, next_batch = [], float('inf'), float('inf'), 0, 0
    running = dict(loss_sum=0., scenarios=0, members=0)
    if (out / 'last.pt').exists():
        old = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        assert old['protocol_sha256'] == protocol_sha and old['world_size'] == world
        net.load_state_dict(old['model'])
        opt.load_state_dict(old['optimizer'])
        start, next_batch = old['next_epoch'], old['next_batch']
        history, best, eligible_best, running = old['history'], old['best'], old['eligible_best'], old['running']
        torch.set_rng_state(old['rng_cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(old['rng_cuda'][rank])
        print(f'resume rank={rank} epoch={start} batch={next_batch}', flush=True)
    ddp = DDP(net, device_ids=[local] if device.type == 'cuda' else None, broadcast_buffers=False)

    def snapshot(next_epoch, batch_index, stats_running):
        cuda_states = [None] * world
        state = torch.cuda.get_rng_state().cpu() if device.type == 'cuda' else None
        dist.all_gather_object(cuda_states, state)
        if rank == 0:
            save(out / 'last.pt', dict(model=net.state_dict(), model_config=net.config,
                                       optimizer=opt.state_dict(), next_epoch=next_epoch, next_batch=batch_index,
                                       history=history, best=best, eligible_best=eligible_best,
                                       running=stats_running, rng_cpu=torch.get_rng_state(), rng_cuda=cuda_states,
                                       world_size=world, protocol_sha256=protocol_sha))
        dist.barrier()

    if (out / 'reference_validation.json').exists():
        reference = read(out / 'reference_validation.json')
    else:
        parent = load_parent(args.parent).to(device)
        reference = evaluate(parent, va, rank, world, times, device, workers)
        del parent
        if rank == 0:
            write(out / 'reference_validation.json', reference)
        dist.barrier()
    if not (out / 'manifest.json').exists():
        initial = evaluate(net, va, rank, world, times, device, workers)
        if rank == 0:
            write(out / 'initial_validation.json', initial)
            write(out / 'manifest.json', dict(model_config=net.config, seed=seed, smoke=args.smoke,
                                              world_size=world, protocol_sha256=protocol_sha,
                                              copied_parent_tensors=len(copied), parent_sha256=parent_sha,
                                              parameters=sum(p.numel() for p in net.parameters()),
                                              train_scenarios=len(tr), train_actual_members=len(tr.pairs),
                                              validation_scenarios=len(va), validation_actual_members=len(va.pairs)))
            print(dict(initial_quality=quality(initial, reference), out=str(out)), flush=True)
        dist.barrier()

    for epoch in range(start, epochs):
        # Fine tuning starts from the validation-selected transfer checkpoint.
        if epoch == (2 if args.smoke else 66) and next_batch == 0:
            chosen = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
            net.load_state_dict(chosen['model'])
            opt.state.clear()
        phase, rates = phase_rates(epoch, epochs, args.smoke)
        for group in opt.param_groups:
            group['lr'] = rates[group['name']]
        sampler.base.epoch = epoch
        sampler.start = next_batch
        ddp.train()
        epoch_started = time.monotonic()
        for batch_index, (f, c, y, lengths) in enumerate(loader, start=next_batch):
            f, c, y = [v.to(device, non_blocking=True) for v in (f, c, y)]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                pred = ddp(f, c)
            value, _ = objective(pred, y, lengths, net.scales, net.pressure_residual_scale, weight=.2, kind='rmse')
            counts = torch.tensor([len(lengths), sum(lengths)], dtype=torch.float64, device=device)
            dist.all_reduce(counts)
            assert torch.isfinite(value)
            (value * (world * len(lengths) / counts[0])).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5, error_if_nonfinite=True)
            opt.step()
            total = value.detach().double() * len(lengths)
            dist.all_reduce(total)
            running['loss_sum'] += float(total)
            running['scenarios'] += int(counts[0])
            running['members'] += int(counts[1])
            if (batch_index + 1) % 25 == 0 and rank == 0:
                write(out / 'progress.json', dict(epoch=epoch, phase=phase, batch=batch_index + 1,
                                                  batches=len(sampler.batches()),
                                                  loss=running['loss_sum'] / running['scenarios'],
                                                  scenarios=running['scenarios'],
                                                  elapsed_seconds=time.monotonic() - began))
            stop = torch.tensor(int(time.monotonic() - began > args.max_hours * 3600), device=device)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if (batch_index + 1) % 100 == 0 or bool(stop):
                snapshot(epoch, batch_index + 1, running)
            if bool(stop):
                if rank == 0:
                    write(out / 'needs_resume.json', dict(epoch=epoch, next_batch=batch_index + 1))
                dist.destroy_process_group()
                return
        assert running['scenarios'] == len(tr) and running['members'] == len(tr.pairs), running
        row = dict(epoch=epoch, phase=phase, train_loss=running['loss_sum'] / len(tr),
                   scenarios=len(tr), members=len(tr.pairs), learning_rates=rates,
                   seconds_this_segment=time.monotonic() - epoch_started)
        state = dict(model=net.state_dict(), model_config=net.config, epoch=epoch,
                     seed=seed, protocol_sha256=protocol_sha, parent_sha256=parent_sha)
        if args.smoke or epoch == 0 or (epoch + 1) % 11 == 0 or epoch + 1 == epochs:
            result = evaluate(net, va, rank, world, times, device, workers)
            score, eligible = quality(result, reference)
            row.update(validation=result['summary'], relative_score=score, eligible_vs_parent=eligible)
            if score < best:
                best = score
                if rank == 0:
                    save(out / 'best.pt', state)
                    write(out / 'validation_best.json', result)
            if eligible and score < eligible_best:
                eligible_best = score
                if rank == 0:
                    save(out / 'eligible.pt', state)
                    write(out / 'validation_eligible.json', result)
        history.append(row)
        running = dict(loss_sum=0., scenarios=0, members=0)
        next_batch = 0
        if rank == 0:
            write(out / 'history.json', history)
            print(row, flush=True)
        snapshot(epoch + 1, 0, running)
    # All 11 times are required before any model can be considered for replacement.
    best_state = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
    net.load_state_dict(best_state['model'])
    full = evaluate(net, va, rank, world, (10,) if args.smoke else tuple(range(11)), device, workers)
    if rank == 0:
        write(out / 'full_validation_best.json', full)
        write(out / 'complete.json', dict(status='training_complete_validation_review_required', epochs=epochs,
                                          smoke=args.smoke, best_score=best,
                                          eligible_checkpoint=(out / 'eligible.pt').exists(),
                                          checkpoint_sha256=sha256(out / 'best.pt'), test_evaluated=False))
        (out / 'needs_resume.json').unlink(missing_ok=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
