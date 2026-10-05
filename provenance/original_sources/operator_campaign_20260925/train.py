"""Distributed, equal-scenario transfer learning with resumable fine tuning.

No test data, manuscript writes, parent modifications, or automatic SOTA claims.
Global batches are partitioned without duplication; unequal final shards receive
the exact DDP scaling world_size * local_scenarios / global_scenarios.
"""
import argparse
import math
import os
import random
import time
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from direct_operator.data import ScenarioPairs, EpochScenarios, collate
from joint_mmd.loss import objective
from joint_precision.metrics import records, summarize
from .common import PACK, OUT, BANK, PARENT, PARENT_SHA, read, write, save, sha
from .model import GeoUNOBasisHead


class ShardedBatches:
    def __init__(self, data, seed, rank, world, batch=16):
        self.base = EpochScenarios(data, seed)
        self.rank, self.world, self.batch = rank, world, batch
        self.start = 0

    def batches(self):
        all_items = list(self.base)
        batches = [all_items[j:j+self.batch] for j in range(0, len(all_items), self.batch)]
        # Merge a small tail so that every rank always has a real scenario.
        if len(batches) > 1 and len(batches[-1]) < self.world:
            tail = batches.pop()
            batches[-1].extend(tail)
        return batches

    def __len__(self):
        return len(self.batches())-self.start

    def __iter__(self):
        for batch in self.batches()[self.start:]:
            idx = np.array_split(np.arange(len(batch)), self.world)[self.rank]
            assert len(idx)
            yield [batch[int(j)] for j in idx]


@torch.no_grad()
def evaluate(net, data, rank, world, times, workers=2):
    net.eval()
    sampler = [(i, ti) for i in range(len(data)) for ti in times][rank::world]
    loader = DataLoader(data, batch_size=1, sampler=sampler, collate_fn=collate,
                        num_workers=workers, pin_memory=True)
    rows = []
    for (i, ti), (f, c, y, lengths) in zip(sampler, loader):
        f, c, y = [v.cuda(non_blocking=True) for v in (f, c, y)]
        outputs = []
        for j in range(0, len(f), 8):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                outputs.append(net(f[j:j+8], c[j:j+8]).float())
        pred = torch.cat(outputs)
        assert torch.isfinite(pred).all()
        rows.append(dict(scen=data.pairs.names[i], time_index=ti,
                         **records(pred, y, f, lengths)[0]))
    gathered = [None]*world
    dist.all_gather_object(gathered, rows)
    rows = sorted([r for part in gathered for r in part], key=lambda r: (r['scen'], r['time_index']))
    assert len(rows) == len(data)*len(times)
    assert len({(r['scen'], r['time_index']) for r in rows}) == len(rows)
    return dict(scenario_count=len(data), actual_members=len(data.pairs), times=list(times),
                summary=summarize(rows), records=rows)


def quality(result, reference):
    a, b = result['summary'], reference['summary']
    ratios = [a[t][k]/b[t][k] for t in ('10', 'all_evaluated_times') for k in ('T_rmse', 'P_rmse')]
    tails = ('inj_P90', 'prod_P90', 'P_scenario_P90', 'P_shape_mse', 'inj_rmse', 'prod_rmse')
    eligible = max(ratios) <= 1.0 and all(a['10'][k] <= 1.02*b['10'][k] for k in tails if b['10'][k] is not None)
    return float(np.mean(ratios)), bool(eligible)


def phase_rates(epoch, epochs, smoke=False):
    warm, joint = (1, 2) if smoke else (11, 66)
    if epoch < warm:
        scale = min(1., (epoch+1)/3)
        return 'head_warmup', dict(new=1e-4*scale, shared=0., temperature=0., pressure=0., level=0.)
    if epoch < joint:
        return 'joint_transfer', dict(new=3e-5, shared=1e-6, temperature=1e-5, pressure=3e-6, level=1e-5)
    scale = .1+.9*(1+math.cos(math.pi*(epoch-joint)/max(1, epochs-joint)))/2
    return 'fine_tune', {k: v*scale for k, v in dict(new=1e-5, shared=3e-7, temperature=3e-6, pressure=1e-6, level=3e-6).items()}


def bucket(name):
    if name.startswith(('trunk.', 'coefficient_delta.')): return 'new'
    if name.startswith('temperature_'): return 'temperature'
    if name.startswith('level_head.'): return 'level'
    if name.startswith(('pressure_', 'geology_')): return 'pressure'
    return 'shared'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pod-rank', type=int, choices=(512, 1024), required=True)
    ap.add_argument('--seed', type=int, default=20260925)
    ap.add_argument('--epochs', type=int, default=132)
    ap.add_argument('--max-hours', type=float, default=22.5)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(1)
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    began = time.monotonic()
    seed = args.seed
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    random.seed(seed); np.random.seed(seed)
    assert sha(PARENT) == PARENT_SHA
    protocol = read(OUT/'protocol.json')
    assert sha(PACK/'meta.csv') == protocol['meta_sha256']
    assert sha(PACK/'stats.json') == protocol['stats_sha256']
    assert sha(BANK) == read(BANK.parent/'complete.json')['bank_sha256']
    for name, digest in protocol['sources_sha256'].items():
        assert sha(PACK.parent/name) == digest, name
    protocol_sha = sha(OUT/'protocol.json')
    out = OUT/('smoke' if args.smoke else 'formal')/f'r{args.pod_rank}_s{seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        dist.destroy_process_group(); return
    stats = read(PACK/'stats.json')
    net = GeoUNOBasisHead(stats['fields_std'], stats['p_res_std_mpa'], pod_rank=args.pod_rank)
    copied = net.initialize_parent(PARENT)
    net.initialize_pod(BANK)
    net.cuda()
    groups = {k: [] for k in ('new', 'shared', 'temperature', 'pressure', 'level')}
    for name, p in net.named_parameters(): groups[bucket(name)].append(p)
    opt = torch.optim.AdamW([dict(params=p, name=k, lr=1e-5) for k, p in groups.items()], weight_decay=1e-4)
    epochs = 3 if args.smoke else args.epochs
    tr = ScenarioPairs(PACK, 'train', 13 if args.smoke else 0, 2 if args.smoke else 0)
    va = ScenarioPairs(PACK, 'val', 4 if args.smoke else 0, 2 if args.smoke else 0)
    assert not set(tr.pairs.names)&set(va.pairs.names)
    if not args.smoke:
        assert (len(tr), len(tr.pairs), len(va), len(va.pairs)) == (6582, 72939, 542, 6275)
    workers = 0 if args.smoke else 2
    sampler = ShardedBatches(tr, seed+3000, rank, world, batch=8 if args.smoke else 16)
    loader = DataLoader(tr, batch_sampler=sampler, collate_fn=collate, num_workers=workers,
                        persistent_workers=workers > 0, pin_memory=True)
    times = (10,) if args.smoke else (4, 7, 10)
    history, best, eligible_best, start, next_batch = [], float('inf'), float('inf'), 0, 0
    running = dict(loss_sum=0., scenarios=0, members=0)
    if (out/'last.pt').exists():
        old = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        assert old['protocol_sha256'] == protocol_sha and old['world_size'] == world
        assert old['basis_sha256'] == sha(BANK)
        net.load_state_dict(old['model']); opt.load_state_dict(old['optimizer'])
        start, next_batch = old['next_epoch'], old['next_batch']
        history, best, eligible_best, running = old['history'], old['best'], old['eligible_best'], old['running']
        torch.set_rng_state(old['rng_cpu']); torch.cuda.set_rng_state(old['rng_cuda'][rank])
        print(f'resume rank={rank} epoch={start} batch={next_batch}', flush=True)
    ddp = DDP(net, device_ids=[local], broadcast_buffers=False)

    def snapshot(next_epoch, batch_index, stats_running):
        cuda_states = [None]*world
        dist.all_gather_object(cuda_states, torch.cuda.get_rng_state().cpu())
        if rank == 0:
            save(out/'last.pt', dict(model=net.state_dict(), model_config=net.config,
                 optimizer=opt.state_dict(), next_epoch=next_epoch, next_batch=batch_index,
                 history=history, best=best, eligible_best=eligible_best, running=stats_running,
                 rng_cpu=torch.get_rng_state(), rng_cuda=cuda_states, world_size=world,
                 protocol_sha256=protocol_sha, parent_sha256=PARENT_SHA,
                 basis_sha256=sha(BANK)))
        dist.barrier()

    if (out/'reference_validation.json').exists():
        reference = read(out/'reference_validation.json')
    else:
        from encoder_study.model import load as load_parent
        parent = load_parent(PARENT).cuda()
        reference = evaluate(parent, va, rank, world, times, workers)
        del parent
        if rank == 0:
            write(out/'reference_validation.json', reference)
        dist.barrier()
    if not (out/'manifest.json').exists():
        initial = evaluate(net, va, rank, world, times, workers)
        if rank == 0:
            write(out/'initial_validation.json', initial)
            write(out/'manifest.json', dict(model_config=net.config, seed=seed, smoke=args.smoke,
                  world_size=world, protocol_sha256=protocol_sha, copied_parent_tensors=len(copied),
                  parent_sha256=PARENT_SHA, basis_sha256=sha(BANK),
                  parameters=sum(p.numel() for p in net.parameters()),
                  train_scenarios=len(tr), train_actual_members=len(tr.pairs),
                  validation_scenarios=len(va), validation_actual_members=len(va.pairs),
                  meta_sha256=sha(PACK/'meta.csv'), gpu=torch.cuda.get_device_name()))
            print(dict(initial_quality=quality(initial, reference), out=str(out)), flush=True)
        dist.barrier()

    for epoch in range(start, epochs):
        # Fine tuning starts from the validation-selected transfer checkpoint.
        if epoch == (2 if args.smoke else 66) and next_batch == 0:
            chosen = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
            net.load_state_dict(chosen['model'])
            opt.state.clear()
        phase, rates = phase_rates(epoch, epochs, args.smoke)
        for group in opt.param_groups: group['lr'] = rates[group['name']]
        sampler.base.epoch = epoch
        sampler.start = next_batch
        ddp.train()
        epoch_started = time.monotonic()
        for batch_index, (f, c, y, lengths) in enumerate(loader, start=next_batch):
            f, c, y = [v.cuda(non_blocking=True) for v in (f, c, y)]
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = ddp(f, c)
            value, _ = objective(pred, y, lengths, net.scales, net.pressure_residual_scale, weight=.2, kind='rmse')
            counts = torch.tensor([len(lengths), sum(lengths)], dtype=torch.float64, device='cuda')
            dist.all_reduce(counts)
            assert torch.isfinite(value)
            (value*(world*len(lengths)/counts[0])).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5, error_if_nonfinite=True)
            opt.step()
            total = value.detach().double()*len(lengths)
            dist.all_reduce(total)
            running['loss_sum'] += float(total)
            running['scenarios'] += int(counts[0]); running['members'] += int(counts[1])
            if (batch_index+1) % 25 == 0 and rank == 0:
                write(out/'progress.json', dict(epoch=epoch, phase=phase, batch=batch_index+1,
                      batches=len(sampler.batches()), loss=running['loss_sum']/running['scenarios'],
                      scenarios=running['scenarios'], elapsed_seconds=time.monotonic()-began))
                print(read(out/'progress.json'), flush=True)
            stop = torch.tensor(int(time.monotonic()-began > args.max_hours*3600), device='cuda')
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if (batch_index+1) % 100 == 0 or bool(stop):
                snapshot(epoch, batch_index+1, running)
            if bool(stop):
                if rank == 0: write(out/'needs_resume.json', dict(epoch=epoch, next_batch=batch_index+1))
                dist.destroy_process_group(); return
        assert running['scenarios'] == len(tr) and running['members'] == len(tr.pairs), running
        row = dict(epoch=epoch, phase=phase, train_loss=running['loss_sum']/len(tr),
                   scenarios=len(tr), members=len(tr.pairs), learning_rates=rates,
                   seconds_this_segment=time.monotonic()-epoch_started)
        state = dict(model=net.state_dict(), model_config=net.config, epoch=epoch,
                     seed=seed, protocol_sha256=protocol_sha, parent_sha256=PARENT_SHA)
        if args.smoke or epoch == 0 or (epoch+1) % 11 == 0 or epoch+1 == epochs:
            result = evaluate(net, va, rank, world, times, workers)
            score, eligible = quality(result, reference)
            row.update(validation=result['summary'], relative_score=score, eligible_vs_parent=eligible)
            if score < best:
                best = score
                if rank == 0:
                    save(out/'best.pt', state); write(out/'validation_best.json', result)
            if eligible and score < eligible_best:
                eligible_best = score
                if rank == 0:
                    save(out/'eligible.pt', state); write(out/'validation_eligible.json', result)
        history.append(row)
        running = dict(loss_sum=0., scenarios=0, members=0)
        next_batch = 0
        if rank == 0:
            write(out/'history.json', history)
            print(row, flush=True)
        snapshot(epoch+1, 0, running)
    # All 11 times are required before any model can be considered for replacement.
    best_state = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
    net.load_state_dict(best_state['model'])
    full = evaluate(net, va, rank, world, (10,) if args.smoke else tuple(range(11)), workers)
    if rank == 0:
        write(out/'full_validation_best.json', full)
        write(out/'complete.json', dict(status='training_complete_validation_review_required', epochs=epochs,
              smoke=args.smoke, best_score=best, eligible_checkpoint=(out/'eligible.pt').exists(),
              checkpoint_sha256=sha(out/'best.pt'), test_evaluated=False, sota_established=False))
        (out/'needs_resume.json').unlink(missing_ok=True)
    assert sha(PARENT) == PARENT_SHA
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
