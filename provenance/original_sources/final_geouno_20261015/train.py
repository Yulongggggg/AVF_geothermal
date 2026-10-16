"""Three-stage training of the final simplified GeoUNO (see common.py). Restart-safe at stage granularity and,
within a stage, at epoch granularity (last.pt)."""
import argparse
from pathlib import Path
import math
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from direct_operator.data import ScenarioPairs, EpochScenarios, collate
from joint_mmd.loss import objective, mmd2
from joint_precision.metrics import evaluate
from temperature_revision.train import atomic_checkpoint
from fair_comparison.train import score
from .model import HeadAblation
from .common import PACK, BANK, REFERENCE, STAGES, BATCH_SCENARIOS, folder, freeze, read, sha, write

TEMPERATURE_PREFIXES = ('temperature_decoder.', 'temperature_head.', 'trunk.temperature.')


def build(stats, arch):
    net = HeadAblation(stats['fields_std'], stats['fields_std'][1]*stats['p_res_std_norm'], arch)
    net.initialize_pod(BANK)
    return net


def temperature_loss(pred, y, lengths, net, mmd_weight=.2):
    """Temperature-only version of joint_mmd.loss.objective (RMSE kind): cooling, cap/bed cooling, cooling MMD."""
    p, y = pred.float(), y.float(); s = net.scales.flatten()[0]; e = (p-y)[:, :1]
    def equal(v):
        return torch.stack([(g.mean()+1e-12).sqrt() for g in torch.split(v, lengths)]).mean()
    parts = dict(temperature_rmse=equal((e[:, :, 16:144]/s).square().flatten(1).mean(1)))
    parts['outside_rmse'] = .1*equal((torch.cat((e[:, :, :16], e[:, :, 144:]), 2)/s).square().flatten(1).mean(1))
    a = (p[:, 0, 16:144]/s).flatten(1); b = (y[:, 0, 16:144]/s).flatten(1)
    parts['mmd'] = mmd_weight*torch.stack([mmd2(x, z) for x, z in zip(torch.split(a, lengths), torch.split(b, lengths))]).mean()
    return sum(parts.values()), parts


def fine_lr(epoch, epochs, peak):
    return peak*(.1+.9*(1+math.cos(math.pi*epoch/max(1, epochs-1)))/2)*min(1., (epoch+1)/3)


def admissible(result, start, stage):
    """Stage selection relative to the stage-entry checkpoint (3-time validation)."""
    a, b = result['summary'], start['summary']
    times = [t for t in a if t != 'all_evaluated_times']
    def ratio(k):
        return [np.mean([a[t][k] for t in times])/max(np.mean([b[t][k] for t in times]), 1e-20), a['10'][k]/max(b['10'][k], 1e-20)]
    if stage == 'temperature':
        r = ratio('T_rmse')
        unchanged = all(math.isclose(a[t]['P_rmse'], b[t]['P_rmse'], rel_tol=2e-3, abs_tol=1e-6) for t in times)
        return float(np.mean(r)), bool(max(r) <= 1.+1e-10 and unchanged)
    r = ratio('T_rmse')+ratio('P_rmse')
    tails = all(a['10'][k] <= 1.02*b['10'][k]+1e-12 for k in ('inj_P90', 'prod_P90', 'P_scenario_P90') if b['10'][k] is not None)
    return float(np.mean(r)), bool(max(r) <= 1.+1e-10 and tails)


def run_stage(stage, net, arch, seed, out, device, smoke, tr, va, workers, stats):
    cfg = STAGES[stage]; sdir = out/stage; sdir.mkdir(parents=True, exist_ok=True)
    if (sdir/'complete.json').exists():
        net.load_state_dict(torch.load(sdir/'best.pt', map_location=device, weights_only=False)['model'])
        return
    epochs = 2 if smoke else cfg['epochs']
    times = (10,) if smoke else (4, 7, 10)
    if stage == 'temperature':
        active = [p for n, p in net.named_parameters() if n.startswith(TEMPERATURE_PREFIXES)]
        for n, p in net.named_parameters():
            p.requires_grad_(n.startswith(TEMPERATURE_PREFIXES))
    else:
        active = list(net.parameters())
        for p in active:
            p.requires_grad_(True)
    opt = torch.optim.AdamW(active, lr=cfg['lr'], weight_decay=1e-4)
    if stage == 'main':
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: .05+.95*(1+math.cos(math.pi*e/epochs))/2)
        reference = read(REFERENCE) if not smoke else evaluate(net, va, device, times, workers, chunk_size=16)
    else:
        sched = None
    start = evaluate(net, va, device, times, workers, chunk_size=16) if stage != 'main' else None
    if stage != 'main':
        atomic_checkpoint(dict(model=net.state_dict(), model_config=net.config, stage=stage, epoch=-1, seed=seed), sdir/'best.pt')
        write(sdir/'validation_start.json', start)
    best, history, first = (float('inf') if stage == 'main' else 1.), [], 0
    if (sdir/'last.pt').exists():
        old = torch.load(sdir/'last.pt', map_location=device, weights_only=False)
        net.load_state_dict(old['model']); opt.load_state_dict(old['optimizer'])
        if sched: sched.load_state_dict(old['scheduler'])
        torch.set_rng_state(old['rng_cpu'].cpu())
        if device == 'cuda': torch.cuda.set_rng_state_all([x.cpu() for x in old['rng_cuda']])
        first, best, history = old['epoch']+1, old['best'], old['history']
    sampler = EpochScenarios(tr, seed+3000+{'main': 0, 'temperature': 1, 'joint': 2}[stage])
    loader = DataLoader(tr, batch_size=BATCH_SCENARIOS, sampler=sampler, collate_fn=collate, num_workers=workers,
                        persistent_workers=workers > 0, pin_memory=device == 'cuda', prefetch_factor=4 if workers else None)
    offset = {'main': 0, 'temperature': STAGES['main']['epochs'], 'joint': STAGES['main']['epochs']+STAGES['temperature']['epochs']}[stage]
    for ep in range(first, epochs):
        sampler.epoch = offset+ep
        if sched is None:
            for g in opt.param_groups: g['lr'] = fine_lr(ep, epochs, cfg['lr'])
        net.train(); totals, count, members, began = {}, 0, 0, time.monotonic()
        for f, c, y, lengths in loader:
            f, c, y = [v.to(device, non_blocking=True) for v in (f, c, y)]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
                pred = net(f, c)
            if stage == 'temperature':
                value, parts = temperature_loss(pred.float(), y, lengths, net)
            else:
                value, parts = objective(pred.float(), y, lengths, net.scales, net.pressure_residual_scale, weight=.2, kind='rmse')
            assert torch.isfinite(value)
            value.backward()
            torch.nn.utils.clip_grad_norm_(active, 5, error_if_nonfinite=True)
            opt.step()
            for k, v in parts.items(): totals[k] = totals.get(k, 0)+float(v.detach())*len(lengths)
            count += len(lengths); members += sum(lengths)
        assert count == len(tr) and members == len(tr.pairs)
        if sched: sched.step()
        row = dict(stage=stage, epoch=ep, seconds=time.monotonic()-began, lr=opt.param_groups[0]['lr'],
                   **{k: v/count for k, v in totals.items()})
        state = dict(model=net.state_dict(), model_config=net.config, stage=stage, epoch=ep, seed=seed)
        if smoke or ep == 0 or (ep+1) % 11 == 0 or ep+1 == epochs:
            val = evaluate(net, va, device, times, workers, chunk_size=16)
            if stage == 'main':
                s = score(val, reference); row.update(validation=val['summary'], score=s)
                if s < best: best = s; atomic_checkpoint(state, sdir/'best.pt'); write(sdir/'validation_best.json', val)
            else:
                s, ok = admissible(val, start, stage); row.update(validation=val['summary'], stage_score=s, admissible=ok)
                if ok and s < best: best = s; atomic_checkpoint(state, sdir/'best.pt'); write(sdir/'validation_best.json', val)
        history.append(row)
        atomic_checkpoint(dict(**state, optimizer=opt.state_dict(), scheduler=sched.state_dict() if sched else None,
                               history=history, best=best, rng_cpu=torch.get_rng_state(),
                               rng_cuda=torch.cuda.get_rng_state_all() if device == 'cuda' else []), sdir/'last.pt')
        write(sdir/'history.json', history); print(row, flush=True)
    chosen = torch.load(sdir/'best.pt', map_location=device, weights_only=False)
    net.load_state_dict(chosen['model'])
    write(sdir/'complete.json', dict(status='stage_complete', stage=stage, epochs=epochs, selected_epoch=chosen['epoch'],
                                     best=best, checkpoint_sha256=sha(sdir/'best.pt')))


def adopt_main(src, out, arch, seed, smoke):
    """Use an identical-protocol 99-epoch run (simple_geouno_20261015 / simple_geouno_lvl_20261015 fast protocol) as the main stage.
    Waits for its training to reach the last epoch; the evaluation phase of that job is not needed."""
    sdir = out/'main'
    if (sdir/'complete.json').exists():
        return
    need = 2 if smoke else STAGES['main']['epochs']
    while True:
        h = read(src/'history.json') if (src/'history.json').exists() else []
        if len(h) >= need and (src/'best.pt').exists():
            break
        print(dict(waiting_for=str(src), epochs_done=len(h), need=need), flush=True); time.sleep(60)
    time.sleep(30)  # let the final atomic checkpoint write settle
    man = read(src/'manifest.json'); rec = man['recipe']
    assert rec['variant'] == arch and rec['epochs'] == STAGES['main']['epochs'] and rec['batch_scenarios'] == BATCH_SCENARIOS
    assert rec['lr'] == STAGES['main']['lr'] and rec['initialization'] == 'scratch' and rec['seed'] == seed
    assert smoke or man['train_members'] == 72939
    ck = torch.load(src/'best.pt', map_location='cpu', weights_only=False)
    assert ck['model_config']['architecture'] == 'geouno_simple' and ck['model_config']['variant'] == arch
    proto = next(q/'protocol.json' for q in src.parents if (q/'protocol.json').exists())
    sdir.mkdir(parents=True, exist_ok=True)
    atomic_checkpoint(dict(model=ck['model'], model_config=ck['model_config'], stage='main', epoch=ck['epoch'],
                           seed=rec['seed'], adopted_from=str(src/'best.pt')), sdir/'best.pt')
    write(sdir/'history.json', h); write(sdir/'validation_best.json', read(src/'validation_best.json'))
    write(sdir/'complete.json', dict(status='stage_complete', stage='main', adopted_from=str(src),
          source_best_sha256=sha(src/'best.pt'), source_protocol=str(proto),
          source_protocol_sha256=sha(proto), selected_epoch=ck['epoch'],
          note='Identical fast protocol (99 epochs, 8 scenarios/step, AdamW 3e-4 cosine to 0.05x, RMSE+0.2 MMD, '
               'validation-selected best.pt); seed '+str(rec['seed'])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--main-from', default=None, help='run folder of a completed 99-epoch fast-protocol run of the same '
                    'architecture (simple_geouno_20261015 or simple_geouno_lvl_20261015); its validation-selected best.pt is adopted as the main stage')
    a = ap.parse_args()
    protocol = freeze(a.arch)
    device = 'cpu' if a.cpu else 'cuda'
    if device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('GPU allocation required')
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = True
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    out = folder(a.arch, a.seed, a.smoke)
    if a.smoke: out = out/device
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists(): return
    stats = read(PACK/'stats.json')
    net = build(stats, a.arch).to(device)
    torch.manual_seed(a.seed+1000); torch.cuda.manual_seed_all(a.seed+1000)
    tr = ScenarioPairs(PACK, 'train', 2 if a.smoke else 0, 2 if a.smoke else 0)
    va = ScenarioPairs(PACK, 'val', 2 if a.smoke else 0, 2 if a.smoke else 0)
    assert not set(tr.pairs.names) & set(va.pairs.names)
    if not a.smoke:
        assert (len(tr), len(tr.pairs), len(va), len(va.pairs)) == (6582, 72939, 542, 6275), 'nominal records must not enter'
    write(out/'manifest.json', dict(arch=a.arch, seed=a.seed, smoke=a.smoke, protocol_sha256=sha(out.parents[1 if not a.smoke else 2]/f'protocol_{a.arch}.json'),
                                    parameters=sum(p.numel() for p in net.parameters()), train_scenarios=len(tr), train_members=len(tr.pairs),
                                    gpu=torch.cuda.get_device_name() if device == 'cuda' else 'cpu'))
    workers = 0 if a.cpu or a.smoke else 10
    if a.main_from:
        adopt_main(Path(a.main_from), out, a.arch, a.seed, a.smoke)
    for stage in ('main', 'temperature', 'joint'):
        run_stage(stage, net, a.arch, a.seed, out, device, a.smoke, tr, va, workers, stats)
    final = out/'final.pt'
    atomic_checkpoint(dict(torch.load(out/'joint/best.pt', map_location='cpu', weights_only=False), final=True), final)
    from .evaluate import evaluate as full_evaluate
    d = out/'final_eval'; d.mkdir(exist_ok=True)
    full_evaluate(final, 'val', d, device, a.smoke, fp32=True)
    full_evaluate(final, 'test', d, device, a.smoke, fp32=True)
    write(out/'complete.json', dict(status='complete', arch=a.arch, seed=a.seed, checkpoint_sha256=sha(final),
                                    nominal_records_used=False, smoke=a.smoke))


if __name__ == '__main__':
    main()
