"""Protect the old checkpoint and freeze this independent training protocol."""
import platform
import shutil
from pathlib import Path
import torch
from .common import ROOT, PACK, OUT, PARENT, PARENT_SHA, read, write, sha


def main():
    assert sha(PARENT) == PARENT_SHA
    protected = OUT/'retained_original'; protected.mkdir(parents=True, exist_ok=True)
    target = protected/'model.pt'
    if not target.exists(): shutil.copy2(PARENT, target)
    assert sha(target) == PARENT_SHA
    for name in ('final_model.json',):
        if not (protected/name).exists(): shutil.copy2(PARENT.parent/name, protected/name)
    paths = [p for p in (ROOT/'operator_campaign_20260925').glob('*.py') if p.name not in ('contract.py', 'freeze.py')]
    paths += [ROOT/name for name in (
        'encoder_study/model.py', 'joint_architecture/model.py', 'direct_operator/model.py',
        'temperature_revision/model.py', 'direct_operator/data.py', 'overnight_bench/data.py',
        'temperature_revision/data.py', 'det_sampler.py', 'joint_mmd/loss.py', 'joint_precision/metrics.py')]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}
    protocol = dict(version=4, user_scope='Retain current GeoUNO; convert its output heads to coefficient-times-basis form; add temperature Fourier features; train and fine tune; do not edit manuscript.',
        architecture='geouno_basis_head', retained='All original encoder, attention, temperature/pressure decoders, regional and pressure-level weights.',
        basis='Training-only POD, bilinear spatial extension, plus independent coordinate MLP functions; temperature MLP has Fourier features.',
        temperature_fourier_cycles=[1,2,4,8,16,32], parent=str(PARENT), parent_sha256=PARENT_SHA,
        recipes=[dict(pod_rank=r, learned_rank=64, seed=20260925, gpus=4) for r in (512,1024)],
        rank_rationale='Preflight validation projection errors at rank256/512 were T=0.399/0.209 K. Rank256 risks a restrictive thermal error floor; compare 512/1024 instead. No fitted basis used validation.',
        epochs=132, stages=dict(head_warmup=[0,10], joint_transfer=[11,65], fine_tune=[66,131]),
        fine_tune_initialization='Best validation-selected transfer checkpoint, optimizer reset.',
        loss='Equal-scenario paired RMSE + 0.2 Gaussian empirical squared MMD, identical to frozen parent final loss.',
        sampling='All 6582 training scenarios, all 72939 actual members each epoch; each scenario sees all 11 times in each 11-epoch cycle; no nominal or duplicated/padded scenarios.',
        global_batch_scenarios=16, validation='All 542 validation scenarios / 6275 actual members at times 4,7,10 for checkpoint selection; final all 11 times.',
        selection='Mean ratios of T/P RMSE at final time and across evaluation times vs frozen parent. Separate eligible checkpoint also protects pressure tails, wells, and shape.',
        test_policy='No test evaluation or tuning in training jobs. Independent seeds, frozen finalist and baseline review required for comparative claims.',
        sota_policy='Not established by architecture name, a single warm-start run, or validation scores alone. Report added compute and inherited training.',
        manuscript_writes=False, original_checkpoint_writes=False,
        meta_sha256=sha(PACK/'meta.csv'), stats_sha256=sha(PACK/'stats.json'),
        sources_sha256=hashes, torch_version=torch.__version__, python=platform.python_version())
    path = OUT/'protocol.json'
    if path.exists():
        assert read(path) == protocol, 'Existing protocol differs; do not silently mutate a running experiment.'
    else:
        write(path, protocol)
        for source in paths:
            dest = OUT/'source_snapshot'/source.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    write(protected/'manifest.json', dict(original=str(PARENT), copy=str(target), sha256=PARENT_SHA,
                                        original_source_config=str(PARENT.parent/'final_model.json')))
    print(dict(protocol=str(path), protocol_sha256=sha(path), retained_checkpoint_sha256=sha(target)), flush=True)


if __name__ == '__main__':
    main()
