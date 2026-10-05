"""Training components and an end-to-end CPU smoke run on a synthetic pack."""
import json
import math
import os

import numpy as np
import pytest
import torch

from geouno.training.data import ScenarioPairs
from geouno.training.synthetic import make_synthetic_pack
from geouno.training.train import ShardedBatches, phase_rates


def test_schedule_has_paper_phase_lengths():
    phases = [phase_rates(e, 132)[0] for e in range(132)]
    assert [phases.count(k) for k in ('head_warmup', 'joint_transfer', 'fine_tune')] == [11, 55, 66]
    # Cosine decay from 1e-5 towards 10 % (reached at epoch 132, i.e. just after the last epoch).
    expected = 1e-5 * (.1 + .9 * (1 + math.cos(math.pi * 65 / 66)) / 2)
    assert phase_rates(131, 132)[1]['new'] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize('size', [13, 17, 23])
@pytest.mark.parametrize('world', [2, 4])
def test_sharded_batches_cover_every_scenario_once(size, world):
    class Fake:
        def __len__(self):
            return size
    fake = Fake()
    fake.pairs = type('Names', (), dict(names=[str(i) for i in range(size)]))()
    rank_batches = [list(ShardedBatches(fake, 7, r, world, 16)) for r in range(world)]
    assert len({len(b) for b in rank_batches}) == 1
    seen = sorted(i for batches in rank_batches for b in batches for i, _ in b)
    assert seen == list(range(size))


def test_dataset_groups_members_by_scenario(tmp_path):
    pack = make_synthetic_pack(tmp_path / 'pack', train=4, val=2, members=3)
    data = ScenarioPairs(pack, 'train')
    field, scalar, truth, n = data[(0, 10)]
    assert len(data) == 4 and n == 3
    assert field.shape == (3, 6, 160, 128) and scalar.shape == (3, 5) and truth.shape == (3, 2, 160, 128)


@pytest.mark.slow
def test_training_smoke_run_on_cpu(tmp_path, monkeypatch):
    """Three smoke epochs of the real training loop (gloo, one process) on random data."""
    from geouno.model.geology import GeologyJointOperator
    from geouno.training import train
    pack = make_synthetic_pack(tmp_path / 'pack', train=13, val=4, members=2)
    stats = json.loads((pack / 'stats.json').read_text())
    torch.manual_seed(0)
    parent = GeologyJointOperator(stats['fields_std'], stats['p_res_std_mpa'], 32, regional=True, gates=False)
    torch.save(dict(model=parent.state_dict(), model_config=parent.config), tmp_path / 'parent.pt')
    q, _ = torch.linalg.qr(torch.randn(20480, 512))
    basis = (q.T * 20480 ** .5).reshape(1, 512, -1).expand(2, -1, -1).clone()
    torch.save(dict(split='train', basis=basis, mean=torch.zeros(2, 20480), rank=512), tmp_path / 'bank.pt')
    monkeypatch.setenv('MASTER_PORT', str(29600 + os.getpid() % 300))
    train.main(['--pack', str(pack), '--parent', str(tmp_path / 'parent.pt'), '--bank', str(tmp_path / 'bank.pt'),
                '--out', str(tmp_path / 'run'), '--pod-rank', '512', '--smoke', '--device', 'cpu',
                '--skip-parent-hash-check'])
    out = tmp_path / 'run' / 'smoke' / 'r512_s20260925'
    complete = json.loads((out / 'complete.json').read_text())
    history = json.loads((out / 'history.json').read_text())
    assert complete['status'] == 'training_complete_validation_review_required'
    assert [h['phase'] for h in history] == ['head_warmup', 'joint_transfer', 'fine_tune']
    assert all(np.isfinite(h['train_loss']) for h in history)
