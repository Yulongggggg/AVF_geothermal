"""Training restarts retain evaluated checkpoints and reject changed input data."""
import copy
import importlib
import json

import pytest
import torch

from geouno.training import final_protocol as protocol
from geouno.training import train
from geouno.training.data import FieldPairs
from geouno.training.synthetic import make_synthetic_pack


def metrics(value=1.0):
    row = dict(T_rmse=value, P_rmse=value, inj_P90=1.0, prod_P90=1.0, P_scenario_P90=1.0)
    return dict(summary={str(t): dict(row) for t in (4, 7, 10)} | {'all_evaluated_times': dict(row)})


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature_head = torch.nn.Linear(1, 1, bias=False)
        self.config = {}

    def forward(self, field, scalar):
        return self.temperature_head(field)


class TinyData:
    class Pairs(list):
        names = ['scenario']

    pairs = Pairs([0])

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return torch.ones(1, 1), torch.zeros(1, 1), torch.zeros(1, 1), 1


def saved_stage(root, stage, last_epoch=1, selected_epoch=0, legacy=False):
    folder = root / stage
    folder.mkdir()
    model = TinyModel()
    with torch.no_grad():
        model.temperature_head.weight.fill_(2)
    best = dict(model=copy.deepcopy(model.state_dict()), model_config={}, stage=stage, epoch=selected_epoch, seed=1)
    if not legacy:
        best.update(selection_score=0.5, validation=metrics(0.5))
    torch.save(best, folder / 'best.pt')
    with torch.no_grad():
        model.temperature_head.weight.fill_(3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train.STAGES[stage]['lr'], weight_decay=1e-4)
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng_cpu=torch.get_rng_state(),
                    rng_cuda=[], epoch=last_epoch, best=0.95,
                    history=[dict(epoch=0, stage_score=0.5)]), folder / 'last.pt')
    (folder / 'validation_start.json').write_text(json.dumps(metrics()))
    with torch.no_grad():
        model.temperature_head.weight.fill_(1)
    return model, folder


@pytest.mark.parametrize('stage', ['temperature', 'joint'])
@pytest.mark.parametrize('legacy', [False, True])
def test_resume_after_last_epoch_retains_selected_weights(tmp_path, monkeypatch, stage, legacy):
    model, folder = saved_stage(tmp_path, stage, legacy=legacy)
    original_start = (folder / 'validation_start.json').read_bytes()
    monkeypatch.setattr(train, 'evaluate', lambda *a, **k: pytest.fail('Entry validation must be reused'))
    train.run_stage(stage, model, 'resid64_p1', 1, tmp_path, 'cpu', True, TinyData(), TinyData(), 0, {})
    assert model.temperature_head.weight.item() == 2.0
    assert (folder / 'validation_start.json').read_bytes() == original_start
    complete = json.loads((folder / 'complete.json').read_text())
    assert complete['selected_epoch'] == 0 and complete['best'] == 0.5


def test_best_saved_before_last_checkpoint_survives_replayed_epoch(tmp_path, monkeypatch):
    model, folder = saved_stage(tmp_path, 'joint', last_epoch=0, selected_epoch=1)
    model.scales, model.pressure_residual_scale = [1., 1.], 1.
    monkeypatch.setattr(train, 'evaluate', lambda *a, **k: metrics(0.9))

    def loss(pred, *args, **kwargs):
        value = pred.square().mean()
        return value, {'loss': value}

    monkeypatch.setattr(train, 'objective', loss)
    train.run_stage('joint', model, 'resid64_p1', 1, tmp_path, 'cpu', True, TinyData(), TinyData(), 0, {})
    assert model.temperature_head.weight.item() == 2.0
    complete = json.loads((folder / 'complete.json').read_text())
    assert complete['best'] == 0.5 and complete['selected_epoch'] == 1
    assert json.loads((folder / 'validation_best.json').read_text()) == metrics(0.5)


def test_resume_rejects_missing_entry_validation(tmp_path):
    model, folder = saved_stage(tmp_path, 'joint')
    (folder / 'validation_start.json').unlink()
    with pytest.raises(ValueError, match='entry validation'):
        train.run_stage('joint', model, 'resid64_p1', 1, tmp_path, 'cpu', True, TinyData(), TinyData(), 0, {})


@pytest.fixture
def frozen_inputs(tmp_path, monkeypatch):
    pack = tmp_path / 'pack'
    pack.mkdir()
    (pack / 'stats.json').write_text('{}')
    (pack / 'meta.csv').write_text('row,scen,split\n0,a,train\n')
    for name in ('real.npy', 'fields.npy', 'scal.npy', '_ref_times.npy'):
        (pack / name).write_bytes(b'fixed input')
    bank, reference = tmp_path / 'bank.pt', tmp_path / 'reference.json'
    bank.write_bytes(b'fixed bank')
    reference.write_text('{}')
    for name, value in dict(PACK=pack, BANK=bank, OUT=tmp_path / 'run', REFERENCE=reference).items():
        monkeypatch.setattr(protocol, name, value)
    return dict(basis_sha256=bank, reference_validation_sha256=reference,
                stats_sha256=pack / 'stats.json', meta_sha256=pack / 'meta.csv', times_sha256=pack / '_ref_times.npy')


def test_unchanged_frozen_inputs_can_resume(frozen_inputs):
    original = protocol.freeze('resid64_p1')
    assert protocol.freeze('resid64_p1') == original
    assert 'inputs.py' in original['sources_sha256']


@pytest.mark.parametrize('key', ['basis_sha256', 'reference_validation_sha256', 'stats_sha256',
                                 'meta_sha256', 'times_sha256'])
def test_freeze_rejects_changed_input_file(frozen_inputs, key):
    protocol.freeze('resid64_p1')
    frozen_inputs[key].write_bytes(b'changed')
    with pytest.raises(ValueError, match=key):
        protocol.freeze('resid64_p1')


def test_freeze_rejects_changed_batch_size(frozen_inputs, monkeypatch):
    protocol.freeze('resid64_p1')
    monkeypatch.setattr(protocol, 'BATCH_SCENARIOS', protocol.BATCH_SCENARIOS + 1)
    with pytest.raises(ValueError, match='batch_scenarios'):
        protocol.freeze('resid64_p1')


@pytest.mark.parametrize('filename', ['real.npy', 'fields.npy', 'scal.npy'])
def test_freeze_rejects_replaced_data_array(frozen_inputs, filename):
    protocol.freeze('resid64_p1')
    (protocol.PACK / filename).write_bytes(b'replacement data array')
    with pytest.raises(ValueError, match='pack_arrays'):
        protocol.freeze('resid64_p1')


@pytest.fixture
def evaluation_cache(tmp_path, monkeypatch):
    module = importlib.import_module('geouno.training.evaluate')
    pack = make_synthetic_pack(tmp_path / 'pack', train=1, val=1, members=2)
    monkeypatch.setattr(protocol, 'PACK', pack)
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'cache identity fixture')
    data = FieldPairs(pack, 'val', scenario_limit=2, member_limit=2, seed=20260913)
    identity = module.evaluation_identity(checkpoint, data, 'val', True, True, 'cpu')
    result = dict(status='complete', checkpoint_sha256=module.sha(checkpoint), smoke=True,
                  evaluation_identity=identity, records=[], summary={})
    monkeypatch.setattr(module, 'load', lambda *a, **k: pytest.fail('Cache validation must precede model loading'))
    return module, checkpoint, tmp_path, result


def test_evaluation_reuses_matching_complete_cache(evaluation_cache):
    module, checkpoint, folder, result = evaluation_cache
    (folder / 'val.json').write_text(json.dumps(result))
    assert module.evaluate(checkpoint, 'val', folder, 'cpu', smoke=True) == result


def test_fresh_evaluation_writes_and_reuses_complete_and_partial_identity(evaluation_cache, monkeypatch):
    module, checkpoint, folder, _ = evaluation_cache

    class ZeroFields(torch.nn.Module):
        def forward(self, field, scalar):
            return field.new_zeros((len(field), 2, 160, 128))

    monkeypatch.setattr(module, 'load', lambda _: ZeroFields())
    monkeypatch.setattr(module, 'load_scope', lambda: {
        row: dict(high_pressure_training_p90=False, above_generator_envelope=False) for row in range(6)
    })
    result = module.evaluate(checkpoint, 'val', folder, 'cpu', smoke=True)
    assert result['scenario_count'] == 1 and result['actual_members'] == 2
    assert result['summary']['10']['T_paired_rmse']['mean'] > 0
    partial = json.loads((folder / 'val.partial.json').read_text())
    assert partial['evaluation_identity'] == result['evaluation_identity']
    (folder / 'val.json').unlink()
    resumed = module.evaluate(checkpoint, 'val', folder, 'cpu', smoke=True)
    assert resumed == result
    monkeypatch.setattr(module, 'load', lambda _: pytest.fail('Matching completed evaluation must be reused'))
    assert module.evaluate(checkpoint, 'val', folder, 'cpu', smoke=True) == result


@pytest.mark.parametrize('partial', [False, True])
@pytest.mark.parametrize('change', ['precision', 'smoke', 'split', 'selection', 'meta', 'stats', 'source', 'legacy'])
def test_evaluation_rejects_stale_complete_and_partial_caches(evaluation_cache, partial, change):
    module, checkpoint, folder, result = evaluation_cache
    identity = result['evaluation_identity']
    if change == 'precision':
        identity['fp32'] = False
    elif change == 'smoke':
        identity['smoke'] = False
    elif change == 'split':
        identity['split'] = 'train'
    elif change == 'selection':
        identity['selection_sha256'] = 'changed'
    elif change in ('meta', 'stats'):
        name = 'meta.csv' if change == 'meta' else 'stats.json'
        identity['data_sha256'][name] = 'changed'
    elif change == 'source':
        identity['sources_sha256']['inputs.py'] = 'changed'
    else:
        del result['evaluation_identity']
    path = folder / ('val.partial.json' if partial else 'val.json')
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='different or missing'):
        module.evaluate(checkpoint, 'val', folder, 'cpu', smoke=True)
