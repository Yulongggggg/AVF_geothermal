"""Regressions found while reconciling the release with the manuscript."""
import csv
import gzip
import importlib
import io
import json
import shutil
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from geouno import artifacts, paper, tasks
from geouno.model.geothermal import GeothermalJointOperator, load_parent
from geouno.model.geology import GeologyJointOperator


@pytest.mark.model_execution
def test_geothermal_name_loads_both_checkpoint_tags(tmp_path):
    assert GeologyJointOperator is GeothermalJointOperator
    torch.manual_seed(9)
    net = GeothermalJointOperator([1., 2.], 1., width=8, regional=True, gates=False).eval()
    assert net.config['architecture'] == 'geothermal_task_decoders'
    field, scalar = torch.randn(1, 6, 160, 128), torch.randn(1, 5)
    field[:, 1] = 1
    field.requires_grad_(True)
    reference = net(field, scalar)
    gradient, = torch.autograd.grad(reference.square().mean(), field)
    for tag in ('geology_task_decoders', 'geothermal_task_decoders'):
        path = tmp_path / f'{tag}.pt'
        torch.save(dict(model=net.state_dict(), model_config={**net.config, 'architecture': tag}), path)
        restored = load_parent(path)
        assert type(restored).__name__ == 'GeothermalJointOperator'
        actual = restored(field, scalar)
        actual_gradient, = torch.autograd.grad(actual.square().mean(), field)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(actual_gradient, gradient, rtol=0, atol=0)


@pytest.mark.parametrize('damage', ['delete', 'corrupt'])
def test_artifact_marker_does_not_hide_damaged_files(tmp_path, monkeypatch, damage):
    source = tmp_path / 'source'
    source.mkdir()
    archive = source / 'test.tar.gz'
    payload = b'validated weights'
    with tarfile.open(archive, 'w:gz') as out:
        entry = tarfile.TarInfo('data.bin')
        entry.size = len(payload)
        out.addfile(entry, io.BytesIO(payload))
    import hashlib
    info = dict(assets={'test.tar.gz': dict(group='core', extract='core', sha256=artifacts.sha256(archive),
                files={'data.bin': hashlib.sha256(payload).hexdigest()})})
    monkeypatch.setattr(artifacts, 'manifest', lambda: info)
    root = tmp_path / 'installed'
    artifacts.fetch(['test.tar.gz'], root, source_dir=source, log=lambda *_: None)
    file = root / 'core/data.bin'
    if damage == 'delete':
        file.unlink()
    else:
        file.write_bytes(b'corrupt weights')
        with pytest.raises(RuntimeError, match='SHA-256'):
            artifacts.path('core/data.bin', root)
    artifacts.fetch(['test.tar.gz'], root, source_dir=source, log=lambda *_: None)
    assert file.read_bytes() == payload
    assert all(artifacts.verify(root).values())


def test_task_timing_and_evaluation_counts_exclude_forecast(monkeypatch):
    backend = SimpleNamespace(evaluations=100, device='cpu')
    registry = SimpleNamespace(entry=lambda _: {'observation_seed': 1})
    monkeypatch.setattr(tasks.artifacts, 'Reservoirs', lambda _: registry)
    monkeypatch.setattr(tasks, 'problem_for', lambda *_: (object(), {'channels': [0, 1]}))
    clock = iter([10., 12., 13., 18., 19.])
    monkeypatch.setattr(tasks.time, 'monotonic', lambda: next(clock))

    def infer(*_, **__):
        backend.evaluations += 20
        return np.zeros((1, 4096)), {'seconds': 1.5}

    def predict(*_, **__):
        backend.evaluations += 5
        return {'score': 0.}, {}

    monkeypatch.setattr(tasks, 'esmda', infer)
    monkeypatch.setattr(tasks, 'forecast', predict)
    _, info = tasks.run('multiwell_opt17', 'test_01', 'esmda', backend=backend)
    assert info['seconds'] == 1.5
    assert info['inference_wall_seconds'] == 2 and info['forecast_seconds'] == 5
    assert info['total_seconds'] == 9
    assert info['forward_member_times'] == 20 and info['forecast_member_times'] == 5


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'nan', 'wrong_seed'])
def test_three_seed_report_rejects_invalid_records(tmp_path, monkeypatch, damage):
    folder = tmp_path / 'table1_three_seed_20261015'
    folder.mkdir()
    source = paper.SOURCE / folder.name
    shutil.copy(source / 'cohort.json.gz', folder)
    with gzip.open(source / 'scenario_time_scores.csv.gz', 'rt') as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    if damage == 'missing':
        rows.pop()
    elif damage == 'duplicate':
        rows.append(rows[0].copy())
    elif damage == 'nan':
        rows[0]['T_field_rmse'] = 'nan'
    else:
        rows[0]['seed'] = '123'
    with gzip.open(folder / 'scenario_time_scores.csv.gz', 'wt') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(paper, 'SOURCE', tmp_path)
    with pytest.raises(ValueError):
        paper.geouno_test_runs()


def test_map_restart_checks_posterior_seed_and_options(tmp_path, monkeypatch):
    module = importlib.import_module('geouno.inverse.map')

    class Quadratic:
        data = torch.ones(1, dtype=torch.float64)

        def cache_key(self):
            return 'quadratic-one'

        def tensor(self, x):
            return torch.as_tensor(x, dtype=torch.float64)

        def potential(self, xi):
            return .5 * (xi.square().sum() + (xi[:1] - self.data).square().sum())

    problem = Quadratic()
    options = {'maxiter': 10}
    result, _ = module.map_optimize(problem, 11, options, tmp_path)
    np.testing.assert_allclose(result[0, 0], .5, atol=1e-8)
    monkeypatch.setattr(module, 'minimize', lambda *_, **__: pytest.fail('Valid restart should reuse starts'))
    cached, _ = module.map_optimize(problem, 11, options, tmp_path)
    np.testing.assert_array_equal(result, cached)
    for seed, opt in [(12, options), (11, {'maxiter': 20})]:
        with pytest.raises(ValueError, match='protocol differs'):
            module.map_optimize(problem, seed, opt, tmp_path)
    problem.cache_key = lambda: 'different-posterior'
    with pytest.raises(ValueError, match='protocol differs'):
        module.map_optimize(problem, 11, options, tmp_path)
    assert json.loads((tmp_path / 'protocol.json').read_text())['seed'] == 11
