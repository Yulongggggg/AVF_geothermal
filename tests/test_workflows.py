"""Interrupted runs and failed verification must not be reported as successful."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from geouno import cli, nuts_runner, tasks


@pytest.fixture
def lightweight_task(monkeypatch):
    backend = SimpleNamespace(evaluations=0, device='cpu')
    registry = SimpleNamespace(entry=lambda _: {'observation_seed': 1})
    monkeypatch.setattr(tasks.artifacts, 'Reservoirs', lambda _: registry)
    monkeypatch.setattr(tasks, 'problem_for', lambda *_: (object(), {'channels': [0, 1]}))
    monkeypatch.setattr(tasks, 'esmda', lambda *_, **__: (np.zeros((1, 4096)), {}))
    monkeypatch.setattr(tasks, 'forecast', lambda *_, **__: ({'score': 0.}, {'mean': np.ones(2)}))
    return lambda out: tasks.run('multiwell_opt17', 'test_01', 'esmda', out=out, backend=backend)


@pytest.mark.parametrize('filename', tasks.TASK_FILES)
def test_resume_requires_every_intact_output(tmp_path, lightweight_task, filename):
    lightweight_task(tmp_path)
    assert tasks.task_complete(tmp_path, 'multiwell_opt17', 'test_01', 'esmda')
    assert not tasks.task_complete(tmp_path, 'multiwell_opt17', 'test_02', 'esmda')
    output = tmp_path / filename
    data = output.read_bytes()
    output.write_bytes(b'interrupted')
    assert not tasks.task_complete(tmp_path, 'multiwell_opt17', 'test_01', 'esmda')
    output.write_bytes(data)
    assert tasks.task_complete(tmp_path, 'multiwell_opt17', 'test_01', 'esmda')
    output.unlink()
    assert not tasks.task_complete(tmp_path, 'multiwell_opt17', 'test_01', 'esmda')


def test_interrupted_replacement_invalidates_old_completion(tmp_path, lightweight_task, monkeypatch):
    lightweight_task(tmp_path)

    def fail(*_, **__):
        raise RuntimeError('interrupted before output completion')

    monkeypatch.setattr(tasks, 'forecast', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        lightweight_task(tmp_path)
    assert (tmp_path / 'scores.json').exists()
    assert not tasks.task_complete(tmp_path, 'multiwell_opt17', 'test_01', 'esmda')


@pytest.mark.parametrize('record', ['{', '[]', '{}'])
def test_invalid_completion_record_is_incomplete(tmp_path, record):
    (tmp_path / 'complete.json').write_text(record)
    assert not tasks.task_complete(tmp_path, 'multiwell_opt17', 'test_01', 'esmda')


def test_unknown_monitoring_config_fails_before_reading_cases(monkeypatch):
    registry = SimpleNamespace(labels=lambda _: [], case=lambda _: pytest.fail('No data should be read'))
    monkeypatch.setattr(tasks.artifacts, 'Reservoirs', lambda _: registry)
    with pytest.raises(ValueError, match='Unknown monitoring configuration'):
        cli.cmd_prepare_cases(SimpleNamespace(data=None, config='misspelled', check=True, out=None))


def test_batch_shard_validation_precedes_loading_model(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts/run_all_tasks.py'
    spec = importlib.util.spec_from_file_location('run_all_tasks', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'load_backend', lambda *_: pytest.fail('Invalid shard loaded the model'))
    for shard in ('0/0', '-1/2', '2/2', 'wrong'):
        monkeypatch.setattr('sys.argv', ['run_all_tasks.py', '--out', str(tmp_path), '--shard', shard])
        with pytest.raises(SystemExit) as result:
            module.main()
        assert result.value.code == 2


@pytest.mark.parametrize('difference', ['none', 'score', 'diagnostic', 'nan'])
def test_nuts_verification_reports_mismatches(tmp_path, monkeypatch, difference):
    folder = tmp_path / 'nuts/test_01'
    folder.mkdir(parents=True)
    diag = dict(rank_rhat_max=1.01, bulk_ess_min=400., tail_ess_min=500., divergences=0,
                convergence_pass=True)
    (folder / 'diagnostics.json').write_text(json.dumps(diag))
    (folder / 'scores.json').write_text(json.dumps({'metric': 1.}))
    new_diag, score = dict(diag), 1.
    if difference == 'score':
        score = 1.1
    elif difference == 'diagnostic':
        new_diag['bulk_ess_min'] = 300.
    elif difference == 'nan':
        score = float('nan')
    checked = []
    monkeypatch.setattr(nuts_runner.artifacts, 'path', lambda rel, root: checked.append(rel))
    monkeypatch.setattr(nuts_runner, 'merge', lambda *_, **__: dict(diagnostics=new_diag, scores={'metric': score}))
    monkeypatch.setattr(nuts_runner, 'headline', lambda scores: scores)
    result = nuts_runner.verify_released('test_01', root=tmp_path)
    assert result['passed'] is (difference == 'none')
    assert len(checked) == 10


def test_nuts_cli_returns_failure_if_any_case_fails(monkeypatch):
    monkeypatch.setattr(tasks, 'load_backend', lambda *_: object())
    monkeypatch.setattr(nuts_runner, 'verify_released', lambda case, **_: {'passed': case != 'test_02'})
    args = SimpleNamespace(device='cpu', threads=1, data=None, case=['test_01'], rtol=1e-4, atol=1e-8)
    assert cli.cmd_nuts_verify(args) == 0
    args.case.append('test_02')
    assert cli.cmd_nuts_verify(args) == 1


def test_reproduced_diagnostic_flag_is_separate_from_numerical_reproduction(tmp_path, monkeypatch):
    folder = tmp_path / 'nuts/test_16'
    folder.mkdir(parents=True)
    diagnostics = dict(rank_rhat_max=1.052, bulk_ess_min=53.6, tail_ess_min=283.6,
                       divergences=0, convergence_pass=False)
    (folder / 'diagnostics.json').write_text(json.dumps(diagnostics))
    (folder / 'scores.json').write_text(json.dumps({'metric': 1.}))
    monkeypatch.setattr(nuts_runner.artifacts, 'path', lambda *_: None)
    monkeypatch.setattr(nuts_runner, 'merge',
                        lambda *_, **__: dict(diagnostics=diagnostics, scores={'metric': 1.}))
    monkeypatch.setattr(nuts_runner, 'headline', lambda scores: scores)
    result = nuts_runner.verify_released('test_16', root=tmp_path)
    assert result['passed']
    assert not result['convergence_pass']
