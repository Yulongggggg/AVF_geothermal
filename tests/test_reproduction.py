"""Reproduction entry points: complete cohort and authenticated artifact guidance."""
from types import SimpleNamespace
import urllib.error

import pytest
import numpy as np

from geouno import artifacts, cli, nuts_runner, tasks


@pytest.mark.parametrize('group', ['fe', 'training', 'prior', 'nuts'])
def test_optional_artifacts_verify_without_core_model(monkeypatch, tmp_path, group):
    source = tmp_path / 'input.bin'
    source.write_bytes(b'released scientific input')
    manifest = {'assets': {'input.bin': {
        'group': group, 'path': 'input.bin', 'sha256': artifacts.sha256(source),
    }}}
    monkeypatch.setattr(artifacts, 'manifest', lambda: manifest)
    args = cli.build_parser().parse_args(['verify', '--group', group, '--data', str(tmp_path)])
    assert cli.cmd_verify(args) == 0
    source.write_bytes(b'changed input')
    assert cli.cmd_verify(args) == 1


def test_nuts_diagnostics_accept_mixed_chains_and_reject_shifted_chain():
    # Exercise the actual ArviZ API in CI without downloading the reservoir artifacts.
    from geouno.inverse import convergence

    draws = np.random.default_rng(20261020).normal(size=(4, 1024, 3))
    infos = [{'chains': [{'divergences': {'chain 0': []}}]} for _ in range(4)]
    good, _ = convergence(draws, infos)
    assert good['convergence_pass']
    draws[0, :, 0] += 5
    bad, _ = convergence(draws, infos)
    assert not bad['convergence_pass'] and bad['rank_rhat_max'] > 1.05


@pytest.mark.parametrize('label', ['test_07', 'test_16', 'test_24'])
def test_extended_nuts_cases_reach_model_loading(monkeypatch, tmp_path, label):
    class ReachedBackend(Exception):
        pass

    monkeypatch.setattr(artifacts, 'Reservoirs', lambda _: object())

    def load(*args):
        raise ReachedBackend

    monkeypatch.setattr(nuts_runner, 'load_backend', load)
    with pytest.raises(ReachedBackend):
        nuts_runner.run_chain(label, 0, tmp_path)
    with pytest.raises(ValueError, match='registered'):
        nuts_runner.run_chain('test_25', 0, tmp_path)


def test_default_nuts_verification_uses_available_archives(monkeypatch):
    seen = []
    monkeypatch.setattr(tasks, 'load_backend', lambda *_: object())

    def verify(case, **kwargs):
        seen.append(case)
        return {'passed': True}

    monkeypatch.setattr(nuts_runner, 'verify_released', verify)
    args = SimpleNamespace(device='cpu', threads=1, data=None, case=None, rtol=1e-4, atol=1e-8)
    assert cli.cmd_nuts_verify(args) == 0
    assert seen == [f'test_{i:02d}' for i in range(1, 25)]


def test_nuts_cli_verifies_explicit_extended_cases(monkeypatch):
    seen = []
    monkeypatch.setattr(tasks, 'load_backend', lambda *_: object())

    def verify(case, **kwargs):
        seen.append(case)
        return {'passed': True}

    monkeypatch.setattr(nuts_runner, 'verify_released', verify)
    args = cli.build_parser().parse_args(['nuts-verify', '--case', 'test_07', 'test_16', 'test_24'])
    assert args.func(args) == 0
    assert seen == ['test_07', 'test_16', 'test_24']


def test_private_lfs_error_does_not_try_a_nonexistent_release(monkeypatch, tmp_path):
    url = 'https://media.githubusercontent.com/media/owner/repo/artifacts/model.pt'

    def unavailable(request):
        raise urllib.error.HTTPError(url, 404, 'Not Found', {}, None)

    monkeypatch.setattr(artifacts.urllib.request, 'urlopen', unavailable)
    monkeypatch.setattr(artifacts.subprocess, 'run', lambda *_args, **_kwargs: pytest.fail('Not a release URL'))
    with pytest.raises(RuntimeError, match='git lfs pull'):
        artifacts._download(url, tmp_path / 'model.pt', repo='owner/repo', tag='artifact-branch')
    assert not (tmp_path / 'model.pt.part').exists()
