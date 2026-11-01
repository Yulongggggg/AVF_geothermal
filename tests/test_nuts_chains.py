"""Chain merging rejects unrelated, truncated and corrupted inference output."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from geouno import artifacts, nuts_runner


@pytest.fixture
def chain_folder(tmp_path, monkeypatch):
    protocol = dict(samples=8, warmup=4, max_depth=7, target_accept=.9, chains=4)
    monkeypatch.setattr(nuts_runner, 'PAPER_PROTOCOL', protocol)
    entry = dict(observation_seed=610000, multiwell_opt17_case_sha256='case-hash')
    registry = SimpleNamespace(entry=lambda _: entry)
    monkeypatch.setattr(artifacts, 'Reservoirs', lambda _: registry)
    for chain in range(4):
        folder = tmp_path / f'chain_{chain}'
        folder.mkdir()
        samples = folder / 'latent_samples.npz'
        np.savez_compressed(samples, y=np.random.default_rng(chain).normal(size=(1, 8, 4096)))
        info = dict(format_version=1, case='test_01', chain=chain, seed=629000 + chain,
                    protocol={k: v for k, v in protocol.items() if k != 'chains'},
                    checkpoint_sha256=artifacts.manifest()['checkpoint_sha256'],
                    case_sha256='case-hash',
                    calibration_sha256=artifacts.sha256(artifacts.resource('calibration.json')),
                    samples_sha256=artifacts.sha256(samples))
        (folder / 'diagnostics.json').write_text(json.dumps(info))
    return tmp_path, registry


def test_valid_chain_identity(chain_folder):
    folder, registry = chain_folder
    y, info, _ = nuts_runner._read_chain('test_01', 2, folder / 'chain_2', registry)
    assert y.shape == (1, 8, 4096)
    assert info['seed'] == 629002


@pytest.mark.parametrize('field,value', [('case', 'test_02'), ('chain', 3), ('seed', 629001),
                                        ('checkpoint_sha256', 'wrong'), ('case_sha256', 'wrong'),
                                        ('calibration_sha256', 'wrong'), ('samples_sha256', 'wrong'),
                                        ('protocol', {})])
def test_merge_rejects_mixed_chain_identity(chain_folder, monkeypatch, field, value):
    folder, _ = chain_folder
    path = folder / 'chain_0/diagnostics.json'
    info = json.loads(path.read_text())
    info[field] = value
    path.write_text(json.dumps(info))
    monkeypatch.setattr(nuts_runner, 'convergence', lambda *_: pytest.fail('Invalid chains reached diagnostics'))
    with pytest.raises(ValueError, match=field):
        nuts_runner.merge('test_01', folder)


@pytest.mark.parametrize('damage', ['truncated', 'nonfinite'])
def test_chain_samples_must_be_complete_and_finite(chain_folder, damage):
    folder, registry = chain_folder
    path = folder / 'chain_0/latent_samples.npz'
    values = np.zeros((1, 7 if damage == 'truncated' else 8, 4096))
    if damage == 'nonfinite':
        values[0, 0, 0] = np.nan
    np.savez_compressed(path, y=values)
    info_path = folder / 'chain_0/diagnostics.json'
    info = json.loads(info_path.read_text())
    info['samples_sha256'] = artifacts.sha256(path)
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match='incomplete or non-finite'):
        nuts_runner._read_chain('test_01', 0, path.parent, registry)


def test_unidentified_legacy_chain_is_rejected(chain_folder):
    folder, registry = chain_folder
    path = folder / 'chain_0/diagnostics.json'
    info = json.loads(path.read_text())
    del info['seed']
    del info['format_version']
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match='legacy chain identity'):
        nuts_runner._read_chain('test_01', 0, path.parent, registry)
