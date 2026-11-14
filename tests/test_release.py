"""Checks against the released artifacts (run ``geouno download`` first; skipped otherwise)."""
import json
import os

import numpy as np
import pytest
import torch

from geouno import artifacts

pytestmark = pytest.mark.release


def _available():
    try:
        return all(artifacts.verify(groups=('core',)).values())
    except FileNotFoundError:
        return False


requires_core = pytest.mark.skipif(not _available(), reason='core artifacts not downloaded')


@pytest.fixture(scope='module')
def backend():
    from geouno.tasks import load_backend
    value = load_backend('cpu', threads=2)
    if os.environ.get('GEOUNO_TEST_RECOMPUTE') == '1':
        from torch.utils.checkpoint import checkpoint

        def wrap(forward):
            return lambda *args: checkpoint(forward, *args, use_reentrant=False)

        for block in (*value.operator.enc, *value.operator.temperature_decoder,
                      *value.operator.pressure_decoder, value.operator.global_mix,
                      ):
            block.forward = wrap(block.forward)
        branch = value.operator.branch
        # Recompute activations for this check only; retain the original batch shape
        # and arithmetic, and compare the resulting gradient with the frozen archive.
        value.operator.branch = lambda *args: checkpoint(branch, *args, use_reentrant=False)
    return value


@requires_core
def test_checkpoint_metadata():
    from geouno.model import CHECKPOINT_SHA256, TRAINABLE_PARAMETERS, load_geouno
    path = artifacts.path(artifacts.manifest()['assets']['geouno_resid64_p1_f39894ff.pt']['path'])
    net = load_geouno(path)
    assert sum(p.numel() for p in net.parameters()) == TRAINABLE_PARAMETERS
    assert net.config['architecture'] == 'geouno_simple'
    assert net.config['variant'] == 'resid64_p1'
    assert artifacts.manifest()['checkpoint_sha256'] == CHECKPOINT_SHA256


@requires_core
def test_regenerated_opt17_cases_equal_released_files():
    from geouno.monitoring import configuration, prepare_case
    from geouno.tasks import design
    reservoirs = artifacts.Reservoirs()
    for label in reservoirs.labels('test')[:4] + ['pilot_01']:
        case = reservoirs.case(label)
        seed = reservoirs.entry(label)['observation_seed']
        values = prepare_case(case, configuration('multiwell_opt17'), design(), seed, label)
        for key in case:
            assert np.array_equal(values[key], case[key]), (label, key)


@requires_core
def test_forward_prediction_close_to_fe_truth(backend):
    reservoirs = artifacts.Reservoirs()
    case = reservoirs.case('test_01')
    base = backend.base_tensor(reservoirs.base('test_01'))
    with torch.no_grad():
        pred = backend.predict_real(base, 10, torch.as_tensor(case['logk_truth']))[0].numpy()
    err = pred[:, 16:144] - case['truth'][10, :, 16:144]
    rmse = np.sqrt((err ** 2).mean((1, 2)))
    # This application model is the seed-20260915 member of Supplementary Table 1.
    assert rmse[0] < 3.0 and rmse[1] < 2.0, rmse


@pytest.mark.skipif(not (artifacts.data_dir() / 'nuts' / 'test_04' / 'diagnostics.json').exists(),
                    reason='NUTS artifacts not downloaded')
def test_released_nuts_diagnostics_recompute():
    from geouno.inverse import convergence
    run = artifacts.data_dir() / 'nuts' / 'test_04'
    ys, infos = [], []
    for c in range(4):
        with np.load(run / f'chain_{c}' / 'latent_samples.npz') as z:
            ys.append(z['y'].copy())
        infos.append(json.loads((run / f'chain_{c}' / 'diagnostics.json').read_text()))
    diagnostics, _ = convergence(np.concatenate(ys), infos)
    archived = json.loads((run / 'diagnostics.json').read_text())
    for key in ('rank_rhat_max', 'bulk_ess_min', 'tail_ess_min'):
        assert np.isclose(diagnostics[key], archived[key], rtol=1e-10), key
    assert diagnostics['divergences'] == 0 and diagnostics['convergence_pass']


@requires_core
def test_map_cache_identity_changes_with_observations(backend):
    from geouno.tasks import problem_for
    problem, _ = problem_for(backend, artifacts.Reservoirs(), 'pilot_01', 'multiwell_opt17')
    key = problem.cache_key()
    assert problem.cache_key() == key
    problem.d[0] += .01
    assert problem.cache_key() != key
