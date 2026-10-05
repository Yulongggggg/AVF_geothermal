"""Checks against the released artifacts (run ``geouno download`` first; skipped otherwise)."""
import json

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
    return load_backend('cpu', threads=2)


@requires_core
def test_checkpoint_metadata():
    from geouno.model import CHECKPOINT_SHA256, TRAINABLE_PARAMETERS, load_geouno
    path = artifacts.path(artifacts.manifest()['assets']['geouno_pod1024_ed043489.pt']['path'])
    net = load_geouno(path)
    assert sum(p.numel() for p in net.parameters()) == TRAINABLE_PARAMETERS
    assert net.config == dict(architecture='geouno_basis_head', scales=[11.32111365383957, 7.98859426337178],
                              pressure_residual_scale=1.603772823092218, pod_rank=1024, learned_rank=64, width=32)
    assert artifacts.manifest()['checkpoint_sha256'] == CHECKPOINT_SHA256


@requires_core
def test_archived_posterior_value_and_gradient(backend):
    from geouno.tasks import problem_for
    problem, _ = problem_for(backend, artifacts.Reservoirs(), 'pilot_01', 'multiwell_opt17')
    with np.load(artifacts.path('reservoirs/validation/pilot_01_value_gradient.npz')) as z:
        ref = {k: z[k].copy() for k in z.files}
    assert problem.nobs == 238 and problem.dim == 4096
    np.testing.assert_allclose(problem.cholesky.numpy(), ref['cholesky'], rtol=1e-12, atol=1e-14)
    y = problem.tensor(ref['y']).requires_grad_(True)
    loss = problem.potential(y)
    gradient, = torch.autograd.grad(loss, y)
    np.testing.assert_allclose(loss.item(), ref['loss'], rtol=1e-5)
    np.testing.assert_allclose(gradient.numpy(), ref['gradient'], rtol=1e-3, atol=1e-2)


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
    # One realization at ten years; Table 1 averages 618 scenarios x 11 times (0.316 K / 0.427 MPa).
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
