"""Physical observation likelihood and ES-MDA equations, without release artifacts."""
import itertools

import numpy as np
import pytest
import torch
from scipy.linalg import solve

from geouno.inverse.esmda import esmda
from geouno.inverse.problem import Problem


class _ObservationBackend:
    """Small differentiable block model with recognizable times/channels/locations."""

    device = torch.device('cpu')

    def __init__(self):
        self.calls = 0

    def base_tensor(self, base):
        return torch.as_tensor(base)

    def observation_basis(self, weights):
        return weights

    def blocks(self, base, times, permeability, weights, projected):
        self.calls += 1
        time = torch.tensor(times, dtype=torch.float64)[None, :, None, None]
        channel = torch.arange(2, dtype=torch.float64)[None, None, :, None]
        location = torch.arange(3, dtype=torch.float64)[None, None, None, :]
        return (100 * time + 10 * channel + location
                + permeability[:, 0, None, None, None]
                + permeability[:, 1, None, None, None] * (1 + channel + location))


def _problem(channels=(0, 1), times=(2, 0)):
    case = dict(prior_mean=np.zeros((128, 128)), weights=np.ones((3, 1, 1)),
                observed=np.arange(18, dtype=float).reshape(3, 2, 3) + .25,
                qoi_weights=np.ones((1, 1)))
    calibration = dict(block_local_variance=[[.3, .4], [.5, .7], [1.1, .9]],
                       pressure_common_variance=[.2, .6, 1.3])
    problem = Problem(_ObservationBackend(), case, np.zeros(1), calibration,
                      channels=channels, times=times, static=False, sigma_t=.7, sigma_p=.3)
    # The prior transform is covered separately in test_math.py. Keep this fixture
    # small so the observation ordering and posterior gradient can be checked exactly.
    problem.dim = 2
    problem.permeability = lambda xi: xi
    return problem


@pytest.mark.parametrize('channels,times', [((0, 1), (0, 2)), ((1, 0), (2, 0)),
                                          ((0,), (2, 0)), ((1,), (2, 0)),
                                          ((), (2, 0)), ((0, 1), ())])
def test_physical_observations_and_covariance_have_the_same_order(channels, times):
    problem = _problem(channels, times)
    coordinates = list(itertools.product(times, channels, range(3)))
    expected_d = np.array([problem.case['observed'][t, c, p] for t, c, p in coordinates])
    expected_R = np.zeros((len(coordinates), len(coordinates)))
    for i, (t, c, p) in enumerate(coordinates):
        for j, (t2, c2, p2) in enumerate(coordinates):
            if (t, c) == (t2, c2):
                if p == p2:
                    expected_R[i, j] += problem.cal['block_local_variance'][t][c] + (.7 if c == 0 else .3) ** 2
                if c == 1:
                    expected_R[i, j] += problem.cal['pressure_common_variance'][t]
    np.testing.assert_array_equal(problem.d.numpy(), expected_d)
    np.testing.assert_allclose(problem.R.numpy(), expected_R, rtol=0, atol=1e-15)
    factor = torch.linalg.cholesky(problem.R)
    torch.testing.assert_close(factor @ factor.T, problem.R)
    xi = problem.tensor([[.2, -.1], [-.4, .3]])
    expected_g = np.array([[100 * t + 10 * c + p + a + b * (1 + c + p)
                            for t, c, p in coordinates] for a, b in xi.numpy()])
    np.testing.assert_allclose(problem.forward(xi).numpy(), expected_g, rtol=0, atol=1e-13)
    np.testing.assert_allclose(problem.forward_numpy(xi.numpy(), chunk=1), expected_g, rtol=0, atol=1e-13)
    assert problem.nobs == len(coordinates)
    if not coordinates:
        assert problem.backend.calls == problem.calls == 0


@pytest.mark.parametrize('channels', [(0, 1), (1,), ()])
def test_posterior_value_and_gradient_match_dense_covariance_quadratic(channels):
    problem = _problem(channels)
    xi = problem.tensor([.15, -.2]).requires_grad_()
    value = problem.potential(xi)
    gradient, = torch.autograd.grad(value, xi)

    reference_xi = xi.detach().clone().requires_grad_()
    residual = problem.forward(reference_xi[None])[0] - problem.d
    expected = .5 * (reference_xi @ reference_xi + residual @ torch.linalg.solve(problem.R, residual))
    expected_gradient, = torch.autograd.grad(expected, reference_xi)
    torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    batched = problem.forward(torch.stack([xi.detach(), -xi.detach()]))
    misfit = problem.data_misfit(batched)
    residual = batched - problem.d
    expected_quadratic = (residual * torch.linalg.solve(problem.R, residual.T).T).sum(-1)
    torch.testing.assert_close(misfit, expected_quadratic, rtol=1e-12, atol=1e-12)


class _LinearProblem:
    """A correlated, differently scaled observation problem with analytic updates."""

    def __init__(self, nobs=3, dim=6):
        rng = np.random.default_rng(37)
        self.A = rng.normal(size=(nobs, dim))
        self.dim, self.nobs = dim, nobs
        factor = rng.normal(size=(nobs, nobs))
        scale = np.geomspace(.2, 3, nobs)
        self.R = self.tensor((factor @ factor.T + np.eye(nobs)) * np.outer(scale, scale))
        self.d = self.tensor(rng.normal(size=nobs))

    @staticmethod
    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)

    def forward_numpy(self, xi):
        return xi @ self.A.T

    def data_misfit(self, predictions):
        residual = predictions - self.d
        return (residual * torch.linalg.solve(self.R, residual.T).T).sum(-1)


@pytest.mark.parametrize('dense', [False, True])
def test_reference_adapter_perturbations_have_the_requested_covariance(dense):
    pytest.importorskip('iterative_ensemble_smoother')
    from geouno.inverse._esmda_reference import PhysicalESMDA

    R = np.array([[2., .8], [.8, 1.]]) if dense else np.array([2., 1.])
    d = np.array([3., -2.])
    solver = PhysicalESMDA(covariance=R, observations=d, alpha=3, seed=19)
    normal = np.random.default_rng(19).standard_normal(size=(2, 7))
    C = np.linalg.cholesky(R) if dense else np.diag(np.sqrt(R))
    expected = d[:, None] + np.sqrt(3) * C @ normal
    actual = solver.perturb_observations(ensemble_size=7, alpha=3)
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-15)
    if dense:
        # The inherited gain solver requires the UPPER factor. Transpose only for
        # perturbations; replacing its stored factor would break the gain instead.
        np.testing.assert_allclose(solver.C_D_L.T @ solver.C_D_L, R, rtol=1e-15, atol=1e-15)
        assert not np.allclose(solver.C_D_L @ solver.C_D_L.T, R)


@pytest.mark.parametrize('nobs,samples,steps', [(3, 12, 1), (3, 12, 8), (9, 5, 3)])
def test_esmda_matches_physical_sample_covariance_update_and_legacy_whitening(nobs, samples, steps):
    reference = pytest.importorskip('iterative_ensemble_smoother')
    problem = _LinearProblem(nobs=nobs)
    seed = 23
    actual, info = esmda(problem, samples=samples, steps=steps, seed=seed)
    R, d = problem.R.numpy(), problem.d.numpy()
    C = np.linalg.cholesky(R)
    expected = np.random.default_rng(seed).normal(size=(problem.dim, samples))
    legacy = expected.copy()
    perturbations = np.random.default_rng(seed + 1)
    white = reference.ESMDA(covariance=np.ones(nobs), observations=np.linalg.solve(C, d),
                            alpha=steps, seed=seed + 1)
    misfits = []
    for _ in range(steps):
        g = problem.forward_numpy(expected.T).T
        delta_xi = expected - expected.mean(axis=1, keepdims=True)
        delta_g = g - g.mean(axis=1, keepdims=True)
        C_xi_g = delta_xi @ delta_g.T / (samples - 1)
        C_gg = delta_g @ delta_g.T / (samples - 1)
        epsilon = np.sqrt(steps) * C @ perturbations.standard_normal(size=(nobs, samples))
        residual = d[:, None] + epsilon - g
        misfits.append(np.mean(np.linalg.solve(C, g - d[:, None]) ** 2))
        expected += C_xi_g @ solve(C_gg + steps * R, residual, assume_a='pos')

        g_white = np.linalg.solve(C, problem.forward_numpy(legacy.T).T)
        white.prepare_assimilation(Y=g_white, truncation=1.)
        legacy = white.assimilate_batch(X=legacy)

    np.testing.assert_allclose(actual, expected.T, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(actual, legacy.T, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose([h['mean_whitened_misfit'] for h in info['history']], misfits,
                               rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(info['inflation'], np.full(steps, steps))
    assert sum(1 / alpha for alpha in info['inflation']) == pytest.approx(1)
    assert info['observation_space'] == 'physical'
    assert info['latent_coordinate'] == 'xi'
    assert info['misfit_evaluated_at'] == 'before_update'


@pytest.mark.parametrize('kwargs', [dict(samples=1), dict(samples=2.5), dict(steps=0), dict(steps=1.5)])
def test_esmda_rejects_invalid_ensemble_or_inflation_length(kwargs):
    pytest.importorskip('iterative_ensemble_smoother')
    with pytest.raises(ValueError):
        esmda(_LinearProblem(), **kwargs)


def test_esmda_requires_dynamic_observations():
    pytest.importorskip('iterative_ensemble_smoother')
    with pytest.raises(ValueError, match='dynamic observations'):
        esmda(_problem(channels=()))


class _GaussianPotential:
    """Known posterior exposed only through the shared potential interface."""

    def __init__(self, dim=2):
        self.dim = dim
        self.calls = 0
        self._A = self.tensor([[1.2, .4], [-.3, .7]])
        self._R = self.tensor([[.4, .15], [.15, .9]])
        self._d = self.tensor([1.3, -.6])

    @staticmethod
    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)

    def potential(self, xi):
        self.calls += 1
        residual = self._A @ xi[:2] - self._d
        return .5 * (xi @ xi + residual @ torch.linalg.solve(self._R, residual))


def test_map_uses_shared_potential_and_recovers_correlated_gaussian_mode():
    from geouno.inverse.map import map_optimize

    # Only two coordinates enter the observations; all other coordinates have
    # an independent standard-normal posterior and therefore a zero MAP value.
    problem = _GaussianPotential(dim=4096)
    result, info = map_optimize(problem, seed=11,
                                options=dict(maxiter=100, maxfun=200, gtol=1e-9, ftol=1e-13))
    A, R, d = problem._A.numpy(), problem._R.numpy(), problem._d.numpy()
    expected = np.zeros(problem.dim)
    expected[:2] = A.T @ np.linalg.solve(A @ A.T + R, d)
    np.testing.assert_allclose(result[0], expected, rtol=1e-7, atol=1e-8)
    assert result.shape == (1, problem.dim)
    assert info['success'] and info['point_estimate']
    assert problem.calls == info['evaluations'] > 0


def test_nuts_uses_shared_potential_and_preserves_seed_and_diagnostics_schema():
    pytest.importorskip('pyro')
    from geouno.inverse.nuts import nuts

    problem = _GaussianPotential()
    progress = []
    draws, info = nuts(problem, samples=8, warmup=8, chains=1, max_depth=2,
                       seed=11, progress=progress.append)
    assert draws.shape == (1, 8, problem.dim)
    assert np.isfinite(draws).all()
    assert problem.calls > 0
    assert progress == info['trace']
    assert len(progress) == 16
    assert all(record['potential_evaluations'] > 0 for record in progress)
    assert 'y' in info['chains'][0] and 'xi' not in info['chains'][0]
    # This checks the RNG/API contract, not convergence from eight smoke-test draws.
    repeated, _ = nuts(_GaussianPotential(), samples=8, warmup=8, chains=1,
                        max_depth=2, seed=11)
    np.testing.assert_array_equal(draws, repeated)
