"""Mathematical identities of the inference and design components (no artifacts needed)."""
import numpy as np
import torch

from geouno.inverse.prior import common_mode_quadratic, condition, transform
from geouno.metrics import empirical_crps, score
from geouno.monitoring.design import reduction, update
from geouno.training.loss import mmd2, objective


def _small_condition_problem(seed=0):
    g = torch.Generator().manual_seed(seed)
    factor = torch.zeros(4096, 4096)
    idx = torch.arange(4096)
    factor[idx, idx] = 0.5 + torch.rand(4096, generator=g)
    blocks = torch.zeros(3, 128, 128, dtype=torch.float64)
    blocks[0, 10:13, 20:23] = 1
    blocks[1, 60:62, 60:64] = 1
    blocks[2, 100:104, 5:8] = 1
    mean = torch.randn(128, 128, generator=g) * .1 - 13.4
    observed = torch.tensor([-13.0, -13.8, -12.9], dtype=torch.float64)
    return factor, blocks, mean, observed


def test_condition_matches_dense_gaussian_update():
    factor, blocks, mean, observed = _small_condition_problem()
    sigma = .1
    post = condition(mean, factor, blocks, observed, sigma=sigma)
    H = post['H'].double()
    gram = H @ H.T + sigma ** 2 * torch.eye(3, dtype=torch.float64)
    expected_mean = H.T @ torch.linalg.solve(gram, observed - post['offset'].double())
    torch.testing.assert_close(post['mean'].double(), expected_mean, rtol=1e-5, atol=1e-6)
    # S S^T v must equal the posterior covariance applied to v, for random v (matrix-free).
    root = post['root'].double()
    v = torch.randn(4096, 5, dtype=torch.float64)

    def apply_s(x):
        return x - root @ (root.T @ x)
    lhs = apply_s(apply_s(v))
    rhs = v - H.T @ torch.linalg.solve(gram, H @ v)
    torch.testing.assert_close(lhs, rhs, rtol=1e-4, atol=1e-5)


def test_transform_identity_without_conditioning():
    y = torch.randn(4, 4096, dtype=torch.float64)
    assert torch.equal(transform(y, None), y)


def test_common_mode_quadratic_equals_dense_inverse():
    g = torch.Generator().manual_seed(1)
    error = torch.randn(3, 4, 6, generator=g, dtype=torch.float64)
    diag = torch.rand(4, 6, generator=g, dtype=torch.float64) + .2
    common = torch.rand(4, generator=g, dtype=torch.float64)
    expected = torch.zeros(3, dtype=torch.float64)
    for t in range(4):
        cov = torch.diag(diag[t] ** 2) + common[t] ** 2 * torch.ones(6, 6, dtype=torch.float64)
        e = error[:, t]
        expected += torch.einsum('bi,ij,bj->b', e, torch.linalg.inv(cov), e)
    torch.testing.assert_close(common_mode_quadratic(error, diag, common), expected)


def test_empirical_crps_matches_definition():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=50), .3
    brute = np.mean(np.abs(x - y)) - .5 * np.mean(np.abs(x[:, None] - x[None, :]))
    assert np.isclose(empirical_crps(x, y), brute)


def test_mixture_quantiles_bracket_the_median():
    rng = np.random.default_rng(4)
    s = score(rng.normal(size=40), np.array(.1), .3)
    assert s['p05'] < s['p50'] < s['p95'] and s['width90'] > 0 and np.isfinite(s['nll'])


def test_design_reduction_equals_trace_drop_after_update():
    rng = np.random.default_rng(5)
    K, N = 9, 5
    H = rng.normal(size=(N, 4, K))
    noise = rng.uniform(.1, .5, size=(N, 4))
    Y = rng.normal(size=(2, K))
    A = rng.normal(size=(K, K))
    S = A @ A.T + np.eye(K)
    gains = reduction(S, H, noise, Y)
    for c in range(N):
        drop = np.trace(Y @ S @ Y.T) - np.trace(Y @ update(S, H[c], noise[c]) @ Y.T)
        assert np.isclose(gains[c], drop)


def test_mmd_is_zero_for_identical_ensembles_and_positive_otherwise():
    x = torch.randn(5, 30)
    assert abs(float(mmd2(x, x))) < 1e-6
    assert float(mmd2(x + 1.0, x)) > 1e-3


def test_objective_parts_are_finite_and_scenario_weighted():
    torch.manual_seed(0)
    pred, truth = torch.randn(6, 2, 160, 128), torch.randn(6, 2, 160, 128)
    value, parts = objective(pred, truth, [2, 1, 3], [11.3, 8.0], 1.6, weight=.2, kind='rmse')
    assert torch.isfinite(value)
    assert set(parts) == {'temperature_rmse', 'pressure_level_rmse', 'pressure_shape_rmse', 'outside_rmse', 'mmd'}
    zero, zparts = objective(truth, truth, [2, 1, 3], [11.3, 8.0], 1.6, weight=.2, kind='rmse')
    assert float(zero) < 1e-4 and abs(float(zparts['mmd'])) < 1e-6
