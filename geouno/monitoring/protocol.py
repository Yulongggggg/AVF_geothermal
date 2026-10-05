"""Registered monitoring configurations and deterministic observation generation.

Source: ``operator_final_20260930/monitoring_prepare.py`` (``configurations`` and the case
preparation loop of ``main``). The 49 configurations comprise 45 full-cohort
configurations (24 test reservoirs) and four noise variants on the first six reservoirs.

Observation model for a configuration with points ``x_j``::

    d_T(t, j) = <truth_T(t), w_j> + sigma_T * e_T(t, j)
    d_p(t, j) = <truth_p(t), w_j> + sigma_p * e_p(t, j)
    d_k(j)    = <logk_truth, w^k_j> + 0.1 * e_k(j)

where ``w_j`` are 6-m averaging blocks and the ``e`` are coordinate-keyed standard normal
draws (:func:`geouno.monitoring.networks.point_noise`) seeded by the reservoir's
``observation_seed``. Readings shared between layouts therefore have identical errors.
"""
import json
from functools import lru_cache

import numpy as np

from ..artifacts import resource
from .geometry import FAMILIES, geometry
from .networks import masks, point_noise

BUDGETS = (2, 3, 5, 7, 9, 13, 17, 25, 33)
SPACE_FILLING_BUDGETS = (5, 9, 17, 33)
ESMDA = dict(samples=256, steps=8)
PRIOR_REFERENCE_SEED = 81071
PRIOR_REFERENCE_SAMPLES = 256
MAP_SEED_OFFSET = 29000
ENSEMBLE_SEED_OFFSET = 19000


def configurations():
    """The 49 registered monitoring configurations, in registration order."""
    configs = []
    for family in FAMILIES:
        for rule, counts in [('optimized', BUDGETS), ('space_filling', SPACE_FILLING_BUDGETS)]:
            for n in counts:
                if n == 2 and family != 'multiwell':
                    continue
                configs.append(dict(key=f'{family}_{"opt" if rule == "optimized" else "space"}{n}', family=family,
                                    rule=rule, n=n, channels=[0, 1], static=True, times=list(range(7)),
                                    sigma_t=.5, sigma_p=.25, kind='budget'))
    for label, channels, static in [('Tp', [0, 1], False), ('T', [0], False), ('p', [1], False),
                                    ('Tk', [0], True), ('pk', [1], True), ('k', [], True)]:
        configs.append(dict(key=f'multiwell_opt9_{label}', family='multiwell', rule='optimized', n=9,
                            channels=channels, static=static, times=list(range(7)), sigma_t=.5, sigma_p=.25,
                            kind='channels'))
    for last in (4, 5):
        configs.append(dict(key=f'multiwell_opt17_end{last}', family='multiwell', rule='optimized', n=17,
                            channels=[0, 1], static=True, times=list(range(last + 1)), sigma_t=.5, sigma_p=.25,
                            kind='duration'))
    for multiplier in (.2, 2.):
        for label, channels in [('Tk', [0]), ('Tpk', [0, 1])]:
            configs.append(dict(key=f'multiwell_opt9_{label}_noise{multiplier:g}', family='multiwell',
                                rule='optimized', n=9, channels=channels, static=True, times=list(range(7)),
                                sigma_t=.5 * multiplier, sigma_p=.25, kind='noise', test_limit=6))
    return configs


def configuration(key):
    for config in configurations():
        if config['key'] == key:
            return config
    raise KeyError(f'Unknown monitoring configuration: {key}')


def design_indices(design, config):
    """Candidate indices of ``config`` from the frozen validation design histories."""
    return next(d['indices'] for d in design['histories'][config['family'] + '_' + config['rule']]
                if d['n'] == config['n'])


@lru_cache(maxsize=1)
def registered_geometries():
    """Candidate points of the 28 registered reservoirs, exactly as frozen in the monitoring protocol."""
    return json.loads(resource('geometries.json').read_text())


def config_points(base_case, config, design, label=None):
    """Monitoring points of ``config`` for one reservoir.

    For a registered reservoir (``label`` such as ``test_07``) the candidate set frozen in the
    protocol is used, as in the original case preparation. Recomputing it from the wells gives
    the same points up to the last floating-point digit, which depends on the platform's
    ``hypot``. Other reservoirs use :func:`geometry` with wells ``base_case['point_xy'][:2]``.
    """
    stored = registered_geometries().get(label, {}).get(config['family']) if label else None
    if stored is not None:
        candidates = [(p[0], float(p[1]), float(p[2])) for p in stored]
    else:
        candidates, _ = geometry(base_case['point_xy'], config['family'])
    return [candidates[j] for j in design_indices(design, config)]


def prepare_case(base_case, config, design, observation_seed, label=None):
    """Return the case arrays of ``config`` generated from a reservoir's base case.

    ``base_case`` supplies ``scenario, case_id, truth, logk_truth, qoi_weights, prior_mean``
    and the well coordinates ``point_xy[:2]`` (any configuration's case file works).
    """
    points = config_points(base_case, config, design, label)
    w, kw = masks(points), masks(points, True)
    xy = np.array([pt[1:] for pt in points])
    distance = np.linalg.norm(xy[:, None] - xy[None], axis=-1) + np.eye(len(xy)) * 1e6
    assert distance.min() >= 6. - 1e-6
    obs = np.einsum('tchw,phw->tcp', base_case['truth'], w)
    for ch, sd in [(0, config['sigma_t']), (1, config['sigma_p'])]:
        obs[:, ch] += sd * point_noise(points, (11,), observation_seed, str(ch), 'monitoring')
    ko = np.einsum('hw,phw->p', base_case['logk_truth'], kw) + .1 * point_noise(points, (), observation_seed, 'k',
                                                                                 'monitoring')
    values = {k: base_case[k] for k in ('scenario', 'case_id', 'truth', 'logk_truth', 'qoi_weights', 'prior_mean')}
    values.update(observed=obs, k_observed=ko, weights=w, k_weights=kw, point_xy=xy)
    return values
