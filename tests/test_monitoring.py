"""Monitoring geometry, noise and the registered protocol (no artifacts needed)."""
import json

import numpy as np

from geouno.artifacts import REPOSITORY, resource
from geouno.monitoring import FAMILIES, candidate_points, configurations, geometry, masks, point_noise

PROTOCOL = json.loads((REPOSITORY / 'protocols' / 'monitoring_protocol.json').read_text())


def test_configurations_match_frozen_protocol():
    configs = configurations()
    assert len(configs) == 49
    assert configs == PROTOCOL['configs']
    full = [c for c in configs if 'test_limit' not in c]
    assert len(full) * 24 + (len(configs) - len(full)) * 6 == 1104


def test_geometries_match_frozen_protocol_for_all_reservoirs():
    # Recomputed points agree with the frozen ones to rounding (the last digit depends on the
    # platform's hypot); case preparation therefore uses the frozen points of registered reservoirs.
    for label, families in PROTOCOL['geometries'].items():
        wells = np.array([p[1:] for p in families['vertical']['points'][:2]], dtype=float)
        for family in FAMILIES:
            points, _ = geometry(wells, family)
            stored = families[family]['points']
            assert [p[0] for p in points] == [p[0] for p in stored], (label, family)
            np.testing.assert_allclose([p[1:] for p in points], [p[1:] for p in stored], rtol=0, atol=1e-9)


def test_packaged_geometries_equal_frozen_protocol():
    from geouno.monitoring.protocol import registered_geometries
    packaged = registered_geometries()
    assert sorted(packaged) == sorted(PROTOCOL['geometries'])
    for label, families in PROTOCOL['geometries'].items():
        for family, record in families.items():
            assert packaged[label][family] == record['points']


def test_candidate_sets_respect_domain_and_spacing():
    wells = np.array([[0., -250.], [100., 200.]])
    sizes = {}
    for family in FAMILIES:
        points, _ = geometry(wells, family)
        xy = np.array([p[1:] for p in points])
        assert np.abs(xy).max() <= 400
        assert np.allclose(xy[:2], wells)
        sizes[family] = len(points)
    assert sizes == {'multiwell': 385, 'vertical': 99, 'inclined': 99}
    assert len({tuple(p[1:]) for p in candidate_points(100., 200.)}) == 385


def test_point_noise_is_keyed_by_location():
    points = [('a', 10., 20.), ('b', -30., 40.), ('c', 50., -60.)]
    full = point_noise(points, (11,), 610000, '0', 'monitoring')
    subset = point_noise(points[::-1][:2], (11,), 610000, '0', 'monitoring')
    assert np.array_equal(full[:, 2], subset[:, 0]) and np.array_equal(full[:, 1], subset[:, 1])
    assert not np.array_equal(point_noise(points, (11,), 610001, '0', 'monitoring'), full)


def test_masks_are_normalized_six_metre_blocks():
    w = masks([('a', 0., -250.), ('b', 100., 100.)])
    assert w.shape == (2, 160, 128) and np.allclose(w.sum((1, 2)), 1)
    k = masks([('a', 0., -250.)], geology=True)
    assert k.shape == (1, 128, 128) and np.isclose(k.sum(), 1)


def test_frozen_design_is_nested_and_starts_at_the_wells():
    design = json.loads(resource('design.json').read_text())
    for key, history in design['histories'].items():
        assert [h['n'] for h in history] == list(range(2, 34)), key
        for a, b in zip(history, history[1:]):
            assert b['indices'][:-1] == a['indices'] and b['indices'][:2] == [0, 1]
