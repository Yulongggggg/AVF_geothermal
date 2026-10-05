"""Monitoring locations, observation blocks and coordinate-keyed noise.

Sources: ``history_matching_networks/networks.py`` (``point_key``, corridor
``network_points``, ``point_noise``), ``history_matching_dense/networks.py``
(``extend``, dense ``network_points``) and ``inference_comparison/geothermal.py``
(``masks``). Locations use well geometry only; truth fields never enter.

A point is a tuple ``(name, x, y)`` in metres; the injector is at (0, -250) m.
"""
import hashlib
from functools import lru_cache

import numpy as np

CORRIDOR_NETWORKS = {
    'wells2': dict(n=2, label='2 points: wells only'),
    'cross5': dict(n=5, label='5 points: original cross'),
    'axial5': dict(n=5, label='5 points: well axis'),
    'corridor9': dict(n=9, label='9 points: cross and axis'),
    'corridor17': dict(n=17, label='17 points: corridor, 120 m flanks'),
    'wide17': dict(n=17, label='17 points: corridor, 240 m flanks'),
}

DENSE_NETWORKS = {
    'corridor17': dict(n=17, label='Original 17-point corridor'),
    'corridor49': dict(n=49, label='49 points covering the inter-well corridor'),
    'global49': dict(n=49, label='49 points covering the reservoir'),
    'hybrid97': dict(n=97, label='49-point corridor plus 48 reservoir points'),
    'hybrid193': dict(n=193, label='97-point hybrid plus 96 reservoir points'),
    'hybrid385': dict(n=385, label='Dense synthetic 385-point hybrid'),
}


def point_key(point):
    return f'{float(point[1]):.6f},{float(point[2]):.6f}'


def corridor_points(wx, wy, network):
    """Geometric corridor networks between the injector (0, -250) and producer (wx, wy)."""
    if network == 'bank':
        points = [p for name in CORRIDOR_NETWORKS for p in corridor_points(wx, wy, name)]
    else:
        if network not in CORRIDOR_NETWORKS:
            raise ValueError(network)
        vx, vy = float(wx), float(wy) + 250.
        length = np.hypot(vx, vy) + 1e-9
        nx, ny = -vy / length, vx / length

        def at(name, fraction, offset=0.):
            return (name, float(np.clip(fraction * vx + offset * nx, -390, 390)),
                    float(np.clip(-250 + fraction * vy + offset * ny, -390, 390)))
        points = [at('inj', 0.), at('prod', 1.)]
        if network == 'axial5':
            points += [at('axial25', .25), at('mid', .5), at('axial75', .75)]
        elif network != 'wells2':
            points += [at('mid', .5), at('perp+', .5, 120), at('perp-', .5, -120)]
            if network in ('corridor9', 'corridor17', 'wide17'):
                points += [at(f'axis{i}', i / 6) for i in (1, 2, 4, 5)]
            if network in ('corridor17', 'wide17'):
                offset = 120 if network == 'corridor17' else 240
                points += [at(f'flank{i}{sign:+d}', i / 5, sign * offset)
                           for i in (1, 2, 3, 4) for sign in (1, -1)]
    unique = {}
    for p in points:
        unique.setdefault(point_key(p), p)
    points = list(unique.values())
    if network != 'bank' and len(points) != CORRIDOR_NETWORKS[network]['n']:
        raise ValueError(f'Clipping collapsed locations for {network}: {(wx, wy)}')
    return points


def extend(points, wx, wy, target, corridor=False):
    """Greedy farthest-point extension on a 21 x 21 lattice (spacing 36 m, |x|,|y| <= 360 m)."""
    axis = np.linspace(-360, 360, 21)
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    candidates = np.stack([xx.ravel(), yy.ravel()], 1)
    if corridor:
        origin = np.array([0., -250.])
        v = np.array([wx, wy + 250.])
        f = np.clip((candidates - origin) @ v / (v @ v), 0, 1)
        candidates = candidates[np.linalg.norm(candidates - origin - f[:, None] * v, axis=1) <= 180]
    points = list(points)
    while len(points) < target:
        xy = np.array([p[1:] for p in points])
        distance = np.square(candidates[:, None] - xy[None]).sum(-1).min(1)
        i = int(np.argmax(distance))
        if distance[i] < 12 ** 2:
            raise ValueError('Insufficient distinct spatial candidates')
        x, y = candidates[i]
        points.append((f'monitor{len(points)}', float(x), float(y)))
        candidates = np.delete(candidates, i, axis=0)
    return points


@lru_cache(maxsize=4096)
def _candidate_points(wx, wy, network):
    base = corridor_points(wx, wy, 'corridor17')
    if network == 'corridor17':
        return base
    if network == 'global49':
        return extend(base, wx, wy, 49)
    dense = extend(base, wx, wy, 49, True)
    if network == 'corridor49':
        return dense
    dense = extend(dense, wx, wy, 97)
    if network == 'hybrid97':
        return dense
    dense = extend(dense, wx, wy, 193)
    if network == 'hybrid193':
        return dense
    if network == 'hybrid385':
        return extend(dense, wx, wy, 385)
    if network == 'bank':
        combined = _candidate_points(wx, wy, 'hybrid385') + _candidate_points(wx, wy, 'global49')
        return list({point_key(p): p for p in combined}.values())
    raise ValueError(network)


def candidate_points(wx, wy, network='hybrid385'):
    """Dense candidate banks seeded by the 17-point corridor (a fresh list each call)."""
    return list(_candidate_points(wx, wy, network))


def point_noise(points, shape, seed, channel, purpose):
    """Standard normal variates keyed by (seed, purpose, channel, coordinates).

    A reading at a given location receives the same error in every layout, independent of
    network size and ordering; changing the noise SD rescales the same draws.
    """
    samples = []
    for p in points:
        key = f'{seed}|{purpose}|{channel}|{point_key(p)}'.encode()
        local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
        samples.append(np.random.default_rng(local_seed).standard_normal(shape))
    return np.stack(samples, axis=-1)


def masks(points, geology=False):
    """Normalized 6-m averaging blocks: ``[P, 160, 128]`` (output grid) or ``[P, 128, 128]``."""
    xx = np.linspace(-400, 400, 128)
    yy = xx if geology else np.linspace(-500, 500, 160)
    x, y = np.meshgrid(xx, yy)
    mask = np.stack([np.hypot(x - px, y - py) <= 6. for _, px, py in points])
    if not mask.any(axis=(1, 2)).all():
        raise ValueError('Empty monitoring block')
    if len({m.tobytes() for m in mask}) != len(mask):
        raise ValueError('Duplicate monitoring block')
    return mask / mask.sum(axis=(1, 2), keepdims=True)
