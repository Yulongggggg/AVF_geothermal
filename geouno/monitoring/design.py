"""Greedy goal-oriented A-optimal monitoring design (Methods Eq. 6) and space-filling controls.

Sources: ``history_matching_sparse/design.py`` (``reduction``, ``update``) and
``final_model/monitoring_budget.py`` (``state``, ``choose``).

For each validation reservoir a linear-Gaussian approximation is built from 64 prior
members: rows of ``H`` are ensemble anomalies of the 15 readings a candidate provides
(temperature and pressure at seven times plus one log-permeability value); seven extra
latent coordinates carry the spatially shared pressure error at each time. ``Y`` holds
the normalized anomalies of the ten-year producer cooling and injector pressure, so the
objective ``trace(Y S Y^T) / 2`` averaged over reservoirs is the mean posterior variance
relative to the prior variance. Placement uses validation priors and well geometry only.
"""
import numpy as np


def reduction(S, H, noise, Y):
    """Variance reduction of the targets for every candidate (``H``: candidate x reading x latent)."""
    # Correlated pressure errors are augmented latent modes, shared by all locations at each time.
    hs = H @ S
    gram = hs @ H.transpose(0, 2, 1)
    i = np.arange(H.shape[1])
    gram[:, i, i] += noise
    cross = hs @ Y.T
    return np.sum(cross * np.linalg.solve(gram, cross), axis=(1, 2))


def update(S, H, noise):
    """Posterior latent covariance after adding one candidate's readings."""
    hs = H @ S
    gram = hs @ H.T + np.diag(noise)
    new = S - hs.T @ np.linalg.solve(gram, hs)
    return (new + new.T) / 2


def state(path_or_arrays, cal):
    """Linear-Gaussian design state from a validation design-prior cache.

    The cache holds ``dynamic`` ``[time, member, channel, point]`` block predictions and
    ``geology`` ``[member, point]`` block log-permeability for 64 prior members.
    """
    a = np.load(path_or_arrays) if isinstance(path_or_arrays, (str, bytes)) or hasattr(path_or_arrays, '__fspath__') \
        else path_or_arrays
    q, k = a['dynamic'], a['geology']
    K = 64
    N = k.shape[1]

    def anomalies(x):
        return (x - x.mean(0, keepdims=True)) / np.sqrt(K - 1)
    H = np.zeros((N, 15, K + 7))
    for ch, start in [(0, 0), (1, 7)]:
        H[:, start:start + 7, :K] = anomalies(q[:7, :, ch].transpose(1, 0, 2)).transpose(2, 1, 0)
    H[:, 14, :K] = anomalies(k).T
    for t in range(7):
        H[:, 7 + t, K + t] = np.sqrt(cal['pressure_common_variance'][t])
    noise = np.tile(np.r_[np.array(cal['block_local_variance'])[:7, 0] + .5 ** 2,
                          np.array(cal['block_local_variance'])[:7, 1] + .25 ** 2, .1 ** 2], (N, 1))
    Y = np.zeros((2, K + 7))
    Y[0, :K] = anomalies(q[10, :, 0, 1])
    Y[1, :K] = anomalies(q[10, :, 1, 0])
    Y /= np.sqrt((Y * Y).sum(1, keepdims=True)).clip(1e-8)
    return dict(H=H, noise=noise, Y=Y, S=np.eye(K + 7))


def choose(states, geometries, rule, maximum=33, line=False):
    """Nested selection from candidate index 0, 1 (the two wells) up to ``maximum`` points.

    ``rule='optimized'`` adds the candidate with the largest mean variance reduction;
    ``rule='space_filling'`` adds the candidate farthest (on average over geometries) from
    those already chosen. Candidates closer than 6 m to a chosen point in any geometry are
    ineligible. Returns one record per budget with the chosen indices and the objective.
    """
    xy = np.array([[p[1:] for p in points] for points in geometries])
    chosen = [0, 1]
    history = []

    def add(j):
        for s in states:
            s['S'] = update(s['S'], s['H'][j], s['noise'][j])
    for j in chosen:
        add(j)

    def record():
        return dict(n=len(chosen), indices=chosen.copy(),
                    normalized_future_variance=float(np.mean([np.trace(s['Y'] @ s['S'] @ s['Y'].T) / 2
                                                              for s in states])))
    history.append(record())
    for _ in range(3, maximum + 1):
        distance = np.square(xy[:, :, None, :] - xy[:, None, chosen, :]).sum(-1).min(-1)
        eligible = (distance >= 6 ** 2 - 1e-8).all(0)
        eligible[chosen] = False
        if line:
            # Avoid exhausting a narrow line with an early greedy choice: leave
            # room for the largest registered budget under the spacing rule.
            step = int(np.ceil(6 / min(np.linalg.norm(x[1] - x[0]) for x in xy) * 128 - 1e-10))
            ordinal = np.r_[0, 128, np.arange(1, 128)]
            for j in np.flatnonzero(eligible):
                positions = np.sort(ordinal[chosen + [int(j)]])
                capacity = 1 + int(np.floor(np.diff(positions) / step).sum())
                if capacity < maximum:
                    eligible[j] = False
        gains = np.mean([reduction(s['S'], s['H'], s['noise'], s['Y']) for s in states], 0) \
            if rule == 'optimized' else distance.mean(0)
        gains[~eligible] = -np.inf
        assert np.isfinite(gains).any(), 'Insufficient feasible distinct monitoring locations'
        j = int(np.argmax(gains))
        chosen.append(j)
        add(j)
        history.append(record())
    return history
