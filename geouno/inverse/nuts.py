"""No-U-Turn sampling of the whitened posterior with Pyro, and convergence diagnostics.

Sources: ``inference_comparison/algorithms.py`` (``nuts``) and the merge step of the
portable NUTS bundle (``portable_runtime/cli.py``). Paper protocol: four chains per
reservoir, 512 warm-up and 1,024 retained draws, target acceptance 0.9, maximum tree depth
7, diagonal adapted mass matrix, step-size adaptation, no JIT. Chain ``c`` uses seed
``observation_seed + 19000 + c`` for both the initial vector and Pyro's RNG.
Gate: max rank-normalized split R-hat <= 1.05, min bulk and tail ESS >= 200, no divergences.
"""
import time

import numpy as np

PAPER_PROTOCOL = dict(samples=1024, warmup=512, max_depth=7, target_accept=.9, chains=4)
CONVERGENCE = dict(rank_rhat_max=1.05, bulk_ess_min=200, tail_ess_min=200, divergences=0)


def nuts(problem, samples=512, warmup=512, chains=4, max_depth=6,
         target_accept=.9, seed=1, progress=None):
    """Run ``chains`` sequential chains; chain ``c`` is seeded with ``seed + c``."""
    import pyro
    from pyro.infer import MCMC, NUTS
    start = time.monotonic()
    chains_out, chain_diagnostics, trace = [], [], []
    for chain in range(chains):
        pyro.set_rng_seed(seed + chain)
        initial = problem.tensor(np.random.default_rng(seed + chain).normal(size=problem.dim))
        evaluations = [0]
        previous = [0]

        def potential(params):
            evaluations[0] += 1
            y = params['y']
            residual = problem.forward(y[None])[0] - problem.data
            return .5 * (y.square().sum() + residual.square().sum())

        def hook(kernel, params, stage, iteration):
            record = dict(chain=chain, stage=stage, iteration=iteration,
                          step_size=float(kernel.step_size),
                          potential_evaluations=evaluations[0] - previous[0])
            previous[0] = evaluations[0]
            trace.append(record)
            if progress:
                progress(record)

        kernel = NUTS(potential_fn=potential, target_accept_prob=target_accept,
                      max_tree_depth=max_depth, full_mass=False,
                      adapt_step_size=True, adapt_mass_matrix=True, jit_compile=False)
        sampler = MCMC(kernel, num_samples=samples, warmup_steps=warmup, num_chains=1,
                       initial_params={'y': initial}, disable_progbar=True, hook_fn=hook)
        sampler.run()
        chains_out.append(sampler.get_samples()['y'].detach().cpu().numpy())
        chain_diagnostics.append(sampler.diagnostics())
    return np.stack(chains_out), dict(seconds=time.monotonic() - start,
                                      chains=chain_diagnostics, trace=trace,
                                      max_tree_depth=max_depth, target_accept=target_accept)


def convergence(y, chain_infos):
    """Rank-normalized R-hat and bulk/tail ESS over all coordinates (ArviZ), plus divergences.

    ``y`` has shape ``[chains, draws, 4096]``; ``chain_infos`` are the per-chain info
    dictionaries returned by :func:`nuts` (one chain each) or stored ``diagnostics.json``.
    Any non-finite coordinate diagnostic fails the gate.
    """
    import arviz as az
    data = az.from_dict(posterior={'y': y})
    arrays = {name: np.asarray(value.y) for name, value in
              [('rhat', az.rhat(data, method='rank')), ('bulk', az.ess(data, method='bulk')),
               ('tail', az.ess(data, method='tail'))]}
    nonfinite = {k: int((~np.isfinite(v)).sum()) for k, v in arrays.items()}
    rhat = float(np.max(arrays['rhat']))
    bulk = float(np.min(arrays['bulk']))
    tail = float(np.min(arrays['tail']))
    divergences = sum(len(v) for info in chain_infos for chain in info['chains']
                      for v in chain.get('divergences', {}).values())
    good = bool(np.isfinite([rhat, bulk, tail]).all() and rhat <= CONVERGENCE['rank_rhat_max']
                and bulk >= CONVERGENCE['bulk_ess_min'] and tail >= CONVERGENCE['tail_ess_min']
                and divergences == CONVERGENCE['divergences'])
    return dict(rank_rhat_max=rhat, bulk_ess_min=bulk, tail_ess_min=tail, divergences=divergences,
                convergence_pass=good, nonfinite_diagnostic_coordinates=nonfinite), arrays


def balanced_draws(y, per_chain=64):
    """Equally spaced retained draws per chain (``linspace(0, 1023, 64)``), pooled."""
    index = np.linspace(0, y.shape[1] - 1, per_chain, dtype=int)
    return y[:, index].reshape(-1, y.shape[-1]), index
