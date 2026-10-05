"""Thin upstream adapters; no reimplementation of ES-MDA or NUTS kernels.

RML subclasses the author's RmapStrategy: its solve() supplies independent data
and prior perturbations. Only the nonlinear optimizer and sample retention are
adapted. See SOURCES.md for source, paper and equation correspondence.
"""
import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import minimize

UPSTREAM = Path(__file__).resolve().parents[1]/'runs/inference_comparison_20260914/upstream'


def rmap_class():
    name = '_geothermal_upstream_rmap'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, UPSTREAM/'rmap/src/__init__.py',
            submodule_search_locations=[str(UPSTREAM/'rmap/src')])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return __import__(name+'.randomization_strategies', fromlist=['RmapStrategy']).RmapStrategy


def esmda(problem, samples=256, steps=8, seed=1, progress=None):
    from iterative_ensemble_smoother import ESMDA
    start = time.monotonic()
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(problem.dim, samples))
    solver = ESMDA(np.ones(problem.nobs), problem.data.cpu().numpy(), alpha=steps, seed=seed+1)
    history = []
    for iteration in range(steps):
        with torch.no_grad():
            pred = problem.forward_numpy(x.T)
        solver.prepare_assimilation(Y=pred.T, truncation=1.)
        x = solver.assimilate_batch(X=x)
        history.append({'iteration': iteration+1, 'mean_whitened_misfit': float(np.mean((pred-problem.data.cpu().numpy())**2))})
        if progress: progress(history[-1])
    return x.T, {'seconds': time.monotonic()-start, 'history': history, 'inflation': solver.alpha.tolist()}


def rml(problem, samples=256, maxiter=300, seed=1, progress=None, gtol=1e-4):
    """Uncorrected nonlinear RML/rMAP, not EnRML or exact posterior MCMC."""
    start = time.monotonic()
    rng = np.random.default_rng(seed)
    retained, diagnostics, perturbations = [], [], []
    Base = rmap_class()

    class TorchRmap(Base):
        def convertCovariancesToMatrices(self):
            # Both covariances are identity after shared whitening. Avoid two
            # dense identity matrices; the upstream random generator accepts 1.
            assert self.inv_noise_covariance == self.inv_prior_covariance == 1.

        def solveRealization(self, data_perturbation, prior_perturbation):
            target = problem.data + problem.tensor(data_perturbation)
            center = problem.tensor(prior_perturbation)

            def objective(y):
                state = problem.tensor(y).requires_grad_(True)
                error = problem.forward(state[None])[0]-target
                loss = .5*((state-center).square().sum()+error.square().sum())
                gradient, = torch.autograd.grad(loss, state)
                if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
                    raise FloatingPointError('Non-finite RML objective or gradient')
                return loss.item(), gradient.detach().cpu().numpy().astype(np.float64)

            result = minimize(objective, prior_perturbation.copy(), jac=True,
                              method='L-BFGS-B', options=dict(maxiter=maxiter,
                              maxls=30, ftol=1e-12, gtol=gtol, maxcor=10))
            retained.append(result.x.copy())
            perturbations.append((data_perturbation.copy(), prior_perturbation.copy()))
            record = dict(sample=len(retained), success=bool(result.success), status=int(result.status),
                          message=str(result.message), iterations=int(result.nit), evaluations=int(result.nfev),
                          objective=float(result.fun), gradient_inf=float(np.max(np.abs(result.jac))))
            diagnostics.append(record)
            if progress: progress(record)
            return result.x

    def normal(mean, covariance, count):
        assert np.asarray(covariance).ndim == 0 and float(covariance) == 1.
        return rng.standard_normal((count, len(mean)))+mean

    solver = TorchRmap(problem.data.cpu().numpy(), SimpleNamespace(shape=(problem.nobs, problem.dim)),
                      1., 0., 1., normal, samples, 'direct')
    upstream_mean = solver.solve()
    draws = np.stack(retained)
    np.testing.assert_allclose(upstream_mean, draws.mean(0))
    return draws, dict(seconds=time.monotonic()-start, optimization=diagnostics,
                       converged_members=sum(r['success'] for r in diagnostics),
                       data_perturbations=np.stack([p[0] for p in perturbations]),
                       prior_perturbations=np.stack([p[1] for p in perturbations]))


def nuts(problem, samples=512, warmup=512, chains=4, max_depth=6,
         target_accept=.9, seed=1, progress=None):
    import pyro
    from pyro.infer import MCMC, NUTS
    start = time.monotonic()
    chains_out, chain_diagnostics, trace = [], [], []
    for chain in range(chains):
        pyro.set_rng_seed(seed+chain)
        initial = problem.tensor(np.random.default_rng(seed+chain).normal(size=problem.dim))
        evaluations = [0]
        previous = [0]

        def potential(params):
            evaluations[0] += 1
            y = params['y']
            residual = problem.forward(y[None])[0]-problem.data
            return .5*(y.square().sum()+residual.square().sum())

        def hook(kernel, params, stage, iteration):
            record = dict(chain=chain, stage=stage, iteration=iteration,
                          step_size=float(kernel.step_size),
                          potential_evaluations=evaluations[0]-previous[0])
            previous[0] = evaluations[0]
            trace.append(record)
            if progress: progress(record)

        kernel = NUTS(potential_fn=potential, target_accept_prob=target_accept,
                      max_tree_depth=max_depth, full_mass=False,
                      adapt_step_size=True, adapt_mass_matrix=True, jit_compile=False)
        sampler = MCMC(kernel, num_samples=samples, warmup_steps=warmup, num_chains=1,
                       initial_params={'y': initial}, disable_progbar=True, hook_fn=hook)
        sampler.run()
        chains_out.append(sampler.get_samples()['y'].detach().cpu().numpy())
        chain_diagnostics.append(sampler.diagnostics())
    return np.stack(chains_out), dict(seconds=time.monotonic()-start,
                                     chains=chain_diagnostics, trace=trace,
                                     max_tree_depth=max_depth, target_accept=target_accept)
