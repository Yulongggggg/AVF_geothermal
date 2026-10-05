"""Run calibration, checked validation pilots, and explicit formal comparisons."""
from .common import ROOT, OUT, verify, sha, write_json
import argparse
import json
import time
import numpy as np
import torch
from .geothermal import Backend, Problem, prepare_case, calibrate
from . import algorithms
from .metrics import score


def finite_json(value):
    if isinstance(value, dict): return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [finite_json(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)): return finite_json(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
    return value


def gradient_check(problem):
    rng = np.random.default_rng(8712)
    y = problem.tensor(rng.normal(size=problem.dim)).requires_grad_(True)
    direction = problem.tensor(rng.normal(size=problem.dim))
    direction /= torch.linalg.norm(direction)
    residual = problem.forward(y[None])[0]-problem.data
    potential = .5*(y.square().sum()+residual.square().sum())
    grad, = torch.autograd.grad(potential, y)
    analytic = (grad*direction).sum().item()
    values = []
    with torch.no_grad():
        for eps in (.1, .03, .01):
            potentials = []
            for sign in (-1, 1):
                yy = y+sign*eps*direction
                error = problem.forward(yy[None])[0]-problem.data
                potentials.append(.5*(yy.square().sum()+error.square().sum()).item())
            fd = (potentials[1]-potentials[0])/(2*eps)
            values.append(dict(step=eps, finite_difference=fd, relative_error=abs(fd-analytic)/max(1., abs(analytic))))
        # Whitening must equal the same dense physical covariance quadratic.
        unwhitened = problem.data@problem.cholesky.T
        q_dense = unwhitened@torch.linalg.solve(problem.cholesky@problem.cholesky.T, unwhitened)
        np.testing.assert_allclose(q_dense.item(), problem.data.square().sum().item(), rtol=1e-9)
    result = dict(analytic=analytic, comparisons=values, potential=float(potential.detach()),
                  gradient_norm=float(torch.linalg.norm(grad)))
    if not torch.isfinite(grad).all() or min(v['relative_error'] for v in values) > .05:
        raise RuntimeError('FP32 surrogate gradient check failed: '+json.dumps(result))
    return result


def forecasts(problem, latent, destination, count=256):
    flat = latent.reshape(-1, problem.dim)
    if latent.ndim == 3:
        # Balanced across chains; full chains are retained for convergence checks.
        per_chain = max(1, count//len(latent))
        index = np.linspace(0, latent.shape[1]-1, min(per_chain, latent.shape[1]), dtype=int)
        selected = latent[:, index].reshape(-1, problem.dim)
    else:
        selected = flat[np.linspace(0, len(flat)-1, min(count, len(flat)), dtype=int)]
    fields, geology, readouts = [], [], []
    with torch.no_grad():
        for begin in range(0, len(selected), 4):
            m = problem.permeability(problem.tensor(selected[begin:begin+4]))
            f = problem.backend.predict_real(problem.gi, 10, m)
            fields.append(f[:, :, 16:144].cpu().numpy())
            geology.append(m.cpu().numpy())
            readouts.append(torch.einsum('bchw,phw->bcp', f.double(), problem.qoi_weights).cpu().numpy())
    fields, geology, qoi = np.concatenate(fields), np.concatenate(geology), np.concatenate(readouts)
    fe = problem.case['truth'][10]
    truth_qoi = np.einsum('chw,phw->cp', fe, problem.case['qoi_weights'])
    result = dict(scoring_members=len(selected), blocks={}, fields={})
    for ch, channel in enumerate(('temperature', 'pressure')):
        position = ch  # producer T; injector p
        for name, draws, truth, sd in [
            ('blocks', qoi[:, ch, position], truth_qoi[ch, position], np.sqrt(problem.cal['qoi_variance'][10][ch][position])),
            ('fields', fields[:, ch], fe[ch, 16:144], np.sqrt(problem.cal['field_variance'][10][ch]))]:
            result[name][channel] = {}
            for level in ('raw', 'with_surrogate_epistemic'):
                scores = score(draws, truth, sd if level != 'raw' else None)
                result[name][channel][level] = {k: float(np.mean(v)) for k, v in scores.items()
                                                if k not in ('p05', 'p50', 'p95')}
    true_k = problem.case['logk_truth']
    lower, upper = np.quantile(geology, [.05, .95], axis=0)
    result['permeability'] = dict(rmse_logk=float(np.sqrt(np.mean((geology.mean(0)-true_k)**2))),
                                   mean_variance=float(np.var(geology, axis=0, ddof=1).mean()),
                                   coverage90=float(((true_k >= lower)&(true_k <= upper)).mean()))
    np.savez_compressed(destination/'forecast.npz', field_mean=fields.mean(0), field_p05=np.quantile(fields, .05, axis=0),
                        field_p95=np.quantile(fields, .95, axis=0), qoi_samples=qoi,
                        logk_mean=geology.mean(0), logk_sd=geology.std(0), logk_p05=lower, logk_p95=upper,
                        truth_fields=fe[:, 16:144], truth_logk=true_k, truth_qoi=truth_qoi)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['calibrate', 'check', 'pilot', 'formal'])
    ap.add_argument('--method', choices=['esmda', 'rml', 'nuts'], default='esmda')
    ap.add_argument('--index', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    protocol = verify()
    if args.mode == 'formal' and not (OUT/'formal_review.json').exists():
        raise RuntimeError('Validation pilot diagnostic review must precede formal launch')
    backend = Backend('test' if args.mode == 'formal' else 'val', args.device)
    calibration = OUT/'calibration.json'
    if args.mode == 'calibrate':
        calibrate(backend, protocol['cases']['calibration'], calibration)
        for entry in protocol['cases']['pilot']:
            prepare_case(backend, entry, OUT/'cases'/f"{entry['label']}.npz")
        backend_test = Backend('test', args.device)
        for entry in protocol['cases']['test']:
            prepare_case(backend_test, entry, OUT/'cases'/f"{entry['label']}.npz")
        write_json(OUT/'case_manifest.json', {p.name: sha(p) for p in sorted((OUT/'cases').glob('*.npz'))})
        return
    phase = 'test' if args.mode == 'formal' else 'pilot'
    entry = protocol['cases'][phase][args.index]
    case_path = OUT/'cases'/f"{entry['label']}.npz"
    expected = json.loads((OUT/'case_manifest.json').read_text())[case_path.name]
    if sha(case_path) != expected: raise RuntimeError('Shared case changed')
    problem = Problem(backend, case_path, calibration)
    if args.mode == 'check':
        result = gradient_check(problem)
        write_json(OUT/f"gradient_check_{entry['label']}.json", result)
        print(json.dumps(result, indent=2))
        return
    destination = OUT/args.mode/entry['label']/args.method
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    last_notice = [0.]
    def progress(record):
        write_json(destination/'progress.json', record | dict(seconds=time.monotonic()-started))
        if time.monotonic()-last_notice[0] > 30:
            print(json.dumps(record), flush=True)
            last_notice[0] = time.monotonic()
    settings = protocol['proposed_formal' if args.mode == 'formal' else 'pilot'][args.method]
    latent, info = getattr(algorithms, args.method)(problem, **settings, seed=entry['observation_seed']+19000, progress=progress)
    np.savez_compressed(destination/'latent_samples.npz', y=latent)
    if args.method == 'rml':
        np.savez_compressed(destination/'randomization.npz', data=info.pop('data_perturbations'), prior=info.pop('prior_perturbations'))
    if args.method == 'nuts':
        import arviz as az
        data = az.from_dict(posterior={'y': latent})
        info['rank_rhat_max'] = float(az.rhat(data, method='rank').y.max())
        info['bulk_ess_min'] = float(az.ess(data, method='bulk').y.min())
        info['tail_ess_min'] = float(az.ess(data, method='tail').y.min())
    info.update(method=args.method, case_sha256=expected, calibration_sha256=sha(calibration),
                protocol_sha256=sha(OUT/'protocol.json'), settings=settings,
                forward_member_times=backend.evaluations, posterior_calls=problem.calls,
                publishable_ranking=False if args.mode == 'pilot' else None)
    write_json(destination/'diagnostics.json', finite_json(info))
    scores = forecasts(problem, latent, destination)
    write_json(destination/'scores.json', scores)
    write_json(destination/'complete.json', dict(status='complete', seconds=time.monotonic()-started,
                                                 method=args.method, case=entry['label']))


if __name__ == '__main__': main()
