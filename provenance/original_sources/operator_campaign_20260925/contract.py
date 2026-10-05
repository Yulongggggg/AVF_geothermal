"""Nontrivial checks of head conversion, function queries, and inverse gradients."""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from direct_operator.data import ScenarioPairs
from encoder_study.model import load as load_parent
from .common import OUT, PACK, BANK, PARENT, PARENT_SHA, sha, read, write, save
from .model import GeoUNOBasisHead, load, coordinates
from .train import ShardedBatches


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    torch.set_num_threads(1); torch.manual_seed(20260925)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = args.device
    assert sha(PARENT) == PARENT_SHA
    stats = read(PACK/'stats.json')
    model = GeoUNOBasisHead(stats['fields_std'], stats['p_res_std_mpa'], pod_rank=256)
    copied = model.initialize_parent(PARENT)
    model.initialize_pod(BANK)
    model.to(device).eval()
    data = ScenarioPairs(PACK, 'train', 13, 2)
    f, c, y, _ = data[(0, 10)]
    f, c, y = [x[:1].to(device) for x in (f, c, y)]
    report = dict(parent_sha256=sha(PARENT), copied_tensors=len(copied), device=device,
                  contract_sha256=sha(__file__), tf32=False)
    with torch.no_grad():
        pred, parts = model(f, c, return_parts=True)
        expanded = torch.einsum('bcr,crn->bcn', parts['coefficients'], parts['basis'])+parts['mean'][None]
        reconstructed = torch.stack(((expanded[:, 0]+parts['constant_coefficients'][:, 0:1])*model.scales[0],
                                     expanded[:, 1]*model.pressure_residual_scale+parts['constant_coefficients'][:, 1:2]*model.scales[1]), 1).reshape_as(pred)
        assert torch.equal(pred, reconstructed)
        old = load_parent(PARENT).to(device)
        parent = old(f, c)
        scale = torch.stack((model.scales[0], model.pressure_residual_scale))[None, :, None]
        pp = parent.flatten(2)
        center = pp.mean(-1, keepdim=True)
        normalized = (pp-center)/scale-parts['mean'][None]
        pod = model.pod.flatten(2)
        coefficients = torch.einsum('bcn,crn->bcr', normalized, pod)/pp.shape[-1]
        projected = (torch.einsum('bcr,crn->bcn', coefficients, pod)+parts['mean'][None])*scale+center
        projection_error = float((projected.reshape_as(pred)-pred).abs().max())
        assert projection_error < 2e-4, projection_error
        idx = torch.tensor([0, 127, 20479, 999, 10240, 8192], device=device)
        query = model(f, c, query=model.sensor_coordinates[idx])
        query_error = float((query-pred.flatten(2)[:, :, idx]).abs().max())
        assert query_error < 2e-3, query_error
        arbitrary = torch.tensor([[.1234, -.321], [-.712, .833]], device=device, requires_grad=True)
        dense_shape = model(f, c, query=coordinates(device, 31, 27)).shape
        assert dense_shape == (1, 2, 31*27)
        report.update(expansion_exact=True, projection_max_difference=projection_error,
                      query_at_nodes_max_difference=query_error, arbitrary_query_shape=list(dense_shape),
                      temperature_fourier_cycles=(model.trunk.frequency/torch.pi).cpu().tolist())
        del old
    f = f.clone().requires_grad_(True)
    pred = model(f, c)
    scalar = pred[:, :, 30:130:7, 20:100:7].double().mean()
    gradient, = torch.autograd.grad(scalar, f)
    assert torch.isfinite(gradient).all() and gradient[:, 0].abs().max() > 0
    direction = torch.zeros_like(f); direction[:, 0] = torch.randn_like(f[:, 0])
    direction /= direction[:, 0].square().mean().sqrt()
    ad = float((gradient.double()*direction.double()).sum())
    sweep = []
    with torch.no_grad():
        for eps in (.01, .003, .001, .0003, .0001):
            plus = model(f+eps*direction, c)[:, :, 30:130:7, 20:100:7].double().mean()
            minus = model(f-eps*direction, c)[:, :, 30:130:7, 20:100:7].double().mean()
            fd = float((plus-minus)/(2*eps))
            relative = abs(ad-fd)/max(abs(ad), abs(fd), 1e-5)
            sweep.append(dict(epsilon=eps, finite_difference=fd, relative_error=relative))
    # Max pooling is piecewise differentiable; the step sweep checks convergence
    # towards the local derivative instead of mistaking a finite jump for it.
    assert sum(r['relative_error'] < .05 for r in sweep) >= 2, (ad, sweep)
    report.update(permeability_gradient_norm=float(gradient[:, 0].norm()),
                  directional_autograd=ad, directional_finite_difference_sweep=sweep)
    qout = model(f.detach(), c, query=arbitrary)
    qgrad, = torch.autograd.grad(qout.sum(), arbitrary)
    assert torch.isfinite(qgrad).all()
    # The trunk must participate in learning after zero-initialized coefficient warm start.
    optimizer = torch.optim.AdamW(list(model.coefficient_delta.parameters())+list(model.trunk.parameters()), lr=1e-5)
    trunk_norms = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        value = ((model(f.detach(), c)-y)/model.scales[None, :, None, None]).square().mean()
        value.backward()
        trunk_norms.append(sum(float(p.grad.norm()) for p in model.trunk.parameters() if p.grad is not None))
        optimizer.step()
    assert trunk_norms[1] > 0 and np.isfinite(trunk_norms).all()
    report['trunk_gradient_norms'] = trunk_norms
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        path = Path(tmp)/'model.pt'
        save(path, dict(model=model.state_dict(), model_config=model.config))
        restored = load(path).to(device)
        with torch.no_grad(): assert torch.equal(model(f.detach(), c), restored(f.detach(), c))
    for size in (13, 17, 19, 23, 6582):
        class Fake:
            def __len__(self): return size
        fake = Fake(); fake.pairs = type('Names', (), dict(names=[str(i) for i in range(size)]))()
        for world in (2, 4):
            samplers = [ShardedBatches(fake, 7, r, world, 16) for r in range(world)]
            rank_batches = [list(s) for s in samplers]
            assert len({len(b) for b in rank_batches}) == 1
            seen = [i for batches in rank_batches for b in batches for i, _ in b]
            assert sorted(seen) == list(range(size))
            for step in zip(*rank_batches):
                counts = [len(b) for b in step]
                losses = np.arange(1, world+1, dtype=float)
                ddp_loss = np.mean(losses*world*np.array(counts)/sum(counts))
                assert np.isclose(ddp_loss, np.average(losses, weights=counts))
    report.update(sampler_no_duplicates_or_omissions=True, ddp_unequal_tail_weighting=True,
                  parent_unchanged=sha(PARENT) == PARENT_SHA, checkpoint_roundtrip_exact=True,
                  status='passed')
    write(OUT/f'contract_{device}.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
