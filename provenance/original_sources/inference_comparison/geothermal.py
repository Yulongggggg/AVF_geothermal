"""One differentiable, whitened posterior shared by every inference adapter."""
from .common import ROOT, OUT, PACK, CHECKPOINTS, FACTOR, sha, write_json
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
import train_t as T
from det_sampler import RealSampler
from history_matching_revision.inputs import random_prior_mean, realized_inputs
from history_matching_dense.conditioning import condition, transform
from history_matching_sparse.networks import network_points


def masks(points, geology=False):
    xx = np.linspace(-400, 400, 128)
    yy = xx if geology else np.linspace(-500, 500, 160)
    x, y = np.meshgrid(xx, yy)
    mask = np.stack([np.hypot(x-px, y-py) <= 6. for _, px, py in points])
    if not mask.any(axis=(1, 2)).all(): raise ValueError('Empty monitoring block')
    if len({m.tobytes() for m in mask}) != len(mask): raise ValueError('Duplicate monitoring block')
    return mask/mask.sum(axis=(1, 2), keepdims=True)


class Backend:
    def __init__(self, split='val', device='cuda'):
        from pressure_revision.inference import load_pressure_replacement
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device(device)
        self.ds = T.GeoSetT(split, n_obs=20, with_targets=True, require_g=False, in_pp='dd')
        self.net = load_pressure_replacement(CHECKPOINTS['pressure'], CHECKPOINTS['background'],
                                            self.ds, device, CHECKPOINTS['temperature'])
        self.net.eval().requires_grad_(False)
        self.sampler = RealSampler(self.ds.st, device)
        self.factor = torch.tensor(np.load(FACTOR), device=device)
        self.fields = np.load(PACK/'fields.npy', mmap_mode='r')
        self.real = np.load(PACK/'real.npy', mmap_mode='r')
        self.meta = list(csv.DictReader((PACK/'meta.csv').open()))
        self.mean = torch.tensor(self.ds.st['fields_mean'], device=device).float()[None, :, None, None]
        self.std = torch.tensor(self.ds.st['fields_std'], device=device).float()[None, :, None, None]
        self.evaluations = 0

    def base(self, gi, ti):
        x = self.ds[gi*self.ds.T+ti][0][None, :13].to(self.device).clone()
        x[:, 1] = 0.
        x[:, 9] = (1.-self.sampler.sm_mean)/self.sampler.sm_std
        return x

    def predict_real(self, gi, ti, m):
        x = realized_inputs(self.base(gi, ti), m, self.sampler.M_mean, self.sampler.M_std)
        self.evaluations += len(x)
        return self.net.mean_modes(x)[0].float()*self.std+self.mean


class Problem:
    def __init__(self, backend, case_path, calibration_path):
        self.backend = backend
        self.device = backend.device
        case = np.load(case_path, allow_pickle=False)
        self.case = case
        self.gi = backend.ds.scen_list.index(str(case['scenario']))
        self.mean = self.tensor(case['prior_mean']).float()
        self.weights = self.tensor(case['weights'])
        self.dim = 4096
        self.times = list(range(7))
        self.conditional = condition(self.mean, backend.factor,
                                     self.tensor(case['k_weights']), self.tensor(case['k_observed']), sigma=.1)
        self.conditional = {k: (v.double() if isinstance(v, torch.Tensor) else v)
                            for k, v in self.conditional.items()}
        self.cal = json.loads(Path(calibration_path).read_text())
        obs = self.tensor(case['observed'][:7])  # time, channel, point
        n = self.weights.shape[0]
        covariances = []
        for ti in self.times:
            for ch in (0, 1):
                variance = self.cal['block_local_variance'][ti][ch]+(.5 if ch == 0 else .25)**2
                cov = variance*torch.eye(n, dtype=torch.float64, device=self.device)
                if ch == 1: cov = cov+self.cal['pressure_common_variance'][ti]*torch.ones_like(cov)
                covariances.append(cov)
        self.cholesky = torch.linalg.cholesky(torch.block_diag(*covariances))
        self.data = self.whiten(obs.flatten()[None])[0].detach()
        self.nobs = len(self.data)
        self.qoi_weights = self.tensor(case['qoi_weights'])
        self.calls = 0

    def tensor(self, x): return torch.as_tensor(x, device=self.device, dtype=torch.float64)

    def whiten(self, predictions):
        return torch.linalg.solve_triangular(self.cholesky, predictions.T, upper=False).T

    def permeability(self, y):
        z = transform(y, self.conditional)
        dev = (z.float()@self.backend.factor.T).reshape(-1, 1, 64, 64)
        return self.mean+F.interpolate(dev, (128, 128), mode='bilinear', align_corners=True)[:, 0]

    def forward(self, y):
        self.calls += 1
        m = self.permeability(y)
        outputs = []
        for ti in self.times:
            fields = self.backend.predict_real(self.gi, ti, m)
            outputs.append(torch.einsum('bchw,phw->bcp', fields.double(), self.weights))
        return self.whiten(torch.stack(outputs, dim=1).flatten(1))

    def forward_numpy(self, y, chunk=4):
        return np.concatenate([self.forward(self.tensor(batch)).detach().cpu().numpy()
                               for batch in np.array_split(y, max(1, int(np.ceil(len(y)/chunk))))])


from pathlib import Path


def prepare_case(backend, entry, destination):
    ds = backend.ds
    gi = ds.scen_list.index(entry['scenario'])
    row = entry['row']
    assert backend.meta[row]['case_id'] == entry['case_id']
    wx, wy = (float(v) for v in ds.S[row, 3:5])
    points = network_points(wx, wy, 'opt17')
    weights, kw = masks(points), masks(points, geology=True)
    qoi = masks([('producer', wx, wy), ('injector', 0., -250.)])
    truth = np.array(backend.fields[row], dtype=np.float64)
    dynamic = np.einsum('tchw,phw->tcp', truth, weights)
    geology = np.array(backend.real[row], copy=True)
    static = np.einsum('hw,phw->p', geology, kw)
    rng = np.random.default_rng(entry['observation_seed'])
    observed = dynamic+rng.standard_normal(dynamic.shape)*np.array([.5, .25])[None, :, None]
    static_observed = static+.1*rng.standard_normal(static.shape)
    prior_mean = random_prior_mean(backend.sampler.L, backend.sampler.M_mean, .5,
                                  entry['prior_seed']).cpu().numpy()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): raise FileExistsError(destination)
    np.savez_compressed(destination, scenario=entry['scenario'], case_id=entry['case_id'],
                        truth=truth, logk_truth=geology, observed=observed,
                        k_observed=static_observed, weights=weights, k_weights=kw,
                        qoi_weights=qoi, prior_mean=prior_mean,
                        point_xy=np.array([(p[1], p[2]) for p in points]))


def calibrate(backend, entries, destination):
    """Shared second-moment covariance, from exact validation FE realizations.

    Fixed per-time variances avoid using future test truth to set score scales.
    The pressure spatial common component is separated before the local fit.
    Field and well-block variances are fitted separately, never interchanged.
    """
    local, common, field, qoi = [[] for _ in range(11)], [[] for _ in range(11)], [[] for _ in range(11)], [[] for _ in range(11)]
    with torch.no_grad():
        for count, entry in enumerate(entries):
            gi = backend.ds.scen_list.index(entry['scenario'])
            row = entry['row']
            wx, wy = (float(v) for v in backend.ds.S[row, 3:5])
            w = masks(network_points(wx, wy, 'opt17'))
            qw = masks([('producer', wx, wy), ('injector', 0., -250.)])
            m = torch.tensor(np.array(backend.real[row], copy=True), device=backend.device)
            for ti in range(11):
                pred = backend.predict_real(gi, ti, m)[0].cpu().numpy()
                error = pred-np.array(backend.fields[row, ti], copy=True)
                block = np.einsum('chw,phw->cp', error, w)
                pc = block[1].mean()
                block[1] -= pc
                local[ti].append((block**2).mean(1))
                common[ti].append(pc**2)
                field[ti].append((error[:, 16:144]**2).mean((1, 2)))
                qoi[ti].append(np.einsum('chw,phw->cp', error, qw)**2)
            print(json.dumps(dict(stage='calibration', completed=count+1, total=len(entries))), flush=True)
    result = dict(block_local_variance=np.maximum(np.mean(local, axis=1), 1e-8),
                  pressure_common_variance=np.mean(common, axis=1),
                  field_variance=np.maximum(np.mean(field, axis=1), 1e-8),
                  qoi_variance=np.maximum(np.mean(qoi, axis=1), 1e-8),
                  entries=entries, precision='FP32 neural network; FP64 likelihood; TF32 disabled',
                  interpretation='Zero-mean Gaussian approximation using validation residual second moments; no test fitting',
                  checkpoint_hashes={str(p): sha(p) for p in CHECKPOINTS.values()},
                  source_sha256=sha(__file__))
    if Path(destination).exists(): raise FileExistsError(destination)
    write_json(destination, result)
