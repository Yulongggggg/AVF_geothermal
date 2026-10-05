"""Verbatim scientific definitions extracted from pinned original source."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from history_matching_dense.conditioning import condition,transform

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
