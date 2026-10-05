"""Author-selected immutable transfer operator in the physical inference interface."""
import csv
import sys
import numpy as np
import torch
from torch import nn
from .common import PACK, FACTOR, DATA_ROOT, candidate, environment

# A few historical inference utilities were never present in this training
# checkout. Their original source files are explicitly pinned in our protocol.
if str(DATA_ROOT) not in sys.path:
    sys.path.append(str(DATA_ROOT))

# Old common modules set paths relative to their code checkout at import time.
# Bind paths before importing GeoSetT; never create a misleading data symlink.
import inference_comparison.common as inherited
inherited.PACK = PACK
inherited.FACTOR = FACTOR
environment()
from inference_comparison.geothermal import Backend as PhysicalBackend
import train_t as T
T.PACK = PACK
from direct_operator.inference import SamplingAdapter
from det_sampler import RealSampler
from operator_campaign_20260925.model import load as load_operator
from operator_parallel_20260927_v2.runtime import fixed_pressure_pool


def load(path):
    net = fixed_pressure_pool(load_operator(path))
    assert net.config['architecture'] == 'geouno_basis_head'
    return net.eval()


class Bundle(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.joint = load(path)

    def forward(self, field, scalar):
        return self.joint(field,scalar)


class Backend(PhysicalBackend):
    def __init__(self, split='val', device='cuda'):
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.mha.set_fastpath_enabled(False)
        environment(); T.PACK = PACK
        self.device = torch.device(device)
        self.ds = T.GeoSetT(split,n_obs=1,with_targets=False,require_g=False,in_pp='abs')
        self.net = SamplingAdapter(Bundle(candidate()['checkpoint']),self.ds.st).to(device).eval().requires_grad_(False)
        self.sampler = RealSampler(self.ds.st,device)
        self.factor = torch.tensor(np.load(FACTOR),device=device)
        self.fields = np.load(PACK/'fields.npy',mmap_mode='r')
        self.real = np.load(PACK/'real.npy',mmap_mode='r')
        self.meta = list(csv.DictReader((PACK/'meta.csv').open()))
        self.mean = torch.tensor(self.ds.st['fields_mean'],device=device).float()[None,:,None,None]
        self.std = torch.tensor(self.ds.st['fields_std'],device=device).float()[None,:,None,None]
        self.evaluations = 0
