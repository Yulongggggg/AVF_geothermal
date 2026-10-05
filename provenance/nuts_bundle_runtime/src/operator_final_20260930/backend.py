"""I/O-only adapter: exact frozen bases replace the large original dataset."""
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from portable_runtime.io import BUNDLE,read,settings
from direct_operator.inference import SamplingAdapter
from operator_campaign_20260925.model import load as load_operator
from operator_parallel_20260927_v2.runtime import fixed_pressure_pool
from history_matching_revision.inputs import realized_inputs

def load(path):
    net=fixed_pressure_pool(load_operator(path))
    assert net.config['architecture']=='geouno_basis_head'
    return net.eval()

class Bundle(nn.Module):
    def __init__(self,path):
        super().__init__();self.joint=load(path)
    def forward(self,field,scalar):return self.joint(field,scalar)

class Backend:
    def __init__(self,split='val',device='cpu'):
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.mha.set_fastpath_enabled(False)
        self.device=torch.device(device)
        self.entries=[e for e in settings()['cases'] if e['split']==split]
        if not self.entries:raise ValueError(f'No exported cases for split {split}')
        st=read(BUNDLE/self.entries[0]['stats_path'])
        self.ds=SimpleNamespace(scen_list=[e['scenario'] for e in self.entries],st=st)
        assert len(set(self.ds.scen_list))==len(self.entries)
        self.net=SamplingAdapter(Bundle(BUNDLE/'data/model.pt'),st).to(device).eval().requires_grad_(False)
        self.sampler=SimpleNamespace(M_mean=float(st['M_mean']),M_std=float(st['M_std']))
        self.factor=torch.tensor(np.load(BUNDLE/'data/prior_factor.npy'),device=device)
        self.mean=torch.tensor(st['fields_mean'],device=device).float()[None,:,None,None]
        self.std=torch.tensor(st['fields_std'],device=device).float()[None,:,None,None]
        self.evaluations=0
        self._base_arrays={}
    def base(self,gi,ti):
        if gi not in self._base_arrays:
            with np.load(BUNDLE/self.entries[gi]['base_path']) as z:self._base_arrays[gi]=z['base'].copy()
        return torch.as_tensor(self._base_arrays[gi][ti],device=self.device)[None].clone()
    # Verbatim PhysicalBackend prediction definition; see provenance original.
    def predict_real(self, gi, ti, m):
        x = realized_inputs(self.base(gi, ti), m, self.sampler.M_mean, self.sampler.M_std)
        self.evaluations += len(x)
        return self.net.mean_modes(x)[0].float()*self.std+self.mean
