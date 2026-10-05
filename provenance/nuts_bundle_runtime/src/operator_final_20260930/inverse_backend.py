"""Fixed-basis observation contractions, equivalent to averaging dense outputs."""
import numpy as np
import torch
from .backend import Backend as CheckedBackend
from inference_comparison.geothermal import Problem as BaseProblem
from history_matching_revision.inputs import realized_inputs


class Backend(CheckedBackend):
    def __init__(self,split='val',device='cpu'):
        super().__init__(split,device)
        self.operator=self.net.bundle.joint
        with torch.no_grad():self.fixed_basis=tuple(v.detach() for v in self.operator.basis())
        original_basis=self.operator.basis
        self.operator.basis=lambda query=None:self.fixed_basis if query is None else original_basis(query)
        self.base_cache={}

    def base(self,gi,ti):
        key=(gi,ti)
        if key not in self.base_cache:self.base_cache[key]=super().base(gi,ti).detach()
        return self.base_cache[key]

    def observation_basis(self,weights):
        basis,mean=self.fixed_basis
        w=weights.reshape(len(weights),-1).double()
        return torch.einsum('crn,pn->crp',basis.double(),w),torch.einsum('cn,pn->cp',mean.double(),w)

    def blocks(self,gi,times,m,weights,projected=None):
        if m.ndim==2:m=m[None]
        fields=[];scalars=[]
        for ti in times:
            x=realized_inputs(self.base(gi,ti),m,self.sampler.M_mean,self.sampler.M_std)
            f,s=self.net.encoder(x);fields.append(f);scalars.append(s)
        f,s=torch.cat(fields),torch.cat(scalars)
        coefficients=[];levels=[]
        for first in range(0,len(f),4):
            c,l=self.operator.branch(f[first:first+4],s[first:first+4]);coefficients.append(c);levels.append(l)
        c,l=torch.cat(coefficients).double(),torch.cat(levels).double()
        basis,mean=self.observation_basis(weights) if projected is None else projected
        v=torch.einsum('bcr,crp->bcp',c,basis)+mean[None]
        t=(v[:,0]+l[:,0:1])*self.operator.scales[0].double()
        p=v[:,1]*self.operator.pressure_residual_scale+l[:,1:2]*self.operator.scales[1].double()
        self.evaluations+=len(f)
        return torch.stack((t,p),1).reshape(len(times),len(m),2,len(weights)).permute(1,0,2,3)


class Problem(BaseProblem):
    def __init__(self,backend,case_path,calibration_path,channels=(0,1),static=True,times=tuple(range(7)),sigma_t=.5,sigma_p=.25):
        super().__init__(backend,case_path,calibration_path)
        self.channels=list(channels);self.times=list(times);self.sigma_t=sigma_t;self.sigma_p=sigma_p
        if not static:self.conditional=None
        self.projected=backend.observation_basis(self.weights)
        n=len(self.weights);covs=[]
        for ti in self.times:
            for ch in self.channels:
                variance=self.cal['block_local_variance'][ti][ch]+(sigma_t if ch==0 else sigma_p)**2
                cov=variance*torch.eye(n,dtype=torch.float64,device=self.device)
                if ch==1:cov=cov+self.cal['pressure_common_variance'][ti]*torch.ones_like(cov)
                covs.append(cov)
        self.cholesky=torch.linalg.cholesky(torch.block_diag(*covs)) if covs else self.tensor(np.empty((0,0)))
        observed=self.tensor(self.case['observed'])[self.times][:,self.channels]
        self.data=self.whiten(observed.flatten()[None])[0].detach() if covs else self.tensor([])
        self.nobs=len(self.data)

    def forward(self,y):
        self.calls+=1
        values=self.backend.blocks(self.gi,self.times,self.permeability(y),self.weights,self.projected)
        return self.whiten(values[:,:,self.channels].flatten(1))
