"""Physical input construction for prior generation and paired calibration."""
import math
import torch
from torch.nn import functional as F
from det_sampler import field_on_grid


def random_prior_mean(cholesky, training_mean, amplitude, seed):
    """Independent smooth log-k mean; no target reservoir enters this function."""
    side=math.isqrt(cholesky.shape[0])
    if side*side!=cholesky.shape[0]:raise ValueError('Prior covariance must be on a square grid')
    g=torch.Generator(device='cpu').manual_seed(seed)
    z=torch.randn(1,side*side,generator=g).to(cholesky.device)
    field=(z@cholesky.T).reshape(1,1,side,side)*amplitude+training_mean
    return F.interpolate(field,size=(128,128),mode='bilinear',align_corners=True)[0,0]


def realized_inputs(base, logk128, mean, std):
    """Replace only log-k, using the exact stored FE realization for calibration."""
    fields=torch.as_tensor(logk128,device=base.device,dtype=torch.float32)
    if fields.ndim==2:fields=fields[None]
    result=base.float().expand(len(fields),-1,-1,-1).clone()
    result[:,0]=(field_on_grid(fields[:,None])[:,0]-mean)/std
    return result
