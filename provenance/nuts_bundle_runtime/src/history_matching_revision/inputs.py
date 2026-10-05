"""Verbatim scientific definitions extracted from pinned original source."""
import torch
from det_sampler import field_on_grid

def realized_inputs(base, logk128, mean, std):
    """Replace only log-k, using the exact stored FE realization for calibration."""
    fields=torch.as_tensor(logk128,device=base.device,dtype=torch.float32)
    if fields.ndim==2:fields=fields[None]
    result=base.float().expand(len(fields),-1,-1,-1).clone()
    result[:,0]=(field_on_grid(fields[:,None])[:,0]-mean)/std
    return result
