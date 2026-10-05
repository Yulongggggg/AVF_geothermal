"""Exact Gaussian conditioning for static local log-permeability data."""
import torch
import torch.nn.functional as F


def condition(mean128,factor,masks,observed,sigma=.1,amplitude=1.):
    """Return latent conditional mean and a symmetric covariance-root factor.

    z = mean + y - (y @ root) @ root.T, y ~ N(0,I).
    This accounts for each static geological observation once. Subsequent T/P
    inversion uses this conditional prior and does not repeat its likelihood.
    """
    device=factor.device;dtype=torch.float64
    masks=masks.to(device=device,dtype=dtype);weights=masks/masks.sum((-2,-1),keepdim=True)
    B=F.interpolate(torch.eye(64,device=device,dtype=dtype).reshape(64,1,64),size=128,mode='linear',align_corners=True)[:,0].T
    A=torch.einsum('pxy,xi,yj->pij',weights,B,B).reshape(len(masks),4096)
    H=amplitude*(A@factor.double())
    offset=(weights*mean128.to(device=device,dtype=dtype)).sum((-2,-1))
    gram=H@H.T;ev,Q=torch.linalg.eigh(gram);ev=ev.clamp_min(0)
    rhs=observed.to(device=device,dtype=dtype)-offset
    latent_mean=H.T@torch.linalg.solve(gram+sigma**2*torch.eye(len(masks),device=device,dtype=dtype),rhs)
    s=torch.sqrt(ev+sigma**2)
    coefficient=1/(s*(s+sigma))
    root=(H.T@Q)*torch.sqrt(coefficient)[None]
    return dict(mean=latent_mean.float(),root=root.float(),H=H.float(),offset=offset.float(),sigma=float(sigma))


def transform(y,posterior):
    if posterior is None:return y
    root=posterior['root'];return posterior['mean'][None]+y-(y@root)@root.T


def common_mode_quadratic(error,diagonal_sigma,common_sigma):
    """Error (...,time,point), diagonal SD (time,point), common SD (time)."""
    # Dense arrays can otherwise lose precision when subtracting a common mode.
    error=error.double()
    inv=diagonal_sigma.double().square().reciprocal()
    common_variance=common_sigma.double().square()
    diagonal=(error.square()*inv).sum(-1)
    correction=(error*inv).sum(-1).square()*common_variance/(1+common_variance*inv.sum(-1))
    return (diagonal-correction).sum(-1)
