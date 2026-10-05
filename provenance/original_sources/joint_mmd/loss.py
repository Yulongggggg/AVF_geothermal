"""L = weighted paired MSE + lambda * empirical squared MMD.

MMD uses the biased (V-statistic) empirical squared RKHS distance, including
diagonal terms, valid also for singleton ensembles. Gretton et al., JMLR 2012:
https://jmlr.org/papers/v13/gretton12a.html . Unlike an unpaired U-statistic,
this training discrepancy is zero for identical empirical ensembles.
"""
import torch


def distances(x, y):
    return ((x*x).sum(1)[:,None]+(y*y).sum(1)[None,:]-2*x@y.T).clamp_min(0)/x.shape[1]


def mmd2(x, y):
    # Bandwidth uses FE members only, detached from the optimization. The
    # floor handles physically identical/singleton responses at early times.
    center=y.mean(0,keepdim=True).detach();x=x-center;y=y-center
    yy=distances(y,y)
    upper=yy.detach()[torch.triu(torch.ones_like(yy,dtype=torch.bool),diagonal=1)]
    bandwidth=upper.median().clamp_min(1e-4) if upper.numel() else yy.new_tensor(1e-4)
    def kernel(d):
        return torch.stack([torch.exp(-d/(2*bandwidth*factor)) for factor in (.25,1.,4.)]).mean(0)
    return kernel(distances(x,x)).mean()+kernel(yy).mean()-2*kernel(distances(x,y)).mean()


def objective(pred, truth, lengths, scales, residual_scale, weight=.1, temperature_weight=1., kind='mse'):
    assert sum(lengths)==len(pred)==len(truth) and all(n>0 for n in lengths)
    p,y=pred.float(),truth.float()
    scales=torch.as_tensor(scales,device=p.device).flatten()
    residual_scale=torch.as_tensor(residual_scale,device=p.device).clamp_min(1e-6)
    error=p-y
    assert kind in ('mse','rmse')
    def equal_scenario(v):
        values=torch.stack([g.mean() for g in torch.split(v,lengths)])
        return (values if kind=='mse' else (values+1e-12).sqrt()).mean()
    pe=error[:,1:2,16:144];level=pe.mean((-2,-1),keepdim=True)
    parts={
        'temperature_'+kind:temperature_weight*equal_scenario((error[:,:1,16:144]/scales[0]).square().flatten(1).mean(1)),
        'pressure_level_'+kind:equal_scenario((level/scales[1]).square().flatten(1).mean(1)),
        'pressure_shape_'+kind:equal_scenario(((pe-level)/residual_scale).square().flatten(1).mean(1)),
        'outside_'+kind:.1*equal_scenario((torch.cat([error[:,:,:16],error[:,:,144:]],2)/scales[None,:,None,None]).square().flatten(1).mean(1)),
    }
    # One sample is one complete paired T/p reservoir response. Joint vectors
    # retain spatial structure and T-p dependence, with fixed training scales.
    norm=scales[None,:,None,None]
    pp=(p[:,:,16:144]/norm).flatten(1);yy=(y[:,:,16:144]/norm).flatten(1)
    parts['mmd']=weight*torch.stack([mmd2(a,b) for a,b in zip(torch.split(pp,lengths),torch.split(yy,lengths))]).mean()
    return sum(parts.values()),parts
