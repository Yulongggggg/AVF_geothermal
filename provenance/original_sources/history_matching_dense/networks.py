"""Nested monitoring arrays chosen from well geometry, without truth access."""
import numpy as np
from functools import lru_cache
from history_matching_networks.networks import network_points as original_points,point_key,point_noise

NETWORKS={
    'corridor17':dict(n=17,label='Original 17-point corridor'),
    'corridor49':dict(n=49,label='49 points covering the inter-well corridor'),
    'global49':dict(n=49,label='49 points covering the reservoir'),
    'hybrid97':dict(n=97,label='49-point corridor plus 48 reservoir points'),
    'hybrid193':dict(n=193,label='97-point hybrid plus 96 reservoir points'),
    'hybrid385':dict(n=385,label='Dense synthetic 385-point hybrid'),
}


def extend(points,wx,wy,target,corridor=False):
    axis=np.linspace(-360,360,21);yy,xx=np.meshgrid(axis,axis,indexing='ij');candidates=np.stack([xx.ravel(),yy.ravel()],1)
    if corridor:
        origin=np.array([0.,-250.]);v=np.array([wx,wy+250.]);f=np.clip((candidates-origin)@v/(v@v),0,1)
        candidates=candidates[np.linalg.norm(candidates-origin-f[:,None]*v,axis=1)<=180]
    points=list(points)
    while len(points)<target:
        xy=np.array([p[1:] for p in points]);distance=np.square(candidates[:,None]-xy[None]).sum(-1).min(1)
        i=int(np.argmax(distance))
        if distance[i]<12**2:raise ValueError('Insufficient distinct spatial candidates')
        x,y=candidates[i];points.append((f'monitor{len(points)}',float(x),float(y)))
        candidates=np.delete(candidates,i,axis=0)
    return points


@lru_cache(maxsize=4096)
def network_points(wx,wy,network):
    base=original_points(wx,wy,'corridor17')
    if network=='corridor17':return base
    if network=='global49':return extend(base,wx,wy,49)
    dense=extend(base,wx,wy,49,True)
    if network=='corridor49':return dense
    dense=extend(dense,wx,wy,97)
    if network=='hybrid97':return dense
    dense=extend(dense,wx,wy,193)
    if network=='hybrid193':return dense
    if network=='hybrid385':return extend(dense,wx,wy,385)
    if network=='bank':
        combined=network_points(wx,wy,'hybrid385')+network_points(wx,wy,'global49')
        return list({point_key(p):p for p in combined}.values())
    raise ValueError(network)
