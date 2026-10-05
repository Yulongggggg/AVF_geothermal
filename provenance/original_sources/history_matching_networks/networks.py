"""Predetermined monitoring networks and random numbers shared by location.

Coordinates use well geometry only. Neither truth fields nor posterior results
are available to the network generator.
"""
import hashlib
import numpy as np

NETWORKS = {
    'wells2': dict(n=2,label='2 points: wells only'),
    'cross5': dict(n=5,label='5 points: original cross'),
    'axial5': dict(n=5,label='5 points: well axis'),
    'corridor9': dict(n=9,label='9 points: cross and axis'),
    'corridor17': dict(n=17,label='17 points: corridor, 120 m flanks'),
    'wide17': dict(n=17,label='17 points: corridor, 240 m flanks'),
}


def point_key(point):
    return f'{float(point[1]):.6f},{float(point[2]):.6f}'


def network_points(wx,wy,network):
    if network == 'bank':
        points = [p for name in NETWORKS for p in network_points(wx,wy,name)]
    else:
        if network not in NETWORKS: raise ValueError(network)
        vx,vy=float(wx),float(wy)+250.
        length=np.hypot(vx,vy)+1e-9
        nx,ny=-vy/length,vx/length
        def at(name,fraction,offset=0.):
            return (name,float(np.clip(fraction*vx+offset*nx,-390,390)),
                    float(np.clip(-250+fraction*vy+offset*ny,-390,390)))
        points=[at('inj',0.),at('prod',1.)]
        if network == 'axial5':
            points += [at('axial25',.25),at('mid',.5),at('axial75',.75)]
        elif network != 'wells2':
            points += [at('mid',.5),at('perp+',.5,120),at('perp-',.5,-120)]
            if network in ('corridor9','corridor17','wide17'):
                points += [at(f'axis{i}',i/6) for i in (1,2,4,5)]
            if network in ('corridor17','wide17'):
                offset=120 if network=='corridor17' else 240
                points += [at(f'flank{i}{sign:+d}',i/5,sign*offset)
                           for i in (1,2,3,4) for sign in (1,-1)]
    unique={}
    for p in points: unique.setdefault(point_key(p),p)
    points=list(unique.values())
    if network != 'bank' and len(points)!=NETWORKS[network]['n']:
        raise ValueError(f'Clipping collapsed locations for {network}: {(wx,wy)}')
    return points


def point_noise(points,shape,seed,channel,purpose):
    """Normal variates whose value is invariant to network size and ordering."""
    samples=[]
    for p in points:
        key=f'{seed}|{purpose}|{channel}|{point_key(p)}'.encode()
        local_seed=int.from_bytes(hashlib.sha256(key).digest()[:8],'little')
        samples.append(np.random.default_rng(local_seed).standard_normal(shape))
    return np.stack(samples,axis=-1)
