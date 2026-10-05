"""Matched budget/engineering-constraint study; frozen GeoUNO and ES-MDA.

Design uses prior draws on validation reservoirs, never test responses. The
linear-Gaussian design criterion is the existing registered greedy criterion.
Field/well forecast scores are evaluated independently on all 24 test truths.
"""
from .common import *
import argparse
import fcntl
import os
import time
import numpy as np

DEST = OUT/'monitoring_budget'
BUDGETS = [2, 3, 5, 7, 9, 13, 17, 25, 33]
CONTROLS = [9, 17, 33]
FAMILIES = ['free', 'line', 'region']


def geometry(points, family):
    """Same named candidate indices across well geometries; no truth inputs."""
    from history_matching_dense.networks import network_points
    inj, prod = np.asarray(points[:2], float)
    if family == 'line':
        # Two existing wells plus 127 possible additional positions on their line.
        xy = np.vstack([inj, prod, inj + np.arange(1,128)[:,None]/128*(prod-inj)])
    elif family == 'region':
        # Illustrative access strip on one side of the existing well axis.
        # All new instruments must lie within the strip; existing wells remain.
        v=prod-inj; normal=np.array([-v[1],v[0]])/np.linalg.norm(v)
        xy=np.vstack([inj,prod,*[inj+f*v+s*normal for f in np.linspace(.05,.95,19)
                                for s in np.linspace(30,150,9)]])
    else:
        assert family=='free'
        xy=np.array([p[1:] for p in network_points(*prod,'hybrid385')])
    assert np.max(np.abs(xy))<=400, 'Access geometry leaves the modeled permeability domain'
    return [(f'monitor{i}',float(x),float(y)) for i,(x,y) in enumerate(xy)]


def configurations():
    result=[]
    for family in FAMILIES:
        for n in BUDGETS:
            if n==2 and family!='free':continue
            reuse=f'opt{n}_Tpk' if family=='free' and n in (5,9,17) else None
            result.append(dict(key=f'{family}_opt{n}',family=family,n=n,rule='optimized',reuse=reuse))
        for n in CONTROLS:
            result.append(dict(key=f'{family}_space{n}',family=family,n=n,rule='space_filling',reuse=None))
    return result


def register():
    verify();bind_monitoring()
    from joint_monitoring.common import protocol as original,SOURCE
    p=original();DEST.mkdir(exist_ok=True)
    dest=DEST/'protocol.json'
    if dest.exists():return protocol()
    cases={e['label']:SOURCE/'cases'/f"{e['label']}.npz" for e in p['cases']['pilot']}
    cases.update({e['label']:APPS/'cases/opt17_Tpk'/f"{e['label']}.npz" for e in p['cases']['test']})
    # Validate every geometry before registering the proposed design space.
    from inference_comparison.geothermal import masks
    for path in cases.values():
        with np.load(path) as a:
            for family in FAMILIES:
                xy=geometry(a['point_xy'],family)
                if family!='line':masks(xy);masks(xy,True)
    result=dict(checkpoint_sha256=p['checkpoint_sha256'],cases=p['cases'],budgets=BUDGETS,
        configs=configurations(),esmda=p['esmda'],prior_samples=256,design_samples=64,
        calibration_sha256=sha(APPS/'calibration.json'),case_paths={k:str(v) for k,v in cases.items()},
        case_hashes={k:sha(v) for k,v in cases.items()},
        source_sha256={str(ROOT/'final_model'/n):sha(ROOT/'final_model'/n) for n in ('monitoring_budget.py','monitoring_budget.sbatch')},
        controls='Same final checkpoint, 24 test reservoirs, prior, 7 dynamic times to 4.02 y, T/p/k readings, covariance, ES-MDA 256 x 8 and initial latent ensemble; only number and coordinates change. Two existing wells count toward every budget.',
        geometry=dict(free='Existing 385-point candidate bank',line='Line segment joining the two fixed wells; optimize position along it, not line orientation.',
            region='Illustrative access strip: 5--95% along the well axis and 30--150 m to its positive-normal side. Two existing wells are retained outside the strip.',
            minimum_separation_m=6,hypothetical=True),
        design='Mean validation-prior normalized ten-year producer-temperature and injector-pressure variance; existing greedy linear-Gaussian reduction. Optimize a common index policy across four validation geometries; no test responses or truth enter selection.',
        baseline='Geometry-only nested farthest-point placement (approximately space-filling, not claimed to be exactly uniform); same candidate sets, endpoints and minimum separation.',
        uncertainty='Unchanged validation-calibrated epistemic covariance and coordinate-keyed aleatoric noise. Forecast scores include epistemic uncertainty.',
        analysis='Report matched per-reservoir field RMSE, well CRPS, NLL, P50 error, P5--P95 coverage and width, plus per-added-location gains. Paired bootstrap intervals across all 24 reservoirs. Compare both channels separately; do not infer saturation from design variance alone. Dense-reference equivalence is descriptive: paired 95% CI for the RMSE-ratio difference lies within +/-0.02, and coverage decreases by no more than 0.05 relative to 33 points. If no budget qualifies, report no demonstrated plateau within the tested range.',
        source_design_sha256=sha(APPS/'design.json'))
    write(dest,result);return result


def protocol():
    verify();p=read(DEST/'protocol.json')
    assert p['checkpoint_sha256']==model()['checkpoint_sha256']
    assert p['calibration_sha256']==sha(APPS/'calibration.json')
    assert all(sha(path)==v for path,v in p['source_sha256'].items())
    return p


def design_cache(backend, entry, family, p):
    import torch
    from joint_monitoring.backend import Problem
    from inference_comparison.geothermal import masks
    path=Path(p['case_paths'][entry['label']]);assert sha(path)==p['case_hashes'][entry['label']]
    with np.load(path) as a:points=geometry(a['point_xy'],family)
    if family=='free':return APPS/'design_prior'/f"{entry['label']}.npz",points
    dest=DEST/'design_prior'/family/f"{entry['label']}.npz";dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():return dest,points
    problem=Problem(backend,path,APPS/'calibration.json');problem.conditional=None
    # Candidate masks may repeat at closely spaced line positions. Selection
    # enforces >=6 m separation and unique actual monitoring blocks below.
    def candidate_masks(geology=False):
        return np.concatenate([masks([point],geology) for point in points])
    w,kw=candidate_masks(),candidate_masks(True)
    ys=np.random.default_rng(81071).normal(size=(64,4096));dynamic=[];geology=[]
    with torch.no_grad():
        for ti in range(11):
            values=[]
            for start in range(0,64,4):
                m=problem.permeability(problem.tensor(ys[start:start+4]));f=backend.predict_real(problem.gi,ti,m)
                values.append(np.einsum('bchw,phw->bcp',f.cpu().numpy(),w))
                if ti==0:geology.append(np.einsum('bhw,phw->bp',m.cpu().numpy(),kw))
            dynamic.append(np.concatenate(values))
    np.savez_compressed(dest,dynamic=np.stack(dynamic),geology=np.concatenate(geology))
    return dest,points


def state(path, cal):
    a=np.load(path);q,k=a['dynamic'],a['geology'];K=64;N=k.shape[1]
    anomalies=lambda x:(x-x.mean(0,keepdims=True))/np.sqrt(K-1)
    H=np.zeros((N,15,K+7))
    for ch,start in [(0,0),(1,7)]:H[:,start:start+7,:K]=anomalies(q[:7,:,ch].transpose(1,0,2)).transpose(2,1,0)
    H[:,14,:K]=anomalies(k).T
    for t in range(7):H[:,7+t,K+t]=np.sqrt(cal['pressure_common_variance'][t])
    noise=np.tile(np.r_[np.array(cal['block_local_variance'])[:7,0]+.5**2,np.array(cal['block_local_variance'])[:7,1]+.25**2,.1**2],(N,1))
    Y=np.zeros((2,K+7));Y[0,:K]=anomalies(q[10,:,0,1]);Y[1,:K]=anomalies(q[10,:,1,0])
    Y/=np.sqrt((Y*Y).sum(1,keepdims=True)).clip(1e-8)
    return dict(H=H,noise=noise,Y=Y,S=np.eye(K+7))


def choose(states, geometries, rule, maximum=33, line=False):
    from history_matching_sparse.design import reduction,update
    xy=np.array([[p[1:] for p in points] for points in geometries]);chosen=[0,1];history=[]
    def add(j):
        for s in states:s['S']=update(s['S'],s['H'][j],s['noise'][j])
    for j in chosen:add(j)
    def record():
        return dict(n=len(chosen),indices=chosen.copy(),normalized_future_variance=float(np.mean([np.trace(s['Y']@s['S']@s['Y'].T)/2 for s in states])))
    history.append(record())
    for n in range(3,maximum+1):
        distance=np.square(xy[:,:,None,:]-xy[:,None,chosen,:]).sum(-1).min(-1)
        eligible=(distance>=6**2-1e-8).all(0);eligible[chosen]=False
        if line:
            # Avoid exhausting a narrow line with an early greedy choice: leave
            # room for the largest registered budget under the spacing rule.
            step=int(np.ceil(6/min(np.linalg.norm(x[1]-x[0]) for x in xy)*128-1e-10))
            ordinal=np.r_[0,128,np.arange(1,128)]
            for j in np.flatnonzero(eligible):
                positions=np.sort(ordinal[chosen+[int(j)]])
                capacity=1+int(np.floor(np.diff(positions)/step).sum())
                if capacity<maximum:eligible[j]=False
        gains=np.mean([reduction(s['S'],s['H'],s['noise'],s['Y']) for s in states],0) if rule=='optimized' else distance.mean(0)
        gains[~eligible]=-np.inf
        assert np.isfinite(gains).any(), 'Insufficient feasible distinct monitoring locations'
        j=int(np.argmax(gains));chosen.append(j);add(j);history.append(record())
    return history


def prepare():
    import torch
    p=register();bind_monitoring()
    if (DEST/'prepared.json').exists():return
    from joint_monitoring.backend import Backend
    from inference_comparison.geothermal import masks
    from history_matching_networks.networks import point_noise
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    b=Backend('val','cpu');cal=read(APPS/'calibration.json');designs={};hashes={}
    for family in FAMILIES:
        cached=[design_cache(b,e,family,p) for e in p['cases']['pilot']]
        for path,_ in cached:hashes[str(path)]=sha(path)
        # Known well coordinates determine feasibility only. No held-out fields,
        # observations, or prior forecasts enter the information criterion.
        geometries=[]
        for path in p['case_paths'].values():
            with np.load(path) as a:geometries.append(geometry(a['point_xy'],family))
        for rule in ['optimized','space_filling']:
            designs[f'{family}_{rule}']=choose([state(path,cal) for path,_ in cached],geometries,rule,line=family=='line')
    # Exact reproduction, before reusing any old monitoring experiment.
    old=read(APPS/'design.json')['designs']
    for n in (5,9,17):
        new=next(d for d in designs['free_optimized'] if d['n']==n)
        assert new['indices']==old[str(n)]['indices']
        np.testing.assert_allclose(new['normalized_future_variance'],old[str(n)]['normalized_future_variance'],rtol=1e-11)
    write(DEST/'design.json',dict(histories=designs,cache_hashes=hashes,validation_only=True,checkpoint_sha256=p['checkpoint_sha256']))
    manifest={}
    for e in p['cases']['test']:
        path=Path(p['case_paths'][e['label']]);assert sha(path)==p['case_hashes'][e['label']]
        with np.load(path) as oldcase:
            for c in p['configs']:
                indices=next(d['indices'] for d in designs[c['family']+'_'+c['rule']] if d['n']==c['n'])
                points=[geometry(oldcase['point_xy'],c['family'])[i] for i in indices]
                w,kw=masks(points),masks(points,True)
                xy=np.array([x[1:] for x in points]);dist=np.linalg.norm(xy[:,None]-xy[None],axis=-1)+np.eye(len(xy))*1e6
                assert dist.min()>=6-1e-6
                obs=np.einsum('tchw,phw->tcp',oldcase['truth'],w)
                for ch,sd in [(0,.5),(1,.25)]:obs[:,ch]+=sd*point_noise(points,(11,),e['observation_seed'],str(ch),'monitoring')
                ko=np.einsum('hw,phw->p',oldcase['logk_truth'],kw)+.1*point_noise(points,(),e['observation_seed'],'k','monitoring')
                values={k:oldcase[k] for k in ['scenario','case_id','truth','logk_truth','qoi_weights','prior_mean']}
                values.update(observed=obs,k_observed=ko,weights=w,k_weights=kw,point_xy=xy)
                if c['reuse']:
                    reused=APPS/'cases'/c['reuse']/f"{e['label']}.npz"
                    with np.load(reused) as a:
                        for k,v in values.items():np.testing.assert_array_equal(a[k],v)
                    dest=reused
                else:
                    dest=DEST/'cases'/c['key']/f"{e['label']}.npz";dest.parent.mkdir(parents=True,exist_ok=True)
                    if not dest.exists():np.savez_compressed(dest,**values)
                manifest[c['key']+'/'+e['label']]=dict(path=str(dest),sha256=sha(dest))
    write(DEST/'case_manifest.json',manifest)
    write(DEST/'prepared.json',dict(status='complete',checkpoint_sha256=p['checkpoint_sha256'],configs=len(p['configs']),reservoirs=24,
        design_sha256=sha(DEST/'design.json'),case_manifest_sha256=sha(DEST/'case_manifest.json'),reused_configurations=['opt5_Tpk','opt9_Tpk','opt17_Tpk'],
        checks=['exact existing design reproduction','all reused observations and masks exactly equal','all geometries inside domain','minimum 6 m separation and distinct field/geology blocks','validation-only design selection']))


def folder(c,e):
    if c['reuse']:return APPS/'configs'/c['reuse']/'formal'/e['label']/'esmda'
    return DEST/'formal'/c['key']/e['label']


def run(c,e,p):
    import torch
    bind_monitoring()
    from joint_monitoring.backend import Backend,Problem
    from inference_comparison.algorithms import esmda
    from inference_comparison.run import forecasts,finite_json
    dest=folder(c,e);dest.mkdir(parents=True,exist_ok=True)
    record=read(DEST/'case_manifest.json')[c['key']+'/'+e['label']]
    assert sha(record['path'])==record['sha256']
    b=Backend('test','cpu');torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    problem=Problem(b,record['path'],APPS/'calibration.json');start=time.monotonic()
    if (dest/'latent_samples.npz').exists():y=np.load(dest/'latent_samples.npz')['y']
    else:
        def progress(r):write(dest/'progress.json',finite_json(r|dict(seconds=time.monotonic()-start)))
        y,info=esmda(problem,**p['esmda'],seed=e['observation_seed']+19000,progress=progress)
        np.savez_compressed(dest/'latent_samples.npz',y=y)
        write(dest/'diagnostics.json',finite_json(info|dict(checkpoint_sha256=p['checkpoint_sha256'],case_sha256=record['sha256'],protocol_sha256=sha(DEST/'protocol.json'))))
    write(dest/'scores.json',forecasts(problem,y,dest))
    write(dest/'complete.json',dict(status='complete',checkpoint_sha256=p['checkpoint_sha256'],case_sha256=record['sha256'],seconds=time.monotonic()-start))


def worker():
    p=protocol();prepared=read(DEST/'prepared.json')
    assert sha(DEST/'case_manifest.json')==prepared['case_manifest_sha256']
    assert sha(DEST/'design.json')==prepared['design_sha256']
    start=time.monotonic()
    # Cross-process file locks prevent duplicate experiments and permit restarts.
    # Interleave counts/constraints across all reservoirs rather than truth-selecting.
    for c in p['configs']:
        if c['reuse']:continue
        for e in p['cases']['test']:
            if time.monotonic()-start>10.5*3600:return
            d=folder(c,e);d.mkdir(parents=True,exist_ok=True)
            with (d/'run.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if not (d/'complete.json').exists():
                    print(c['key'],e['label'],flush=True);run(c,e,p)


def collect():
    if not (DEST/'protocol.json').exists():return None
    p=read(DEST/'protocol.json');counts={c['key']:sum((folder(c,e)/'complete.json').exists() for e in p['cases']['test']) for c in p['configs']}
    result=dict(status='complete' if all(v==24 for v in counts.values()) else 'running',counts=counts,total=sum(counts.values()),expected=24*len(counts),checkpoint_sha256=p['checkpoint_sha256'])
    write(DEST/'status.json',result)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['register','prepare','worker','report']);args=ap.parse_args()
    if args.mode=='register':print(register())
    elif args.mode=='prepare':prepare()
    elif args.mode=='worker':worker()
    else:print(collect())
