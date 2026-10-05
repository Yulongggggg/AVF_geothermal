"""Validation-prior design and matched observations on redesigned access geometries."""
import time
import numpy as np
import torch
from .common import *
from .inverse_backend import Backend,Problem
from .monitoring_geometry import geometry,FAMILIES
from inference_comparison.geothermal import masks
from history_matching_networks.networks import point_noise
from final_model.monitoring_budget import state,choose

DEST=OUT/'monitoring'
BUDGETS=(2,3,5,7,9,13,17,25,33)


def configurations():
    configs=[]
    for family in FAMILIES:
        for rule,counts in [('optimized',BUDGETS),('space_filling',(5,9,17,33))]:
            for n in counts:
                if n==2 and family!='multiwell':continue
                configs.append(dict(key=f'{family}_{"opt" if rule=="optimized" else "space"}{n}',family=family,
                    rule=rule,n=n,channels=[0,1],static=True,times=list(range(7)),sigma_t=.5,sigma_p=.25,kind='budget'))
    for label,channels,static in [('Tp',[0,1],False),('T',[0],False),('p',[1],False),('Tk',[0],True),('pk',[1],True),('k',[],True)]:
        configs.append(dict(key=f'multiwell_opt9_{label}',family='multiwell',rule='optimized',n=9,channels=channels,
            static=static,times=list(range(7)),sigma_t=.5,sigma_p=.25,kind='channels'))
    for last in (4,5):
        configs.append(dict(key=f'multiwell_opt17_end{last}',family='multiwell',rule='optimized',n=17,channels=[0,1],
            static=True,times=list(range(last+1)),sigma_t=.5,sigma_p=.25,kind='duration'))
    for multiplier in (.2,2.):
        for label,channels in [('Tk',[0]),('Tpk',[0,1])]:
            configs.append(dict(key=f'multiwell_opt9_{label}_noise{multiplier:g}',family='multiwell',rule='optimized',n=9,channels=channels,
                static=True,times=list(range(7)),sigma_t=.5*multiplier,sigma_p=.25,kind='noise',test_limit=6))
    return configs


def register():
    DEST.mkdir(exist_ok=True)
    path=DEST/'protocol.json'
    if path.exists():return read(path)
    candidate();assert read(OUT/'qualification/inverse_cpu.json')['status']=='passed'
    assert read(OUT/'calibration_complete.json')['checkpoint_sha256']==EXPECTED_SHA
    cases=read(SOURCE/'protocol.json')['cases'];raw={};geometries={}
    for e in cases['pilot']+cases['test']:
        f=SOURCE/'cases'/f"{e['label']}.npz";a=np.load(f)
        assert str(a['scenario'])==e['scenario'] and str(a['case_id'])==e['case_id']
        raw[e['label']]=dict(path=str(f),sha256=sha(f))
        geometries[e['label']]={}
        for family in FAMILIES:
            points,details=geometry(a['point_xy'],family);masks(points);masks(points,True)
            geometries[e['label']][family]=dict(points=points,**details)
    p=dict(checkpoint_sha256=EXPECTED_SHA,calibration_sha256=sha(OUT/'calibration.json'),
        cases={k:cases[k] for k in ('pilot','test')},raw_cases=raw,geometries=geometries,
        configs=configurations(),esmda=dict(samples=256,steps=8),design_samples=64,prior_samples=256,
        prior_seed=81071,design_seed=81071,noise='Coordinate-keyed common Gaussian draws; changing sigma rescales the same errors; permeability sigma=0.1 dec',
        budget_includes_two_operating_wells=True,counts_are_observation_locations_not_drilling_cost=True,
        geometry='Vertical and inclined independent monitoring boreholes replace the previous connecting line and strip; unrestricted access represents multiple parallel monitoring laterals.',
        selection='Common candidate-index policy optimized over four validation priors; only known well geometry determines feasible test candidates. No test response enters placement.',
        noise_subset='First six registered test reservoirs, fixed before new inference results',
        analysis='Equal-reservoir means; paired bootstrap 10000 resamples; compare producer-temperature width, P50 error and CRPS together; analyze field RMSE separately.',
        sources={str(f):sha(f) for f in [Path(__file__),ROOT/'operator_final_20260930/monitoring_geometry.py',ROOT/'operator_final_20260930/inverse_backend.py',DATA_ROOT/'final_model/monitoring_budget.py',DATA_ROOT/'history_matching_sparse/design.py']})
    write(path,p);return p


def main():
    p=register()
    for f,h in p['sources'].items():assert sha(f)==h,f
    if (DEST/'prepared.json').exists():return
    b=Backend('val','cpu');torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    cal=read(OUT/'calibration.json');designs={};cache_hashes={}
    for family in FAMILIES:
        cache=[]
        for e in p['cases']['pilot']:
            path=DEST/'design_prior'/family/f"{e['label']}.npz";path.parent.mkdir(parents=True,exist_ok=True)
            points=p['geometries'][e['label']][family]['points']
            if not path.exists():
                problem=Problem(b,p['raw_cases'][e['label']]['path'],OUT/'calibration.json',static=False)
                w=problem.tensor(masks(points));kw=masks(points,True)
                projected=b.observation_basis(w);ys=np.random.default_rng(81071).normal(size=(64,4096))
                dynamic=[];geology=[]
                with torch.no_grad():
                    for start in range(0,64,4):
                        m=problem.permeability(problem.tensor(ys[start:start+4]))
                        values=b.blocks(problem.gi,list(range(11)),m,w,projected)
                        dynamic.append(values.cpu().numpy());geology.append(np.einsum('bhw,phw->bp',m.cpu().numpy(),kw))
                tmp=path.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,dynamic=np.concatenate(dynamic).transpose(1,0,2,3),geology=np.concatenate(geology))
                os.replace(tmp,path)
            cache.append(path);cache_hashes[str(path)]=sha(path)
            print('design prior',family,e['label'],flush=True)
        geometries=[p['geometries'][e['label']][family]['points'] for e in p['cases']['pilot']+p['cases']['test']]
        for rule in ('optimized','space_filling'):
            designs[family+'_'+rule]=choose([state(path,cal) for path in cache],geometries,rule,maximum=33,line=False)
    write(DEST/'design.json',dict(histories=designs,cache_hashes=cache_hashes,validation_only=True,checkpoint_sha256=EXPECTED_SHA))
    manifest={}
    for e in p['cases']['pilot']+p['cases']['test']:
        raw=p['raw_cases'][e['label']];assert sha(raw['path'])==raw['sha256']
        with np.load(raw['path']) as old:
            for c in p['configs']:
                if c.get('test_limit') and e in p['cases']['test'][c['test_limit']:]:continue
                indices=next(d['indices'] for d in designs[c['family']+'_'+c['rule']] if d['n']==c['n'])
                points=[p['geometries'][e['label']][c['family']]['points'][j] for j in indices]
                w,kw=masks(points),masks(points,True);xy=np.array([pt[1:] for pt in points])
                distance=np.linalg.norm(xy[:,None]-xy[None],axis=-1)+np.eye(len(xy))*1e6
                assert distance.min()>=6.-1e-6
                obs=np.einsum('tchw,phw->tcp',old['truth'],w)
                for ch,sd in [(0,c['sigma_t']),(1,c['sigma_p'])]:
                    obs[:,ch]+=sd*point_noise(points,(11,),e['observation_seed'],str(ch),'monitoring')
                ko=np.einsum('hw,phw->p',old['logk_truth'],kw)+.1*point_noise(points,(),e['observation_seed'],'k','monitoring')
                values={k:old[k] for k in ('scenario','case_id','truth','logk_truth','qoi_weights','prior_mean')}
                values.update(observed=obs,k_observed=ko,weights=w,k_weights=kw,point_xy=xy)
                file=DEST/'cases'/c['key']/f"{e['label']}.npz";file.parent.mkdir(parents=True,exist_ok=True)
                if not file.exists():
                    tmp=file.with_suffix('.tmp.npz');np.savez_compressed(tmp,**values);os.replace(tmp,file)
                else:
                    with np.load(file) as a:
                        for key,value in values.items():np.testing.assert_array_equal(a[key],value)
                manifest[c['key']+'/'+e['label']]=dict(path=str(file),sha256=sha(file))
    write(DEST/'case_manifest.json',manifest)
    write(DEST/'prepared.json',dict(status='complete',checkpoint_sha256=EXPECTED_SHA,
        design_sha256=sha(DEST/'design.json'),case_manifest_sha256=sha(DEST/'case_manifest.json'),
        configs=len(p['configs']),raw_truth_reused=True,old_predictions_reused=False,
        geometry_checks='Independent well paths, >=20 m operating-well clearance, >=6 m reading separation, distinct dynamic/static blocks'))


if __name__=='__main__':main()
