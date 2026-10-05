"""Matched final-model ES-MDA, MAP and prior forecasts with resumable tasks."""
import argparse,fcntl,time,inspect
import numpy as np
import torch
from scipy.optimize import minimize
from .common import *
from .inverse_backend import Backend,Problem
from inference_comparison.algorithms import esmda
from inference_comparison.metrics import score
from inference_comparison.run import finite_json,gradient_check

DEST=OUT/'monitoring'


def atomic_npz(path,**values):
    path=Path(path);tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**values);os.replace(tmp,path)


def forecast(problem,latent,dest,save_members=False,point_estimate=False):
    latent=np.asarray(latent).reshape(-1,4096);fields=[];geology=[];qoi=[]
    with torch.no_grad():
        for first in range(0,len(latent),4):
            m=problem.permeability(problem.tensor(latent[first:first+4]));f=problem.backend.predict_real(problem.gi,10,m)
            fields.append(f[:,:,16:144].cpu().numpy());geology.append(m.cpu().numpy())
            qoi.append(torch.einsum('bchw,phw->bcp',f.double(),problem.qoi_weights).cpu().numpy())
    fields=np.concatenate(fields);geology=np.concatenate(geology);qoi=np.concatenate(qoi)
    truth=problem.case['truth'][10,:,16:144]
    truth_qoi=np.einsum('chw,phw->cp',problem.case['truth'][10],problem.case['qoi_weights'])
    result=dict(members=len(latent),field_rmse=np.sqrt(np.mean((fields.mean(0)-truth)**2,axis=(1,2))).tolist(),
        permeability_rmse=float(np.sqrt(np.mean((geology.mean(0)-problem.case['logk_truth'])**2))),blocks={})
    for ch,label in enumerate(('temperature','pressure')):
        draws=qoi[:,ch,ch];target=truth_qoi[ch,ch];sd=np.sqrt(problem.cal['qoi_variance'][10][ch][ch])
        result['blocks'][label]={}
        if point_estimate:
            assert len(draws)==1
            result['blocks'][label]=dict(absolute_error=float(abs(draws[0]-target)),point_prediction=float(draws[0]),posterior_interval=None)
            continue
        for level,error_sd in [('raw',None),('with_surrogate_epistemic',sd)]:
            result['blocks'][label][level]={k:float(v) for k,v in score(draws,target,error_sd).items()}
    payload=dict(field_mean=fields.mean(0),field_sd=fields.std(0),field_p05=np.quantile(fields,.05,axis=0),
        field_p95=np.quantile(fields,.95,axis=0),qoi_samples=qoi,truth_qoi=truth_qoi,truth_fields=truth,
        logk_mean=geology.mean(0),logk_sd=geology.std(0),truth_logk=problem.case['logk_truth'])
    if save_members:payload.update(field_samples=fields,logk_samples=geology)
    atomic_npz(dest/'forecast.npz',**payload);write(dest/'scores.json',result)
    return result


def map_optimize(problem,dest,seed,options):
    starts=[np.zeros(4096)]+list(np.random.default_rng(seed).normal(size=(2,4096)));summaries=[];solutions=[]
    for i,initial in enumerate(starts):
        file=dest/f'start_{i}.npz';meta=dest/f'start_{i}.json'
        if file.exists() and meta.exists():solutions.append(np.load(file)['y']);summaries.append(read(meta));continue
        count=0;begin=time.monotonic()
        def objective(z):
            nonlocal count
            y=problem.tensor(z).requires_grad_(True);r=problem.forward(y[None])[0]-problem.data
            u=.5*(y.square().sum()+r.square().sum());g,=torch.autograd.grad(u,y)
            assert torch.isfinite(u) and torch.isfinite(g).all()
            count+=1
            if count%20==0:write(dest/'progress.json',dict(start=i,evaluations=count,objective=u.item(),seconds=time.monotonic()-begin))
            return u.item(),g.detach().cpu().numpy()
        r=minimize(objective,initial,jac=True,method='L-BFGS-B',options=options)
        info=dict(start=i,success=bool(r.success),message=str(r.message),objective=float(r.fun),
            gradient_inf=float(abs(r.jac).max()),iterations=int(r.nit),evaluations=int(r.nfev),seconds=time.monotonic()-begin)
        atomic_npz(file,y=r.x);write(meta,info);solutions.append(r.x);summaries.append(info)
    best=int(np.argmin([s['objective'] for s in summaries]));return solutions[best][None],dict(starts=summaries,
        selected_start=best,success=summaries[best]['success'],seconds=sum(s['seconds'] for s in summaries),
        evaluations=sum(s['evaluations'] for s in summaries),point_estimate=True)


def register():
    p=read(DEST/'protocol.json');path=DEST/'inference_protocol.json'
    if path.exists():return read(path)
    configs=p['configs'];priority=['multiwell_opt17','multiwell_opt9','multiwell_opt9_T','multiwell_opt9_p',
        'multiwell_opt9_Tp','multiwell_opt9_Tk','multiwell_opt9_pk','multiwell_opt9_k']
    configs=sorted(configs,key=lambda c:(priority.index(c['key']) if c['key'] in priority else len(priority),c['key']))
    tasks=[]
    for c in configs:
        for e in p['cases']['test'][:c.get('test_limit',24)]:tasks.append(dict(config=c['key'],case=e['label'],method='esmda'))
    tasks=[dict(config='multiwell_opt17',case=e['label'],method='map') for e in p['cases']['test']]+tasks
    result=dict(checkpoint_sha256=EXPECTED_SHA,monitoring_protocol_sha256=sha(DEST/'protocol.json'),
        calibration_sha256=sha(OUT/'calibration.json'),tasks=tasks,
        map_optimizer=dict(maxiter=1000,maxfun=20000,maxls=40,gtol=.0001,ftol=1e-10,maxcor=10),
        map_starts='zero plus two seeded prior vectors, select lowest posterior objective',
        nuts=dict(samples=1024,warmup=512,chains=4,max_depth=7,target_accept=.9,cases=6),
        paired_ensemble_seed='case observation_seed + 19000, same across all configurations',
        sources={str(f):sha(f) for f in [Path(__file__),ROOT/'operator_final_20260930/inverse_backend.py',
          Path(inspect.getfile(esmda)),Path(inspect.getfile(score))]})
    write(path,result);return result


def problem_for(b,c,e,p):
    record=read(DEST/'case_manifest.json')[c['key']+'/'+e['label']]
    assert sha(record['path'])==record['sha256']
    return Problem(b,record['path'],OUT/'calibration.json',**{k:c[k] for k in ('channels','static','times','sigma_t','sigma_p')}),record


def run_task(b,c,e,method,p,rules,phase='formal'):
    dest=DEST/phase/c['key']/e['label']/method;dest.mkdir(parents=True,exist_ok=True)
    with (dest/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        if (dest/'complete.json').exists():return False
        problem,record=problem_for(b,c,e,p);began=time.monotonic();before=b.evaluations
        latent=dest/'latent_samples.npz'
        if latent.exists():y=np.load(latent)['y'];info=read(dest/'diagnostics.json')
        else:
            def progress(r):write(dest/'progress.json',finite_json(r|dict(seconds=time.monotonic()-began)))
            if method=='map':y,info=map_optimize(problem,dest,e['observation_seed']+29000,rules['map_optimizer'])
            elif c['channels']:y,info=esmda(problem,**p['esmda'],seed=e['observation_seed']+19000,progress=progress)
            else:
                # Match the actual ES-MDA member vectors, not only its seed.
                # esmda() draws parameters x members and then transposes.
                y=np.random.default_rng(e['observation_seed']+19000).normal(
                    size=(problem.dim,p['esmda']['samples'])).T
                info=dict(seconds=0.,method='analytic static conditioning')
            info.update(checkpoint_sha256=EXPECTED_SHA,case_sha256=record['sha256'],calibration_sha256=p['calibration_sha256'],
                method=method,device=str(b.device),threads=torch.get_num_threads(),
                hardware=torch.cuda.get_device_name(b.device) if b.device.type=='cuda' else __import__('platform').processor(),
                monitoring_response_evaluations=(sum(s['evaluations'] for s in info['starts']) if method=='map' else (2048 if c['channels'] else 0)),
                forward_member_times=b.evaluations-before,job_id=os.environ.get('SLURM_JOB_ID'))
            # Diagnostics commit before latent commit makes resume safe.
            write(dest/'diagnostics.json',finite_json(info));atomic_npz(latent,y=y)
        forecast(problem,y,dest,save_members=c['key']=='multiwell_opt17',point_estimate=method=='map')
        if c['key']=='multiwell_opt17':
            prior=np.random.default_rng(p['prior_seed']).normal(size=(256,4096));conditional=problem.conditional
            for label,value in [('prior',None),('static_prior',conditional)]:
                d=DEST/phase/'references'/e['label']/label;d.mkdir(parents=True,exist_ok=True)
                with (d/'run.lock').open('a') as other:
                    fcntl.flock(other,fcntl.LOCK_EX)
                    if not (d/'complete.json').exists():
                        problem.conditional=value;forecast(problem,prior,d,save_members=True)
                        write(d/'complete.json',dict(status='complete',checkpoint_sha256=EXPECTED_SHA))
            problem.conditional=conditional
        write(dest/'complete.json',dict(status='complete',checkpoint_sha256=EXPECTED_SHA,case_sha256=record['sha256'],
            inference_protocol_sha256=sha(DEST/'inference_protocol.json'),total_task_seconds=time.monotonic()-began))
        print(dict(case=e['label'],config=c['key'],method=method,seconds=time.monotonic()-began),flush=True)
        return True


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('register','pilot','worker'));ap.add_argument('--device',default='cpu')
    ap.add_argument('--max-hours',type=float,default=22.5);ap.add_argument('--method',choices=('all','esmda','map'),default='all');a=ap.parse_args()
    rules=register()
    if a.mode=='register':print(dict(tasks=len(rules['tasks'])));return
    p=read(DEST/'protocol.json');assert read(DEST/'prepared.json')['status']=='complete'
    assert sha(DEST/'protocol.json')==rules['monitoring_protocol_sha256'] and sha(OUT/'calibration.json')==rules['calibration_sha256']
    for file,digest in rules['sources'].items():assert sha(file)==digest,file
    b=Backend('val' if a.mode=='pilot' else 'test',a.device)
    torch.set_num_threads(1 if b.device.type=='cuda' else int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    configs={c['key']:c for c in p['configs']};entries={e['label']:e for e in p['cases']['test']}
    if a.mode=='pilot':
        e=p['cases']['pilot'][0];c=configs['multiwell_opt17'];problem,_=problem_for(b,c,e,p)
        write(DEST/'pilot_gradient.json',gradient_check(problem))
        for method in ('esmda','map'):run_task(b,c,e,method,p,rules,'pilot')
        write(DEST/'pilot_complete.json',dict(status='passed',checkpoint_sha256=EXPECTED_SHA,
            scope='Value/gradient and executable full-budget validation pilot; not test performance or posterior convergence proof'))
        return
    assert read(DEST/'pilot_complete.json')['status']=='passed'
    begin=time.monotonic()
    # Each worker scans one fixed finite task list; locks prevent duplicate work.
    for task in rules['tasks']:
        if a.method!='all' and task['method']!=a.method:continue
        if time.monotonic()-begin>a.max_hours*3600:return
        run_task(b,configs[task['config']],entries[task['case']],task['method'],p,rules)


if __name__=='__main__':main()
