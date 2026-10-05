"""Four-chain final-operator NUTS; preserve all six predeclared reservoirs."""
import argparse,fcntl,time
import numpy as np
import torch
from .common import *
from .inverse_backend import Backend
from .inverse_run import atomic_npz,problem_for,forecast
from inference_comparison.algorithms import nuts
from inference_comparison.run import finite_json,gradient_check

MON=OUT/'monitoring'
DEST=OUT/'hamiltonian'


def finalize(problem,dest,rules):
    parts=[dest/f'chain_{i}' for i in range(4)]
    if not all((p/'complete.json').exists() for p in parts):return
    with (dest/'finalize.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        if (dest/'complete.json').exists():return
        y=np.concatenate([np.load(p/'latent_samples.npz')['y'] for p in parts]);infos=[read(p/'diagnostics.json') for p in parts]
        import arviz as az
        data=az.from_dict(posterior={'y':y})
        rhat=float(az.rhat(data,method='rank').y.max());bulk=float(az.ess(data,method='bulk').y.min());tail=float(az.ess(data,method='tail').y.min())
        divergences=sum(len(v) for info in infos for chain in info['chains'] for v in chain.get('divergences',{}).values())
        info=dict(rank_rhat_max=rhat,bulk_ess_min=bulk,tail_ess_min=tail,divergences=divergences,
            convergence_pass=rhat<=1.05 and bulk>=200 and tail>=200 and divergences==0,
            seconds=sum(i['seconds'] for i in infos),potential_evaluations=sum(i['potential_evaluations'] for i in infos),
            chains=infos,checkpoint_sha256=EXPECTED_SHA)
        atomic_npz(dest/'latent_samples.npz',y=y);write(dest/'diagnostics.json',finite_json(info))
        # Balanced posterior propagation: 64 draws from each complete chain.
        selected=y[:,np.linspace(0,len(y[0])-1,64,dtype=int)].reshape(256,4096)
        forecast(problem,selected,dest,save_members=True)
        write(dest/'complete.json',dict(status='complete',convergence_pass=info['convergence_pass'],checkpoint_sha256=EXPECTED_SHA))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('pilot','worker'));ap.add_argument('--device',default='cpu')
    ap.add_argument('--max-hours',type=float,default=22.5);a=ap.parse_args();DEST.mkdir(exist_ok=True)
    p=read(MON/'protocol.json');assert read(MON/'prepared.json')['status']=='complete'
    b=Backend('val' if a.mode=='pilot' else 'test',a.device);torch.set_num_threads(1)
    c=next(c for c in p['configs'] if c['key']=='multiwell_opt17')
    file=DEST/'protocol.json'
    if not file.exists():
        write(file,dict(checkpoint_sha256=EXPECTED_SHA,monitoring_protocol_sha256=sha(MON/'protocol.json'),
            inference_protocol_sha256=sha(MON/'inference_protocol.json'),cases=p['cases']['test'][:6],
            formal=dict(samples=1024,warmup=512,max_depth=7,target_accept=.9,chains=4),
            pilot=dict(samples=32,warmup=64,max_depth=4,target_accept=.9,chains=2),
            source_sha256=sha(__file__),convergence=dict(rank_rhat_max=1.05,bulk_ess_min=200,tail_ess_min=200,divergences=0),
            retention='All six reservoirs and all four chains retained, including failures of convergence criteria; no truth-based selection'))
    rules=read(file);assert rules['source_sha256']==sha(__file__) and rules['checkpoint_sha256']==EXPECTED_SHA
    if a.mode=='pilot':
        e=p['cases']['pilot'][0];problem,_=problem_for(b,c,e,p);dest=DEST/'pilot';dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():return
        write(dest/'gradient_check.json',gradient_check(problem))
        begin=time.monotonic()
        def progress(r):write(dest/'progress.json',finite_json(r|dict(seconds=time.monotonic()-begin)))
        y,info=nuts(problem,**rules['pilot'],seed=e['observation_seed']+19000,progress=progress)
        assert np.isfinite(y).all();atomic_npz(dest/'latent_samples.npz',y=y);write(dest/'diagnostics.json',finite_json(info))
        write(dest/'complete.json',dict(status='passed',checkpoint_sha256=EXPECTED_SHA,formal_convergence_not_established=True))
        return
    assert read(DEST/'pilot/complete.json')['status']=='passed'
    began=time.monotonic()
    for e in rules['cases']:
        problem,record=problem_for(b,c,e,p);dest=DEST/e['label'];dest.mkdir(exist_ok=True)
        for chain in range(4):
            if time.monotonic()-began>a.max_hours*3600:return
            part=dest/f'chain_{chain}';part.mkdir(exist_ok=True)
            with (part/'run.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if (part/'complete.json').exists():continue
                started=time.monotonic();calls=problem.calls
                def progress(r):write(part/'progress.json',finite_json(r|dict(seconds=time.monotonic()-started)))
                settings=rules['formal']|dict(chains=1)
                y,info=nuts(problem,**settings,seed=e['observation_seed']+19000+chain,progress=progress)
                assert np.isfinite(y).all();atomic_npz(part/'latent_samples.npz',y=y)
                info.update(checkpoint_sha256=EXPECTED_SHA,case_sha256=record['sha256'],calibration_sha256=p['calibration_sha256'],
                    potential_evaluations=problem.calls-calls,hardware=torch.cuda.get_device_name(b.device) if b.device.type=='cuda' else __import__('platform').processor(),
                    device=str(b.device),job_id=os.environ.get('SLURM_JOB_ID'))
                write(part/'diagnostics.json',finite_json(info));write(part/'complete.json',dict(status='complete',checkpoint_sha256=EXPECTED_SHA))
                print('NUTS complete',e['label'],chain,info['seconds'],flush=True)
            finalize(problem,dest,rules)
        finalize(problem,dest,rules)


if __name__=='__main__':main()
