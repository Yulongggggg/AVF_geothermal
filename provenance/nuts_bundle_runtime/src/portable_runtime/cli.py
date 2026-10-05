"""Execution plumbing only; original posterior, NUTS and scoring are imported."""
import argparse,datetime,fcntl,importlib.metadata,json,os,platform,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from .io import BUNDLE,read,write,sha,settings

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def verify():
    manifest=read(BUNDLE/'MANIFEST.json')
    for name,record in manifest['files'].items():
        path=BUNDLE/name
        if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:raise RuntimeError(f'Hash/size mismatch: {name}; run git lfs pull and verify checkout')
    return manifest

def hardware():
    gpus=[]
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i)
        gpus.append(dict(device=f'cuda:{i}',name=p.name,total_memory_bytes=p.total_memory,multiprocessors=p.multi_processor_count))
    return dict(utc=now(),architecture=platform.machine(),system=platform.system(),python=sys.version,
        cpu_model=platform.processor(),logical_cpu_count=os.cpu_count(),affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        ram_bytes=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'),torch=torch.__version__,cuda_runtime=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),gpus=gpus,packages={k:importlib.metadata.version(k) for k in settings()['software_versions']})

def check_versions():
    expected=settings()['software_versions']
    actual={k:importlib.metadata.version(k) for k in expected}
    errors={k:dict(expected=v,actual=actual[k]) for k,v in expected.items() if actual[k].split('+')[0]!=v.split('+')[0]}
    if errors:raise RuntimeError(f'Frozen core dependency versions differ: {errors}')
    return actual

def source_origins():
    names=list(settings()['source_provenance'])+['operator_final_20260930.backend','portable_runtime.cli']
    origins={name:str(Path(__import__(name,fromlist=['x']).__file__).resolve()) for name in names}
    for name,path in origins.items():
        if not Path(path).is_relative_to(BUNDLE/'src'):raise RuntimeError(f'Source escaped bundle: {name}: {path}')
    return origins

def backend(split,device):
    from operator_final_20260930.inverse_backend import Backend
    return Backend(split,device)

def problem_for(b,e):
    from operator_final_20260930.inverse_backend import Problem
    c=settings()['observation_config']
    return Problem(b,BUNDLE/e['case_path'],BUNDLE/'data/calibration.json',**{k:c[k] for k in ('channels','static','times','sigma_t','sigma_p')})

def value_gradient(p,y):
    v=p.tensor(y).requires_grad_(True);prediction=p.forward(v[None])[0]
    loss=.5*(v.square().sum()+(prediction-p.data).square().sum())
    gradient,=torch.autograd.grad(loss,v)
    if not torch.isfinite(loss) or not torch.isfinite(gradient).all():raise RuntimeError('Nonfinite potential or input gradient')
    return dict(y=v.detach().cpu().numpy(),prediction=prediction.detach().cpu().numpy(),gradient=gradient.cpu().numpy(),loss=np.array(loss.item()),
        data=p.data.cpu().numpy(),cholesky=p.cholesky.cpu().numpy(),permeability=p.permeability(v.detach()[None]).cpu().numpy())

def smoke(device,output,toy=True):
    verify();check_versions();h=hardware();origins=source_origins()
    e=next(e for e in settings()['cases'] if e['role']=='validation_pilot');b=backend('val',device);p=problem_for(b,e)
    assert p.dim==4096 and p.nobs==238
    with np.load(BUNDLE/settings()['reference']['path']) as z:ref={k:z[k].copy() for k in z.files}
    begin=time.monotonic();got=value_gradient(p,ref['y']);checks={}
    # CPU comparisons use the same frozen binary build and thread count. CUDA
    # allows explicit FP32 cross-device roundoff, recorded alongside each error.
    for key,target in ref.items():
        atol,rtol=(1e-10,1e-10) if b.device.type=='cpu' else ((3e-4,3e-4) if key=='gradient' else (2e-5,2e-5))
        arr=got[key];err=float(np.max(np.abs(arr-target)))
        np.testing.assert_allclose(arr,target,atol=atol,rtol=rtol,err_msg=key)
        checks[key]=dict(max_abs_difference=err,atol=atol,rtol=rtol)
    output.mkdir(parents=True,exist_ok=True)
    from operator_final_20260930.inverse_run import atomic_npz
    atomic_npz(output/'saved_input_result.npz',**got)
    with np.load(output/'saved_input_result.npz') as restored:again=value_gradient(p,restored['y'])
    for key in got:np.testing.assert_array_equal(got[key],again[key],err_msg=f'saved-input replay: {key}')
    toy_result=None
    if toy:
        from inference_comparison.algorithms import nuts
        class Toy:
            dim=4
            data=torch.zeros(4,dtype=torch.float64,device=b.device)
            def tensor(self,x):return torch.as_tensor(x,dtype=torch.float64,device=b.device)
            def forward(self,x):return x
        a,info=nuts(Toy(),samples=8,warmup=8,chains=1,max_depth=2,target_accept=.9,seed=123)
        assert a.shape==(1,8,4) and np.isfinite(a).all()
        class InterruptedToy(Toy):
            calls=0
            def forward(self,x):
                self.calls+=1
                if self.calls==5:raise RuntimeError('deliberate validation interruption')
                return x
        try:nuts(InterruptedToy(),samples=8,warmup=8,chains=1,max_depth=2,target_accept=.9,seed=123)
        except RuntimeError as exc:assert 'deliberate validation interruption' in str(exc)
        else:raise AssertionError('Toy interruption was not reached')
        replay,_=nuts(Toy(),samples=8,warmup=8,chains=1,max_depth=2,target_accept=.9,seed=123)
        np.testing.assert_array_equal(a,replay)
        # Synthetic fixture validates reuse and corruption rejection, never a
        # formal result. No toy marker is placed in the six-reservoir outputs.
        fixture=output/'synthetic_chain_fixture';fixture.mkdir(exist_ok=True)
        e0=entries()[0];atomic_npz(fixture/'latent_samples.npz',y=np.zeros((1,1024,4096)))
        write(fixture/'diagnostics.json',dict(synthetic_fixture=True))
        write(fixture/'complete.json',dict(status='complete',seed=e0['observation_seed']+19000,checkpoint_sha256=settings()['checkpoint_sha256'],protocol_sha256=settings()['protocol_sha256'],case_sha256=e0['case_sha256'],calibration_sha256=settings()['calibration_sha256'],manifest_sha256=sha(BUNDLE/'MANIFEST.json'),artifacts={n:sha(fixture/n) for n in ['latent_samples.npz','diagnostics.json']}))
        assert chain_complete(fixture,e0,0)
        write(fixture/'diagnostics.json',dict(synthetic_fixture=True,deliberately_changed=True))
        try:chain_complete(fixture,e0,0)
        except RuntimeError:pass
        else:raise AssertionError('Corruption gate did not reject synthetic fixture')
        for child in fixture.iterdir():child.unlink()
        fixture.rmdir()
        toy_result=dict(shape=list(a.shape),seconds=info['seconds'],scientific_result=False,interrupted_restart_from_seed='bitwise identical to uninterrupted toy chain',completed_chain_reuse_and_corruption_gate='passed synthetic fixture',transition_state_resume=False)
    receipt=dict(status='passed',device=str(b.device),utc=now(),seconds=time.monotonic()-begin,
        checkpoint_sha256=settings()['checkpoint_sha256'],manifest_sha256=sha(BUNDLE/'MANIFEST.json'),
        reference_sha256=settings()['reference']['sha256'],comparisons=checks,input_artifact_replay='bitwise identical; not sampler-state/transition resume',
        toy_nuts_installation_check=toy_result,source_origins=origins,hardware=h)
    if b.device.type=='cuda':receipt['max_cuda_memory_allocated_bytes']=torch.cuda.max_memory_allocated(b.device)
    write(output/'complete.json',receipt)
    print(json.dumps(receipt,indent=2));return receipt

def chain_complete(part,e,chain):
    marker=part/'complete.json'
    if not marker.exists():return False
    r=read(marker)
    if r.get('status')!='complete' or r.get('checkpoint_sha256')!=settings()['checkpoint_sha256'] or r.get('seed')!=e['observation_seed']+19000+chain or r.get('protocol_sha256')!=settings()['protocol_sha256'] or r.get('case_sha256')!=e['case_sha256'] or r.get('calibration_sha256')!=settings()['calibration_sha256'] or r.get('manifest_sha256')!=sha(BUNDLE/'MANIFEST.json') or set(r.get('artifacts',{}))!={'latent_samples.npz','diagnostics.json'}:
        raise RuntimeError(f'Existing chain marker has incompatible provenance: {marker}')
    for name,digest in r['artifacts'].items():
        if sha(part/name)!=digest:raise RuntimeError(f'Existing chain artifact corrupt: {part/name}')
    with np.load(part/'latent_samples.npz') as z:
        if z['y'].shape!=(1,1024,4096) or not np.isfinite(z['y']).all():raise RuntimeError(f'Invalid completed chain: {part}')
    return True

def merged_complete(dest,e):
    marker=dest/'complete.json'
    if not marker.exists():return False
    r=read(marker);rules=settings()
    expected=dict(status='complete',checkpoint_sha256=rules['checkpoint_sha256'],protocol_sha256=rules['protocol_sha256'],case_sha256=e['case_sha256'],calibration_sha256=rules['calibration_sha256'],manifest_sha256=sha(BUNDLE/'MANIFEST.json'))
    if any(r.get(k)!=v for k,v in expected.items()):raise RuntimeError(f'Existing merged marker has incompatible provenance: {marker}')
    hashes={f'chain_{i}/complete.json':sha(dest/f'chain_{i}/complete.json') for i in range(4)}
    if r.get('chain_markers')!=hashes:raise RuntimeError(f'Existing merged result is not bound to current four chains: {marker}')
    if set(r.get('artifacts',{}))!={'latent_samples.npz','diagnostics.json','forecast.npz','scores.json'}:raise RuntimeError(f'Incomplete merged artifact manifest: {marker}')
    for name,digest in r['artifacts'].items():
        if sha(dest/name)!=digest:raise RuntimeError(f'Invalid merged artifact: {dest/name}')
    return True

def entries(case=None):
    cases=[e for e in settings()['cases'] if e['role']=='formal']
    if case is not None:
        cases=[e for e in cases if e['label']==case]
        if not cases:raise ValueError('Unknown formal case')
    return cases

def worker(a):
    verify();check_versions();source_origins()
    from inference_comparison.algorithms import nuts
    from inference_comparison.run import finite_json
    from operator_final_20260930.inverse_run import atomic_npz
    b=backend('test',a.device);started=time.monotonic();rules=settings()
    for e in entries(a.case):
        p=None
        for chain in ([a.chain] if a.chain is not None else range(4)):
            if time.monotonic()-started>a.max_hours*3600:return
            part=a.output/e['label']/f'chain_{chain}';part.mkdir(parents=True,exist_ok=True)
            with (part/'run.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if chain_complete(part,e,chain):continue
                if p is None:p=problem_for(b,e)
                seed=e['observation_seed']+19000+chain;attempt=now()
                # Interrupted runs are restarted from their original fixed seed.
                write(part/'attempt.json',dict(status='running',utc=attempt,seed=seed,device=a.device,pid=os.getpid(),resume_policy='complete-chain reuse; interrupted chain restarts from fixed seed'))
                calls=p.calls;begin=time.monotonic()
                def progress(r):write(part/'progress.json',finite_json(r|dict(seconds=time.monotonic()-begin,case=e['label'],chain_index=chain)))
                try:
                    y,info=nuts(p,**(rules['formal']|dict(chains=1)),seed=seed,progress=progress)
                    assert y.shape==(1,1024,4096) and np.isfinite(y).all()
                    atomic_npz(part/'latent_samples.npz',y=y)
                    info.update(checkpoint_sha256=rules['checkpoint_sha256'],protocol_sha256=rules['protocol_sha256'],case_sha256=e['case_sha256'],
                        calibration_sha256=rules['calibration_sha256'],seed=seed,potential_evaluations=p.calls-calls,device=a.device,hardware=hardware(),source_origins=source_origins())
                    write(part/'diagnostics.json',finite_json(info))
                    write(part/'complete.json',dict(status='complete',utc=now(),seed=seed,case_sha256=e['case_sha256'],calibration_sha256=rules['calibration_sha256'],manifest_sha256=sha(BUNDLE/'MANIFEST.json'),checkpoint_sha256=rules['checkpoint_sha256'],protocol_sha256=rules['protocol_sha256'],
                        artifacts={n:sha(part/n) for n in ['latent_samples.npz','diagnostics.json']}))
                    print('NUTS chain complete',e['label'],chain,info['seconds'],flush=True)
                except BaseException:
                    write(part/'last_failure.json',dict(utc=now(),seed=seed,traceback=traceback.format_exc()));raise

def merge(output,device):
    verify();check_versions()
    from operator_final_20260930.inverse_run import atomic_npz,forecast
    from inference_comparison.run import finite_json
    import arviz as az
    rules=settings();b=None;rows=[]
    for e in entries():
        dest=output/e['label'];dest.mkdir(parents=True,exist_ok=True)
        parts=[dest/f'chain_{i}' for i in range(4)]
        complete=[chain_complete(p,e,i) for i,p in enumerate(parts)]
        if not all(complete):rows.append(dict(case=e['label'],status='incomplete',completed_chains=sum(complete)));continue
        with (dest/'finalize.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            marker=dest/'complete.json'
            if not merged_complete(dest,e):
                ys=[]
                for part in parts:
                    with np.load(part/'latent_samples.npz') as z:ys.append(z['y'].copy())
                y=np.concatenate(ys);infos=[read(p/'diagnostics.json') for p in parts]
                data=az.from_dict(posterior={'y':y})
                diagnostic_arrays={name:np.asarray(value.y) for name,value in [('rhat',az.rhat(data,method='rank')),('bulk',az.ess(data,method='bulk')),('tail',az.ess(data,method='tail'))]}
                nonfinite={k:int((~np.isfinite(v)).sum()) for k,v in diagnostic_arrays.items()}
                rhat=float(np.max(diagnostic_arrays['rhat']));bulk=float(np.min(diagnostic_arrays['bulk']));tail=float(np.min(diagnostic_arrays['tail']))
                divergences=sum(len(v) for info in infos for chain in info['chains'] for v in chain.get('divergences',{}).values())
                good=bool(np.isfinite([rhat,bulk,tail]).all() and rhat<=1.05 and bulk>=200 and tail>=200 and divergences==0)
                info=dict(rank_rhat_max=rhat,bulk_ess_min=bulk,tail_ess_min=tail,divergences=divergences,convergence_pass=good,nonfinite_diagnostic_coordinates=nonfinite,
                    seconds=sum(i['seconds'] for i in infos),potential_evaluations=sum(i['potential_evaluations'] for i in infos),
                    timing_interpretation='Sum of chain computation seconds; not parallel elapsed time',chains=infos,checkpoint_sha256=rules['checkpoint_sha256'])
                atomic_npz(dest/'latent_samples.npz',y=y);write(dest/'diagnostics.json',finite_json(info))
                selected=y[:,np.linspace(0,len(y[0])-1,64,dtype=int)].reshape(256,4096)
                if b is None:b=backend('test',device)
                forecast(problem_for(b,e),selected,dest,save_members=True)
                write(marker,dict(status='complete',convergence_pass=good,checkpoint_sha256=rules['checkpoint_sha256'],protocol_sha256=rules['protocol_sha256'],case_sha256=e['case_sha256'],calibration_sha256=rules['calibration_sha256'],manifest_sha256=sha(BUNDLE/'MANIFEST.json'),chain_markers={f'chain_{i}/complete.json':sha(parts[i]/'complete.json') for i in range(4)},
                    selected_draw_indices=np.linspace(0,1023,64,dtype=int).tolist(),artifacts={n:sha(dest/n) for n in ['latent_samples.npz','diagnostics.json','forecast.npz','scores.json']}))
            rows.append(dict(case=e['label'],status='complete',completed_chains=4,diagnostics=read(dest/'diagnostics.json'),scores=read(dest/'scores.json')))
    full=all(r['status']=='complete' for r in rows)
    report=dict(status='complete' if full else 'incomplete',cases=rows,expected_cases=6,expected_chains=24,
        completed_chains=sum(r['completed_chains'] for r in rows),retention='All six predeclared reservoirs retained including failed convergence; no selected-case mean',
        checkpoint_sha256=rules['checkpoint_sha256'],protocol_sha256=rules['protocol_sha256'],utc=now())
    if full:
        def mean_tree(values):
            if isinstance(values[0],dict):return {k:mean_tree([v[k] for v in values]) for k in values[0]}
            return np.mean(np.asarray(values),axis=0).tolist()
        report['equal_reservoir_mean_scores']=mean_tree([r['scores'] for r in rows])
        report['convergence_pass_cases']=sum(r['diagnostics']['convergence_pass'] for r in rows)
        report['scientific_status']='all_converged' if report['convergence_pass_cases']==6 else 'complete_but_convergence_failures_retained'
    write(output/'report.json',finite_json(report));print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
    return report

def launch(a):
    verify();check_versions()
    devices=a.devices.split(',') if a.devices else [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    if not devices:raise RuntimeError('No CUDA device found; explicitly choose --devices cpu for slow CPU execution')
    if len(set(devices))!=len(devices):raise ValueError('Repeated device; use --workers deliberately to share a device')
    workers=a.workers or len(devices)
    if workers<1:raise ValueError('--workers must be positive')
    for device in set(devices):
        if device.startswith('cuda:') and int(device.split(':')[1])>=torch.cuda.device_count():raise ValueError(f'Unavailable {device}')
        receipt=a.output/'smoke'/device.replace(':','_')/'complete.json'
        if not receipt.exists():raise RuntimeError(f'Run smoke --device {device} --output {a.output} first')
        s=read(receipt)
        if s.get('status')!='passed' or s.get('manifest_sha256')!=sha(BUNDLE/'MANIFEST.json'):raise RuntimeError(f'Stale or failed smoke: {receipt}')
    a.output.mkdir(parents=True,exist_ok=True);processes=[];begin=time.monotonic()
    write(a.output/'launch.json',dict(utc=now(),workers=workers,devices=devices,hardware=hardware(),max_hours_between_chain_checks=a.max_hours))
    for i in range(workers):
        device=devices[i%len(devices)]
        cmd=[sys.executable,str(BUNDLE/'run.py'),'_worker','--device',device,'--output',str(a.output),'--max-hours',str(a.max_hours)]
        if a.case:cmd+=['--case',a.case]
        if a.chain is not None:cmd+=['--chain',str(a.chain)]
        logfile=(a.output/f'worker_{i}.log').open('a',buffering=1)
        processes.append((subprocess.Popen(cmd,stdout=logfile,stderr=subprocess.STDOUT),logfile))
    try:codes=[p.wait() for p,_ in processes]
    except KeyboardInterrupt:
        for p,_ in processes:p.terminate()
        for p,_ in processes:p.wait()
        raise
    finally:
        for _,f in processes:f.close()
    worker_elapsed=time.monotonic()-begin
    merge(a.output,devices[0])
    write(a.output/'launch_finished.json',dict(utc=now(),elapsed_seconds=time.monotonic()-begin,worker_elapsed_seconds=worker_elapsed,timing_interpretation='Launcher wall time includes worker subprocesses and final merge; worker_elapsed_seconds excludes final merge',worker_exit_codes=codes))
    if any(codes):raise SystemExit('One or more workers failed; read worker logs. Completed chains remain reusable.')

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    for name in ['preflight','smoke','run','merge','_worker']:
        p=sub.add_parser(name);p.add_argument('--output',type=Path,default=BUNDLE/'outputs')
        if name in ('preflight','smoke','merge','_worker'):p.add_argument('--device',default='cpu')
        if name in ('run','_worker'):
            p.add_argument('--case',choices=[f'test_{i:02}' for i in range(1,7)]);p.add_argument('--chain',type=int,choices=range(4));p.add_argument('--max-hours',type=float,default=1e6)
        if name=='run':p.add_argument('--devices');p.add_argument('--workers',type=int)
    a=ap.parse_args();a.output=a.output.resolve()
    if a.command=='preflight':
        verify();check_versions();r=hardware()|dict(status='passed',source_origins=source_origins(),checkpoint_sha256=settings()['checkpoint_sha256']);write(a.output/'preflight.json',r);print(json.dumps(r,indent=2))
    elif a.command=='smoke':smoke(a.device,a.output/'smoke'/a.device.replace(':','_'))
    elif a.command=='_worker':worker(a)
    elif a.command=='merge':merge(a.output,a.device)
    else:launch(a)
