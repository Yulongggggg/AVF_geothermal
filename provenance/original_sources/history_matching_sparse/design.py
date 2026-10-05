"""Greedy ensemble linear-Gaussian forecast-information design, validation only."""
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'runs/history_matching_dense_20260910';OUT=ROOT/'runs/history_sparse_20260912'


def reduction(S,H,noise,Y):
    # H: candidate, reading, latent. Correlated pressure errors are augmented
    # latent modes, shared by all locations at each observed time.
    hs=H@S;gram=hs@H.transpose(0,2,1)
    i=np.arange(H.shape[1]);gram[:,i,i]+=noise
    cross=hs@Y.T
    return np.sum(cross*np.linalg.solve(gram,cross),axis=(1,2))


def update(S,H,noise):
    hs=H@S;gram=hs@H.T+np.diag(noise)
    new=S-hs.T@np.linalg.solve(gram,hs)
    return (new+new.T)/2


def main():
    if (OUT/'selection.json').exists():raise FileExistsError('Sparse design already frozen')
    source=OLD/'pilot/hybrid385_Tpk/inverse_z2_dense_hybrid385_Tpk_pilot.npz'
    calpath=OLD/'calibration.json';cal=json.loads(calpath.read_text())
    with np.load(source,allow_pickle=True) as d:records=list(d['results'])
    assert len(records)==4 and not set(r['scen'] for r in records)&set(cal['validation_scenarios'])
    states=[]
    for r in records:
        qT,qP,k=[np.asarray(r[n],dtype=float) for n in ('qT_prior','qP_prior','k_prior')]
        K=qT.shape[1];assert K==64 and len(r['pts'])==385
        # Only prior samples and known geometry are used. No observed or true
        # temperatures, pressure, permeability, or posterior values enter design.
        anomalies=lambda x:(x-x.mean(axis=0,keepdims=True))/np.sqrt(K-1)
        ht=anomalies(qT[:7].transpose(1,0,2)).transpose(2,1,0)
        hp=anomalies(qP[:7].transpose(1,0,2)).transpose(2,1,0)
        hk=anomalies(k).T[:,None,:]
        H=np.zeros((385,15,K+7));H[:,:7,:K]=ht;H[:,7:14,:K]=hp;H[:,14:,:K]=hk
        for t in range(7):H[:,7+t,K+t]=cal['pressure_common_sd'][t]
        noise=np.empty((385,15))
        for ch,q,start,inst in [('T',qT,0,.5),('P',qP,7,.25)]:
            c=cal['calib'][ch]
            noise[:,start:start+7]=(inst**2+np.array(c['alpha'][:7])[:,None]**2+(c['beta']*q[:7].mean(1))**2).T
        noise[:,14]=.1**2
        ip=next(i for i,p in enumerate(r['pts']) if p[0]=='prod');ii=next(i for i,p in enumerate(r['pts']) if p[0]=='inj')
        Y=np.zeros((2,K+7));Y[0,:K]=anomalies(qT[10,:,ip]);Y[1,:K]=anomalies(qP[10,:,ii])
        Y/=np.sqrt(np.sum(Y**2,axis=1,keepdims=True)).clip(1e-8)
        states.append(dict(H=H,noise=noise,Y=Y,S=np.eye(K+7)))
    selected=[0,1];history=[]
    for s in states:
        for j in selected:s['S']=update(s['S'],s['H'][j],s['noise'][j])
    for count in range(3,18):
        gains=np.mean([reduction(s['S'],s['H'],s['noise'],s['Y']) for s in states],axis=0)
        gains[selected]=-np.inf;j=int(np.argmax(gains));selected.append(j)
        for s in states:s['S']=update(s['S'],s['H'][j],s['noise'][j])
        score=float(np.mean([np.trace(s['Y']@s['S']@s['Y'].T)/2 for s in states]))
        assert np.isfinite(score) and score>=-1e-8
        history.append(dict(points=count,added_index=j,normalized_future_variance=score,indices=selected.copy()))
        print(history[-1],flush=True)
    designs={str(n):next(r for r in history if r['points']==n) for n in (5,9,17)}
    best=designs['17']['normalized_future_variance']
    primary=min(n for n in (5,9,17) if designs[str(n)]['normalized_future_variance']<=1.1*best)
    result=dict(status='frozen_before_sparse_formal_results',designs=designs,primary_points=primary,display_key=f'opt{primary}_Tpk',
        showcase='v4m00000',validation_scenarios=[r['scen'] for r in records],criterion='Smallest of 5/9/17 with normalized prior-ensemble future target variance within 10% of the 17-point design.',
        objective='Equal relative variance of ten-year producer-block T and injector-block p; linear-Gaussian ensemble information approximation. No guarantee of nonlinear posterior forecast performance or global geological recovery.',
        fixed_wells=True,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),calibration_sha256=hashlib.sha256(calpath.read_bytes()).hexdigest(),
        design_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        configs=[dict(network=f'opt{n}',channel='Tpk',key=f'opt{n}_Tpk') for n in (5,9,17)]+[dict(network='opt9',channel='Tp',key='opt9_Tp'),dict(network='corridor9',channel='Tpk',key='corridor9_Tpk')],
        disclosure='Design method and budget changed after the earlier 13 formal outcomes and user feedback; new designs use only the four saved validation prior ensembles, not formal test truth or posterior images.')
    (OUT/'selection.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
