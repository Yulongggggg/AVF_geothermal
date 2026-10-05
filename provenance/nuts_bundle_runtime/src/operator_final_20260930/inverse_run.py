"""Verbatim scientific definitions extracted from pinned original source."""
import os
from pathlib import Path
import numpy as np
import torch
from inference_comparison.metrics import score
from portable_runtime.io import write

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
