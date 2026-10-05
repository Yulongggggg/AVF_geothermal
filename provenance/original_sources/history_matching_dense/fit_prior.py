"""Estimate stationary geological covariance from training fields only.

The prior mean stays the previous independently generated smooth random field.
Its independent covariance is added to the training-field covariance because
the inverse residual is true geology minus that random prior mean.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=__import__("geouno_paths").ROOT;OUT=ROOT/'runs/history_matching_dense_20260910'


def main():
    import argparse
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, default=__import__('geouno_paths').PACK)
    parser.add_argument('--original-factor', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    OUT = args.out
    torch.set_num_threads(4);pack=args.pack;OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'prior_factor.npy').exists():raise FileExistsError('Prior already fitted')
    meta=list(csv.DictReader((pack/'meta.csv').open()))
    # One actual member per scenario gives scenarios equal weight; hashing is
    # independent of values and excludes every validation/test row explicitly.
    by={}
    for row in meta:
        if row['split']=='train' and row['real']!='nom':by.setdefault(row['scen'],[]).append(row)
    names=sorted(by,key=lambda s:hashlib.sha256(('prior20260910:'+s).encode()).digest())[:1024]
    selected=[sorted(by[n],key=lambda r:r['real'])[0] for n in names]
    real=np.load(pack/'real.npy',mmap_mode='r');stats=json.loads((pack/'stats.json').read_text());mean=float(stats['M_mean'])
    B=F.interpolate(torch.eye(64,dtype=torch.float64).reshape(64,1,64),size=128,mode='linear',align_corners=True)[:,0].T
    pinv=torch.linalg.pinv(B);power=torch.zeros(64,64,dtype=torch.float64);residual=[]
    for first in range(0,len(selected),32):
        rows=[int(r['row']) for r in selected[first:first+32]]
        fields=torch.tensor(np.array(real[rows],copy=True),dtype=torch.float64)-mean
        grid=pinv@fields@pinv.T;power+=torch.fft.fft2(grid).abs().square().sum(0)
        residual.extend((fields-B@grid@B.T).square().mean((-2,-1)).sqrt().tolist())
    power/=len(selected)
    kernel=torch.fft.ifft2(power).real/4096
    # Circular stationarity is an explicit modelling assumption. This estimate
    # stores averaged second moments, not any individual geological map.
    yy,xx=torch.meshgrid(torch.arange(64),torch.arange(64),indexing='ij');coords=torch.stack([yy.flatten(),xx.flatten()],1)
    dy=(coords[:,None,0]-coords[None,:,0])%64;dx=(coords[:,None,1]-coords[None,:,1])%64
    covariance=kernel[dy,dx];del dy,dx
    original=args.original_factor;L=torch.from_numpy(np.load(original)).double()
    covariance+=.25*(L@L.T)
    jitter=1e-6;covariance.diagonal().add_(jitter)
    factor=torch.linalg.cholesky(covariance).float()
    relative=float(torch.linalg.vector_norm(factor.double()@factor.double().T-covariance)/torch.linalg.vector_norm(covariance))
    if relative>1e-5:raise ValueError('Covariance factorization inaccurate')
    np.save(OUT/'prior_factor.npy',factor.numpy());np.save(OUT/'training_power.npy',power.numpy())
    report=dict(training_scenarios=len(names),selected_rows=[int(r['row']) for r in selected],selected_cases=[r['case_id'] for r in selected],
        prior_mean='Unchanged: training-wide mean plus the independent original 0.5-dex smooth Gaussian field, seed777+gi',
        covariance='Training-only stationary 64-grid second moment about training global mean plus independent random-M0 covariance (0.25 original L L^T), jitter1e-6.',
        stationarity='Periodic covariance on the 64-grid; fitted PSD is averaged across training scenarios. No validation/test geology fitted or stored.',
        marginal_sd=float(covariance.diagonal().mean().sqrt()),factor_relative_error=relative,
        median_training_projection_error=float(np.median(residual)),original_factor_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        factor_sha256=hashlib.sha256((OUT/'prior_factor.npy').read_bytes()).hexdigest(),meta_sha256=hashlib.sha256((pack/'meta.csv').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'prior_fit.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if not k.startswith('selected')})


if __name__=='__main__':main()
