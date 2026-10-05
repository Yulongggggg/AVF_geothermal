"""Validation-only covariance on a fixed spatial block bank, independent of layouts."""
import argparse
import time
import numpy as np
import torch
from .common import *
from .backend import Backend
from inference_comparison.geothermal import masks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=('cpu','cuda'),default='cpu');a=ap.parse_args()
    m=candidate();gate=read(OUT/'qualification/cpu.json');assert gate['status']=='passed' and gate['checkpoint_sha256']==EXPECTED_SHA
    dest=OUT/'calibration';dest.mkdir(exist_ok=True)
    if (OUT/'calibration.json').exists():
        assert read(OUT/'calibration.json')['checkpoint_sha256']==EXPECTED_SHA;return
    entries=read(SOURCE/'protocol.json')['cases']['calibration'];assert len(entries)==120
    b=Backend('val',a.device);torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    points=[(f'cal_{i}_{j}',float(x),float(y)) for i,x in enumerate(np.linspace(-375,375,17)) for j,y in enumerate(np.linspace(-375,375,17))]
    w=masks(points);results=[];began=time.monotonic()
    for i,e in enumerate(entries):
        assert b.meta[e['row']]['split']=='val' and b.meta[e['row']]['real']!='nom'
        file=dest/f"{i:03d}_{e['case_id']}.json"
        if file.exists():
            r=read(file);assert r['checkpoint_sha256']==EXPECTED_SHA and r['entry']==e
        else:
            gi=b.ds.scen_list.index(e['scenario']);row=e['row'];wx,wy=map(float,b.ds.S[row,3:5])
            qw=masks([('producer',wx,wy),('injector',0.,-250.)]);values=[]
            permeability=torch.tensor(np.array(b.real[row],copy=True),device=b.device)
            with torch.no_grad():
                for ti in range(11):
                    pred=b.predict_real(gi,ti,permeability)[0].cpu().numpy()
                    error=pred-np.array(b.fields[row,ti],copy=True)
                    block=np.einsum('chw,phw->cp',error,w);common=float(block[1].mean());block[1]-=common
                    values.append(dict(time_index=ti,block_local_variance=(block**2).mean(1).tolist(),
                        pressure_common_variance=common**2,field_variance=(error[:,16:144]**2).mean((1,2)).tolist(),
                        qoi_variance=(np.einsum('chw,phw->cp',error,qw)**2).tolist()))
            r=dict(entry=e,checkpoint_sha256=EXPECTED_SHA,values=values);write(file,r)
        results.append(r)
        write(dest/'progress.json',dict(completed=i+1,total=len(entries),seconds=time.monotonic()-began))
        print(dict(calibration=i+1,total=120),flush=True)
    keys=('block_local_variance','pressure_common_variance','field_variance','qoi_variance')
    summary={k:np.maximum(np.mean([[v[k] for v in r['values']] for r in results],axis=0),1e-8).tolist() for k in keys}
    summary.update(checkpoint_sha256=EXPECTED_SHA,checkpoint_hashes={m['checkpoint']:EXPECTED_SHA},entries=entries,
        precision='FP32 operator, FP64 likelihood, TF32 disabled',
        observation_blocks=points,block_radius_m=6,
        interpretation='Zero-mean spatially stationary Gaussian residual-second-moment approximation fitted only on120validation realizations at289predeclared reservoir blocks. Pressure spatial common component separated before local variance. No test geometry or response chooses calibration blocks.',
        source_sha256=sha(__file__),seconds=time.monotonic()-began)
    write(OUT/'calibration.json',summary)
    write(OUT/'calibration_complete.json',dict(status='complete',checkpoint_sha256=EXPECTED_SHA,
        calibration_sha256=sha(OUT/'calibration.json'),entries=120,times=11,blocks=289,used_test_data=False))


if __name__=='__main__':main()
