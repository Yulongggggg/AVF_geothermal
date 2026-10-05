"""Identical physical inputs and true member identities for architecture benchmarks."""
import csv
import numpy as np
import torch
from temperature_revision.data import ThermalPairs,EpochPairs,FixedPairs
from det_sampler import field_on_grid


class FieldPairs(ThermalPairs):
    def __init__(self,*args,probabilistic=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.probabilistic=probabilistic
        self.scales=torch.tensor(self.stats['fields_std'],dtype=torch.float32)[:,None,None]
        self.mc=np.load(self.pack/'inputs.npy',mmap_mode='r') if probabilistic else None
        self.index=np.load(self.pack/'index.npy',mmap_mode='r') if probabilistic else None
        self.weights=np.array([1/len(self.selected[str(s)]) for s in self.row_scen],dtype=np.float32)
        self.weights/=self.weights.mean()
        meta=list(csv.DictReader((self.pack/'meta.csv').open()))
        self.meta={int(r['row']):r for r in meta}

    def __getitem__(self,item):
        field,scalar,_,idx,ti=super().__getitem__(item)
        row=int(self.rows[idx])
        truth=torch.from_numpy(np.array(self.fields[row,ti],copy=True))/self.scales
        if self.probabilistic:
            mc=field_on_grid(torch.from_numpy(np.array(self.mc[int(self.index[row])],copy=True))[None])[0]
            # Characterization mean, confidence and amplitude replace realized geology.
            field=torch.cat([mc[0:1]+13.5,field[1:],mc[1:2],torch.full_like(mc[1:2],float(self.scal[row,2]))],0)
        return field,scalar,truth,idx,ti


def packed_input(field,scalar):
    return torch.cat([field,scalar[:,:,None,None].expand(-1,-1,*field.shape[-2:])],1)
