"""Opt-in temperature replacement for the existing realization sampler.

The input is a physical realization in legacy normalized channels. Keep C and
S_max in the upstream RealSampler so they still control the ensemble law.
"""
import numpy as np
import torch
from torch import nn

from .model import ThermalFieldNet


def point_readout(fields, positions_m):
    """Bilinear point values on x=[-400,400], y=[-500,500], preserving gradients."""
    if positions_m.shape != (len(fields),2):
        raise ValueError("positions_m must have shape (batch,2), in x/y metres")
    grid=positions_m.to(device=fields.device,dtype=fields.dtype)/fields.new_tensor([400.,500.])
    if (grid.abs()>1).any(): raise ValueError("Point lies outside the output grid")
    return torch.nn.functional.grid_sample(fields,grid[:,None,None,:],mode="bilinear",
                                           align_corners=True)[:,:,0,0]


class PhysicalInputEncoder(nn.Module):
    def __init__(self, stats):
        super().__init__()
        self.stats = stats
        coeff = np.polyfit(stats["tau_norm"], np.log10(stats["times_s"]), 1)
        if np.max(np.abs(np.polyval(coeff, stats["tau_norm"])-np.log10(stats["times_s"]))) > 1e-6:
            raise ValueError("Legacy tau must be an affine normalization of log time")
        self.register_buffer("tau_to_log10", torch.tensor(coeff, dtype=torch.float32))

    def forward(self, x):
        if x.shape[1] != 13:
            raise ValueError("This revision requires the 13-channel depth-aware input")
        st = self.stats
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(torch.linspace(-500,500,h,device=x.device),
                                torch.linspace(-400,400,w,device=x.device), indexing="ij")
        coords = torch.stack([(yy.abs() <= 400).float(), xx/400, yy/500])[None].expand(b,-1,-1,-1)
        m = x[:,0:1].float()*st["M_std"]+st["M_mean"]
        field = torch.cat([m+13.5, coords, x[:,5:7].float()], dim=1)
        scalar = x[:,:,0,0].float()
        q = scalar[:,7]*st["scal_std"][0]+st["scal_mean"][0]
        tinj = scalar[:,8]*st["scal_std"][1]+st["scal_mean"][1]
        pp = scalar[:,10]*st["pp_norm"][1]+st["pp_norm"][0]
        depth = scalar[:,12]*st["depth_norm"][1]+st["depth_norm"][0]
        logt = scalar[:,11]*self.tau_to_log10[0]+self.tau_to_log10[1]
        cond = torch.stack([q/3.5, (tinj-303)/10, (pp-20)/10, (logt-6)/2, (depth-3.06)/.6], dim=1)
        return field, cond


class ThermalReplacement(nn.Module):
    """Frozen legacy pressure plus the independent deterministic temperature.

Only mean_modes is exposed: uncertainty must be propagated by drawing physical
permeability inputs, as in det_sampler.RealSampler.draw, not internal modes.
Freezing pressure parameters preserves differentiation with respect to inputs.
"""
    def __init__(self, pressure, thermal, stats, pressure_drop=(2,3,4), full_domain=False):
        super().__init__()
        self.pressure = pressure.requires_grad_(False).eval()
        self.thermal = thermal
        self.encoder = PhysicalInputEncoder(stats)
        self.pressure_drop = tuple(pressure_drop)
        self.full_domain = bool(full_domain)
        self.temp_offset = float(stats["fields_mean"][0]/stats["fields_std"][0])
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        self.pressure.eval()
        return self

    def mean_modes(self, x):
        field, cond = self.encoder(x)
        xp = x.clone()
        if self.pressure_drop:
            xp[:, self.pressure_drop] = 0
        mu, mt, mp = self.pressure.mean_modes(xp)
        temp = self.thermal(field, cond).float()-self.temp_offset
        if not self.full_domain:
            # A reservoir-only checkpoint has no trained cap/bed outputs.
            # Retain the already-trained legacy prediction there.
            temp=torch.where(field[:,1:2].bool(),temp,mu[:,0:1])
        return torch.cat([temp, mu[:,1:2]], dim=1), torch.zeros_like(mt), mp

    def forward(self, x):
        return self.mean_modes(x)[0]

    def temperature_change_at(self,x,positions_m):
        if self.full_domain:
            field,cond=self.encoder(x)
            dt=self.thermal(field,cond).float()*self.encoder.stats["fields_std"][0]
        else:
            dt=self(x)[:,0:1]*self.encoder.stats["fields_std"][0]+self.encoder.stats["fields_mean"][0]
        return point_readout(dt,positions_m)[:,0]


def load_replacement(thermal_path, pressure_path, ds, device="cuda"):
    from .audit import load_legacy
    ck = torch.load(thermal_path, map_location="cpu", weights_only=False)
    if not np.isclose(ck["temperature_scale"], ds.st["fields_std"][0], rtol=1e-7):
        raise ValueError("Temperature checkpoint and evaluation normalization differ")
    temp = ThermalFieldNet(**ck["model_config"])
    temp.load_state_dict(ck["model"], strict=True)
    pressure, conf = load_legacy(pressure_path, ds, device)
    drop = tuple(int(v) for v in conf.get("drop_ch", "").split(",") if v)
    return ThermalReplacement(pressure, temp, ds.st, drop,
                              full_domain=ck.get("args",{}).get("outside_weight",0)>0).to(device).eval()
