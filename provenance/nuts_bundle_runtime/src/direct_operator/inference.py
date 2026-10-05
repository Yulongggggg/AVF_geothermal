"""Two deployment options sharing the same physical input/output interface."""
import torch
from torch import nn
from temperature_revision.inference import PhysicalInputEncoder, point_readout
from .model import DirectFieldOperator


def load_field(path):
    checkpoint = torch.load(path,map_location='cpu',weights_only=False)
    model = DirectFieldOperator(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'],strict=True)
    return model.eval()


class FieldBundle(nn.Module):
    def __init__(self, paths):
        super().__init__()
        if 'joint' in paths:
            assert set(paths)=={'joint'}
            self.joint = load_field(paths['joint'])
            assert self.joint.kind=='joint'
        else:
            assert set(paths)=={'temperature','pressure'}
            self.joint = None
            self.temperature = load_field(paths['temperature'])
            self.pressure = load_field(paths['pressure'])
            assert self.temperature.kind=='temperature' and self.pressure.kind=='pressure'

    def forward(self, field, scalar):
        if self.joint is not None:
            return self.joint(field,scalar)
        return torch.cat([self.temperature(field,scalar),self.pressure(field,scalar)],1)


class SamplingAdapter(nn.Module):
    """Differentiable interface for existing geological sampling and inversion."""
    def __init__(self, bundle, stats):
        super().__init__()
        self.bundle = bundle
        self.encoder = PhysicalInputEncoder(stats)
        self.register_buffer('scale',torch.tensor(stats['fields_std'])[None,:,None,None])
        self.register_buffer('offset',torch.tensor(stats['fields_mean'])[None,:,None,None])

    def physical_fields(self,x):
        return self.bundle(*self.encoder(x)).float()

    def mean_modes(self,x):
        fields = self.physical_fields(x)
        empty = fields.new_zeros((len(fields),0,*fields.shape[-2:]))
        return (fields-self.offset)/self.scale,empty,empty

    def forward(self,x):
        return self.mean_modes(x)[0]

    def temperature_change_at(self,x,positions_m):
        return point_readout(self.physical_fields(x)[:,0:1],positions_m)[:,0]

    def well_delta_pressure(self,x):
        # Both well outputs are read from the same predicted pressure field.
        field,_ = self.encoder(x)
        r2 = field[:,4].square()
        wx = 100*(r2[:,:,0]-r2[:,:,-1]).mean(1)
        wy = 80*(r2[:,0,:]-r2[:,-1,:]).mean(1)
        pressure = self.physical_fields(x)[:,1:2]
        injector = torch.stack([wx*0,wy*0-250],1)
        producer = torch.stack([wx,wy],1)
        return torch.cat([point_readout(pressure,injector),point_readout(pressure,producer)],1)
