"""One shared encoder with task-specific decoders and a spatial pressure readout."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from direct_operator.model import DirectFieldOperator
from direct_operator.inference import load_field


class JointTaskOperator(nn.Module):
    def __init__(self, scales, pressure_residual_scale, width=32):
        super().__init__()
        self.config=dict(architecture='task_decoders',scales=list(scales),
                         pressure_residual_scale=float(pressure_residual_scale),width=width)
        self.kind='joint'
        base=DirectFieldOperator('joint',scales,pressure_residual_scale,width)
        self.register_buffer('scales',base.scales.clone())
        self.register_buffer('pressure_residual_scale',base.pressure_residual_scale.clone())
        for name in ('condition','enc','global_mix','position'):setattr(self,name,getattr(base,name))
        self.temperature_decoder=copy.deepcopy(base.dec)
        self.pressure_decoder=base.dec
        self.temperature_head=nn.Conv2d(width,1,1)
        self.pressure_head=nn.Conv2d(width,1,1)
        self.pressure_grid=nn.Sequential(nn.Conv2d(6*width,16,1),nn.GELU(),nn.AdaptiveAvgPool2d((4,4)))
        self.level_head=nn.Sequential(nn.Linear(12*width+12+5+256,256),nn.GELU(),
                                      nn.Linear(256,128),nn.GELU(),nn.Linear(128,1))

    def initialize(self, thermal_path, pressure_path, start):
        """Copy training weights once; no external predictor remains in the model."""
        thermal=load_field(thermal_path)
        pressure=load_field(pressure_path)
        shared=thermal if start=='temperature' else pressure
        for name in ('condition','enc','global_mix','position'):
            getattr(self,name).load_state_dict(getattr(shared,name).state_dict())
        source=thermal if start=='temperature' else pressure
        self.temperature_decoder.load_state_dict(source.dec.state_dict())
        self.pressure_decoder.load_state_dict(pressure.dec.state_dict())
        with torch.no_grad():
            self.temperature_head.weight.copy_(source.field_head.weight[:1])
            self.temperature_head.bias.copy_(source.field_head.bias[:1])
            self.pressure_head.weight.copy_(pressure.field_head.weight[1:2])
            self.pressure_head.bias.copy_(pressure.field_head.bias[1:2])
            self.level_head[0].weight.zero_()
            size=pressure.level_head[0].in_features
            self.level_head[0].weight[:,:size].copy_(pressure.level_head[0].weight)
            self.level_head[0].bias.copy_(pressure.level_head[0].bias)
            for i in (2,4):self.level_head[i].load_state_dict(pressure.level_head[i].state_dict())
        return self

    @staticmethod
    def decode(h,skips,condition,blocks):
        for block,skip in zip(blocks,reversed(skips[:-1])):
            h=F.interpolate(h,size=skip.shape[-2:],mode='bilinear',align_corners=True)
            h=block(torch.cat([h,skip],1),condition)
        return h

    def forward(self,field,scalar,return_parts=False):
        condition=self.condition(scalar);h=field;skips=[]
        for i,block in enumerate(self.enc):
            if i:h=F.avg_pool2d(h,2)
            h=block(h,condition);skips.append(h)
        pos=F.interpolate(field[:,2:4],size=h.shape[-2:],mode='bilinear',align_corners=True)
        shape=h.shape
        h=self.global_mix((h+self.position(pos)).flatten(2).transpose(1,2)).transpose(1,2).reshape(shape)
        features=torch.cat([h.mean((-2,-1)),h.amax((-2,-1))],1)
        grid=self.pressure_grid(h).flatten(1)
        ht=self.decode(h,skips,condition,self.temperature_decoder)
        hp=self.decode(h,skips,condition,self.pressure_decoder)
        with torch.autocast(device_type=field.device.type,enabled=False):
            temperature=self.temperature_head(ht.float())*self.scales[0]
            raw=self.pressure_head(hp.float())
            residual=(raw-raw.mean((-2,-1),keepdim=True))*self.pressure_residual_scale
            features=torch.cat([features.float(),DirectFieldOperator.well_features(field),scalar.float(),grid.float()],1)
            level=self.level_head(features)[:,:,None,None]*self.scales[1]
            pressure=level+residual
            result=torch.cat([temperature,pressure],1)
        if return_parts:return result,dict(temperature=temperature,pressure=pressure,pressure_level=level,pressure_residual=residual)
        return result


def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    config=ck['model_config'].copy()
    assert config.pop('architecture')=='task_decoders'
    model=JointTaskOperator(**config)
    model.load_state_dict(ck['model'],strict=True)
    return model.eval()
