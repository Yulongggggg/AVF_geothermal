"""One shared encoder with separate temperature and pressure decoders.

Source: ``joint_architecture/model.py`` (class ``JointTaskOperator``). The historical
``initialize`` helper, which assembled this model from two separately trained
single-field predictors earlier in the training lineage, is not needed to use or
fine-tune the released checkpoints and is omitted.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .direct import DirectFieldOperator


class JointTaskOperator(nn.Module):
    def __init__(self, scales, pressure_residual_scale, width=32):
        super().__init__()
        self.config = dict(architecture='task_decoders', scales=list(scales),
                           pressure_residual_scale=float(pressure_residual_scale), width=width)
        self.kind = 'joint'
        base = DirectFieldOperator('joint', scales, pressure_residual_scale, width)
        self.register_buffer('scales', base.scales.clone())
        self.register_buffer('pressure_residual_scale', base.pressure_residual_scale.clone())
        for name in ('condition', 'enc', 'global_mix', 'position'):
            setattr(self, name, getattr(base, name))
        self.temperature_decoder = copy.deepcopy(base.dec)
        self.pressure_decoder = base.dec
        self.temperature_head = nn.Conv2d(width, 1, 1)
        self.pressure_head = nn.Conv2d(width, 1, 1)
        self.pressure_grid = nn.Sequential(nn.Conv2d(6 * width, 16, 1), nn.GELU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.level_head = nn.Sequential(nn.Linear(12 * width + 12 + 5 + 256, 256), nn.GELU(),
                                        nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))

    @staticmethod
    def decode(h, skips, condition, blocks):
        for block, skip in zip(blocks, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            h = block(torch.cat([h, skip], 1), condition)
        return h

    def forward(self, field, scalar, return_parts=False):
        condition = self.condition(scalar)
        h = field
        skips = []
        for i, block in enumerate(self.enc):
            if i:
                h = F.avg_pool2d(h, 2)
            h = block(h, condition)
            skips.append(h)
        pos = F.interpolate(field[:, 2:4], size=h.shape[-2:], mode='bilinear', align_corners=True)
        shape = h.shape
        h = self.global_mix((h + self.position(pos)).flatten(2).transpose(1, 2)).transpose(1, 2).reshape(shape)
        features = torch.cat([h.mean((-2, -1)), h.amax((-2, -1))], 1)
        grid = self.pressure_grid(h).flatten(1)
        ht = self.decode(h, skips, condition, self.temperature_decoder)
        hp = self.decode(h, skips, condition, self.pressure_decoder)
        with torch.autocast(device_type=field.device.type, enabled=False):
            temperature = self.temperature_head(ht.float()) * self.scales[0]
            raw = self.pressure_head(hp.float())
            residual = (raw - raw.mean((-2, -1), keepdim=True)) * self.pressure_residual_scale
            features = torch.cat([features.float(), DirectFieldOperator.well_features(field),
                                  scalar.float(), grid.float()], 1)
            level = self.level_head(features)[:, :, None, None] * self.scales[1]
            pressure = level + residual
            result = torch.cat([temperature, pressure], 1)
        if return_parts:
            return result, dict(temperature=temperature, pressure=pressure,
                                pressure_level=level, pressure_residual=residual)
        return result
