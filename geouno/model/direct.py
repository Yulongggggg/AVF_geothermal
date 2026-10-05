"""Shared encoder/decoder backbone and well-neighbourhood features.

Source: ``direct_operator/model.py`` (class ``DirectFieldOperator``). The final GeoUNO
does not use this class directly; :class:`geouno.model.joint.JointTaskOperator` borrows its
conditioning network, encoder, attention block and position embedding, and the static
:meth:`DirectFieldOperator.well_features` supplies scalar features to the coefficient head.
The constructor is kept intact so that module creation order (and therefore seeded random
initialization) matches the original training code.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import CondBlock


class DirectFieldOperator(nn.Module):
    def __init__(self, kind, scales, pressure_residual_scale, width=32):
        super().__init__()
        assert kind in ('joint', 'temperature', 'pressure')
        self.kind = kind
        self.config = dict(kind=kind, scales=list(scales),
                           pressure_residual_scale=float(pressure_residual_scale), width=width)
        self.register_buffer('scales', torch.tensor(scales, dtype=torch.float32))
        self.register_buffer('pressure_residual_scale', torch.tensor(float(pressure_residual_scale)))
        dims = (width, 2 * width, 4 * width, 6 * width)
        self.condition = nn.Sequential(nn.Linear(5, 128), nn.GELU(), nn.Linear(128, 128))
        self.enc = nn.ModuleList([CondBlock(6, dims[0])] + [CondBlock(a, b) for a, b in zip(dims, dims[1:])])
        layer = nn.TransformerEncoderLayer(dims[-1], 6, 2 * dims[-1], dropout=0,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.global_mix = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.position = nn.Conv2d(2, dims[-1], 1)
        self.dec = nn.ModuleList([CondBlock(dims[i + 1] + dims[i], dims[i]) for i in (2, 1, 0)])
        self.field_head = nn.Conv2d(width, 2 if kind == 'joint' else 1, 1)
        if kind != 'temperature':
            # Pool geology near both wells at three scales, without another network.
            self.level_head = nn.Sequential(nn.Linear(2 * dims[-1] + 12 + 5, 256), nn.GELU(),
                                            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))

    @staticmethod
    def well_features(field):
        """Twelve log-permeability summaries around the producer (channel 4) and injector (5).

        For each well and Gaussian radius 6, 24 and 96 m: the weighted arithmetic mean of
        ``m = log10(k) + 13.5`` and the log10 of the weighted harmonic mean of ``k``.
        """
        m = field[:, 0].float()
        features = []
        for channel in (4, 5):
            distance = field[:, channel].float() * 400
            for radius in (6., 24., 96.):
                logw = -.5 * (distance / radius).square()
                logw = logw.flatten(1) - torch.logsumexp(logw.flatten(1), 1, keepdim=True)
                features.append((logw.exp() * m.flatten(1)).sum(1))
                features.append(-torch.logsumexp(logw - m.flatten(1) * math.log(10), 1) / math.log(10))
        return torch.stack(features, 1)

    def forward(self, field, scalar, return_parts=False):
        assert field.shape[1] == 6 and scalar.shape[1] == 5
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
        global_features = torch.cat([h.mean((-2, -1)), h.amax((-2, -1))], 1)
        for block, skip in zip(self.dec, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            h = block(torch.cat([h, skip], 1), condition)
        # Regress physical outputs in FP32, avoiding BF16 pressure quantization.
        with torch.autocast(device_type=field.device.type, enabled=False):
            raw = self.field_head(h.float())
            parts = {}
            if self.kind != 'pressure':
                temperature = raw[:, 0:1] * self.scales[0]
                parts['temperature'] = temperature
            if self.kind != 'temperature':
                features = torch.cat([global_features.float(), self.well_features(field), scalar.float()], 1)
                level = self.level_head(features)[:, :, None, None] * self.scales[1]
                shape_field = raw[:, 1:2] if self.kind == 'joint' else raw
                residual = (shape_field - shape_field.mean((-2, -1), keepdim=True)) * self.pressure_residual_scale
                pressure = level + residual
                parts.update(pressure=pressure, pressure_level=level, pressure_residual=residual)
        output = torch.cat([temperature, pressure], 1) if self.kind == 'joint' else parts[self.kind]
        return (output, parts) if return_parts else output
