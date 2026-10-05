"""Geology summaries and the regional pressure-level readout.

Source: ``encoder_study/model.py`` (``geology_maps``, ``GeologyGate``, ``RegionalReadout``,
``GeologyJointOperator``, ``load``). The selected GeoUNO extends
``GeologyJointOperator(regional=True, gates=False)``; the transfer parent (checkpoint
``cf278e64...``) is an instance of that class. Input ``m`` is ``log10(k) + 13.5``.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .direct import DirectFieldOperator
from .joint import JointTaskOperator


def geology_maps(field):
    """Six inputs plus local log-k moments at two physical neighbourhood scales.

    5/17 cells span approximately 31/107 m. Replicate padding avoids introducing
    zero-permeability values at domain edges. Arithmetic/harmonic quantities are
    log10 means of linear k, shifted by 13.5, computed with stable exponent shifts.
    """
    m = field[:, :1].float()
    hi, lo = m.amax((-2, -1), keepdim=True), m.amin((-2, -1), keepdim=True)
    positive, inverse = ((m - hi) * math.log(10)).exp(), ((lo - m) * math.log(10)).exp()
    result = [field.float()]
    for size in (5, 17):
        def pool(x):
            return F.avg_pool2d(F.pad(x, [size // 2] * 4, mode='replicate'), size, stride=1)
        mean = pool(m)
        spread = (pool(m.square()) - mean.square()).clamp_min(0)
        arithmetic = hi + pool(positive).clamp_min(1e-30).log() / math.log(10)
        harmonic = lo - pool(inverse).clamp_min(1e-30).log() / math.log(10)
        result += [mean, m - mean, spread, arithmetic, harmonic]
    return torch.cat(result, 1)


class GeologyGate(nn.Module):
    """Optional geology-conditioned feature modulation (disabled in the selected model)."""

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(18, 16, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(16, 2 * channels, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, geology):
        g = F.adaptive_avg_pool2d(geology, h.shape[-2:])
        g = torch.cat([g, h.mean(1, keepdim=True), h.amax(1, keepdim=True)], 1)
        gain, shift = self.net(g).chunk(2, 1)
        return h * (1 + gain.tanh()) + shift


class RegionalReadout(nn.Module):
    """Learned reservoir/near-well pooling without changing well coordinates.

    Seven softmax pooling maps (one free map plus Gaussian priors of radius 6, 24 and
    96 m around each well) are restricted to the reservoir mask and applied to 16
    geology feature channels, giving 7 x 16 features for the pressure-level head.
    """
    output_size = 7 * 16

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.GELU())
        self.salience = nn.Conv2d(16, 7, 1)
        nn.init.zeros_(self.salience.weight)
        nn.init.zeros_(self.salience.bias)

    def forward(self, maps, field):
        h = self.features(maps)
        log_priors = [torch.zeros_like(field[:, :1])]
        for channel in (4, 5):
            for radius in (6., 24., 96.):
                log_priors.append(-.5 * (field[:, channel:channel + 1] * 400 / radius).square())
        logits = self.salience(h).float() + torch.cat(log_priors, 1).float()
        logits = logits.masked_fill(field[:, 1:2] < .5, -1e9)
        weights = logits.flatten(2).softmax(-1)
        return torch.einsum('brn,bcn->brc', weights, h.float().flatten(2)).flatten(1)


class GeologyJointOperator(JointTaskOperator):
    def __init__(self, scales, pressure_residual_scale, width=32, regional=False, gates=True):
        super().__init__(scales, pressure_residual_scale, width)
        self.config.update(architecture='geology_task_decoders', regional=regional, gates=gates)
        self.geology_gates = nn.ModuleList([GeologyGate(c * width) for c in (1, 2, 4, 6)]) if gates else None
        self.geology_readout = RegionalReadout() if regional else None
        if regional:
            old = self.level_head[0]
            new = nn.Linear(old.in_features + RegionalReadout.output_size, old.out_features)
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:, :old.in_features].copy_(old.weight)
                new.bias.copy_(old.bias)
            self.level_head[0] = new

    def forward(self, field, scalar, return_parts=False):
        with torch.autocast(field.device.type, enabled=False):
            geology = geology_maps(field)
        condition = self.condition(scalar)
        h, skips = field, []
        for i, block in enumerate(self.enc):
            if i:
                h = F.avg_pool2d(h, 2)
            h = block(h, condition)
            if self.geology_gates is not None:
                h = self.geology_gates[i](h, geology)
            skips.append(h)
        pos = F.interpolate(field[:, 2:4], size=h.shape[-2:], mode='bilinear', align_corners=True)
        shape = h.shape
        h = self.global_mix((h + self.position(pos)).flatten(2).transpose(1, 2)).transpose(1, 2).reshape(shape)
        features = torch.cat([h.mean((-2, -1)), h.amax((-2, -1))], 1)
        grid = self.pressure_grid(h).flatten(1)
        region = self.geology_readout(geology, field) if self.geology_readout is not None else None
        ht = self.decode(h, skips, condition, self.temperature_decoder)
        hp = self.decode(h, skips, condition, self.pressure_decoder)
        with torch.autocast(field.device.type, enabled=False):
            temperature = self.temperature_head(ht.float()) * self.scales[0]
            raw = self.pressure_head(hp.float())
            residual = (raw - raw.mean((-2, -1), keepdim=True)) * self.pressure_residual_scale
            features = torch.cat([features.float(), DirectFieldOperator.well_features(field),
                                  scalar.float(), grid.float()], 1)
            if region is not None:
                features = torch.cat([features, region.float()], 1)
            level = self.level_head(features)[:, :, None, None] * self.scales[1]
            pressure = level + residual
            result = torch.cat([temperature, pressure], 1)
        if return_parts:
            return result, dict(temperature=temperature, pressure=pressure,
                                pressure_level=level, pressure_residual=residual)
        return result


def load_parent(path, map_location='cpu'):
    """Load a grid-output predictor checkpoint (e.g. the transfer parent ``cf278e64...``)."""
    ck = torch.load(path, map_location=map_location, weights_only=True)
    cfg = ck['model_config'].copy()
    arch = cfg.pop('architecture')
    if arch == 'task_decoders':
        net = JointTaskOperator(**cfg)
    elif arch == 'geology_task_decoders':
        net = GeologyJointOperator(**cfg)
    else:
        raise ValueError(f'Unsupported parent architecture: {arch}')
    net.load_state_dict(ck['model'], strict=True)
    return net.eval()
