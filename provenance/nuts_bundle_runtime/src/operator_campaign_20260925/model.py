"""GeoUNO output-head conversion, retaining ALL original trained modules.

u_c(x;m,v) = constant_c(m,v) + mean_c(x)
            + sum_j coefficient_cj(m,v) * basis_cj(x).

Decoder features generate coefficients, never bypassing the spatial expansion.
Training-only POD modes are extended by bilinear interpolation; learned modes
depend only on query coordinates. Temperature uses multiscale Fourier features.
Inputs still use fixed branch sensors; input-mesh invariance is not claimed.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from encoder_study.model import GeologyJointOperator, geology_maps
from direct_operator.model import DirectFieldOperator


def coordinates(device='cpu', height=160, width=128):
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=device),
                            torch.linspace(-1, 1, width, device=device), indexing='ij')
    return torch.stack((xx, yy), -1).reshape(-1, 2)


class CoordinateTrunk(nn.Module):
    def __init__(self, rank, width=128, octaves=6):
        super().__init__()
        self.register_buffer('frequency', math.pi * 2.**torch.arange(octaves))
        def mlp(inputs):
            return nn.Sequential(nn.Linear(inputs, width), nn.GELU(), nn.Linear(width, width),
                                 nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, rank))
        # Temperature: 1,2,4,8,16,32 cycles over each normalized coordinate domain.
        self.temperature = mlp(2+4*octaves)
        self.pressure = mlp(2)

    def forward(self, xy):
        angle = (xy[..., None] * self.frequency).flatten(1)
        fourier = torch.cat((xy, angle.sin(), angle.cos()), 1)
        return torch.stack((self.temperature(fourier).T, self.pressure(xy).T))


class GeoUNOBasisHead(GeologyJointOperator):
    def __init__(self, scales, pressure_residual_scale, pod_rank=512, learned_rank=64, width=32):
        super().__init__(scales, pressure_residual_scale, width, regional=True, gates=False)
        self.config = dict(architecture='geouno_basis_head', scales=list(scales),
                           pressure_residual_scale=float(pressure_residual_scale),
                           pod_rank=pod_rank, learned_rank=learned_rank, width=width)
        self.register_buffer('sensor_coordinates', coordinates())
        self.register_buffer('pod', torch.zeros(2, pod_rank, 160, 128))
        self.register_buffer('mean_function', torch.zeros(2, 1, 160, 128))
        self.trunk = CoordinateTrunk(learned_rank)
        self.coefficient_delta = nn.Sequential(nn.LayerNorm(769), nn.Linear(769, 256), nn.GELU(),
                                               nn.Linear(256, 2*(pod_rank+learned_rank)))
        nn.init.zeros_(self.coefficient_delta[-1].weight)
        nn.init.zeros_(self.coefficient_delta[-1].bias)
        self.branch_chunk = 4

    def initialize_parent(self, path):
        old = torch.load(path, map_location='cpu', weights_only=False)['model']
        missing, unexpected = self.load_state_dict(old, strict=False)
        assert not unexpected
        assert all(n.startswith(('trunk.', 'coefficient_delta.', 'pod', 'mean_function', 'sensor_coordinates')) for n in missing)
        assert all(torch.equal(self.state_dict()[n].cpu(), v) for n, v in old.items())
        return list(old)

    def initialize_pod(self, path):
        bank = torch.load(path, map_location='cpu', weights_only=False)
        assert bank['split'] == 'train'
        r = self.config['pod_rank']
        self.pod.copy_(bank['basis'][:, :r].reshape_as(self.pod))
        self.mean_function.copy_(bank['mean'].reshape_as(self.mean_function))

    def branch(self, field, scalar):
        if field.shape[-2:] != (160, 128):
            raise ValueError('Sample the input function on the fixed 160 x 128 branch sensors.')
        condition = self.condition(scalar)
        h, skips = field, []
        for i, block in enumerate(self.enc):
            h = block(F.avg_pool2d(h, 2) if i else h, condition)
            skips.append(h)
        pos = F.interpolate(field[:, 2:4], size=h.shape[-2:], mode='bilinear', align_corners=True)
        shape = h.shape
        h = self.global_mix((h+self.position(pos)).flatten(2).transpose(1, 2)).transpose(1, 2).reshape(shape)
        global_features = torch.cat((h.mean((-2, -1)), h.amax((-2, -1))), 1)
        grid = self.pressure_grid(h).flatten(1)
        ht = self.decode(h, skips, condition, self.temperature_decoder)
        hp = self.decode(h, skips, condition, self.pressure_decoder)
        with torch.autocast(field.device.type, enabled=False):
            geology = geology_maps(field)
        region = self.geology_readout(geology, field)
        with torch.autocast(field.device.type, enabled=False):
            features = torch.cat((global_features.float(), DirectFieldOperator.well_features(field),
                                  scalar.float(), grid.float(), region.float()), 1)
            # Old 1x1 weights supply scalar features to the coefficient head.
            # These grid features NEVER reach the output without basis expansion.
            raw = torch.cat((self.temperature_head(ht.float()), self.pressure_head(hp.float())), 1).flatten(2)
            raw_mean = raw.mean(-1, keepdim=True)
            centered = raw-raw_mean-self.mean_function.flatten(2).squeeze(1)[None]
            coef = torch.einsum('bcn,crn->bcr', centered, self.pod.flatten(2))/raw.shape[-1]
            delta = self.coefficient_delta(features).reshape(len(field), 2, -1)
            coeff = torch.cat((coef, coef.new_zeros(len(field), 2, self.config['learned_rank'])), -1) + delta
            levels = torch.cat((raw_mean[:, 0], self.level_head(features)), 1)
        return coeff, levels

    def basis(self, query=None):
        xy = self.sensor_coordinates if query is None else query
        learned = self.trunk(xy.float()).float()
        ref = learned if query is None else self.trunk(self.sensor_coordinates).float()
        centers = ref.mean(-1, keepdim=True)
        norms = (ref-centers).square().mean(-1, keepdim=True).clamp_min(1e-6).sqrt()
        learned = (learned-centers)/norms
        if query is None:
            pod, mean = self.pod.flatten(2), self.mean_function.flatten(2).squeeze(1)
        else:
            grid = xy.reshape(1, 1, -1, 2).expand(2, -1, -1, -1)
            pod = F.grid_sample(self.pod, grid, align_corners=True, mode='bilinear').squeeze(2)
            mean = F.grid_sample(self.mean_function, grid, align_corners=True, mode='bilinear').squeeze(2).squeeze(1)
        return torch.cat((pod, learned), 1), mean

    def forward(self, field, scalar, query=None, return_parts=False):
        if query is not None and (query.ndim != 2 or query.shape[1] != 2):
            raise ValueError('query must be [N,2] normalized physical coordinates (x/400,y/500).')
        coefs, levels = [], []
        for j in range(0, len(field), self.branch_chunk):
            args = (field[j:j+self.branch_chunk], scalar[j:j+self.branch_chunk])
            c, b = checkpoint(self.branch, *args, use_reentrant=False) if self.training and torch.is_grad_enabled() else self.branch(*args)
            coefs.append(c); levels.append(b)
        coeff, level = torch.cat(coefs), torch.cat(levels)
        with torch.autocast(field.device.type, enabled=False):
            basis, mean = self.basis(query)
            values = torch.einsum('bcr,crn->bcn', coeff.float(), basis) + mean[None]
            t = (values[:, 0] + level[:, 0:1]) * self.scales[0]
            p = values[:, 1] * self.pressure_residual_scale + level[:, 1:2] * self.scales[1]
            out = torch.stack((t, p), 1)
            if query is None:
                out = out.reshape(-1, 2, 160, 128)
        if return_parts:
            return out, dict(coefficients=coeff, basis=basis, mean=mean, constant_coefficients=level)
        return out


def load(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    cfg = state['model_config'].copy()
    assert cfg.pop('architecture') == 'geouno_basis_head'
    net = GeoUNOBasisHead(**cfg)
    net.load_state_dict(state['model'], strict=True)
    return net.eval()
