# SPDX-License-Identifier: GPL-3.0-only
"""Spatial backbone adaptation of Xue et al.'s official PCNO v1.0.

Upstream: https://git.codeocean.com/capsule-8000337.git
Commit: dc4cd4417ed7f97715ffb47496690ce60f7665a3
Author: Zhenqian Xue. Original files/license retained in upstream/.

The 4-D FFT becomes a 2-D FFT; the per-time 3-D U-Net becomes 2-D.
Global fusion, four spectral/pointwise blocks, reused gated U-Net features,
GroupNorm, padding and separate T/p cores are retained. Five physical scalar
controls include time. This is NOT the full physics-constrained PCNO: neither
the 3-D/time transform nor its original PDE/wellbore losses is reproduced.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class Spectral2D(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.width, self.modes = width, tuple(modes)
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.view_as_real(torch.rand(width, width, *modes, dtype=torch.cfloat) / width**2))
                for _ in range(2)
            ]
        )

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            xf = torch.fft.rfft2(x.float())
            out = xf.new_zeros(x.shape[0], self.width, x.shape[-2], x.shape[-1] // 2 + 1)
            m1, m2 = min(self.modes[0], x.shape[-2] // 2), min(self.modes[1], xf.shape[-1])
            for sl, weight in zip((slice(0, m1), slice(-m1, None)), self.weights):
                out[:, :, sl, :m2] = torch.einsum(
                    "bixy,ioxy->boxy", xf[:, :, sl, :m2], torch.view_as_complex(weight.contiguous())[:, :, :m1, :m2]
                )
            return torch.fft.irfft2(out, s=x.shape[-2:])


def convs(cin, channels):
    layers = []
    for cout in channels:
        layers += [nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.ReLU()]
        cin = cout
    return nn.Sequential(*layers)


class UNet2D(nn.Module):
    def __init__(self, width):
        super().__init__()
        c = width // 2
        self.enc1, self.enc2 = convs(width, (c, c)), convs(c, (2 * c, 2 * c))
        self.bottom = convs(2 * c, (4 * c, 2 * c))
        self.up1, self.up2 = nn.ConvTranspose2d(2 * c, 2 * c, 2, stride=2), nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec1, self.dec2 = convs(4 * c, (2 * c,)), convs(2 * c, (c,))
        self.out = nn.Conv2d(c, width, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.bottom(F.max_pool2d(b, 2))
        d = self.dec1(torch.cat([b, self.up1(c)], 1))
        return self.out(self.dec2(torch.cat([a, self.up2(d)], 1)))


class Core(nn.Module):
    def __init__(self, width=32, modes=(14, 14)):
        super().__init__()
        self.lift = nn.Conv2d(6, width, 1)
        self.global_fusion = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, width))
        self.fusion = nn.Conv2d(2 * width, width, 1)
        self.fusion_norm = nn.GroupNorm(8, width)
        self.spectral = nn.ModuleList([Spectral2D(width, modes) for _ in range(4)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(4)])
        self.norm = nn.ModuleList([nn.GroupNorm(8, width) for _ in range(4)])
        self.unet = UNet2D(width)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.head = nn.Sequential(nn.Conv2d(width, 4 * width, 1), nn.GELU(), nn.Conv2d(4 * width, 1, 1))

    def forward(self, field, scalar):
        x = F.pad(self.lift(field), [16] * 4)
        g = self.global_fusion(scalar)[:, :, None, None].expand(-1, -1, *x.shape[-2:])
        x = self.fusion_norm(self.fusion(torch.cat([x, g], 1)))
        u = checkpoint(self.unet, x, use_reentrant=False) if self.training else self.unet(x)
        for spectral, linear, norm in zip(self.spectral, self.pointwise, self.norm):

            def step(h, unet, spectral=spectral, linear=linear, norm=norm):
                return F.gelu(norm(spectral(h) + linear(h) + self.alpha * unet))

            x = checkpoint(step, x, u, use_reentrant=False) if self.training else step(x, u)
        return self.head(x[:, :, 16:-16, 16:-16])


class PCNOSpatialBackbone(nn.Module):
    def __init__(self, scales, pressure_residual_scale, width=(56, 64), modes=(14, 14)):
        super().__init__()
        self.config = dict(
            architecture="pcno_spatial_backbone",
            scales=list(scales),
            pressure_residual_scale=float(pressure_residual_scale),
            width=list(width),
            modes=list(modes),
        )
        self.kind = "joint"
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32).flatten())
        self.register_buffer("pressure_residual_scale", torch.tensor(float(pressure_residual_scale)))
        self.temperature_core, self.pressure_core = Core(width[0], modes), Core(width[1], modes)

    def forward(self, field, scalar):
        return torch.cat(
            [
                self.temperature_core(field, scalar).float() * self.scales[0],
                self.pressure_core(field, scalar).float() * self.scales[1],
            ],
            1,
        )


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    config = ck["model_config"].copy()
    if config.pop("architecture") != "pcno_spatial_backbone":
        raise ValueError("Expected a PCNO spatial-backbone checkpoint")
    net = PCNOSpatialBackbone(**config)
    net.load_state_dict(ck["model"], strict=True)
    return net.eval()
