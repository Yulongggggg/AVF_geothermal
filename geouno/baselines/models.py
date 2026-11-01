"""Geothermal comparison architectures with checkpoint-compatible parameter names.

These are the two-dimensional, two-field adaptations used in the common
99-epoch comparison. Spatial inputs have six channels and controls five.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _cbr(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, 2, 1), nn.GroupNorm(8, cout), nn.GELU())


class DeepONet(nn.Module):
    def __init__(self, cin=10, p=256, tw=256, octaves=7):
        super().__init__()
        self.p = p
        self.octaves = octaves
        self.branch = nn.Sequential(
            _cbr(cin, 48),
            _cbr(48, 96),
            _cbr(96, 128),
            _cbr(128, 192),
            _cbr(192, 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 2 * p),
        )
        din = 2 + 4 * octaves
        self.trunk = nn.Sequential(
            nn.Linear(din, tw), nn.GELU(), nn.Linear(tw, tw), nn.GELU(), nn.Linear(tw, tw), nn.GELU(), nn.Linear(tw, p)
        )
        self.bias = nn.Parameter(torch.zeros(2))
        self._cc = {}

    def _coords(self, Hh, Ww, dev):
        key = (Hh, Ww, dev)
        if key not in self._cc:
            y = torch.linspace(-1, 1, Hh, device=dev)
            x = torch.linspace(-1, 1, Ww, device=dev)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            c = torch.stack([xx, yy], -1).reshape(-1, 2)
            fs = [c]
            for k in range(self.octaves):
                w = (2.0**k) * torch.pi
                fs += [torch.sin(w * c), torch.cos(w * c)]
            self._cc[key] = torch.cat(fs, -1)
        return self._cc[key]

    def forward(self, x):
        B, _, Hh, Ww = x.shape
        coef = self.branch(x).view(B, 2, self.p)  # [B,2,p]
        basis = self.trunk(self._coords(Hh, Ww, x.device))  # [N,p]
        out = torch.einsum("bcp,np->bcn", coef, basis).view(B, 2, Hh, Ww)
        return out + self.bias.view(1, 2, 1, 1)


class CondBlock(nn.Module):
    def __init__(self, cin, cout, cdim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.film = nn.Linear(cdim, 2 * cout)
        nn.init.normal_(self.film.weight, std=0.01)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, c):
        h = F.gelu(self.norm1(self.conv1(x)))
        gain, shift = self.film(c).chunk(2, dim=1)
        h = h * (1 + torch.tanh(gain)[:, :, None, None]) + shift[:, :, None, None]
        h = self.norm2(self.conv2(F.gelu(h)))
        return F.gelu(h + self.skip(x))


class ThermalFieldNet(nn.Module):
    def __init__(self, global_context=True, width=32):
        super().__init__()
        if width % 8:
            raise ValueError("width must be divisible by 8")
        self.config = dict(global_context=global_context, width=width)
        dims = [width, 2 * width, 4 * width, 6 * width]
        self.condition = nn.Sequential(nn.Linear(5, 128), nn.GELU(), nn.Linear(128, 128))
        self.enc = nn.ModuleList([CondBlock(6, dims[0])] + [CondBlock(a, b) for a, b in zip(dims, dims[1:])])
        if global_context:
            layer = nn.TransformerEncoderLayer(
                dims[-1], 6, 2 * dims[-1], dropout=0, activation="gelu", batch_first=True, norm_first=True
            )
            self.global_mix = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.position = nn.Conv2d(2, dims[-1], 1)
        else:
            self.global_mix = None
        self.dec = nn.ModuleList([CondBlock(dims[i + 1] + dims[i], dims[i]) for i in (2, 1, 0)])
        self.out = nn.Conv2d(width, 1, 1)

    def forward(self, field, scalar):
        if field.shape[1] != 6 or scalar.shape[1] != 5:
            raise ValueError("Expected six physical field channels and five scalar controls")
        c = self.condition(scalar)
        skips = []
        h = field
        for i, block in enumerate(self.enc):
            if i:
                h = F.avg_pool2d(h, 2)
            h = block(h, c)
            skips.append(h)
        if self.global_mix is not None:
            pos = F.interpolate(field[:, 2:4], size=h.shape[-2:], mode="bilinear", align_corners=True)
            shape = h.shape
            tok = (h + self.position(pos)).flatten(2).transpose(1, 2)
            h = self.global_mix(tok).transpose(1, 2).reshape(shape)
        for block, skip in zip(self.dec, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=True)
            h = block(torch.cat([h, skip], 1), c)
        return self.out(h)


def packed_input(field, scalar):
    return torch.cat([field, scalar[:, :, None, None].expand(-1, -1, *field.shape[-2:])], 1)


class SpectralConv2d(nn.Module):
    """Real storage of complex weights permits multi-GPU replication."""

    def __init__(self, cin, cout, m1, m2):
        super().__init__()
        self.cout = cout
        self.m1 = m1
        self.m2 = m2
        for name in ("w1", "w2"):
            self.register_parameter(
                name, nn.Parameter(torch.view_as_real(torch.rand(cin, cout, m1, m2, dtype=torch.cfloat) / (cin * cout)))
            )

    def forward(self, x):
        dt = x.dtype
        with torch.autocast(x.device.type, enabled=False):
            b, _, h, w = x.shape
            xf = torch.fft.rfft2(x.float(), norm="ortho")
            out = xf.new_zeros(b, self.cout, h, w // 2 + 1)
            m1 = min(self.m1, h // 2)
            m2 = min(self.m2, w // 2 + 1)
            for sl, weight in ((slice(0, m1), self.w1), (slice(-m1, None), self.w2)):
                out[:, :, sl, :m2] = torch.einsum(
                    "bixy,ioxy->boxy", xf[:, :, sl, :m2], torch.view_as_complex(weight.contiguous())[:, :, :m1, :m2]
                )
            return torch.fft.irfft2(out, s=(h, w), norm="ortho").to(dt)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, cin=11, cout=2, width=32):
        super().__init__()
        dims = [width, 2 * width, 4 * width, 6 * width]
        self.enc = nn.ModuleList([Block(cin, dims[0])] + [Block(a, b) for a, b in zip(dims, dims[1:])])
        self.dec = nn.ModuleList([Block(dims[i + 1] + dims[i], dims[i]) for i in (2, 1, 0)])
        self.head = nn.Conv2d(width, cout, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else F.avg_pool2d(x, 2))
            skips.append(x)
        for block, skip in zip(self.dec, reversed(skips[:-1])):
            x = block(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True), skip], 1))
        return self.head(x)


class Fourier(nn.Module):
    def __init__(self, unet=False, width=32, modes=(16, 12), depth=4):
        super().__init__()
        self.lift = nn.Conv2d(11, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, *modes) for _ in range(depth)])
        self.skip = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.local = nn.ModuleList(
            [UNet(width, width, 16) if unet and i >= depth - 2 else nn.Identity() for i in range(depth)]
        )
        self.unet = unet
        self.head = nn.Sequential(nn.Conv2d(width, 64, 1), nn.GELU(), nn.Conv2d(64, 2, 1))

    def forward(self, x):
        x = self.lift(x)
        for i, (spec, skip, local) in enumerate(zip(self.spec, self.skip, self.local)):
            y = spec(x) + skip(x)
            if self.unet and i >= len(self.spec) - 2:
                y = y + local(x)
            x = F.gelu(y)
        return self.head(x)


class ResNet(nn.Module):
    def __init__(self, width=64, depth=8):
        super().__init__()
        self.lift = nn.Conv2d(11, width, 1)
        self.blocks = nn.ModuleList([Block(width, width) for _ in range(depth)])
        self.head = nn.Conv2d(width, 2, 1)

    def forward(self, x):
        x = self.lift(x)
        for b in self.blocks:
            x = x + b(x)
        return self.head(x)


class Predictor(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "deeponet":
            self.net = DeepONet(cin=11, p=256, tw=256)
        elif kind in ("fno", "ufno"):
            self.net = Fourier(kind == "ufno")
        elif kind == "unet":
            self.net = UNet()
        elif kind == "resnet":
            self.net = ResNet()
        elif kind == "attention_unet":
            self.net = ThermalFieldNet(global_context=True, width=32)
            self.net.out = nn.Conv2d(32, 2, 1)
        else:
            raise ValueError(kind)

    def forward(self, field, scalar):
        if self.kind == "attention_unet":
            return self.net(field, scalar)
        return self.net(packed_input(field, scalar))


BASELINES = ("deeponet", "fno", "ufno", "unet", "resnet", "attention_unet")
COMPARISON_MODELS = (*BASELINES, "geouno", "resid64_p1")


class PhysicalBaseline(nn.Module):
    def __init__(self, kind, scales, pressure_residual_scale):
        super().__init__()
        self.config = dict(
            architecture="fair_physical_baseline",
            kind=kind,
            scales=list(scales),
            pressure_residual_scale=float(pressure_residual_scale),
        )
        self.predictor = Predictor(kind)
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        self.register_buffer("pressure_residual_scale", torch.tensor(pressure_residual_scale, dtype=torch.float32))

    def forward(self, field, scalar):
        return self.predictor(field, scalar).float() * self.scales[None, :, None, None]


def build(kind, scales, pressure_residual_scale, bank_path=None):
    if kind in ("geouno", "resid64_p1"):
        from ..model.final import FinalGeoUNO

        if bank_path is None:
            raise ValueError("GeoUNO requires the training-only POD bank (--bank)")
        model = FinalGeoUNO(scales, pressure_residual_scale, "resid64_p1")
        model.initialize_pod(bank_path)
        return model
    if kind not in BASELINES:
        raise ValueError(f"Unknown baseline {kind!r}; choose one of {BASELINES}")
    return PhysicalBaseline(kind, scales, pressure_residual_scale)


def load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"].copy()
    architecture = config.pop("architecture")
    if architecture == "fair_physical_baseline":
        model = PhysicalBaseline(**config)
    elif architecture == "geouno_simple":
        from ..model.final import FinalGeoUNO

        model = FinalGeoUNO(**config)
    else:
        raise ValueError(f"Unsupported comparison checkpoint architecture {architecture!r}")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()
