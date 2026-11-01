"""Study-specific PNO architecture and composite proper energy score.

Method: Bülte, Scholl and Kutyniok (2025), https://github.com/cbuelt/pfno.
The architecture and training recipe here are the geothermal study adaptation.
"""

import torch
from torch import nn
from torch.nn import functional as F


def packed_input(field, scalar):
    return torch.cat([field, scalar[:, :, None, None].expand(-1, -1, *field.shape[-2:])], 1)


class InitialSpectralConv2d(nn.Module):
    """Complex-weight initialization used before the real-storage substitution."""

    def __init__(self, cin, cout, m1, m2):
        super().__init__()
        self.cin, self.cout, self.m1, self.m2 = cin, cout, m1, m2
        s = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(s * torch.rand(cin, cout, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(s * torch.rand(cin, cout, m1, m2, dtype=torch.cfloat))

    def forward(self, x):
        dt = x.dtype
        with torch.autocast("cuda", enabled=False):
            x = x.float()
            B, C, H, W = x.shape
            xf = torch.fft.rfft2(x, norm="ortho")
            out = torch.zeros(B, self.cout, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
            m1 = min(self.m1, H // 2)
            m2 = min(self.m2, W // 2 + 1)
            out[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
            out[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
            y = torch.fft.irfft2(out, s=(H, W), norm="ortho")
        return y.to(dt)


class FNOBlock(nn.Module):
    def __init__(self, width, m1, m2, local=True):
        super().__init__()
        self.spec = InitialSpectralConv2d(width, width, m1, m2)
        self.w = nn.Conv2d(width, width, 1)
        self.loc = nn.Conv2d(width, width, 3, padding=1) if local else None
        self.norm = nn.GroupNorm(8, width)

    def forward(self, x):
        y = self.spec(x) + self.w(x)
        if self.loc is not None:
            y = y + self.loc(x)
        return F.gelu(self.norm(y)) + x


class NoiseInject(nn.Module):
    """Latent-dependent affine feature modulation."""

    def __init__(self, width, lat):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(lat, 2 * width), nn.GELU(), nn.Linear(2 * width, 2 * width))
        self.width = width

    def forward(self, x, xi):
        g, b = self.f(xi).chunk(2, -1)  # [B,width] each
        return x * (1.0 + g[:, :, None, None]) + b[:, :, None, None]


class PNO(nn.Module):
    """One conditional field sample per forward pass."""

    def __init__(self, cin=10, width=64, modes=(24, 20), depth=5, lat=64, noise_ch=2):
        super().__init__()
        self.lat, self.noise_ch = lat, noise_ch
        self.lift = nn.Sequential(nn.Conv2d(cin + noise_ch, width, 1), nn.GELU(), nn.Conv2d(width, width, 1))
        self.blocks = nn.ModuleList([FNOBlock(width, modes[0], modes[1]) for _ in range(depth)])
        self.inj = nn.ModuleList([NoiseInject(width, lat) for _ in range(depth)])
        self.head = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, 2, 1))

    def forward(self, x, xi=None):
        B, _, H, W = x.shape
        if xi is None:
            xi = torch.randn(B, self.lat, device=x.device, dtype=x.dtype)
        nz = torch.randn(B, self.noise_ch, H, W, device=x.device, dtype=x.dtype)
        h = self.lift(torch.cat([x, nz], 1))
        for blk, inj in zip(self.blocks, self.inj):
            h = inj(blk(h), xi)
        return self.head(h)

    def sample(self, x, K):
        """Return K conditional field samples."""
        outs = [self.forward(x) for _ in range(K)]
        return torch.stack(outs, 1)


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


class AMPReplica(nn.Module):
    def __init__(self, net, samples=1):
        super().__init__()
        self.net = net
        self.samples = samples

    def forward(self, field, scalar):
        with torch.autocast(device_type=field.device.type, dtype=torch.bfloat16, enabled=field.is_cuda):
            if self.samples == 1:
                return self.net(field, scalar).float()
            return torch.stack([self.net(field, scalar).float() for _ in range(self.samples)], 1)


def energy_per_sample(pred, truth):
    def norm(d):
        return (
            d.square().mean(tuple(range(3, d.ndim))).clamp_min(1e-20).sqrt()
            if d.ndim >= 6
            else d.square().mean(tuple(range(2, d.ndim))).clamp_min(1e-20).sqrt()
        )

    # Explicit axes avoid ambiguity between field and projected vector scores.
    a = (pred - truth[:, None]).flatten(2).square().mean(-1).clamp_min(1e-20).sqrt().mean(1)
    k = pred.shape[1]
    pair = (pred[:, :, None] - pred[:, None, :]).flatten(3)
    d = pair.square().mean(-1).clamp_min(1e-20).sqrt()
    off = ~torch.eye(k, dtype=torch.bool, device=pred.device)
    return a - 0.5 * d[:, off].mean(1)


def loss(pred, truth, field, weights, probabilistic=False, return_per=False):
    mask = field[:, 1:2]

    def mse(d, mask=mask):
        return (d.square() * mask).sum((-2, -1)) / mask.sum((-2, -1)).clamp_min(1)

    if probabilistic:
        per = energy_per_sample(pred[:, :, :, 16:144], truth[:, :, 16:144])
        per = per + 0.1 * energy_per_sample(pred * (1 - mask[:, None]), truth * (1 - mask))
        gx = pred[..., 1:] - pred[..., :-1]
        tx = truth[..., 1:] - truth[..., :-1]
        gy = pred[..., 1:, :] - pred[..., :-1, :]
        ty = truth[..., 1:, :] - truth[..., :-1, :]
        per = per + 0.1 * (energy_per_sample(gx, tx) + energy_per_sample(gy, ty)) / (0.02**0.5)
        per = per + 0.25 * energy_per_sample(pred[:, :, :, 16:144].mean((-2, -1)), truth[:, :, 16:144].mean((-2, -1)))
    else:
        e = pred - truth
        per = mse(e).mean(1)
        outside = 1 - mask
        per = per + 0.1 * mse(e, outside).mean(1)
        gx = e[..., 1:] - e[..., :-1]
        gy = e[..., 1:, :] - e[..., :-1, :]
        per = per + 0.1 * (gx.square().mean((1, 2, 3)) + gy.square().mean((1, 2, 3))) / 0.02
        per = per + 0.25 * (e[:, 1, 16:144].mean((-2, -1))).square()
    return per if return_per else (per * weights).sum() / weights.sum()


class Predictor(nn.Module):
    def __init__(self, kind="pno"):
        super().__init__()
        if kind != "pno":
            raise ValueError(kind)
        self.kind = kind
        self.net = PNO(cin=13, width=32, modes=(16, 12), depth=4, lat=64)
        for block in self.net.blocks:
            block.spec = SpectralConv2d(32, 32, 16, 12)

    def forward(self, field, scalar):
        return self.net(packed_input(field, scalar))


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["kind"] != "pno":
        raise ValueError("Expected a PNO checkpoint")
    net = Predictor()
    net.load_state_dict(ck["model"], strict=True)
    return net.eval()
