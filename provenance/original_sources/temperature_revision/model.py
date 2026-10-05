"""Full-resolution temperature decoder with conditional multiscale features.

This is an experimental deterministic operator, not a claimed PCNO reproduction.
The local variant provides the ablation for the optional global attention.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CondBlock(nn.Module):
    def __init__(self, cin, cout, cdim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.film = nn.Linear(cdim, 2*cout)
        nn.init.normal_(self.film.weight, std=.01); nn.init.zeros_(self.film.bias)

    def forward(self, x, c):
        h = F.gelu(self.norm1(self.conv1(x)))
        gain, shift = self.film(c).chunk(2, dim=1)
        h = h*(1+torch.tanh(gain)[:, :, None, None])+shift[:, :, None, None]
        h = self.norm2(self.conv2(F.gelu(h)))
        return F.gelu(h+self.skip(x))


class ThermalFieldNet(nn.Module):
    def __init__(self, global_context=True, width=32):
        super().__init__()
        if width % 8:
            raise ValueError("width must be divisible by 8")
        self.config = dict(global_context=global_context, width=width)
        dims = [width, 2*width, 4*width, 6*width]
        self.condition = nn.Sequential(nn.Linear(5, 128), nn.GELU(), nn.Linear(128, 128))
        self.enc = nn.ModuleList([CondBlock(6, dims[0])] + [CondBlock(a,b) for a,b in zip(dims, dims[1:])])
        if global_context:
            layer = nn.TransformerEncoderLayer(dims[-1], 6, 2*dims[-1], dropout=0,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.global_mix = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.position = nn.Conv2d(2, dims[-1], 1)
        else:
            self.global_mix = None
        self.dec = nn.ModuleList([CondBlock(dims[i+1]+dims[i], dims[i]) for i in (2,1,0)])
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
            h = block(h, c); skips.append(h)
        if self.global_mix is not None:
            pos = F.interpolate(field[:, 2:4], size=h.shape[-2:], mode="bilinear", align_corners=True)
            shape = h.shape
            tok = (h+self.position(pos)).flatten(2).transpose(1,2)
            h = self.global_mix(tok).transpose(1,2).reshape(shape)
        for block, skip in zip(self.dec, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=True)
            h = block(torch.cat([h, skip], 1), c)
        return self.out(h)


def temperature_loss(pred, truth, field, grad_weight=.1, well_weight=.1, outside_weight=.1):
    """Mean-square field, spatial-gradient and producer-readout losses.

    All terms use the same temperature units (dT / train standard deviation).
    A fixed one-cell gradient scale prevents per-example zero-front singularities.
    This is supervised gradient matching, not a conservation/PDE residual.
    """
    mask = field[:, 1:2]
    mse = ((pred-truth).square()*mask).sum()/mask.sum().clamp_min(1)
    outside = ((pred-truth).square()*(1-mask)).sum()/(1-mask).sum().clamp_min(1)
    gx = (pred[..., 1:]-pred[..., :-1])-(truth[..., 1:]-truth[..., :-1])
    gy = (pred[..., 1:, :]-pred[..., :-1, :])-(truth[..., 1:, :]-truth[..., :-1, :])
    mx = mask[..., 1:]*mask[..., :-1]
    my = mask[..., 1:, :]*mask[..., :-1, :]
    # A difference of 0.1 normalized temperature per cell is about 1.13 K.
    grad = .5*((gx.square()*mx).sum()/mx.sum().clamp_min(1)
               +(gy.square()*my).sum()/my.sum().clamp_min(1))/.01
    w = torch.exp(-.5*(field[:, 4:5]*400/5).square())
    readout = ((pred-truth)*w).sum((-2,-1))/w.sum((-2,-1)).clamp_min(1e-8)
    well = readout.square().mean()
    return mse+grad_weight*grad+well_weight*well+outside_weight*outside, {"mse": mse, "gradient": grad, "well": well, "outside": outside}
