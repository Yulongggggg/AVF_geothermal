"""Joint saturation-pressure models on the 96 by 200 CO2 benchmark grid."""

import torch
from torch import nn
from torch.nn import functional as F
from ..baselines.models import Predictor, Block, DeepONet, CondBlock, ThermalFieldNet
from ..model.final import FinalGeoUNO, spd_solve
from ..model.direct import DirectFieldOperator
from ..model.geouno import coordinates

H, W, CIN, NSCAL = 96, 200, 8, 6
PAPER_MODELS = ("deeponet", "fno", "ufno", "unet", "resnet", "geouno", "geouno_r2048", "geouno_r4096")
KINDS = (*PAPER_MODELS, "attention_unet")


def widen_conv(conv, cin):
    new = nn.Conv2d(cin, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        nn.init.kaiming_uniform_(new.weight, a=5**0.5)
        new.weight[:, : conv.in_channels].copy_(conv.weight)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def widen_linear(linear, insert_at, extra):
    """Insert `extra` zero-initialised input columns at `insert_at` (behaviour unchanged for zero inputs)."""
    new = nn.Linear(linear.in_features + extra, linear.out_features, bias=linear.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :insert_at].copy_(linear.weight[:, :insert_at])
        new.weight[:, insert_at + extra :].copy_(linear.weight[:, insert_at:])
        if linear.bias is not None:
            new.bias.copy_(linear.bias)
    return new


def widen_condblock(block, cin):
    new = CondBlock(cin, block.conv1.out_channels)
    new.load_state_dict(
        {k: v for k, v in block.state_dict().items() if not k.startswith(("conv1.", "skip."))}, strict=False
    )
    new.conv1 = widen_conv(block.conv1, cin)
    if isinstance(block.skip, nn.Conv2d):
        new.skip = widen_conv(block.skip, cin)
    return new


class UFNOThermalFieldNet(ThermalFieldNet):
    def forward(self, field, scalar):
        if field.shape[1] != 8 or scalar.shape[1] != 6:
            raise ValueError("Expected eight field channels and six scalar controls")
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


class PhysicalBaseline(nn.Module):
    def __init__(self, kind, scales, pressure_residual_scale):
        super().__init__()
        self.config = dict(
            architecture="ufno_physical_baseline",
            kind=kind,
            scales=list(scales),
            pressure_residual_scale=float(pressure_residual_scale),
        )
        self.predictor = Predictor(kind)
        net = self.predictor.net
        packed = CIN + NSCAL
        if kind == "deeponet":
            self.predictor.net = DeepONet(cin=packed, p=256, tw=256)
        elif kind in ("fno", "ufno", "resnet"):
            net.lift = widen_conv(net.lift, packed)
        elif kind == "unet":
            net.enc[0] = Block(packed, net.enc[0].net[0].out_channels)
        elif kind == "attention_unet":
            new = UFNOThermalFieldNet(global_context=True, width=32)
            new.out = nn.Conv2d(32, 2, 1)
            new.load_state_dict(net.state_dict())
            new.enc[0] = widen_condblock(new.enc[0], CIN)
            new.condition[0] = widen_linear(new.condition[0], 5, NSCAL - 5)
            self.predictor.net = new
        else:
            raise ValueError(kind)
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        self.register_buffer("pressure_residual_scale", torch.tensor(float(pressure_residual_scale)))

    def forward(self, field, scalar):
        return self.predictor(field, scalar).float() * self.scales[None, :, None, None]


class UFNOSimpleHead(FinalGeoUNO):
    """resid64_p1 (no correction MLP, no P0; 12 near-well statistics in the level network) on a 96 x 200 grid."""

    def __init__(self, scales, pressure_residual_scale, pod_rank=1024, learned_rank=64, width=32):
        if width != 32:
            raise ValueError("The published CO2 architecture uses width=32")
        super().__init__(
            scales,
            pressure_residual_scale,
            variant="resid64_p1",
            pod_rank=pod_rank,
            learned_rank=learned_rank,
            width=width,
        )
        self.config.update(architecture="ufno_geouno_simple", grid=[H, W], cin=CIN, nscalar=NSCAL)
        self.grid = (H, W)
        for name, value in (
            ("sensor_coordinates", coordinates(height=H, width=W)),
            ("pod", torch.zeros(2, pod_rank, H, W)),
            ("mean_function", torch.zeros(2, 1, H, W)),
        ):
            delattr(self, name)
            self.register_buffer(name, value)
        self.enc[0] = widen_condblock(self.enc[0], CIN)
        self.condition[0] = widen_linear(self.condition[0], 5, NSCAL - 5)
        # level features: [global 2*6*width | scalars | pressure grid 256 | near-well statistics 12]
        self.level_head[0] = widen_linear(self.level_head[0], 2 * 6 * width + 5, NSCAL - 5)

    def forward(self, field, scalar, query=None, return_parts=False):
        coeff, level = self.branch(field, scalar)  # no gradient checkpointing, same expansion in train and eval
        with torch.autocast(field.device.type, enabled=False):
            basis, mean = self.basis(None)
            values = torch.einsum("bcr,crn->bcn", coeff.float(), basis) + mean[None]
            t = (values[:, 0] + level[:, 0:1]) * self.scales[0]
            p = values[:, 1] * self.pressure_residual_scale + level[:, 1:2] * self.scales[1]
            return torch.stack((t, p), 1).reshape(-1, 2, *self.grid)

    def branch(self, field, scalar):
        if field.shape[-2:] != self.grid:
            raise ValueError("Sample the input function on the fixed CO2 branch sensors.")
        condition = self.condition(scalar)
        h, skips = field, []
        for i, block in enumerate(self.enc):
            h = block(F.avg_pool2d(h, 2) if i else h, condition)
            skips.append(h)
        pos = F.interpolate(field[:, 2:4], size=h.shape[-2:], mode="bilinear", align_corners=True)
        shape = h.shape
        h = self.global_mix((h + self.position(pos)).flatten(2).transpose(1, 2)).transpose(1, 2).reshape(shape)
        features = torch.cat((h.mean((-2, -1)), h.amax((-2, -1)), scalar, self.pressure_grid(h).flatten(1)), 1)
        ht = self.decode(h, skips, condition, self.temperature_decoder)
        hp = self.decode(h, skips, condition, self.pressure_decoder)
        with torch.autocast(field.device.type, enabled=False):
            if self.well_inputs:  # P1: 12 near-well statistics
                features = torch.cat((features.float(), DirectFieldOperator.well_features(field)), 1)
            mt, mp = self.temperature_head(ht.float()), self.pressure_head(hp.float())  # B, 1(+L), H, W
            raw = torch.cat((mt[:, :1], mp[:, :1]), 1).flatten(2)  # B, 2, N
            raw_mean = raw.mean(-1, keepdim=True)
            hc = raw - raw_mean - self.mean_function.flatten(2).squeeze(1)[None]  # centred decoder maps
            pod = self.pod.flatten(2)
            n = raw.shape[-1]
            a = torch.einsum("bcn,crn->bcr", hc, pod) / n  # POD coefficients
            phi = self.basis(None)[0][:, -self.config["learned_rank"] :]  # 2, L, N
            if self.method == "chan64":  # C: dedicated decoder channels
                g = torch.stack((mt[:, 1:], mp[:, 1:]), 1).flatten(3)  # B, 2, L, N
                c = (g * phi[None]).sum(-1) / n
            else:
                Gff = torch.einsum("ckn,cjn->ckj", phi, phi) / n  # 2, L, L
                if self.method == "resid64":  # A: residual projection
                    r = hc - torch.einsum("bcr,crn->bcn", a, pod)
                    c = spd_solve(Gff, torch.einsum("bcn,ckn->bck", r, phi) / n)
                else:  # B: joint least squares
                    Gpf = torch.einsum("crn,ckn->crk", pod, phi) / n  # 2, R, L
                    S = Gff - torch.einsum("crk,crj->ckj", Gpf, Gpf)  # Schur complement
                    rhs = torch.einsum("bcn,ckn->bck", hc, phi) / n - torch.einsum("bcr,crk->bck", a, Gpf)
                    c = spd_solve(S, rhs)
                    a = a - torch.einsum("crk,bck->bcr", Gpf, c)
            if hasattr(self, "learned_gain"):
                c = c * self.learned_gain[None]
            levels = torch.cat((raw_mean[:, 0], self.level_head(features.float())), 1)
        return torch.cat((a, c), -1), levels


def build(kind, scales, residual, bank_path=None):
    if kind.startswith("geouno"):
        rank = int(kind.split("_r")[1]) if "_r" in kind else 1024
        net = UFNOSimpleHead(scales, residual, pod_rank=rank, learned_rank=64)
        if bank_path is not None:
            net.initialize_pod(str(bank_path))
        return net
    return PhysicalBaseline(kind, scales, residual)


def count(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"].copy()
    architecture = config.pop("architecture")
    if architecture == "ufno_physical_baseline":
        net = PhysicalBaseline(**config)
    elif architecture == "ufno_geouno_simple":
        for key, expected in (("grid", [H, W]), ("cin", CIN), ("nscalar", NSCAL), ("variant", "resid64_p1")):
            if config.pop(key) != expected:
                raise ValueError(f"Unexpected CO2 model setting {key}")
        net = UFNOSimpleHead(**config)
    else:
        raise ValueError(f"Unsupported CO2 checkpoint architecture: {architecture}")
    net.load_state_dict(checkpoint["model"], strict=True)
    return net.eval()
