"""Convert the public U-FNO tensors and fit training-only POD basis functions."""

import argparse
from pathlib import Path
import time
import numpy as np
import torch
from .protocol import read, write, sha
from ..baselines.protocol import file_identity
from .data import PRESSURE_OFFSET_BAR, PRESSURE_SCALE_BAR, normalize_pressure

RAW = PREP = None
H, W, T = 96, 200, 24


def convert(split, n, test_pressure_units="bar"):
    a = torch.load(RAW / f"sg_{split}_a.pt", mmap=True, weights_only=True)
    sg = torch.load(RAW / f"sg_{split}_u.pt", mmap=True, weights_only=True)
    dp = torch.load(RAW / f"dP_{split}_u.pt", mmap=True, weights_only=True)
    assert a.shape == (n, H, W, T, 12) and sg.shape == dp.shape == (n, H, W, T), (a.shape, sg.shape)
    PREP.mkdir(parents=True, exist_ok=True)
    fields = np.lib.format.open_memmap(PREP / f"{split}_fields.npy", "w+", np.float32, (n, 8, H, W))
    scalars = np.zeros((n, 5), np.float32)
    targets = np.lib.format.open_memmap(PREP / f"{split}_targets.npy", "w+", np.float32, (n, T, 2, H, W))
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    for i in range(n):
        x0 = a[i, :, :, 0, :].numpy()  # (H, W, 12) at the first time
        kx, ky, phi, perf = x0[..., 0], x0[..., 1], x0[..., 2], x0[..., 3]
        mask = (kx != 0).astype(np.float32)
        pr, pc = np.nonzero(perf > 0.5)
        if len(pr):
            d = np.sqrt((yy[..., None] - pr) ** 2 + (xx[..., None] - pc) ** 2).min(-1) / float(max(H, W))
        else:
            d = np.ones((H, W), np.float32)
        fields[i, 0] = kx
        fields[i, 1] = mask
        fields[i, 2] = 2 * x0[..., 9] - 1
        fields[i, 3] = 2 * x0[..., 10] - 1
        fields[i, 4] = d
        fields[i, 5] = d
        fields[i, 6] = ky
        fields[i, 7] = phi
        scalars[i] = x0[0, 0, 4:9]
        targets[i, :, 0] = sg[i].numpy().transpose(2, 0, 1)
        pressure = dp[i].numpy().transpose(2, 0, 1)
        if split == "test" and test_pressure_units == "bar":
            pressure = normalize_pressure(pressure)
        targets[i, :, 1] = pressure
        if (i + 1) % 250 == 0:
            print(split, i + 1, "/", n, flush=True)
    np.save(PREP / f"{split}_scalars.npy", scalars)
    if split == "train":
        np.save(PREP / "time_grid.npy", a[0, 0, 0, :, 11].numpy().astype(np.float32))
    fields.flush()
    targets.flush()
    return dict(n=n)


def stats_and_pod(rank=1024, snapshots_per_case=2):
    tr_t = np.load(PREP / "train_targets.npy", mmap_mode="r")
    tr_f = np.load(PREP / "train_fields.npy", mmap_mode="r")
    n = tr_t.shape[0]
    rng = np.random.default_rng(20261003)
    sel = [(i, int(t)) for i in range(n) for t in rng.choice(T, snapshots_per_case, replace=False)]
    x = np.empty((len(sel), 2, H * W), np.float32)
    for k, (i, t) in enumerate(sel):
        x[k] = tr_t[i, t].reshape(2, -1)
    mask = np.stack([tr_f[i, 1].reshape(-1) for i, _ in sel])
    scales = [float(np.sqrt((x[:, c] ** 2 * mask).sum() / mask.sum())) for c in range(2)]
    level = (x[:, 1] * mask).sum(1) / mask.sum(1)
    p_res = float(np.sqrt((((x[:, 1] - level[:, None]) ** 2) * mask).sum() / mask.sum()))
    stats = dict(
        fields_std=scales,
        p_res_std=p_res,
        snapshots=len(sel),
        rank=rank,
        dp_test_normalized=True,
        dp_normalization=dict(scale=PRESSURE_SCALE_BAR, offset=PRESSURE_OFFSET_BAR),
    )
    write(PREP / "stats.json", stats)
    torch.manual_seed(20261003)
    bank = dict(split="train", rank=rank, basis=torch.zeros(2, rank, H * W), mean=torch.zeros(2, H * W))
    for c, scale in enumerate([scales[0], p_res]):
        y = torch.from_numpy(x[:, c]) / scale
        y = y - y.mean(1, keepdim=True)
        mean = y.mean(0)
        mean = mean - mean.mean()
        y = y - mean
        _, s, v = torch.pca_lowrank(y, q=rank + 32, center=False, niter=4)
        q, _ = torch.linalg.qr(v[:, :rank] - v[:, :rank].mean(0), mode="reduced")
        modes = q.T * (H * W) ** 0.5
        orth = float((modes @ modes.T / (H * W) - torch.eye(rank)).abs().max())
        assert orth < 1e-3, orth
        bank["basis"][c] = modes
        bank["mean"][c] = mean
        energy = (s[:rank] ** 2).cumsum(0) / (y**2).sum()
        stats[f"pod_energy_c{c}"] = {str(r): float(energy[r - 1]) for r in (64, 256, 512, 1024) if r <= rank}
        print("channel", c, "orth", orth, "energy", stats[f"pod_energy_c{c}"], flush=True)
    torch.save(bank, PREP / "bank.pt")
    write(PREP / "stats.json", stats)
    return stats


def bank_large(rank=4096, snapshots_per_case=2):
    tr_t = np.load(PREP / "train_targets.npy", mmap_mode="r")
    n = tr_t.shape[0]
    rng = np.random.default_rng(20261003)
    sel = [(i, int(t)) for i in range(n) for t in rng.choice(T, snapshots_per_case, replace=False)]
    x = np.empty((len(sel), 2, H * W), np.float32)
    for k, (i, t) in enumerate(sel):
        x[k] = tr_t[i, t].reshape(2, -1)
    stats = read(PREP / "stats.json")
    torch.manual_seed(20261003)
    bank = dict(
        split="train",
        rank=rank,
        basis=torch.zeros(2, rank, H * W),
        mean=torch.zeros(2, H * W),
        nested_from=str(PREP / "bank.pt"),
    )
    for c, scale in enumerate([stats["fields_std"][0], stats["p_res_std"]]):
        y = torch.from_numpy(x[:, c]) / scale
        y = y - y.mean(1, keepdim=True)
        mean = y.mean(0)
        mean = mean - mean.mean()
        y = y - mean
        _, s, v = torch.pca_lowrank(y, q=min(rank + 32, len(sel) - 1), center=False, niter=4)
        q, _ = torch.linalg.qr(v[:, :rank] - v[:, :rank].mean(0), mode="reduced")
        modes = q.T * (H * W) ** 0.5
        orth = float((modes @ modes.T / (H * W) - torch.eye(rank)).abs().max())
        assert orth < 5e-3, orth
        bank["basis"][c] = modes
        bank["mean"][c] = mean
        energy = (s[:rank] ** 2).cumsum(0) / (y**2).sum()
        stats[f"pod_energy_r4096_c{c}"] = {str(r): float(energy[r - 1]) for r in (1024, 2048, 4096)}
        print("channel", c, "orth", orth, stats[f"pod_energy_r4096_c{c}"], flush=True)
    torch.save(bank, PREP / "bank_r4096.pt")
    write(PREP / "stats.json", stats)


def main(argv=None):
    global RAW, PREP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", default="test,val,train")
    parser.add_argument("--pod", action="store_true", help="Fit the 1,024-function POD bank and training scales")
    parser.add_argument("--pod-large", action="store_true", help="Fit the 4,096-function bank; requires --pod output")
    parser.add_argument("--test-pressure-units", choices=("bar", "normalized"), default="bar")
    args = parser.parse_args(argv)
    RAW, PREP = Path(args.raw), Path(args.out)
    PREP.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sizes = dict(train=4500, val=500, test=500)
    for split in filter(None, args.splits.split(",")):
        if split not in sizes:
            raise ValueError(f"Unknown split: {split}")
        marker = PREP / f"{split}_conversion.json"
        raw_names = [f"sg_{split}_a.pt", f"sg_{split}_u.pt", f"dP_{split}_u.pt"]
        output_names = [f"{split}_{kind}.npy" for kind in ("fields", "scalars", "targets")]
        if split == "train":
            output_names.append("time_grid.npy")
        identity = dict(
            raw=file_identity(RAW, raw_names),
            cases=sizes[split],
            source_pressure_units=args.test_pressure_units if split == "test" else "normalized",
            sources_sha256={name: sha(Path(__file__).with_name(name)) for name in ("prepare.py", "data.py")},
        )
        if marker.exists():
            previous = read(marker)
            if previous.get("identity") != identity:
                raise ValueError("Conversion inputs or settings changed; choose a new output directory")
            if previous.get("output_files") != file_identity(PREP, output_names):
                raise ValueError("Converted arrays changed; choose a new output directory")
            continue
        if any((PREP / f"{split}_{kind}.npy").exists() for kind in ("fields", "scalars", "targets")):
            raise ValueError(f"{split} arrays already exist without a conversion record; choose a new output directory")
        result = convert(split, sizes[split], args.test_pressure_units)
        result.update(
            identity=identity,
            output_files=file_identity(PREP, output_names),
            source_pressure_units=args.test_pressure_units if split == "test" else "normalized",
            output_pressure_units="normalized",
            pressure_offset_bar=PRESSURE_OFFSET_BAR,
            pressure_scale_bar=PRESSURE_SCALE_BAR,
        )
        write(marker, result)
    if args.pod:
        stats_and_pod()
    if args.pod_large:
        bank_large()
    print(f"Elapsed seconds: {time.monotonic() - started:.1f}")


if __name__ == "__main__":
    main()
