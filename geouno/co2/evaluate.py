"""Masked CO2 benchmark errors in normalized and physical pressure units."""

import torch
from torch.utils.data import DataLoader

from .data import PRESSURE_SCALE_BAR, Samples
from .models import load
from .protocol import write, sha


@torch.no_grad()
def evaluate(net, data, device, batch, scales, workers):
    net.eval()
    loader = DataLoader(data, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=device == "cuda")
    denominator = torch.zeros(2, device=device)
    squares = torch.zeros(2, device=device)
    relative = []
    count = 0
    for field, scalar, target, _, _ in loader:
        field, scalar, target = field.to(device), scalar.to(device), target.to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            pred = net(field, scalar).float()
        mask = field[:, 1:2]
        error = ((pred - target).square() * mask).sum((-2, -1))
        truth = (target.square() * mask).sum((-2, -1))
        relative.append(error.clamp_min(1e-30).sqrt() / truth.clamp_min(1e-30).sqrt())
        squares += error.sum(0)
        denominator += mask.sum((-2, -1)).sum(0).expand(2)
        count += len(field)
    relative = torch.cat(relative)
    rmse = (squares / denominator).sqrt()
    normalized = rmse.tolist()
    return dict(
        samples=count,
        rel_l2=relative.mean(0).tolist(),
        rel_l2_median=relative.median(0).values.tolist(),
        rmse=normalized,
        rmse_scaled=(rmse / torch.tensor(scales, device=device)).tolist(),
        physical_rmse=[normalized[0], normalized[1] * PRESSURE_SCALE_BAR],
        physical_units=["saturation fraction", "bar"],
        relative_l2_pressure_target="offset-normalized pressure",
    )


def main(argv=None):
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out", required=True, help="Output JSON file")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    data = Samples(args.prepared, "test", cases=range(2) if args.smoke else None)
    if not args.smoke and len(data) != 12000:
        raise ValueError("The benchmark test set must contain 500 cases and 24 times")
    result = evaluate(load(args.checkpoint).to(args.device), data, args.device, args.batch, data.scales, args.workers)
    result.update(
        checkpoint_sha256=sha(args.checkpoint), stats_sha256=sha(Path(args.prepared) / "stats.json"), smoke=args.smoke
    )
    write(args.out, result)
    print(result)


if __name__ == "__main__":
    main()
