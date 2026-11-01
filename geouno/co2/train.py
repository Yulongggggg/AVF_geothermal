"""Train the CO2 benchmark models with the shared 20-epoch masked-RMSE recipe."""

import argparse
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Samples
from .evaluate import evaluate
from .models import KINDS, build, count
from .protocol import atomic_checkpoint, freeze, read, sha, write


def masked_rmse(pred, target, mask):
    error = ((pred - target).square() * mask).sum((-2, -1)) / mask.sum((-2, -1)).clamp_min(1)
    return error.clamp_min(1e-12).sqrt().mean(0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--kind", dest="model", choices=KINDS, required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bank", help="Training-only POD bank; defaults to the prepared directory")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        parser.error("epochs and batch must be positive, and workers nonnegative")
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA allocation required; use --device cpu for CPU execution")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    prepared = Path(args.prepared)
    out = Path(args.out) / ("smoke" if args.smoke else "formal") / f"{args.model}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    bank = None
    if args.model.startswith("geouno"):
        bank = Path(args.bank) if args.bank else prepared / ("bank.pt" if args.model == "geouno" else "bank_r4096.pt")
    protocol_sha = freeze(args, out, bank)
    if (out / "complete.json").exists():
        previous = read(out / "complete.json")
        if previous["protocol_sha256"] != protocol_sha or previous["checkpoint_sha256"] != sha(out / "best.pt"):
            raise ValueError("Completed CO2 checkpoint or protocol changed")
        return
    stats = read(prepared / "stats.json")
    scales = stats["fields_std"]
    net = build(args.model, scales, stats["p_res_std"], bank).to(device)
    cases = range(64) if args.smoke else None
    train = Samples(prepared, "train", cases=cases)
    val = Samples(prepared, "val", times=(5, 12, 23), cases=range(32) if args.smoke else None)
    test = Samples(prepared, "test", cases=range(32) if args.smoke else None)
    if not args.smoke and (len(train), len(val), len(test)) != (108000, 1500, 12000):
        raise ValueError("Expected 4,500 training, 500 validation and 500 test cases at the benchmark times")
    epochs = 1 if args.smoke else args.epochs
    loader = DataLoader(
        train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device == "cuda",
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = epochs * len(loader)
    warm = min(500, steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            min(1.0, (step + 1) / max(1, warm))
            * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, steps)))))
        ),
    )
    scale = torch.tensor(scales, device=device)[None, :, None, None]
    write(
        out / "manifest.json",
        dict(
            kind=args.model,
            seed=args.seed,
            epochs=epochs,
            batch=args.batch,
            lr=args.lr,
            parameters=count(net),
            model_config=net.config,
            train_samples=len(train),
            val_samples=len(val),
            test_samples=len(test),
            stats=stats,
            stats_sha256=sha(prepared / "stats.json"),
            bank_sha256=sha(bank) if bank else None,
            protocol_sha256=protocol_sha,
            objective="sum of two fieldwise mean masked RMSEs after training-scale normalization",
            gpu=torch.cuda.get_device_name() if device == "cuda" else "cpu",
        ),
    )
    history, best, first = [], float("inf"), 0
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location=device, weights_only=False)
        if state["protocol_sha256"] != protocol_sha:
            raise ValueError("Checkpoint protocol changed")
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng_cpu"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in state["rng_cuda"]])
        first, best, history = state["epoch"] + 1, state["best"], state["history"]
        if args.workers > 0:
            # Restarting persistent workers needs a worker seed, but continuing
            # workers do not draw one between epochs. Keep the sampler RNG at
            # the saved state while the restarted workers obtain their seed.
            loader.generator = torch.Generator().set_state(torch.get_rng_state())
    # best.pt may already be durable when interruption prevents writing last.pt.
    if (out / "best.pt").exists():
        selected = torch.load(out / "best.pt", map_location=device, weights_only=False)
        if selected["protocol_sha256"] != protocol_sha or selected["kind"] != args.model:
            raise ValueError("Selected CO2 checkpoint belongs to another protocol")
        best = float(selected["selection_score"])
        if not math.isfinite(best):
            raise ValueError("Selected CO2 checkpoint has an invalid score")
    elif (out / "last.pt").exists():
        raise ValueError("Resumed CO2 run is missing best.pt")
    started = time.monotonic()
    for epoch in range(first, epochs):
        net.train()
        totals = torch.zeros(2, device=device)
        batches = 0
        began = time.monotonic()
        for field, scalar, target, _, _ in loader:
            field, scalar, target = [v.to(device, non_blocking=True) for v in (field, scalar, target)]
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                prediction = net(field, scalar)
            field_loss = masked_rmse(prediction.float() / scale, target / scale, field[:, 1:2])
            loss = field_loss.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            totals += field_loss.detach()
            batches += 1
            if args.smoke and batches >= 5:
                break
        validation = evaluate(net, val, device, 64, scales, args.workers)
        row = dict(
            epoch=epoch,
            train_rmse_scaled=(totals / batches).tolist(),
            seconds=time.monotonic() - began,
            lr=scheduler.get_last_lr()[0],
            val=validation,
        )
        history.append(row)
        score = sum(validation["rmse_scaled"])
        state = dict(
            model=net.state_dict(),
            model_config=net.config,
            epoch=epoch,
            kind=args.model,
            seed=args.seed,
            protocol_sha256=protocol_sha,
        )
        if score < best:
            best = score
            atomic_checkpoint(dict(state, selection_score=score), out / "best.pt")
        atomic_checkpoint(
            dict(
                state,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                best=best,
                history=history,
                rng_cpu=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state_all() if device == "cuda" else [],
            ),
            out / "last.pt",
        )
        write(out / "history.json", history)
        print(row, flush=True)
    net.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=False)["model"])
    result = evaluate(net, test, device, 64, scales, args.workers)
    write(out / "test.json", result)
    write(
        out / "complete.json",
        dict(
            status="complete",
            kind=args.model,
            seed=args.seed,
            epochs=epochs,
            smoke=args.smoke,
            best_val=best,
            test=result,
            protocol_sha256=protocol_sha,
            checkpoint_sha256=sha(out / "best.pt"),
            elapsed_hours=(time.monotonic() - started) / 3600,
        ),
    )


if __name__ == "__main__":
    main()
