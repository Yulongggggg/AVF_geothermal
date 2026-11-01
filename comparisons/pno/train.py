"""Train the conditional PNO with the 44-epoch supplementary protocol."""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from comparisons.protocol import atomic_checkpoint, freeze, identity, read, sha, write
from .data import EpochPairs, FieldPairs
from .model import AMPReplica, Predictor, loss
from .validation import validation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, choices=(20260913, 20260914), default=20260913)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    device = args.device
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    if not args.smoke and (device != "cuda" or torch.cuda.device_count() != 2):
        raise RuntimeError("The published PNO training protocol uses exactly two visible GPUs")
    pack = Path(args.pack)
    epochs, batch_size, workers = (2, 4, 0) if args.smoke else (44, 64, 4)
    recipe = dict(
        seed=args.seed,
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
        lr=0.0003,
        latent_draws=4,
        weight_decay=0.0001,
        eta_min_fraction=0.03,
        gradient_clip=5,
        validation_times=[4, 7, 10],
        validation_seed=97113,
        validation_scenarios=2 if args.smoke else 64,
        validation_members=2 if args.smoke else 4,
        training_gpus=torch.cuda.device_count() if device == "cuda" else 0,
    )
    out, protocol_sha = freeze(
        Path(args.out) / ("smoke" if args.smoke else "formal") / f"pno_s{args.seed}",
        dict(recipe=recipe, smoke=args.smoke, inputs=identity(pack, __file__, device, ("inputs.npy", "index.npy"))),
    )
    if (out / "completion.json").exists():
        done = read(out / "completion.json")
        if done["protocol_sha256"] != protocol_sha or done["checkpoint_sha256"] != sha(out / "best.pt"):
            raise ValueError("Completed PNO checkpoint or protocol changed")
        return
    tr = FieldPairs(
        pack,
        "train",
        scenario_limit=2 if args.smoke else 0,
        member_limit=2 if args.smoke else 0,
        seed=20260913,
        probabilistic=True,
    )
    va = FieldPairs(
        pack,
        "val",
        scenario_limit=2 if args.smoke else 64,
        member_limit=2 if args.smoke else 4,
        seed=20260913,
        probabilistic=True,
    )
    assert not set(tr.names) & set(va.names)
    net = Predictor().to(device)
    wrapped = AMPReplica(net, 4)
    model = torch.nn.DataParallel(wrapped) if device == "cuda" and torch.cuda.device_count() > 1 else wrapped
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0003, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.0003 * 0.03)
    sampler = EpochPairs(tr, 0, args.seed)
    loader = DataLoader(
        tr,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=device == "cuda",
    )
    manifest = dict(
        kind="pno",
        seed=args.seed,
        recipe=recipe,
        protocol_sha256=protocol_sha,
        train_scenarios=len(tr.names),
        train_actual_members=len(tr),
        validation_membership=va.selected,
        parameters_real=sum(p.numel() for p in net.parameters()),
        normalization=tr.stats["fields_std"],
        validation_rule="Proper composite energy score with four fixed latent draws",
    )
    if (out / "manifest.json").exists() and read(out / "manifest.json") != manifest:
        raise ValueError("PNO manifest changed")
    write(out / "manifest.json", manifest)
    start, best, history = 0, float("inf"), []
    resumed = (out / "last.pt").exists()
    if resumed:
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        if ck["protocol_sha256"] != protocol_sha:
            raise ValueError("PNO optimizer checkpoint belongs to another protocol")
        net.load_state_dict(ck["model"], strict=True)
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        torch.set_rng_state(ck["rng_cpu"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([r.cpu() for r in ck["rng_cuda"]])
    if (out / "best.pt").exists():
        selected = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        if selected["protocol_sha256"] != protocol_sha:
            raise ValueError("PNO selected checkpoint belongs to another protocol")
        best = float(selected["selection_score"])
        if not math.isfinite(best):
            raise ValueError("Invalid selected PNO score")
        write(out / "validation_best.json", selected["validation"])
    elif resumed:
        raise ValueError("Resumed PNO run is missing best.pt")

    def checkpoint(epoch):
        return dict(
            kind="pno",
            model=net.state_dict(),
            seed=args.seed,
            epochs=epochs,
            epoch=epoch,
            best=best,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if device == "cuda" else [],
            manifest=manifest,
            protocol_sha256=protocol_sha,
            history=history,
        )

    for ep in range(start, epochs):
        sampler.epoch = ep
        model.train()
        total, count, started = 0.0, 0, time.monotonic()
        # Persistent workers reuse their original iterator in an uninterrupted run.
        # Recreating them after resume must not consume an extra model RNG draw.
        rng_before_iterator = torch.get_rng_state() if resumed and workers and ep == start else None
        batches = iter(loader)
        if rng_before_iterator is not None:
            torch.set_rng_state(rng_before_iterator)
        for field, scalar, truth, idx, _ti in batches:
            field, scalar, truth = field.to(device), scalar.to(device), truth.to(device)
            weights = torch.as_tensor(tr.weights[idx.numpy()], device=device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(field, scalar)
            value = loss(pred, truth, field, weights, True)
            if not torch.isfinite(value):
                raise FloatingPointError("Nonfinite PNO loss")
            value.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            total += value.item() * len(field)
            count += len(field)
        scheduler.step()
        record = dict(epoch=ep, loss=total / count, actual_member_time_pairs=count, seconds=time.monotonic() - started)
        if ep == 0 or (ep + 1) % 4 == 0 or ep + 1 == epochs:
            metrics = validation(model, va, device, batch_size, workers)
            score = metrics["selection_score"]
            record["validation"] = metrics
            if score < best:
                best = score
                atomic_checkpoint(dict(checkpoint(ep), selection_score=score, validation=metrics), out / "best.pt")
                write(out / "validation_best.json", metrics)
        history.append(record)
        atomic_checkpoint(checkpoint(ep), out / "last.pt")
        write(out / "history.json", history)
        print(record, flush=True)
    write(
        out / "completion.json",
        dict(
            status="training_complete",
            best=best,
            epochs=epochs,
            protocol_sha256=protocol_sha,
            checkpoint_sha256=sha(out / "best.pt"),
        ),
    )


if __name__ == "__main__":
    main()
