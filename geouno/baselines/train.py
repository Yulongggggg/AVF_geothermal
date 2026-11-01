"""Train GeoUNO or a geothermal baseline using the shared 99-epoch comparison recipe."""

import argparse
import math
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..training.data import ScenarioPairs, EpochScenarios, collate
from ..training.loss import objective
from ..training.validation import evaluate
from .protocol import atomic_checkpoint
from .protocol import score
from .models import COMPARISON_MODELS, build as build_baseline
from . import protocol
from .protocol import freeze, read, recipe as make_recipe, sha, write


def build(stats, variant):
    scales = stats["fields_std"]
    residual = scales[1] * stats["p_res_std_norm"] if "p_res_std_norm" in stats else stats["p_res_std_mpa"]
    return build_baseline(variant, scales, residual, bank_path=protocol.BANK)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "--variant", dest="variant", choices=COMPARISON_MODELS, required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank", help="Training-only POD bank, required for --model geouno")
    ap.add_argument("--reference")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    if a.variant == "geouno":
        a.variant = "resid64_p1"
    if a.variant == "resid64_p1" and a.bank is None:
        ap.error("--model geouno requires --bank")
    protocol.configure(a.pack, a.out, a.reference, a.bank)
    device = "cpu" if a.cpu else a.device
    out, protocol_sha = freeze(a.variant, a.seed, a.smoke, device)
    recipe = make_recipe(a.variant, a.seed)
    seed = recipe["seed"]
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        done = read(out / "complete.json")
        if done["protocol_sha256"] != protocol_sha or done["checkpoint_sha256"] != sha(out / "best.pt"):
            raise ValueError("Completed baseline checkpoint or protocol changed")
        return
    stats = read(protocol.PACK / "stats.json")
    net = build(stats, a.variant).to(device)
    torch.manual_seed(seed + 1000)
    torch.cuda.manual_seed_all(seed + 1000)
    groups = [dict(params=list(net.parameters()), lr=recipe["lr"])]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    epochs = 2 if a.smoke else recipe["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 0.05 + 0.95 * (1 + math.cos(math.pi * e / epochs)) / 2)
    tr = ScenarioPairs(protocol.PACK, "train", 2 if a.smoke else 0, 2 if a.smoke else 0)
    va = ScenarioPairs(protocol.PACK, "val", 2 if a.smoke else 0, 2 if a.smoke else 0)
    assert not set(tr.pairs.names) & set(va.pairs.names)
    if not a.smoke:
        assert (len(tr), len(tr.pairs), len(va), len(va.pairs)) == (6582, 72939, 542, 6275)
    sampler = EpochScenarios(tr, seed + 3000)
    workers = 0 if device == "cpu" or a.smoke else 10
    loader = DataLoader(
        tr,
        batch_size=recipe["batch_scenarios"],
        sampler=sampler,
        collate_fn=collate,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=device == "cuda",
        prefetch_factor=4 if workers else None,
    )
    times = (10,) if a.smoke else (4, 7, 10)
    if (out / "last.pt").exists():
        old = torch.load(out / "last.pt", map_location=device, weights_only=False)
        assert old["protocol_sha256"] == protocol_sha and old["recipe"] == recipe
        assert read(out / "manifest.json")["smoke"] == a.smoke
        net.load_state_dict(old["model"])
        opt.load_state_dict(old["optimizer"])
        scheduler.load_state_dict(old["scheduler"])
        torch.set_rng_state(old["rng_cpu"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in old["rng_cuda"]])
        start, best, history = old["epoch"] + 1, old["best"], old["history"]
        reference = read(out / "reference_validation.json")
    else:
        # Same selection reference as the published baselines (precomputed, identical in all 15 runs).
        reference = read(protocol.REFERENCE)
        if a.smoke:
            reference = evaluate(build(stats, a.variant).to(device), va, device, times, workers, chunk_size=16)
        write(out / "reference_validation.json", reference)
        write(
            out / "manifest.json",
            dict(
                recipe=recipe,
                smoke=a.smoke,
                protocol_sha256=protocol_sha,
                initialization_checkpoint_sha256=None,
                basis_sha256=sha(protocol.BANK) if a.variant == "resid64_p1" else None,
                parameters_real=sum(v.numel() for v in net.parameters()),
                gpu=torch.cuda.get_device_name() if device == "cuda" else "cpu",
                validation_membership=va.pairs.selected,
                train_members=len(tr.pairs),
                train_scenarios=len(tr),
                train_membership_sha256=sha(protocol.PACK / "meta.csv"),
            ),
        )
        start, best, history = 0, float("inf"), []
    # A best candidate is atomic and can precede the first optimizer checkpoint.
    # Retain it when replaying an interrupted epoch from the previous state.
    if (out / "best.pt").exists():
        selected = torch.load(out / "best.pt", map_location=device, weights_only=False)
        if selected["protocol_sha256"] != protocol_sha or selected["recipe"] != recipe:
            raise ValueError("Selected baseline checkpoint belongs to another protocol")
        best = float(selected["selection_score"])
        if not math.isfinite(best):
            raise ValueError("Selected baseline checkpoint has an invalid score")
        write(out / "validation_best.json", selected["validation"])
    elif (out / "last.pt").exists():
        raise ValueError("Resumed baseline run is missing best.pt")
    for ep in range(start, epochs):
        sampler.epoch = recipe["time_offset"] + ep
        net.train()
        totals, count, members, began = {}, 0, 0, time.monotonic()
        for f, c, y, lengths in loader:
            f, c, y = [v.to(device, non_blocking=True) for v in (f, c, y)]
            opt.zero_grad(set_to_none=True)
            outputs = []
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                outputs.append(net(f, c))  # fast protocol: whole batch, no checkpointing
            pred = torch.cat(outputs).float()
            value, parts = objective(pred, y, lengths, net.scales, net.pressure_residual_scale, weight=0.2, kind="rmse")
            assert torch.isfinite(value)
            value.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5, error_if_nonfinite=True)
            opt.step()
            for k, v in parts.items():
                totals[k] = totals.get(k, 0) + float(v.detach()) * len(lengths)
            count += len(lengths)
            members += sum(lengths)
        assert count == len(tr) and members == len(tr.pairs)
        scheduler.step()
        row = dict(
            epoch=ep,
            seconds=time.monotonic() - began,
            scenarios=count,
            members=members,
            **{k: v / count for k, v in totals.items()},
        )
        state = dict(
            model=net.state_dict(),
            model_config=net.config,
            seed=seed,
            epoch=ep,
            recipe=recipe,
            protocol_sha256=protocol_sha,
        )
        if a.smoke or ep == 0 or (ep + 1) % 11 == 0 or ep + 1 == epochs:
            val = evaluate(net, va, device, times, workers, chunk_size=16)
            row.update(validation=val["summary"])
            val_score = score(val, reference)
            if val_score < best:
                best = val_score
                atomic_checkpoint(dict(state, selection_score=val_score, validation=val), out / "best.pt")
                write(out / "validation_best.json", val)
        history.append(row)
        atomic_checkpoint(
            dict(
                **state,
                optimizer=opt.state_dict(),
                scheduler=scheduler.state_dict(),
                history=history,
                best=best,
                rng_cpu=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state_all() if device == "cuda" else [],
            ),
            out / "last.pt",
        )
        write(out / "history.json", history)
        print(row, flush=True)
    # Same full evaluation as the baselines: all-time validation, then test, for best.pt only.
    from .evaluate import evaluate as full_evaluate

    directory = out / "best"
    directory.mkdir(exist_ok=True)
    full_evaluate(out / "best.pt", "val", directory, protocol.PACK, device, a.smoke)
    full_evaluate(out / "best.pt", "test", directory, protocol.PACK, device, a.smoke)
    write(
        out / "complete.json",
        dict(
            status="complete",
            epochs=epochs,
            smoke=a.smoke,
            protocol_sha256=protocol_sha,
            checkpoint_sha256=sha(out / "best.pt"),
        ),
    )


if __name__ == "__main__":
    main()
