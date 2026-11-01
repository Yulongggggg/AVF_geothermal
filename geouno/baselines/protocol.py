"""Data paths and the shared 99-epoch geothermal comparison recipe."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from ..artifacts import resource
from .models import BASELINES

SEEDS = (20260915, 20260916, 20260917)
PACK = OUT = BANK = None
REFERENCE = resource("training_final_20261015/reference_validation.json")


def configure(pack, out, reference=None, bank=None):
    global PACK, OUT, REFERENCE, BANK
    PACK, OUT = Path(pack), Path(out)
    BANK = Path(bank) if bank is not None else None
    REFERENCE = Path(reference) if reference else resource("training_final_20261015/reference_validation.json")


def recipe(model, seed):
    if model == "geouno":
        model = "resid64_p1"
    if model not in (*BASELINES, "resid64_p1"):
        raise ValueError(f"Unknown baseline: {model}")
    return dict(
        variant=model, seed=seed, initialization="scratch", epochs=99, batch_scenarios=8, time_offset=0, lr=0.0003
    )


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_checkpoint(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def score(result, reference):
    return float(
        np.mean(
            [
                result["summary"][t][k] / reference["summary"][t][k]
                for t in ("10", "all_evaluated_times")
                for k in ("T_rmse", "P_rmse")
            ]
        )
    )


def file_identity(directory, names):
    """Identify large input arrays without rereading the full simulation corpus."""
    directory = Path(directory).resolve()
    files = {}
    for name in names:
        stat = (directory / name).stat()
        files[name] = dict(bytes=stat.st_size, modified_ns=stat.st_mtime_ns)
    return dict(directory=str(directory), files=files)


def freeze(model, seed, smoke=False, device="cuda"):
    if model == "geouno":
        model = "resid64_p1"
    if model == "resid64_p1" and BANK is None:
        raise ValueError("The common 99-epoch GeoUNO recipe requires a POD bank")
    package = Path(__file__).resolve().parent.parent
    paths = list((package / "baselines").glob("*.py"))
    paths.append(package / "common99.py")
    if model == "resid64_p1":
        paths.extend((package / "model").glob("*.py"))
    paths += [
        package / relative
        for relative in (
            "training/data.py",
            "training/loss.py",
            "training/validation.py",
            "training/metrics.py",
            "training/full_metrics.py",
            "inputs.py",
        )
    ]
    value = dict(
        recipe=recipe(model, seed),
        smoke=smoke,
        device=device,
        torch=str(torch.__version__),
        numpy=str(np.__version__),
        training_scenarios=6582,
        training_members=72939,
        validation_scenarios=542,
        validation_members=6275,
        test_scenarios=618,
        test_members=8621,
        evaluation_times=11,
        validation_times=[4, 7, 10],
        objective="paired RMSE + 0.2 MMD squared",
        weight_decay=1e-4,
        gradient_clip=5,
        evaluation_precision="BF16 autocast on CUDA",
        table_metrics_precision="FP32; TF32 disabled" if model == "resid64_p1" else "BF16 autocast; TF32 disabled",
        table_inference_batch_size=16 if model == "resid64_p1" else 32,
        reference_sha256=sha(REFERENCE),
        basis_sha256=sha(BANK) if model == "resid64_p1" else None,
        stats_sha256=sha(PACK / "stats.json"),
        meta_sha256=sha(PACK / "meta.csv"),
        times_sha256=sha(PACK / "_ref_times.npy"),
        pack_arrays=file_identity(PACK, ("real.npy", "fields.npy", "scal.npy")),
        sources_sha256={p.relative_to(package).as_posix(): sha(p) for p in sorted(paths)},
    )
    directory = OUT / ("smoke" if smoke else "formal") / f"{model}_s{seed}"
    target = directory / "protocol.json"
    if target.exists():
        old = read(target)
        changed = [key for key in sorted(set(old) | set(value)) if old.get(key) != value.get(key)]
        if changed:
            raise ValueError("Baseline inputs or protocol changed: " + ", ".join(changed))
    else:
        write(target, value)
    return directory, sha(target)
