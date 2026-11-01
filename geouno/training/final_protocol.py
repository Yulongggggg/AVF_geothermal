"""Portable paths and protocol for the 15 October 2026 final model."""

import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from ..artifacts import resource

PACK = BANK = OUT = None
REFERENCE = resource("training_final_20261015/reference_validation.json")
BANK_SHA = "296b9975a490526b30840e45c01abe5c256d2af6e0fa7f60fa14fa6206d7a997"
SEEDS = (20260915, 20260925, 20260926)
STAGES = dict(main=dict(epochs=99, lr=3e-4), temperature=dict(epochs=44, lr=1e-5), joint=dict(epochs=22, lr=1e-5))
BATCH_SCENARIOS = 8


def configure(pack, bank, out, reference=None):
    global PACK, BANK, OUT, REFERENCE
    PACK, BANK, OUT = Path(pack), Path(bank), Path(out)
    REFERENCE = Path(reference) if reference else resource("training_final_20261015/reference_validation.json")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_checkpoint(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


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


def folder(arch, seed, smoke=False):
    return OUT / ("smoke" if smoke else "formal") / f"{arch}_s{seed}"


def freeze(arch):
    if arch != "resid64_p1":
        raise ValueError("The paper recipe uses resid64_p1")
    pkg = Path(__file__).resolve().parent.parent
    sources = {
        p.relative_to(pkg).as_posix(): sha(p) for sub in ("training", "model") for p in sorted((pkg / sub).glob("*.py"))
    }
    sources["inputs.py"] = sha(pkg / "inputs.py")
    arrays = {}
    for name in ("real.npy", "fields.npy", "scal.npy"):
        stat = (PACK / name).stat()
        arrays[name] = dict(bytes=stat.st_size, modified_ns=stat.st_mtime_ns)
    value = dict(
        architecture=arch,
        stages=STAGES,
        batch_scenarios=BATCH_SCENARIOS,
        seeds=list(SEEDS),
        training_records=72939,
        training_scenarios=6582,
        nominal_records_used=False,
        basis_sha256=sha(BANK),
        reference_validation_sha256=sha(REFERENCE),
        stats_sha256=sha(PACK / "stats.json"),
        meta_sha256=sha(PACK / "meta.csv"),
        times_sha256=sha(PACK / "_ref_times.npy"),
        pack_path=str(PACK.resolve()),
        pack_arrays=arrays,
        sources_sha256=sources,
        final_evaluation="FP32, TF32 disabled",
    )
    target = OUT / f"protocol_{arch}.json"
    if target.exists():
        old = read(target)
        changed = [key for key in sorted(set(old) | set(value)) if old.get(key) != value.get(key)]
        if changed:
            raise ValueError("Training inputs or protocol changed since this run began: " + ", ".join(changed))
        return old
    write(target, value)
    return value
