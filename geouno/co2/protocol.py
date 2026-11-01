"""Protocol records for the 20-epoch CO2 benchmark comparison."""

from pathlib import Path

import numpy as np
import torch

from ..baselines.protocol import read, write, sha, atomic_checkpoint, file_identity

__all__ = ["read", "write", "sha", "atomic_checkpoint", "freeze"]


def freeze(args, directory, bank):
    package = Path(__file__).resolve().parent.parent
    paths = [p for sub in ("co2", "baselines", "model") for p in (package / sub).glob("*.py")]
    prepared = Path(args.prepared)
    value = dict(
        model=args.model,
        seed=args.seed,
        epochs=1 if args.smoke else args.epochs,
        batch=args.batch,
        learning_rate=args.lr,
        weight_decay=1e-4,
        gradient_clip=5,
        smoke=args.smoke,
        device=args.device,
        workers=args.workers,
        torch=str(torch.__version__),
        numpy=str(np.__version__),
        stats_sha256=sha(prepared / "stats.json"),
        time_grid_sha256=sha(prepared / "time_grid.npy"),
        bank_sha256=sha(bank) if bank else None,
        prepared_arrays=file_identity(prepared, [f"{split}_{kind}.npy" for split in ("train", "val", "test")
                                                 for kind in ("fields", "scalars", "targets")]),
        sources_sha256={p.relative_to(package).as_posix(): sha(p) for p in sorted(paths)},
    )
    path = directory / "protocol.json"
    if path.exists():
        old = read(path)
        changed = [key for key in sorted(set(value) | set(old)) if old.get(key) != value.get(key)]
        if changed:
            raise ValueError("CO2 inputs or protocol changed: " + ", ".join(changed))
    else:
        write(path, value)
    return sha(path)
