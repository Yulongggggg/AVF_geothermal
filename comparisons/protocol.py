"""Identities and atomic state for the supplementary comparison programs."""

from pathlib import Path

import numpy as np
import torch

from geouno.baselines.protocol import atomic_checkpoint as atomic_checkpoint, file_identity, read, sha, write


def identity(pack, module, device, extra_arrays=()):
    pack = Path(pack)
    package = Path(__file__).resolve().parents[1] / "geouno"
    source = list(Path(module).parent.glob("*.py")) + [Path(__file__)]
    source += list((package / "training").glob("*.py"))
    source += [package / "inputs.py", package / "baselines/protocol.py"]
    return dict(
        device=device,
        torch=str(torch.__version__),
        numpy=str(np.__version__),
        meta_sha256=sha(pack / "meta.csv"),
        stats_sha256=sha(pack / "stats.json"),
        times_sha256=sha(pack / "_ref_times.npy"),
        arrays=file_identity(pack, ("real.npy", "scal.npy", "fields.npy", *extra_arrays)),
        sources_sha256={str(p.relative_to(package.parent)): sha(p) for p in sorted(source)},
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
    )


def freeze(directory, value):
    directory = Path(directory)
    path = directory / "protocol.json"
    if path.exists():
        old = read(path)
        changed = sorted(k for k in set(value) | set(old) if value.get(k) != old.get(k))
        if changed:
            raise ValueError("Comparison protocol or inputs changed: " + ", ".join(changed))
    else:
        write(path, value)
    return directory, sha(path)
