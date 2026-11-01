"""All-time conditional-distribution metrics with fixed scenario/time latent seeds."""

import argparse
import hashlib
from pathlib import Path

import torch

from comparisons.protocol import identity, read, sha, write
from geouno.training.full_metrics import metric_record, summarize
from .data import FieldPairs
from .model import AMPReplica, load


@torch.no_grad()
def evaluate(path, split, folder, pack, device="cuda", smoke=False):
    path, folder, pack = Path(path), Path(folder), Path(pack)
    torch.set_num_threads(1)
    torch.manual_seed(87213)
    target, partial = folder / f"{split}.json", folder / f"{split}.partial.json"
    ident = dict(
        checkpoint_sha256=sha(path),
        split=split,
        smoke=smoke,
        inputs=identity(pack, __file__, device, ("inputs.npy", "index.npy")),
        inference_batch_size=32,
        latent_seed=87213,
    )
    if target.exists():
        saved = read(target)
        if saved.get("identity") != ident:
            raise ValueError("PNO evaluation inputs changed")
        return saved
    ds = FieldPairs(
        pack, split, scenario_limit=2 if smoke else 0, member_limit=4 if smoke else 0, seed=20260913, probabilistic=True
    )
    model = AMPReplica(load(path).to(device).eval(), 1)
    records = []
    if partial.exists():
        saved = read(partial)
        if saved.get("identity") != ident:
            raise ValueError("Partial PNO evaluation inputs changed")
        records = saved["records"]
    done = {(r["scen"], r["time_index"]) for r in records}
    lookup = {int(row): i for i, row in enumerate(ds.rows)}
    times = (4, 10) if smoke else tuple(range(11))
    for si, name in enumerate(ds.names):
        indices = [lookup[r] for r in ds.selected[name]]
        for ti in times:
            if (name, ti) in done:
                continue
            torch.manual_seed(int.from_bytes(hashlib.sha256(f"87213:{name}:{ti}".encode()).digest()[:4], "little"))
            pp, tt = [], []
            for lo in range(0, len(indices), 32):
                batch = indices[lo : lo + 32]
                items = [ds[(i, ti)] for i in batch]
                field, scalar, truth = [torch.stack([v[j] for v in items]).to(device) for j in range(3)]
                prediction = model(field, scalar) * ds.scales.to(device)
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("Nonfinite PNO prediction")
                pp.append(prediction)
                tt.append(truth * ds.scales.to(device))
            records.append(metric_record(torch.cat(pp), torch.cat(tt), ds, indices, ti, True, {}))
        if (si + 1) % 24 == 0 or si + 1 == len(ds.names):
            write(partial, dict(identity=ident, records=records))
    assert len(records) == len(ds.names) * len(times)
    assert {(r["scen"], r["time_index"]) for r in records} == {(s, t) for s in ds.names for t in times}
    assert all(r["rows"] == ds.selected[r["scen"]] for r in records)
    result = dict(
        status="complete",
        kind="pno",
        split=split,
        smoke=smoke,
        checkpoint=str(path),
        checkpoint_sha256=sha(path),
        identity=ident,
        task="conditional_distribution",
        scenario_count=len(ds.names),
        actual_members=len(ds),
        summary=summarize(records),
        records=records,
    )
    write(target, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args(argv)
    evaluate(a.checkpoint, a.split, a.out, a.pack, a.device, a.smoke)


if __name__ == "__main__":
    main()
