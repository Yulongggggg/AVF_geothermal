# SPDX-License-Identifier: GPL-3.0-only
"""Full-trajectory validation and test evaluation of the geothermal PCNOs."""

from pathlib import Path

import torch
from geouno.training.data import FieldPairs
from geouno.training.full_metrics import metric_record, summarize, load_scope
from geouno.training.evaluate import evaluation_identity as input_identity
from .model import load
from geouno.baselines.protocol import read, write, sha


def evaluate(path, split, folder, pack, device="cuda", smoke=False):
    path, folder, pack = Path(path), Path(folder), Path(pack)
    target = folder / f"{split}.json"
    partial = folder / f"{split}.partial.json"
    digest = sha(path)
    ds = FieldPairs(pack, split, scenario_limit=2 if smoke else 0, member_limit=2 if smoke else 0, seed=20260913)
    identity = dict(
        checkpoint_sha256=digest,
        stats_sha256=sha(pack / "stats.json"),
        meta_sha256=sha(pack / "meta.csv"),
        times_sha256=sha(pack / "_ref_times.npy"),
        evaluator_sha256=sha(Path(__file__)),
        split=split,
        device=device,
        smoke=smoke,
        inputs=input_identity(path, ds, split, smoke, device != "cuda", device),
        comparison_sources_sha256={p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
    )
    if target.exists():
        saved = read(target)
        if saved.get("identity") != identity:
            raise ValueError("Cached PCNO evaluation belongs to different inputs or settings")
        return saved
    net = load(path).to(device).eval()
    scope = load_scope()
    rows = []
    if partial.exists():
        previous = read(partial)
        if previous.get("identity") != identity:
            raise ValueError("Partial PCNO evaluation belongs to different inputs or settings")
        rows = previous["records"]
    done = {(r["scen"], r["time_index"]) for r in rows}
    lookup = {int(row): i for i, row in enumerate(ds.rows)}
    with torch.no_grad():
        for si, name in enumerate(ds.names):
            indices = [lookup[row] for row in ds.selected[name]]
            for ti in (10,) if smoke else range(11):
                if (name, ti) in done:
                    continue
                pred, truth = [], []
                for start in range(0, len(indices), 4):
                    items = [ds[(i, ti)] for i in indices[start : start + 4]]
                    f, c, y = [torch.stack([item[k] for item in items]).to(device) for k in (0, 1, 2)]
                    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                        p = net(f, c)
                    assert torch.isfinite(p).all()
                    pred.append(p.float())
                    truth.append(y * ds.scales.to(device))
                rows.append(metric_record(torch.cat(pred), torch.cat(truth), ds, indices, ti, False, scope))
            if (si + 1) % 24 == 0 or si + 1 == len(ds.names):
                write(partial, dict(checkpoint_sha256=digest, identity=identity, records=rows))
    expected_times = {10} if smoke else set(range(11))
    assert len(rows) == len(ds.names) * len(expected_times)
    assert {(r["scen"], r["time_index"]) for r in rows} == {(s, t) for s in ds.names for t in expected_times}
    assert all(r["rows"] == ds.selected[r["scen"]] for r in rows)
    result = dict(
        status="complete",
        split=split,
        checkpoint=str(path),
        checkpoint_sha256=digest,
        identity=identity,
        smoke=smoke,
        scenario_count=len(ds.names),
        actual_members=len(ds),
        records=rows,
        summary=summarize(rows),
    )
    write(target, result)
    return result


def main(argv=None):
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    evaluate(Path(args.checkpoint), args.split, folder, Path(args.pack), args.device, args.smoke)


if __name__ == "__main__":
    main()
