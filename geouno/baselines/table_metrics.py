"""Scenario-weighted field, relative-L2 and near-well metrics used in Table 1."""

import collections
import math
from pathlib import Path
import statistics
import time

import torch

from ..training.data import FieldPairs
from ..training.evaluate import evaluation_identity as input_identity
from .models import load
from .protocol import read, sha, write


def summarize(records):
    cases = collections.defaultdict(dict)
    for row in records:
        if row["time_index"] in cases[row["scenario"]]:
            raise ValueError("Duplicate scenario-time record")
        cases[row["scenario"]][row["time_index"]] = row
    times = sorted(next(iter(cases.values())))
    if any(sorted(case) != times for case in cases.values()):
        raise ValueError("Scenarios do not contain the same forecast times")
    scores = {}
    for horizon, selected_times in (("all", times), ("ten", [10])):
        for field in ("T", "P"):
            for metric in ("field_rmse", "nearwell_rmse"):
                scores[f"{horizon}_{field}_{metric}"] = statistics.mean(
                    statistics.mean(case[t][field + "_" + metric] for t in selected_times) for case in cases.values()
                )
            scores[f"{horizon}_{field}_relative"] = statistics.mean(
                math.sqrt(
                    math.fsum(case[t][field + "_field_rmse"] ** 2 for t in selected_times)
                    / math.fsum(case[t][field + "_truth_rms"] ** 2 for t in selected_times)
                )
                for case in cases.values()
            )
    return scores


def evaluate(checkpoint, pack, destination, device="cuda", smoke=False, split="test"):
    checkpoint, pack, destination = Path(checkpoint), Path(pack), Path(destination)
    data = FieldPairs(pack, split, scenario_limit=2 if smoke else 0, member_limit=2 if smoke else 0, seed=20260913)
    config = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_config"]
    is_geouno = config["architecture"] == "geouno_simple"
    # Table 1 re-evaluates GeoUNO in FP32, using 16 members per inference call.
    # The six baselines use BF16 on CUDA and batches of 32 members.
    amp = device == "cuda" and not is_geouno
    batch = 16 if is_geouno else 32
    identity = dict(
        checkpoint_sha256=sha(checkpoint),
        evaluator_sha256=sha(Path(__file__)),
        stats_sha256=sha(pack / "stats.json"),
        meta_sha256=sha(pack / "meta.csv"),
        times_sha256=sha(pack / "_ref_times.npy"),
        precision="bf16 autocast cuda" if amp else "fp32 " + device,
        inference_batch_size=batch,
        matmul_tf32=False,
        cudnn_tf32=False,
        split=split,
        smoke=smoke,
        inputs=input_identity(checkpoint, data, split, smoke, not amp, device),
        baseline_sources_sha256={p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    )
    records = []
    if destination.exists():
        saved = read(destination)
        if saved.get("identity") != identity:
            raise ValueError("Saved table metrics belong to different inputs or settings")
        records = saved["records"]
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    net = load(checkpoint).to(device).eval().requires_grad_(False)
    if not smoke:
        expected = (618, 8621) if split == "test" else (542, 6275)
        assert (len(data.names), len(data)) == expected
    lookup = {int(row): i for i, row in enumerate(data.rows)}
    done = {(row["scenario"], row["time_index"]) for row in records}
    times = [10] if smoke else list(range(11))
    scales = data.scales.to(device)
    started = time.monotonic()
    with torch.inference_mode():
        for si, name in enumerate(data.names):
            rows = data.selected[name]
            indices = [lookup[row] for row in rows]
            control = data.scal[rows[0]]
            masks = [
                torch.hypot(data.x - float(control[3]), data.y - float(control[4])) <= 6,
                torch.hypot(data.x, data.y + 250) <= 6,
            ]
            assert all(mask.any() and (data.inside[mask] == 1).all() for mask in masks)
            masks = [mask.to(device) for mask in masks]
            for ti in times:
                if (name, ti) in done:
                    continue
                predictions, truths = [], []
                for start in range(0, len(indices), batch):
                    items = [data[(i, ti)] for i in indices[start : start + batch]]
                    fields, scalars, truth = [torch.stack([item[k] for item in items]).to(device) for k in (0, 1, 2)]
                    with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                        pred = net(fields, scalars)
                    predictions.append(pred.float())
                    truths.append(truth * scales)
                pred, truth = torch.cat(predictions).double(), torch.cat(truths).double()
                assert torch.isfinite(pred).all()
                row = dict(
                    scenario=name,
                    time_index=ti,
                    members=len(rows),
                    rows=rows,
                    producer_cells=int(masks[0].sum()),
                    injector_cells=int(masks[1].sum()),
                )
                for channel, label in enumerate(("T", "P")):
                    error = pred[:, channel] - truth[:, channel]
                    row[label + "_field_rmse"] = float(error[:, 16:144].square().mean().sqrt())
                    row[label + "_truth_rms"] = float(truth[:, channel, 16:144].square().mean().sqrt())
                    row[label + "_nearwell_rmse"] = float(error[:, masks[channel]].square().mean().sqrt())
                records.append(row)
            if (si + 1) % 24 == 0 or si + 1 == len(data.names):
                write(
                    destination,
                    dict(
                        status="partial",
                        identity=identity,
                        checkpoint=str(checkpoint),
                        scenario_count=si + 1,
                        records=records,
                        seconds=time.monotonic() - started,
                    ),
                )
    expected_ids = {(name, ti): tuple(data.selected[name]) for name in data.names for ti in times}
    actual_ids = {(row["scenario"], row["time_index"]): tuple(row["rows"]) for row in records}
    assert len(records) == len(actual_ids) == len(expected_ids) and actual_ids == expected_ids
    result = dict(
        status="complete",
        identity=identity,
        checkpoint=str(checkpoint),
        scenario_count=len(data.names),
        actual_members=len(data),
        records=records,
        scores=summarize(records),
        seconds=time.monotonic() - started,
    )
    write(destination, result)
    return result


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--out", required=True, help="Output JSON file")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    result = evaluate(args.checkpoint, args.pack, args.out, args.device, args.smoke, args.split)
    print(result["scores"])


if __name__ == "__main__":
    main()
