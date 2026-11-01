"""Recompute current manuscript tables and abstract claims from bundled Source Data."""

import csv
import gzip
import hashlib
import json
import math
import statistics as st
from collections import defaultdict

from .artifacts import resource

SOURCE = resource("source_data")
FINAL = "final_model_20261015"
MODEL_NAMES = dict(
    deeponet="DeepONet", fno="FNO", ufno="U-FNO", unet="U-Net", resnet="ResNet",
    pcno="PCNO", attention_unet="Attn U-Net", geouno="GeoUNO"
)
COMMON_MODELS = ("deeponet", "fno", "ufno", "unet", "resnet", "attention_unet", "geouno")
COMMON_SEEDS = (20260915, 20260916, 20260917)


def read(name):
    return json.loads((SOURCE / FINAL / name).read_text())


def _mean_sd(values):
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError("Invalid Source Data")
    return st.mean(values), st.stdev(values)


def summarize(runs):
    return tuple(_mean_sd(list(c)) for c in zip(*runs, strict=True))


def common_training_runs():
    """Read all 21 common-protocol runs on one 618-scenario, 11-time test cohort."""
    folder = SOURCE / "common99"
    metadata = json.loads((folder / "sources.json").read_text())
    expected_runs = {(model, seed) for model in COMMON_MODELS for seed in COMMON_SEEDS}
    sources = metadata["sources"]
    recorded_runs = {(r["model"], r["seed"]) for r in sources}
    if (metadata.get("complete") is not True or metadata.get("completed_runs") != 21
            or len(sources) != 21 or recorded_runs != expected_runs):
        raise ValueError("Common comparison requires all seven models and three completed seeds per model")
    for name in ("cohort.json.gz", "scenario_time_scores.csv.gz"):
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != metadata["files"][name]:
            raise ValueError(f"Common-protocol Source Data checksum differs: {name}")
    with gzip.open(folder / "cohort.json.gz", "rt") as handle:
        cohort = json.load(handle)
    expected = {(name, int(t)): tuple(members) for name, t, members in cohort}
    scenarios = sorted({name for name, _ in expected})
    if (len(expected) != len(cohort) or len(scenarios) != 618
            or set(expected) != {(name, t) for name in scenarios for t in range(11)}):
        raise ValueError("Incomplete or duplicate common test cohort")
    for t in range(11):
        members = [member for (name, time), rows in expected.items() if time == t for member in rows]
        if len(members) != 8621 or len(set(members)) != 8621:
            raise ValueError("Incorrect common test-member count")
    if any(expected[name, t] != expected[name, 0] for name in scenarios for t in range(11)):
        raise ValueError("Common test membership changes over time")
    columns = ("T_field_rmse", "P_field_rmse", "T_nearwell_rmse", "P_nearwell_rmse",
               "T_truth_rms", "P_truth_rms")
    runs, truth = defaultdict(dict), {}
    with gzip.open(folder / "scenario_time_scores.csv.gz", "rt") as handle:
        for row in csv.DictReader(handle):
            run = row["model"], int(row["seed"])
            key = row["scenario"], int(row["time_index"])
            if (run not in expected_runs or key not in expected or key in runs[run]
                    or int(row["members"]) != len(expected[key])):
                raise ValueError("Duplicate or mismatched common test record")
            if int(row["injector_cells"]) != 4 or not 1 <= int(row["producer_cells"]) <= 4:
                raise ValueError("Incorrect near-well masks")
            values = tuple(float(row[c]) for c in columns)
            if not all(math.isfinite(v) and v >= 0 for v in values) or min(values[4:]) <= 0:
                raise ValueError("Invalid common test metric")
            if key in truth and truth[key] != values[4:]:
                raise ValueError("Common test truth norms differ between runs")
            truth[key] = values[4:]
            runs[run][key] = values
    if set(runs) != expected_runs or any(set(rows) != set(expected) for rows in runs.values()):
        raise ValueError("Incomplete common test results")
    out = defaultdict(list)
    for model in COMMON_MODELS:
        for seed in COMMON_SEEDS:
            rows, scores = runs[model, seed], {}
            for horizon, times in (("all", range(11)), ("ten", (10,))):
                rmse = [st.mean(st.mean(rows[s, t][i] for t in times) for s in scenarios) for i in range(4)]
                rel = [
                    st.mean(math.sqrt(math.fsum(rows[s, t][i] ** 2 for t in times)
                                      / math.fsum(rows[s, t][i + 4] ** 2 for t in times)) for s in scenarios)
                    for i in range(2)
                ]
                scores[horizon] = tuple(rmse[:2] + rel + rmse[2:])
            out[model].append(scores)
    return dict(out)


def table1(runs=None):
    runs = common_training_runs() if runs is None else runs
    return {MODEL_NAMES[model]: summarize([r["all"] for r in runs[model]]) for model in COMMON_MODELS}


def supplementary_table19(runs=None):
    runs = common_training_runs() if runs is None else runs
    out = {}
    for model in COMMON_MODELS:
        for metric, offset in (("RMSE", 0), ("relative L2", 2)):
            values = [tuple(r[horizon][i + offset] for horizon in ("all", "ten") for i in range(2))
                      for r in runs[model]]
            out[MODEL_NAMES[model] + " / " + metric] = summarize(values)
    return out


def geouno_test_runs():
    """Equal-scenario/time RMSE and scenario-pooled relative L2, three final seeds."""
    folder = SOURCE / "table1_three_seed_20261015"
    with gzip.open(folder / "cohort.json.gz", "rt") as handle:
        cohort = json.load(handle)
    expected = {(name, int(t)): len(members) for name, t, members in cohort}
    if len(expected) != 618 * 11 or len(cohort) != len(expected):
        raise ValueError("Incomplete or duplicate test cohort")
    if any(sum(n for (_, t), n in expected.items() if t == ti) != 8621 for ti in range(11)):
        raise ValueError("Incorrect test-member count")
    cols = ("T_field_rmse", "P_field_rmse", "T_nearwell_rmse", "P_nearwell_rmse", "T_truth_rms", "P_truth_rms")
    runs = defaultdict(dict)
    with gzip.open(folder / "scenario_time_scores.csv.gz", "rt") as handle:
        for row in csv.DictReader(handle):
            seed, key = int(row["seed"]), (row["scenario"], int(row["time_index"]))
            if key in runs[seed] or expected.get(key) != int(row["members"]):
                raise ValueError("Duplicate or mismatched test record")
            values = tuple(float(row[k]) for k in cols)
            if not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError("Invalid test metric")
            runs[seed][key] = values
    if set(runs) != {20260915, 20260925, 20260926} or any(set(r) != set(expected) for r in runs.values()):
        raise ValueError("Incomplete three-seed test results")
    out = []
    for seed in sorted(runs):
        rows = runs[seed]
        rmse = [st.mean(v[i] for v in rows.values()) for i in range(4)]
        scenarios = sorted({key[0] for key in rows})
        rel = [
            st.mean(
                math.sqrt(
                    sum(rows[(s, t)][i] ** 2 for t in range(11)) / sum(rows[(s, t)][i + 4] ** 2 for t in range(11))
                )
                for s in scenarios
            )
            for i in range(2)
        ]
        out.append(tuple(rmse[:2] + rel + rmse[2:]))
    return out


def supplementary_table1():
    out = {MODEL_NAMES[k]: summarize([r["values"] for r in runs]) for k, runs in read("baseline_per_seed.json").items()}
    out["GeoUNO"] = summarize(geouno_test_runs())
    return out


def table2():
    groups = read("inference_numbers.json")["groups"]
    methods = {
        "Prior": "prior",
        "Static conditioning": "static_prior",
        "MAP": "multiwell_opt17/map",
        "ES-MDA": "multiwell_opt17/esmda",
        "NUTS": "nuts",
    }
    columns = ("logk_RMSE", "T_field_RMSE", "p_field_RMSE", "T_nearwell_RMSE", "p_nearwell_RMSE")
    cases = [f"test_{i:02d}" for i in range(1, 25)]
    return {"Table 2 (24 reservoirs)": {
        label: summarize([[groups[key][case][col] for col in columns] for case in cases])
        for label, key in methods.items()
    }}


def co2():
    out = {
        MODEL_NAMES[k]: summarize([r["values"] for r in runs]) for k, runs in read("co2_baseline_per_seed.json").items()
    }
    out.update(
        {
            f"GeoUNO ({int(rank):,} POD)": summarize([r["values"] for r in d["runs"]])
            for rank, d in read("co2_final.json").items()
        }
    )
    return out


def abstract():
    groups = read("inference_numbers.json")["groups"]

    def mean(config, metric):
        return st.mean(r[metric] for r in groups[config].values())

    return {
        "pressure-field error reduction vs prior (%)": 100
        * (1 - mean("multiwell_opt17/esmda", "p_field_RMSE") / mean("prior", "p_field_RMSE")),
        "temperature-field error reduction vs prior (%)": 100
        * (1 - mean("multiwell_opt17/esmda", "T_field_RMSE") / mean("prior", "T_field_RMSE")),
        "near-producer error reduction from guided placement (%)": 100
        * (1 - mean("multiwell_opt9/esmda", "T_nearwell_RMSE") / mean("multiwell_space9/esmda", "T_nearwell_RMSE")),
    }


def check():
    """Check all cells in Tables 1–2, Supplementary Tables 1, 2 and 19, and three abstract claims."""
    common = common_training_runs()
    computed = {
        "Table 1": table1(common),
        **table2(),
        "Supplementary Table 1": supplementary_table1(),
        "Supplementary Table 2": co2(),
        "Supplementary Table 19": supplementary_table19(common),
        "Abstract": abstract(),
    }
    expected = read("printed_values.json")
    report = []
    if set(computed) != set(expected):
        raise ValueError("Computed table set differs from printed tables")
    for table, rows in expected.items():
        if set(computed[table]) != set(rows):
            raise ValueError(f"Computed rows differ from printed rows: {table}")
        for row, printed in rows.items():
            value = computed[table][row]
            if isinstance(printed, int):
                report.append((table, row, "", value, printed, round(value) == printed))
                continue
            for col, (v, p) in enumerate(zip(value, printed, strict=True)):
                digits = 4 if table == "Supplementary Table 2" and col == 2 else 3
                ok = all(round(a, digits) == b for a, b in zip(v, p, strict=True))
                report.append((table, row, col, v, p, ok))
    return report


def main():
    report = check()
    for table, row, col, value, printed, ok in report:
        print(f"{'OK ' if ok else 'BAD'} {table} / {row} [{col}]: {value} -> {printed}")
    failures = [r for r in report if not r[-1]]
    print(json.dumps(dict(checked=len(report), mismatches=len(failures), common_protocol_runs=21)))
    return int(bool(failures))
