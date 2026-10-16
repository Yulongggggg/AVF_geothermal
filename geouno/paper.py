"""Recompute the 15 October final-model paper tables from saved Source Data."""

import csv
import gzip
import json
import math
import statistics as st
from collections import defaultdict

from .artifacts import resource

SOURCE = resource("source_data")
FINAL = "final_model_20261015"
MODEL_NAMES = dict(
    deeponet="DeepONet", fno="FNO", ufno="U-FNO", unet="U-Net", resnet="ResNet", pcno="PCNO", geouno="GeoUNO"
)
SIX = [f"test_{i:02d}" for i in range(1, 7)]


def read(name):
    return json.loads((SOURCE / FINAL / name).read_text())


def _mean_sd(values):
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError("Invalid Source Data")
    return st.mean(values), st.stdev(values)


def summarize(runs):
    return tuple(_mean_sd(list(c)) for c in zip(*runs, strict=True))


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


def table1():
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
    out = {}
    for n in (24, 6):
        table = f"Table 2 ({'24' if n == 24 else 'six'} reservoirs)"
        out[table] = {}
        for label, key in methods.items():
            if n == 24 and label == "NUTS":
                continue
            cases = [f"test_{i:02d}" for i in range(1, n + 1)]
            out[table][label] = summarize([[groups[key][case][col] for col in columns] for case in cases])
    return out


def supplementary_table1():
    out = {}
    for name, d in read("uniform_final.json").items():
        if not d["complete"]:
            continue
        if len(d["runs"]) != 3:
            raise ValueError("Uniform comparison needs three completed seeds")
        for metric, key in [("RMSE", "rmse"), ("relative L2", "rel")]:
            out[MODEL_NAMES[name] + " / " + metric] = summarize([r[key] for r in d["runs"]])
    return out


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
    """Return (table, row, column, recomputed, printed, ok); pending cells are omitted."""
    computed = {
        "Table 1": table1(),
        **table2(),
        "Supplementary Table 1": supplementary_table1(),
        "CO2 benchmark": co2(),
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
                digits = 4 if table == "CO2 benchmark" and col == 2 else 3
                ok = all(round(a, digits) == b for a, b in zip(v, p, strict=True))
                report.append((table, row, col, v, p, ok))
    return report


def main():
    report = check()
    for table, row, col, value, printed, ok in report:
        print(f"{'OK ' if ok else 'BAD'} {table} / {row} [{col}]: {value} -> {printed}")
    failures = [r for r in report if not r[-1]]
    pending = [MODEL_NAMES[k] for k, d in read("uniform_final.json").items() if not d["complete"]]
    print(json.dumps(dict(checked=len(report), mismatches=len(failures), pending_uniform_models=pending)))
    return int(bool(failures))
