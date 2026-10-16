"""Final metrics from overnight_bench/evaluate.py."""

import hashlib
import json
import gzip
import numpy as np
import torch
from ..artifacts import resource


def metric_record(pred, truth, ds, indices, ti, probabilistic, scope):
    rows = [int(ds.rows[i]) for i in indices]
    name = str(ds.row_scen[indices[0]])
    p = pred[:, :, 16:144].float()
    y = truth[:, :, 16:144].float()
    n = len(rows)
    r = dict(scen=name, rows=rows, n=n, time_index=ti, distribution_eligible=n >= 4)
    for ch, label in enumerate(("T", "P")):
        r[label + "_mean_field_rmse"] = float((p[:, ch].mean(0) - y[:, ch].mean(0)).square().mean().sqrt())
        if not probabilistic:
            mse = (p[:, ch] - y[:, ch]).square().mean((-2, -1))
            r[label + "_paired_rmse"] = float(mse.mean().sqrt())
            for subset in ("high_pressure", "within_envelope", "above_envelope"):
                key = "high_pressure_training_p90" if subset == "high_pressure" else "above_generator_envelope"
                selected = [j for j, row in enumerate(rows) if bool(scope[row][key]) == (subset != "within_envelope")]
                if selected:
                    r[label + "_" + subset + "_paired_rmse"] = float(mse[selected].mean().sqrt())
        if n >= 4:
            r[label + "_W1"] = float((p[:, ch].sort(0).values - y[:, ch].sort(0).values).abs().mean())
    if n >= 4:
        scale = ds.scales.to(p.device)
        a = (p / scale).flatten(1)
        b = (y / scale).flatten(1)
        d = a.shape[1] ** 0.5
        r["energy_score"] = float((torch.cdist(a, b).mean() - 0.5 * torch.cdist(a, a).mean()) / d)
        s = ds.scal[rows[0]]
        for label, ch, x, ypos in (
            ("prod_T", 0, float(s[3]), float(s[4])),
            ("inj_P", 1, 0.0, -250.0),
            ("prod_P", 1, float(s[3]), float(s[4])),
        ):
            mask = (torch.hypot(ds.x - x, ds.y - ypos) <= 6).to(p.device)
            assert mask.any()
            pv = pred[:, ch, mask].mean(1)
            tv = truth[:, ch, mask].mean(1)
            r[label + "_block_P90_abs"] = float((torch.quantile(pv, 0.9) - torch.quantile(tv, 0.9)).abs())
    if n >= 8:
        seed = int.from_bytes(hashlib.sha256(f"20260913:{name}:cross_member".encode()).digest()[:8], "little")
        order = np.random.default_rng(seed).permutation(n)
        k = n // 2
        draw = order[:k]
        held = order[k : 2 * k]
        # Surrogate inputs and FE references are disjoint members. PNO uses independent latent draws.
        cp = p[draw]
        cy = y[held]
        for ch, label in enumerate(("T", "P")):
            r["cross_" + label + "_mean_field_rmse"] = float(
                (cp[:, ch].mean(0) - cy[:, ch].mean(0)).square().mean().sqrt()
            )
            r["cross_" + label + "_W1"] = float((cp[:, ch].sort(0).values - cy[:, ch].sort(0).values).abs().mean())
        scale = ds.scales.to(p.device)
        aa = (cp / scale).flatten(1)
        bb = (cy / scale).flatten(1)
        r["cross_energy_score"] = float(
            (torch.cdist(aa, bb).mean() - 0.5 * torch.cdist(aa, aa).mean()) / (aa.shape[1] ** 0.5)
        )
        s = ds.scal[rows[0]]
        for label, ch, x, ypos in (
            ("prod_T", 0, float(s[3]), float(s[4])),
            ("inj_P", 1, 0.0, -250.0),
            ("prod_P", 1, float(s[3]), float(s[4])),
        ):
            mask = (torch.hypot(ds.x - x, ds.y - ypos) <= 6).to(p.device)
            pv = pred[draw, ch][:, mask].mean(1)
            tv = truth[held, ch][:, mask].mean(1)
            r["cross_" + label + "_block_P90_abs"] = float((torch.quantile(pv, 0.9) - torch.quantile(tv, 0.9)).abs())
    return r


def summarize(records):
    result = {}
    for ti in sorted({r["time_index"] for r in records}):
        rr = [r for r in records if r["time_index"] == ti]
        keys = set().union(*(r.keys() for r in rr)) - {"scen", "rows", "n", "time_index", "distribution_eligible"}
        result[str(ti)] = {
            k: dict(mean=float(np.mean(v)), scenario_p90=float(np.quantile(v, 0.9)), scenarios=len(v))
            for k in sorted(keys)
            if (v := [r[k] for r in rr if k in r])
        }
    return result


def load_scope():
    with gzip.open(resource("training_final_20261015/pressure_scope.json.gz"), "rt") as f:
        return {int(r["row"]): r for r in json.load(f)["records"]}
