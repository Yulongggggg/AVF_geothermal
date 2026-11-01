"""Checkpoint selection by the fixed-draw composite energy score."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import FixedPairs
from .model import loss


@torch.no_grad()
def validation(net, ds, device, batch_size=64, workers=4):
    net.eval()
    sums = {}
    energy = {}
    loader = DataLoader(ds, batch_size=batch_size, sampler=FixedPairs(ds, (4, 7, 10)), num_workers=workers)
    # Repeatable PNO validation draws; restore training random state afterwards.
    devices = list(range(torch.cuda.device_count())) if device == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(97113)
        for f, c, y, ii, tis in loader:
            f, c, y = f.to(device), c.to(device), y.to(device)
            pred = net(f, c)
            if pred.ndim == 5:
                weights = torch.ones(len(f), device=device)
                es = loss(pred, y, f, weights, True, return_per=True).cpu().numpy()
                pred = pred.mean(1)
            else:
                es = None
            mse = ((pred - y).square()[:, :, 16:144]).mean((-2, -1)).cpu().numpy()
            for j, (idx, ti) in enumerate(zip(ii.tolist(), tis.tolist())):
                key = (str(ds.row_scen[idx]), ti)
                sums.setdefault(key, []).append(mse[j])
                if es is not None:
                    energy.setdefault(key, []).append(float(es[j]))
    per = np.array([np.mean(v, 0) for v in sums.values()])
    rmse = np.sqrt(per)
    scale = ds.scales[:, 0, 0].numpy()
    result = dict(
        equal_scenario_normalized_rmse=float(rmse.mean()),
        T_rmse_K=float((rmse[:, 0] * scale[0]).mean()),
        P_rmse_MPa=float((rmse[:, 1] * scale[1]).mean()),
        P_scenario_p90_MPa=float(np.quantile(rmse[:, 1] * scale[1], 0.9)),
        scenario_time_pairs=len(sums),
    )
    result["selection_score"] = (
        float(np.mean([np.mean(v) for v in energy.values()])) if energy else result["equal_scenario_normalized_rmse"]
    )
    result["selection_objective"] = "proper_composite_energy_score" if energy else "normalized_paired_rmse"
    return result
