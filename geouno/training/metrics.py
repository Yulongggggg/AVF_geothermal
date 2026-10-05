"""Scenario-weighted validation errors with an exact pressure MSE decomposition.

Sources: ``joint_precision/metrics.py`` (``records``, ``summarize``) and
``direct_operator/model.py`` (``block_values``).
"""
import numpy as np
import torch


def block_values(value, field):
    """Mean of ``value`` within 6 m of the injector (channel 5) and producer (channel 4)."""
    values = []
    for ch in (5, 4):
        mask = (field[:, ch:ch + 1] * 400 <= 6).float()
        assert bool((mask.sum((-2, -1)) > 0).all())
        values.append((value * mask).sum((-2, -1)) / mask.sum((-2, -1)))
    return torch.cat(values, 1)


def records(pred, truth, field, lengths):
    result = []
    for p, y, f in zip(torch.split(pred.float(), lengths),
                       torch.split(truth.float(), lengths), torch.split(field, lengths)):
        e = p[:, :, 16:144] - y[:, :, 16:144]
        pressure = e[:, 1]
        offset = pressure.mean((-2, -1), keepdim=True)
        dc_mse = offset.square().mean()
        shape_mse = (pressure - offset).square().mean()
        total_mse = pressure.square().mean()
        assert torch.allclose(dc_mse + shape_mse, total_mse, rtol=2e-5, atol=1e-7)
        pw = block_values(p[:, 1:2], f)
        yw = block_values(y[:, 1:2], f)
        row = dict(
            T_rmse=float(e[:, 0].square().mean().sqrt()),
            P_rmse=float(total_mse.sqrt()), P_mse=float(total_mse),
            P_offset_mse=float(dc_mse), P_shape_mse=float(shape_mse),
            P_bias=float(offset.mean()),
            inj_rmse=float((pw[:, 0] - yw[:, 0]).square().mean().sqrt()),
            prod_rmse=float((pw[:, 1] - yw[:, 1]).square().mean().sqrt()))
        # Match the manuscript's response-quantile eligibility rule.
        row.update(inj_P90=None, prod_P90=None)
        if len(p) >= 4:
            row.update(inj_P90=float((torch.quantile(pw[:, 0], .9) - torch.quantile(yw[:, 0], .9)).abs()),
                       prod_P90=float((torch.quantile(pw[:, 1], .9) - torch.quantile(yw[:, 1], .9)).abs()))
        result.append(row)
    return result


def summarize(rows):
    result = {}
    for ti in sorted({r['time_index'] for r in rows}):
        rr = [r for r in rows if r['time_index'] == ti]
        s = {}
        for k in rr[0]:
            if k in ('scen', 'time_index'):
                continue
            v = [r[k] for r in rr if r[k] is not None]
            s[k] = float(np.mean(v)) if v else None
        s['P_scenario_P90'] = float(np.quantile([r['P_rmse'] for r in rr], .9))
        s['P_offset_fraction_of_MSE'] = s['P_offset_mse'] / max(s['P_mse'], 1e-20)
        result[str(ti)] = s
    overall = {}
    for k in next(iter(result.values())):
        v = [s[k] for s in result.values() if s[k] is not None]
        overall[k] = float(np.mean(v)) if v else None
    result['all_evaluated_times'] = overall
    return result
