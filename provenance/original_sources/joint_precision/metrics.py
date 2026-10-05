"""Scenario-weighted physical errors and an exact pressure MSE decomposition."""
import numpy as np
import torch
from direct_operator.model import block_values


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
        s = {k: float(np.mean(v)) if (v := [r[k] for r in rr if r[k] is not None]) else None for k in rr[0]
             if k not in ('scen', 'time_index')}
        s['P_scenario_P90'] = float(np.quantile([r['P_rmse'] for r in rr], .9))
        s['P_offset_fraction_of_MSE'] = s['P_offset_mse'] / max(s['P_mse'], 1e-20)
        result[str(ti)] = s
    result['all_evaluated_times'] = {k: float(np.mean(v)) if (v := [s[k] for s in result.values() if s[k] is not None]) else None
                                     for k in next(iter(result.values()))}
    return result


@torch.no_grad()
def evaluate(model, data, device, times=(4, 7, 10), workers=0, chunk_size=16):
    from torch.utils.data import DataLoader
    from direct_operator.data import collate
    model.eval()
    sampler = [(i, ti) for i in range(len(data)) for ti in times]
    loader = DataLoader(data, batch_size=1, sampler=sampler, num_workers=workers,
                        collate_fn=collate, pin_memory=device == 'cuda')
    rows = []
    for (i, ti), (field, scalar, truth, lengths) in zip(sampler, loader):
        field, scalar, truth = [v.to(device) for v in (field, scalar, truth)]
        outputs = []
        for start in range(0, len(field), chunk_size):
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
                out = model(field[start:start+chunk_size], scalar[start:start+chunk_size])
            assert torch.isfinite(out).all()
            outputs.append(out.float())
        row = records(torch.cat(outputs), truth, field, lengths)[0]
        rows.append(dict(scen=data.pairs.names[i], time_index=ti, **row))
    return dict(scenario_count=len(data), actual_members=len(data.pairs), times=list(times),
                records=rows, summary=summarize(rows))
