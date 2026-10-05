"""Posterior forecast at ten years and its scores.

Source: ``operator_final_20260930/inverse_run.py`` (``forecast``). Members are propagated
through the deterministic GeoUNO at stored time index 10 (10.01 years). Field RMSE compares
the ensemble-mean reservoir field (rows 16..143) with the FE truth; permeability RMSE
compares the ensemble-mean log10 permeability with the true field. Producer temperature and
injector pressure are 6-m block averages (``qoi_weights``). ``raw`` scores use the
propagated ensemble only; ``with_surrogate_epistemic`` adds the validation-fitted residual
variance as a Gaussian mixture (separate diagnostic).
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

from ..metrics import score

FORECAST_TIME_INDEX = 10


def forecast(problem, latent, save_members=False, point_estimate=False, time_index=FORECAST_TIME_INDEX):
    """Return ``(scores, arrays)`` for whitened latent members ``latent`` (``[N, 4096]``)."""
    latent = np.asarray(latent).reshape(-1, 4096)
    fields = []
    geology = []
    qoi = []
    with torch.no_grad():
        for first in range(0, len(latent), 4):
            m = problem.permeability(problem.tensor(latent[first:first + 4]))
            f = problem.backend.predict_real(problem.base, time_index, m)
            fields.append(f[:, :, 16:144].cpu().numpy())
            geology.append(m.cpu().numpy())
            qoi.append(torch.einsum('bchw,phw->bcp', f.double(), problem.qoi_weights).cpu().numpy())
    fields = np.concatenate(fields)
    geology = np.concatenate(geology)
    qoi = np.concatenate(qoi)
    truth = problem.case['truth'][time_index, :, 16:144]
    truth_qoi = np.einsum('chw,phw->cp', problem.case['truth'][time_index], problem.case['qoi_weights'])
    result = dict(members=len(latent),
                  field_rmse=np.sqrt(np.mean((fields.mean(0) - truth) ** 2, axis=(1, 2))).tolist(),
                  permeability_rmse=float(np.sqrt(np.mean((geology.mean(0) - problem.case['logk_truth']) ** 2))),
                  blocks={})
    for ch, label in enumerate(('temperature', 'pressure')):
        draws = qoi[:, ch, ch]
        target = truth_qoi[ch, ch]
        sd = np.sqrt(problem.cal['qoi_variance'][time_index][ch][ch])
        result['blocks'][label] = {}
        if point_estimate:
            assert len(draws) == 1
            result['blocks'][label] = dict(absolute_error=float(abs(draws[0] - target)),
                                           point_prediction=float(draws[0]), posterior_interval=None)
            continue
        for level, error_sd in [('raw', None), ('with_surrogate_epistemic', sd)]:
            result['blocks'][label][level] = {k: float(v) for k, v in score(draws, target, error_sd).items()}
    arrays = dict(field_mean=fields.mean(0), field_sd=fields.std(0), field_p05=np.quantile(fields, .05, axis=0),
                  field_p95=np.quantile(fields, .95, axis=0), qoi_samples=qoi, truth_qoi=truth_qoi,
                  truth_fields=truth, logk_mean=geology.mean(0), logk_sd=geology.std(0),
                  truth_logk=problem.case['logk_truth'])
    if save_members:
        arrays.update(field_samples=fields, logk_samples=geology)
    return result, arrays


def headline(result):
    """The manuscript metrics of one forecast (``scores.json``) as a flat dictionary."""
    row = dict(temperature_field_RMSE_K=result['field_rmse'][0], pressure_field_RMSE_MPa=result['field_rmse'][1],
               log10_permeability_RMSE_dec=result['permeability_rmse'])
    for label, unit in (('temperature', 'K'), ('pressure', 'MPa')):
        block = result['blocks'][label]
        if 'raw' in block:
            row[f'{label}_well_raw_p50_absolute_error_{unit}'] = block['raw']['p50_absolute_error']
            row[f'{label}_well_raw_width90_{unit}'] = block['raw']['width90']
        else:
            row[f'{label}_well_point_error_{unit}'] = block['absolute_error']
    return row


def write_forecast(dest, result, arrays):
    """Write ``forecast.npz`` and ``scores.json`` atomically into ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / 'forecast.tmp.npz'
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, dest / 'forecast.npz')
    tmp = dest / f'scores.json.{os.getpid()}.tmp'
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, dest / 'scores.json')
