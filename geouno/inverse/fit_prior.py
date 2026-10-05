"""Training-only stationary covariance for the inverse permeability prior.

Source: ``history_matching_dense/fit_prior.py`` (``main``) and
``geouno/scripts/refit_prior.py`` (reconstruction from released inputs).

One actual member from each of 1,024 hash-selected training scenarios is projected onto
the bilinear 64 x 64 latent basis about the training-wide mean ``M_mean``. Averaged squared
Fourier amplitudes define a periodic stationary covariance; the covariance of the
independently randomized prior mean (``0.25 L0 L0^T`` with the original 0.5-dec smooth
factor ``L0``) is added, plus 1e-6 diagonal jitter, and the Cholesky factor is stored in
FP32. Periodicity and stationarity are modelling assumptions. No validation or test geology
is used. The paper's marginal SD is 0.9759736257386044 dec (0.976 in the manuscript).
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .prior import interpolation_matrix

PAPER_MARGINAL_SD = 0.9759736257386044


def select_rows(meta_rows, count=1024):
    """One member (lowest ``real`` label) from each of ``count`` hash-ordered training scenarios."""
    by = {}
    for row in meta_rows:
        if row['split'] == 'train' and row['real'] != 'nom':
            by.setdefault(row['scen'], []).append(row)
    names = sorted(by, key=lambda s: hashlib.sha256(('prior20260910:' + s).encode()).digest())[:count]
    return names, [sorted(by[n], key=lambda r: r['real'])[0] for n in names]


def fit_prior(pack, original_factor, threads=4):
    """Return ``(factor [4096, 4096] float32, power [64, 64], report)`` for a packed corpus."""
    torch.set_num_threads(threads)
    pack = Path(pack)
    meta = list(csv.DictReader((pack / 'meta.csv').open()))
    # One actual member per scenario gives scenarios equal weight; hashing is
    # independent of values and excludes every validation/test row explicitly.
    names, selected = select_rows(meta)
    real = np.load(pack / 'real.npy', mmap_mode='r')
    stats = json.loads((pack / 'stats.json').read_text())
    mean = float(stats['M_mean'])
    B = interpolation_matrix()
    pinv = torch.linalg.pinv(B)
    power = torch.zeros(64, 64, dtype=torch.float64)
    residual = []
    for first in range(0, len(selected), 32):
        rows = [int(r['row']) for r in selected[first:first + 32]]
        fields = torch.tensor(np.array(real[rows], copy=True), dtype=torch.float64) - mean
        grid = pinv @ fields @ pinv.T
        power += torch.fft.fft2(grid).abs().square().sum(0)
        residual.extend((fields - B @ grid @ B.T).square().mean((-2, -1)).sqrt().tolist())
    power /= len(selected)
    kernel = torch.fft.ifft2(power).real / 4096
    # Circular stationarity is an explicit modelling assumption. This estimate
    # stores averaged second moments, not any individual geological map.
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
    coords = torch.stack([yy.flatten(), xx.flatten()], 1)
    dy = (coords[:, None, 0] - coords[None, :, 0]) % 64
    dx = (coords[:, None, 1] - coords[None, :, 1]) % 64
    covariance = kernel[dy, dx]
    del dy, dx
    L = torch.from_numpy(np.load(original_factor)).double()
    covariance += .25 * (L @ L.T)
    jitter = 1e-6
    covariance.diagonal().add_(jitter)
    factor = torch.linalg.cholesky(covariance).float()
    relative = float(torch.linalg.vector_norm(factor.double() @ factor.double().T - covariance)
                     / torch.linalg.vector_norm(covariance))
    if relative > 1e-5:
        raise ValueError('Covariance factorization inaccurate')
    report = dict(training_scenarios=len(names), selected_rows=[int(r['row']) for r in selected],
                  selected_cases=[r['case_id'] for r in selected],
                  marginal_sd=float(covariance.diagonal().mean().sqrt()), factor_relative_error=relative,
                  median_training_projection_error=float(np.median(residual)))
    return factor.numpy(), power.numpy(), report


def refit_from_release(inputs, out):
    """Rebuild the 1,024-field selection pack from released inputs and refit the prior.

    ``inputs`` is the extracted ``prior_fit_inputs`` asset (``fit_rows.json``,
    ``fit_fields.npz``, ``stats.json``, ``original_factor_lower.npz``, ``prior_fit.json``,
    ``prior_factor.npy``). Returns a dictionary of comparisons with the paper prior.
    """
    inputs, out = Path(inputs), Path(out)
    pack = out / 'selected_training_pack'
    pack.mkdir(parents=True, exist_ok=True)
    rows = json.loads((inputs / 'fit_rows.json').read_text())
    for i, r in enumerate(rows):
        r['row'] = str(i)
    with (pack / 'meta.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.save(pack / 'real.npy', np.load(inputs / 'fit_fields.npz')['fields'])
    (pack / 'stats.json').write_text((inputs / 'stats.json').read_text())
    L = np.zeros((4096, 4096), dtype=np.float64)
    L[np.tril_indices(4096)] = np.load(inputs / 'original_factor_lower.npz')['lower']
    original = out / 'L_chol64.npy'
    np.save(original, L)
    saved = json.loads((inputs / 'prior_fit.json').read_text())
    assert hashlib.sha256(original.read_bytes()).hexdigest() == saved['original_factor_sha256']
    factor, power, report = fit_prior(pack, original)
    np.save(out / 'prior_factor.npy', factor)
    reference = np.load(inputs / 'prior_factor.npy')
    np.testing.assert_allclose(report['marginal_sd'], saved['marginal_sd'], rtol=1e-12)
    np.testing.assert_allclose(factor, reference, rtol=2e-5, atol=1e-7)
    assert report['selected_cases'] == saved['selected_cases']
    return dict(marginal_sd=report['marginal_sd'], paper_marginal_sd=saved['marginal_sd'],
                max_abs_factor_difference=float(np.max(np.abs(factor - reference))),
                factor_bitwise_equal=bool(np.array_equal(factor, reference)),
                selected_cases_equal=True, factor_relative_error=report['factor_relative_error'])
