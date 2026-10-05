"""Plot GeoUNO and FE fields of one released reservoir at one stored time.

    python scripts/plot_forecast.py --case test_01 --time 10 --out forecast_test_01.png
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from geouno import artifacts
from geouno.tasks import load_backend


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', default='test_01')
    ap.add_argument('--time', type=int, default=10, help='Stored time index (0-10; 10 = 10.01 years).')
    ap.add_argument('--out', default='forecast.png')
    a = ap.parse_args()
    backend = load_backend('cpu')
    reservoirs = artifacts.Reservoirs()
    case = reservoirs.case(a.case)
    base = backend.base_tensor(reservoirs.base(a.case))
    with torch.no_grad():
        pred = backend.predict_real(base, a.time, torch.as_tensor(case['logk_truth']))[0].numpy()
    truth = case['truth'][a.time]
    years = backend.stats['times_s'][a.time] / 3.15576e7
    extent = (-400, 400, -500, 500)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), constrained_layout=True)
    axes[0, 0].imshow(case['logk_truth'], origin='lower', extent=(-400, 400, -400, 400), cmap='cividis')
    axes[0, 0].set_title('log10 k (m$^2$)')
    axes[1, 0].axis('off')
    for row, (name, unit) in enumerate((('Cooling $\\Delta T$', 'K'), ('Pressure change $\\Delta p$', 'MPa'))):
        lo, hi = np.percentile(truth[row], [1, 99])
        for col, (title, field) in enumerate((('FE', truth[row]), ('GeoUNO', pred[row])), start=1):
            im = axes[row, col].imshow(field, origin='lower', extent=extent, vmin=lo, vmax=hi, cmap='inferno')
            axes[row, col].set_title(f'{name}, {title}')
        fig.colorbar(im, ax=axes[row, 1:3], label=unit, shrink=.8)
        err = pred[row] - truth[row]
        lim = np.abs(err).max()
        im = axes[row, 3].imshow(err, origin='lower', extent=extent, vmin=-lim, vmax=lim, cmap='RdBu_r')
        rmse = np.sqrt(np.mean(err[16:144] ** 2))
        axes[row, 3].set_title(f'GeoUNO - FE (reservoir RMSE {rmse:.2f} {unit})')
        fig.colorbar(im, ax=axes[row, 3], label=unit, shrink=.8)
    for ax in axes.flat:
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
    fig.suptitle(f'{a.case} at {years:.2f} years (true permeability)')
    fig.savefig(a.out, dpi=150)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
