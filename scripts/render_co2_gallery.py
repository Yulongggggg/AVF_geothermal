"""Render the paper CO2 gallery from saved fields; no model execution."""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from geouno.artifacts import resource
import figure_style as style

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("figures/co2"))
args = parser.parse_args()
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)
with np.load(resource("source_data/final_model_20261015/co2_visualization.npz")) as data:
    truth, pred, mask, cases, ti = [data[k].copy() for k in ("truth", "prediction", "mask", "cases", "shown_indices")]
with plt.rc_context(style.rc(7)):
    fig = plt.figure(figsize=(style.WIDTH_MAIN, 8.65))
    outer = fig.add_gridspec(4, 2, hspace=0.40, wspace=0.24, left=0.065, right=0.99, bottom=0.045, top=0.91)
    for channel, (name, cmap, unit) in enumerate(
        [(r"Gas saturation $S_g$", "inferno", ""), (r"Pressure build-up $\Delta p$", "Blues", "bar")]
    ):
        fig.text(0.018 if channel == 0 else 0.512, 0.987, "ab"[channel], fontsize=10, fontweight="bold", va="top")
        fig.text(0.068 if channel == 0 else 0.56, 0.987, name, fontsize=9, va="top")
        for k, case in enumerate(cases):
            sub = outer[k, channel].subgridspec(5, 3, height_ratios=[1, 1, 1, 0.18, 0.11], hspace=0.16, wspace=0.13)
            y0, y1 = np.flatnonzero(mask[k].any(1))[[0, -1]]
            y1 += 1
            lo = min(float(truth[k, :, channel, mask[k]].min()), float(pred[k, :, channel, mask[k]].min()), 0.0)
            hi = max(float(truth[k, :, channel, mask[k]].max()), float(pred[k, :, channel, mask[k]].max()))
            delta = pred[k, :, channel] - truth[k, :, channel]
            lim = float(np.abs(delta[:, mask[k]]).max())
            for row, t in enumerate(ti):
                for col, arr in enumerate((truth[k, row, channel], pred[k, row, channel], delta[row])):
                    ax = fig.add_subplot(sub[row, col])
                    im = ax.imshow(
                        np.ma.array(arr, mask=~mask[k]),
                        origin="lower",
                        aspect="auto",
                        interpolation="nearest",
                        extent=[0, 200, 0, 96],
                        cmap=cmap if col < 2 else "RdBu_r",
                        vmin=lo if col < 2 else -lim,
                        vmax=hi if col < 2 else lim,
                        rasterized=True,
                    )
                    ax.set_ylim(y0, y1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_visible(True)
                        sp.set_linewidth(0.4)
                    if col == 0:
                        ax.set_ylabel(f"Time {t + 1}", fontsize=5.8, labelpad=3)
                    if row == 0:
                        ax.set_title(["Reference", "GeoUNO", "Difference"][col], fontsize=6.5, pad=3)
                        if col == 0:
                            ax.text(
                                0,
                                1.74,
                                f"Case {chr(65 + k)} · test {case + 1}",
                                fontsize=7,
                                fontweight="bold",
                                transform=ax.transAxes,
                                va="bottom",
                            )
                    if row == 2 and col in (1, 2):
                        ca = fig.add_subplot(sub[4, :2] if col == 1 else sub[4, 2])
                        cb = fig.colorbar(im, cax=ca, orientation="horizontal")
                        cb.locator = MaxNLocator(3)
                        cb.update_ticks()
                        cb.outline.set_linewidth(0.4)
                        cb.ax.tick_params(length=1.5, pad=1, labelsize=5.5)
                        cb.set_label(
                            ("Difference" if col == 2 else ("$S_g$" if channel == 0 else "$\\Delta p$"))
                            + (f" ({unit})" if unit else ""),
                            fontsize=6,
                            labelpad=1,
                        )
    fig.savefig(OUT / "fig_co2_transfer.pdf")
    fig.savefig(OUT / "fig_co2_transfer.png", dpi=300)
    plt.close(fig)
