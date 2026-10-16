"""Shared house style for the polished manuscript figures (2026-10-01).

Extends the author's qstyle.py / supplement_style.py conventions:
real-size drawing (183 mm double column, 131 mm SI width), Nimbus Sans /
Helvetica at 6-8 pt, open left/bottom spines, temperature red / pressure blue,
light palettes.  Bar charts follow ChenLiu-1996/figures4papers
(figure_ImmunoStruct/plot_bars.py): solid bars, capped black whiskers, open
top/right axes, no category ticks, a separate legend column; colours are a
single-hue blue ramp as requested by the author.
"""

from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

# --- house colours -----------------------------------------------------------
T_COLOR, P_COLOR = "#C44E52", "#4C72B0"  # temperature red / pressure blue
INK, MUTED, GREY, GREY_L = "#1A1A1A", "#6F737A", "#9A9FA6", "#D8DBDF"
NAVY = "#1F4E79"
HOUSE_BLUE = "#3775BA"  # ImmunoStruct "ours" blue (55,117,186)
INJ_FC, INJ_EC = "#38b6ff", "black"  # injector star
FAMILY = dict(multiwell="#285F96", vertical="#B86745", inclined="#3C8071")
FAMILY_NAME = dict(multiwell="Multiwell", vertical="Vertical well", inclined="Inclined well")
DESIGNED, SPACEFILL = "#285F96", "#B96C38"
CM_K, CM_T, CM_P, CM_S = "viridis", "inferno", "Blues", "magma"
WIDTH_MAIN = 7.09  # 180 mm
WIDTH_SI = (372 * 72 / 72.27) / 72  # manuscript text width (~5.1 in)

BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#F3F7FB", "#C6D9EC", "#8DB4D9", "#4F8AC4", HOUSE_BLUE, NAVY]
)


def blue_ramp(n, lo=0.28, hi=1.0):
    """ImmunoStruct ablation ramp: one hue, alpha from light to full."""
    r, g, b = to_rgb(HOUSE_BLUE)
    return [(r, g, b, a) for a in np.linspace(lo, hi, n)]


def rc(size=7.0):
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": size,
        "axes.titlesize": size + 0.5,
        "axes.labelsize": size,
        "xtick.labelsize": size - 0.6,
        "ytick.labelsize": size - 0.6,
        "legend.fontsize": size - 0.6,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "mathtext.fontset": "dejavusans",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.unicode_minus": False,
        "figure.dpi": 150,
    }


def letter(ax, s, dx=-0.22, dy=1.06, size=9):
    ax.text(
        dx,
        dy,
        s,
        transform=ax.transAxes,
        fontsize=size,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="black",
        clip_on=False,
    )


def fletter(fig, x, y, s, size=9):
    fig.text(x, y, s, fontsize=size, fontweight="bold", va="top", ha="left", color="black")


def tidy(ax, lw=0.6):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(lw)
    ax.tick_params(length=2.4, width=0.6, pad=2)


def bars(ax, values, errors, colors, width=0.74, ecolor="black"):
    """ImmunoStruct bar template at manuscript scale."""
    values = np.asarray(values, float)
    errors = np.asarray(errors, float)
    b = ax.bar(
        np.arange(len(values)),
        values,
        width=width,
        color=colors,
        edgecolor="none",
        yerr=errors,
        error_kw=dict(ecolor=ecolor, elinewidth=0.7, capsize=2.2, capthick=0.7),
        zorder=3,
    )
    ax.set_xticks([])
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    ax.tick_params(axis="y", width=0.7, length=2.6, pad=2)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_ylim(bottom=0)
    return b


def legend_column(ax, labels, colors, title=None, note=None, fs=6.6):
    ax.axis("off")
    handles = [Patch(facecolor=c, edgecolor="none", label=l) for l, c in zip(labels, colors)]
    lg = ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#D5DAE0",
        fontsize=fs,
        handlelength=1.4,
        handleheight=1.0,
        borderpad=0.7,
        labelspacing=0.6,
        borderaxespad=0,
        title=title,
        title_fontsize=fs,
    )
    lg.get_frame().set_linewidth(0.5)
    if note:
        ax.text(0.0, 0.02, note, transform=ax.transAxes, fontsize=fs - 0.6, color=MUTED, va="bottom", linespacing=1.4)
    return lg


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, stem, dpi=300, audit=None, tight=False):
    stem = OUT / stem
    kw = dict(bbox_inches="tight", pad_inches=0.02) if tight else {}
    fig.savefig(stem.with_suffix(".pdf"), **kw)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, **kw)
    if audit is not None:
        audit = dict(audit, pdf_sha256=sha(stem.with_suffix(".pdf")))
        stem.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    plt.close(fig)
    print("wrote", stem.with_suffix(".pdf"))
    return stem.with_suffix(".pdf")
