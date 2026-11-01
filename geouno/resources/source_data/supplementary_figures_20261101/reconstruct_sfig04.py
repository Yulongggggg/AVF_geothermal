"""Reconstruct Supplementary Fig. 4 arrays at their original resolution.

The source data are the native fracture mask and the recorded PCG64 state.
Every output is checked against the original displayed array's byte hash.
Reference environment: NumPy 2.4.6 and SciPy 1.15.3.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, rotate
from scipy.signal import fftconvolve

EK_SIZE, EK_SX, EK_SY, EK_ANG = 31, 8.0, 4.0, 25.0
SIGMA_LOCAL, SIGMA_CLUSTER = 40.0, 45.0
ALPHA_FRAC_WELL = 1.0
K_MIN, K_MAX = 4.0e-15, 1.35e-12
KAP2_RANGE = (0.08, 0.20)
X_INJ, Y_INJ = 0.0, -250.0


def normalize_field(a):
    a = np.asarray(a, float) - np.nanmin(a)
    m = np.nanmax(a)
    return a / m if m > 0 else a


def ellipse_kernel():
    ax = np.arange(-(EK_SIZE // 2), EK_SIZE // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-0.5 * ((xx / EK_SX) ** 2 + (yy / EK_SY) ** 2))
    k = rotate(k, EK_ANG, reshape=False, order=1, mode="constant", cval=0.0)
    return k / k.sum()


def _aniso_gauss(field, theta_deg, sig_long, sig_short):
    """Rotate, apply separable Gaussian smoothing, then rotate back."""
    r = rotate(field, theta_deg, reshape=False, order=1, mode="reflect")
    r = gaussian_filter(r, sigma=(sig_short, sig_long))
    return rotate(r, -theta_deg, reshape=False, order=1, mode="reflect")


def construct(mask, rng_state):
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = rng_state
    m = mask
    ny, nx = m.shape
    x = np.linspace(-400, 400, nx)
    y = np.linspace(-400, 400, ny)
    X, Y = np.meshgrid(x, y)
    n_lobe = int(rng.integers(1, 4))
    W = np.zeros_like(X)
    for li in range(n_lobe):
        if li == 0:
            cx, cy = X_INJ, Y_INJ
            r_a = rng.uniform(70.0, 140.0)
        else:
            off_r = rng.uniform(60.0, 200.0)
            off_t = rng.uniform(0, 2 * np.pi)
            cx, cy = X_INJ + off_r * np.cos(off_t), Y_INJ + off_r * np.sin(off_t)
            r_a = rng.uniform(120.0, 260.0)
        ar = rng.uniform(1.2, 3.0)
        th = rng.uniform(0, np.pi)
        wl = 1.0 if li == 0 else rng.uniform(0.4, 0.9)
        u = (X - cx) * np.cos(th) + (Y - cy) * np.sin(th)
        v = -(X - cx) * np.sin(th) + (Y - cy) * np.cos(th)
        W += wl * np.exp(-((u / r_a) ** 2) - (v / (r_a / ar)) ** 2)
    W = normalize_field(W)
    E = normalize_field(fftconvolve(m.astype(float), ellipse_kernel(), mode="same"))
    th_c = rng.uniform(0, 180.0)
    sig_long = rng.uniform(80.0, 160.0)
    sig_short = rng.uniform(25.0, 45.0)
    Fc = normalize_field(_aniso_gauss(E, th_c, sig_long, sig_short))
    src = E * (1.0 + ALPHA_FRAC_WELL * W)
    Ft = normalize_field(
        gaussian_filter(src, SIGMA_LOCAL) + gaussian_filter(src, SIGMA_CLUSTER)
    )
    sig_tex = rng.uniform(45.0, 90.0)
    Es = normalize_field(gaussian_filter(E, sig_tex))
    beta = rng.uniform(0.4, 0.8)
    Gn = normalize_field(gaussian_filter(rng.standard_normal((ny, nx)), sig_tex))
    Gmix = normalize_field((1.0 - beta) * Gn + beta * Es)
    tau = rng.uniform(0.50, 0.62)
    kap = rng.uniform(0.03, 0.09)
    Gp = 1.0 / (1.0 + np.exp(-(Gmix - tau) / kap))
    w = {
        "w": rng.uniform(0.30, 0.55),
        "f": rng.uniform(0.20, 0.45),
        "c": rng.uniform(0.20, 0.45),
        "g": rng.uniform(0.45, 0.75),
    }
    raw = w["w"] * W + w["f"] * Ft + w["c"] * Fc + w["g"] * Gp
    S0 = np.clip(raw, 0.0, 1.0)
    family = "channelized" if rng.random() < 0.5 else "smooth"
    q_hi = rng.uniform(0.55, 0.75)
    kap2 = rng.uniform(*KAP2_RANGE)
    tau2 = float(np.quantile(S0, q_hi))
    Sf_c = 1.0 / (1.0 + np.exp(-(S0 - tau2) / kap2))
    core = np.exp(-((X - X_INJ) ** 2 + (Y - Y_INJ) ** 2) / 25.0**2)

    sf = Sf_c if family == "channelized" else S0
    well = 0.95 * core
    score = 1.0 - (1.0 - sf) * (1.0 - well)
    lo, hi = np.log10(K_MIN), np.log10(K_MAX)
    mean = lo + score * (hi - lo)
    return {
        "W": W,
        "Ft": Ft,
        "Fc": Fc,
        "Gp": Gp,
        "S0": S0,
        "Sf_channelized": Sf_c,
        "injector_term": well,
        "score": score,
        "M": mean,
        "x_m": x,
        "y_m": y,
    }


def array_identity(value):
    value = np.ascontiguousarray(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional output NPZ; omitting it verifies arrays in memory",
    )
    args = parser.parse_args(argv)
    here = Path(__file__).resolve().parent
    metadata = json.loads((here / "sfig04_definitions.json").read_text())
    with np.load(here / "sfig04_native_inputs.npz") as source:
        mask = np.array(source["fracture_mask"], copy=True)
    if array_identity(mask) != metadata["native_input_array_index"]["fracture_mask"]:
        raise ValueError(
            "The native fracture mask differs from the published source data"
        )
    arrays = construct(
        mask, metadata["reconstruction"]["rng_state_before_construction"]
    )
    for name, values in arrays.items():
        if array_identity(values) != metadata["array_index"][name]:
            raise ValueError(
                f"Array {name} differs from the original. Reference: NumPy 2.4.6 / SciPy 1.15.3"
            )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **arrays)
    print("Verified all nine full-resolution maps and both coordinate arrays.")


if __name__ == "__main__":
    main()
