# SPDX-License-Identifier: LGPL-2.1-only
import argparse, math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_opening, binary_closing, gaussian_filter, rotate, distance_transform_edt
from scipy.signal import fftconvolve

BD = Path(".")
OUT = BD / "prior_fields"
IMG_DIR = Path("images")
Lx = Ly = 800.0
X_WELL, Y_WELL = 0.0, -250.0

K_MIN, K_MAX = 2.7e-15, 1.35e-12
LOGK_MIN, LOGK_MAX = np.log(K_MIN), np.log(K_MAX)

CHANNEL_KAP2_RANGE = (0.04, 0.10)

W_WELL, W_FRAC = 0.80, 0.70
SIGMA_LOCAL, SIGMA_CLUSTER = 40.0, 45.0
ALPHA_FRAC_WELL = 1.0
EK_SIZE, EK_SX, EK_SY, EK_ANG = 31, 8.0, 4.0, 25.0
R_WELL_MIN, R_WELL_MAX = 150.0, 300.0

def normalize_field(a):
    a = np.asarray(a, float) - np.nanmin(a)
    m = np.nanmax(a)
    return a / m if m > 0 else a

def fracture_mask(img_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    w = np.flipud(img)
    R, G, B = w[:, :, 0].astype(float), w[:, :, 1].astype(float), w[:, :, 2].astype(float)
    gray = 0.2989*R + 0.5870*G + 0.1140*B
    red = (R > 180) & (G < 100) & (B < 100)
    m = (gray < 190) & (~red)
    m = binary_opening(m, np.ones((2, 2)))
    m = binary_closing(m, np.ones((2, 2)))
    return m

def ellipse_kernel():
    ax = np.arange(-(EK_SIZE//2), EK_SIZE//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-0.5*((xx/EK_SX)**2 + (yy/EK_SY)**2))
    k = rotate(k, EK_ANG, reshape=False, order=1, mode="constant", cval=0.0)
    return k / k.sum()

def build_heatmap_logk(img_path, r_well, ek):

    m = fracture_mask(img_path)
    ny, nx = m.shape
    x = np.linspace(-Lx/2, Lx/2, nx); y = np.linspace(-Ly/2, Ly/2, ny)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - X_WELL)**2 + (Y - Y_WELL)**2)
    well_effect = normalize_field(np.exp(-(r/r_well)**2))
    ell = normalize_field(fftconvolve(m.astype(float), ek, mode="same"))
    source = ell * (1.0 + ALPHA_FRAC_WELL*well_effect)
    frac_effect = normalize_field(gaussian_filter(source, SIGMA_LOCAL)
                                  + gaussian_filter(source, SIGMA_CLUSTER))
    score = np.clip(W_WELL*well_effect + W_FRAC*frac_effect, 0.0, 1.0)
    return LOGK_MIN + score*(LOGK_MAX - LOGK_MIN), x, y

def build_damage_zone_logk(img_path, T_m=4.0, L_srv=300.0):

    m = fracture_mask(img_path)
    ny, nx = m.shape
    mpp = Lx / nx
    d = distance_transform_edt(~m) * mpp
    x = np.linspace(-Lx/2, Lx/2, nx); y = np.linspace(-Ly/2, Ly/2, ny)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - X_WELL)**2 + (Y - Y_WELL)**2)
    sigma_dz = T_m / (2.0*math.sqrt(2.0*math.log(2.0)))
    lg = LOGK_MIN + (LOGK_MAX - LOGK_MIN) * np.exp(-0.5*(d/sigma_dz)**2) * np.exp(-0.5*(r/L_srv)**2)
    return lg, x, y

def build_heatmap_v2_logk(img_path, r_well, ek, e_ref, aniso=2.0,
                           w_well=0.45, w_frac=0.85, sig_local=20.0, sig_cluster=60.0, wl=0.6,
                           r_core=25.0, w_core=0.95):

    m = fracture_mask(img_path)
    ny, nx = m.shape
    x = np.linspace(-Lx/2, Lx/2, nx); y = np.linspace(-Ly/2, Ly/2, ny)
    X, Y = np.meshgrid(x, y)
    r2 = (X-X_WELL)**2 + (Y-Y_WELL)**2
    W = np.exp(-((X-X_WELL)/r_well)**2 - ((Y-Y_WELL)/(r_well/aniso))**2)
    Wcore = np.exp(-r2/r_core**2)
    E = np.clip(fftconvolve(m.astype(float), ek, mode="same")/e_ref, 0, 1.5)
    src = E*(1.0 + ALPHA_FRAC_WELL*W)
    Ft = wl*gaussian_filter(src, sig_local) + (1-wl)*gaussian_filter(src, sig_cluster)
    Ft = np.clip(Ft/np.percentile(Ft, 99), 0, 1)

    score = 1.0 - (1.0 - w_well*W)*(1.0 - w_frac*Ft)*(1.0 - w_core*Wcore)
    return LOGK_MIN + score*(LOGK_MAX - LOGK_MIN), x, y

def _aniso_gauss(field, theta_deg, sig_long, sig_short):

    r = rotate(field, theta_deg, reshape=False, order=1, mode="reflect")
    r = gaussian_filter(r, sigma=(sig_short, sig_long))
    return rotate(r, -theta_deg, reshape=False, order=1, mode="reflect")

def build_heatmap_struct_logk(img_path, rng, ek):

    m = fracture_mask(img_path)
    ny, nx = m.shape
    mpp = Lx / nx
    x = np.linspace(-Lx/2, Lx/2, nx); y = np.linspace(-Ly/2, Ly/2, ny)
    X, Y = np.meshgrid(x, y)

    n_lobe = int(rng.integers(1, 4))
    W = np.zeros_like(X)
    lobes = []
    for li in range(n_lobe):
        if li == 0:
            cx, cy = X_WELL, Y_WELL
            r_a = rng.uniform(70.0, 140.0)
        else:
            off_r = rng.uniform(60.0, 200.0)
            off_t = rng.uniform(0, 2*np.pi)
            cx, cy = X_WELL + off_r*np.cos(off_t), Y_WELL + off_r*np.sin(off_t)
            r_a = rng.uniform(120.0, 260.0)
        ar = rng.uniform(1.2, 3.0)
        th = rng.uniform(0, np.pi)
        wl = 1.0 if li == 0 else rng.uniform(0.4, 0.9)
        dx, dy = X - cx, Y - cy
        u = dx*np.cos(th) + dy*np.sin(th)
        v = -dx*np.sin(th) + dy*np.cos(th)
        W += wl * np.exp(-(u/r_a)**2 - (v/(r_a/ar))**2)
        lobes.append(f"lobe{li}(r={r_a:.0f},ar={ar:.1f},th={np.degrees(th):.0f})")
    W = normalize_field(W)

    ell = normalize_field(fftconvolve(m.astype(float), ek, mode="same"))
    th_c = rng.uniform(0, 180.0)
    sig_long = rng.uniform(80.0, 160.0)
    sig_short = rng.uniform(25.0, 45.0)
    Fc = normalize_field(_aniso_gauss(ell, th_c, sig_long, sig_short))

    source = ell * (1.0 + ALPHA_FRAC_WELL*W)
    Ft = normalize_field(gaussian_filter(source, SIGMA_LOCAL)
                         + gaussian_filter(source, SIGMA_CLUSTER))

    sig_tex = rng.uniform(45.0, 90.0)
    E_s = normalize_field(gaussian_filter(ell, sig_tex))
    beta = rng.uniform(0.4, 0.8)
    Gn = normalize_field(gaussian_filter(rng.standard_normal((ny, nx)), sig_tex))
    G = normalize_field((1.0 - beta)*Gn + beta*E_s)
    tau = rng.uniform(0.50, 0.62)
    kap = rng.uniform(0.03, 0.09)
    Gp = 1.0/(1.0 + np.exp(-(G - tau)/kap))

    w_w = rng.uniform(0.30, 0.55)
    w_f = rng.uniform(0.20, 0.45)
    w_c = rng.uniform(0.20, 0.45)
    w_g = rng.uniform(0.45, 0.75)
    score = np.clip(w_w*W + w_f*Ft + w_c*Fc + w_g*Gp, 0.0, 1.0)

    family = "blob"
    if rng.random() < 0.5:
        family = "channel"
        q_hi = rng.uniform(0.55, 0.75)
        tau2 = float(np.quantile(score, q_hi))
        kap2 = rng.uniform(*CHANNEL_KAP2_RANGE)
        score = 1.0/(1.0 + np.exp(-(score - tau2)/kap2))
    r2 = (X - X_WELL)**2 + (Y - Y_WELL)**2
    core = np.exp(-r2/25.0**2)
    score = 1.0 - (1.0 - score)*(1.0 - 0.95*core)
    desc = (f"{family}, {n_lobe}lobe[{';'.join(lobes)}], corridor(th={th_c:.0f},"
            f"sl={sig_long:.0f}px,ss={sig_short:.0f}px), tex(sig={sig_tex:.0f}px,"
            f"tau={tau:.2f},beta={beta:.2f}), w=({w_w:.2f},{w_f:.2f},{w_c:.2f},{w_g:.2f})")
    return LOGK_MIN + score*(LOGK_MAX - LOGK_MIN), x, y, desc

def ellipse_density_ref(idxs, ek):

    vals = []
    for i in idxs:
        p = IMG_DIR/f"nf_slice_{i:04d}.png"
        if p.exists():
            vals.append(np.percentile(fftconvolve(fracture_mask(p).astype(float), ek, mode="same"), 95))
    return float(np.median(vals))

def write_pml(field_k, x, y, path, style):
    vals = field_k.T.reshape(-1, order="F")
    with Path(path).open("w") as f:
        f.write(f"# prior mean permeability, style={style} (final_scheme/build_prior_field.py)\n\n")
        f.write("AXIS X\n" + " ".join(f"{v:.6f}" for v in x))
        f.write("\n\nAXIS Y\n" + " ".join(f"{v:.6f}" for v in y))
        f.write("\n\nDATA\n")
        for v in vals:
            f.write(f"{v:.12e}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", default="1-15")
    ap.add_argument("--style", choices=["heatmap", "heatmap_v2", "heatmap_struct", "damage_zone"], default="heatmap")
    ap.add_argument("--seed", type=int, default=42, help="r_well 采样种子（heatmap style）")
    ap.add_argument("--T", type=float, default=4.0, help="damage_zone: 厚度 FWHM [m]")
    ap.add_argument("--L-srv", type=float, default=300.0, help="damage_zone: SRV 尺度 [m]")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if "-" in args.indices:
        a, b = args.indices.split("-"); idxs = list(range(int(a), int(b)+1))
    else:
        idxs = [int(v) for v in args.indices.split(",")]

    rng = np.random.default_rng(args.seed)
    r_wells = {i: float(rng.uniform(R_WELL_MIN, R_WELL_MAX)) for i in idxs}
    ek = ellipse_kernel()
    prefix = {"heatmap": "prior", "heatmap_v2": "prior_v2", "heatmap_struct": "prior_st",
              "damage_zone": "prior_dz"}[args.style]
    e_ref = ellipse_density_ref(idxs, ek) if args.style == "heatmap_v2" else None

    for i in idxs:
        img = IMG_DIR / f"nf_slice_{i:04d}.png"
        if not img.exists():
            print(f"[skip] {img} 不存在"); continue
        if args.style == "heatmap":
            lg, x, y = build_heatmap_logk(img, r_wells[i], ek)
            extra = f"r_well={r_wells[i]:.1f}m"
        elif args.style == "heatmap_struct":
            lg, x, y, extra = build_heatmap_struct_logk(img, rng, ek)
        elif args.style == "heatmap_v2":
            lg, x, y = build_heatmap_v2_logk(img, r_wells[i], ek, e_ref)
            extra = f"r_well={r_wells[i]:.1f}m, aniso-SRV+softOR"
        else:
            lg, x, y = build_damage_zone_logk(img, args.T, args.L_srv)
            extra = f"T={args.T}m, L_srv={args.L_srv}m"
        out_i = OUT / f"{prefix}_{i:04d}_k.i"
        write_pml(np.exp(lg), x, y, out_i, args.style)
        lg10 = lg / np.log(10)
        print(f"{prefix}_{i:04d} ({args.style}, {extra}): log10k∈[{lg10.min():.2f},{lg10.max():.2f}] -> {out_i.name}")
        if args.preview:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax_ = plt.subplots(figsize=(6, 6))
            im = ax_.imshow(lg10, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower",
                            vmin=np.log10(K_MIN), vmax=np.log10(K_MAX), cmap="turbo")
            ax_.scatter([X_WELL], [Y_WELL], s=25, edgecolors="k", facecolors="w")
            ax_.set_title(f"{prefix}_{i:04d}  ({args.style}, {extra})")
            fig.colorbar(im, ax=ax_, shrink=0.8, label="log10 k [m2]")
            fig.savefig(OUT / f"{prefix}_{i:04d}_preview.png", dpi=110, bbox_inches="tight")
            plt.close(fig)

if __name__ == "__main__":
    main()
