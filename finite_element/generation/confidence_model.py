# SPDX-License-Identifier: LGPL-2.1-only
import numpy as np
from scipy.ndimage import gaussian_filter

NC = 40
LX = 800.0
INJ = (0.0, -250.0)
xc = np.linspace(-LX/2, LX/2, NC)
yc = np.linspace(-LX/2, LX/2, NC)
XC, YC = np.meshgrid(xc, yc)
S_MIN = 0.05

def coarse_fracture_density(mask):

    Em = gaussian_filter(mask.astype(float), 25)
    st = mask.shape[0] // NC
    Ec = np.flipud(Em[:NC*st, :NC*st].reshape(NC, st, NC, st).mean((1, 3)))
    return (Ec - Ec.min()) / (Ec.max() - Ec.min() + 1e-12)

def _disk(cx, cy, r, a):
    return a * np.exp(-((XC-cx)**2 + (YC-cy)**2) / r**2)

def _corridor(p0, p1, halfw, a):

    d = np.hypot(p1[0]-p0[0], p1[1]-p0[1]) + 1e-9
    t = np.clip(((XC-p0[0])*(p1[0]-p0[0]) + (YC-p0[1])*(p1[1]-p0[1])) / d**2, 0, 1)
    dist2 = (XC - (p0[0]+t*(p1[0]-p0[0])))**2 + (YC - (p0[1]+t*(p1[1]-p0[1])))**2
    return a * np.exp(-dist2 / halfw**2)

def sample_confidence_cfg(rng, level=None):

    y_t = float(rng.uniform(50.0, 350.0))
    if level == "sparse":
        cfg = dict(bg=0.05, aw=0.85, rw=30., use_test_disk=False,
                   corr_a=0.0, corr_w=80., a_s=0.50, r_s=180., rich=0.2)
    elif level == "typical":
        cfg = dict(bg=0.10, aw=0.90, rw=40., use_test_disk=True,
                   corr_a=0.50, corr_w=100., a_s=0.65, r_s=260., rich=0.5)
    elif level == "dense":
        cfg = dict(bg=0.15, aw=0.95, rw=55., use_test_disk=True,
                   corr_a=0.70, corr_w=130., a_s=0.80, r_s=320., rich=0.9)
    else:

        rich = float(rng.uniform(0.0, 1.0))
        cfg = dict(bg=float(rng.uniform(0.03, 0.15)),
                   aw=float(rng.uniform(0.80, 0.95)) if rng.random() < 0.6 + 0.4*rich else 0.0,
                   rw=float(rng.uniform(25., 60.)),
                   use_test_disk=bool(rng.random() < 0.3 + 0.6*rich),
                   corr_a=(float(rng.uniform(0.35, 0.70))
                           if rng.random() < 0.25 + 0.65*rich else 0.0),
                   corr_w=float(rng.uniform(60., 150.)),
                   a_s=(float(rng.uniform(0.40, 0.80))
                        if rng.random() < 0.5 + 0.5*rich else 0.0),
                   r_s=float(rng.uniform(180., 320.)))
        cfg["rich"] = rich
    cfg["y_t"] = y_t
    return cfg

def eval_confidence(cfg, X, Y, Ec):

    def disk(cx, cy, r, a):
        return a * np.exp(-((X-cx)**2 + (Y-cy)**2) / r**2)
    comps = [np.full_like(X, cfg["bg"])]
    if cfg["aw"] > 0:
        comps.append(disk(*INJ, cfg["rw"], cfg["aw"]))
    if cfg["use_test_disk"] and cfg["aw"] > 0:
        comps.append(disk(0.0, cfg["y_t"], cfg["rw"], cfg["aw"]))
    if cfg["corr_a"] > 0:
        p0, p1 = INJ, (0.0, cfg["y_t"])
        d = np.hypot(p1[0]-p0[0], p1[1]-p0[1]) + 1e-9
        t = np.clip(((X-p0[0])*(p1[0]-p0[0]) + (Y-p0[1])*(p1[1]-p0[1])) / d**2, 0, 1)
        dist2 = (X - (p0[0]+t*(p1[0]-p0[0])))**2 + (Y - (p0[1]+t*(p1[1]-p0[1])))**2
        comps.append(cfg["corr_a"] * np.exp(-dist2 / cfg["corr_w"]**2))
    if cfg["a_s"] > 0:
        I = np.exp(-((X-INJ[0])**2 + (Y-INJ[1])**2)/cfg["r_s"]**2) * (0.35 + 0.65*Ec)
        I = (I - I.min()) / (I.max() - I.min() + 1e-12)
        comps.append(cfg["a_s"] * I)
    one_minus = np.ones_like(X)
    for c in comps:
        one_minus *= (1.0 - np.clip(c, 0.0, 0.999))
    return 1.0 - one_minus

def cfg_desc(cfg):
    return (f"rich={cfg.get('rich', -1):.2f} bg={cfg['bg']:.2f} "
            f"well(a={cfg['aw']:.2f},r={cfg['rw']:.0f}) test_disk={cfg['use_test_disk']} "
            f"corr(a={cfg['corr_a']:.2f},w={cfg['corr_w']:.0f}) "
            f"seis(a={cfg['a_s']:.2f},rs={cfg['r_s']:.0f}) y_t={cfg['y_t']:.0f}")

def build_confidence_map(rng, Ec, level=None):

    cfg = sample_confidence_cfg(rng, level=level)
    return eval_confidence(cfg, XC, YC, Ec), cfg_desc(cfg)

def confidence_to_S(C, S_max, S_min=S_MIN):

    return S_min + (1.0 - C) * (S_max - S_min)

def sample_realization_dev(S, L, z):

    return (S.ravel() * (L @ z)).reshape(NC, NC)
