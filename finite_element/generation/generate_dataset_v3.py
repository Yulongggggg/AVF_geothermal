# SPDX-License-Identifier: LGPL-2.1-only
import argparse, csv, glob, os, re, subprocess, time
from pathlib import Path
import numpy as np

BD = Path(__file__).resolve().parent
DATA = BD / "dataset_v3"
TEMPLATE = BD / "THM_template_final.i"
IMG_DIR = Path(os.environ.get("GEO_IMG_DIR",
    "images"))
MESH_DIR = Path(os.environ.get("GEO_MESH_DIR",
    "mesh"))
N_IMAGES = 2000
MPIEXEC = os.environ.get("GEO_MPIEXEC", "mpiexec")
BIN = os.environ.get("GEO_BIN", "cavity-opt")
RANKS, PIN_CORES = 12, "12-23"

LX = 800.0
NPERT = 64
NSTATE = 128
NSCREEN = 40
PERM_N = 400
xp = np.linspace(-LX/2, LX/2, NPERT); yp = xp.copy()
xs = np.linspace(-LX/2, LX/2, NSTATE); ys = xs.copy()
x400 = np.linspace(-LX/2, LX/2, PERM_N); y400 = x400.copy()
XP, YP = np.meshgrid(xp, yp)
XS, YS = np.meshgrid(xs, ys)

L_CORR = 120.0
S_MAX_RANGE = (0.15, 0.5)
CLIP_MIN, CLIP_MAX = 2.7e-16, 4.3e-12
Q_RANGE = (1.0, 3.5)
TINJ_RANGE = (293.0, 313.0)
WELL_Y_CHOICES = (50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0)
LZ_THICKNESS_M = 5.0
DT0, DT_GROWTH = 1000, 2.0
CASE_TIMEOUT = 1200

PP_CEILING = 65.0
MAX_Z_RETRY = 40

def _CEIL(sp):

    return PP_CEILING

N_SCENARIOS, N_REAL = 5600, 5
GOLD_SCENARIOS, GOLD_REAL = 10, 200
SMOKE_SCEN, SMOKE_REAL = 4, 2

H, W = 160, 128
END_TIME = 3.16e8
XG, YG = np.meshgrid(np.linspace(-400, 400, W), np.linspace(-500, 500, H))
T_INIT = 354.15 - 0.023*YG
P_INIT = 30e6 - 9810.0*YG

import sys
sys.path.insert(0, str(BD))
from generate_uq_dataset import read_pml, write_pml, upsample, coarsen
from build_prior_field import build_heatmap_struct_logk, ellipse_kernel, fracture_mask
from confidence_model import sample_confidence_cfg, eval_confidence, cfg_desc, confidence_to_S
from injectivity_screen import predicted_pp
from scipy.ndimage import gaussian_filter

def get_chol():
    f = DATA / "L_chol64.npy"
    if f.exists():
        return np.load(f)
    pts = np.column_stack([XP.ravel(), YP.ravel()])
    d2 = ((pts[:, None, :] - pts[None, :, :])**2).sum(-1)
    L = np.linalg.cholesky(np.exp(-0.5*d2/L_CORR**2) + 1e-8*np.eye(NPERT*NPERT))

    tmp = f.with_suffix(f".npy.{os.getpid()}.tmp")
    np.save(tmp, L)
    os.replace(tmp, f)
    return L

def frac_density_full(mask):

    Em = gaussian_filter(mask.astype(float), 25)
    return (Em - Em.min()) / (Em.max() - Em.min() + 1e-12)

def to_grid_field(px, py, v):
    from scipy.interpolate import griddata
    g = griddata(np.column_stack([px, py]), v, (XG, YG), method="linear")
    hole = np.isnan(g)
    if hole.any():
        g[hole] = griddata(np.column_stack([px, py]), v, (XG[hole], YG[hole]), method="nearest")
    return g.astype(np.float32)

def sub_once(t, old, new, tag):
    if t.count(old) != 1:
        raise RuntimeError(f"patch {tag}")
    return t.replace(old, new)

def build_case_template():

    t = TEMPLATE.read_text()
    t = re.sub(r"(\[point_sample1\][\s\S]*?execute_on = )FINAL",
               r"\1'TIMESTEP_END'", t, count=1)
    return t

def replace_top(t, key, val):
    pat = re.compile(rf"^({re.escape(key)}\s*=\s*).*$", re.MULTILINE)
    new, n = pat.subn(lambda m: m.group(1) + str(val), t, count=1)
    if n != 1:
        raise RuntimeError(f"var {key}")
    return new

def run_case(cid, ifile, cases, logs, ranks, pin):
    env = dict(os.environ)
    env["PATH"] = str(Path(MPIEXEC).parent) + ":" + env.get("PATH", "")
    for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
        env[f"{v}_NUM_THREADS"] = "1"
    cmd = ["taskset", "-c", pin, MPIEXEC, "-n", str(ranks), BIN,
           "-i", str(ifile), f"Outputs/file_base={cases/cid}"]
    t0 = time.time()
    rc = -9
    with (logs / f"{cid}.log").open("w") as lf:
        try:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                 cwd=str(cases), env=env, timeout=CASE_TIMEOUT)
        except subprocess.TimeoutExpired:
            pass
    return rc, round(time.time() - t0, 1)

def collect_and_clean(cid, cases, fields_dir, z, rc=0):

    if rc != 0:
        return False
    sc = cases / f"{cid}.csv"
    if not sc.exists():
        return False
    try:
        srows = list(csv.DictReader(open(sc)))
        if not srows or abs(float(srows[-1]["time"]) - END_TIME) > 1e-3*END_TIME:
            return False
    except Exception:
        return False
    finals = [f for f in sorted(glob.glob(str(cases / f"{cid}_point_sample1_*.csv")))
              if os.path.getsize(f) > 100]
    if not finals:
        return False
    rows = list(csv.DictReader(open(finals[-1])))
    px = np.array([float(r["x"]) for r in rows]); py = np.array([float(r["y"]) for r in rows])
    T = np.array([float(r["T"]) for r in rows]); p = np.array([float(r["pp"]) for r in rows])
    fld = np.stack([T_INIT - to_grid_field(px, py, T),
                    (to_grid_field(px, py, p) - P_INIT)/1e6], 0)
    if fld.shape != (2, H, W) or not np.isfinite(fld).all():
        return False
    np.savez_compressed(fields_dir / f"{cid}.npz", fields=fld.astype(np.float32),
                        z=z.astype(np.float32))
    for f in (glob.glob(str(cases / f"{cid}_point_sample1_*.csv"))
              + [str(cases / f"{cid}_perm.i")]):
        try:
            os.remove(f)
        except OSError:
            pass
    return True

def scenario_params(sid, gold=False):
    rng = np.random.default_rng(910000 + sid if gold else 110000 + sid)
    return dict(
        img=int(rng.integers(1, N_IMAGES + 1)),
        s_max=round(float(rng.uniform(*S_MAX_RANGE)), 3),
        well_y=float(rng.choice(WELL_Y_CHOICES)),
        q=round(float(rng.uniform(*Q_RANGE)), 2),
        t_inj=round(float(rng.uniform(*TINJ_RANGE)), 1),
        rng=rng)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--ranks", type=int, default=RANKS)
    ap.add_argument("--pin", default=PIN_CORES)
    args = ap.parse_args()
    k, K = (int(v) for v in args.shard.split("/"))
    if args.smoke:
        n_scen, n_real, tag = SMOKE_SCEN, SMOKE_REAL, "smoke"
    elif args.gold:
        n_scen, n_real, tag = GOLD_SCENARIOS, GOLD_REAL, "gold"
    else:
        n_scen, n_real, tag = N_SCENARIOS, N_REAL, "train"
    tag = globals().get("TAG_OVERRIDE") or tag

    inputs = DATA/"inputs"; cases = DATA/"cases"; logs = DATA/"logs"; fields = DATA/"fields"
    for d in (inputs, cases, logs, fields):
        d.mkdir(parents=True, exist_ok=True)
    L = get_chol()
    ek = ellipse_kernel()
    tmpl = build_case_template()
    meta_csv = DATA / f"metadata_{tag}_shard{k}.csv"
    meta_fields = ["case_id", "scen", "real", "img", "s_max", "well_y", "well_x", "q", "t_inj",
                   "prod_dd", "prod_p_mpa",
                   "rich", "cbar", "scen_pp_pred", "z_rejects", "clip_rate",
                   "rc", "wall_s", "collected", "cfg"]
    if not meta_csv.exists():
        with meta_csv.open("w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=meta_fields).writeheader()

    def meta_append(row):
        with meta_csv.open("a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=meta_fields).writerow(row)

    failed_cids = set()
    if meta_csv.exists():
        with meta_csv.open() as fh:
            for row in csv.DictReader(fh):
                if str(row.get("rc")) in ("-9", "1"):
                    failed_cids.add(row["case_id"])
    if failed_cids:
        print(f"[resume] 跳过 {len(failed_cids)} 个历史失败 case(超时/物理失败)", flush=True)

    n_scen_skipped = 0
    for sid in range(n_scen):
        if sid % K != k:
            continue
        sp = scenario_params(sid, gold=args.gold)
        scen_id = f"{tag}{sid:05d}"
        mesh = MESH_DIR / f"mesh_{sp['img']}.msh"
        img = IMG_DIR / f"nf_slice_{sp['img']:04d}.png"
        if not mesh.exists() or not img.exists():
            print(f"[skip] {scen_id}: 缺 mesh/img {sp['img']}", flush=True)
            continue

        lg_nat, xf, yf, recipe = build_heatmap_struct_logk(img, sp["rng"], ek)
        lg10 = lg_nat / np.log(10.0)
        M_400 = coarsen(lg10, PERM_N)
        M_state = upsample(M_400, x400, y400, xs, ys)
        M_pert = upsample(M_400, x400, y400, xp, yp)
        M_screen = coarsen(lg10, NSCREEN)

        mask = fracture_mask(img)
        Ef = frac_density_full(mask)
        E_400 = coarsen(Ef, PERM_N)
        cfg = sample_confidence_cfg(sp["rng"])
        C_state = eval_confidence(cfg, XS, YS, upsample(E_400, x400, y400, xs, ys))
        C_pert = eval_confidence(cfg, XP, YP, upsample(E_400, x400, y400, xp, yp))
        S_pert = confidence_to_S(C_pert, sp["s_max"])

        from injectivity_screen import P_BASE

        _pos = sp.get("_sweep_positions")
        _cand = _pos if _pos else [(sp.get("well_x", 0.0), sp["well_y"])]
        pp25 = max(predicted_pp(M_screen, prod_y=_wy, rate=2.5, prod_x=_wx) for _wx, _wy in _cand)

        _ceil = _CEIL(sp)
        q_adm = 2.5*(_ceil - P_BASE)/(pp25 - P_BASE) if pp25 > P_BASE else Q_RANGE[1]
        if q_adm < Q_RANGE[0]:
            n_scen_skipped += 1
            print(f"[screen] {scen_id}: 最低流量也不可行 (pp@2.5={pp25:.1f}) → 跳过", flush=True)
            continue
        q_hi = min(Q_RANGE[1], q_adm)
        if globals().get("FIXED_Q_NO_RESAMPLE"):

            sp["q"] = round(float(min(sp["q"], q_hi)), 2)
        else:
            sp["q"] = round(float(sp["rng"].uniform(Q_RANGE[0], q_hi)), 2)
        pp_pred = predicted_pp(M_screen, prod_y=sp["well_y"], rate=sp["q"], prod_x=sp.get("well_x", 0.0))

        scen_npz = inputs / f"{scen_id}.npz"
        if not scen_npz.exists():
            np.savez_compressed(scen_npz,
                                M=M_state.astype(np.float32),
                                C=C_state.astype(np.float32),
                                s_max=sp["s_max"], q=sp["q"], t_inj=sp["t_inj"],
                                well_y=sp["well_y"], well_x=sp.get("well_x", 0.0), ell=L_CORR, img=sp["img"],
                                prod_dd=(np.nan if sp.get("prod_dd") is None else sp["prod_dd"]),
                                prod_p_mpa=(np.nan if sp.get("prod_p_mpa") is None else sp["prod_p_mpa"]),
                                rich=cfg.get("rich", -1.0), recipe=recipe,
                                cfg_desc=cfg_desc(cfg))

        _nom_last = bool(globals().get("NOMINAL_LAST", False))
        for r in range(n_real):
            _is_nom = _nom_last and r == n_real - 1
            cid = f"{scen_id}_rnom" if _is_nom else f"{scen_id}_r{r:03d}"

            _zsid = globals().get("Z_SID_OVERRIDE")
            zrng = np.random.default_rng(770000 + (sid if _zsid is None else _zsid)*1000 + r)
            z = np.zeros(NPERT*NPERT) if _is_nom else zrng.standard_normal(NPERT*NPERT)
            if (fields / f"{cid}.npz").exists() or cid in failed_cids:
                continue
            rejects = 0
            while True:
                dev = (S_pert.ravel() * (L @ z)).reshape(NPERT, NPERT)
                m40 = coarsen(M_pert + dev, NSCREEN) if NPERT % NSCREEN == 0 else \
                    upsample(M_pert + dev, xp, yp,
                             np.linspace(-LX/2, LX/2, NSCREEN), np.linspace(-LX/2, LX/2, NSCREEN))

                if globals().get("NO_Z_RESAMPLE"):
                    break
                if predicted_pp(m40, prod_y=sp["well_y"], rate=sp["q"],
                                prod_x=sp.get("well_x", 0.0)) < _CEIL(sp):
                    break
                rejects += 1
                if rejects >= MAX_Z_RETRY:
                    break
                z = zrng.standard_normal(NPERT*NPERT)
            if rejects >= MAX_Z_RETRY:
                print(f"[screen] {cid}: 实现级筛查 {MAX_Z_RETRY} 次未过 → 跳过", flush=True)
                meta_append(dict(case_id=cid, scen=scen_id, real=r, img=sp["img"],
                                 s_max=sp["s_max"], well_y=sp["well_y"], well_x=sp.get("well_x", 0.0), q=sp["q"],
                                 t_inj=sp["t_inj"],
                                 prod_dd=sp.get("prod_dd"), prod_p_mpa=sp.get("prod_p_mpa"),
                                 rich=round(cfg.get("rich", -1), 3),
                                 cbar=round(float(C_state.mean()), 3),
                                 scen_pp_pred=round(pp_pred, 1), z_rejects=rejects,
                                 clip_rate=None, rc=None, wall_s=None, collected=False,
                                 cfg=cfg_desc(cfg)))
                continue

            m_fine = M_400 + upsample(dev, xp, yp, x400, y400)
            clip_rate = float(((m_fine < np.log10(CLIP_MIN)) |
                               (m_fine > np.log10(CLIP_MAX))).mean())
            k400 = np.clip(10.0**m_fine, CLIP_MIN, CLIP_MAX)
            perm = cases / f"{cid}_perm.i"
            write_pml(k400, x400, y400, perm)
            bh = cases / f"{cid}.bh"
            _wx = sp.get("well_x", 0.0)

            _rw = 0.1 if "prod_line_length" in tmpl else 0.2
            if "prod_line_length" in tmpl:
                bh.write_text(f"{_rw} {_wx:.6f} {sp['well_y']:.6f} 0\n")
            else:
                bh.write_text(f"{_rw} {_wx:.6f} {sp['well_y']:.6f} 1\n"
                              f"{_rw} {_wx:.6f} {sp['well_y']:.6f} 0\n")
            t = tmpl
            t = replace_top(t, "mesh_file", str(mesh))
            t = replace_top(t, "perm_block_file", f"'{perm}'")
            t = replace_top(t, "prod_bh_file", str(bh))
            t = replace_top(t, "prod_pt", f"'{sp.get('well_x', 0.0):g} {sp['well_y']:g} 0'")

            if sp.get("prod_p_mpa") is not None:
                t = replace_top(t, "prod_p",
                                f"{sp['prod_p_mpa']*1e6:.6g}  # v4-r5: p0(y_prod) - dd")
            t = replace_top(t, "inj_T", f"{sp['t_inj']:g}")
            t = replace_top(t, "injection_mass_rate", f"{sp['q']:g}")
            t = replace_top(t, "Lz", f"{LZ_THICKNESS_M:g}")
            t = replace_top(t, "dt0", f"{DT0:g}")
            t = replace_top(t, "dt_growth", f"{DT_GROWTH:g}")
            ifile = cases / f"{cid}.i"
            ifile.write_text(t)
            row = dict(case_id=cid, scen=scen_id, real=r, img=sp["img"], s_max=sp["s_max"],
                       well_y=sp["well_y"], well_x=sp.get("well_x", 0.0), q=sp["q"], t_inj=sp["t_inj"],
                       prod_dd=sp.get("prod_dd"), prod_p_mpa=sp.get("prod_p_mpa"),
                       rich=round(cfg.get("rich", -1), 3), cbar=round(float(C_state.mean()), 3),
                       scen_pp_pred=round(pp_pred, 1), z_rejects=rejects,
                       clip_rate=round(clip_rate, 4), rc=None, wall_s=None,
                       collected=None, cfg=cfg_desc(cfg))
            if not args.generate_only:
                rc, wall = run_case(cid, ifile, cases, logs, args.ranks, args.pin)
                ok = collect_and_clean(cid, cases, fields, z, rc)
                row.update(rc=rc, wall_s=wall, collected=ok)
                print(f"  {cid}: rc={rc} wall={wall}s npz={ok} rej={rejects}", flush=True)
            meta_append(row)
    print(f"\ndone (shard {k}/{K}); 场景级筛掉 {n_scen_skipped}", flush=True)

if __name__ == "__main__":
    main()
