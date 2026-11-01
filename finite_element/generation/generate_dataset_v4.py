# SPDX-License-Identifier: LGPL-2.1-only
import argparse, csv, glob, hashlib, json, os, re, sys
from pathlib import Path

import numpy as np

import netCDF4

BD = Path(__file__).resolve().parent
sys.path.insert(0, str(BD))
import generate_dataset_v3 as G

DATA4 = BD / "dataset_v4_coussy_r5"
PREVIOUS_DATA4 = BD / "dataset_v4_coussy_r4"
LEGACY_DATA4 = BD / "dataset_v4_coussy"

PHYSICS_VERSION = "v4-prodbhp-variable-r5"

R3_K_MIN = 4e-15
R3_CLIP_MIN = 4e-16
R3_CHANNEL_KAP2 = (0.08, 0.20)

import build_prior_field as _bpf
_bpf.K_MIN = R3_K_MIN
_bpf.LOGK_MIN = np.log(R3_K_MIN)
_bpf.CHANNEL_KAP2_RANGE = R3_CHANNEL_KAP2
G.CLIP_MIN = R3_CLIP_MIN

PHI_REF = 3.9e-2
CAP_BED_PHI_REF = 1.0e-3
BIOT_ALPHA = 0.58
YOUNGS_MODULUS = 31.0e9
POISSON_RATIO = 0.32
SOLID_LINEAR_THERMAL_EXPANSION = 1.0e-5
DRAINED_BULK_MODULUS = YOUNGS_MODULUS / (3.0 * (1.0 - 2.0 * POISSON_RATIO))
GRAIN_BULK_MODULUS = DRAINED_BULK_MODULUS / (1.0 - BIOT_ALPHA)
POROSITY_EPV_COEFF = BIOT_ALPHA - PHI_REF
POROSITY_P_COEFF = POROSITY_EPV_COEFF / GRAIN_BULK_MODULUS
POROSITY_T_COEFF = (
    -3.0 * SOLID_LINEAR_THERMAL_EXPANSION * POROSITY_EPV_COEFF
)
CAP_BED_POROSITY_EPV_COEFF = BIOT_ALPHA - CAP_BED_PHI_REF
CAP_BED_POROSITY_P_COEFF = CAP_BED_POROSITY_EPV_COEFF / GRAIN_BULK_MODULUS
CAP_BED_POROSITY_T_COEFF = (
    -3.0 * SOLID_LINEAR_THERMAL_EXPANSION * CAP_BED_POROSITY_EPV_COEFF
)

WELL_Y_RANGE = (0.0, 360.0)
WELL_X_RANGE = (-180.0, 180.0)
INJ_XY = (0.0, -250.0)
DIST_RANGE = (250.0, 620.0)
SEED_MAIN = 410000
SEED_TRUTH = 730000
N_MAIN_DEFAULT = 20000

N_GEO = 400

_gx, _gy = np.meshgrid(np.linspace(-180.0, 180.0, 7), np.linspace(0.0, 360.0, 7))
_d = np.hypot(_gx - INJ_XY[0], _gy - INJ_XY[1])
XY_GRID = [(round(float(x), 1), round(float(y), 1))
           for x, y, d in zip(_gx.ravel(), _gy.ravel(), _d.ravel())
           if DIST_RANGE[0] <= d <= DIST_RANGE[1]]
N_Y = len(XY_GRID)
N_TRUTH = 20
SNAP_MIN = 6
_FAIL_MAX = 5
_FAIL_STREAK = 0

PP_KILL_PA = 7.2e7
PP_GUARD_POLL_S = 15

SCREEN_NC = 120

SCREEN_FIT_A, SCREEN_FIT_B = 54.6, -684.1
SCREEN_GAMMA = 1.3
PP_CEILING_R2 = 60.0

PROD_DD_RANGE = (4.0, 14.0)
SEED_PRODDD = 990000

PROD_P_REF_MPA = 20.0
PP_IC_PA, DPDY_PA_M, Y_REF_M = 30e6, -9810.0, 0.0

SCREEN_BHP_SLOPE = 1.0

def draw_prod_dd(sid, truth=False):

    base = SEED_PRODDD + (10**6 if truth else 0)
    return round(float(np.random.default_rng(base + int(sid)).uniform(*PROD_DD_RANGE)), 3)

def prod_p_mpa_of(well_y, dd_mpa):

    return (PP_IC_PA + DPDY_PA_M * (float(well_y) - Y_REF_M)) / 1e6 - float(dd_mpa)

def pp_shift_mpa(prod_p_mpa):

    return (float(prod_p_mpa) - PROD_P_REF_MPA) * SCREEN_BHP_SLOPE

def eff_ceiling(sp_or_prod_p, default=PP_CEILING_R2):

    pp = sp_or_prod_p.get("prod_p_mpa") if isinstance(sp_or_prod_p, dict) else sp_or_prod_p
    if pp is None:
        return default
    return default - pp_shift_mpa(pp)

def _inj_proxy_fast(lg10_coarse, nc, inj=(0.0, -250.0), prod=(0.0, 150.0)):

    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    LX = 800.0
    xc = np.linspace(-LX/2, LX/2, nc); yc = np.linspace(-LX/2, LX/2, nc)
    k = 10.0 ** np.asarray(lg10_coarse, float)
    idx = np.arange(nc * nc).reshape(nc, nc)
    rows, cols, vals = [], [], []
    for (pa, pb, ta) in (
        (idx[:-1, :].ravel(), idx[1:, :].ravel(),
         (2.0 / (1.0 / k[:-1, :] + 1.0 / k[1:, :])).ravel()),
        (idx[:, :-1].ravel(), idx[:, 1:].ravel(),
         (2.0 / (1.0 / k[:, :-1] + 1.0 / k[:, 1:])).ravel()),
    ):
        rows += [pa, pb, pa, pb]; cols += [pa, pb, pb, pa]
        vals += [ta, ta, -ta, -ta]
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(nc * nc, nc * nc)).tocsr()
    ii = int(np.argmin(np.abs(yc - inj[1]))); ji = int(np.argmin(np.abs(xc - inj[0])))
    ip = int(np.argmin(np.abs(yc - prod[1]))); jp = int(np.argmin(np.abs(xc - prod[0])))
    src, snk = int(idx[ii, ji]), int(idx[ip, jp])
    A = A.tolil(); A[snk, :] = 0; A[snk, snk] = 1.0
    b = np.zeros(nc * nc); b[src] = 1.0
    return float(spsolve(A.tocsr(), b)[src])

def predicted_pp_r2(lg10_coarse, prod_y=150.0, rate=2.5, prod_x=0.0):

    import injectivity_screen as scr
    a = np.asarray(lg10_coarse, float)
    if a.shape[0] != SCREEN_NC:
        src_ax = np.linspace(-400.0, 400.0, a.shape[0])
        dst_ax = np.linspace(-400.0, 400.0, SCREEN_NC)
        a = G.upsample(a, src_ax, src_ax, dst_ax, dst_ax)
    raw = _inj_proxy_fast(a, SCREEN_NC, prod=(prod_x, prod_y))
    op25 = max(SCREEN_FIT_A * np.log10(raw) + SCREEN_FIT_B - scr.P_BASE, 0.0)
    return scr.P_BASE + op25 * (rate / 2.5) ** SCREEN_GAMMA

CURVE_COLS = ["T_prod", "pp_inj", "pp_prod", "Qth_MW", "m_prod_kgs",
              "cum_heat_GJ", "pressure_drop_MPa"]

_tmpl_v3 = G.build_case_template
_get_chol_v3 = G.get_chol
_chol_mem = None

def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _chol_cache_path():

    current = DATA4 / "L_chol64.npy"
    previous = PREVIOUS_DATA4 / "L_chol64.npy"
    legacy = LEGACY_DATA4 / "L_chol64.npy"
    if current.exists():
        return current
    if previous.exists():
        return previous
    if legacy.exists():
        return legacy
    return None

def get_chol_v4():

    global _chol_mem
    if _chol_mem is not None:
        return _chol_mem
    cache = _chol_cache_path()
    if cache is not None:
        _chol_mem = np.load(cache)
    else:
        _chol_mem = _get_chol_v3()
    expected = (G.NPERT * G.NPERT, G.NPERT * G.NPERT)
    if _chol_mem.shape != expected or not np.isfinite(_chol_mem).all():
        raise RuntimeError(f"Cholesky 缓存不合格: shape={_chol_mem.shape}")
    return _chol_mem

def build_case_template_v4(snap=False):

    t = _tmpl_v3()

    t = G.replace_top(
        t, "epv_coeff",
        f"{POROSITY_EPV_COEFF:.12g} # v4: alpha - phi_ref (current-volume porosity)",
    )
    t = G.replace_top(
        t, "P_coeff",
        f"{POROSITY_P_COEFF:.12g} # v4: (alpha - phi_ref) / K_s",
    )
    t = G.replace_top(
        t, "T_coeff",
        f"{POROSITY_T_COEFF:.12g} # v4: -3 alpha_T^s (alpha - phi_ref)",
    )
    t = G.replace_top(
        t, "epv_coeff_cap_bed",
        f"{CAP_BED_POROSITY_EPV_COEFF:.12g} # v4: alpha - phi_ref (cap/bed)",
    )
    t = G.replace_top(
        t, "P_coeff_cap_bed",
        f"{CAP_BED_POROSITY_P_COEFF:.12g} # v4: (alpha - phi_ref) / K_s (cap/bed)",
    )
    t = G.replace_top(
        t, "T_coeff_cap_bed",
        f"{CAP_BED_POROSITY_T_COEFF:.12g} # v4: -3 alpha_T^s (alpha - phi_ref; cap/bed)",
    )

    assert "[pp_prod]" not in t
    t = t.replace("  [pp_inj]\n", "  [pp_prod]\n    type = PointValue\n    variable = pp\n"
                                  "    point = ${prod_pt}\n  []\n  [pp_inj]\n", 1)

    assert "dtmax" not in t
    t = t.replace("  end_time = ${end_time}", "  end_time = ${end_time}\n  dtmax = 1.58e7  # 半年(v4 井曲线分辨率)", 1)
    old = "  [csv_out]\n    type = CSV\n    execute_on = FINAL\n  []"
    assert old in t, "csv_out 块与预期不符"
    new = ("  [csv_out]\n    type = CSV\n    execute_on = 'INITIAL TIMESTEP_END FINAL'\n"
           "    execute_vector_postprocessors_on = FINAL\n  []")
    if snap:

        new += ("\n  [snap_out]\n    type = Exodus\n"
                "    show = 'T pp'\n"
                "    time_step_interval = 3\n"
                "    execute_on = 'TIMESTEP_END FINAL'\n  []")
    t = t.replace(old, new, 1)
    return t

def _physics_manifest():

    rendered = build_case_template_v4(snap=True)
    chol = _chol_cache_path()
    return {
        "physics_version": PHYSICS_VERSION,
        "dataset_root": DATA4.name,
        "previous_dataset_root": PREVIOUS_DATA4.name,
        "legacy_dataset_root": LEGACY_DATA4.name,
        "porosity_definition": "current bulk volume (Eulerian/local porosity)",
        "lagrangian_to_current_linearization": "Phi=J*phi, J≈1+eps_v",
        "porosity_pressure_reference": "spatial pre-injection pressure p0(y)",
        "porosity_temperature_reference": "spatial pre-injection temperature T0(y)",
        "mechanical_stress_free_temperature": "spatial pre-injection temperature T0(y)",
        "lower_dimensional_fracture_elements": False,
        "fracture_representation": "static equivalent matrix permeability only",
        "bottom_mechanical_boundary": "uy=0 on base; ux=0 at bottom-center node",
        "solid_bulk_density_for_gravity_kg_m3": 2772.68093782,
        "heatmap_k_min_m2": R3_K_MIN,
        "heatmap_k_max_m2": 1.35e-12,
        "clip_range_m2": [R3_CLIP_MIN, 4.3e-12],
        "channel_kap2_range": list(R3_CHANNEL_KAP2),
        "well_x_range_m": list(WELL_X_RANGE),
        "well_y_range_m": list(WELL_Y_RANGE),
        "injector_xy_m": list(INJ_XY),
        "interwell_distance_range_m": list(DIST_RANGE),
        "sweep_xy_grid_m": [list(xy) for xy in XY_GRID],
        "sweep_position_count": N_Y,

        "producer_bhp_mode": "sampled_drawdown",
        "producer_drawdown_range_mpa": list(PROD_DD_RANGE),
        "producer_drawdown_seed": SEED_PRODDD,
        "producer_bhp_formula": "prod_p = pp_ic + dpdy*(y_prod - y_ref) - drawdown",
        "producer_bhp_reference_mpa": PROD_P_REF_MPA,
        "screen_bhp_slope": SCREEN_BHP_SLOPE,
        "phi_ref": PHI_REF,
        "cap_bed_phi_ref": CAP_BED_PHI_REF,
        "biot_alpha": BIOT_ALPHA,
        "youngs_modulus_Pa": YOUNGS_MODULUS,
        "poisson_ratio": POISSON_RATIO,
        "drained_bulk_modulus_Pa": DRAINED_BULK_MODULUS,
        "grain_bulk_modulus_Pa": GRAIN_BULK_MODULUS,
        "epv_coeff": POROSITY_EPV_COEFF,
        "P_coeff_per_Pa": POROSITY_P_COEFF,
        "T_coeff_per_K": POROSITY_T_COEFF,
        "cap_bed_epv_coeff": CAP_BED_POROSITY_EPV_COEFF,
        "cap_bed_P_coeff_per_Pa": CAP_BED_POROSITY_P_COEFF,
        "cap_bed_T_coeff_per_K": CAP_BED_POROSITY_T_COEFF,
        "producer_completion_note": (
            "single .bh point at z=0 plus prod_line_length=1.0 m in the unit-thickness 2D "
            "model, i.e. the producer fully penetrates the Lz=5 m stage. r3 and earlier wrote "
            "two points at z=1 and z=0; the z=1 point lies outside the strictly 2D mesh and was "
            "silently dropped, halving the Peaceman well index."),
        "producer_line_length_m": 1.0,
        "borehole_radius_m": 0.1,
        "peaceman_strain_at_nearest_qp": False,
        "peaceman_strain_mapping_note": (
            "nearest-QP material is not enabled: the installed MOOSE build segfaults during "
            "INITIAL elemental AuxKernel execution; this model's Peaceman well uses qp "
            "permeability and nodal fluid mobility, not nodal strain-dependent porosity"
        ),
        "producer_bhp_initialization": "local hydrostatic p0(x_prod,y_prod)",
        "Qth_definition": "produced enthalpy minus m_prod times injected-fluid specific enthalpy",
        "template_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "cholesky_sha256": _sha256_file(chol) if chol is not None else None,
    }

def ensure_physics_manifest():

    DATA4.mkdir(parents=True, exist_ok=True)
    manifest = DATA4 / "PHYSICS_MANIFEST.json"
    expected = _physics_manifest()
    if manifest.exists():
        found = json.loads(manifest.read_text())
        if found != expected:
            raise RuntimeError(
                f"{manifest} 与当前 {PHYSICS_VERSION} 不一致；"
                "为防止混合物理版本，拒绝续跑"
            )
        return
    unexpected = [p.name for p in DATA4.iterdir() if p.name != "L_chol64.npy"]
    if unexpected:
        raise RuntimeError(
            f"{DATA4} 已有文件但没有 PHYSICS_MANIFEST.json: {unexpected[:5]}；"
            "为防止混合物理版本，拒绝写入"
        )

    tmp = manifest.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, manifest)
    if json.loads(manifest.read_text()) != expected:
        raise RuntimeError(f"{manifest} 写入后与本进程期望不一致（并发冷启动？）")

def _draw_wellpos(rng):

    for _ in range(256):
        wx = float(rng.uniform(*WELL_X_RANGE)); wy = float(rng.uniform(*WELL_Y_RANGE))
        d = float(np.hypot(wx - INJ_XY[0], wy - INJ_XY[1]))
        if DIST_RANGE[0] <= d <= DIST_RANGE[1]:
            return round(wx, 1), round(wy, 1)
    return 0.0, 150.0

def scenario_params_v4(sid, gold=False):

    rng = np.random.default_rng(SEED_MAIN + sid)
    img = int(rng.integers(1, G.N_IMAGES + 1))
    s_max = round(float(rng.uniform(*G.S_MAX_RANGE)), 3)
    wx, wy = _draw_wellpos(rng)
    dd = draw_prod_dd(sid)
    return dict(
        img=img, s_max=s_max, well_y=wy, well_x=wx,
        q=round(float(rng.uniform(*G.Q_RANGE)), 2),
        t_inj=round(float(rng.uniform(*G.TINJ_RANGE)), 1),
        prod_dd=dd, prod_p_mpa=round(prod_p_mpa_of(wy, dd), 4),
        rng=rng)

def scenario_params_sweep(sid, gold=False):

    geo, yj = divmod(int(sid), N_Y)
    sp = scenario_params_v4(geo)
    sp["well_x"], sp["well_y"] = XY_GRID[yj]

    sp["prod_p_mpa"] = round(prod_p_mpa_of(sp["well_y"], sp["prod_dd"]), 4)
    G.Z_SID_OVERRIDE = geo
    G.FIXED_Q_NO_RESAMPLE = True
    G.NO_Z_RESAMPLE = True
    sp["_geo"] = geo

    feas = _sweep_feasible_positions(geo, sp["img"])
    if (sp["well_x"], sp["well_y"]) in feas:
        sp["_sweep_positions"] = feas
    else:
        sp["_sweep_positions"] = [(sp["well_x"], sp["well_y"])]
    return sp

_SWEEP_FEAS_CACHE = {}

def _sweep_feasible_positions(geo, img_id):

    if geo in _SWEEP_FEAS_CACHE:
        return _SWEEP_FEAS_CACHE[geo]
    import injectivity_screen as scr
    sp2 = scenario_params_v4(geo)
    lg_nat, _, _, _ = G.build_heatmap_struct_logk(
        G.IMG_DIR / f"nf_slice_{sp2['img']:04d}.png", sp2["rng"], G.ellipse_kernel())
    m_scr = G.coarsen(lg_nat / np.log(10.0), SCREEN_NC)
    feas = []
    for wx, wy in XY_GRID:
        pp25 = predicted_pp_r2(m_scr, prod_y=wy, rate=2.5, prod_x=wx)

        ceil_wy = eff_ceiling(prod_p_mpa_of(wy, sp2["prod_dd"]))
        q_adm = 2.5 * (ceil_wy - scr.P_BASE) / max(pp25 - scr.P_BASE, 1e-9)
        if q_adm >= G.Q_RANGE[0]:
            feas.append((wx, wy))
    _SWEEP_FEAS_CACHE[geo] = feas
    print(f"[sweep-feas] geo={geo}: {len(feas)}/{len(XY_GRID)} 位置可行", flush=True)
    return feas

def scenario_params_truth(sid, gold=False):

    rng = np.random.default_rng(SEED_TRUTH + int(sid))
    img = int(rng.integers(1, G.N_IMAGES + 1))
    s_max = round(float(rng.uniform(*G.S_MAX_RANGE)), 3)
    wx, wy = _draw_wellpos(rng)
    dd = draw_prod_dd(sid, truth=True)
    return dict(img=img, s_max=s_max, well_y=wy, well_x=wx,
                q=round(float(rng.uniform(*G.Q_RANGE)), 2),
                t_inj=round(float(rng.uniform(*G.TINJ_RANGE)), 1),
                prod_dd=dd, prod_p_mpa=round(prod_p_mpa_of(wy, dd), 4), rng=rng)

_run_v3 = G.run_case

def run_case_v4(cid, ifile, cases, logs, ranks, pin):

    import subprocess, time, os
    env = dict(os.environ)
    env["PATH"] = str(Path(G.MPIEXEC).parent) + ":" + env.get("PATH", "")
    for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
        env[f"{v}_NUM_THREADS"] = "1"
    cmd = ["taskset", "-c", pin, G.MPIEXEC, "-n", str(ranks), G.BIN,
           "-i", str(ifile), f"Outputs/file_base={cases/cid}",
           f"Outputs/snap_out/file_base={cases/cid}_snap"]

    import signal, threading
    t0 = time.time(); rc = -9
    ppkill = []

    def _pp_watchdog(proc):
        csvf = cases / f"{cid}.csv"
        col = None
        while proc.poll() is None:
            time.sleep(PP_GUARD_POLL_S)
            try:
                lines = csvf.read_text().strip().splitlines()
                if len(lines) < 2:
                    continue
                if col is None:
                    col = lines[0].split(",").index("pp_inj")
                v = float(lines[-1].split(",")[col])
                if v > PP_KILL_PA:
                    ppkill.append(v)
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        pass
                    return
            except (OSError, ValueError, IndexError):
                continue

    with (logs / f"{cid}.log").open("w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                cwd=str(cases), env=env, start_new_session=True)
        threading.Thread(target=_pp_watchdog, args=(proc,), daemon=True).start()
        try:
            rc = proc.wait(timeout=G.CASE_TIMEOUT)
        except subprocess.TimeoutExpired:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(proc.pid), sig)
                except (ProcessLookupError, PermissionError):
                    break
                try:
                    proc.wait(timeout=15); break
                except subprocess.TimeoutExpired:
                    continue
            rc = -9
        except BaseException:
            try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception: pass
            raise
    if ppkill:
        (cases / f"{cid}.ppkill").write_text(f"{ppkill[0]:.6g}\n")
        print(f"  [ppguard] {cid}: pp_inj={ppkill[0]/1e6:.1f} MPa > {PP_KILL_PA/1e6:.0f} → 截停(物理拒绝)",
              flush=True)
        rc = -15
    return rc, round(time.time() - t0, 1)

_collect_v3 = G.collect_and_clean

def collect_and_clean_v4(cid, cases, fields_dir, z, rc=0):

    mk = cases / f"{cid}.ppkill"
    if mk.exists():

        print(f"  [ppguard] {cid}: 越限截停已记账(pp={mk.read_text().strip()} Pa),清理中间产物", flush=True)
        for f in sorted(cases.glob(f"{cid}*")):
            try: f.unlink()
            except OSError: pass
        return False
    ok = _collect_v3(cid, cases, fields_dir, z, rc)
    if not ok:
        return False
    f_npz = fields_dir / f"{cid}.npz"
    try:
        sc = cases / f"{cid}.csv"
        rows = list(csv.DictReader(open(sc)))
        tvec = np.array([float(r["time"]) for r in rows])
        keep = np.concatenate([[True], np.diff(tvec) > 0])

        keep &= tvec > 0
        tvec = tvec[keep]
        missing = [c for c in CURVE_COLS if c not in rows[0]]
        if missing:
            raise RuntimeError(f"曲线缺列 {missing}")
        curves = np.stack([np.array([float(r[c]) for r in rows])[keep] for c in CURVE_COLS],
                          1).astype(np.float32)
        names = list(CURVE_COLS)
        if not (np.isfinite(curves).all() and len(tvec) >= 10
                and abs(tvec[-1] - G.END_TIME) < 1e-3*G.END_TIME):
            raise RuntimeError(f"曲线不合格(nt={len(tvec)}, t_end={tvec[-1]:.3g})")
        payload = dict(np.load(f_npz))
        payload.update(curve_t=tvec.astype(np.float64), curves=curves,
                       curve_cols=np.array(names),
                       physics_version=np.array(PHYSICS_VERSION))

        efile = cases / f"{cid}_snap.e"
        if efile.exists():
            ds = netCDF4.Dataset(str(efile))
            et = np.array(ds.variables["time_whole"][:])
            names = ["".join(c.decode() for c in row if c != b"") for row in
                     ds.variables["name_nod_var"][:].data]
            iT = names.index("T"); ip = names.index("pp")
            px = np.array(ds.variables["coordx"][:]); py = np.array(ds.variables["coordy"][:])

            sel = [j for j, tj in enumerate(et) if tj > 0 and tj < 0.995*G.END_TIME]
            fs, ts = [], []
            for j in sel:
                T = np.array(ds.variables[f"vals_nod_var{iT+1}"][j])
                pp_ = np.array(ds.variables[f"vals_nod_var{ip+1}"][j])
                fs.append(np.stack([G.T_INIT - G.to_grid_field(px, py, T),
                                    (G.to_grid_field(px, py, pp_) - G.P_INIT)/1e6], 0))
                ts.append(float(et[j]))
            ds.close()
            if len(fs) < SNAP_MIN:
                raise RuntimeError(f"快照仅 {len(fs)} 帧 < {SNAP_MIN}")
            payload.update(snap_fields=np.stack(fs).astype(np.float32),
                           snap_times=np.array(ts))
            os.remove(efile)
        else:
            raise RuntimeError("缺 Exodus 快照文件")

        tmp = f_npz.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, f_npz)
        os.remove(sc)
        globals()["_FAIL_STREAK"] = 0
        return True
    except Exception as e:

        globals()["_FAIL_STREAK"] = globals().get("_FAIL_STREAK", 0) + 1
        print(f"  [collect_v4] {cid}: {e} (连续失败 {_FAIL_STREAK}/{_FAIL_MAX};删除不完整产物)", flush=True)

        for f in (f_npz, f_npz.with_suffix(".tmp.npz")):
            try: f.unlink()
            except OSError: pass
        if globals()["_FAIL_STREAK"] >= _FAIL_MAX:
            raise RuntimeError(f"收集连续失败 {_FAIL_MAX} 次 —— 疑似环境/依赖故障,终止 worker") from e
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["main", "sweep", "truth"], default="main")
    ap.add_argument("--n", type=int, default=N_MAIN_DEFAULT, help="main 模式目标场景数")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--ranks", type=int, default=6)
    ap.add_argument("--pin", default="0-5")
    a = ap.parse_args()

    G.DATA = DATA4
    DATA4.mkdir(parents=True, exist_ok=True)
    G.get_chol = get_chol_v4
    get_chol_v4()
    ensure_physics_manifest()

    G.CASE_TIMEOUT = 2400

    _fd = DATA4 / "fields"
    if _fd.exists():

        import time as _time
        _now = _time.time()
        _bad = 0
        for _f in sorted(_fd.glob("*.npz")):
            try:
                _age = _now - _f.stat().st_mtime
            except OSError:
                continue
            if _age > 172800.0 or _age < 3600.0:
                continue
            _ok = False
            for _ in range(3):
                try:
                    with np.load(_f) as _d:
                        _ok = {"curves", "curve_t", "snap_fields", "snap_times"} <= set(_d.files)
                    break
                except Exception:
                    _time.sleep(2.0)
            if not _ok:
                _f.unlink(missing_ok=True); _bad += 1
        for _t in _fd.glob("*.tmp.npz"):
            try:
                if _now - _t.stat().st_mtime > 3600.0:
                    _t.unlink(missing_ok=True); _bad += 1
            except OSError:
                pass
        if _bad:
            print(f"[resume] 清理不完整产物 {_bad} 个(将自动重跑)", flush=True)
    G.collect_and_clean = collect_and_clean_v4
    G.run_case = run_case_v4

    G.predicted_pp = predicted_pp_r2
    G.PP_CEILING = PP_CEILING_R2

    G._CEIL = lambda sp: eff_ceiling(sp, PP_CEILING_R2)
    G.NSCREEN = SCREEN_NC

    G.NOMINAL_LAST = True

    G.N_REAL = int(os.environ.get("V4_N_REAL", "16"))
    if a.mode == "main":
        G.scenario_params = scenario_params_v4
        G.N_SCENARIOS = a.n
        G.build_case_template = lambda: build_case_template_v4(snap=True)
    elif a.mode == "sweep":
        G.scenario_params = scenario_params_sweep
        G.N_SCENARIOS = N_GEO * N_Y
        G.build_case_template = lambda: build_case_template_v4(snap=True)
    else:
        G.scenario_params = scenario_params_truth
        G.N_SCENARIOS = N_TRUTH
        G.build_case_template = lambda: build_case_template_v4(snap=True)
    tagmap = {"main": "v4m", "sweep": "v4s", "truth": "v4t"}

    G.TAG_OVERRIDE = tagmap[a.mode]

    sys.argv = [sys.argv[0], "--shard", a.shard, "--ranks", str(a.ranks), "--pin", a.pin] \
               + (["--smoke"] if a.smoke else [])
    G.main()

if __name__ == "__main__":
    main()
