#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UQ Operator 数据集生成器 —— 20k 设计（2026-07-10 重规划；旧 2250 版留档 legacy/）。

核心原则: 算子泛化吃"unique 输入"数量, energy score 是 strictly proper —— 每场景少量标签即可,
预算应堆在场景数上。
  场景 = (裂隙图 i∈1..2000, r_well, 监测配置, σ, 井位y) 全部随机采样(种子可复现)
  训练:  N_SCENARIOS=4600 场景 × N_REAL=4 实现 = 18,400 次 MOOSE
  金标准: --gold → 8 个固定场景 × 200 实现 = 1,600 次(留出, 只做 CRPS/覆盖率评测)
  合计 ≈ 20,000

磁盘策略: 渗透率场写 400×400(~2.9MB); 每 case 跑完立刻把场输出插值成 npz(~1MB, 含5个时间快照+z向量)
并删除重文件(perm .i / point_sample csv)。全程 ~40GB。

用法:
  python3 generate_uq_dataset.py --smoke                      # 4场景×2实现=8 case 端到端
  python3 generate_uq_dataset.py --shard 0/2 --ranks 12 --pin 0-11  &   # 双分片并行
  python3 generate_uq_dataset.py --shard 1/2 --ranks 12 --pin 12-23 &
  python3 generate_uq_dataset.py --gold                       # 金标准集
"""
import argparse, csv, glob, os, re, subprocess, time
from pathlib import Path
import numpy as np

# ================= 配置 =================
BD = Path("/data/yl3825/OL_Geothermal_Claude/final_scheme")
DATA = BD / "dataset"
TEMPLATE = BD / "THM_template_final.i"
IMG_DIR = Path("/home/yl3825/Pycharm_projects/Journal_Inverse_problem/Geothermal_generation/Final/natural fracture networks/images")
MESH_DIR = Path("/home/yl3825/Pycharm_projects/Journal_Inverse_problem/Geothermal_generation/Final/mesh")
N_IMAGES = 2000                            # nf_slice_0001..2000 与 mesh_1..2000 一一对应
MPIEXEC = "/home/yl3825/miniforge/envs/moose/bin/mpiexec"
BIN = "/home/yl3825/projects/cavity/cavity-opt"
RANKS = 20; PIN_CORES = "0-19"             # 单流默认; 双分片用 --ranks/--pin 覆盖

NC = 40; LX = 800.0                        # kriging 粗网格
PERM_N = 400                               # 渗透率场写出分辨率(2m)
SIGMA = 0.3                                # 参考 σ（出图用）
SIGMA_RANGE = (0.15, 0.5)                  # 每场景 σ~U; 上限=收敛研究上界
L_CORR = 120.0
OBS_NOISE = 0.10                           # 仪器精度[dec]: 流量测井/封隔器~2倍内(Paillet'98), NMR类似(Coates'99)
JITTER = 1e-8                             # 纯数值稳定项(非物理nugget, 非测量噪声)
K_MIN, K_MAX = 2.7e-15, 1.35e-11           # m0 映射范围(G/I 设计点, 等效开度0.93mm)
# clip 界与映射界分离(专家意见): m0 本身顶到映射界, 波动必然越界 → clip 用更宽的物理筛查界,
# 否则边界概率质量堆积破坏高斯性。筛查界: 裂隙岩下限(Brace'80)到开度~1.4mm(立方定律)。
CLIP_MIN, CLIP_MAX = 2.7e-16, 4.3e-11      # = K_MIN/10, K_MAX*10^0.5
INJ = (0.0, -250.0)
PROD_LINE_OBS_Y = (50, 150, 250, 350)
WELL_Y_CHOICES = (50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0)
SYNC_TIMES = "1.58e7 3.16e7 7.9e7 1.58e8"
INJ_T = 296.0
INJ_RATE = 2.5                             # 模板原设计值 50/20 (case I 实测 47.1 MPa ✅)
ZERO_RESIDUAL = False                      # True = 零残差消融(y:=H·m_base → m̄=m_base, 不抽 m*):
                                           # 映射不变, 但均值通道失去"仪器处kriging鼓包"的部署真实性;
                                           # 用于消融实验"均值通道真实性重不重要"。默认走完整虚拟勘探。
LZ_THICKNESS_M = 5.0

N_SCENARIOS = 4600; N_REAL = 4             # 训练: 18,400 runs
GOLD_SCENARIOS = 8; GOLD_REAL = 200        # 金标准: 1,600 runs
SMOKE_SCEN, SMOKE_REAL = 4, 2

# 输出场网格(与 collect_fields 一致)
H, W = 160, 128
XG, YG = np.meshgrid(np.linspace(-400, 400, W), np.linspace(-500, 500, H))
T_INIT = 354.15 - 0.023*YG; P_INIT = 30e6 - 9810.0*YG
SYNC_LIST = [1.58e7, 3.16e7, 7.9e7, 1.58e8]; FINAL_TIME = 3.16e8

# ================= 场 IO =================
def read_pml(p):
    sec, cur = {}, None
    for raw in Path(p).read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"): continue
        if s in {"AXIS X", "AXIS Y", "DATA"}: cur = s; sec[cur] = []; continue
        if cur: sec[cur].append(s)
    ax = np.array(" ".join(sec["AXIS X"]).split(), float)
    ay = np.array(" ".join(sec["AXIS Y"]).split(), float)
    d = np.array(" ".join(sec["DATA"]).split(), float)
    return ax, ay, d.reshape((len(ax), len(ay)), order="F").T

def write_pml(field, ax, ay, path):
    vals = field.T.reshape(-1, order="F")
    with Path(path).open("w") as f:
        f.write("# possible-truth permeability (final_scheme 20k)\n\n")
        f.write("AXIS X\n" + " ".join(f"{v:.4f}" for v in ax))
        f.write("\n\nAXIS Y\n" + " ".join(f"{v:.4f}" for v in ay))
        f.write("\n\nDATA\n")
        for v in vals: f.write(f"{v:.6e}\n")

def upsample(coarse, xc_, yc_, xf, yf):
    from scipy.interpolate import RegularGridInterpolator
    it = RegularGridInterpolator((yc_, xc_), coarse, bounds_error=False, fill_value=None)
    XF, YF = np.meshgrid(xf, yf)
    return it(np.column_stack([YF.ravel(), XF.ravel()])).reshape(len(yf), len(xf))

def coarsen(a, n):
    f = a.shape[0]//n
    return a[:n*f, :n*f].reshape(n, f, n, f).mean((1, 3))

# ================= kriging 机器 =================
xc = np.linspace(-LX/2, LX/2, NC); yc = np.linspace(-LX/2, LX/2, NC)
XC, YC = np.meshgrid(xc, yc); PTS = np.column_stack([XC.ravel(), YC.ravel()])
d2 = ((PTS[:, None, :] - PTS[None, :, :])**2).sum(-1)
R_UNIT = np.exp(-0.5*d2/L_CORR**2)
C_PRIOR = SIGMA**2 * R_UNIT + JITTER*np.eye(NC*NC)
L_PRIOR = np.linalg.cholesky(C_PRIOR)
x400 = np.linspace(-LX/2, LX/2, PERM_N); y400 = np.linspace(-LX/2, LX/2, PERM_N)

def nearest_idx(pt):
    return int(np.argmin((PTS[:, 0]-pt[0])**2 + (PTS[:, 1]-pt[1])**2))
WELL_OBS = [nearest_idx(INJ)] + [nearest_idx((0.0, y)) for y in PROD_LINE_OBS_Y]

def sample_obs_scenario(rng):
    n_extra = int(rng.integers(0, 36))       # 0..35 个加密仪器, 连续覆盖勘探成熟度
    extra = list(rng.choice(NC*NC, n_extra, replace=False)) if n_extra else []
    return sorted(set(WELL_OBS + extra)), n_extra

def belief_state(obs_idx, sigma, m_base, rng):
    """完整虚拟勘探链(专家审阅版):
       m_star ~ N(m_base, C)  ->  y = H m_star + ε  ->  (m̄, Σpost)  ->  采样算子 A。
    Σpost 仍与 y 无关, 但后验均值 m̄ 依赖 y —— 不再隐含 y=H m_base 的零残差假设。
    m_star 只用于产生 y 与 m̄, 不跑 MOOSE(成本不变)。"""
    o = np.array(obs_idx)
    C = sigma**2 * R_UNIT + JITTER*np.eye(NC*NC)
    if ZERO_RESIDUAL:                                           # 消融: y:=H·m_base(显式零残差假设)
        m_star = m_base.copy()
        y = m_base[o].copy()
    else:
        Lc = np.linalg.cholesky(C)
        m_star = m_base + Lc @ rng.standard_normal(NC*NC)       # 潜在真实场
        y = m_star[o] + OBS_NOISE*rng.standard_normal(len(o))   # 虚拟观测
    Coo = C[np.ix_(o, o)] + OBS_NOISE**2*np.eye(len(o))
    Kg = C[:, o] @ np.linalg.inv(Coo)
    mbar = m_base + Kg @ (y - m_base[o])                        # 后验均值(网络的均值输入)
    Cpost = C - Kg @ C[o, :]
    s = np.sqrt(np.clip(np.diag(Cpost), 0, None))
    # 特征分解: 非零σn下 Σpost 理论正定, 用 eigh 是为了对小/数值负特征值稳健(截断)
    w, V = np.linalg.eigh(Cpost); w = np.clip(w, 0, None)
    return m_star, y, mbar, s, V * np.sqrt(w)

def conditioned_uncertainty(obs_idx, sigma=SIGMA):
    """(保留给出图/旧接口) 仅返回 (s, A) —— 与 y 无关的部分。"""
    o = np.array(obs_idx)
    C = sigma**2 * R_UNIT + JITTER*np.eye(NC*NC)
    Coo = C[np.ix_(o, o)] + OBS_NOISE**2*np.eye(len(o))
    Kg = C[:, o] @ np.linalg.inv(Coo)
    Cpost = C - Kg @ C[o, :]
    s = np.sqrt(np.clip(np.diag(Cpost), 0, None))
    w, V = np.linalg.eigh(Cpost); w = np.clip(w, 0, None)
    return s, V * np.sqrt(w)

# ================= m0 on-the-fly =================
_EK = None
def build_m0(img_idx, r_well):
    """裂隙图+r_well -> m0 (log10): 返回 (400网格, 40网格)"""
    global _EK
    import sys; sys.path.insert(0, str(BD))
    from build_prior_field import build_heatmap_logk, ellipse_kernel
    if _EK is None: _EK = ellipse_kernel()
    lg, _, _ = build_heatmap_logk(IMG_DIR/f"nf_slice_{img_idx:04d}.png", r_well, _EK)  # 自然对数,1200²
    lg10 = lg/np.log(10)
    return coarsen(lg10, PERM_N), coarsen(lg10, NC).ravel()

# ================= 模板 =================
def sub_once(t, old, new, tag):
    if t.count(old) != 1: raise RuntimeError(f"patch {tag}")
    return t.replace(old, new)

def build_case_template():
    t = TEMPLATE.read_text()
    t = sub_once(t, "    sort_by   = 'x'\n    execute_on = FINAL",
                 "    sort_by   = 'x'\n    execute_on = 'TIMESTEP_END'", "vpp")
    t = sub_once(t, "  [csv_out]\n    type = CSV\n    execute_on = FINAL",
                 "  [csv_snap]\n    type = CSV\n    execute_on = 'TIMESTEP_END'\n"
                 f"    sync_only = true\n    sync_times = '{SYNC_TIMES}'\n  []\n"
                 "  [csv_out]\n    type = CSV\n    execute_on = FINAL", "snap")
    return t

def replace_top(t, key, val):
    pat = re.compile(rf"^({re.escape(key)}\s*=\s*).*$", re.MULTILINE)
    new, n = pat.subn(lambda m: m.group(1) + str(val), t, count=1)
    if n != 1: raise RuntimeError(f"var {key}")
    return new

# ================= 运行 + 就地收场 =================
def run_case(cid, ifile, cases, logs, ranks, pin):
    env = dict(os.environ)
    env["PATH"] = "/home/yl3825/miniforge/envs/moose/bin:" + env.get("PATH", "")
    for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"): env[f"{v}_NUM_THREADS"] = "1"
    cmd = ["taskset", "-c", pin, MPIEXEC, "-n", str(ranks), BIN,
           "-i", str(ifile), f"Outputs/file_base={cases/cid}"]
    t0 = time.time()
    with (logs/f"{cid}.log").open("w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(cases), env=env)
    return rc, round(time.time()-t0, 1)

def grid_field(px, py, v):
    from scipy.interpolate import griddata
    g = griddata(np.column_stack([px, py]), v, (XG, YG), method="linear")
    hole = np.isnan(g)
    if hole.any():
        g[hole] = griddata(np.column_stack([px, py]), v, (XG[hole], YG[hole]), method="nearest")
    return g.astype(np.float32)

def collect_and_clean(cid, cases, fields_dir, z):
    """point_sample -> [S,2,H,W] npz (ΔT,Δp; 快照+终态), 存z; 删重文件。成功返回True。"""
    def read_vpp(f):
        rows = list(csv.DictReader(open(f)))
        return (np.array([float(r["x"]) for r in rows]), np.array([float(r["y"]) for r in rows]),
                np.array([float(r["T"]) for r in rows]), np.array([float(r["pp"]) for r in rows]))
    snaps = sorted(glob.glob(str(cases/f"{cid}_csv_snap_point_sample1_*.csv")))
    finals = sorted(glob.glob(str(cases/f"{cid}_point_sample1_*.csv")))
    finals = [f for f in finals if os.path.getsize(f) > 100]
    if not finals: return False
    good_snaps = [f for f in snaps if os.path.getsize(f) > 100][:len(SYNC_LIST)]
    files = good_snaps + [finals[-1]]
    times = SYNC_LIST[:len(good_snaps)] + [FINAL_TIME]
    out = []
    for f in files:
        px, py, T, p = read_vpp(f)
        out.append(np.stack([T_INIT - grid_field(px, py, T),
                             (grid_field(px, py, p) - P_INIT)/1e6], 0))
    np.savez_compressed(fields_dir/f"{cid}.npz",
                        fields=np.stack(out, 0).astype(np.float32),
                        times=np.array(times), z=z.astype(np.float32))
    for f in snaps + finals + [str(cases/f"{cid}_perm.i")]:
        try: os.remove(f)
        except OSError: pass
    return True

# ================= 主流程 =================
def scenario_params(sid, gold=False):
    """场景参数由 (sid, 用途) 的独立种子唯一决定 → 断点/分片/复现都安全"""
    rng = np.random.default_rng(900000 + sid if gold else 100000 + sid)
    img = int(rng.integers(1, N_IMAGES + 1))
    r_well = float(rng.uniform(150.0, 300.0))
    sigma = round(float(rng.uniform(*SIGMA_RANGE)), 3)
    obs_idx, n_extra = sample_obs_scenario(rng)
    well_y = float(rng.choice(WELL_Y_CHOICES))
    return dict(img=img, r_well=r_well, sigma=sigma, obs_idx=obs_idx,
                n_extra=n_extra, well_y=well_y, rng=rng)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--shard", default="0/1", help="k/K: 只跑 sid%%K==k 的场景")
    ap.add_argument("--ranks", type=int, default=RANKS)
    ap.add_argument("--pin", default=PIN_CORES)
    args = ap.parse_args()
    k, K = (int(v) for v in args.shard.split("/"))
    if args.smoke:   n_scen, n_real, tag = SMOKE_SCEN, SMOKE_REAL, "smoke"
    elif args.gold:  n_scen, n_real, tag = GOLD_SCENARIOS, GOLD_REAL, "gold"
    else:            n_scen, n_real, tag = N_SCENARIOS, N_REAL, "train"

    inputs = DATA/"inputs"; cases = DATA/"cases"; logs = DATA/"logs"; fields = DATA/"fields"
    for d in (inputs, cases, logs, fields): d.mkdir(parents=True, exist_ok=True)
    tmpl = build_case_template()
    meta_csv = DATA/f"metadata_{tag}_shard{k}.csv"
    meta_rows = []

    for sid in range(n_scen):
        if sid % K != k: continue
        sp = scenario_params(sid, gold=args.gold)
        scen_id = f"{tag}{sid:05d}"
        mesh = MESH_DIR/f"mesh_{sp['img']}.msh"
        img = IMG_DIR/f"nf_slice_{sp['img']:04d}.png"
        if not mesh.exists() or not img.exists():
            print(f"[skip] {scen_id}: 缺 mesh/img {sp['img']}"); continue
        try:
            m0_400, m0_40 = build_m0(sp["img"], sp["r_well"])   # m_base
        except Exception as e:
            print(f"[skip] {scen_id}: m0 失败 {e}"); continue
        m_star, y_obs, mbar, s, A = belief_state(sp["obs_idx"], sp["sigma"], m0_40, sp["rng"])
        h_obs = np.zeros(NC*NC, np.float32); h_obs[np.array(sp["obs_idx"])] = 1.0
        sat_rate = float((m0_400 >= np.log10(K_MAX) - 1e-9).mean())   # prior score 饱和率
        scen_npz = inputs/f"{scen_id}.npz"
        if not scen_npz.exists():
            np.savez_compressed(scen_npz,
                                mbar=mbar.reshape(NC, NC).astype(np.float32),   # 网络均值输入
                                s=s.reshape(NC, NC).astype(np.float32),
                                h_obs=h_obs.reshape(NC, NC),                    # 监测位置mask输入
                                m_base=m0_40.reshape(NC, NC).astype(np.float32),
                                m_star=m_star.reshape(NC, NC).astype(np.float32),
                                y=y_obs.astype(np.float32),
                                obs_idx=np.array(sp["obs_idx"]), sigma=sp["sigma"],
                                ell=L_CORR, sigma_n=OBS_NOISE,
                                img=sp["img"], r_well=sp["r_well"], well_y=sp["well_y"])
        for r in range(n_real):
            cid = f"{scen_id}_r{r:03d}"
            z = sp["rng"].standard_normal(NC*NC)      # 先抽(保证断点续跑时序列一致)
            if (fields/f"{cid}.npz").exists():
                continue                              # 断点续跑
            dev40 = ((mbar - m0_40) + A @ z).reshape(NC, NC)   # 围绕 m̄ 采样(偏差相对 m_base)
            m_fine = m0_400 + upsample(dev40, xc, yc, x400, y400)
            clip_rate = float(((m_fine < np.log10(CLIP_MIN)) | (m_fine > np.log10(CLIP_MAX))).mean())
            k400 = np.clip(10.0**m_fine, CLIP_MIN, CLIP_MAX)
            perm = cases/f"{cid}_perm.i"; write_pml(k400, x400, y400, perm)
            bh = cases/f"{cid}.bh"
            bh.write_text(f"0.2 0.000000 {sp['well_y']:.6f} 1\n0.2 0.000000 {sp['well_y']:.6f} 0\n")
            t = tmpl
            t = replace_top(t, "mesh_file", str(mesh))
            t = replace_top(t, "perm_block_file", f"'{perm}'")
            t = replace_top(t, "prod_bh_file", str(bh))
            t = replace_top(t, "prod_pt", f"'0 {sp['well_y']:g} 0'")
            t = replace_top(t, "inj_T", f"{INJ_T:g}")
            t = replace_top(t, "injection_mass_rate", f"{INJ_RATE:g}")
            t = replace_top(t, "Lz", f"{LZ_THICKNESS_M:g}")
            ifile = cases/f"{cid}.i"; ifile.write_text(t)
            row = dict(case_id=cid, scen=scen_id, real=r, **{kk: sp[kk] for kk in
                       ("img", "r_well", "sigma", "n_extra", "well_y")}, n_obs=len(sp["obs_idx"]),
                       clip_rate=round(clip_rate, 4), sat_rate=round(sat_rate, 4))
            if not args.generate_only:
                rc, wall = run_case(cid, ifile, cases, logs, args.ranks, args.pin)
                ok = collect_and_clean(cid, cases, fields, z)
                row.update(rc=rc, wall_s=wall, collected=ok)
                print(f"  {cid}: rc={rc} wall={wall}s npz={ok}", flush=True)
            meta_rows.append(row)
            with meta_csv.open("w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=sorted({x for rr in meta_rows for x in rr}))
                wr.writeheader(); wr.writerows(meta_rows)
    print(f"\ndone: {len(meta_rows)} cases -> {meta_csv}")

if __name__ == "__main__":
    main()
