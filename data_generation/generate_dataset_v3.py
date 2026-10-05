#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(M, C) 方案数据集生成器 v3 —— 30k 设计（2026-07-12；经验与坑见 structure_experiments_July11/FINDINGS.md）。

链条: 场景采样(img, 结构配方, C旋钮[监测预算分层], S_max, w, q, T_inj)
      → M = heatmap_struct(细1200²) → C/S 同一 cfg 在 128²(存图)+64²(扰动) 求值
      → 双级流量感知注入筛查(场景级 M+q; 实现级 m^(j)+q, 重抽 z)
      → m^(j) = M_400 + upsample(D_S L z) → MOOSE(终态, 非时变) → npz 就地收+删重文件

预算: 训练 5,600 场景 × N=5 = 28,000 + 金标准 10×200 = 2,000 → 30,000 runs
用法:
  python3 generate_dataset_v3.py --smoke                       # 4场景×2 = 8 case 端到端
  python3 generate_dataset_v3.py --shard 0/2 --ranks 12 --pin 0-11  &
  python3 generate_dataset_v3.py --shard 1/2 --ranks 12 --pin 12-23 &
  python3 generate_dataset_v3.py --gold
  python3 generate_dataset_v3.py --smoke --generate-only       # 干跑(不占核): 只出输入/场文件
"""
import argparse, csv, glob, os, re, subprocess, time
from pathlib import Path
import numpy as np

# 路径可移植(2026-07-12, 跨服务器协议): 仓库内相对 + 环境变量覆盖外部资产, 见 DISTRIBUTED.md
BD = Path(__file__).resolve().parent
DATA = BD / "dataset_v3"
TEMPLATE = BD / "THM_template_final.i"
IMG_DIR = Path(os.environ.get("GEO_IMG_DIR",
    "/home/yl3825/Pycharm_projects/Journal_Inverse_problem/Geothermal_generation/Final/natural fracture networks/images"))
MESH_DIR = Path(os.environ.get("GEO_MESH_DIR",
    "/home/yl3825/Pycharm_projects/Journal_Inverse_problem/Geothermal_generation/Final/mesh"))
N_IMAGES = 2000
MPIEXEC = os.environ.get("GEO_MPIEXEC", "/home/yl3825/miniforge/envs/moose/bin/mpiexec")
BIN = os.environ.get("GEO_BIN", "/home/yl3825/projects/cavity/cavity-opt")
RANKS, PIN_CORES = 12, "12-23"

# ---- 网格（分辨率升级 2026-07-12: 扰动 64², 状态图 128²）----
LX = 800.0
NPERT = 64                                  # 扰动/采样网格 (12.5 m)
NSTATE = 128                                # M/C/S 状态图存储分辨率
NSCREEN = 40                                # 筛查网格(=injectivity_screen 标定网格, 勿动)
PERM_N = 400                                # 渗透率写场分辨率 (2 m)
xp = np.linspace(-LX/2, LX/2, NPERT); yp = xp.copy()
xs = np.linspace(-LX/2, LX/2, NSTATE); ys = xs.copy()
x400 = np.linspace(-LX/2, LX/2, PERM_N); y400 = x400.copy()
XP, YP = np.meshgrid(xp, yp)
XS, YS = np.meshgrid(xs, ys)

# ---- 物理常数（E/D 系列落地后在此更新）----
L_CORR = 120.0
S_MAX_RANGE = (0.15, 0.5)
CLIP_MIN, CLIP_MAX = 2.7e-16, 4.3e-12       # E系列定案: K_MAX=1.35e-12, CLIP=K_MAX*10^0.5
Q_RANGE = (1.0, 3.5)                        # kg/s per stage（设计参考 2.5）
TINJ_RANGE = (293.0, 313.0)                 # K, 20-40°C（2026-07-12 用户: 注温最高40度）
WELL_Y_CHOICES = (50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0)
LZ_THICKNESS_M = 5.0
DT0, DT_GROWTH = 1000, 2.0                  # D系列定案: 场RMSE 1.03K ≪ 实现间3.56K, 省34%墙钟
CASE_TIMEOUT = 1200                         # 健康run≤400s(dt加速); 硬案例(~5%)浪费封顶20min,
                                            # 场景保留其余标签(N=5容错), 见 FINDINGS §14
PP_CEILING = 65.0                           # 筛查天花板 [MPa]
MAX_Z_RETRY = 40


def _CEIL(sp):
    """场景级筛查天花板 [MPa]。默认 = PP_CEILING(v3/v4≤r4 行为逐位不变)。
       v4-r5 起生产井 BHP 可变,由 generate_dataset_v4 覆写本钩子按 prod_p 平移天花板
       (等价于平移预测值,但保持 pp25−P_BASE = op25,q_adm 的解析反解不受影响)。"""
    return PP_CEILING

N_SCENARIOS, N_REAL = 5600, 5               # 训练 28,000 (2026-07-12: 场景降、标签升)
GOLD_SCENARIOS, GOLD_REAL = 10, 200         # 金标准 2,000
SMOKE_SCEN, SMOKE_REAL = 4, 2

# ---- 输出场网格（非时变: 只收终态）----
H, W = 160, 128
END_TIME = 3.16e8          # 与 THM_template_final.i 的 Executioner end_time 一致(审计#5 校验用)
XG, YG = np.meshgrid(np.linspace(-400, 400, W), np.linspace(-500, 500, H))
T_INIT = 354.15 - 0.023*YG
P_INIT = 30e6 - 9810.0*YG

import sys
sys.path.insert(0, str(BD))
from generate_uq_dataset import read_pml, write_pml, upsample, coarsen   # 场 IO 复用
from build_prior_field import build_heatmap_struct_logk, ellipse_kernel, fracture_mask
from confidence_model import sample_confidence_cfg, eval_confidence, cfg_desc, confidence_to_S
from injectivity_screen import predicted_pp
from scipy.ndimage import gaussian_filter

# ---- 相关矩阵 Cholesky（64², 一次缓存）----
def get_chol():
    f = DATA / "L_chol64.npy"
    if f.exists():
        return np.load(f)
    pts = np.column_stack([XP.ravel(), YP.ravel()])
    d2 = ((pts[:, None, :] - pts[None, :, :])**2).sum(-1)
    L = np.linalg.cholesky(np.exp(-0.5*d2/L_CORR**2) + 1e-8*np.eye(NPERT*NPERT))
    # ★2026-08-05:原子落盘。裸 np.save 在多 worker 同时冷启动(空数据目录)时会互相撕裂,
    #   各进程读到不同字节 → v4 的 PHYSICS_MANIFEST 自检因 cholesky_sha256 不一致而全体拒跑。
    #   os.replace 在同一文件系统上是原子的;内容本身是确定性的,谁赢都一样。
    tmp = f.with_suffix(f".npy.{os.getpid()}.tmp")
    np.save(tmp, L)
    os.replace(tmp, f)
    return L

def frac_density_full(mask):
    """1200² 掩膜 → 平滑密度(1200²)∈[0,1]; 用 upsample 插值到任意网格"""
    Em = gaussian_filter(mask.astype(float), 25)
    return (Em - Em.min()) / (Em.max() - Em.min() + 1e-12)

def to_grid_field(px, py, v):
    from scipy.interpolate import griddata
    g = griddata(np.column_stack([px, py]), v, (XG, YG), method="linear")
    hole = np.isnan(g)
    if hole.any():
        g[hole] = griddata(np.column_stack([px, py]), v, (XG[hole], YG[hole]), method="nearest")
    return g.astype(np.float32)

# ---- 模板（终态输出; 非时变）----
def sub_once(t, old, new, tag):
    if t.count(old) != 1:
        raise RuntimeError(f"patch {tag}")
    return t.replace(old, new)

def build_case_template():
    """point_sample 每步执行(终态文件才有数据, FINAL 去重坑), csv_out 保持 FINAL。
    注: 与旧生成器不同——它的 vpp 每步 + csv_out FINAL 只出一个含数据的终态文件, 最优。"""
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
    """终态 point_sample → npz{dT,dp}(float32, [2,H,W]) + z; 删重文件。

    ★审计#5:必须做**终态硬校验**再落盘,否则超时/发散的半截 run 也会被当成有效数据:
      ① rc == 0;② 标量 CSV 末行 time == END_TIME;③ 场有限且形状 == (2,H,W)。
    """
    if rc != 0:                                                  # ① 求解器必须正常退出
        return False
    sc = cases / f"{cid}.csv"                                     # ② 终态时间校验
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
    if fld.shape != (2, H, W) or not np.isfinite(fld).all():      # ③ 场有限 + 形状正确
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

# ---- 场景采样（独立种子 → 断点/分片/复现安全）----
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
    tag = globals().get("TAG_OVERRIDE") or tag   # v4 猴补钩(v3 行为不变)

    inputs = DATA/"inputs"; cases = DATA/"cases"; logs = DATA/"logs"; fields = DATA/"fields"
    for d in (inputs, cases, logs, fields):
        d.mkdir(parents=True, exist_ok=True)
    L = get_chol()
    ek = ellipse_kernel()
    tmpl = build_case_template()
    meta_csv = DATA / f"metadata_{tag}_shard{k}.csv"
    meta_fields = ["case_id", "scen", "real", "img", "s_max", "well_y", "well_x", "q", "t_inj",
                   "prod_dd", "prod_p_mpa",          # ★v4-r5 动作维度(v3 跑时恒为 None)
                   "rich", "cbar", "scen_pp_pred", "z_rejects", "clip_rate",
                   "rc", "wall_s", "collected", "cfg"]
    if not meta_csv.exists():
        with meta_csv.open("w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=meta_fields).writeheader()

    def meta_append(row):
        with meta_csv.open("a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=meta_fields).writerow(row)

    # 断点续跑优化(2026-07-12): 跳过之前已记 rc=-9(超时)/rc=1(物理失败)的 case ——
    # 它们 z 种子固定, 重跑必然同样失败, 纯白烧。每个场景由单机独占(sid%K), 跳过不影响
    # A/B 一致性; z 仍照抽(与 npz-skip 同路径), rng 序列一致。见 FINDINGS §16。
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

        # ---- M（细）+ 多分辨率视图 ----
        lg_nat, xf, yf, recipe = build_heatmap_struct_logk(img, sp["rng"], ek)
        lg10 = lg_nat / np.log(10.0)                     # 1200²
        M_400 = coarsen(lg10, PERM_N)
        M_state = upsample(M_400, x400, y400, xs, ys)    # 128² 存图
        M_pert = upsample(M_400, x400, y400, xp, yp)     # 64² 扰动基
        M_screen = coarsen(lg10, NSCREEN)                # 40² 筛查(标定网格)

        # ---- C/S（同一 cfg 两个分辨率）----
        mask = fracture_mask(img)
        Ef = frac_density_full(mask)
        E_400 = coarsen(Ef, PERM_N)
        cfg = sample_confidence_cfg(sp["rng"])
        C_state = eval_confidence(cfg, XS, YS, upsample(E_400, x400, y400, xs, ys))
        C_pert = eval_confidence(cfg, XP, YP, upsample(E_400, x400, y400, xp, yp))
        S_pert = confidence_to_S(C_pert, sp["s_max"])

        # ---- 场景级筛查 + 流量自适应（2026-07-12: 运行方知道注入能力后才定流量,
        #      q 上限 = 天花板反解 → 场景不浪费, q-注入能力相关性符合现实）----
        from injectivity_screen import P_BASE
        # ★v4s 联合筛查(2026-07-28 专家 P0-1):sweep 模式下 q 上限取**全部候选井位**的最严值,
        #   否则各位置按自己的 q_hi 裁剪 ⇒ 同地质不同位置拿到不同 q,配对失效。
        _pos = sp.get("_sweep_positions")
        _cand = _pos if _pos else [(sp.get("well_x", 0.0), sp["well_y"])]
        pp25 = max(predicted_pp(M_screen, prod_y=_wy, rate=2.5, prod_x=_wx) for _wx, _wy in _cand)
        # ★v4-r5:生产井 BHP 可变时,把修正折算到**天花板**而非预测值 —— 预测值口径不变,
        #   pp25−P_BASE 仍等于 op25,下面的 q_adm 解析反解才成立。sp 无 prod_p_mpa(v3/旧版)
        #   时 _CEIL(sp) 原样返回 PP_CEILING ⇒ 向后完全兼容。
        _ceil = _CEIL(sp)
        q_adm = 2.5*(_ceil - P_BASE)/(pp25 - P_BASE) if pp25 > P_BASE else Q_RANGE[1]
        if q_adm < Q_RANGE[0]:
            n_scen_skipped += 1
            print(f"[screen] {scen_id}: 最低流量也不可行 (pp@2.5={pp25:.1f}) → 跳过", flush=True)
            continue
        q_hi = min(Q_RANGE[1], q_adm)
        if globals().get("FIXED_Q_NO_RESAMPLE"):
            # ★v4s(专家 P0):q 必须在同地质全部井位间一致;仅裁剪到可行上限,不重抽。
            sp["q"] = round(float(min(sp["q"], q_hi)), 2)
        else:
            sp["q"] = round(float(sp["rng"].uniform(Q_RANGE[0], q_hi)), 2)   # 可行区间内重抽
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

        # ★v4 钩子(2026-07-28):NOMINAL_LAST=True 时,最后一轮 = **零扰动实现**(z≡0,m=M),
        #   即 g = MOOSE(M) 名义响应。自带快照与井曲线,与本场景 5 个实现完全同口径。
        #   v3 默认 False → 行为逐字节不变(v3 的 g 走独立的 generate_mean_response.py)。
        _nom_last = bool(globals().get("NOMINAL_LAST", False))
        for r in range(n_real):
            _is_nom = _nom_last and r == n_real - 1
            cid = f"{scen_id}_rnom" if _is_nom else f"{scen_id}_r{r:03d}"
            # ★审计#4 修复:z 改用**每个实现独立的确定性种子**(sid, r),不再共用场景级 rng 流。
            #   旧写法在实现级筛查拒绝时会多抽 z,而断点续跑遇到已存在的 npz 只抽一次就跳过,
            #   导致后续实现的随机序列整体错位(已实测产生过重复 z)。独立种子从构造上消除该问题:
            #   实现 r 的 z 只取决于 (sid, r),与之前抽了多少次、是否续跑完全无关。
            # ★v4s 配对钩子(2026-07-28 专家 P0):Z_SID_OVERRIDE 令同地质的全部井位共享
            #   同一 z 流(否则 sid 含井位编号 ⇒ 每个井位抽到不同 z,配对扫描失效)。
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
                # ★sweep(专家 P0-1 + 2026-07-28 实测):z 由 geo 播种、**禁止重抽** ——
                #   重抽会让该井位换到不同 z,配对即失效(实测联合筛查下 rej 高达 10+)。
                #   本井位若不可行,由场景级筛查跳过整个位置(缺位可接受,优化本就不会选它)。
                if globals().get("NO_Z_RESAMPLE"):
                    break
                if predicted_pp(m40, prod_y=sp["well_y"], rate=sp["q"],
                                prod_x=sp.get("well_x", 0.0)) < _CEIL(sp):
                    break
                rejects += 1
                if rejects >= MAX_Z_RETRY:
                    break
                z = zrng.standard_normal(NPERT*NPERT)   # 重抽走同一独立流
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
            # ★2026-08-05 修复:旧写法是 z=1 与 z=0 两个点(照搬 MOOSE 的 3D 例子),但本模型
            #   网格是严格 2D(全部节点 z=0)⇒ z=1 那个点找不到,被 MOOSE 默认的
            #   point_not_found_behavior=IGNORE **静默丢弃**,只剩 half_seg_len=0.5 的半段,
            #   生产井 Peaceman 井指数恰好只有设计值的一半(实测:改成 ERROR 会报
            #   "Point (x,y,z)=(...,1) not found in block(s) 1, 2, 3")。
            #   2D 正确写法 = 单点 + 模板里的 line_length(见 THM_template_final.i 的两个
            #   PorousFlowPeacemanBorehole)。两点都放 z=0 会触发 zero-segment 报错,勿试。
            #   影响实测(3 场景 A/B):ΔT 场 0.04 K(1.1%)、Δp 场形状 0.009 MPa(0.2%),
            #   但域平均压力偏高 0.96-5.83 MPa(占实现间不可约离散 85%),且随渗透率变化。
            # 护栏:本文件同时是 v3 生成器与 v4 的基座。v3 冻结模板没有 line_length,
            #   配单点会零长度报错,故按模板内容自适应——旧模板仍走旧写法,保 v3 可复现。
            # r_w:0.2 m(15.75")对地热生产段偏大,且实测 1.6% 算例因 Peaceman 的
            #   "单元有效尺寸须 > r_w" 检查崩掉(有效尺寸 0.158-0.176 m),被剔除的样本
            #   系统性偏向"生产井位于裂隙密集区" ⇒ 选择偏倚。r4 起降到 0.1 m(≈8.5" 裸眼)。
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
            # ★v4-r5:生产井 BHP 逐 case 代入(prod_p = p0(y_prod) − Δ)。sp 无 prod_p_mpa 时
            #   完全跳过 ⇒ v3/v4≤r4 的 .i 逐字节不变。注意 replace_top 的正则要求 key 后紧跟
            #   \s*=,所以匹配的是 "prod_p = 20e6" 而不会误命中 "prod_pt = ..."。
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
                ok = collect_and_clean(cid, cases, fields, z, rc)   # ★rc 传入做硬校验
                row.update(rc=rc, wall_s=wall, collected=ok)
                print(f"  {cid}: rc={rc} wall={wall}s npz={ok} rej={rejects}", flush=True)
            meta_append(row)
    print(f"\ndone (shard {k}/{K}); 场景级筛掉 {n_scen_skipped}", flush=True)

if __name__ == "__main__":
    main()
