#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据集 v4 生成器(2026-07-27,规格见 DATASET_V4_SPEC.md)——服务两个下游任务:
① 地热 history matching(需要井曲线时间序列;实现级前向代理 m→d(t))
② 逐 stage 井位优化 + lateral 斜线(需要连续 2D 井位覆盖 + 同地质 2D 扫掠验证集)

设计原则:**猴补复用 v3 全部管线**(场景播种/M/C 生成/双级筛查/写 .i/断点续跑/分片协议),
v4 在 v3 管线上增加:
  a) 生产井连续 2D 采样:well_x∈[-180,180] m、well_y∈[0,360] m，并施加井距约束;
     种子基 410000+sid(与 v3 的 110000/910000 不冲突);
  b) 模板:Coussy 拉格朗日孔隙率换算为当前体积孔隙率,使用平衡的初始
     p/T 参考场和滚动底边界,并仅保留等效基质渗透率中的裂隙表征;
  c) 标量 postprocessor 每步写 CSV(井曲线);加 pp_prod;全部模式保存稀疏时序场;
  d) 收集:npz 增存 curves(time × 列)——终态场保持 v3 同格式,向后兼容;
  e) 三种模式:--mode main | sweep | truth(见下)。

模式:
  main : 连续 2D 井位新场景。--n 控制目标场景数(默认 20000),每场景 5 个随机实现+1 个名义实现。
  sweep: 同地质 2D 井位扫掠(井位优化验证金标准)。sid 编码 (geo, pos_idx):
         geo ∈ [0,N_GEO),pos_idx ∈ [0,N_Y)。同 geo 的全部井位共享同一 rng 流
         (地质/C/S_max/q/T_inj/z 全同,仅 well_x/well_y 不同 → 严格配对)。
  truth: 与 main/sweep 独立播种的合成真值历史(HM 反演靶)。

分片协议(多机多批,DISTRIBUTED.md 铁律不变):
  - 同一 commit;sid % K == k 路由;K 开跑后不可改(建议 K=24);
  - 断点/跳失败/append 元数据全继承 v3;两机永不同 shard。
用法:
  python generate_dataset_v4.py --mode main  --shard 0/24 --ranks 6 --pin 0-5
  python generate_dataset_v4.py --mode sweep --shard 0/24 --ranks 6 --pin 0-5
  python generate_dataset_v4.py --mode truth --shard 0/1  --ranks 6 --pin 0-5
  python generate_dataset_v4.py --mode main --smoke        # 烟测 2 场景×2 实现
"""
import argparse, csv, glob, hashlib, json, os, re, sys
from pathlib import Path

import numpy as np
# ★硬依赖前置(2026-07-28 WSB 指出):netCDF4 用于读 Exodus 快照。此前它在 collect 的 try 内,
#   缺失时每个 case 照烧 400s MOOSE 再把 npz 删掉、只多打一行日志 —— 最贵的失败模式。
#   放顶层 ⇒ 环境不全时启动瞬间崩。安装:conda install -n moose -c conda-forge netcdf4 "hdf5=*=mpi_mpich*"
import netCDF4  # noqa: F401

BD = Path(__file__).resolve().parent
sys.path.insert(0, str(BD))
import generate_dataset_v3 as G

# ---------------- v4 常数 ----------------
# 每次改变前向物理都使用独立目录。dataset_v4_coussy 使用了非平衡的常数
# p/T 参考态和显式低维裂隙块，不能与本版本混合或断点续跑。
DATA4 = BD / "dataset_v4_coussy_r5"
PREVIOUS_DATA4 = BD / "dataset_v4_coussy_r4"
LEGACY_DATA4 = BD / "dataset_v4_coussy"
# ⚠️ r4→r5 的差别是**逐 case 替换** prod_p,而 template_sha256 哈希的是替换**之前**的基模板
#    ⇒ 哈希不变、manifest 门禁抓不到这次改动。所以本行必须手动 bump,别指望自检。
PHYSICS_VERSION = "v4-prodbhp-variable-r5"

# ---- r3 heatmap 再平衡(2026-07-30,用户决策:不引导管补偿叙事,只动 heatmap)----
# r2 实测(标定批 14 场景 + 生产 220 case):无导管物理下背景 k 成为路径瓶颈,q 中位被压到
# 1.11、Δp 标签尺度 ~7-10 MPa;通道族陡 sigmoid 造成"井在高渗墙外注定憋死"(拒绝率 ~25%)。
# 离线扫描(仿射重映射 + logit 软化,14 场景代理电阻定量):选定
#   K_MIN 2.7e-15 → 4e-15(×1.5,仍居损伤带文献带中段;K_MAX/CLIP_MAX/开度/Soultz 锚不动)
#   CHANNEL_KAP2 (0.04,0.10) → (0.08,0.20)(过渡带宽 ×2,通道族保持双峰)
# ⇒ q_adm 中位 1.85→2.42、拒绝→~0%、Δp 中位 ~21 MPa、可采性约束保留(p10=1.6)。
R3_K_MIN = 4e-15
R3_CLIP_MIN = 4e-16                     # 保持 CLIP = K_MIN/10^0.5 的既有关系
R3_CHANNEL_KAP2 = (0.08, 0.20)

# ★import 时即生效(calib/离线脚本与生产 worker 共用同一映射,防"分析用旧场、生产用新场")
import build_prior_field as _bpf
_bpf.K_MIN = R3_K_MIN
_bpf.LOGK_MIN = np.log(R3_K_MIN)
_bpf.CHANNEL_KAP2_RANGE = R3_CHANNEL_KAP2
G.CLIP_MIN = R3_CLIP_MIN

# Coussy 小应变热孔弹性关系先写成单位参考体积的拉格朗日孔隙率，再线性化为
# MOOSE PorousFlowPorosityLinear 所需的当前体积孔隙率。Ks 由
# alpha = 1 - Kd/Ks 与模板中的 E、nu、alpha 保持一致。
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

WELL_Y_RANGE = (0.0, 360.0)                  # 保留纵向覆盖
WELL_X_RANGE = (-180.0, 180.0)               # 收紧横向覆盖，避免横向偏移主导井位差异
INJ_XY = (0.0, -250.0)
DIST_RANGE = (250.0, 620.0)                  # 井间距约束(v3 轴上范围 300-600 的闭包外扩)
SEED_MAIN = 410000
SEED_TRUTH = 730000   # ★独立种子(2026-07-28 专家 P0):v4t 曾 = v4m 前 20 场景 ⇒ HM 数据泄漏
N_MAIN_DEFAULT = 20000    # ★上限不是承诺(2026-07-28 用户"越多越好"):sid 确定性播种 ⇒
                          #   任意进度停下都是完整数据集(前 N 个场景);机器一直跑到人工喊停。
                          #   参考:v3 是 9063 场景。
N_GEO = 400                                   # sweep 地质数。2026-08-09 150→400(用户指令扩容):
                                              # 训练侧确诊 sweep 臂是压力 DC 的覆盖洞(Δ 独立样本
                                              # =地质数)。加性:geo 0-149 的 sid/种子/Δ 逐位不变,
                                              # 新 geo 150-399 追加 sid 6750-17999;地质/Δ 与
                                              # v4m sid 150-399 共享(与 0-149 的既有约定一致,
                                              # 打包 split 按 v4g{geo} 已绑定两臂)。
_gx, _gy = np.meshgrid(np.linspace(-180.0, 180.0, 7), np.linspace(0.0, 360.0, 7))
_d = np.hypot(_gx - INJ_XY[0], _gy - INJ_XY[1])
XY_GRID = [(round(float(x), 1), round(float(y), 1))
           for x, y, d in zip(_gx.ravel(), _gy.ravel(), _d.ravel())
           if DIST_RANGE[0] <= d <= DIST_RANGE[1]]
N_Y = len(XY_GRID)                            # 网格点数(名称沿用,分片/编码不变)
N_TRUTH = 20
SNAP_MIN = 6           # 快照最少帧数(低于此判为不合格样本)
_FAIL_MAX = 5          # 连续收集失败熔断阈值(环境级故障快速失败,不烧算力)
_FAIL_STREAK = 0

# ★THM 运行时压力守卫(2026-07-30 两阶段演化):
#   初版 97 MPa = IF97 崩溃保护(越过 100 → DIVERGED_FUNCTION_DOMAIN → dt 死循环烧满超时)。
#   r3 定版收紧到 72 MPa ≈ S_hmin(70.7)+余量 = **可采性硬执法**:代理筛查散差大(σ≈27),
#   个别病例(标定批 sid9)代理不可见、会被放行到 pp>Shmin;守卫按物理线截停后,
#   数据集内每个保留 case 都是 THM 直接验证过全程 pp_inj < Shmin 的——筛查降级为 q 分配器,
#   安全性由守卫全权负责。截停写 {cid}.ppkill → 收集侧按物理拒绝记账(rc=-15),
#   不计入 _FAIL_STREAK(那是环境熔断,不是物理不可采)。
PP_KILL_PA = 7.2e7
PP_GUARD_POLL_S = 15

# ★r2 注入筛查重标定(2026-07-30 标定批:14 场景 × q=1.0 × 10 年 THM 真值,dataset_v4_calib/)。
#   旧 FIT(19.3,−203.4) 是 v3 轴上井+lower-D 导管标定的;r2 移除导管后系统性低估
#   (571 case r1 审计:σ=14.8、18% 超 Shmin;r2 烟测 sid0 撞 IF97 死循环)。
#   新口径:120² 细网格两点电阻(NC=40 分辨不出无导管物理的低渗瓶颈,corr 0.72→0.84),
#   超压随流量超线性 op∝q^γ(γ=1.3,冷水黏度效应,sid0 双流量点标定),
#   pp25 = A·log10(proxy120) + B 拟合于 γ 外推的 q=2.5 等效真值。
#   ceiling=60:13 实测场景仿真显示放行场景在最大放行流量下真实 pp ≤ 58.2 MPa < Shmin,
#   深度病例(q_adm<1)场景级拒绝;残余尾部由 PP_KILL 运行时守卫兜底。σ=44 ⇒ 本筛查
#   只是粗闸,精确执法在守卫;勿把 pp_pred 当真值用。
SCREEN_NC = 120
# r3 重标定(dataset_v4_calib_r3,14 场景 q=1.0 十年真值,γ 口径 120² 代理):
#   pp25 = 54.6·log10(proxy) − 684.1,resid σ=27.5 —— 只当 q 分配器用,安全由 72 MPa 守卫执法。
#   C=60 ⇒ q_adm 中位 1.92、p10 1.23、场景拒绝 0%(r2 值 76.2/−974.4 已随场统计失效)。
SCREEN_FIT_A, SCREEN_FIT_B = 54.6, -684.1
SCREEN_GAMMA = 1.3
PP_CEILING_R2 = 60.0

# ★v4-r5(2026-08-05):生产井井底压力从模板常数变成可采样的**动作维度**。
#   采的是"降压幅度" Δ = p0(y_prod) − prod_p 而**不是**绝对 BHP —— p0 随井深变,
#   直接采绝对值会让新维度继续与井位共线(r4 的毛病:Δ∈[6.47,10.00] 完全由 y_prod 决定)。
#   范围依据见 V4_R5_PRODBHP_SPEC.md:下界由 rc=−15 随 Δ 的单调趋势定(小 Δ 才危险,
#   储层被迫整体增压撞守卫),上界由最浅井位的闪蒸余量定(prod_p=12.47 MPa vs 0.05 MPa)。
PROD_DD_RANGE = (4.0, 14.0)        # 降压幅度 Δ [MPa]
SEED_PRODDD = 990000               # ★独立种子:绝不从 sp["rng"]/SEED_MAIN 流里取。
                                   #   任何改变随机数**消耗次数**的改动都会改写 (sid→场景)
                                   #   映射(见 _draw_wellpos 上方的回退记录),与已产数据冲突。
                                   #   独立播种还保证 Δ 的边缘分布与分片/机器无关 ⇒ 跨机自动均衡。
PROD_P_REF_MPA = 20.0              # r4 及以前的固定 BHP;筛查拟合就是在这个值上标的
PP_IC_PA, DPDY_PA_M, Y_REF_M = 30e6, -9810.0, 0.0    # 必须与模板 pp_ic/dpdy/y_ref 一致
# 筛查的 BHP 一阶修正:生产端压力水平变化整体平移储层压力 ⇒ 注入压等量平移。
# 实测斜率 −1.17 但混入了井距共线效应,故先用物理的 1.0,由标定批复核。
SCREEN_BHP_SLOPE = 1.0


def draw_prod_dd(sid, truth=False):
    """Δ [MPa],逐**场景**抽一次(与 q、t_inj 同级的运行条件,不是逐实现)。"""
    base = SEED_PRODDD + (10**6 if truth else 0)
    return round(float(np.random.default_rng(base + int(sid)).uniform(*PROD_DD_RANGE)), 3)


def prod_p_mpa_of(well_y, dd_mpa):
    """由井位深度与降压幅度导出生产井 BHP [MPa]。"""
    return (PP_IC_PA + DPDY_PA_M * (float(well_y) - Y_REF_M)) / 1e6 - float(dd_mpa)


def pp_shift_mpa(prod_p_mpa):
    """筛查用:相对 20 MPa 基准的注入压平移量 [MPa](prod_p 更低 ⇒ 平移为负 ⇒ 余量更大)。"""
    return (float(prod_p_mpa) - PROD_P_REF_MPA) * SCREEN_BHP_SLOPE


def eff_ceiling(sp_or_prod_p, default=PP_CEILING_R2):
    """把 BHP 修正折算到**天花板**而不是折算到预测值 —— 这样 predicted_pp_r2 的
       签名和 q_adm 的解析反解公式(pp25−P_BASE = op25)都不用动,改动面最小。"""
    pp = sp_or_prod_p.get("prod_p_mpa") if isinstance(sp_or_prod_p, dict) else sp_or_prod_p
    if pp is None:
        return default
    return default - pp_shift_mpa(pp)


def _inj_proxy_fast(lg10_coarse, nc, inj=(0.0, -250.0), prod=(0.0, 150.0)):
    """向量化装配的两点电阻代理(sweep 模式 45 井位联合筛查需要 <1s/解)。"""
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
    """r2 重标定筛查:120² 代理 + γ 超线性流量律。lg10_coarse 允许任意方阵,内部重采到 SCREEN_NC。"""
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


# ---------------- 模板 v4 ----------------
_tmpl_v3 = G.build_case_template            # 原函数引用(防覆盖后递归)
_get_chol_v3 = G.get_chol
_chol_mem = None


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chol_cache_path():
    """优先用 r2 缓存；否则复用物理无关的旧版 Cholesky 因子。"""
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
    """所有机器加载同一缓存；无缓存时才按 v3 算法生成。"""
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
    """v3 基础模板 + v4 孔隙率映射、井曲线和时序场快照。"""
    t = _tmpl_v3()                            # 已含 point_sample→TIMESTEP_END 的 v3 修复
    # 先把 Coussy 拉格朗日孔隙率换算为 MOOSE 当前体积孔隙率系数。
    # PorousFlowFullySaturated 已另外加入 M div(v_s)，因此 epv_coeff 必须是
    # alpha-phi_ref；若仍填 alpha，会把 phi_ref div(v_s) 重复计算。
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
    # ① 生产井井底压力(pp_inj 已有,补对称的 pp_prod)
    assert "[pp_prod]" not in t
    t = t.replace("  [pp_inj]\n", "  [pp_prod]\n    type = PointValue\n    variable = pp\n"
                                  "    point = ${prod_pt}\n  []\n  [pp_inj]\n", 1)
    # ② 井曲线:标量 postprocessor 每步输出;场向量默认仍只在 FINAL(主存储不变)
    # dtmax:原模板自适应步长无上限 → 10 年仅 ~33 步、3-10 年只有 2-3 个点,曲线读不出突破时间。
    assert "dtmax" not in t
    t = t.replace("  end_time = ${end_time}", "  end_time = ${end_time}\n  dtmax = 1.58e7  # 半年(v4 井曲线分辨率)", 1)
    old = "  [csv_out]\n    type = CSV\n    execute_on = FINAL\n  []"
    assert old in t, "csv_out 块与预期不符"
    new = ("  [csv_out]\n    type = CSV\n    execute_on = 'INITIAL TIMESTEP_END FINAL'\n"
           "    execute_vector_postprocessors_on = FINAL\n  []")
    if snap:                                  # ③ 快照:Exodus 二进制,每 3 个时间步一帧
        # ★两版教训(2026-07-28 烟测):sync_times 反复切断自适应步长(×3.6);
        #   逐步 CSV VPP 写 138k 行 ×10MB 文本(+57s/步)。Exodus 二进制两坑全避。
        # ★show 限定:默认写全部变量(disp/stress/aux…)= 19MB/步;只留 T/pp → ~2MB/步。
        # ★三版教训(2026-07-28 烟测):①sync_times 切断自适应步长(×3.6)
        #   ②逐步 CSV VPP 写 138k 行文本(+57s/步)③逐步 Exodus 仍 +78s/步
        #   (开销在每步的变量投影/MPI 收集,与文件大小无关)。
        #   解:time_step_interval —— 每 N 步写一帧,**不切步长**;46 步取 5 → ~9 帧,
        #   收集期就近抽 8 个设计时刻(t 是模型输入,逐 case 时刻不同无妨)。
        new += ("\n  [snap_out]\n    type = Exodus\n"
                "    show = 'T pp'\n"
                "    time_step_interval = 3\n"
                "    execute_on = 'TIMESTEP_END FINAL'\n  []")
    t = t.replace(old, new, 1)
    return t


def _physics_manifest():
    """返回可机读的物理版本；模板哈希防止跨机器悄悄混入不同方程。"""
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
        # ★r5 动作维度。template_sha256 抓不到 prod_p 的逐 case 替换,靠这几项 + physics_version
        #   把差别写进 manifest,跨机比对才有依据。
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
    """拒绝在无版本或不同模板的数据目录上续跑。"""
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
    # ★2026-08-05:原子落盘 + 竞态复核。多 worker 同时冷启动时,先前的裸 write_text 会让
    #   后到者读到半截 JSON 或与自己算的不一致(实测:6 进程并发导致 cholesky_sha256 分叉,
    #   全体 RuntimeError 拒跑)。写 tmp + os.replace 保证任一时刻读到的都是完整文件;
    #   写完再读回比对,输给别人也无妨——内容是确定性的,不一致才是真问题。
    tmp = manifest.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, manifest)
    if json.loads(manifest.read_text()) != expected:
        raise RuntimeError(f"{manifest} 写入后与本进程期望不一致（并发冷启动？）")


# ---------------- 场景参数 ----------------
# ⚠️ Peaceman 网格尺寸问题(2026-07-30,v4s00049 教训 + 回退记录):
#   井位偶落进裂隙迹线附近的加密单元(h_eff < r_w=0.2)→ MOOSE 启动即 abort(rc=1, ~7s)。
#   曾尝试在 _draw_wellpos 内做 mesh 预筛 —— **已回退,勿复活**:
#   ①任何影响 rng 消耗的过滤都会改写 (sid→场景) 映射,与已产出数据冲突(实测 sid1 井位漂移);
#   ②离线近似判据会误杀实际可跑的位置(sid1 老井位被误判,而它 rc=0 跑通过)。
#   现行策略 = 让 MOOSE 自己校验:坏位置 6 实现 × ~7s 秒失败,rc=1 记入 metadata,代价可忽略。

def _draw_wellpos(rng):
    """2D 井位:矩形内拒绝采样至满足井间距约束(拒绝概率有限,循环必终止)。"""
    for _ in range(256):
        wx = float(rng.uniform(*WELL_X_RANGE)); wy = float(rng.uniform(*WELL_Y_RANGE))
        d = float(np.hypot(wx - INJ_XY[0], wy - INJ_XY[1]))
        if DIST_RANGE[0] <= d <= DIST_RANGE[1]:
            return round(wx, 1), round(wy, 1)
    return 0.0, 150.0                                     # 理论不可达的兜底


def scenario_params_v4(sid, gold=False):
    """main:连续 2D 井位 (well_x, well_y)。抽取顺序固定,保 rng 可比。"""
    rng = np.random.default_rng(SEED_MAIN + sid)
    img = int(rng.integers(1, G.N_IMAGES + 1))
    s_max = round(float(rng.uniform(*G.S_MAX_RANGE)), 3)
    wx, wy = _draw_wellpos(rng)
    dd = draw_prod_dd(sid)                   # ★r5:独立流,不消耗 rng ⇒ 上面几行逐位同 r4
    return dict(
        img=img, s_max=s_max, well_y=wy, well_x=wx,
        q=round(float(rng.uniform(*G.Q_RANGE)), 2),
        t_inj=round(float(rng.uniform(*G.TINJ_RANGE)), 1),
        prod_dd=dd, prod_p_mpa=round(prod_p_mpa_of(wy, dd), 4),
        rng=rng)


def scenario_params_sweep(sid, gold=False):
    """sweep:sid = geo*N_Y + 格点号。**严格配对**(2026-07-28 专家 P0 修复):
       同 geo 的全部 45 个井位共享 (M, C, S_max, q, T_inj, z),仅 (well_x, well_y) 变化。
       - z:由 Z_SID_OVERRIDE=geo 播种(而非 sid —— sid 含井位编号会让每个位置抽到不同 z);
       - q:FIXED_Q_NO_RESAMPLE 关闭井位相关的重抽,并预先取 45 位置的**联合可行上限**,
            使同一 q 在所有位置都通过场景级筛查(否则不同位置各自裁剪 → q 不一致)。
    """
    geo, yj = divmod(int(sid), N_Y)
    sp = scenario_params_v4(geo)             # 地质/其余参数 = geo 的 main 场景定义
    sp["well_x"], sp["well_y"] = XY_GRID[yj]
    # ★r5:Δ 由 geo 播种(随 sp 带过来)、45 个井位共享同一个 Δ ⇒ Δ ⊥ 井位 = 正交设计。
    #   prod_p 仍随各位置的 p0(y) 变化,所以要在井位落定之后重算。
    sp["prod_p_mpa"] = round(prod_p_mpa_of(sp["well_y"], sp["prod_dd"]), 4)
    G.Z_SID_OVERRIDE = geo                   # ★同地质全部井位共享 z 流
    G.FIXED_Q_NO_RESAMPLE = True             # ★q 不随井位重抽
    G.NO_Z_RESAMPLE = True                   # ★z 不重抽(重抽即破坏配对)
    sp["_geo"] = geo
    # ★可行子集过滤(2026-07-30,r2 重标定后实测):45 位置联合取最坏值会让 ~85% 地质
    #   被一个远角位置整体枪毙。按原设计意图「缺位可接受」改为:
    #   位置级预筛 → q 由**可行子集**联合决定(同 geo 全部幸存位置仍共享同一 q,配对不破);
    #   自身位置不可行的 sid 把 _sweep_positions 设为仅自身 → 走正常场景级拒绝。
    #   对全 45 可行的地质,新旧规则 q 完全一致 ⇒ 已产出数据向后兼容。
    feas = _sweep_feasible_positions(geo, sp["img"])
    if (sp["well_x"], sp["well_y"]) in feas:
        sp["_sweep_positions"] = feas        # q 上限与 z 接受按可行子集
    else:
        sp["_sweep_positions"] = [(sp["well_x"], sp["well_y"])]   # → 场景级拒绝本位置
    return sp


_SWEEP_FEAS_CACHE = {}

def _sweep_feasible_positions(geo, img_id):
    """按 geo 缓存:重建该地质的 M_screen(与主循环同一确定性播种/同一 coarsen 路径),
    对 45 个网格位置逐个算 q_adm,返回 q_adm ≥ Q_RANGE[0] 的位置列表。"""
    if geo in _SWEEP_FEAS_CACHE:
        return _SWEEP_FEAS_CACHE[geo]
    import injectivity_screen as scr
    sp2 = scenario_params_v4(geo)            # 独立 rng,同种子 ⇒ 与主循环逐字节同 M
    lg_nat, _, _, _ = G.build_heatmap_struct_logk(
        G.IMG_DIR / f"nf_slice_{sp2['img']:04d}.png", sp2["rng"], G.ellipse_kernel())
    m_scr = G.coarsen(lg_nat / np.log(10.0), SCREEN_NC)
    feas = []
    for wx, wy in XY_GRID:
        pp25 = predicted_pp_r2(m_scr, prod_y=wy, rate=2.5, prod_x=wx)
        # ★r5:天花板逐位置随该位置的 prod_p 平移(Δ 同 geo 共享,但 p0(y) 随位置变)。
        ceil_wy = eff_ceiling(prod_p_mpa_of(wy, sp2["prod_dd"]))
        q_adm = 2.5 * (ceil_wy - scr.P_BASE) / max(pp25 - scr.P_BASE, 1e-9)
        if q_adm >= G.Q_RANGE[0]:
            feas.append((wx, wy))
    _SWEEP_FEAS_CACHE[geo] = feas
    print(f"[sweep-feas] geo={geo}: {len(feas)}/{len(XY_GRID)} 位置可行", flush=True)
    return feas


def scenario_params_truth(sid, gold=False):
    """v4t:**独立种子**(SEED_TRUTH),地质图/(M,C)/z/井位全部与 v4m/v4s 不相交 ⇒ HM 无泄漏。
       (此前 = scenario_params_v4(sid),即 v4m 前 20 场景,模型训练时见过 → 反演结果不可信。)"""
    rng = np.random.default_rng(SEED_TRUTH + int(sid))
    img = int(rng.integers(1, G.N_IMAGES + 1))
    s_max = round(float(rng.uniform(*G.S_MAX_RANGE)), 3)
    wx, wy = _draw_wellpos(rng)
    dd = draw_prod_dd(sid, truth=True)       # ★r5:truth 支再偏移 1e6,与 v4m/v4s 的 Δ 不相交
    return dict(img=img, s_max=s_max, well_y=wy, well_x=wx,
                q=round(float(rng.uniform(*G.Q_RANGE)), 2),
                t_inj=round(float(rng.uniform(*G.TINJ_RANGE)), 1),
                prod_dd=dd, prod_p_mpa=round(prod_p_mpa_of(wy, dd), 4), rng=rng)


# ---------------- run_case v4:snap 输出对象独立 file_base ----------------
_run_v3 = G.run_case

def run_case_v4(cid, ifile, cases, logs, ranks, pin):
    """同 v3,仅当模板含 snap_out 时追加 CLI 覆盖其 file_base(否则两个 CSV 对象抢同名文件)。"""
    import subprocess, time, os
    env = dict(os.environ)
    env["PATH"] = str(Path(G.MPIEXEC).parent) + ":" + env.get("PATH", "")
    for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
        env[f"{v}_NUM_THREADS"] = "1"
    cmd = ["taskset", "-c", pin, G.MPIEXEC, "-n", str(ranks), G.BIN,
           "-i", str(ifile), f"Outputs/file_base={cases/cid}",
           f"Outputs/snap_out/file_base={cases/cid}_snap"]
    # ★孤儿 MPI 代码级修复(2026-07-28 专家 P0-4):独立进程组 + 超时/异常时杀整组。
    #   subprocess.call 的 timeout 只杀 mpiexec,MPI 子进程会继续空转(实测留下 35 个孤儿)。
    import signal, threading
    t0 = time.time(); rc = -9
    ppkill = []                                     # 守卫命中时装入越限值 [Pa]

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
        except BaseException:                       # Ctrl-C / 父进程被杀:同样清整组
            try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception: pass
            raise
    if ppkill:
        (cases / f"{cid}.ppkill").write_text(f"{ppkill[0]:.6g}\n")
        print(f"  [ppguard] {cid}: pp_inj={ppkill[0]/1e6:.1f} MPa > {PP_KILL_PA/1e6:.0f} → 截停(物理拒绝)",
              flush=True)
        rc = -15
    return rc, round(time.time() - t0, 1)


# ---------------- 收集 v4:终态场 + 井曲线(+快照) ----------------
_collect_v3 = G.collect_and_clean

def collect_and_clean_v4(cid, cases, fields_dir, z, rc=0):
    """先按 v3 收终态场与硬校验;成功后把井曲线(与可选快照)并入同一 npz。"""
    mk = cases / f"{cid}.ppkill"
    if mk.exists():
        # ★压力守卫截停 = 物理拒绝(该实现/流量不可采),不是环境故障:
        #   清中间产物、不动 _FAIL_STREAK,rc=-15 已由 run_case_v4 记入 metadata。
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
        keep = np.concatenate([[True], np.diff(tvec) > 0])       # INITIAL/FINAL 与步末重复行去重
        # ★去掉 t=0 伪行(2026-07-28 专家):首行是 postprocessor 初始化值(T_prod=pp=ṁ=0),
        #   不是物理初态;留着会让曲线头学到一个假的零点。
        keep &= tvec > 0
        tvec = tvec[keep]
        missing = [c for c in CURVE_COLS if c not in rows[0]]
        if missing:                                              # ★7 列必须齐(专家 P0-3)
            raise RuntimeError(f"曲线缺列 {missing}")
        curves = np.stack([np.array([float(r[c]) for r in rows])[keep] for c in CURVE_COLS],
                          1).astype(np.float32)                   # [nt, 7]
        names = list(CURVE_COLS)
        if not (np.isfinite(curves).all() and len(tvec) >= 10
                and abs(tvec[-1] - G.END_TIME) < 1e-3*G.END_TIME):
            raise RuntimeError(f"曲线不合格(nt={len(tvec)}, t_end={tvec[-1]:.3g})")
        payload = dict(np.load(f_npz))
        payload.update(curve_t=tvec.astype(np.float64), curves=curves,
                       curve_cols=np.array(names),
                       physics_version=np.array(PHYSICS_VERSION))
        # 全模式时序场快照(Exodus,每 3 个自适应时间步一帧)
        efile = cases / f"{cid}_snap.e"
        if efile.exists():
            ds = netCDF4.Dataset(str(efile))
            et = np.array(ds.variables["time_whole"][:])
            names = ["".join(c.decode() for c in row if c != b"") for row in
                     ds.variables["name_nod_var"][:].data]
            iT = names.index("T"); ip = names.index("pp")
            px = np.array(ds.variables["coordx"][:]); py = np.array(ds.variables["coordy"][:])
            # ★保存**全部非终态帧**(2026-07-28 专家):nearest-select+去重只留 7 帧,
            #   而 Exodus 本就写了 ~10 帧;t 是模型输入,无需强行对齐固定设计时刻。
            #   末帧(≈END_TIME)排除:终态由 fields(element 采样)权威提供,避免双标签。
            sel = [j for j, tj in enumerate(et) if tj > 0 and tj < 0.995*G.END_TIME]
            fs, ts = [], []
            for j in sel:
                T = np.array(ds.variables[f"vals_nod_var{iT+1}"][j])
                pp_ = np.array(ds.variables[f"vals_nod_var{ip+1}"][j])
                fs.append(np.stack([G.T_INIT - G.to_grid_field(px, py, T),
                                    (G.to_grid_field(px, py, pp_) - G.P_INIT)/1e6], 0))
                ts.append(float(et[j]))
            ds.close()
            if len(fs) < SNAP_MIN:                                # ★快照数量门槛(专家 P0-3)
                raise RuntimeError(f"快照仅 {len(fs)} 帧 < {SNAP_MIN}")
            payload.update(snap_fields=np.stack(fs).astype(np.float32),
                           snap_times=np.array(ts))
            os.remove(efile)
        else:
            raise RuntimeError("缺 Exodus 快照文件")                # ★不再静默接受无快照样本
        # ★原子写入(专家 P0-3):先写 .tmp 再 os.replace,中途被杀不会留半截 npz
        #   (resume 只查 npz 是否存在 ⇒ 半成品会被永久跳过)。
        tmp = f_npz.with_suffix(".tmp.npz")   # ★必须以 .npz 结尾:savez 会给非 .npz 路径自动追加后缀
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, f_npz)
        os.remove(sc)                                             # 曲线已入 npz,清理原始 csv
        globals()["_FAIL_STREAK"] = 0
        return True
    except Exception as e:
        # ★2026-07-28(WSB 指出):收集期异常此前一律 unlink npz + return False =
        #   「烧算力 + 删产出 + 继续跑」。改为:终态场与曲线只要已写成功就保留(快照降级缺失),
        #   同时累计连续失败;连续 _FAIL_MAX 次 → 抛出终止 worker(环境级故障应当快速失败)。
        globals()["_FAIL_STREAK"] = globals().get("_FAIL_STREAK", 0) + 1
        print(f"  [collect_v4] {cid}: {e} (连续失败 {_FAIL_STREAK}/{_FAIL_MAX};删除不完整产物)", flush=True)
        # ★不完整样本一律删除(专家 P0-3):resume 只查存在性,留下半成品会被永久跳过。
        for f in (f_npz, f_npz.with_suffix(".tmp.npz")):
            try: f.unlink()
            except OSError: pass
        if globals()["_FAIL_STREAK"] >= _FAIL_MAX:
            raise RuntimeError(f"收集连续失败 {_FAIL_MAX} 次 —— 疑似环境/依赖故障,终止 worker") from e
        return False


# ---------------- 入口:猴补后委托 v3.main ----------------
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
    # v4 超时重标定(2026-07-28):dtmax 使健康 run 350→~700s,1200s 上限只剩 1.7× 余量
    # → 首日实测 27% 误杀。按 v3 同余量比(3.4×)重标:2400s。
    G.CASE_TIMEOUT = 2400
    # ★启动期完整性扫描(2026-07-29):v3 collector 先写 npz(fields+z),v4 collector 再追加
    #   curves/快照并原子替换。两步之间被杀 ⇒ 留下只有 fields/z 的半成品,而 resume 只查
    #   文件存在性 ⇒ 该 case 被永久跳过。启动时把不完整的清掉,让它重跑。
    _fd = DATA4 / "fields"
    if _fd.exists():
        _bad = 0
        for _f in sorted(_fd.glob("*.npz")):
            try:
                with np.load(_f) as _d:
                    _ok = {"curves", "curve_t", "snap_fields", "snap_times"} <= set(_d.files)
            except Exception:
                _ok = False
            if not _ok:
                _f.unlink(missing_ok=True); _bad += 1
        for _t in _fd.glob("*.tmp.npz"):
            _t.unlink(missing_ok=True); _bad += 1
        if _bad:
            print(f"[resume] 清理不完整产物 {_bad} 个(将自动重跑)", flush=True)
    G.collect_and_clean = collect_and_clean_v4
    G.run_case = run_case_v4
    # ★r2 筛查重标定接线(2026-07-30):细网格代理 + γ 流量律 + 收紧天花板。
    #   v3 的 q_adm 线性反解在 q<2.5 相对 γ=1.3 是保守方向,保留不动。
    G.predicted_pp = predicted_pp_r2
    G.PP_CEILING = PP_CEILING_R2
    # ★r5:天花板随生产井 BHP 平移。sp 没有 prod_p_mpa 时退回 PP_CEILING_R2 ⇒ 与 r4 逐位一致。
    G._CEIL = lambda sp: eff_ceiling(sp, PP_CEILING_R2)
    G.NSCREEN = SCREEN_NC
    # ★每场景多跑一个零扰动实现 = g(t) 名义轨迹(2026-07-28 用户指出 v4 缺 g 层)。
    #   名义 run 自带 12 帧快照与井曲线,口径与实现完全一致。
    # ★2026-08-04:随机实现 5 → 10(共 11/场景)。动机 = 把经验均值/标准差的采样噪声降下来
    #   (采样地板 ∝ 1/√N_s,10 比 5 低 41%;σ̂ 的相对误差 35%→24%)。
    #   ⚠️ 向后兼容已验证:z 由 default_rng(770000+sid*1000+r) **逐实现独立播种**,
    #   与 N_REAL 无关 ⇒ 已产的 r000-r004 与 rnom 逐位不变(600 例实测:556 首抽命中、
    #   44 重抽命中且与 metadata z_rejects 逐一吻合、0 无法解释)。本改动为纯增量补跑 r005-r009。
    G.NOMINAL_LAST = True
    # ★2026-08-09:随机实现 10 → 15(共 16/场景,末槽仍为 rnom)。用户指令(降组均/组σ采样噪声:
    #   组均 SE 0.85→0.70 MPa,σ̂ 相对误差 23.6%→18.9%)。向后兼容与 5→10 那次同构:
    #   z = default_rng(770000+zsid*1000+r) 逐实现独立播种,与 N_REAL 无关;nominal 按**名字**
    #   (_rnom)而非槽位判重,z≡0 与槽位无关 ⇒ 已产 r000-r009 与 rnom 逐位不变,
    #   本改动 = 纯增量补跑 r010-r014。
    G.N_REAL = 16
    if a.mode == "main":
        G.scenario_params = scenario_params_v4
        G.N_SCENARIOS = a.n
        G.build_case_template = lambda: build_case_template_v4(snap=True)   # 方案B:全模式快照
    elif a.mode == "sweep":
        G.scenario_params = scenario_params_sweep
        G.N_SCENARIOS = N_GEO * N_Y
        G.build_case_template = lambda: build_case_template_v4(snap=True)   # 方案B:全模式快照
    else:                                     # truth
        G.scenario_params = scenario_params_truth
        G.N_SCENARIOS = N_TRUTH
        G.build_case_template = lambda: build_case_template_v4(snap=True)
    tagmap = {"main": "v4m", "sweep": "v4s", "truth": "v4t"}
    # 场景命名前缀 = 模式名,避免与 v3/其它模式冲突(v3 用 "train"/"gold"/"smoke")
    G.TAG_OVERRIDE = tagmap[a.mode]

    sys.argv = [sys.argv[0], "--shard", a.shard, "--ranks", str(a.ranks), "--pin", a.pin] \
               + (["--smoke"] if a.smoke else [])
    G.main()


if __name__ == "__main__":
    main()
