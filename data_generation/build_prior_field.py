#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天然裂隙图 -> prior 渗透率 heatmap（= 数据集的 prior 均值场 m0）。

【默认 style=heatmap：用户原版方案的忠实移植】
来源: legacy/generate_June19.py 的 build_mean_logk_from_image（多次核对过的版本）。
物理图景: 2D 面 = 压裂面(SHmax-Sv 面)，面上每点的 k 是沿【厚度方向(面外, 破碎带厚度~几米, 模板 Lz)】
平均后的等效渗透率 —— 所以场是平滑 heatmap（体积平均的结果），不是面内细丝。
配方: 裂隙掩膜 -> 椭圆核卷积(裂隙密度) -> 乘(1+井邻近因子) -> 双尺度高斯平滑(σ=40,45px)
      -> score = clip(0.80*井因子 + 0.70*裂隙因子) -> log-线性映射到 [k_min, k_max]。
r_well 按原版每个场从 [150,300]m 均匀采样（--seed 可复现）。

【可选 style=damage_zone：面内细破碎带方案，留作对照实验】

用法: python3 build_prior_field.py [--indices 1-15] [--style heatmap] [--seed 42] [--preview]
产物: prior_fields/prior_XXXX_k.i (MOOSE PiecewiseMultilinear) + 预览 PNG
"""
import argparse, math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_opening, binary_closing, gaussian_filter, rotate, distance_transform_edt
from scipy.signal import fftconvolve

BD = Path("/data/yl3825/OL_Geothermal_Claude/final_scheme")
OUT = BD / "prior_fields"
IMG_DIR = Path("/home/yl3825/Pycharm_projects/Journal_Inverse_problem/Geothermal_generation/Final/natural fracture networks/images")
Lx = Ly = 800.0
X_WELL, Y_WELL = 0.0, -250.0             # 注入井（固定）
# 2026-07-10 定稿(G 设计点, sanity_Lz 实测 pp_inj=40.7MPa, ~10MW/井对, 热突破+温度场有结构):
# 原方案配方不变, 只把 k 范围整体标定到 Cornell 量级(等效单缝开度~0.9mm, 立方定律范围内)。
# 旧范围 [6e-16, 5e-13] 是 F/G 实验前的历史值。
# 2026-07-12 终版(E系列): k_max 降1个量级 → 突破谱系打开(351/335/316K), 等效开度0.43mm
# (水力开度文献区间0.1-0.5mm 正中), 回归原版 5e-13 量级(×2.7)。演化史见 FINDINGS §14。
K_MIN, K_MAX = 2.7e-15, 1.35e-12
LOGK_MIN, LOGK_MAX = np.log(K_MIN), np.log(K_MAX)   # 原版用自然对数
# 通道族 sigmoid 陡度范围(2026-07-30 提为模块常数;默认值=历史内联值,v3 行为不变。
# v4-r3 由 generate_dataset_v4 覆写为 (0.08,0.20):高低渗过渡放宽一倍,治"井在高渗墙外注定憋死")
CHANNEL_KAP2_RANGE = (0.04, 0.10)
# ---- 原版常量（legacy/generate_June19.py L103-112, L75-76）----
W_WELL, W_FRAC = 0.80, 0.70
SIGMA_LOCAL, SIGMA_CLUSTER = 40.0, 45.0   # 高斯平滑 σ [像素]（1200 网格上 ≈ 26.7 / 30 m）
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
    """原版 build_mean_logk_from_image 的忠实移植（返回自然对数 log k）"""
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
    """可选对照：面内细破碎带方案（distance transform + SRV）"""
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
    """v2.1 heatmap：各向异性SRV + soft-OR + 真双尺度 + 跨图统一归一化
    + 完井核(数值保护垫): 注入井 r_core 内保证高导——物理=完井/射孔/主缝起裂区;
      数值=定流量点源永远坐在高k单元里, 防止 pp 冲破 Water97 上限(尤其 Lz=几米时)。
    sig_local=20px≈13m(≥3个网格单元, 压掉单元级k跳变; 原15px≈10m偏细)。"""
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
    # soft-OR 三项: SRV背景 ∨ 裂隙结构 ∨ 完井核
    score = 1.0 - (1.0 - w_well*W)*(1.0 - w_frac*Ft)*(1.0 - w_core*Wcore)
    return LOGK_MIN + score*(LOGK_MAX - LOGK_MIN), x, y


def _aniso_gauss(field, theta_deg, sig_long, sig_short):
    """大尺度各向异性平滑：旋转→分离高斯→转回（走廊/大团结构用）"""
    r = rotate(field, theta_deg, reshape=False, order=1, mode="reflect")
    r = gaussian_filter(r, sigma=(sig_short, sig_long))
    return rotate(r, -theta_deg, reshape=False, order=1, mode="reflect")


def build_heatmap_struct_logk(img_path, rng, ek):
    """结构多样化 heatmap（2026-07-11 实验版）：
    (1) 受激区 = 1-3 个各向异性高斯 lobe（随机方向/纵横比/偏心, 主 lobe 锚在注入井）
        —— 物理: 微震云沿 SHmax/裂隙走向拉长、常不对称/多瓣 (Mayerhofer 2010)
    (2) 大尺度裂隙走廊 = 裂隙密度经大各向异性核平滑 —— 大团连通结构, 保证流体能流动
    (3) 原版双尺度裂隙项（保持原方案成分）
    (4) 完井核 r=25m —— 收敛保护垫 (v2.1 教训: 定流量点源必须坐在高 k 区)
    全部 ≥25m 尺度平滑; 每次调用随机采样一个配方, 返回 (log k, x, y, 配方描述)"""
    m = fracture_mask(img_path)
    ny, nx = m.shape
    mpp = Lx / nx                                     # m per pixel
    x = np.linspace(-Lx/2, Lx/2, nx); y = np.linspace(-Ly/2, Ly/2, ny)
    X, Y = np.meshgrid(x, y)

    # (1) 多 lobe 各向异性受激区
    n_lobe = int(rng.integers(1, 4))
    W = np.zeros_like(X)
    lobes = []
    for li in range(n_lobe):
        if li == 0:
            cx, cy = X_WELL, Y_WELL                   # 注入井紧凑 patch(2026-07-11晚: 原180-300m
            r_a = rng.uniform(70.0, 140.0)            # 大晕使井区全场雷同且饱和, 改小让斑块决定形态)
        else:
            off_r = rng.uniform(60.0, 200.0)
            off_t = rng.uniform(0, 2*np.pi)
            cx, cy = X_WELL + off_r*np.cos(off_t), Y_WELL + off_r*np.sin(off_t)
            r_a = rng.uniform(120.0, 260.0)
        ar = rng.uniform(1.2, 3.0)                    # 纵横比
        th = rng.uniform(0, np.pi)                    # 方向
        wl = 1.0 if li == 0 else rng.uniform(0.4, 0.9)
        dx, dy = X - cx, Y - cy
        u = dx*np.cos(th) + dy*np.sin(th)
        v = -dx*np.sin(th) + dy*np.cos(th)
        W += wl * np.exp(-(u/r_a)**2 - (v/(r_a/ar))**2)
        lobes.append(f"lobe{li}(r={r_a:.0f},ar={ar:.1f},th={np.degrees(th):.0f})")
    W = normalize_field(W)

    # (2) 大尺度裂隙走廊
    ell = normalize_field(fftconvolve(m.astype(float), ek, mode="same"))
    th_c = rng.uniform(0, 180.0)
    sig_long = rng.uniform(80.0, 160.0)               # px: ~53-107 m
    sig_short = rng.uniform(25.0, 45.0)               # px: ~17-30 m
    Fc = normalize_field(_aniso_gauss(ell, th_c, sig_long, sig_short))

    # (3) 原版双尺度裂隙项
    source = ell * (1.0 + ALPHA_FRAC_WELL*W)
    Ft = normalize_field(gaussian_filter(source, SIGMA_LOCAL)
                         + gaussian_filter(source, SIGMA_CLUSTER))

    # (3b) 斑块纹理 (Wu et al. 2021 图6 风格): 平滑高斯场 → sigmoid 阈值化 → 离散高对比 patch,
    #      patch 尺度 ~60-180 m —— 流动走通道、温度场出手指状羽流的来源。
    #      2026-07-11晚(用户): 斑块必须由裂隙分布调制——受激斑块本来就是既有裂隙滑移/扩容
    #      的产物。做法: 噪声 + β·裂隙密度 的加性偏置(裂隙密集处更易越过阈值), β 逐场采样。
    sig_tex = rng.uniform(45.0, 90.0)                 # px ≈ 30-60 m
    E_s = normalize_field(gaussian_filter(ell, sig_tex))      # 裂隙密度平滑到纹理同尺度
    beta = rng.uniform(0.4, 0.8)                      # 裂隙调制强度
    Gn = normalize_field(gaussian_filter(rng.standard_normal((ny, nx)), sig_tex))
    G = normalize_field((1.0 - beta)*Gn + beta*E_s)
    tau = rng.uniform(0.50, 0.62)
    kap = rng.uniform(0.03, 0.09)                     # 过渡更陡(原0.06-0.12)
    Gp = 1.0/(1.0 + np.exp(-(G - tau)/kap))

    # (4) 组合 + 完井核 (lobe 权重调低给纹理让路; lobe+核仍是连通性背板)
    w_w = rng.uniform(0.30, 0.55)
    w_f = rng.uniform(0.20, 0.45)
    w_c = rng.uniform(0.20, 0.45)
    w_g = rng.uniform(0.45, 0.75)
    score = np.clip(w_w*W + w_f*Ft + w_c*Fc + w_g*Gp, 0.0, 1.0)
    # 通道化家族 (2026-07-11晚, 用户: 高低渗过渡要短, 流体才憋在条带里 → 温度场成条状):
    # 按概率对组合分数整体做陡 sigmoid → 双峰化, 过渡 ~1-2 个纹理相关长度的一小段;
    # 另一半保持渐变大团家族 —— 数据集混合多种结构。
    family = "blob"
    if rng.random() < 0.5:
        family = "channel"
        q_hi = rng.uniform(0.55, 0.75)                # 高渗条带占面积 25-45%
        tau2 = float(np.quantile(score, q_hi))        # 自适应阈值(固定阈值会切出"红海")
        kap2 = rng.uniform(*CHANNEL_KAP2_RANGE)
        score = 1.0/(1.0 + np.exp(-(score - tau2)/kap2))
    r2 = (X - X_WELL)**2 + (Y - Y_WELL)**2
    core = np.exp(-r2/25.0**2)
    score = 1.0 - (1.0 - score)*(1.0 - 0.95*core)     # soft-OR 完井核
    desc = (f"{family}, {n_lobe}lobe[{';'.join(lobes)}], corridor(th={th_c:.0f},"
            f"sl={sig_long:.0f}px,ss={sig_short:.0f}px), tex(sig={sig_tex:.0f}px,"
            f"tau={tau:.2f},beta={beta:.2f}), w=({w_w:.2f},{w_f:.2f},{w_c:.2f},{w_g:.2f})")
    return LOGK_MIN + score*(LOGK_MAX - LOGK_MIN), x, y, desc


def ellipse_density_ref(idxs, ek):
    """跨图统一归一化参考: 图库椭圆核密度的95分位中位数"""
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
    r_wells = {i: float(rng.uniform(R_WELL_MIN, R_WELL_MAX)) for i in idxs}  # 原版：每场一个
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
