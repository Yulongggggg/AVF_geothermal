#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""置信度图 C(x) 的物理模型（2026-07-11 深夜定稿方案）：
C 不是随机纹理——几何由监测物理决定，= 四种证据足迹的 soft-OR 并集：
  ① 井筒盘 C_well（光纤 DTS/DAS + 测井）: 高幅小足迹 (a~0.85-0.95, r~25-60 m)
  ② 井间干扰走廊 C_corr（tracer/压力干扰）: 沿注入井→测试井的拉长带 (a~0.4-0.7)
  ③ 微震云足迹 C_seis = a_s·I(x): 照明∝exp(-r²/r_s²)·裂隙密度调制, a_s~0.5-0.8
     (严格<1: 定位误差+无震滑移 Guglielmi 2015)
  ④ 区域背景 C_bg~0.03-0.15（地质类比）
组合: C = 1 − Π(1−C_i)。置信度→不确定度: S = S_min + (1−C)(S_max−S_min)。
分布: m|M,C ~ N(M, D_S R_ℓ D_S)；实现 m^(j) = M + D_S·L·z（细 M + 粗扰动合成在调用方做）。
所有场在 40×40 粗网格（与实现采样网格一致）。"""
import numpy as np
from scipy.ndimage import gaussian_filter

NC = 40
LX = 800.0
INJ = (0.0, -250.0)
xc = np.linspace(-LX/2, LX/2, NC)
yc = np.linspace(-LX/2, LX/2, NC)
XC, YC = np.meshgrid(xc, yc)
S_MIN = 0.05                      # 不可约下限 [dec]（上游流程分辨率/模型误差）


def coarse_fracture_density(mask):
    """1200² 裂隙掩膜 → 40² 平滑密度 ∈[0,1]（微震照明的调制项）"""
    Em = gaussian_filter(mask.astype(float), 25)
    st = mask.shape[0] // NC
    Ec = np.flipud(Em[:NC*st, :NC*st].reshape(NC, st, NC, st).mean((1, 3)))
    return (Ec - Ec.min()) / (Ec.max() - Ec.min() + 1e-12)


def _disk(cx, cy, r, a):
    return a * np.exp(-((XC-cx)**2 + (YC-cy)**2) / r**2)


def _corridor(p0, p1, halfw, a):
    """点到线段距离的高斯带（tracer/压力干扰的流路足迹）"""
    d = np.hypot(p1[0]-p0[0], p1[1]-p0[1]) + 1e-9
    t = np.clip(((XC-p0[0])*(p1[0]-p0[0]) + (YC-p0[1])*(p1[1]-p0[1])) / d**2, 0, 1)
    dist2 = (XC - (p0[0]+t*(p1[0]-p0[0])))**2 + (YC - (p0[1]+t*(p1[1]-p0[1])))**2
    return a * np.exp(-dist2 / halfw**2)


def sample_confidence_cfg(rng, level=None):
    """只采样旋钮(不求值)——同一 cfg 可在任意分辨率网格上求值(128²存图/64²扰动)。"""
    y_t = float(rng.uniform(50.0, 350.0))            # 测试/监测井位（独立于未来设计井 w）
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
        # 数据集用（2026-07-12 用户: 足迹随机搭配, 泛化到不同监测场景——"有时穷有时强"）:
        # 分层采样: 先抽"监测预算" rich~U(0,1), 各足迹的出现概率与幅度随预算走。
        # 穷场景可能只有背景+注入井盘; 富场景四件全齐且幅度高。
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
    """在给定网格 (X,Y,Ec 同形状) 上求值 C。"""
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
    """旧接口(40²模块网格)——run_confidence_sens.py 兼容用。"""
    cfg = sample_confidence_cfg(rng, level=level)
    return eval_confidence(cfg, XC, YC, Ec), cfg_desc(cfg)


def confidence_to_S(C, S_max, S_min=S_MIN):
    """C∈[0,1] → 局部标准差 S [dec]（方向相反，勿混同）"""
    return S_min + (1.0 - C) * (S_max - S_min)


def sample_realization_dev(S, L, z):
    """粗网格扰动 dev40 = D_S·(L·z)；调用方负责 m_fine = M_400 + upsample(dev40)"""
    return (S.ravel() * (L @ z)).reshape(NC, NC)
