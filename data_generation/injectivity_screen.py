#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""注入能力筛查（2026-07-11 深夜，B2_st_0032 憋死教训）：
40² 粗网格稳态达西（调和平均面导度，注入点单位流量、生产点定压）→ 代理超压 p_inj。
用 7 个 B/B2 实测 pp_max 标定: pp ≈ 20.3·log10(proxy) − 217.7 (R²=0.66)。
筛查规则: log10(proxy) < THR=13.9（预测 pp≲65 MPa）。失败案例 st_0032 proxy=2.2e14
比全部通过案例(≤7.5e13)高 3×，分离清晰。
物理正当性: 现实里吞不下设计流量的 stage 不会按 2.5 kg/s 运行——这是可采性约束，
不是任意截断；数据集里记录 reject 率。生成器应在【场景级(M)与实现级(m^(j))】都筛。"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

NC = 40
LX = 800.0
xc = np.linspace(-LX/2, LX/2, NC)
yc = np.linspace(-LX/2, LX/2, NC)
THR_LOG10 = 13.9          # 预测 pp<65 MPa; 阈值可随新数据点收紧
FIT_A, FIT_B = 19.3, -203.4   # 2026-07-12 重标定(17点含2爆点); 残差+36存在(瞬态尖峰类不可先验分离)


def inj_proxy(lg10_coarse, inj=(0.0, -250.0), prod_y=150.0, prod_x=0.0):
    """lg10_coarse: (NC,NC) log10 k。返回代理值（∝超压/流量）。"""
    k = 10.0 ** np.asarray(lg10_coarse, float)
    n = NC * NC
    def idx(i, j): return i * NC + j
    A = lil_matrix((n, n)); b = np.zeros(n)
    for i in range(NC):
        for j in range(NC):
            p = idx(i, j)
            if i + 1 < NC:
                q = idx(i + 1, j)
                t = 2.0 / (1.0 / k[i, j] + 1.0 / k[i + 1, j])
                A[p, p] += t; A[q, q] += t; A[p, q] -= t; A[q, p] -= t
            if j + 1 < NC:
                q = idx(i, j + 1)
                t = 2.0 / (1.0 / k[i, j] + 1.0 / k[i, j + 1])
                A[p, p] += t; A[q, q] += t; A[p, q] -= t; A[q, p] -= t
    ii = int(np.argmin(np.abs(yc - inj[1]))); ji = int(np.argmin(np.abs(xc - inj[0])))
    ip = int(np.argmin(np.abs(yc - prod_y))); jp = int(np.argmin(np.abs(xc - prod_x)))
    src, snk = idx(ii, ji), idx(ip, jp)
    A[snk, :] = 0; A[snk, snk] = 1.0; b[snk] = 0.0
    b[src] += 1.0
    return float(spsolve(csr_matrix(A), b)[src])


P_BASE = 26.7           # 准静水基线 [MPa]（旧标定公式的截距, 超压在其上线性叠加）
RATE_REF = 2.5          # 标定流量 [kg/s]


def predicted_pp(lg10_coarse, prod_y=150.0, rate=RATE_REF, prod_x=0.0):
    """流量感知(2026-07-11深夜, 方案加入 q~U(1,3.5)): 超压 ∝ 流量, 标定点在 2.5 kg/s。"""
    pp25 = FIT_A * np.log10(inj_proxy(lg10_coarse, prod_y=prod_y, prod_x=prod_x)) + FIT_B
    return P_BASE + (pp25 - P_BASE) * rate / RATE_REF


def is_admissible(lg10_coarse, prod_y=150.0, rate=RATE_REF, pp_ceiling=65.0):
    return predicted_pp(lg10_coarse, prod_y=prod_y, rate=rate) < pp_ceiling
