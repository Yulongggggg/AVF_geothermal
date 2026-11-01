# SPDX-License-Identifier: LGPL-2.1-only
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

NC = 40
LX = 800.0
xc = np.linspace(-LX/2, LX/2, NC)
yc = np.linspace(-LX/2, LX/2, NC)
THR_LOG10 = 13.9
FIT_A, FIT_B = 19.3, -203.4

def inj_proxy(lg10_coarse, inj=(0.0, -250.0), prod_y=150.0, prod_x=0.0):

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

P_BASE = 26.7
RATE_REF = 2.5

def predicted_pp(lg10_coarse, prod_y=150.0, rate=RATE_REF, prod_x=0.0):

    pp25 = FIT_A * np.log10(inj_proxy(lg10_coarse, prod_y=prod_y, prod_x=prod_x)) + FIT_B
    return P_BASE + (pp25 - P_BASE) * rate / RATE_REF

def is_admissible(lg10_coarse, prod_y=150.0, rate=RATE_REF, pp_ceiling=65.0):
    return predicted_pp(lg10_coarse, prod_y=prod_y, rate=rate) < pp_ceiling
