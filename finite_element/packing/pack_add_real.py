# SPDX-License-Identifier: LGPL-2.1-only
import argparse, csv, os, sys, time
from pathlib import Path
import numpy as np
from multiprocessing import Pool

PACK = Path(os.environ.get("OLGEO_PACK", "pack_v4r6t1"))
ROOT = {"v4m": "data/r5",
        "v4s": "data/r5",
        "v4t": "data/r5",
        "v4n": "data/r6",
        "v4u": "data/r6",
        "v4x": "data/r6",
        "v4d": "data/r6d"}
LX, NPERT, NSTATE, S_MIN = 800.0, 64, 128, 0.05
xp = np.linspace(-LX/2, LX/2, NPERT); xs = np.linspace(-LX/2, LX/2, NSTATE)
_L = None

def upsample(coarse, xc_, yc_, xf, yf):
    from scipy.interpolate import RegularGridInterpolator
    it = RegularGridInterpolator((yc_, xc_), coarse, bounds_error=False, fill_value=None)
    XF, YF = np.meshgrid(xf, yf)
    return it(np.column_stack([YF.ravel(), XF.ravel()])).reshape(len(yf), len(xf))

def _init():
    global _L
    _L = np.load(ROOT["v4m"] + "/L_chol64.npy")

def one(args):
    row, case_id, tag, real, M, C, s_max = args
    if real == "nom":
        return row, M.astype(np.float32), 0.0
    d = np.load(f"{ROOT[tag]}/fields/{case_id}.npz")
    z = np.asarray(d["z"], np.float64)
    C_pert = upsample(C, xs, xs, xp, xp)
    S_pert = S_MIN + (1.0 - C_pert)*(s_max - S_MIN)
    dev = (S_pert.ravel() * (_L @ z)).reshape(NPERT, NPERT)
    m = M + upsample(dev, xp, xp, xs, xs)
    return row, m.astype(np.float32), float(dev.std())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    rows = list(csv.DictReader(open(PACK/"meta.csv")))
    if a.limit: rows = rows[:a.limit]
    I = np.load(PACK/"inputs.npy", mmap_mode="r"); S = np.load(PACK/"scal.npy")
    N = len(rows); print(f"[real] N={N} pack={PACK}", flush=True)
    out_path = PACK/("real.npy" if not a.limit else "real_smoke.npy")
    R = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(N, NSTATE, NSTATE))
    jobs = ((int(r["row"]), r["case_id"], r["tag"], r["real"],
             np.asarray(I[int(r["scen_row"])][0], np.float64), np.asarray(I[int(r["scen_row"])][1], np.float64),
             float(S[int(r["row"]), 2])) for r in rows)
    t0 = time.time(); devs = []; n_nom = 0
    with Pool(a.workers, initializer=_init) as pool:
        for k, (row, m, dstd) in enumerate(pool.imap_unordered(one, jobs, chunksize=16)):
            R[row] = m; devs.append(dstd); n_nom += (dstd == 0.0)
            if (k+1) % 5000 == 0: print(f"  {k+1}/{N} {time.time()-t0:.0f}s", flush=True)
    R.flush()
    devs = np.array(devs)
    print(f"[real] done {N} rows in {time.time()-t0:.0f}s | nom {n_nom} | dev64 std 中位 {np.median(devs[devs>0]):.3f} dec"
          f" | real-M 均 {float(np.abs(np.asarray(R[:200]) - I[[int(r['scen_row']) for r in rows[:200]], 0]).mean()):.3f}", flush=True)
    assert np.isfinite(np.asarray(R[::97])).all(), "real.npy 有 NaN/Inf"
    print("PACK ADD REAL DONE")

if __name__ == "__main__":
    main()
