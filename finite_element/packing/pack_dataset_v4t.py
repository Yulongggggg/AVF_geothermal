# SPDX-License-Identifier: LGPL-2.1-only
import argparse, csv, glob, hashlib, json, os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np

D = Path("data/r5")
OUT = Path("pack_v4r5t")
H, W, NS = 160, 128, 128
T = 11
N_Y = 45
VAL_PCT, TEST_PCT = 8, 8
PHYS = "v4-prodbhp-variable-r5"
PHYS_R6D = "v4-coussy-depth-r6d"
D_REF_KM, PP_IC_REF = 3.06, 30.0

PREFIX_GEO_DIV = {"v4cg": 4, "v4cc": 1, "v4m": 1, "v4s": N_Y,
                  "v4n": 1, "v4u": 7, "v4x": 1, "v4d": 1}
_PREFIXES = sorted(PREFIX_GEO_DIV, key=len, reverse=True)
R5_PREFIXES = ("v4m", "v4s")
R6_PREFIXES = ("v4n", "v4u", "v4x", "v4cg", "v4cc")

_XG = np.linspace(-400, 400, W)[None, :].repeat(H, 0)
_YG = np.linspace(-500, 500, H)[:, None].repeat(W, 1)
INJ_MASK = (np.hypot(_XG - 0.0, _YG + 250.0) <= 15.0)

def prefix_of(scen):
    for p in _PREFIXES:
        if scen.startswith(p):
            return p
    raise SystemExit(f"❌ 未注册的场景前缀: {scen}")

def sid_of(scen):
    return int(scen[len(prefix_of(scen)):])

def geo_of(scen):

    p = prefix_of(scen)
    g = sid_of(scen) // PREFIX_GEO_DIV[p]
    return g if p in R5_PREFIXES else f"{p}:{g}"

def split_of(scen, img=None):

    p = prefix_of(scen)
    if p in R5_PREFIXES:
        h = int(hashlib.md5(f"v4g{geo_of(scen)}".encode()).hexdigest()[:8], 16) % 100
    else:
        if img is None or int(img) < 0:
            raise SystemExit(f"❌ {scen}: r6 块 split 需要 img(metadata 缺行或缺 img 列)")
        h = int(hashlib.md5(f"r6img{int(img)}".encode()).hexdigest()[:8], 16) % 100
    if h < TEST_PCT:
        s = "test"
    elif h < TEST_PCT + VAL_PCT:
        s = "val"
    else:
        s = "train"
    if p == "v4x" and s != "train":
        return "xheld"
    return s

def scen_of(cid):
    return cid.split("_")[0]

def real_of(cid):
    r = cid.split("_")[1]
    return "nom" if r == "rnom" else r[1:]

def read_fp(job):

    p, exp_phys = job
    try:
        d = np.load(p)
        z = d["z"]
        st_ = np.asarray(d["snap_times"], np.float64)
        tend = float(np.asarray(d["curve_t"], np.float64)[-1])
        pv = str(d["physics_version"])
        d.close()
        if pv != exp_phys:
            return p, None, None, None, None, f"physics_version={pv} ≠ {exp_phys}(混根?)"
        fp = hashlib.sha1(np.ascontiguousarray(z, dtype=np.float32).tobytes()).hexdigest()[:16]
        return p, fp, st_, tend, pv, None
    except Exception as e:
        return p, None, None, None, None, f"读取失败 {e}"

_W = {}
def _init_w(out):
    _W["F"] = np.load(Path(out)/"fields.npy", mmap_mode="r+")
    _W["ref_times"] = np.load(Path(out)/"_ref_times.npy")

def write_one(job):

    p, row = job
    try:
        d = np.load(p)
        snap = np.asarray(d["snap_fields"], np.float32)
        fin = np.asarray(d["fields"], np.float32)
        st_ = np.asarray(d["snap_times"], np.float64)
        d.close()
        if snap.shape != (T-1, 2, H, W) or fin.shape != (2, H, W):
            return p, None, None, f"形状异常 snap={snap.shape} fin={fin.shape}"
        if not np.allclose(st_, _W["ref_times"][:T-1], rtol=0, atol=0.5):
            return p, None, None, f"snap_times 偏离参考 {st_[:3]}..."
        ft = np.concatenate([snap, fin[None]], 0)
        if not np.isfinite(ft).all():
            return p, None, None, "含 NaN/Inf"
        _W["F"][row] = ft
        dc = ft.mean(axis=(2, 3))
        inj = ft[:, 1][:, INJ_MASK].mean(axis=1) - dc[:, 1]
        return p, dc.astype(np.float32), inj.astype(np.float32), None
    except Exception as e:
        return p, None, None, f"写入失败 {e}"

def load_meta(roots):
    meta = {}
    for root in roots:
        for f in sorted(glob.glob(str(root/"metadata_*_shard*.csv"))):
            for r in csv.DictReader(open(f)):
                if r.get("case_id"):
                    meta[r["case_id"]] = r
    return meta

def load_depth_sidecar(r6d):

    p = Path(r6d)/"v4d_sidecar.csv"
    if not p.exists():
        raise SystemExit(f"❌ r6d 根缺 v4d_sidecar.csv: {p}")
    sc = {}
    for r in csv.DictReader(open(p)):
        sid = int(r["sid"]); v = (float(r["depth_km"]), float(r["pp_ic_mpa"]))
        if sid in sc and any(abs(a-b) > 1e-6 for a, b in zip(sc[sid], v)):
            raise SystemExit(f"❌ v4d_sidecar sid={sid} 重复且不一致: {sc[sid]} vs {v}")
        sc[sid] = v
    return sc

def fnum(m, k, dflt=0.0):
    try: return float(m.get(k, dflt))
    except (TypeError, ValueError): return dflt

def img_of_row(r):

    try: return int(float(r.get("img", -1) or -1))
    except (TypeError, ValueError): return -1

def verify(out):
    F = np.load(out/"fields.npy", mmap_mode="r")
    I = np.load(out/"inputs.npy", mmap_mode="r")
    S = np.load(out/"scal.npy"); IDX = np.load(out/"index.npy")
    DC = np.load(out/"row_dcmean.npy"); INJ = np.load(out/"row_injexc15.npy")
    st = json.loads((out/"stats.json").read_text())
    rows = list(csv.DictReader(open(out/"meta.csv")))
    ok = True
    def chk(cond, msg):
        nonlocal ok
        print(("  ✅ " if cond else "  ❌ ") + msg); ok = ok and cond
    chk(len(rows) == F.shape[0] == S.shape[0] == IDX.shape[0] == DC.shape[0] == INJ.shape[0],
        f"行数一致 ({len(rows)})")
    chk(F.shape[1:] == (T, 2, H, W), f"fields 形状 {F.shape}")
    chk(S.shape[1] in (7, 9), f"scal 列数 {S.shape[1]}(7=老 r5 pack / 9=含 depth,pp_ic)")
    chk(IDX.max() < I.shape[0], f"场景索引不越界 (max {IDX.max()} < {I.shape[0]})")
    chk(all(r["split"] == split_of(r["scen"], img_of_row(r)) for r in rows),
        "split 与哈希纪律一致(r5=地质 md5 / r6=img md5;增量安全)")

    by_geo = defaultdict(set)
    for r in rows: by_geo[geo_of(r["scen"])].add(r["split"])
    chk(all(len(v) == 1 for v in by_geo.values()),
        f"同地质不跨 split({len(by_geo)} 个地质键,无泄漏)")

    by_img = defaultdict(set)
    for r in rows:
        if prefix_of(r["scen"]) not in R5_PREFIXES:
            by_img[img_of_row(r)].add(r["split"])
    _bad = [i for i, v in by_img.items()
            if "train" in v and (v & {"val", "test", "xheld"})]
    chk(not _bad,
        f"r6 无 train↔评测同图泄漏({len(by_img)} 张图;违例 {len(_bad)}: {_bad[:5]})")
    chk(all(r["split"] in ("train", "xheld") for r in rows if prefix_of(r["scen"]) == "v4x"),
        "v4x 尾臂永不进标准 val/test")
    sub = np.random.default_rng(0).choice(F.shape[0], min(200, F.shape[0]), replace=False)
    chk(bool(np.isfinite(F[np.sort(sub)]).all()), "抽样 200 行 × 11 时刻标签无 NaN/Inf")

    chk(bool((np.abs(np.asarray(F[np.sort(sub)])).max(axis=(1, 2, 3, 4)) > 0).all()),
        "抽样行无全零标签(P0-04)")
    chk(bool(np.isfinite(I).all()), "全部 M/C 无 NaN/Inf")
    chk(bool(np.isfinite(S).all() and np.isfinite(DC).all() and np.isfinite(INJ).all()), "标量/预计算量有限")
    chk(bool(S[:, 0].min() >= 0.5), f"q 全部 >= 0.5 (min {S[:,0].min():.2f}) —— 无零标量垃圾行")
    chk(bool(S[:, 1].min() >= 293.0), f"t_inj 全部 >= 293 K (min {S[:,1].min():.1f})")
    chk(bool((S[:, 3].min() >= -180.1) and (S[:, 3].max() <= 180.1)),
        f"well_x ∈ [-180,180] (实测 [{S[:,3].min():.1f},{S[:,3].max():.1f}])")
    chk(bool((S[:, 4].min() >= -0.1) and (S[:, 4].max() <= 360.1)),
        f"well_y ∈ [0,360] (实测 [{S[:,4].min():.1f},{S[:,4].max():.1f}])")
    chk(bool((S[:, 6].min() >= 3.99) and (S[:, 6].max() <= 14.01)),
        f"prod_dd ∈ [4,14] MPa (实测 [{S[:,6].min():.2f},{S[:,6].max():.2f}])")
    if S.shape[1] >= 9:
        isd = np.array([prefix_of(r["scen"]) == "v4d" for r in rows])
        chk(bool((np.abs(S[~isd, 7] - D_REF_KM) < 1e-3).all() and
                 (np.abs(S[~isd, 8] - PP_IC_REF) < 1e-3).all()),
            "非 v4d 行 depth=3.06 / pp_ic=30(锚切片)")
        if isd.any():
            chk(bool((S[isd, 7] >= 2.399).all() and (S[isd, 7] <= 3.601).all()),
                f"v4d depth ∈ [2.4,3.6] km (实测 [{S[isd,7].min():.3f},{S[isd,7].max():.3f}])")
            chk(bool((S[isd, 8] >= 23.4).all() and (S[isd, 8] <= 35.4).all()),
                f"v4d pp_ic ∈ [23.5,35.3] MPa (实测 [{S[isd,8].min():.2f},{S[isd,8].max():.2f}])")

    for pfx, share_well in (("v4cg", True), ("v4u", False)):
        grp = defaultdict(list)
        for i, r in enumerate(rows):
            if prefix_of(r["scen"]) == pfx:
                grp[geo_of(r["scen"])].append(i)
        if grp:
            bad = 0
            for g, ii in grp.items():
                cols = (0, 3, 4, 6) if share_well else (0, 6)
                for c in cols:
                    if np.ptp(S[ii, c]) > 0.51: bad += 1
            chk(bad == 0, f"{pfx} 配对共享量一致({len(grp)} 组)")

    scen_nom = {r["scen"] for r in rows if r["real"] == "nom"}
    scen_all = {r["scen"] for r in rows}
    chk(len(scen_nom) > 0.8*len(scen_all),
        f"nom 锚覆盖 {len(scen_nom)}/{len(scen_all)} 场景 ({100*len(scen_nom)/max(len(scen_all),1):.0f}%)")

    chk(len(st["times_s"]) == T and st["times_s"][-1] > 3.1e8,
        f"时间轴 11 档,末端 {st['times_s'][-1]:.3g}s = {st['times_s'][-1]/3.156e7:.2f} y")
    tau = np.array(st["tau_norm"])
    chk(abs(tau.mean()) < 1e-6 and abs(tau.std() - 1.0) < 1e-2, "tau_norm 已 z-score")
    n_nom = sum(1 for r in rows if r["real"] == "nom")
    tags = sorted({prefix_of(r["scen"]) for r in rows})
    print(f"  ℹ️ nom 行 {n_nom} | 实现行 {len(rows)-n_nom} | "
          f"split: " + " ".join(f"{s}={sum(1 for r in rows if r['split']==s)}"
                                for s in ("train", "val", "test", "xheld"))
          + " | tag: " + " ".join(f"{t}={sum(1 for r in rows if prefix_of(r['scen'])==t)}" for t in tags))
    print("\n" + ("✅ pack 自洽" if ok else "❌ pack 有问题"))
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="冒烟:只打前 N 个场景(stats 无效,勿用于训练)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--r5-root", default=str(D), help="r5 数据根;空串 = 不打 r5")
    ap.add_argument("--r6-root", default="", help="r6 同物理根(v4n/v4u/v4x/v4cg/v4cc);空 = 无")
    ap.add_argument("--r6d-root", default="", help="r6 深度根(v4d,新物理块);空 = 无")
    ap.add_argument("--allow-bad", action="store_true",
                    help="⚠️ 读失败/物理不符/写失败只剔除不硬停(应急用;默认硬失败,P0-04)")
    a = ap.parse_args()
    out = Path(a.out)
    if a.verify:
        raise SystemExit(0 if verify(out) else 1)
    out.mkdir(parents=True, exist_ok=True)

    sources, roots = [], []
    if a.r5_root:
        r5 = Path(a.r5_root); roots.append(r5)
        for pfx in R5_PREFIXES:
            sources += [(p, PHYS) for p in sorted(glob.glob(str(r5/"fields"/f"{pfx}*.npz")))]
    if a.r6_root:
        r6 = Path(a.r6_root); roots.append(r6)
        for pfx in R6_PREFIXES:
            sources += [(p, PHYS) for p in sorted(glob.glob(str(r6/"fields"/f"{pfx}*.npz")))]
    depth_sc = {}
    if a.r6d_root:
        r6d = Path(a.r6d_root); roots.append(r6d)
        sources += [(p, PHYS_R6D) for p in sorted(glob.glob(str(r6d/"fields"/"v4d*.npz")))]
        depth_sc = load_depth_sidecar(r6d)
    if not sources:
        raise SystemExit("❌ 没有任何数据源(检查 --r5-root/--r6-root/--r6d-root)")
    root_of = {}
    for p, _ in sources:
        root_of[p] = Path(p).parent.parent

    meta = load_meta(roots)
    if a.limit:
        keep_scens = sorted({scen_of(Path(p).stem) for p, _ in sources})[:a.limit]
        sources = [(p, e) for p, e in sources if scen_of(Path(p).stem) in set(keep_scens)]
        print(f"⚠️ --limit {a.limit}:只打 {len(sources)} 个 npz(冒烟,stats 无效)")
    print(f"候选 npz: {len(sources)}  (metadata 行 {len(meta)})", flush=True)

    fps, errs = {}, []
    ref_times = None; ref_tend = None
    with Pool(a.procs) as pool:
        for i, (p, fp, st_, tend, pv, err) in enumerate(pool.imap_unordered(read_fp, sources, chunksize=64)):
            if err:
                errs.append(f"{Path(p).stem}: {err}"); continue
            if ref_times is None:
                ref_times = st_; ref_tend = tend
                print(f"参考时间轴: {st_} + 终态 {tend:.6g}s", flush=True)
            else:
                if not np.allclose(st_, ref_times, rtol=0, atol=0.5):
                    raise SystemExit(f"❌ {Path(p).stem} snap_times 与参考不一致: {st_}")
                if abs(tend - ref_tend) > 1.0:
                    raise SystemExit(f"❌ {Path(p).stem} curve_t[-1]={tend} ≠ {ref_tend}")
            fps[p] = fp
            if (i+1) % 5000 == 0: print(f"  指纹 {i+1}/{len(sources)}", flush=True)
    print(f"读入指纹 {len(fps)},失败/剔除 {len(errs)}", flush=True)
    if errs and not a.allow_bad:
        print("❌ Pass1 存在坏文件/混根(前 20 条):\n  " + "\n  ".join(errs[:20]))
        raise SystemExit("硬失败(P0-04 纪律)。回传批次应先过 manifest 对账;确要跳过用 --allow-bad")

    by_scen = defaultdict(list)
    for p in fps: by_scen[scen_of(Path(p).stem)].append(p)
    keep, dup = [], []
    for scen in sorted(by_scen):
        seen = {}
        for p in sorted(by_scen[scen]):
            if fps[p] in seen:
                dup.append(f"{Path(p).stem} ← 与 {Path(seen[fps[p]]).stem} 同 z"); continue
            seen[fps[p]] = p; keep.append(p)
    print(f"去重后 {len(keep)} (剔除 {len(dup)})", flush=True)

    scens = sorted({scen_of(Path(p).stem) for p in keep})
    scen_row = {s: i for i, s in enumerate(scens)}
    N, S_ = len(keep), len(scens)

    times = np.concatenate([ref_times, [ref_tend]])
    np.save(out/"_ref_times.npy", times)
    lg = np.log10(times)
    tau_norm = (lg - lg.mean())/lg.std()

    F = np.lib.format.open_memmap(out/"fields.npy", mode="w+", dtype=np.float32, shape=(N, T, 2, H, W))
    del F
    I = np.lib.format.open_memmap(out/"inputs.npy", mode="w+", dtype=np.float32, shape=(S_, 2, NS, NS))
    SC = np.zeros((N, 9), np.float32); IDX = np.zeros(N, np.int64)
    DC = np.zeros((N, T, 2), np.float32); INJ = np.zeros((N, T), np.float32)

    row_of = {p: i for i, p in enumerate(keep)}
    errs2 = []
    with Pool(a.procs, initializer=_init_w, initargs=(str(out),)) as pool:
        for i, (p, dc, inj, err) in enumerate(pool.imap_unordered(write_one, list(row_of.items()), chunksize=16)):
            if err:
                errs2.append(f"{Path(p).stem}: {err}")
            else:
                DC[row_of[p]] = dc; INJ[row_of[p]] = inj
            if (i+1) % 2000 == 0: print(f"  写入 {i+1}/{N}", flush=True)
    if errs2:
        print("❌ Pass2 写入失败(会留下全零标签行):\n  " + "\n  ".join(errs2[:20]))
        if not a.allow_bad:
            raise SystemExit("硬失败:pack 不允许含全零行(P0-04)。修复源数据后重跑")
        raise SystemExit("--allow-bad 也无法救 pass2:全零行已写入 memmap,必须修复源数据后重跑")

    rows = []; miss_inp = []; xchk_bad = []
    inp_cache = {}
    for i, p in enumerate(keep):
        cid = Path(p).stem; scen = scen_of(cid); pfx = prefix_of(scen)
        if scen not in inp_cache:
            ip = root_of[p]/"inputs"/f"{scen}.npz"
            if not ip.exists():
                miss_inp.append(scen); inp_cache[scen] = None
            else:
                sd = np.load(ip, allow_pickle=True)
                I[scen_row[scen]] = np.stack([sd["M"], sd["C"]], 0)
                inp_cache[scen] = dict(q=float(sd["q"]), t_inj=float(sd["t_inj"]), s_max=float(sd["s_max"]),
                                       well_x=float(sd["well_x"]), well_y=float(sd["well_y"]),
                                       prod_dd=float(sd["prod_dd"]),
                                       prod_p=float(sd["prod_p_mpa"]) if "prod_p_mpa" in sd.files else None,
                                       cbar=float(np.asarray(sd["C"]).mean()))
                sd.close()
        d_ = inp_cache[scen]
        if d_ is None:
            raise SystemExit(f"❌ 场景 {scen} 缺 inputs npz —— v4 标量以 inputs 为准,不能带垃圾行")
        m = meta.get(cid, {})
        if m:
            for k, col in (("q", "q"), ("well_y", "well_y"), ("well_x", "well_x"), ("prod_dd", "prod_dd")):
                mv = fnum(m, col, float("nan"))
                if np.isfinite(mv) and abs(mv - d_[k]) > 0.51:
                    xchk_bad.append(f"{cid}: {k} inputs={d_[k]} metadata={mv}")
        img = int(fnum(m, "img", -1)) if m else -1
        if pfx not in R5_PREFIXES and img < 0:
            raise SystemExit(f"❌ {cid}: r6 块缺 metadata img(split 依据),不能打包")
        if pfx == "v4d":
            sid = sid_of(scen)
            if sid not in depth_sc:
                raise SystemExit(f"❌ v4d {scen} 缺 sidecar 深度行(v4d_sidecar.csv)")
            dkm, ppic = depth_sc[sid]
        else:
            dkm, ppic = D_REF_KM, PP_IC_REF

        if d_["prod_p"] is not None:
            pp_pred = ppic - 0.00981*d_["well_y"] - d_["prod_dd"]
            if abs(pp_pred - d_["prod_p"]) > 0.02:
                xchk_bad.append(f"{cid}: prod_p 恒等式失败 pred={pp_pred:.4f} inputs={d_['prod_p']:.4f}"
                                f" (depth={dkm})")
        SC[i] = [d_["q"], d_["t_inj"], d_["s_max"], d_["well_x"], d_["well_y"], d_["cbar"], d_["prod_dd"],
                 dkm, ppic]
        IDX[i] = scen_row[scen]
        rows.append(dict(case_id=cid, scen=scen, tag=pfx, real=real_of(cid),
                         geo=geo_of(scen), split=split_of(scen, img), row=i, scen_row=scen_row[scen],
                         img=img, depth_km=dkm))
    I.flush()
    if xchk_bad:
        print("❌ inputs/metadata 标量不一致:\n  " + "\n  ".join(xchk_bad[:10]))
        raise SystemExit(1)

    np.save(out/"scal.npy", SC); np.save(out/"index.npy", IDX)
    np.save(out/"row_dcmean.npy", DC); np.save(out/"row_injexc15.npy", INJ)
    with (out/"meta.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case_id", "scen", "tag", "real", "geo", "split",
                                           "row", "scen_row", "img", "depth_km"])
        w.writeheader(); w.writerows(rows)

    F = np.load(out/"fields.npy", mmap_mode="r")
    tr = np.array(sorted(r["row"] for r in rows if r["split"] == "train" and r["real"] != "nom"))
    trs = sorted({r["scen_row"] for r in rows if r["split"] == "train"})
    s1 = np.zeros(2); s2 = np.zeros(2); cnt = 0
    for j in range(0, len(tr), 256):
        blk = np.asarray(F[tr[j:j+256]], np.float64)
        s1 += blk.sum(axis=(0, 1, 3, 4)); s2 += (blk**2).sum(axis=(0, 1, 3, 4))
        cnt += blk.shape[0]*T*H*W
        if (j//256) % 10 == 0: print(f"  stats {j}/{len(tr)}", flush=True)
    fm = s1/cnt; fs = np.sqrt(np.maximum(s2/cnt - fm**2, 1e-12))
    stats = dict(fields_mean=fm.tolist(), fields_std=fs.tolist(),
                 M_mean=float(I[trs, 0].mean()), M_std=float(I[trs, 0].std()),
                 scal_mean=SC[tr].mean(0).tolist(), scal_std=(SC[tr].std(0) + 1e-6).tolist(),
                 times_s=times.tolist(), tau_norm=tau_norm.tolist(),
                 N=N, S=S_, H=H, W=W, T=T, physics_version=PHYS,
                 physics_versions=sorted({PHYS} | ({PHYS_R6D} if a.r6d_root else set())),
                 roots=[str(r) for r in roots],
                 n_dup_removed=len(dup), n_read_errors=len(errs),
                 limited=bool(a.limit))
    (out/"stats.json").write_text(json.dumps(stats, indent=1))
    (out/"pack_report.json").write_text(json.dumps(
        dict(dup_removed=dup[:100], read_errors=errs[:50], missing_inputs=miss_inp[:50],
             split_counts={s: sum(1 for r in rows if r["split"] == s)
                           for s in ("train", "val", "test", "xheld")},
             tag_counts={t: sum(1 for r in rows if r["tag"] == t)
                         for t in sorted({r["tag"] for r in rows})},
             n_nom=sum(1 for r in rows if r["real"] == "nom")), indent=1))
    print(f"\n打包完成 → {out}")
    print(f"  fields {F.shape} ({F.shape[0]*T*2*H*W*4/1e9:.1f} GB)")
    print(f"  dT mean {fm[0]:.3f} std {fs[0]:.3f} K | dp mean {fm[1]:.3f} std {fs[1]:.3f} MPa "
          f"(全时刻合并,物理单位口径)")
    print()
    if not verify(out):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
