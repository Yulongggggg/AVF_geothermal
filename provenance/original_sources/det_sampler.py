#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""输入侧实现采样器(确定性算子路线,合作者 2026-09-01):x_i ~ μ = N(M, Σ(C)) 在 GPU 上现场抽样,
推过确定性算子 f_θ 得到 {f_θ(x_i)}。与生成器同一公式:dev64 = S(C)·(L z),S = S_MIN+(1−C)(s_max−S_MIN),
上采到状态网格后加到 M 上(pack_add_real.py 离线重建的训练用实现场与此同源)。"""
import os
import numpy as np
import torch
import torch.nn.functional as F

S_MIN, NPERT = 0.05, 64
H, W = 160, 128
PPIC_PER_KM = 9.80392          # pack scal[:,8] = 9.80392·depth_km(r6 数据集精确线性;3.06 km → 30.0 MPa)
D_REF = 3.06
NO_REAL_SCREEN_FAMILIES = ("v4s", "v4u")   # 井位扫描块:同 geo 共用 z,生成器 NO_Z_RESAMPLE(generate_dataset_v4.py:481-483 / generate_v4u.py:73-75)
GUARD_DEPTH_SLOPE = 15.7                   # v4d 护栏 PP_KILL = 72 + 15.7·(d−3.06) MPa(generate_v4d.py:112-113);其余家族常数 72
Y_INJ, GUARD_R = -250.0, 15.0  # 注入点 (0,−250);护栏口径 = 注入井块(r≤15 m)Δp 均值 + 注入点初始孔压
DP_DY = 0.00981                # MPa/m 静水梯度(pack 的 prod_p = pp_ic − 0.00981·wy − dd 同口径)


def _screen_one(job):
    import admissibility as AD
    m64, q, wx, wy = job
    return AD.screen_pp(m64, q, wx, wy)


def field_on_grid(t, span_y=400.0):
    """(B,C,n,n) 覆盖 [-400,400]² 的方形场 → (B,C,H,W) 计算网格(y∈[-500,500],缓冲带复制边缘)。"""
    yq = torch.linspace(-500, 500, H, device=t.device).clamp(-span_y, span_y)/span_y
    xq = torch.linspace(-400, 400, W, device=t.device)/400.0
    gy, gx = torch.meshgrid(yq, xq, indexing="ij")
    grid = torch.stack([gx, gy], -1)[None].expand(t.shape[0], -1, -1, -1).to(t.dtype)
    return F.grid_sample(t, grid, mode="bilinear", align_corners=True)


class RealSampler:
    """screen(★2026-09-06):=True 时对内部抽的 z 做拒绝抽样,复现数据生成器的实现级可采性筛查(admissibility.py:
    64² 实现 → 120² 两点 Darcy 代理注入压 < 天花板 60+15.7(d−3.06)−(prod_p−20))。FE 数据集里的实现集合本来就是
    "通过筛查"的条件样本;评测/应用若无条件抽样,贴天花板的场景会喂给算子本该被拒的高压实现 ⇒ 压力域均正偏、散布过宽
    (diag_screen_upo5tD.json:20 个贴天花板测试储层 |bias| 3.3→1.1 MPa、散布比 1.9→1.15,其余 44 个不变)。
    默认取环境变量 OLGEO_SCREEN(=1 开);筛查所需标量(q, 井位, prod_p, depth)从 x 的常值通道解码,也可经 meta 显式传入。
    显式传 z 时不筛(潜变量由调用方负责;可用 admissible(x, z) 取掩码)。"""
    def __init__(self, st, dev, L_path="/mnt/home/yliu5/projects/v4r5/dataset_v4_coussy_r5/L_chol64.npy",
                 screen=None, procs=None, ds=None, max_rounds=8):
        self.L = torch.from_numpy(np.load(L_path)).float().to(dev)          # (4096,4096)
        self.M_mean, self.M_std = float(st["M_mean"]), float(st["M_std"])
        self.sm_mean, self.sm_std = float(st["scal_mean"][2]), float(st["scal_std"][2])
        self.q_mean, self.q_std = float(st["scal_mean"][0]), float(st["scal_std"][0])
        self.dd_mean, self.dd_std = float(st["scal_mean"][6]), float(st["scal_std"][6])
        # prod_p / depth 通道的归一化:优先 ds 实例(GeoSetT 现场统计;缺省取 train_t._DS_REGISTRY 里最近构造的数据集),
        # 其次 stats.json 的 pp_norm/depth_norm
        if ds is None:
            import sys
            _reg = getattr(sys.modules.get("train_t"), "_DS_REGISTRY", None) if "train_t" in sys.modules else None
            if _reg: ds = _reg[-1]
        self.in_pp = getattr(ds, "in_pp", "abs")
        self.pp_mean, self.pp_std = (float(ds.pp_mean), float(ds.pp_std)) if ds is not None and hasattr(ds, "pp_mean") \
            else ((float(st["pp_norm"][0]), float(st["pp_norm"][1])) if "pp_norm" in st else (None, None))
        self.d_mean, self.d_std = (float(ds.d_mean), float(ds.d_std)) if ds is not None and hasattr(ds, "d_mean") \
            else ((float(st["depth_norm"][0]), float(st["depth_norm"][1])) if "depth_norm" in st else (None, None))
        self.screen = (os.environ.get("OLGEO_SCREEN", "0") == "1") if screen is None else bool(screen)
        # ★护栏模拟(2026-09-06):生成器的 THM 运行时守卫 pp_inj(注入点绝对孔压,逐输出时刻)> 72 MPa 即截停(rc=−15),
        #   数据集里每个保留实现都满足全程 pp_inj < 72 MPa;热场景被截停的比例可达 50–85%(metadata rc)。
        #   评测抽样必须复现:用算子自己在 11 个存储时刻的注入井块预测 + 初始孔压,超过阈值 guard_mpa 的实现剔除。
        #   阈值按元数据逐场景中止率标定(calib_guard.py;块均值 < 点值,故阈值 < 72)。OLGEO_GUARD=<MPa>,0 = 关。
        # ★配对输入模式(2026-09-06):OLGEO_PAIRED=1 ⇒ 对数据集里的场景,inputs()/draw() 不抽新实现,而是把该场景 FE 用过的实现
        #   (pack real.npy,训练时同一拼装)原样作为算子输入 ⇒ 集合审计两边同一批实现("算子替代 FE 求解"的字面含义)。
        #   仅审计脚本使用;反演/同化等应用必须用新抽实现(先验)。查不到场景 ⇒ 退回抽样。
        self.paired = os.environ.get("OLGEO_PAIRED", "0") == "1"
        if self.paired: print("[sampler] 配对输入模式:数据集场景用 FE 同一批实现作输入(查不到场景才抽样)", flush=True)
        self.guard_mpa = float(os.environ.get("OLGEO_GUARD", "0") or 0)
        # ★护栏的精确模拟(2026-09-06):生成器代理注入压在场景内部对"幸存/被 72 MPa 中止"的排序 AUC 中位 1.00(29k 尝试实现,508 场景);
        #   每个数据集场景的中止率 a_s = 元数据 rc=−15 占尝试数(abort_frac_all.json)是 FE 实验的已知属性 ⇒ 抽样时按代理压
        #   剔除最热的 a_s 比例,即得"条件于幸存"的实现分布。OLGEO_ABORT=1(缺省文件)或路径;查不到场景 ⇒ 不剔除。
        _ab = os.environ.get("OLGEO_ABORT", "")
        self.abort_frac = None
        if _ab:
            import json
            pth = _ab if _ab not in ("1", "true", "True") else "/mnt/home/yliu5/projects/ol_geo_train/abort_frac_all.json"
            _d = json.load(open(pth)); self.abort_frac = {k: (v["ab"]/v["tot"] if v["tot"] else 0.0) for k, v in _d.items()}
            print(f"[sampler] 护栏模拟(按场景中止率剔除最热实现)开启:{pth},{len(self.abort_frac)} 场景", flush=True)
        self.tau = np.asarray(st["tau_norm"], np.float32) if "tau_norm" in st else None
        self.fm = np.asarray(st["fields_mean"], np.float32); self.fsd = np.asarray(st["fields_std"], np.float32)
        yy = torch.linspace(-500, 500, H, device=dev)[:, None].expand(H, W); xx = torch.linspace(-400, 400, W, device=dev)[None].expand(H, W)
        self.inj_mask = ((xx**2 + (yy - Y_INJ)**2) <= GUARD_R**2).float()
        if self.guard_mpa > 0: print(f"[sampler] 护栏模拟开启:注入井块绝对压力(11 时刻最大)< {self.guard_mpa:g} MPa", flush=True)
        self.procs = int(os.environ.get("OLGEO_SCREEN_PROCS", "4")) if procs is None else int(procs)
        self.max_rounds = int(max_rounds); self._pool = None; self.last_screen = None; self._ds = ds; self.n_lookup = [0, 0]
        if self.screen: print(f"[sampler] 实现级可采性筛查开启(拒绝抽样,{self.procs} 进程,最多 {self.max_rounds} 轮)", flush=True)
        # 64² 扰动网格点在计算网格里的采样坐标(x/400, y/500)
        xp = torch.linspace(-400, 400, NPERT, device=dev)
        gy, gx = torch.meshgrid(xp/500.0, xp/400.0, indexing="ij")
        self.grid64 = torch.stack([gx, gy], -1)[None]                        # (1,64,64,2)

    # ---------- 可采性筛查 ----------
    def _ds_lookup(self, x, meta):
        """从已加载的 GeoSetT(sys.modules 里的 train_t 数据集实例,或 ds=)按标量键 + 场值匹配找回该场景的 128² M,
        用与生成器同源的单次插值(128²→64²,admissibility.M64_from_M128)得到 M64;匹配不到返回 None(退回网格插值)。"""
        import sys
        dss = [self._ds] if self._ds is not None else []
        tt = sys.modules.get("train_t")
        if tt is not None: dss += [d for d in getattr(tt, "_DS_REGISTRY", []) if d is not self._ds]
        key = (round(meta["q"], 3), round(meta["wx"], 1), round(meta["wy"], 1))
        x0 = x[0, 0].float()
        best = None
        for ds in dss:
            reg = getattr(ds, "_scen_key_index", None)
            if reg is None:
                reg = {}
                for s_, rows in ds.scen_rows.items():
                    r0 = rows[0]; k_ = (round(float(ds.S[r0, 0]), 3), round(float(ds.S[r0, 3]), 1), round(float(ds.S[r0, 4]), 1))
                    reg.setdefault(k_, []).append(r0)
                ds._scen_key_index = reg
            for r0 in reg.get(key, []):
                mc = ds._mc_on_grid(int(ds.IDX[r0]))
                err = float((((mc[0].to(x0.device) - self.M_mean)/self.M_std - x0).abs().mean()))
                if best is None or err < best[0]: best = (err, ds, r0)
        if best is None or best[0] > 1e-3: return None
        _, ds, r0 = best
        import admissibility as AD
        self.last_scen = next((s_ for s_, rows in ds.scen_rows.items() if rows[0] == r0), None)
        return AD.M64_from_M128(np.asarray(ds.I[int(ds.IDX[r0])][0], float))     # 只回查 M64;S64 按当前 x 的 C(可能已更新)算

    def _pool_get(self):
        """筛查工作池(fork)。父进程若已开 BLAS/OMP 多线程,fork 出的工人偶发 futex 死锁(2026-09-06 登录节点复现),
        故 _screen_batch 对 map 设超时:超时 ⇒ 重建池一次 ⇒ 再超时退回串行。评测作业请 OMP_NUM_THREADS=1。"""
        if self._pool is None and self.procs > 1:
            if os.environ.get("OMP_NUM_THREADS", "") != "1":
                print("[sampler] OMP_NUM_THREADS≠1:fork 工作池会死锁(OpenMP 版 BLAS),筛查改串行;评测作业请 export OMP_NUM_THREADS=1", flush=True)
                self.procs = 1; return None
            import multiprocessing as mp
            self._pool = mp.get_context("fork").Pool(self.procs)
        return self._pool

    def _screen_batch(self, jobs):
        import multiprocessing as mp
        pool = self._pool_get()
        if pool is None: return [_screen_one(j) for j in jobs]
        for attempt in range(2):
            try:
                return pool.map_async(_screen_one, jobs).get(timeout=30.0 + 0.5*len(jobs))
            except mp.TimeoutError:
                print(f"[sampler] 筛查工作池超时(疑似 fork 死锁,第 {attempt+1} 次)—— {'重建池' if attempt == 0 else '退回串行'}", flush=True)
                pool.terminate(); pool.join(); self._pool = None
                if attempt == 0: pool = self._pool_get()
        self.procs = 1
        return [_screen_one(j) for j in jobs]

    def decode_meta(self, x, meta=None):
        """从 x 的常值通道解码筛查标量(q, wx, wy, prod_p, depth);meta 里给出的键优先。通道:7=q,5=r_prod,10=prod_p(abs)/dd,12=depth。"""
        x0 = x[0].float(); meta = dict(meta or {})
        if "q" not in meta: meta["q"] = float(x0[7, 0, 0])*self.q_std + self.q_mean
        if "wx" not in meta or "wy" not in meta:
            r2 = (x0[5]*400.0)**2                                     # (H,W) 到生产井的距离²
            i, j = np.unravel_index(int(torch.argmin(r2)), r2.shape); i = min(max(i, 1), H-2); j = min(max(j, 1), W-2)
            XL = torch.linspace(-400, 400, W, device=x.device); YL = torch.linspace(-500, 500, H, device=x.device)
            # r²(j+1)−r²(j−1) = (x_{j+1}²−x_{j−1}²) − 2wx(x_{j+1}−x_{j−1}) ⇒ 精确解(连续 r,线性一元)
            wx = float(((XL[j+1]**2 - XL[j-1]**2) - (r2[i, j+1] - r2[i, j-1]))/(2*(XL[j+1] - XL[j-1])))
            wy = float(((YL[i+1]**2 - YL[i-1]**2) - (r2[i+1, j] - r2[i-1, j]))/(2*(YL[i+1] - YL[i-1])))
            meta.setdefault("wx", wx); meta.setdefault("wy", wy)
        if "smax" not in meta: meta["smax"] = float(x0[9, 0, 0])*self.sm_std + self.sm_mean
        if "depth" not in meta:
            meta["depth"] = float(x0[12, 0, 0])*self.d_std + self.d_mean if (x0.shape[0] > 12 and self.d_mean is not None) else D_REF
        if "prod_p" not in meta:
            if self.in_pp == "abs":
                if self.pp_mean is None: raise RuntimeError("筛查需要 prod_p 归一化(传 ds= 或 stats.json 的 pp_norm)")
                meta["prod_p"] = float(x0[10, 0, 0])*self.pp_std + self.pp_mean
            else:
                dd = float(x0[10, 0, 0])*self.dd_std + self.dd_mean
                meta["prod_p"] = PPIC_PER_KM*meta["depth"] - 0.00981*meta["wy"] - dd
        return meta

    @torch.no_grad()
    def admissible(self, x, z, meta=None):
        """z (K,4096) → bool 掩码 (K,) 与代理注入压 (K,):按生成器规则判定每个实现是否可采。"""
        import admissibility as AD
        meta = self.decode_meta(x, meta)
        self.last_scen = None
        M64 = meta["M64"] if "M64" in meta else self._ds_lookup(x, meta)
        fam = meta.get("family") or (self.last_scen[:3] if self.last_scen else None); self.last_family = fam
        if M64 is not None:
            M64 = np.asarray(M64, float); self.n_lookup[0] += 1
        else:
            M64 = (F.grid_sample(x[:, 0:1].float(), self.grid64, mode="bilinear", align_corners=True)[0, 0]*self.M_std + self.M_mean).cpu().numpy(); self.n_lookup[1] += 1
        S64 = self._S64(x)[0]                                        # 扰动幅度始终取当前 x 的 C 通道(P4 分级 C / 孪生环更新后的 C′)
        dev64 = (S64.reshape(1, -1)*(z.to(x.device, torch.float32) @ self.L.T)).view(-1, NPERT, NPERT).cpu().numpy()
        m64 = M64[None] + dev64                                       # (K,64,64) log10 k 实现
        jobs = [(m64[k], meta["q"], meta["wx"], meta["wy"]) for k in range(dev64.shape[0])]
        pps = np.array(self._screen_batch(jobs))
        ceil = AD.eff_ceiling(meta["prod_p"], meta["depth"])
        self.last_m64, self.last_dev64, self.last_M64, self.last_pps = m64, dev64, M64, pps
        if fam in NO_REAL_SCREEN_FAMILIES:                            # 扫描块:生成器 NO_Z_RESAMPLE,实现级筛查不存在 ⇒ 全部可采(pps 仅作特征)
            return torch.ones(z.shape[0], dtype=torch.bool, device=x.device), pps, ceil
        return torch.from_numpy(pps < ceil).to(x.device), pps, ceil

    def _S64(self, x):
        C64 = F.grid_sample(x[:, 1:2].float(), self.grid64, mode="bilinear", align_corners=True)[:, 0]   # (1,64,64)
        s_max = x[:, 9, 0, 0].float()*self.sm_std + self.sm_mean
        return S_MIN + (1.0 - C64)*(s_max[:, None, None] - S_MIN)                                       # (1,64,64)

    def _cache_key(self, x, meta):
        """同一场景(不同时刻只差 τ 通道)共用一份可采 z:键 = 筛查标量 + 通道 0 的粗采样值。"""
        x0 = x[0, 0].float()
        x1 = x[0, 1].float()
        return (round(meta["q"], 3), round(meta["wx"], 1), round(meta["wy"], 1), round(meta["prod_p"], 2), round(float(x[0, 9, 0, 0]), 4),
                tuple(np.round(x0[::20, ::16].cpu().numpy().ravel(), 3).tolist()), tuple(np.round(x1[::20, ::16].cpu().numpy().ravel(), 3).tolist()))

    @torch.no_grad()
    def draw_z(self, x, K, meta=None):
        """内部抽样:screen 关 ⇒ randn;开 ⇒ 拒绝抽样直到 K 个可采(最多 max_rounds 轮,不足则按代理压最低补齐)。
        ★缓存:同一场景再次请求时复用已筛出的可采 z(各时刻同一批实现,与 FE 集合的成员跨时刻一致相符;省 11× 筛查)。"""
        if not self.screen:
            return torch.randn(K, NPERT*NPERT, device=x.device)
        meta = self.decode_meta(x, meta); key = self._cache_key(x, meta)
        if not hasattr(self, "_zcache"): self._zcache = {}
        hit = self._zcache.get(key)
        if hit is not None and hit[0].shape[0] >= K:
            self.last_screen = dict(hit[1], cached=True); return hit[0][:K].to(x.device)
        acc_z, rej = ([hit[0].to(x.device)] if hit is not None else []), []; n_acc = (hit[0].shape[0] if hit is not None else 0); n_try = 0; n_acc0 = n_acc
        for r in range(self.max_rounds):
            need = K - n_acc; n_draw = int(np.ceil(need*(1.6 if r == 0 else max(1.2*n_try/max(n_acc, 1), 1.5))))
            if r == 0 and self.abort_frac is not None:                # 首轮按可能的中止率放大抽样量(场景名要先查,用 decode 后的 meta 键估计)
                n_draw = int(np.ceil(n_draw/max(1e-3, 1 - min(0.9, getattr(self, "last_abort", 0.0) or 0.0))))
            z = torch.randn(n_draw, NPERT*NPERT, device=x.device); ok, pps, ceil = self.admissible(x, z, meta)
            a_s = self.abort_frac.get(self.last_scen, None) if (self.abort_frac is not None and self.last_scen) else None
            if a_s:                                                   # 护栏精确模拟:通过筛查的候选里按代理压剔除最热的 a_s 比例
                okn = ok.cpu().numpy(); idx_ok = np.where(okn)[0]
                n_drop = int(round(a_s*len(idx_ok)))
                if n_drop > 0:
                    hot = idx_ok[np.argsort(-pps[idx_ok])[:n_drop]]; okn[hot] = False; ok = torch.from_numpy(okn).to(x.device)
                self.last_abort = a_s
            else:
                ps = self.survival_prob(meta, self.last_family)
                if ps is not None:                                    # 无已知中止率:退回生存分类器的概率稀疏
                    keep = torch.from_numpy(np.random.default_rng(int(abs(hash(key)) % 2**31) + r).random(len(ps)) < ps).to(x.device)
                    ok = ok & keep; self.last_surv = float(np.mean(ps))
            n_try += n_draw; acc_z.append(z[ok]); n_acc += int(ok.sum()); rej.append((z[~ok], pps[~ok.cpu().numpy()]))
            if n_acc >= K: break
        zs = torch.cat(acc_z, 0)[:K]
        if zs.shape[0] < K:                                            # 极端场景:补齐代理压最低的被拒实现(生成器 40 次重抽后同样放行)
            zr = torch.cat([a for a, _ in rej], 0); pr = np.concatenate([b for _, b in rej]); zs = torch.cat([zs, zr[np.argsort(pr)[:K - zs.shape[0]]]], 0)
        self.last_screen = dict(n_try=n_try, n_acc=n_acc - n_acc0, accept=(n_acc - n_acc0)/max(n_try, 1), ceil=float(ceil) if n_try else None,
                                family=self.last_family, scen=self.last_scen, abort_frac=getattr(self, "last_abort", None), surv_mean=getattr(self, "last_surv", None))
        self.last_abort = None
        if len(self._zcache) >= 1024: self._zcache.pop(next(iter(self._zcache)))
        self._zcache[key] = (zs.detach().clone(), dict(self.last_screen))
        return zs

    @torch.no_grad()
    def paired_inputs(self, x, K, meta=None):
        """配对输入:查到场景 ⇒ 返回 (n,C,H,W),n = 该场景 FE 实现数(≤K,不含 nom);查不到 ⇒ None。"""
        meta = self.decode_meta(x, meta); self.last_scen = None; self._ds_lookup(x, meta)
        if not self.last_scen: return None
        import sys
        for ds in ([self._ds] if self._ds is not None else []) + list(getattr(sys.modules.get("train_t"), "_DS_REGISTRY", [])):
            if ds is None or self.last_scen not in getattr(ds, "scen_rows", {}): continue
            R = getattr(ds, "R", None)
            if R is None:                                                                  # 审计脚本 det=0 ⇒ 数据集未载 real.npy,按 pack 路径直接载
                if getattr(self, "_R", None) is None:
                    from pathlib import Path
                    pk = Path(os.environ.get("OLGEO_PACK", "/mnt/home/yliu5/projects/ol_geo_train/pack_v4r6t1"))
                    self._R = np.load(pk/"real.npy", mmap_mode="r"); print(f"[sampler] 配对输入:载入 {pk/'real.npy'}", flush=True)
                R = self._R
            rows = [r for r in ds.scen_rows[self.last_scen]][:K]
            m = torch.from_numpy(np.stack([np.array(R[r], copy=True) for r in rows])).float()[:, None]
            Rg = field_on_grid(m)[:, 0].to(x.device)                                       # (n,H,W) 原始 log10
            n = len(rows)
            if n < K and os.environ.get("OLGEO_PAIRED_PAD", "1") == "1":                  # 调用方按 K 预分配 ⇒ 循环复制补齐(经验分布不变)
                Rg = Rg[torch.arange(K, device=x.device) % n]
            X = x.float().expand(Rg.shape[0], -1, -1, -1).clone(); X[:, 0] = (Rg - self.M_mean)/self.M_std
            self.last_screen = dict(paired=True, n=n, k=int(Rg.shape[0]), scen=self.last_scen); self.last_z = None
            return X.to(x.dtype)
        return None

    @torch.no_grad()
    def inputs(self, x, K, z=None, meta=None):
        """x (1,C,H,W) 单个 (M,C) 条件 → (K,C,H,W) K 个实现的输入(通道 0 换成实现场,其余复制)。
        z=None ⇒ 内部抽样(screen 开时为拒绝抽样);z=(K,4096) 标准正态 ⇒ 潜变量显式传入(SMC/反演用,不筛)。"""
        if z is None and self.paired:
            Xp = self.paired_inputs(x, K, meta)
            if Xp is not None: return Xp
        S64 = self._S64(x)
        if z is None:
            z = self.draw_z(x, K, meta)
        else:
            z = z.to(x.device, torch.float32); K = z.shape[0]
        self.last_z = z
        dev64 = (S64.reshape(1, -1) * (z @ self.L.T)).view(K, 1, NPERT, NPERT)               # (K,1,64,64)
        dev = field_on_grid(dev64)[:, 0]/self.M_std                                          # 标准化单位
        X = x.float().expand(K, -1, -1, -1).clone()
        X[:, 0] = X[:, 0] + dev
        return X.to(x.dtype)

    # ---------- 生存模型(72 MPa 护栏的统计模拟) ----------
    def _survival_load(self):
        """OLGEO_SURVIVAL=<survival_model.json | survival_mlp.pt>:由元数据里全部尝试实现(幸存/rc=−15)拟合的 P(中止|实现特征)。"""
        if getattr(self, "_surv", None) is not None: return self._surv
        pth = os.environ.get("OLGEO_SURVIVAL", "")
        if not pth: self._surv = False; return False
        import json
        if pth.endswith(".json"):
            d = json.load(open(pth)); self._surv = dict(kind="logreg", feats=d["feats"], mu=np.array(d["mu"]), sd=np.array(d["sd"]), w=np.array(d["w"]))
        else:
            d = torch.load(pth, map_location="cpu", weights_only=False)
            n_in = len(d["feats"]); net = torch.nn.Sequential(torch.nn.Linear(n_in, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1))
            net.load_state_dict(d["state"]); net.eval(); self._surv = dict(kind="mlp", feats=d["feats"], mu=np.asarray(d["mu"]), sd=np.asarray(d["sd"]), net=net)
        print(f"[sampler] 生存模型({self._surv['kind']})开启:{pth}", flush=True)
        return self._surv

    def survival_features(self, meta, fam):
        """与 build_survival_set.py / fit_survival.py 的 FEATS 逐字同口径(用 admissible() 留下的 m64/dev64/pps)。"""
        import admissibility as AD
        m64, dev, pps, M64 = self.last_m64, self.last_dev64, self.last_pps, self.last_M64
        xp = AD.xp; ji = int(np.argmin(np.abs(xp - 0.0))); ii = int(np.argmin(np.abs(xp + 250.0)))
        jp = int(np.argmin(np.abs(xp - meta["wx"]))); ip = int(np.argmin(np.abs(xp - meta["wy"])))
        blk = m64[:, ii-1:ii+2, ji-1:ji+2].reshape(len(m64), -1); m_inj = blk.mean(1); m_inj_h = np.log10(1/np.mean(1/10**blk, 1))
        m_prod = m64[:, max(ip-1, 0):ip+2, max(jp-1, 0):jp+2].reshape(len(m64), -1).mean(1)
        ceil = AD.eff_ceiling(meta["prod_p"], meta["depth"]); q = meta["q"]; n = len(m64)
        cols = {"pp_rel": pps - ceil, "pp_log": np.log10(np.maximum(pps - 26.7, 0.1)), "q": np.full(n, q), "m_inj": m_inj, "m_inj_h": m_inj_h, "m_prod": m_prod,
                "dev_mean": dev.reshape(n, -1).mean(1), "dev_std": dev.reshape(n, -1).std(1), "m_mean": m64.reshape(n, -1).mean(1),
                "M_inj": np.full(n, M64[ii-1:ii+2, ji-1:ji+2].mean()), "depth": np.full(n, meta["depth"]), "prod_p": np.full(n, meta["prod_p"]),
                "smax": np.full(n, meta.get("smax", np.nan)), "is_v4s": np.full(n, float(fam == "v4s")), "is_v4d": np.full(n, float(fam == "v4d")), "is_v4n": np.full(n, float(fam == "v4n")),
                "q_over_k": np.log10(q) - m_inj_h}
        return np.column_stack([cols[k] for k in self._surv["feats"]])

    def survival_prob(self, meta, fam):
        S_ = self._survival_load()
        if not S_: return None
        X = (self.survival_features(meta, fam) - S_["mu"])/S_["sd"]
        if S_["kind"] == "logreg":
            p_ab = 1/(1 + np.exp(-(np.column_stack([np.ones(len(X)), X]) @ S_["w"])))
        else:
            with torch.no_grad(): p_ab = torch.sigmoid(S_["net"](torch.tensor(X, dtype=torch.float32))[:, 0]).numpy()
        return 1.0 - p_ab

    @torch.no_grad()
    def guard_metric(self, net, x, z, drop=(), chunk=64):
        """z (K,4096) → 每个实现在 11 个存储时刻的注入井块绝对压力最大值 [MPa](K,):pp_ic(depth) + 静水修正到 y=−250 + Δp_block(t)。"""
        meta = self.decode_meta(x); p0 = PPIC_PER_KM*meta["depth"] - DP_DY*Y_INJ
        X = self.inputs(x, z.shape[0], z=z)
        if drop: X[:, list(drop)] = 0.0
        w = self.inj_mask/self.inj_mask.sum(); best = torch.full((z.shape[0],), -1e9, device=x.device)
        for ti in range(len(self.tau)):
            X[:, 11] = float(self.tau[ti])
            for i0 in range(0, X.shape[0], chunk):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(x.device.type == "cuda")):
                    mu, _, _ = net.mean_modes(X[i0:i0+chunk])
                dp = (mu[:, 1].float()*float(self.fsd[1]) + float(self.fm[1]))                # (b,H,W) MPa
                best[i0:i0+chunk] = torch.maximum(best[i0:i0+chunk], (dp*w).sum((-2, -1)))
        return best + p0

    @torch.no_grad()
    def draw_z_guarded(self, net, x, K, drop=(), meta=None):
        """代理筛查 + 护栏模拟的拒绝抽样(缓存同 draw_z,键加 'g')。"""
        meta = self.decode_meta(x, meta); key = ("g",) + self._cache_key(x, meta)
        if not hasattr(self, "_zcache"): self._zcache = {}
        hit = self._zcache.get(key)
        if hit is not None and hit[0].shape[0] >= K:
            self.last_screen = dict(hit[1], cached=True); return hit[0][:K].to(x.device)
        acc = []; n_acc = 0; n_try = 0; worst = []
        for r in range(self.max_rounds):
            need = K - n_acc; n_draw = int(np.ceil(need*(1.5 if r == 0 else max(1.3*n_try/max(n_acc, 1), 1.5))))
            z = self.draw_z(x, n_draw, meta) if not self.screen else self._fresh_screened(x, n_draw, meta)
            g = self.guard_metric(net, x, z, drop); thr = self.guard_mpa + GUARD_DEPTH_SLOPE*(meta["depth"] - D_REF); ok = g < thr
            acc.append(z[ok]); n_acc += int(ok.sum()); n_try += z.shape[0]; worst.append((z[~ok], g[~ok]))
            if n_acc >= K: break
        zs = torch.cat(acc, 0)[:K] if acc else torch.zeros(0, NPERT*NPERT, device=x.device)
        if zs.shape[0] < K:                                            # 极端热场景:护栏几乎全拒 ⇒ 按压力最低补齐(记录 guard_short)
            zr = torch.cat([a for a, _ in worst], 0); gr = torch.cat([b for _, b in worst], 0)
            zs = torch.cat([zs, zr[torch.argsort(gr)[:K - zs.shape[0]]]], 0)
        prev = dict(self.last_screen or {})
        self.last_screen = dict(n_try=n_try, n_acc=n_acc, accept=n_acc/max(n_try, 1), guard_mpa=float(thr), family=self.last_family, proxy=prev)
        if len(self._zcache) >= 1024: self._zcache.pop(next(iter(self._zcache)))
        self._zcache[key] = (zs.detach().clone(), dict(self.last_screen))
        return zs

    def _fresh_screened(self, x, n, meta):
        """代理筛查后的新鲜 z(不走缓存,供护栏拒绝抽样逐轮调用)。"""
        acc, rej = [], []; n_acc = 0; n_try = 0
        for r in range(self.max_rounds):
            need = n - n_acc; n_draw = int(np.ceil(need*(1.6 if r == 0 else max(1.2*n_try/max(n_acc, 1), 1.5))))
            z = torch.randn(n_draw, NPERT*NPERT, device=x.device); ok, pps, ceil = self.admissible(x, z, meta)
            n_try += n_draw; acc.append(z[ok]); n_acc += int(ok.sum()); rej.append((z[~ok], pps[~ok.cpu().numpy()]))
            if n_acc >= n: break
        zs = torch.cat(acc, 0)[:n]
        if zs.shape[0] < n:
            zr = torch.cat([a for a, _ in rej], 0); pr = np.concatenate([b for _, b in rej]); zs = torch.cat([zs, zr[np.argsort(pr)[:n - zs.shape[0]]]], 0)
        self.last_screen = dict(n_try=n_try, n_acc=n_acc, accept=n_acc/max(n_try, 1), ceil=float(ceil))
        return zs

    def draw(self, net, x, K, drop=(), z=None, meta=None):
        if z is None and self.guard_mpa > 0 and not self.paired:
            z = self.draw_z_guarded(net, x, K, drop=drop, meta=meta)
        X = self.inputs(x, K, z=z, meta=meta)
        if drop: X[:, list(drop)] = 0.0
        mu, _, _ = net.mean_modes(X)
        return mu[None]                                                                      # (1,K,2,H,W)
