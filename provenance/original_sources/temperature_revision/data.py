"""Paired physical inputs, preserving the corpus row and time identities."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from det_sampler import field_on_grid


def stable_key(text, seed=20260909):
    return hashlib.sha256(f"{seed}:{text}".encode()).digest()


class ThermalPairs(Dataset):
    """One actual FE realization and stored time, without padded members.

    Spatial inputs: m, reservoir mask, x, y, producer/injector distances.
    Scalars: flow, injection temperature, producer BHP, log time, depth.
    C and S_max define the sampling law; they are not physical PDE inputs
    once a permeability realization has been supplied.
    """
    def __init__(self, pack, split, scenario_limit=0, member_limit=0, seed=20260909):
        self.pack = Path(pack)
        self.stats = json.loads((self.pack / "stats.json").read_text())
        if self.stats.get("limited"):
            raise ValueError("A smoke pack cannot supply training normalization")
        self.scale = float(self.stats["fields_std"][0])
        self.real = np.load(self.pack / "real.npy", mmap_mode="r")
        self.fields = np.load(self.pack / "fields.npy", mmap_mode="r")
        self.scal = np.load(self.pack / "scal.npy", mmap_mode="r")
        self.times = np.load(self.pack / "_ref_times.npy")
        with (self.pack / "meta.csv").open() as handle:
            meta = [r for r in csv.DictReader(handle) if r["split"] == split and r["real"] != "nom"]
        by_scen = {}
        for row in meta:
            by_scen.setdefault(row["scen"], []).append(row)
        self.names = sorted(by_scen, key=lambda s: stable_key(s, seed))
        if scenario_limit:
            self.names = self.names[:scenario_limit]
        self.rows, self.row_scen = [], []
        self.selected = {}
        for name in self.names:
            rr = sorted(by_scen[name], key=lambda r: r["real"])
            if member_limit:
                rr = rr[:member_limit]
            self.selected[name] = [int(r["row"]) for r in rr]
            self.rows.extend(self.selected[name]); self.row_scen.extend([name]*len(rr))
        self.rows = np.asarray(self.rows, dtype=np.int64)
        self.row_scen = np.asarray(self.row_scen)
        y, x = torch.meshgrid(torch.linspace(-500, 500, 160), torch.linspace(-400, 400, 128), indexing="ij")
        self.x, self.y = x, y
        self.inside = (y.abs() <= 400).float()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        idx, ti = item if isinstance(item, tuple) else (item, 10)
        row = int(self.rows[idx]); s = self.scal[row]
        m = torch.from_numpy(np.array(self.real[row], copy=True))[None, None]
        m = field_on_grid(m)[0, 0]
        rp = torch.hypot(self.x-float(s[3]), self.y-float(s[4]))/400
        ri = torch.hypot(self.x, self.y+250)/400
        field = torch.stack([(m+13.5), self.inside, self.x/400, self.y/500, rp, ri])
        depth = float(s[7]) if len(s) >= 9 else 3.06
        p0 = float(s[8]) if len(s) >= 9 else 30.0
        pp = p0 - .00981*float(s[4]) - float(s[6])
        # Fixed physical scales do not change when packs or splits change.
        cond = torch.tensor([float(s[0])/3.5, (float(s[1])-303)/10,
                             (pp-20)/10, (np.log10(float(self.times[ti]))-6)/2,
                             (depth-3.06)/.6], dtype=torch.float32)
        target = torch.from_numpy(np.array(self.fields[row, ti, 0], copy=True))[None]/self.scale
        return field, cond, target, idx, ti


class EpochPairs:
    """A reproducible time-balanced traversal; worker state is irrelevant."""
    def __init__(self, dataset, epoch, seed):
        self.dataset, self.epoch, self.seed = dataset, epoch, seed

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for idx in rng.permutation(len(self.dataset)):
            # Each row sees every snapshot once in 11 successive epochs.
            offset = int.from_bytes(stable_key(str(int(self.dataset.rows[idx])), self.seed)[:4], "little")
            yield int(idx), (offset+self.epoch) % len(self.dataset.times)


class FixedPairs:
    def __init__(self, dataset, times=(4, 7, 10)):
        self.dataset, self.times = dataset, tuple(times)

    def __len__(self):
        return len(self.dataset)*len(self.times)

    def __iter__(self):
        for idx in range(len(self.dataset)):
            for ti in self.times:
                yield idx, ti
