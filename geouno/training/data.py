"""Scenario-grouped training data from a packed FE corpus.

Sources: ``temperature_revision/data.py`` (``stable_key``, ``ThermalPairs``),
``overnight_bench/data.py`` (``FieldPairs``) and ``direct_operator/data.py``
(``ScenarioPairs``, ``EpochScenarios``, ``collate``).

Pack layout (directory; see ``docs/DATA.md``)::

    meta.csv         one row per FE record: case_id, scen, split, row, real, ...
    stats.json       training normalization (fields_mean/std, M_mean/std, p_res_std_mpa, ...)
    real.npy         [N, 128, 128] realized log10 permeability (m^2), float32
    fields.npy       [N, 11, 2, 160, 128] Delta T [K] and Delta p [MPa], float32
    scal.npy         [N, >=9] q, T_inj, -, x_prod, y_prod, -, drawdown, depth, p0
    _ref_times.npy   [11] stored times in seconds

Only actual (non-nominal) realizations are used; each scenario groups its members.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..inputs import field_on_grid


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
            self.rows.extend(self.selected[name])
            self.row_scen.extend([name] * len(rr))
        self.rows = np.asarray(self.rows, dtype=np.int64)
        self.row_scen = np.asarray(self.row_scen)
        y, x = torch.meshgrid(torch.linspace(-500, 500, 160), torch.linspace(-400, 400, 128), indexing="ij")
        self.x, self.y = x, y
        self.inside = (y.abs() <= 400).float()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        idx, ti = item if isinstance(item, tuple) else (item, 10)
        row = int(self.rows[idx])
        s = self.scal[row]
        m = torch.from_numpy(np.array(self.real[row], copy=True))[None, None]
        m = field_on_grid(m)[0, 0]
        rp = torch.hypot(self.x - float(s[3]), self.y - float(s[4])) / 400
        ri = torch.hypot(self.x, self.y + 250) / 400
        field = torch.stack([(m + 13.5), self.inside, self.x / 400, self.y / 500, rp, ri])
        depth = float(s[7]) if len(s) >= 9 else 3.06
        p0 = float(s[8]) if len(s) >= 9 else 30.0
        pp = p0 - .00981 * float(s[4]) - float(s[6])
        # Fixed physical scales do not change when packs or splits change.
        cond = torch.tensor([float(s[0]) / 3.5, (float(s[1]) - 303) / 10,
                             (pp - 20) / 10, (np.log10(float(self.times[ti])) - 6) / 2,
                             (depth - 3.06) / .6], dtype=torch.float32)
        target = torch.from_numpy(np.array(self.fields[row, ti, 0], copy=True))[None] / self.scale
        return field, cond, target, idx, ti


class FieldPairs(ThermalPairs):
    """Both output channels, scaled by the training standard deviations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scales = torch.tensor(self.stats['fields_std'], dtype=torch.float32)[:, None, None]
        self.weights = np.array([1 / len(self.selected[str(s)]) for s in self.row_scen], dtype=np.float32)
        self.weights /= self.weights.mean()
        meta = list(csv.DictReader((self.pack / 'meta.csv').open()))
        self.meta = {int(r['row']): r for r in meta}

    def __getitem__(self, item):
        field, scalar, _, idx, ti = super().__getitem__(item)
        row = int(self.rows[idx])
        truth = torch.from_numpy(np.array(self.fields[row, ti], copy=True)) / self.scales
        return field, scalar, truth, idx, ti


class ScenarioPairs(Dataset):
    """All actual members of one scenario at one stored time; targets in physical units."""

    def __init__(self, pack, split, scenario_limit=0, member_limit=0):
        self.pairs = FieldPairs(pack, split, scenario_limit=scenario_limit, member_limit=member_limit, seed=20260913)
        lookup = {int(row): i for i, row in enumerate(self.pairs.rows)}
        self.indices = [[lookup[row] for row in self.pairs.selected[name]] for name in self.pairs.names]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        group, ti = item
        items = [self.pairs[(i, ti)] for i in self.indices[group]]
        field, scalar, truth = [torch.stack([x[k] for x in items]) for k in (0, 1, 2)]
        return field, scalar, truth * self.pairs.scales, len(items)


class EpochScenarios:
    """Seeded scenario permutation; each scenario visits every stored time once per 11 epochs."""

    def __init__(self, data, seed):
        self.data, self.seed, self.epoch = data, seed, 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in np.random.default_rng(self.seed + self.epoch).permutation(len(self.data)):
            name = self.data.pairs.names[i]
            offset = int.from_bytes(stable_key(name, 20260913)[:4], 'little')
            yield int(i), (offset + self.epoch) % 11


def collate(batch):
    return (*[torch.cat([row[k] for row in batch]) for k in (0, 1, 2)], [row[3] for row in batch])
