"""PNO conditioning: characterization mean, confidence and amplitude."""

import csv
import numpy as np
import torch
from geouno.training.data import ThermalPairs, stable_key
from geouno.inputs import field_on_grid


class FieldPairs(ThermalPairs):
    def __init__(self, *args, probabilistic=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.probabilistic = probabilistic
        self.scales = torch.tensor(self.stats["fields_std"], dtype=torch.float32)[:, None, None]
        self.mc = np.load(self.pack / "inputs.npy", mmap_mode="r") if probabilistic else None
        self.index = np.load(self.pack / "index.npy", mmap_mode="r") if probabilistic else None
        self.weights = np.array([1 / len(self.selected[str(s)]) for s in self.row_scen], dtype=np.float32)
        self.weights /= self.weights.mean()
        meta = list(csv.DictReader((self.pack / "meta.csv").open()))
        self.meta = {int(r["row"]): r for r in meta}

    def __getitem__(self, item):
        field, scalar, _, idx, ti = super().__getitem__(item)
        row = int(self.rows[idx])
        truth = torch.from_numpy(np.array(self.fields[row, ti], copy=True)) / self.scales
        if self.probabilistic:
            mc = field_on_grid(torch.from_numpy(np.array(self.mc[int(self.index[row])], copy=True))[None])[0]
            # Characterization mean, confidence and amplitude replace realized geology.
            field = torch.cat(
                [mc[0:1] + 13.5, field[1:], mc[1:2], torch.full_like(mc[1:2], float(self.scal[row, 2]))], 0
            )
        return field, scalar, truth, idx, ti


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
            yield int(idx), (offset + self.epoch) % len(self.dataset.times)


class FixedPairs:
    def __init__(self, dataset, times=(4, 7, 10)):
        self.dataset, self.times = dataset, tuple(times)

    def __len__(self):
        return len(self.dataset) * len(self.times)

    def __iter__(self):
        for idx in range(len(self.dataset)):
            for ti in self.times:
                yield idx, ti
