"""Case-time samples from the prepared U-FNO benchmark arrays."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..baselines.protocol import read

PRESSURE_OFFSET_BAR = 4.172939172019009
PRESSURE_SCALE_BAR = 18.772821433027488


def normalize_pressure(value):
    return (value - PRESSURE_OFFSET_BAR) / PRESSURE_SCALE_BAR


class Samples(Dataset):
    def __init__(self, prepared, split, times=None, cases=None):
        prepared = Path(prepared)
        self.fields = np.load(prepared / f"{split}_fields.npy", mmap_mode="r")
        self.scalars = np.load(prepared / f"{split}_scalars.npy")
        self.targets = np.load(prepared / f"{split}_targets.npy", mmap_mode="r")
        log_time = np.log10(np.load(prepared / "time_grid.npy"))
        self.log_time = (2 * (log_time - log_time.min()) / (log_time.max() - log_time.min()) - 1).astype(np.float32)
        self.stats = read(prepared / "stats.json")
        self.scales = np.array(self.stats["fields_std"], np.float32)
        if not self.stats.get("dp_test_normalized"):
            raise ValueError("Prepared targets must use the same pressure normalization in every split")
        self.times = list(range(24)) if times is None else list(times)
        self.cases = list(range(self.fields.shape[0])) if cases is None else list(cases)
        self.index = [(case, ti) for case in self.cases for ti in self.times]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        case, ti = self.index[item]
        field = torch.from_numpy(np.array(self.fields[case], dtype=np.float32))
        scalar = torch.from_numpy(np.concatenate([self.scalars[case], [self.log_time[ti]]]).astype(np.float32))
        target = torch.from_numpy(np.array(self.targets[case, ti], dtype=np.float32))
        return field, scalar, target, case, ti
