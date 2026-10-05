import numpy as np
import torch
from torch.utils.data import Dataset
from temperature_revision.data import stable_key
from overnight_bench.data import FieldPairs


class ScenarioPairs(Dataset):
    def __init__(self, pack, split, scenario_limit=0, member_limit=0):
        self.pairs = FieldPairs(pack,split,scenario_limit=scenario_limit,member_limit=member_limit,seed=20260913)
        lookup = {int(row):i for i,row in enumerate(self.pairs.rows)}
        self.indices = [[lookup[row] for row in self.pairs.selected[name]] for name in self.pairs.names]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        group, ti = item
        items = [self.pairs[(i,ti)] for i in self.indices[group]]
        field,scalar,truth = [torch.stack([x[k] for x in items]) for k in (0,1,2)]
        return field,scalar,truth*self.pairs.scales,len(items)


class EpochScenarios:
    def __init__(self, data, seed):
        self.data,self.seed,self.epoch = data,seed,0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in np.random.default_rng(self.seed+self.epoch).permutation(len(self.data)):
            name = self.data.pairs.names[i]
            offset = int.from_bytes(stable_key(name,20260913)[:4],'little')
            yield int(i),(offset+self.epoch)%11


def collate(batch):
    return (*[torch.cat([row[k] for row in batch]) for k in (0,1,2)], [row[3] for row in batch])
