import json
from functools import lru_cache
from pathlib import Path
from history_matching_dense.networks import NETWORKS as CALIBRATION_NETWORKS, network_points as dense_points,point_noise
from history_matching_networks.networks import network_points as original_points

OUT=Path(__file__).resolve().parents[1]/'runs/history_sparse_20260912'
NETWORKS={**{f'opt{n}':dict(n=n,label=f'{n} validation-designed monitoring locations') for n in (5,9,17)},
          'corridor9':dict(n=9,label='9-point geometric corridor control')}


@lru_cache(maxsize=512)
def network_points(wx,wy,network):
    if network=='bank':return dense_points(wx,wy,'bank')
    if network=='corridor9':return original_points(wx,wy,network)
    selection=json.loads((OUT/'selection.json').read_text())
    points=dense_points(wx,wy,'hybrid385')
    return [points[i] for i in selection['designs'][network[3:]]['indices']]
