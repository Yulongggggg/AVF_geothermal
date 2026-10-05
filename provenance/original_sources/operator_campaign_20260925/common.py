import hashlib
import json
import os
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'pack_v4r6t2'
OUT = ROOT / 'runs/operator_campaign_20260925'
BANK = OUT / 'basis_r1024/bank.pt'
PARENT = ROOT / 'runs/final_regional_20260917/model.pt'
PARENT_SHA = 'cf278e646fbe53db11ec4a03f0545b6354ef81daa9d8c2282ebaf24704ccff9b'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024**2), b''):
            h.update(part)
    return h.hexdigest()

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(value, temp); os.replace(temp, path)

def read(path):
    return json.loads(Path(path).read_text())
