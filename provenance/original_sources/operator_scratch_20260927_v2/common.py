import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path('/mnt/home/yliu5/projects/ol_geo_train')
PACK = DATA_ROOT / 'pack_v4r6t2'
OUT = DATA_ROOT / 'runs/operator_scratch_20260927_v2'
BANK = DATA_ROOT / 'runs/operator_campaign_20260925/basis_r1024/bank.pt'
BANK_SHA = '296b9975a490526b30840e45c01abe5c256d2af6e0fa7f60fa14fa6206d7a997'
REFERENCE = DATA_ROOT / 'runs/operator_campaign_20260925/formal/r1024_s20260925/reference_validation.json'
LM = DATA_ROOT / 'runs/learned_modes_20260925'
LM_PROTOCOL_SHA = 'd366c5be81adf6f075cf9561947c3da9f4e4a8487695c5e7936a8a509d492d2e'
SEED = 20260925


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def verify_protocol():
    p = read(OUT / 'protocol.json')
    assert sha(PACK / 'meta.csv') == p['meta_sha256']
    assert sha(PACK / 'stats.json') == p['stats_sha256']
    assert sha(BANK) == p['basis_sha256'] == BANK_SHA
    assert sha(REFERENCE) == p['reference_validation_sha256']
    for name, digest in p['sources_sha256'].items():
        assert sha(ROOT / name) == digest, name
    return p
