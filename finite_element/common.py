# SPDX-License-Identifier: LGPL-2.1-only
"""Shared source verification for the finite-element example."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_files():
    manifest = json.loads((HERE / 'files.json').read_text())
    for relative, entry in manifest.items():
        path = HERE / relative
        if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError(f'File identity mismatch: {relative}')
    return manifest


def generator():
    sys.path.insert(0, str(HERE / 'generation'))
    return importlib.import_module('generate_dataset_v4')
