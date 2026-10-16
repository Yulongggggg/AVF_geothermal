"""Manuscript tables and abstract numbers recompute from the bundled Source Data."""
import hashlib
from pathlib import Path

from geouno import paper


def test_all_printed_values_recompute():
    report = paper.check()
    assert len(report) == 170
    assert all(ok for *_, ok in report), [r for r in report if not r[-1]]


def test_source_data_hashes():
    root = paper.SOURCE
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        digest, rel = line.split(None, 1)
        assert hashlib.sha256(Path(root, rel.strip()).read_bytes()).hexdigest() == digest, rel
