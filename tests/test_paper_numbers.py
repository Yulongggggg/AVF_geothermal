"""Current manuscript tables and abstract claims recompute from bundled Source Data."""

from collections import Counter
import hashlib
from pathlib import Path

from geouno import paper


def test_all_printed_values_recompute():
    report = paper.check()
    assert Counter(row[0] for row in report) == {
        "Table 1": 7 * 6,
        "Table 2 (24 reservoirs)": 5 * 5,
        "Supplementary Table 1": 7 * 6,
        "Supplementary Table 2": 8 * 4,
        "Supplementary Table 19": 14 * 4,
        "Abstract": 3,
    }
    common = {"DeepONet", "FNO", "U-FNO", "U-Net", "ResNet", "Attn U-Net", "GeoUNO"}
    assert {row for table, row, *_ in report if table == "Table 1"} == common
    assert {row for table, row, *_ in report if table == "Supplementary Table 1"} == (
        common - {"Attn U-Net"} | {"PCNO"}
    )
    assert {row for table, row, *_ in report if table == "Supplementary Table 19"} == {
        model + " / " + metric for model in common for metric in ("RMSE", "relative L2")
    }
    assert len(report) == 200
    assert all(ok for *_, ok in report), [r for r in report if not r[-1]]


def test_source_data_hashes():
    root = paper.SOURCE
    recorded = []
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split(None, 1)
        recorded.append(rel.strip())
        assert hashlib.sha256(Path(root, rel.strip()).read_bytes()).hexdigest() == digest, rel
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS" and "__pycache__" not in path.parts
    }
    assert len(recorded) == len(set(recorded))
    assert set(recorded) == actual
