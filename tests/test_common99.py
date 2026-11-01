"""All comparison models must use three complete runs and one shared test cohort."""

import csv
import gzip
import hashlib
import json
import shutil

import pytest

from geouno import paper


@pytest.mark.parametrize("damage", ["missing_seed", "duplicate_record", "truth_norm"])
def test_common_comparison_rejects_inconsistent_records(tmp_path, monkeypatch, damage):
    folder = tmp_path / "common99"
    shutil.copytree(paper.SOURCE / "common99", folder)
    metadata = json.loads((folder / "sources.json").read_text())
    if damage == "missing_seed":
        metadata["sources"].pop()
    else:
        path = folder / "scenario_time_scores.csv.gz"
        with gzip.open(path, "rt") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        if damage == "duplicate_record":
            rows.append(rows[0].copy())
        else:
            rows[0]["T_truth_rms"] = str(float(rows[0]["T_truth_rms"]) + 1)
        with gzip.open(path, "wt") as handle:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            writer.writerows(rows)
        metadata["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (folder / "sources.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(paper, "SOURCE", tmp_path)
    with pytest.raises(ValueError):
        paper.common_training_runs()
