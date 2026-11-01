"""Final evaluation with caches tied to the checkpoint, data and numerical protocol."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from ..artifacts import resource
from .data import FieldPairs
from .full_metrics import metric_record, summarize, load_scope
from ..model.final import load as load_ablation
from . import final_protocol as protocol
from .final_protocol import read, write, sha


def load(path):
    return load_ablation(path)


def evaluation_identity(path, data, split, smoke, fp32, device):
    """Inputs that must agree before completed or partial evaluation can be reused."""
    pack = data.pack.resolve()
    package = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), package / 'training/data.py', package / 'training/full_metrics.py',
               package / 'inputs.py', *sorted((package / 'model').glob('*.py'))]
    selection = json.dumps(data.selected, sort_keys=True, separators=(',', ':')).encode()
    payloads = {}
    for name in ('real.npy', 'fields.npy', 'scal.npy'):
        stat = (pack / name).stat()
        # These arrays may occupy hundreds of GB. File identity supplements the
        # content hashes of metadata/statistics without rereading the entire corpus.
        payloads[name] = dict(bytes=stat.st_size, modified_ns=stat.st_mtime_ns)
    return dict(
        format_version=1, checkpoint_sha256=sha(path), split=split, smoke=bool(smoke), fp32=bool(fp32),
        device_type=torch.device(device).type, torch=str(torch.__version__), numpy=str(np.__version__),
        pack=str(pack), pack_files=payloads,
        data_sha256={name: sha(pack / name) for name in ('meta.csv', 'stats.json', '_ref_times.npy')},
        selection_sha256=hashlib.sha256(selection).hexdigest(), scenarios=len(data.names), members=len(data),
        pressure_scope_sha256=sha(resource('training_final_20261015/pressure_scope.json.gz')),
        sources_sha256={p.relative_to(package).as_posix(): sha(p) for p in sources},
    )


def validate_cache(saved, identity):
    if saved.get('evaluation_identity') != identity:
        raise ValueError('Evaluation cache has different or missing input/precision metadata; '
                         'use a new output directory')


def evaluate(path, split, folder, device="cuda", smoke=False, fp32=True):
    target = folder / f"{split}.json"
    partial = folder / f"{split}.partial.json"
    digest = sha(path)
    ds = FieldPairs(
        protocol.PACK, split, scenario_limit=2 if smoke else 0, member_limit=2 if smoke else 0, seed=20260913
    )
    identity = evaluation_identity(path, ds, split, smoke, fp32, device)
    if target.exists():
        saved = read(target)
        validate_cache(saved, identity)
        return saved
    rows = []
    if partial.exists():
        previous = read(partial)
        validate_cache(previous, identity)
        rows = previous["records"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False  # FP32 test evaluation as in the paper
    net = load(path).to(device).eval()
    scope = load_scope()
    done = {(r["scen"], r["time_index"]) for r in rows}
    lookup = {int(row): i for i, row in enumerate(ds.rows)}
    with torch.no_grad():
        for si, name in enumerate(ds.names):
            indices = [lookup[row] for row in ds.selected[name]]
            for ti in (10,) if smoke else range(11):
                if (name, ti) in done:
                    continue
                pred, truth = [], []
                for start in range(0, len(indices), 16):
                    items = [ds[(i, ti)] for i in indices[start : start + 16]]
                    f, c, y = [torch.stack([item[k] for item in items]).to(device) for k in (0, 1, 2)]
                    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda" and not fp32):
                        p = net(f, c)
                    assert torch.isfinite(p).all()
                    pred.append(p.float())
                    truth.append(y * ds.scales.to(device))
                rows.append(metric_record(torch.cat(pred), torch.cat(truth), ds, indices, ti, False, scope))
            if (si + 1) % 24 == 0 or si + 1 == len(ds.names):
                write(partial, dict(checkpoint_sha256=digest, evaluation_identity=identity, records=rows))
    expected_times = {10} if smoke else set(range(11))
    assert len(rows) == len(ds.names) * len(expected_times)
    assert {(r["scen"], r["time_index"]) for r in rows} == {(s, t) for s in ds.names for t in expected_times}
    assert all(r["rows"] == ds.selected[r["scen"]] for r in rows)
    result = dict(
        status="complete",
        split=split,
        checkpoint=str(path),
        checkpoint_sha256=digest,
        evaluation_identity=identity,
        smoke=smoke,
        scenario_count=len(ds.names),
        actual_members=len(ds),
        records=rows,
        summary=summarize(rows),
    )
    write(target, result)
    return result
