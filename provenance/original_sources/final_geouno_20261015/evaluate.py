"""Transcription of fair_comparison/evaluate.py with a loader for the POD operator."""
import torch
from overnight_bench.data import FieldPairs
from overnight_bench.evaluate import metric_record, summarize, load_scope
from .model import load as load_ablation
from .common import PACK, read, write, sha


def load(path):
    return load_ablation(path)


def evaluate(path, split, folder, device='cuda', smoke=False, fp32=True):
    target = folder/f'{split}.json'
    partial = folder/f'{split}.partial.json'
    digest = sha(path)
    if target.exists():
        saved = read(target)
        assert saved['checkpoint_sha256'] == digest and saved['smoke'] == smoke
        return saved
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False  # FP32 test evaluation as in the paper
    net = load(path).to(device).eval()
    ds = FieldPairs(PACK, split, scenario_limit=2 if smoke else 0,
                    member_limit=2 if smoke else 0, seed=20260913)
    scope = load_scope()
    rows = []
    if partial.exists():
        previous = read(partial)
        assert previous['checkpoint_sha256'] == digest
        rows = previous['records']
    done = {(r['scen'], r['time_index']) for r in rows}
    lookup = {int(row): i for i, row in enumerate(ds.rows)}
    with torch.no_grad():
        for si, name in enumerate(ds.names):
            indices = [lookup[row] for row in ds.selected[name]]
            for ti in ((10,) if smoke else range(11)):
                if (name, ti) in done:
                    continue
                pred, truth = [], []
                for start in range(0, len(indices), 16):
                    items = [ds[(i, ti)] for i in indices[start:start+16]]
                    f, c, y = [torch.stack([item[k] for item in items]).to(device) for k in (0, 1, 2)]
                    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda' and not fp32):
                        p = net(f, c)
                    assert torch.isfinite(p).all()
                    pred.append(p.float())
                    truth.append(y*ds.scales.to(device))
                rows.append(metric_record(torch.cat(pred), torch.cat(truth), ds, indices, ti, False, scope))
            if (si+1) % 24 == 0 or si+1 == len(ds.names):
                write(partial, dict(checkpoint_sha256=digest, records=rows))
    expected_times = {10} if smoke else set(range(11))
    assert len(rows) == len(ds.names)*len(expected_times)
    assert {(r['scen'], r['time_index']) for r in rows} == {(s, t) for s in ds.names for t in expected_times}
    assert all(r['rows'] == ds.selected[r['scen']] for r in rows)
    result = dict(status='complete', split=split, checkpoint=str(path), checkpoint_sha256=digest,
                  smoke=smoke, scenario_count=len(ds.names), actual_members=len(ds),
                  records=rows, summary=summarize(rows))
    write(target, result)
    return result
