"""Three-time validation during training (BF16 on CUDA)."""

import torch
from .metrics import records, summarize


@torch.no_grad()
def evaluate(model, data, device, times=(4, 7, 10), workers=0, chunk_size=16):
    from torch.utils.data import DataLoader
    from .data import collate

    model.eval()
    sampler = [(i, ti) for i in range(len(data)) for ti in times]
    loader = DataLoader(
        data, batch_size=1, sampler=sampler, num_workers=workers, collate_fn=collate, pin_memory=device == "cuda"
    )
    rows = []
    for (i, ti), (field, scalar, truth, lengths) in zip(sampler, loader):
        field, scalar, truth = [v.to(device) for v in (field, scalar, truth)]
        outputs = []
        for start in range(0, len(field), chunk_size):
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model(field[start : start + chunk_size], scalar[start : start + chunk_size])
            assert torch.isfinite(out).all()
            outputs.append(out.float())
        row = records(torch.cat(outputs), truth, field, lengths)[0]
        rows.append(dict(scen=data.pairs.names[i], time_index=ti, **row))
    return dict(
        scenario_count=len(data),
        actual_members=len(data.pairs),
        times=list(times),
        records=rows,
        summary=summarize(rows),
    )
