# SPDX-License-Identifier: GPL-3.0-only
"""The 99-epoch PCNO spatial-backbone recipe."""

from pathlib import Path

from geouno.baselines.protocol import (
    atomic_checkpoint as atomic_checkpoint,
    read as read,
    score as score,
    sha,
    write as write,
)
from comparisons import protocol as common

PACK = OUT = None
REFERENCE = Path(__file__).with_name("reference_validation.json")


def configure(pack, out, reference=None):
    global PACK, OUT, REFERENCE
    PACK, OUT = Path(pack), Path(out)
    REFERENCE = Path(reference) if reference else Path(__file__).with_name("reference_validation.json")


def recipe(model, seed):
    if model != "pcno_backbone":
        raise ValueError(model)
    return dict(
        variant=model, seed=seed, initialization="scratch", epochs=99, batch_scenarios=1, time_offset=0, lr=0.0003
    )


def freeze(model, seed, smoke=False, device="cuda"):
    value = dict(
        recipe=recipe(model, seed),
        smoke=smoke,
        reference_sha256=sha(REFERENCE),
        inputs=common.identity(PACK, __file__, device),
        objective="paired RMSE + 0.2 MMD squared",
        validation_times=[4, 7, 10],
        inference_chunk=4,
        training_precision="BF16 autocast on CUDA",
        selection="Mean reference-normalized T/P RMSE at time 10 and all evaluated times",
    )
    return common.freeze(OUT / ("smoke" if smoke else "formal") / f"{model}_s{seed}", value)
