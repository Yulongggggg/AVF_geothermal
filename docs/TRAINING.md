# Training GeoUNO

Training needs the packed FE corpus (see [DATA.md](DATA.md#fe-corpus-zenodo-on-publication), released on Zenodo
with the paper) and GPUs. The paper model was trained on four GPUs; the transfer stage took 132 epochs.

## Lineage

1. **Inherited grid-output predictor** `cf278e64...` (`geouno_parent_cf278e64.pt`, asset group `training`):
   a `GeothermalJointOperator(regional=True, gates=False)` trained for 253 epochs through earlier stages of the
   project. Its training history is archived in the
   [OL-geothermal-final-scheme](https://github.com/Yulongggggg/OL-geothermal-final-scheme) repository
   (`geouno/` directory); it is released here as the starting point of the final stage, not retrained.
2. **POD basis** (this repository): 1,024 modes per field from training snapshots only.
3. **Transfer training** (this repository): 132 epochs from the parent, giving checkpoint `ed043489...`.

## 1. POD basis

```bash
python -m geouno.training.pod --pack /path/to/pack --out runs/basis_r1024 --rank 1024
```

For every one of the 6,582 training scenarios with actual members, two member/time snapshots are drawn
(`default_rng(20260925)`; times `o` and `o + 5` mod 11 with a scenario-hashed offset `o`). Each channel is scaled
(temperature by `fields_std[0]`, pressure by `p_res_std_mpa`), spatially centred, and decomposed with a randomized
SVD (`torch.pca_lowrank`, rank + 32 oversampling, 4 power iterations, seed `20260925 + channel`). The script also
reports validation projection errors at ranks 128-1,024 (no validation data enter the fit).

## 2. Transfer training

```bash
torchrun --nproc_per_node 4 -m geouno.training.train \
    --pack /path/to/pack --parent artifacts/model/geouno_parent_cf278e64.pt \
    --bank runs/basis_r1024/bank.pt --out runs/transfer --pod-rank 1024
```

| Epochs | Phase | Learning rates (new head / shared / T decoder / p decoder / level head) |
|---|---|---|
| 0-10 | head warm-up | 1e-4 ramp / 0 / 0 / 0 / 0 |
| 11-65 | joint transfer | 3e-5 / 1e-6 / 1e-5 / 3e-6 / 1e-5 |
| 66-131 | fine tuning from the best transfer checkpoint, optimizer reset | 1e-5 / 3e-7 / 3e-6 / 1e-6 / 3e-6, cosine decay towards 10 % |

AdamW (weight decay 1e-4), gradient-norm clipping at 5, BF16 autocast, a global batch of 16 complete scenarios
split across ranks without duplication, loss `geouno.training.loss.objective(..., weight=0.2, kind='rmse')`.
Each epoch visits all 72,939 actual training members; each scenario cycles through all eleven stored times
every eleven epochs. Validation on all 542 validation scenarios at time indices 4, 7 and 10 (every eleventh epoch,
plus the first and last) selects `best.pt` by the mean ratio of T/p RMSE to the parent; `full_validation_best.json`
evaluates all eleven times. For the paper model the selected checkpoint is the final epoch (131).

`--max-hours` checkpoints and exits for scheduler time limits; rerunning the same command resumes from `last.pt`.
`protocol.json` records input hashes, package source hashes and numerical runtime settings. A changed
model implementation or runtime requires a new output directory rather than silently resuming the old run.

A three-epoch smoke run on a synthetic pack exercises the full loop on a CPU:

```bash
pytest -m slow tests/test_training.py
```

## 3. Evaluating a trained model

Load your checkpoint with `geouno.model.load_geouno(path, expected_sha256=None)`. Table 1 of the paper averages
paired RMSE over all 618 test scenarios (8,621 members per time) and eleven times. The archived per-scenario and
per-time scores of the released model (618 x 11 x 2 rows, exported from the evaluation archive, not recomputed
here) are in `geouno/resources/source_data/forward_scenario_time_scores.csv`. These describe the transferred
application model. Table 1 now uses three independently initialized, refined POD models (198 executed epochs
per seed); their complete FP32 scores and frozen protocol are in
`geouno/resources/source_data/table1_three_seed_20261007/`. The historical baseline evaluation summaries
are in `geouno/resources/source_data/table1_nearwell/`. The transfer-training command above does not rerun
the separate independent-initialization/refinement experiment; its training scripts and records are in
the manuscript repository, `scripts/table1_three_seed_20261007/` and the associated Source Data.

## Reproducibility notes

* GPU training is not bitwise reproducible across hardware or library versions; expect statistically similar,
  not identical, checkpoints. The released checkpoint is the transferred model used in the application experiments.
* The training sources were snapshotted on 2 October 2026 for the manuscript audit; the run protocol that pinned
  their hashes on the training cluster is not part of the public archive ([PROVENANCE.md](PROVENANCE.md)).
* The baselines of Table 1 (DeepONet, FNO, U-FNO, U-Net, ResNet, PCNO) used separate training scripts that are
  archived in the manuscript repository's audit snapshot; their evaluation summaries are in
  `geouno/resources/source_data/table1_nearwell` and `geouno/resources/source_data/uniform_baselines_20261004`.
