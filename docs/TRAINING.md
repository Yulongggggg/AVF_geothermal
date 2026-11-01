# Geothermal training and evaluation

This checkout provides training and evaluation for the final GeoUNO model and the six deterministic
geothermal baselines. Supply the FE corpus in the format described in [DATA.md](DATA.md).

The final `resid64_p1` model starts from random neural initialization and a fixed POD bank fitted to training
outputs. Training uses 72,939 permeability realizations in 6,582 scenarios, with eight complete scenarios per step.
Each eleven-epoch cycle visits the eleven stored times. Seeds: 20260915, 20260925, 20260926.

| Stage | Epochs | Parameters | Learning rate |
|---|---:|---|---|
| Main | 99 | All | AdamW 3e-4, cosine to 0.05× |
| Temperature | 44 | Temperature decoder, output layer and coordinate network | 1e-5 peak |
| Joint | 22 | All | 1e-5 peak |

The objective is paired scenario RMSE plus 0.2 MMD², with the paper's temperature, pressure-mean,
pressure-shape and cap/bed terms. Weight decay is 1e-4 and gradient clipping is 5. Refinement uses three-epoch
warm-up and cosine decay to 0.1× peak. Validation selection uses all 542 scenarios at times 4, 7 and 10;
final validation and test evaluation use all eleven times in FP32. Temperature-stage selection preserves
pressure, and joint selection checks both fields and pressure tails; the exact conditions are in
[`train.admissible`](../geouno/training/train.py).

## Running with the FE corpus

The corpus layout is documented in [DATA.md](DATA.md). Download the training-only POD bank with
`geouno download --group training`, or fit a new bank using `python -m geouno.training.pod --help`.
The released bank SHA-256 is `296b9975a490526b30840e45c01abe5c256d2af6e0fa7f60fa14fa6206d7a997`.

```bash
python -m geouno.training.train \
  --pack /path/to/fe-pack \
  --bank artifacts/training/pod_bank_296b9975.pt \
  --out runs/final --seed 20260915
```

The command writes stage histories and checkpoints under `formal/resid64_p1_s20260915/`, then
`final.pt`, `final_eval/val.json` and `final_eval/test.json`. Completed stages are reused; interrupted stages
resume from the last completed epoch. `--main-from` can adopt a completed main-stage run after validating its protocol and seed.
The packaged three-time reference validation supplies the main-stage score normalization.

## Common 99-epoch comparison

Main Table 1 uses the 99-epoch stage, eight scenarios per batch and seeds 20260915–20260917.
The six baselines use the same input encoding, objective, optimizer and validation reference as GeoUNO.

```bash
python -m geouno.common99 \
  --pack /path/to/fe-pack --bank artifacts/training/pod_bank_296b9975.pt \
  --out runs/common99 --model geouno --seed 20260915 --device cuda

python -m geouno.baselines.train \
  --pack /path/to/fe-pack --out runs/common99 \
  --model attention_unet --seed 20260915 --device cuda
```

Model choices are `deeponet`, `fno`, `ufno`, `unet`, `resnet` and `attention_unet`.
Each run writes its selected `best.pt`, epoch checkpoint `last.pt`, training history and final validation/test
records. The baseline training command resumes from `last.pt`; `--smoke --device cpu` runs two small epochs.
The common trainer writes its original BF16 validation/test records. Use the table evaluator below
for the current paper comparison: it evaluates the six baselines in BF16 on CUDA and GeoUNO in FP32
with TF32 disabled. GeoUNO uses batches of 16 realizations; the other models use 32.

```bash
python -m geouno.baselines.table_metrics \
  --pack /path/to/fe-pack \
  --checkpoint runs/common99/formal/attention_unet_s20260915/best.pt \
  --out runs/common99/attention_s20260915_metrics.json --device cuda
```

The table evaluator records field and near-well errors for every scenario and time. Relative $L_2$
pools squared errors and reference norms over times within each scenario before averaging scenarios.
This differs from averaging the eleven relative errors.
For GeoUNO, pass `runs/common99/formal/resid64_p1_s20260915/best.pt` to the same evaluator.
Repeat each architecture with seeds 20260915, 20260916 and 20260917; the tables report the mean and
sample standard deviation of these three independent runs. The common training command ends after
99 epochs. The application-model command above continues through temperature and joint refinement.

The source used for the final paper runs is preserved in
[`provenance/original_sources/final_geouno_20261015`](../provenance/original_sources/final_geouno_20261015).
The earlier transfer trainer is retained as `legacy_train.py` for historical checkpoints.
