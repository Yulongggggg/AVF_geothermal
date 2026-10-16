# Training the final model

The final `resid64_p1` model starts from random neural initialization and a fixed POD bank fitted to training
outputs. Training uses 72,939 non-nominal realizations in 6,582 scenarios, with eight complete scenarios per step.
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
`final.pt`, `final_eval/val.json` and `final_eval/test.json`. Completed stages and partial epochs resume from
saved state. `--main-from` can adopt a completed main-stage run after validating its protocol and seed.
The packaged three-time reference validation supplies the main-stage score normalization.

The common-protocol appendix comparison uses the 99-epoch stage, eight scenarios per batch and seeds
20260915–20260917. It has no refinement stages. Pending rows are identified in the packaged numerical data.

The source used for the final paper runs is preserved in
[`provenance/original_sources/final_geouno_20261015`](../provenance/original_sources/final_geouno_20261015).
The earlier transfer trainer is retained as `legacy_train.py` for historical checkpoints.
