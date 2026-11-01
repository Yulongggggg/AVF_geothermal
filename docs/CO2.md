# CO₂ storage benchmark

The comparison in Supplementary Table 2 retrains eight two-field models on the U-FNO benchmark:
DeepONet, FNO, U-FNO, U-Net, ResNet and GeoUNO with 1,024, 2,048 or 4,096 POD functions per field.
Each model uses seeds 20261003, 20261004 and 20261005, 20 epochs, batch size 32 and the same
masked-RMSE objective. The split contains 4,500 training, 500 validation and 500 test cases, each
with 24 times on a 96 × 200 grid. Validation selection uses zero-based time indices 5, 12 and 23; testing uses all times.

## Data preparation

Obtain the data through the [upstream U-FNO README](https://github.com/gegewen/ufno/blob/8315fd7b5bd75282b7efe42ee6b8de86543d13cc/README.md),
which links the [data folder](https://drive.google.com/drive/folders/1fZQfMn_vsjKUXAfRV0q_gswtl8JEkVGo).
For each of `train`, `val` and `test`, place `sg_<split>_a.pt`, `sg_<split>_u.pt` and
`dP_<split>_u.pt` in one raw-data directory. The upstream repository commit is
`8315fd7b5bd75282b7efe42ee6b8de86543d13cc`; the README does not assign a numbered data release.
The data retain their provider's terms; see [THIRD_PARTY.md](../THIRD_PARTY.md#co₂-benchmark).

```bash
python -m geouno.co2.prepare \
  --raw /path/to/ufno-raw --out /path/to/ufno-prepared --pod --pod-large
```

Preparation creates field, scalar and target arrays, training scales and POD banks fitted to training
outputs. The 2,048- and 4,096-function models use the first 2,048 and all 4,096 functions of
`bank_r4096.pt`. The 1,024-function model uses `bank.pt`.

The original training and validation pressure tensors are already normalized. The original test
pressure tensor is in bar and is converted once using
`z_p = (Δp − 4.172939172019009) / 18.772821433027488`.
For a test tensor already expressed in these normalized units, pass `--test-pressure-units normalized`.
The unit choice and input identities are recorded with each conversion. Preparation requires the
full benchmark tensors; the training smoke option below does not download or prepare data.

## Training and evaluation

```bash
python -m geouno.co2.train \
  --prepared /path/to/ufno-prepared --out runs/co2 \
  --model geouno_r4096 --seed 20261003 --device cuda

python -m geouno.co2.evaluate \
  --prepared /path/to/ufno-prepared \
  --checkpoint runs/co2/formal/geouno_r4096_s20261003/best.pt \
  --out runs/co2/geouno_r4096_s20261003_test.json --device cuda
```

Model names are `deeponet`, `fno`, `ufno`, `unet`, `resnet`, `geouno`, `geouno_r2048` and
`geouno_r4096`. Repeat each with all three seeds to calculate the table's mean and sample standard
deviation. `attention_unet` is also accepted for checkpoint compatibility and is outside the published
CO₂ comparison. Training resumes from `last.pt` with its optimizer, learning-rate schedule and random
state. `best.pt` is selected by the sum of the two validation RMSEs divided by training scales.
`--smoke --device cpu --workers 0` runs five training batches and a reduced validation/test set.

CUDA evaluation uses BF16 autocast. Relative errors are averaged over case–time pairs within the
reservoir mask. Pressure relative error uses the normalized target; physical pressure RMSE restores
the scale and pools squared errors over masked cells, cases and times. Evaluation records both
quantities, their units, and checkpoint/statistics hashes.

The packaged seed-level results are in
[`co2_final.json`](../geouno/resources/source_data/final_model_20261015/co2_final.json) and
[`co2_baseline_per_seed.json`](../geouno/resources/source_data/final_model_20261015/co2_baseline_per_seed.json).
`geouno paper-numbers` recomputes the displayed aggregates. The field gallery uses the saved arrays
and selection indices described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md#co₂-benchmark).
