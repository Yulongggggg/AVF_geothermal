# Supplementary architecture comparisons

These programs reproduce the paper's model-specific PCNO and PNO comparisons. Run them from this source checkout after installing GeoUNO and obtaining the geothermal training pack described in [DATA.md](../docs/DATA.md). They use the same existing pack and numerical metric functions as GeoUNO. The common 99-epoch Table 1 comparison has its own [training entry points](../docs/TRAINING.md).

Set the input pack and output directories before running the commands:

```bash
PACK=/path/to/fe-pack
PCNO_OUT=runs/pcno
PNO_OUT=runs/pno
```

## PCNO spatial backbone

The [PCNO directory](pcno/) is separately licensed under **GPL-3.0-only**, including the two-dimensional adaptation. It retains Zhenqian Xue's official Code Ocean v1.0 source, commit `dc4cd4417ed7f97715ffb47496690ce60f7665a3`, and its [GPL licence](pcno/LICENSE). The upstream files in `pcno/upstream/` are reference material; the two-dimensional adaptation requires only the installed GeoUNO dependencies. The upstream four-dimensional model additionally imports `iapws` and is not used by these commands.

The adaptation has 23,383,572 parameters, temperature/pressure widths 56/64 and 14 spatial Fourier modes. It uses 99 epochs, one complete scenario per optimizer step, four-member activation-checkpoint chunks, BF16 autocast on CUDA, and paired RMSE plus 0.2 squared MMD. AdamW starts at 0.0003 with weight decay 0.0001 and a cosine multiplier ending at 0.05. Each eleven-epoch cycle visits all stored times. Checkpoint selection averages reference-normalized temperature and pressure RMSE at time index 10 and across validation indices 4, 7 and 10. The saved `reference_validation.json` is identical in all three original runs; it supplies the selection denominators and does not initialize the model. No POD bank is needed.

```bash
for seed in 20260915 20260916 20260917; do
  python -m comparisons.pcno.train --pack "$PACK" --out "$PCNO_OUT" --seed "$seed"
done
python -m comparisons.pcno.evaluate \
  --pack "$PACK" --checkpoint "$PCNO_OUT/formal/pcno_backbone_s20260915/best.pt" \
  --out "$PCNO_OUT/test_s20260915" --split test
```

Training uses one GPU. Full validation and test evaluation use BF16 autocast and four-member chunks on CUDA. The supervised RMSE/MMD objective and scalar-time two-dimensional interface are the adaptation reported in this paper; the original four-dimensional physics constraints are documented in the cited PCNO publication.

## Conditional PNO

The [PNO directory](pno/) contains the study authors' MIT-licensed adaptation of the probabilistic neural operator method of [Bülte, Scholl and Kutyniok (2025)](https://openreview.net/forum?id=gangoPXSRw). The [method repository](https://github.com/cbuelt/pfno) states MIT licensing in its README. Its source is not vendored here. The model conditions on characterization mean, confidence, amplitude and physical controls, with a 64-component latent vector and two spatial Gaussian-noise channels. It has 3,223,170 parameters. Initialization preserves the original construction order, including the replacement of complex Fourier weights with real-storage Fourier weights.

PNO also reads `inputs.npy` (7,814 characterization mean/confidence pairs, float32, shape 7,814 × 2 × 128 × 128) and `index.npy` (95,966 int64 scenario indices). Both are included in the full corpus manifest linked from [DATA.md](../docs/DATA.md); together they add 1,024,964,592 bytes to the six arrays and metadata used by the common comparison. Each index matches the corresponding `scen_row` in `meta.csv`.

The published recipe uses exactly two visible GPUs through `DataParallel`, global batch size 64, four latent draws per training target, 44 epochs, AdamW at 0.0003 with weight decay 0.0001, and a cosine schedule ending at 3% of the initial learning rate. A fixed validation subset of 64 scenarios and up to four actual members per scenario is evaluated at time indices 4, 7 and 10. Four repeatable latent draws per target determine the composite energy score used for checkpoint selection. Both training seeds are included below.

```bash
for seed in 20260913 20260914; do
  python -m comparisons.pno.train --pack "$PACK" --out "$PNO_OUT" --seed "$seed"
  python -m comparisons.pno.evaluate \
    --pack "$PACK" --checkpoint "$PNO_OUT/formal/pno_s${seed}/best.pt" \
    --out "$PNO_OUT/test_s${seed}" --split test
done
```

Full evaluation uses one GPU, BF16 autocast, a batch size of 32 and a deterministic latent seed for each scenario/time pair. It uses all actual members and all eleven times. Distribution and disjoint-member metrics describe conditional response samples; PNO samples are not paired predictions for the supplied permeability realization.

## Smoke tests and interrupted runs

Append `--device cpu --smoke` to either training command to run two epochs on two scenarios with up to two members each. Add the same flags to evaluation commands for a small validation/test subset. Formal PNO training checks that two GPUs are visible; its CPU smoke run exercises the same model and energy score with a smaller batch.

Training resumes from the last completed epoch when the same command is repeated. The optimizer, scheduler, model and random-number states are restored, and the selected checkpoint survives an interruption between selection and the optimizer-state write. Run identities include the source, configuration, normalization and metadata hashes, large-array identities, device, and NumPy/PyTorch versions. Partial evaluation results are reused only when their identities match. Use a new output directory after changing code, data or protocol. Completed experiments in the paper need not be retrained to use their reported Source Data.

The original source hashes and protocol hashes are recorded in each directory's `sources.json`.
