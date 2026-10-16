# Reproducing the final-model paper

Install with `pip install -e ".[all]"`. Commands below distinguish saved-result checks from experiments.
The release update itself used saved results and did not rerun training or inference.

## Saved-result checks

```bash
geouno paper-numbers
geouno download
geouno verify
geouno prepare-cases --check --out runs/cases
```

`paper-numbers` independently aggregates saved per-scenario/time, per-reservoir and per-seed values and
checks **170 printed entries**, with zero mismatches. It covers Table 1, both Table 2 cohorts, all six
uniform-protocol models, the CO₂ table and three abstract percentages. It validates the three final test seeds,
618 × 11 scenario/time records per seed and 8,621 realizations per time. The common 99-epoch comparison includes all 18 completed training runs.
Baseline RMSE uses its original saved evaluation; pooled relative L2 uses the saved 13 October FP32 evaluation.

`verify` checks all selected artifact hashes and strictly loads the final checkpoint, checking 3,520,755
parameters. It does not predict fields. `prepare-cases` regenerates observation arrays from saved FE fields,
new optimized layouts and coordinate-keyed noise; it does not run the operator.

## Inference and monitoring experiments

There are 64 monitoring configurations: 60 over 24 reservoirs and four noise variants over six reservoirs,
giving 1,464 monitoring tasks. The 24 MAP tasks bring the total to 1,488. The score package also contains
48 prior/static-reference tasks and six NUTS forecasts, for 1,542 records.

```bash
geouno run --config multiwell_opt17 --case test_01 --method esmda --threads 4
geouno run --config multiwell_opt17 --case test_01 --method map --threads 4
python scripts/run_all_tasks.py --out runs/paper --threads 4 --shard 0/8
python scripts/run_all_tasks.py --out runs/paper --summary
```

| Method | Seed | Settings |
|---|---|---|
| ES-MDA | observation seed + 19000 | 256 members, eight updates, alpha = 8, no localization |
| MAP | observation seed + 29000 | origin + two normal starts, L-BFGS-B, 1,000 iterations/start |
| Prior / static reference | 81071 | 256 members |
| NUTS | observation seed + 19000 + chain | four chains, 512 warm-up + 1,024 draws, target acceptance 0.9, depth 7 |

Observations span times 0–6 (4.02 years), with 0.5 K / 0.25 MPa dynamic noise and 0.1-dec static noise.
Validation-fitted model-error covariance is included. Forecasts target ten years. Measured MAP and ES-MDA
inference times average 6.7 and 10.6 minutes on four CPU threads. NUTS uses H100/H200 GPUs with two concurrent
chains per GPU; four-chain summed time averages 19.6 chain-hours per reservoir.

![Monitoring results](images/monitoring.png)

## NUTS and prior

```bash
geouno download --group nuts
geouno nuts-verify
geouno nuts-chain --case test_01 --chain 0 --device cuda:0 --out runs/nuts
geouno nuts-merge --case test_01 --dir runs/nuts
geouno download --group prior
geouno refit-prior --out runs/prior
```

`nuts-verify` recomputes diagnostics from saved draws and forecasts the balanced 256-member subset;
this command executes the model. `nuts-chain` starts a new inference experiment. The saved final runs have
maximum rank R-hat 1.01131, minimum bulk ESS 234.115 and zero divergences. Diagnostic JSON is included in each
NUTS artifact and in Source Data.

The inverse-prior fit inputs remain the original training-only inputs, with marginal SD 0.976 dec.
The main FE corpus is required for retraining; see [TRAINING.md](TRAINING.md) and [DATA.md](DATA.md).

## Release validation

Current checks are recorded in [PROVENANCE.md](PROVENANCE.md). Earlier reports in
[REVIEW_20261007.md](REVIEW_20261007.md) describe the previous model release.

## CO₂ gallery

`python scripts/render_co2_gallery.py --out figures/co2` renders the 72 field panels from packaged saved arrays. It does not execute a model. Selection indices and pressure conversion are in `co2_gallery_manifest.json`.
