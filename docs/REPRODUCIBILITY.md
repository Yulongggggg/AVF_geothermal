# Reproducing the final-model paper

Install with `pip install -e ".[all]"`. The commands below cover saved table values, forward prediction,
inverse experiments and training. Use [environment.yml](../environment.yml) for the reported inference versions.

## Saved-result checks

```bash
geouno paper-numbers
geouno download
geouno verify
geouno prepare-cases --check --out runs/cases
```

`paper-numbers` aggregates packaged scenario/time, reservoir and training-seed records and compares
with the printed tables. Main Table 1 is the common 99-epoch comparison; Supplementary Table 1 is
the refined GeoUNO comparison. Source Data also contains the complete 24-reservoir Table 2,
CO₂ results, monitoring experiments and the three abstract percentages.

`verify` checks the selected artifact hashes. For the `core` or `all` group it also strictly loads the
final checkpoint, checking 3,520,755 parameters. It does not predict fields. `prepare-cases` regenerates observation arrays from saved FE fields,
new optimized layouts and coordinate-keyed noise; it does not run the operator.

## Inference and monitoring experiments

There are 64 monitoring configurations: 60 over 24 reservoirs and four noise variants over six reservoirs,
giving 1,464 monitoring tasks. The 24 MAP tasks bring the total to 1,488. The score package also contains
48 prior/static-reference tasks and 24 NUTS forecasts, for 1,560 records.

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
chains per GPU; four-chain summed time averages 19.5 ± 4.6 chain-hours across 24 reservoirs. Completed-chain times
exclude interrupted attempts and queue waits.

![Monitoring results](images/monitoring.png)

## NUTS and prior

The NUTS extra requires ArviZ 0.x (`arviz>=0.20,<1`); its 1.x conversion API is incompatible with the
archived diagnostic implementation. Reinstall `.[nuts]` or `.[all]` after updating the checkout.

```bash
geouno download --group nuts
geouno nuts-verify --device cuda:0
geouno nuts-chain --case test_01 --chain 0 --device cuda:0 --out runs/nuts
geouno nuts-merge --case test_01 --dir runs/nuts
geouno download --group prior
geouno refit-prior --out runs/prior
```

`nuts-verify` recomputes diagnostics from saved draws and forecasts the balanced 256-member subset;
this command executes the model. `nuts-chain` starts a new inference experiment. All 24 final-model reservoirs have four completed chains. Twenty-three pass the registered criteria
(rank R-hat ≤ 1.05, bulk/tail ESS ≥ 200, zero divergences). `test_16` has R-hat 1.05191 and bulk ESS 53.580;
the cohort minimum tail ESS is 283.596 and there are no divergences. Every reservoir remains included
in the tables and comparison figure. Per-reservoir results, compact diagnostics, costs and CSV exports
are in Source Data. All 24 raw-chain bundles are downloadable (3.94 GiB in total), and `nuts-verify`
checks all 24 by default. Its result distinguishes agreement with the archived computation from the
convergence flag. `nuts-chain` accepts `test_01` through `test_24` with the same four-chain protocol.

The archived NUTS forecasts use FP32 on H100/H200 GPUs with TF32 disabled. Recomputing the 24 released
draw sets on two H100 GPUs with PyTorch 2.8.0+cu128 took 4.53 minutes and reproduced all diagnostic,
forecast-score and per-reservoir Table 2 values exactly. The command above selects this GPU path;
it evaluates saved chains rather than sampling new ones.

CPU evaluation reproduces the diagnostics exactly and gives the same rounded Table 2 entries.
Across 24 reservoirs, maximum field-RMSE differences from the archived GPU calculation were
1.81 × 10⁻⁶ K and 6.67 × 10⁻⁴ MPa; the largest difference among pressure-well summary scores was
9.26 × 10⁻³ MPa. The unchanged score tolerances (`rtol=1e-4`, `atol=1e-8`) flag some of these CPU
comparisons as failed. CPU remains the default when no `--device` is supplied.

The inverse-prior fit inputs remain the original training-only inputs, with marginal SD 0.976 dec.
The main FE corpus is required for retraining; see [TRAINING.md](TRAINING.md) and [DATA.md](DATA.md).
Portable GeoUNO and baseline training commands are in [TRAINING.md](TRAINING.md).
The DFN/MOOSE simulation-generation workflow and FE corpus are separate from the inference assets.

## Forward demo and environment

```bash
geouno predict --case test_01 --times 10 --threads 4 --out prediction.npz
```

The demo evaluates the released application model against one FE realization. Expected reservoir-field
errors are approximately 0.757 K and 0.837 MPa. The output archive contains predicted and reference fields.
The reproduction environment is Rocky Linux 9.4 (x86-64), Python 3.11.15, PyTorch 2.8.0+cpu,
NumPy 2.4.6 and SciPy 1.15.3;
[environment.yml](../environment.yml) pins the inference dependencies. CPU execution is supported.
Creating a fresh Python 3.11.15 environment and installing its dependencies and build tools took
6.16 minutes on an Intel Xeon Platinum 8568Y+ HPC login node with Rocky Linux 9.4 and a shared
network filesystem. Pip reused 40 cached wheels and downloaded two; this timing excludes GeoUNO
installation, artifact downloads and verification commands.
[`requirements-tested-cpu.txt`](../requirements-tested-cpu.txt) records all installed dependency
versions, including plotting and development tools. Install it with
`python -m pip install -r requirements-tested-cpu.txt` in a fresh Python 3.11 environment to use
this CPU configuration. CUDA training and GPU NUTS use the separate GPU configuration below.
GPU training and NUTS use PyTorch 2.8.0 with CUDA 12.8; the original runs used H100/H200 GPUs. A common 99-epoch Attention
U-Net run takes approximately three hours on an H100, followed by validation and test evaluation.
The [physical-input example](MODEL.md#predicting-from-your-own-permeability-map) completed one joint
field prediction in 3.6 seconds, including imports and loading, on four cluster CPU threads.

[PROVENANCE.md](PROVENANCE.md) identifies the scientific source implementations and verification commands.

## CO₂ benchmark

`python scripts/render_co2_gallery.py --out figures/co2` renders the 72 field panels from packaged saved arrays. It does not execute a model. Selection indices and pressure conversion are in `co2_gallery_manifest.json`.
The [CO₂ guide](CO2.md) gives the upstream data source and commands for preparation, training and prediction-based evaluation.
