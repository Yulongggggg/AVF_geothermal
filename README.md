# GeoUNO: joint operator learning for geothermal forecasting and inference under uncertainty

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Geothermal data: CC BY 4.0](https://img.shields.io/badge/geothermal%20data-CC%20BY%204.0-lightgrey.svg)](docs/DATA.md#licence)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Tests](https://github.com/Yulongggggg/AVF_geothermal/actions/workflows/tests.yml/badge.svg)](https://github.com/Yulongggggg/AVF_geothermal/actions/workflows/tests.yml)

GeoUNO jointly predicts transient temperature and pressure in geothermal reservoirs. Its differentiable
operator connects ensemble forecasting, permeability inference and monitoring design.

![GeoUNO overview](docs/images/overview.png)

> Yulong Liu, Jonah Botvinick-Greenhouse, Yunan Yang and Chloé Arson.
> *Joint operator learning for geothermal forecasting and inference under uncertainty.* Manuscript, 2026.

## Highlights

- **Compact operator** — 3,520,755 parameters, 1,024 fixed training-fitted POD functions and
  64 learned functions per field.
- **Accurate forecasts** — under the common 99-epoch protocol, three independent seeds achieve
  **0.368 ± 0.008 K** and **0.430 ± 0.006 MPa** paired field RMSE over 618 test scenarios and eleven times.
- **Fast** — one realization's joint trajectory takes **0.015 s** on the measured GPU setup.
- **History matching** — across 24 reservoirs, ES-MDA reduces pressure and temperature field errors by
  **97% and 48%** relative to prior forecasts.
- **Monitoring design** — guided nine-location multiwell placement reduces near-producer error by
  **37%** relative to space-filling placement.

## What is in this release

- The application checkpoint: the refined seed-20260915 model in Supplementary Table 1, SHA-256
  `f39894ff83e04f47b2a9098f0e387540e1390cf12b9d7f8726dc815dffd16dbb`.
- Updated observation layouts and calibration, 24 test reservoirs, 1,560 saved task/reference score
  records, and the paper's numerical [Source Data](geouno/resources/source_data/final_model_20261015).
- NUTS forecasts and diagnostics for all 24 reservoirs (96 completed chains): 23 pass convergence
  criteria; `test_16` remains included with its diagnostic flag. Mean NUTS permeability,
  temperature-field and pressure-field RMSEs are **0.645 dec, 6.631 K and 0.474 MPa**.
  The downloadable raw-chain bundles cover all 24 reservoirs; their scores and diagnostic summaries
  are bundled in the Source Data.

## Quick start

```bash
git clone https://github.com/Yulongggggg/AVF_geothermal.git
cd AVF_geothermal
python -m pip install -e ".[all]"
geouno paper-numbers                   # recompute tables from packaged Source Data
geouno download                        # checkpoint, prior factor and reservoir cases
geouno verify                          # hashes and strict checkpoint loading; no prediction
```

### Model and data assets

Large final-model assets are on the [`artifacts-final-20261101`](https://github.com/Yulongggggg/AVF_geothermal/tree/artifacts-final-20261101)
Git LFS branch. Each asset has a URL and SHA-256 in
[`release_manifest.json`](geouno/resources/release_manifest.json).
For a local asset copy, use `geouno download --from-dir /path/to/assets`.
For authenticated repository access, obtain the assets with SSH:

```bash
git clone --branch artifacts-final-20261101 --single-branch git@github.com:Yulongggggg/AVF_geothermal.git ../geouno-assets
cd ../geouno-assets
git lfs install --local
git lfs pull
cd ../AVF_geothermal
geouno download --from-dir ../geouno-assets
```

## Using the operator and inverse methods

```bash
geouno predict --case test_01 --times 10 --out prediction.npz
geouno run --config multiwell_opt17 --case test_01 --method esmda --threads 4
geouno run --config multiwell_opt17 --case test_01 --method map --threads 4
geouno prepare-cases --check --out runs/cases
geouno download --group nuts
```

Prediction and inverse commands run the model. The paper-number and artifact-verification commands operate
on saved results and model weights. See [reproduction instructions](docs/REPRODUCIBILITY.md) for batch tasks,
NUTS and training, and [data formats](docs/DATA.md) for units and array layouts.

| Component | Implementation |
|---|---|
| Final `resid64_p1` architecture | [`geouno/model/final.py`](geouno/model/final.py), [model description](docs/MODEL.md) |
| Main / temperature / joint training, 99 / 44 / 22 epochs | [`geouno/training`](geouno/training), [training instructions](docs/TRAINING.md) |
| Gaussian conditioning, MAP, ES-MDA, NUTS | [`geouno/inverse`](geouno/inverse) |
| 64 monitoring configurations and new optimized layouts | [`geouno/monitoring`](geouno/monitoring) |
| Tables and abstract numbers | [`geouno/paper.py`](geouno/paper.py), [Source Data](geouno/resources/source_data/final_model_20261015) |
| Exact original final-model source and implementation notes | [`provenance`](provenance), [source map](docs/PROVENANCE.md) |

The common 99-epoch comparison is the main Table 1. The refined GeoUNO model and the earlier baseline
comparison are in Supplementary Table 1. Portable training and evaluation entry points are provided
for GeoUNO and the six deterministic baselines; see [training instructions](docs/TRAINING.md).
Retraining requires the FE corpus described in [DATA.md](docs/DATA.md).

The separate [finite-element example](finite_element/README.md) runs the MOOSE reference simulation
for `test_01` from its supplied mesh, permeability and coupled THM input. It includes the original
`cavity` application source, a pinned MOOSE version and comparisons against the archived fields
and well curves. Its source code retains the LGPL-2.1 licence.

## CO₂ storage transfer

Retraining the same two-field design on the public U-FNO benchmark gives relative saturation and
normalized-pressure errors of **0.063** and **0.042** with 4,096 POD functions, below all five baselines.
Pressure normalization is `z_p = (Δp − 4.17294 bar) / 18.77282 bar`; physical pressure RMSE is reported separately.
The gallery shows saved predictions for four test cases and three times.
Data preparation, training and evaluation commands are in the [CO₂ guide](docs/CO2.md).

![CO2 field gallery](docs/images/co2.png)

## Development

```bash
python -m pip install -e ".[all,dev]"
ruff check .
pytest -q -m "not slow and not release and not model_execution"
```

The [source map](docs/PROVENANCE.md) identifies the archived implementations and numerical verification
commands. The [reproduction guide](docs/REPRODUCIBILITY.md) gives expected results and execution costs.

## Licences and citation

The GeoUNO Python package is MIT licensed, with component-specific licences listed in
[THIRD_PARTY.md](THIRD_PARTY.md). The authors' geothermal weights and data are CC BY 4.0;
third-party benchmark inputs and reference fields retain their provider's terms.
Please cite the manuscript; bibliographic metadata are in [CITATION.cff](CITATION.cff).

The paper's model-specific PCNO and conditional PNO comparisons have separate
[training and evaluation programs](comparisons/README.md). The PCNO adaptation retains its own
GPL-3.0-only licence; it is outside the MIT-licensed GeoUNO package.
