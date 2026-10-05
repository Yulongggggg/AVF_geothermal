# Provenance

The `geouno` package is a reorganized, documented transcription of the research code that produced the
paper's results. Scientific operations are copied statement by statement in the same order; only module names,
data access (stored base inputs instead of the 180 GB cluster pack), path handling and orchestration differ.
`provenance/verify_equivalence.py` checks the package against the archived code bit for bit
(results: `provenance/equivalence_report.json`).

## Sources

| Archive | Commit / location | Role |
|---|---|---|
| [OL-geothermal-final-scheme](https://github.com/Yulongggggg/OL-geothermal-final-scheme) | `main` @ `229a6b7` | historical GeoUNO archive (`geouno/`), prior-fit inputs, parent checkpoint |
| same repository | `codex/workstationd-nuts-20261001` @ `137d61a` | frozen NUTS bundle (checkpoint, 25 reservoirs, protocols, portable runtime) |
| same repository | `claude/workstationd-nuts-results-20261005` @ `7c3b2f7` | returned six-reservoir NUTS results |
| [AVF-Nature-communication---Yulong-Jonah](https://github.com/Yulongggggg/AVF-Nature-communication---Yulong-Jonah) | `main` @ `1281250` | manuscript, 2 October 2026 scientific audit (corrected sources, Source Data), 1 October evidence archive |

Every copied original file is in `provenance/original_sources/` with its SHA-256 and origin in
`provenance/SOURCES.json`. 31 of the 41 files equal a hash pinned by the frozen experiment protocols, by the NUTS
bundle or by the audited runtime import record of the paper inference; the remaining ten are the training sources
(whose run protocol is not public), the portable prior-fit script, the training data loader, and the corrected
`inverse_run.py` (see below).

## Module map

| `geouno` module | Original (in `provenance/original_sources/`) |
|---|---|
| `model/blocks.py` | `temperature_revision/model.py` (`CondBlock`) |
| `model/direct.py` | `direct_operator/model.py` (`DirectFieldOperator`) |
| `model/joint.py` | `joint_architecture/model.py` (`JointTaskOperator`; historical `initialize` omitted) |
| `model/geology.py` | `encoder_study/model.py` (`geology_maps`, `GeologyGate`, `RegionalReadout`, `GeologyJointOperator`, `load`) |
| `model/geouno.py` | `operator_campaign_20260925/model.py` (`coordinates`, `CoordinateTrunk`, `GeoUNOBasisHead`, `load`) |
| `model/numerics.py` | `operator_parallel_20260927_v2/runtime.py`; settings of `operator_final_20260930/backend.py` |
| `inputs.py` | `det_sampler.py` (`field_on_grid`), `history_matching_revision/inputs.py` (`realized_inputs`), `temperature_revision/inference.py` (`PhysicalInputEncoder`, `point_readout`), `direct_operator/inference.py` (`SamplingAdapter`) |
| `backend.py` | `operator_final_20260930/backend.py` (portable bundle version) and `inverse_backend.py` (`Backend`) |
| `inverse/prior.py` | `history_matching_dense/conditioning.py` |
| `inverse/problem.py` | `inference_comparison/geothermal.py` (`Problem`) merged with `operator_final_20260930/inverse_backend.py` (`Problem`) |
| `inverse/esmda.py` | `inference_comparison/algorithms.py` (`esmda`); static-only draw of the corrected `inverse_run.py` |
| `inverse/map.py` | `operator_final_20260930/inverse_run.py` (`map_optimize`) |
| `inverse/nuts.py` | `inference_comparison/algorithms.py` (`nuts`); diagnostics of the NUTS bundle `merge` |
| `inverse/forecast.py` | `operator_final_20260930/inverse_run.py` (`forecast`) |
| `inverse/fit_prior.py` | `history_matching_dense/fit_prior.py`, `geouno/scripts/refit_prior.py` |
| `metrics.py` | `inference_comparison/metrics.py` |
| `monitoring/networks.py` | `history_matching_networks/networks.py`, `history_matching_dense/networks.py`, `inference_comparison/geothermal.py` (`masks`) |
| `monitoring/geometry.py` | `operator_final_20260930/monitoring_geometry.py` |
| `monitoring/design.py` | `history_matching_sparse/design.py` (`reduction`, `update`), `final_model/monitoring_budget.py` (`state`, `choose`) |
| `monitoring/protocol.py` | `operator_final_20260930/monitoring_prepare.py` (`configurations`, case preparation) |
| `training/data.py` | `temperature_revision/data.py`, `overnight_bench/data.py`, `direct_operator/data.py` |
| `training/loss.py` | `joint_mmd/loss.py` |
| `training/metrics.py` | `joint_precision/metrics.py`, `direct_operator/model.py` (`block_values`) |
| `training/pod.py` | `operator_campaign_20260925/prepare_basis.py` |
| `training/train.py` | `operator_campaign_20260925/train.py` |

## Intentional differences

* **Static-only member order (scientific correction, 2 October 2026).** The registered `inverse_run.py` drew the
  static-only ensemble as `(members, parameters)`, while ES-MDA draws `(parameters, members)` and transposes, so
  equal seeds gave different vectors. The corrected draw (`static_members`) matches the ES-MDA initial ensemble
  exactly. The paper reports the corrected results; `reference/paper_task_scores.json` keeps both.
* **Registered geometries.** Case preparation for the 28 registered reservoirs uses the candidate points frozen in
  the monitoring protocol, as the original did. Recomputing them gives identical points except for one coordinate
  of one reservoir that differs in the last floating-point digit (platform `hypot`).
* **Safe loading.** Checkpoints are loaded with `torch.load(..., weights_only=True)`.
* **Portability.** Data paths are arguments; the training script writes its own `protocol.json` instead of reading
  a cluster protocol, and has a CPU/gloo path for smoke tests. The historical `rml` adapter (not used in the
  paper), characterization-input dataset options and cluster orchestration are omitted.
* **Prior-fit script.** The archived portable `fit_prior.py` differs from the cluster file recorded in
  `prior_fit.json` (`7f682486...`) by its argument handling; `geouno refit-prior` reproduces the released factor.

## Release assembly

`provenance/assemble_release.py` records how the release assets, packaged resources, protocols and reference
scores were built from the archives, with hash checks of every input and deterministic tar archives.
