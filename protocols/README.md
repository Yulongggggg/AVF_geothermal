# Frozen experiment protocols

Copies of the protocols registered before the reported inference runs (checkpoint `ed043489...`). Absolute
paths inside them are identifiers of the original cluster files, not runtime dependencies.

| File | SHA-256 | Content |
|---|---|---|
| `monitoring_protocol.json` | `968aa31e...` | registered monitoring study: 4 validation and 24 test reservoirs with their seeds, candidate geometries, 49 configurations, ES-MDA settings, reference-ensemble seeds, source pins |
| `inference_protocol.json` | `c09205d5...` | the 1,128 registered tasks, MAP optimizer options, NUTS settings, source pins |
| `hamiltonian_protocol.json` | `b60804ad...` | six-reservoir NUTS cohort, sampler settings and convergence gate |
| `model_selection.json` | `5d3e7d7c...` | author selection of the checkpoint, configuration and training lineage |
| `monitoring_design.json` | `8d9d89d9...` | validation-only nested designs (same file as `geouno/resources/design.json`) |
| `calibration.json` | `b22e64c3...` | validation-fitted surrogate-error second moments (same as `geouno/resources/calibration.json`) |
| `case_manifest.json` | `11c58ca1...` | SHA-256 of 1,300 historical monitoring case files; this archive includes configurations beyond the current registered task set |

`geouno.monitoring.configurations()` reproduces the registered configurations exactly, and the candidate points of
the registered reservoirs are packaged in `geouno/resources/geometries.json` (tests in `tests/test_monitoring.py`).
The current 49 configurations produce 1,104 test-case files plus 49 validation-pilot files; the four noise
variants are restricted to the first six test reservoirs.
