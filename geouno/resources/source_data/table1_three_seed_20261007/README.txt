THREE INDEPENDENT GEOUNO TEST RUNS — 2026-10-07

Current Table 1 GeoUNO row, Supplementary Tables 6 and 8, and the test panel
of Supplementary Table 19 use all three frozen independently initialized
POD1024+64 models (seeds 20260925, 20260926, 20260927). Each executed 132
initial training epochs and 44 temperature-only plus 22 joint refinement epochs.
The three checkpoints are fixed by the original BF16 validation selection.
No test-based checkpoint or seed selection is performed.

The old transferred checkpoint ed043489 remains the model for application
figures, distribution diagnostics, design, inverse and monitoring experiments.
It is not included as a fourth seed. Original two-seed baseline and three-seed
PCNO results remain unchanged in the comparison tables. Training budgets and
objectives differ; this is not a budget-matched architecture ablation.

Evaluation job 125044 uses FP32 throughout with BF16 and TF32 disabled, the
precision of the previous Table 1 GeoUNO row. All models passed independent
CPU/GPU validation-input prediction checks before ANY test prediction. The
four Table 1 RMSE metrics were independently recomputed in NumPy float64.
The previous BF16 archived-validation replay failed (124860, 124869). Diagnostic
125043 reproduced seed 27 exactly on the same GPU model but found differences
for seeds 25/26 trained on H100/H200. Those attempts are preserved and are not
claimed to have passed. FP32 evaluation is an explicit numerical-protocol
revision, not a relaxed tolerance or a substitute validation score.

Cohort: all 618 test scenarios, all 8,621 actual non-nominal realizations and
all 11 stored times for each model. No ensemble-size filter for field/near-well
RMSE. Near-well temperature uses a producer-centred 6-m disk; pressure uses a
6-m disk at the injector (0,-250 m). These regions contain 1–4 and 4 grid points.
Within each scenario/time, squared paired member/cell errors are pooled and
the square root is taken; resulting scenario/time scores are averaged equally.
Across the three run-level scores, report mean and sample SD (ddof=1).

Ensemble-mean field error compares predicted and FE mean fields over reservoir
rows 16:144. The pressure-tail stratum is the original training-defined group:
960 realizations in 229 scenarios. Well-pressure P90 errors retain the original
532 scenarios with at least four members. Do not relabel producer pressure
as producer temperature; Table 1 uses pointwise near-well temperature RMSE.

Files:
- scenario_time_scores.csv.gz: 20,394 scenario/time records, all metrics,
  seed identities and cohort counts.
- cohort.json.gz: common scenario/time identities and actual member-row IDs.
- summary.json: each run, across-seed means/sample SDs, checkpoint hashes and
  execution identity.
- protocol.json: frozen data/source/checkpoint hashes, precision, metric and
  preflight definitions.
- ../../evidence/table1_three_seed_20261007: replay receipts, completion marker,
  numeric/protection checks, GPU diagnostic, build log and PDF checks.

Reproduce aggregation on the original HPC workspace:
  python scripts/table1_three_seed_20261007/collect.py
  python scripts/table1_three_seed_20261007/install.py
Verify the portable packaged record without GPU inference:
  python scientific_audit_20261002/reproduction/verify_three_seed_tables.py
  python scientific_audit_20261002/reproduction/verify_table1_nearwell.py

Current Table 1: 全场 T 0.369 ± 0.006；全场 p 0.413 ± 0.005；生产井 T 0.964 ± 0.011；注入井 p 0.412 ± 0.004.
Temperature units K; pressure units MPa. Seed SD is optimization variability,
not geological uncertainty, a confidence interval or an ensemble prediction.
