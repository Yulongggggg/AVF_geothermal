TABLE 1: WHOLE-FIELD AND NEAR-WELL PAIRED RMSE

Scientific question
How accurately do the fixed delivered predictors reproduce paired FE fields,
both across the reservoir and locally around the engineering readout wells?
This augmentation evaluates existing checkpoints; it does not train, tune or
select a new predictor. Original differences in training budgets remain.

Metric and cohort
Temperature: grid points within a 6 m disk centred on the producer.
Pressure: grid points within a 6 m disk centred on the injector (0, -250 m).
The producer disk contains 1–4 output-grid points, the injector disk 4.
For each scenario and stored time, square each paired member/cell error,
average over actual members and region cells, then take the square root.
Average the resulting RMSEs equally over 618 scenarios and eleven times.
Each time includes all 8,621 actual test members, without padding or an
ensemble-size eligibility filter. These are local-field RMSEs, not errors
after spatially averaging well responses and not response-quantile errors.
Units: temperature K; pressure MPa. Whole-field columns use reservoir cells.

Training repeats
DeepONet, FNO, U-FNO, U-Net, ResNet and Attention U-Net: seeds 20260913/14.
PCNO adaptation: seeds 20260915/16/17.
GeoUNO: one fixed transferred POD1024+64 operator, checkpoint SHA-256
ed04348969fe27903bc9bd084781164f65c610c7e46da0ff75bee60553fdc18e.
Means and sample standard deviations are calculated across run-level scores
last. GeoUNO has no across-training standard deviation. These standard
deviations are training-run variation, not predictive ensemble uncertainty.

Files
scenario_time_rmse.csv.gz: all scenario/time scores for all 16 fixed runs.
  T/P_field_rmse and T/P_nearwell_rmse come from the same fresh predictions.
  T/P_reference_field_rmse retain the original reports for numerical checks.
cohort.json.gz: common scenario IDs, member-row identities and time indices.
run_summary.csv: independent equal-scenario/time aggregation for each run.
summary.json: mean/sample SD, complete-cohort checks, hashes and drift from
  original whole-field reports, including maximum individual differences.
protocol.json: checkpoint/reference hashes, batch sizes and fixed methods.
precision_diagnostic.json: original versus new forward path on the same GPU,
  with independent NumPy float64 local-RMSE verification for one scenario.
data_identity.json and source_checks.json: input metadata hashes and comparison
  of current source files against the code snapshots already in this repository.
jobs.json: scheduler provenance, not a live job-status interface.

Reproduction
Reaggregate the public CSV and verify every printed Table 1 cell:
  python scientific_audit_20261002/reproduction/verify_table1_nearwell.py
Fresh inference additionally requires the FE pack and checkpoints at the paths
in protocol.json. They are not embedded in these CSV files. The exact evaluator
and aggregation scripts are in reproduction/table1_nearwell/. Copy protocol.json
beside evaluate.py, preserve its checkpoint/data paths or explicitly document
any relocation, then run evaluate.py --run INDEX for INDEX 0 through 15.
The evaluator writes results beside itself; aggregate.py --root EVALUATION_DIR
--out EXPORT_DIR verifies every run before exporting. The original raw JSON
outputs remain under geouno-scientific-audit-20261002/table1_nearwell_20261003/
results/ on the project filesystem; summary.json records their hashes.

Precision
As in the original evaluation, baseline and PCNO inference uses BF16 automatic
mixed precision; GeoUNO uses FP32 with TF32 disabled. Actual GPU and PyTorch
versions are recorded per run. Fresh BF16 results need not be bitwise identical
to archived results. All four printed columns use fresh inference, and the
original report values remain separate rather than being silently overwritten.
This does not turn the delivered-model comparison into matched-budget training.
