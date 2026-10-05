Uniform 99-epoch geothermal baselines — 2026-10-04

Published set: DeepONet, FNO, U-FNO, U-Net and ResNet, each with seeds
20260915/16/17. All 15 published runs completed 99 epochs, full validation
(542 scenarios, 6,275 members) and full test evaluation (618 scenarios,
8,621 members, 11 times). The current paper uses the historical transferred
POD1024+64 checkpoint separately; it is not a member of this common-budget set.

The frozen campaign originally included an additional comparator. Its three
completed runs remain in this numerical archive, but the author retired it
from all manuscript comparisons on 2026-10-04. Thus the archive has 18 complete
baseline runs and the published table has 15. The old grid-output architecture
is recorded separately in discontinued_grid_control.json: one complete run,
two incomplete runs discontinued when resources shifted to the POD model.
The reporting policy is docs/baseline_reporting_policy.json. No raw output or
checkpoint was deleted, relabelled or reselected to change the published set.

per_scenario_time.csv.gz: archived paired field RMSE per scenario and time,
in kelvin and MPa. All 18 completed baseline runs are retained. These are
already spatial/member RMSEs, not pointwise prediction arrays.
per_seed.csv: equal-scenario, equal-time mean RMSE and ten-year mean RMSE.
summary.json: per-seed values, across-seed means and sample SD (ddof=1),
source/checkpoint hashes, and the explicit published subset.
table_rows.tex: published five-model rows for the named TeX table asset.
protocol.json: unchanged frozen training protocol (including original scope).
verification.json: portable numerical reaggregation and table checks.

For each scenario/time, take the square root after averaging squared paired
errors over permeability members and reservoir-grid cells. Then average
scenario RMSEs equally and, for full-horizon scores, average eleven times
equally. Finally compute the mean and sample SD across three training seeds.
The SD is training-run variability, not predictive spread or a confidence band.
No distribution scores, interval-hit columns or retired reporting metrics are
exported in the performance table.

Portable verification (standard-library Python):
  python scientific_audit_20261002/reproduction/verify_uniform_baselines.py
Full audit/export from the archived campaign:
  python scientific_audit_20261002/reproduction/complete_uniform_baselines.py \
    --campaign /path/to/runs/fair_comparison_20260920 \
    --output scientific_audit_20261002/source_data/uniform_baselines_20261004

The export checks frozen protocol and checkpoint identity, completion flags,
99 consecutive epochs, training member counts, identical evaluation membership,
all eleven time indices, and the archived per-time aggregation. This revision
recomputes aggregation from saved scenario metrics; it is not fresh inference.
The underlying metric implementation is overnight_bench.evaluate.metric_record,
with reservoir crop [:, :, 16:144]; fair_comparison.evaluate evaluates best.pt.
Checkpoint selection uses minimum reference-normalized validation score; the
separate eligible-versus-parent flag does not restrict the reported best.pt.
Architecture citations remain in the manuscript baseline Methods paragraph.
