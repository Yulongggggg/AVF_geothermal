# Reference results of the paper

## `paper_task_scores.json`

Archived `scores.json` of every registered inference task of the paper (checkpoint `ed043489...`), keyed by

* `<configuration>/<reservoir>/esmda` - 1,080 ES-MDA monitoring evaluations;
* `multiwell_opt9_k/<reservoir>/static` - the 24 static-only forecasts after the 2 October 2026 correction of the
  member draw order (reported in the paper); `.../static_historical_member_order` keeps the superseded records;
* `multiwell_opt17/<reservoir>/map` - 24 MAP point estimates;
* `prior/<reservoir>`, `static_prior/<reservoir>` - reference ensembles (256 members);
* `multiwell_opt17/test_0X/nuts` - the six four-chain NUTS forecasts.

Each record has `field_rmse` (temperature [K], pressure [MPa]) of the ensemble-mean ten-year fields,
`permeability_rmse` [dec] and, per well target, raw-ensemble and surrogate-mixture scores. `geouno run` compares
new runs with these records; `geouno.inverse.headline` extracts the manuscript metrics.

## `source_data/`

Copied from the manuscript's scientific audit (`scientific_audit_20261002/source_data`, revision of 7 October 2026);
`SHA256SUMS` lists every file.

| File | Content |
|---|---|
| `monitoring_primary_source_data.csv` | per-reservoir field RMSE, well P50 absolute error and P5-P95 width of every monitoring configuration, MAP and the references (independently recomputed; canonical for Table 2 and Figs. 5-6) |
| `sensor_marginal_source_data.csv` | changes between nested location counts with paired bootstrap intervals |
| `fig6_layout_budget_data.json` | inputs of main Fig. 6 |
| `approximation_error_diagnostic.csv` | scores of the separate surrogate-error mixture diagnostic |
| `forward_scenario_time_scores.csv` | archived per-scenario/time forward scores of the selected model (618 test scenarios x 11 times x 2 fields) |
| `information_recomputed.json`, `operating_design_recomputed.json` | local-information and operating-design summaries |
| `table1_nearwell/` | Original baseline archive; the transferred GeoUNO row is superseded in Table 1 |
| `table1_three_seed_20261007/` | Full FP32 test scores of the three independent GeoUNO models now used in Table 1 |
| `uniform_baselines_20261004/` | Supplementary Table 1: five baselines x three seeds under the uniform protocol |
| `nuts_20261005/` | six-reservoir NUTS per-reservoir scores, diagnostics summary and notes |

`geouno paper-numbers` recomputes Tables 1, 2, S1 and the abstract numbers from these files. The archived
attention U-Net runs in `table1_nearwell` were retired from reporting by the authors on 4 October 2026 and are
excluded from the tables.
