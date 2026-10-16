# Reference results for the final model

[`paper_task_scores.json`](../geouno/resources/paper_task_scores.json) contains the saved scores for
checkpoint `f39894ff…16dbb`: 1,392 ES-MDA monitoring tasks, 72 static-only monitoring tasks, 24 MAP tasks,
48 prior/static-reference ensembles and six NUTS forecasts (1,542 records).
Keys are `<configuration>/<reservoir>/<method>`, with `prior/<reservoir>` and `static_prior/<reservoir>`
for the reference ensembles. The optimized observation locations and calibration use the final model.

The current numerical Source Data are in
[`final_model_20261015`](../geouno/resources/source_data/final_model_20261015) and
[`table1_three_seed_20261015`](../geouno/resources/source_data/table1_three_seed_20261015).
They include per-reservoir spatial errors, saved three-seed scenario/time scores, per-seed baseline values,
all CO₂ rank comparisons, completed uniform-protocol results, timings, diagnostics and the CO₂ gallery arrays.

`geouno paper-numbers` independently aggregates 138 entries and checks them against the updated manuscript.
Four unfinished uniform-protocol model rows are dashes. `source_data/SHA256SUMS` covers every packaged file.
Older dated folders retain earlier evaluations; the current table checker uses them only where explicitly
needed for unchanged baseline data. The application checkpoint is one of the three Table 1 models.
