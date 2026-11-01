# Source Data

This index follows the current manuscript. The geothermal application checkpoint is
`f39894ff83e04f47b2a9098f0e387540e1390cf12b9d7f8726dc815dffd16dbb`.
The common 99-epoch comparison and the refined application model are separate experiments.

## Tables and abstract

`geouno paper-numbers` recomputes 200 numerical cells across the five tables below
and three percentage claims in the abstract. Printed expectations are extracted from
the manuscript; `final_model_20261015/printed_values_source.json` records the source
file hashes and table labels.

| Manuscript item | Files | Statistical unit and content |
| --- | --- | --- |
| Main Table 1: common 99-epoch comparison | `common99/scenario_time_scores.csv.gz`, `common99/cohort.json.gz`, `common99/sources.json` | Seven models, three seeds each; 618 scenarios, 11 times and 8,621 realizations per time. Six reported metrics, with equal scenario weighting and sample standard deviation across seeds. |
| Main Table 2: permeability inference | `final_model_20261015/inference_numbers.json`, `inference_scores_24.csv`, `nuts_summary_24.json` | All 24 reservoirs; permeability, temperature/pressure fields and spatial near-well RMSE. |
| Supplementary Table 1: refined GeoUNO and baseline comparison | `table1_three_seed_20261015/scenario_time_scores.csv.gz`, its shared `cohort.json.gz`, `final_model_20261015/baseline_per_seed.json` | Refined GeoUNO seeds 20260915, 20260925 and 20260926; baseline runs identified in the per-seed records. |
| Supplementary Table 2: CO₂ benchmark | `final_model_20261015/co2_baseline_per_seed.json`, `co2_final.json` | Three seeds per model/POD rank; relative field errors and physical-unit RMSE. Pressure RMSE is in bar. |
| Supplementary Table 19: common-protocol trajectory and endpoint errors | `common99/` | The same 21 runs as Main Table 1; full-trajectory and ten-year field RMSE and relative L₂. |
| Abstract: inference and monitoring error reductions | `final_model_20261015/inference_numbers.json` | Equal-reservoir means for the stated 24-reservoir comparisons. |

Within `common99/`, every CSV row identifies its model, seed, scenario, time and
realization count. The shared cohort file lists the actual realization row indices.
`sources.json` records evaluation, checkpoint, training-protocol and evaluator hashes,
and the precision used for each run. Relative L₂ pools squared errors and truth norms
over the selected times within each scenario before averaging scenarios. The CSV
retains both truth norms, so this calculation can be repeated independently.

## Main figures

The current application-figure data are in
[`current_figures_20261101/`](current_figures_20261101/README.md). Its manifest links
the numerical arrays to the exact manuscript PDF hashes, checkpoint and original
rendering inputs. The figure filenames retain an earlier date; the manifest identifies
the final model and the data actually shown.

| Manuscript item | Files in `current_figures_20261101/` | Content |
| --- | --- | --- |
| Figure 1: numerical examples in the overview | `fig1_workflow_arrays.npz`, `fig1_stage_fe_fields.npz`, `fig1_*json`, `fig1_inference_cooling_curves.csv` | The six FE stage fields, displayed model inputs and outputs, illustrative basis shapes, ensemble forecast maps and inference curves. The design inset uses `fig4_location_group_0.npz`. |
| Figures 2 and 3: temperature and pressure prediction | `fig2_3_plot_arrays.npz`, `fig2_3_map_panels.json`, `fig2_3_selection.json`, `fig2_3_*csv.gz` | Displayed field maps and time slices, member response curves, thermal power and cumulative pressure work, and scenario-level error distributions. |
| Figures 2 and 3: full member responses and cohort | `paired_well_members.npz`, `scenario_cohort.json`, `display_fields_histories.npz`, `well_distribution_w1.npz` | Reference and predicted well responses for all 8,621 realizations at 11 times, actual scenario membership and the original plotting inputs. |
| Figure 4: location and operating design | `fig4_location_candidates.csv`, `fig4_location_group_*.npz`, `fig4_ranking_and_selection.csv`, `fig4_operating_settings.csv`, `fig4_*json` | All 164 candidate configurations and three objectives, six ensembles of field maps, ranking and selection statistics, and 18 operating settings. |
| Figure 5: permeability inference and forecasts | `fig5_fields_and_pressure_members.npz`, `fig5_map_panels.json`, `fig5_pressure_curves.csv`, `fig5_reservoir_errors.csv`, `fig5_inference_times.csv`, `fig5_summary.json` | Displayed prior/posterior maps, individual forecast members and intervals, all 24 reservoir errors and original inference times. NUTS times sum the four chains, including warm-up and sampling. |
| Figure 6: monitoring designs | `fig6_monitoring_geometry.json`, `fig6_budget_curves.csv`, `fig6_history_duration.csv`, `fig6_reservoir_metrics.csv.gz`, `fig6_reservoir_ratios.csv.gz` | Candidate and selected sensor locations, budget and history-duration curves, per-reservoir errors and paired ratios. The geometry file records bootstrap settings and the displayed duration-whisker convention. |

The directory README states units, crops, selections and transformations used in
each panel. Array-level index files identify the stored plot arrays independently
of the NPZ containers. These files cover the numerical content of the six main
figures; graphic elements and panel layout remain in the manuscript figure assets.

## Supplementary figures

The files below are in
[`supplementary_figures_20261101/`](supplementary_figures_20261101/README.md), whose
manifest identifies the corresponding current figure assets and original numerical
inputs.

| Manuscript item | Files | Content |
| --- | --- | --- |
| Supplementary Fig. 1: observation-noise sensitivity | `sfig01_reservoir_values.csv`, `sfig01_definitions.json` | 216 reservoir-level values for four metrics, three observation combinations and three temperature-noise levels. |
| Supplementary Fig. 2: reservoir setup | `sfig02_arrays.npz`, `sfig02_definitions.json` | The displayed permeability realization, well coordinates and physical-domain definitions. |
| Supplementary Fig. 3: fracture-network illustration | `sfig03_arrays.npz`, `sfig03_definitions.json` | Prescribed distributions, sampled fracture geometry, pair-distance curves and the displayed native fracture mask. |
| Supplementary Fig. 4: permeability construction | `sfig04_native_inputs.npz`, `sfig04_definitions.json`, `reconstruct_sfig04.py` | Native fracture mask and fixed random state; deterministic reconstruction of all nine original full-resolution maps, with byte hashes. |
| Supplementary Fig. 5: uncertainty propagation | `sfig05_display_arrays.npz`, `sfig05_array_index.json`, `sfig05_definitions.json` | Displayed characterization and endpoint maps, and all 20 member well responses at 11 times. |
| Supplementary Fig. 6: network architecture | `sfig06_definitions.json` | Tensor dimensions and architecture definitions; this schematic contains no sampled, measured or predicted result arrays. |
| Supplementary Fig. 7: prediction regression | `sfig07_regression.npz`, `sfig07_definitions.json` | Reference and predicted reservoir means for all 618 scenarios and 11 times, plus the displayed well means and regression statistics. |
| Supplementary Fig. 8: monitoring geometry | `sfig08_arrays.npz`, `sfig08_locations.csv`, `sfig08_definitions.json` | Candidate coordinates, borehole paths and nested monitoring selections for all six displayed panels. |
| Supplementary Figs. 9 and 10: monitoring placement and budget | `sfig09_*csv`, `sfig10_*csv` | 300 plotted curve points and the underlying values for all 24 reservoirs, across access geometries and monitoring budgets. |
| Supplementary Fig. 11: history duration | `sfig11_reservoir_values.csv`, `sfig11_definitions.json` | 672 values for the prior and three history windows at fixed observation locations. |
| Supplementary Fig. 12: observation combinations | `sfig12_reservoir_values.csv`, `sfig12_definitions.json` | 3,696 values spanning seven observation combinations, three access geometries and the prior. |
| Supplementary Fig. 13: paired reservoir comparisons | `sfig13_reservoir_values.csv`, `sfig13_definitions.json` | Forecast and prior errors, and 840 corresponding error ratios. |

Supplementary Fig. 14 uses `final_model_20261015/co2_visualization.npz` and
`co2_gallery_manifest.json`. The gallery contains the reference fields, predictions
and reservoir masks for the four displayed cases at three times. Its manifest
specifies case selection, time indices and pressure conversion.

## Additional numerical results

The following files are in `final_model_20261015/`:

| Result | Files |
| --- | --- |
| Operating designs and control settings | `operating_design.json` |
| Monitoring budgets, observation channels, observation durations and paired reservoir comparisons | `monitoring_report.json` |
| Forecast timing | `forward_speed.json` |
| MAP, ES-MDA and NUTS computational costs across 24 reservoirs | `inference_cost_24.json` |
| Likelihood and gradient evaluation costs | `likelihood_cost.json` |
| Four-chain NUTS diagnostics for all 24 reservoirs | `nuts_diagnostics_24.csv`, `nuts_diagnostics_24.json`, `nuts_summary_24.json` |
| CO₂ field gallery | `co2_visualization.npz`, `co2_gallery_manifest.json` |

The 24 original NUTS chain archives
are separate download assets, indexed by the package release manifest; each contains
four chains and their saved forecasts and diagnostics. The original `test_16`
convergence flag is retained.

## Historical records

Files under `nuts_20261005/`, `table1_three_seed_20261007/`,
`table1_nearwell/` and `uniform_baselines_20261004/` document earlier experiments.
The root-level files `forward_scenario_time_scores.csv`,
`monitoring_primary_source_data.csv`, `sensor_marginal_source_data.csv`,
`approximation_error_diagnostic.csv`, `fig6_layout_budget_data.json`,
`information_recomputed.json` and `operating_design_recomputed.json` originate from
the earlier `operator_final_20260930` experiment. They are preserved as historical
records and are not inputs to the current table checks.

`final_model_20261015/uniform_final.json` and `table1_final.json` retain earlier table
layouts and summaries. The current common-protocol comparison uses `common99/`.

`SHA256SUMS` identifies the files in this directory. Data and upstream benchmark
licences are described in [DATA.md](../../../docs/DATA.md) and
[THIRD_PARTY.md](../../../THIRD_PARTY.md).
