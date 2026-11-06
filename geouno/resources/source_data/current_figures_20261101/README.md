# Source Data for Figures 1–6

These files contain the saved numerical inputs of the current manuscript figures,
using the final GeoUNO checkpoint with SHA-256
`f39894ff83e04f47b2a9098f0e387540e1390cf12b9d7f8726dc815dffd16dbb`.
The predictions were evaluated in FP32 with TF32 disabled. `manifest.json` links
the data to the exact manuscript PDF files and records the source and file hashes.

## Figure 1

| File | Contents and panels |
| --- | --- |
| `fig1_stage_fe_fields.npz` | Six saved FE temperature fields used on the stimulation stages in panel a, with coordinates, cooling, initial temperature and case IDs |
| `fig1_stage_definitions.json` | FE realization selection, pack-row IDs, temperature transformation and array hashes |
| `fig1_workflow_arrays.npz` | Numerical examples in panels b and c: permeability inputs, saved forecasts, basis illustrations, characterization maps, gradient and producer-cooling histories |
| `fig1_array_index.json` | Shape, dtype and SHA-256 of each workflow array |
| `fig1_panel_definitions.json` | Scenario and member selections, quantity definitions, colour limits and links to reused design data |
| `fig1_inference_cooling_curves.csv` | Prior and ES-MDA intervals, ES-MDA median, FE truth and observations in panel c |

Panel a shows FE temperature in °C with a 20–100 °C colour scale. Panel b uses
the first member of the saved `v4s00000` ensemble for its input/output thumbnails.
Its four basis thumbnails illustrate the representation using singular vectors
of the centered saved temperature ensemble; the fitted model's POD bank is
provided with the training assets. The small coordinate and distance tiles are
schematic inputs.

Panel c's forecasting example is `v4s01745`, using its first and last stored
permeability realizations and its twenty-member mean and standard deviation.
The design example reuses `fig4_location_group_0.npz`: 45 candidate locations
coloured by `up_op_p90` in MPa. The inverse example is `test_01`, scenario
`v4m02656`, with seventeen monitoring locations; its permeability maps are the
same arrays provided for Figure 5. Its temperature curves use producer-block
cooling in K. The gradient thumbnail contains the saved derivative with respect
to 4,096 whitened latent coordinates, reshaped to 64 × 64 for display.

## Figures 2 and 3

| File | Contents and panels |
| --- | --- |
| `fig2_3_plot_arrays.npz` | Displayed arrays for panels a–e, with keys listed below |
| `fig2_3_map_panels.json` | Panel a array names, scenarios, units and colour limits |
| `fig2_3_array_index.json` | Shape, dtype and SHA-256 of each displayed array |
| `fig2_3_selection.json` | Scenario IDs, exact ranks/percentiles within one percentage point of P25/P50/P75/P90, members, well coordinates and operating parameters |
| `fig2_3_selected_member_histories.csv.gz` | Individual FE and GeoUNO responses and derived power/integral for panel c |
| `fig2_3_scenario_time_scores.csv.gz` | All 532 eligible scenarios at eleven times for panels d and e |
| `fig2_3_summary_curves.csv.gz` | Medians and interquartile limits displayed in panels d and e |
| `fig2_3_display_fields_histories_20261105.npz` | Current Figure 2/3 renderer input, selected from the unchanged full-cohort cache |
| `display_fields_histories.npz` | Original renderer input retained for Figure 1 and the previous Figure 2/3 provenance |
| `well_distribution_w1.npz` | Unmodified saved well-response Wasserstein-1 arrays, in `distribution_scenarios` order |
| `paired_well_members.npz` | FE and GeoUNO well responses for all 8,621 members at eleven times |
| `scenario_cohort.json` | All 618 scenarios, pack-row IDs, member slices, stored times and the 532-scenario distribution subset |


The gallery cases lie within one percentage point of P25, P50, P75 and P90 of ten-year ensemble-mean field RMSE. Within these windows, examples were chosen to show clear permeability structure and spatial and temporal reservoir responses. Temperature examples additionally require positive producer cooling at ten years for every FE and predicted member. Full time histories, including early times, are retained. The selection file records the exact ranks and percentiles. Panels d and e retain the full 532-scenario cohort.

The `fig2_3_plot_arrays.npz` keys use `figure2` for temperature and `figure3` for
pressure. Rows are zero based, in P25/P50/P75/P90 order:

- `{figure}_a_row{row}_{quantity}`: the five panel-a maps, each 128 × 128.
- `{figure}_b_{fe|geouno}_fields`: the five panel-b maps for the P50 scenario.
- `{figure}_c_row{row}_{fe|geouno}_response`: member responses at the nine displayed times.
- `{figure}_c_row{row}_{fe|geouno}_lower`: the corresponding thermal power or integrated pressure-change proxy.
- `{figure}_c_row{row}_geouno_{response|lower}_quantiles`: P5, P50 and P95 curves, in that order.
- `{figure}_d_field_rmse_quantiles` and `{figure}_e_w1_quantiles`: P25, P50 and P75 curves, in that order.
- `{figure}_d_well_error_median`: the median absolute error of the well-block ensemble mean.

`time_s` and `time_years` give the nine displayed times. The film keys give the
five panel-b time indices and years. The CSV files retain all eleven stored times;
`displayed=1` identifies the nine plotted times. One year is 365.25 days. Pack rows,
member indices, time indices and scenario slices are zero based; `stop` is exclusive.

Temperature fields and producer-block responses are cooling,
`T_initial − T`, in K. Pressure fields and injector-block responses are pressure
change in MPa. Mean log-permeability is `log10(k / m²)`. Map rows increase upward;
the displayed extent is −400 to 400 m on each axis. The field outputs are cropped
from native 160 × 128 arrays using rows `16:144`; mean-permeability maps are already
128 × 128. Figure 3a shows the pressure error after subtracting each map's spatial
mean. The full RMSE and removed mean offset are recorded separately in the
selection file.

For Figure 2c, thermal power is
`q × 4180 × (T_initial − cooling − T_injection) / 1000` kW.
For Figure 3c, each complete pressure history is integrated by the trapezoidal
rule, starting at zero pressure change at time zero, with water density
1000 kg m⁻³; the result is in TJ. Quantiles are computed across the resulting
member curves. The selection file supplies the initial block temperature and
operating parameters used in the original plot.

In `paired_well_members.npz`, `T_fe`/`T_pred` are producer cooling;
`P_fe`/`P_pred` are injector pressure change; `P_prod_fe`/`P_prod_pred` are
producer pressure change. Each array has shape `(8621, 11)`. The scenario cohort
provides member slices. Wasserstein-1 compares the sorted FE and GeoUNO responses
within each scenario at each time. Panels d and e give each of the 532 scenarios
equal weight.

## Figure 4

| File | Contents and panels |
| --- | --- |
| `fig4_location_design.npz` | Exact scatter values, rank correlations and selection penalties for panels a–d |
| `fig4_location_group_0.npz` … `fig4_location_group_5.npz` | Candidate positions, mean log-permeability and candidate objectives for the six ensembles |
| `fig4_location_candidates.csv` | All 164 configurations and three objectives; separate exact scatter and map values preserve the original stored dtypes |
| `fig4_ranking_and_selection.csv` | Eighteen ensemble/objective rank correlations, penalties and selected scenario IDs |
| `fig4_location_groups.json` | Scenario membership, shared permeability-member hashes and controls |
| `fig4_schematic_trajectories.json` | Optimal locations and straight-line fits used for the panel-e schematic |
| `fig4_operating_settings.csv` | All eighteen settings plotted in panels f–h, with FE/GeoUNO values, feasibility and choices |
| `fig4_operating_definitions.json` | Selection rule, physical references and forecast time |

The location objectives `cool_p50`, `op_p50` and `op_p90` are median producer
cooling (K), median injector pressure change (MPa), and P90 injector pressure change
(MPa). Each is minimized. Groups 0–5 correspond to E1–E6. In each group NPZ, `pos`
contains producer coordinates in m, `M` contains mean log-permeability, and
`fe_*`/`up_*` contain FE/GeoUNO objectives. Maps show the two largest groups; the
scatter plots include all six. Panel e uses groups 0–4 and a schematic stage
spacing of 150 m.

The operating-setting CSV reports output thermal power in kW and P90 **absolute**
injector pressure in MPa. Thermal power equals the original stored objective in
kg K s⁻¹ multiplied by 4.18 kJ kg⁻¹ K⁻¹. Panel-h percentage differences are
`100 × (GeoUNO − FE) / FE`. Ensemble A/B correspond to group 0/1 and each contains
nine settings evaluated on four shared permeability realizations.

All NPZ files can be read with `numpy.load(path, allow_pickle=False)`. CSV files
use UTF-8, and `.csv.gz` files use standard gzip compression.

## Figure 5

| File | Contents and panels |
| --- | --- |
| `fig5_fields_and_pressure_members.npz` | Eighteen gallery maps for panels a–c; raw injector-pressure members, FE truth, observations, times and monitoring coordinates for panel d |
| `fig5_map_panels.json` | Row/method mapping, colour limits and hashes of the eighteen maps |
| `fig5_array_index.json` | Shape, dtype and SHA-256 of the NPZ arrays |
| `fig5_pressure_curves.csv` | Ensemble means and P5/P95 curves, FE truth and seven observations for panel d |
| `fig5_reservoir_errors.csv` | All 24 reservoirs, three methods and three quantities for panel e |
| `fig5_inference_times.csv` | Per-reservoir inference times for panel f |
| `fig5_summary.json` | Case order, observations, medians' underlying ratios and time summaries |

The gallery uses `test_01`. Columns are `prior` (geologic prior), `static_prior`
(conditioned on local permeability measurements), `map`, `esmda`, `nuts`, and
`truth`. Field suffixes are `log10_permeability`, `cooling_K`, and
`pressure_change_MPa`; all maps have shape 128 × 128, origin at the bottom left,
and extent −400 to 400 m on both axes. The log-permeability unit is
`log10(k / m²)`; cooling and pressure change use the same definitions as Figures 2
and 3. `monitoring_locations_xy_m` gives the seventeen marked coordinates.

`{method}_injector_pressure_members_MPa` retains individual propagated responses
at all eleven stored times. Means and P5/P95 quantiles are taken across members.
MAP contributes a single response without an interval. The error CSV contains
physical RMSE and each reservoir's ratio to its geologic-prior RMSE. The
`marked_cross` column identifies the point shown with a cross in panel e.
Inference-time values exclude data loading and forecast generation; NUTS values
are the sum of four chains including warm-up and sampling. The panel-f bars are
means with standard errors across the 24 reservoirs.

## Figure 6

| File | Contents and panels |
| --- | --- |
| `fig6_monitoring_geometry.json` | Candidate coordinates, nested 5/9/17-location indices, well paths and nine-location comparisons for panels a–d; configuration labels and bootstrap settings |
| `fig6_budget_curves.csv` | Exact means and bootstrap limits for panels e–g and i–k |
| `fig6_history_duration.csv` | Mean physical RMSE, displayed whiskers and bootstrap quantiles for panels h and l |
| `fig6_reservoir_ratios.csv.gz` | Individual-reservoir forecast/prior ratios for panel m, including the vertical/inclined means' underlying values |
| `fig6_reservoir_metrics.csv.gz` | Physical RMSE for every plotted configuration and reservoir |

All comparisons use the same 24 reservoirs. Ratios divide each reservoir's
forecast RMSE by its own prior RMSE before averaging. Budget bands use 10,000
reservoir bootstrap resamples of size 24, generated with NumPy
`default_rng(20261002)`; limits are P2.5 and P97.5 of the resampled means.
Duration whiskers are centered on the mean with half-width equal to half that
bootstrap interval's span; the CSV records both displayed endpoints and the
bootstrap quantiles. Temperature RMSE is in K, pressure RMSE in MPa, and
log-permeability RMSE in decimal logarithmic units. Geometry coordinates are in
metres and candidate indices are zero based.
