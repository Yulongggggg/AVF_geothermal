# Supplementary figure data

The numbering below follows the current manuscript. `manifest.json` links each
dataset to the exact figure asset and the original numerical input. The application
checkpoint is `f39894ff83e04f47b2a9098f0e387540e1390cf12b9d7f8726dc815dffd16dbb`.

| Figure | Data | Contents |
| --- | --- | --- |
| Supplementary Fig. 1 | `sfig01_reservoir_values.csv`, `sfig01_definitions.json` | Four metrics, three observation combinations, three temperature-noise levels and six reservoirs: 216 displayed values. |
| Supplementary Fig. 2 | `sfig02_arrays.npz`, `sfig02_definitions.json` | The exact stored permeability field and well coordinates used by the THM setup drawing. |
| Supplementary Fig. 3 | `sfig03_arrays.npz`, `sfig03_definitions.json` | Prescribed density curves, sampled illustrative fracture geometry, cumulative pair-distance curves and the archived fracture-trace mask. |
| Supplementary Fig. 4 | `sfig04_native_inputs.npz`, `sfig04_definitions.json`, `reconstruct_sfig04.py` | Native fracture mask and exact recipe reconstructing all nine full-resolution displayed maps; original shape, dtype and byte hash for every map. |
| Supplementary Fig. 5 | `sfig05_display_arrays.npz`, `sfig05_array_index.json`, `sfig05_definitions.json` | Characterization fields, five displayed permeability/FE/GeoUNO endpoint maps and all twenty realization-level well responses at eleven times. |
| Supplementary Fig. 6 | `sfig06_definitions.json` | Architecture schematic and tensor dimensions; this figure contains no sampled, measured or predicted numerical result arrays. |
| Supplementary Fig. 7 | `sfig07_regression.npz`, `sfig07_definitions.json` | Reservoir means for 618 scenarios at eleven times and well-response means at time indices 4 and 10; all four panel statistics. |
| Supplementary Fig. 8 | `sfig08_arrays.npz`, `sfig08_locations.csv`, `sfig08_definitions.json` | Candidate coordinates, borehole paths and nested 5/9/17-location selections for six geometry/placement panels of reservoir test_01. |
| Supplementary Fig. 9 | `sfig09_reservoir_values.csv`, `sfig09_curves.csv` | Guided and space-filling arrays at five monitoring budgets for all three access geometries: 90 curve points and the underlying 24-reservoir values. |
| Supplementary Fig. 10 | `sfig10_reservoir_values.csv`, `sfig10_curves.csv` | Seven metrics for nested guided arrays: 210 curve points and the underlying 24-reservoir values. Multiwell access includes the additional counts 4, 6 and 8. |
| Supplementary Fig. 11 | `sfig11_reservoir_values.csv`, `sfig11_definitions.json` | Prior and three history windows at seventeen fixed locations: 672 values and the displayed means and intervals. |
| Supplementary Fig. 12 | `sfig12_reservoir_values.csv`, `sfig12_definitions.json` | Seven observation combinations across three access geometries, with the prior reference: 3,696 values. |
| Supplementary Fig. 13 | `sfig13_reservoir_values.csv`, `sfig13_definitions.json` | Forecast errors, prior errors and their ratios for five targets, seven observation combinations and 24 reservoirs: 840 ratios. |

Monitoring CSV metric names carry their units: temperature in K, pressure in MPa
and base-ten log-permeability in dec. Near-well RMSE is a spatial error within 6 m
of the relevant well; `well_raw_width90` is the P5–P95 width of the block-averaged
response. Curves average reservoirs equally. Confidence intervals use 10,000
reservoir-bootstrap samples with seed 20261002 and the 2.5th/97.5th percentiles.
The full input report is `../final_model_20261015/monitoring_report.json`.

In `sfig05_display_arrays.npz`, `member_numbers` uses the one-based realization
numbers printed in the figure. The two `mean_confidence` channels are mean
log-permeability and confidence. Endpoint field channels are cooling (K) and
pressure change (MPa), cropped to the central 128 reservoir rows. Maps have
horizontal and vertical extents −400 to 400 m, with the lower edge at row zero.
Well-response arrays have shape `[2, 20, 11]`: producer-block cooling and
injector-block pressure change, followed by realization and stored time.
Coordinates are in metres and `times_s` is in seconds. The accompanying JSON
retains the displayed P5, P50 and P95 curves and colour limits.

In `sfig07_regression.npz`, `PT` and `TT` are GeoUNO and FE reservoir means,
respectively, with shape `[618, 11, 2]`; channels are cooling (K) and pressure
change (MPa). `ty` gives years. Keys beginning `wb_` give the scenario-ensemble
well means: `dT` is producer cooling, `dp` is injector pressure change, `scfe` is
FE and `scpred` is GeoUNO. Scenario order and realization row identities follow
`../current_figures_20261101/scenario_cohort.json`. Each scenario contributes one
point at each displayed time.

The CO₂ gallery in Supplementary Fig. 14 uses
`../final_model_20261015/co2_visualization.npz` and
`../final_model_20261015/co2_gallery_manifest.json`;
its third-party reference fields retain the provider's terms described in
[THIRD_PARTY.md](../../../../THIRD_PARTY.md).

In Supplementary Fig. 2, `logk_truth` is log10(k / m²), and `point_xy_m` gives
injector and producer coordinates in metres. The reservoir occupies
[−400, 400] × [−400, 400] m; the cap and bed extend the vertical domain to
[−500, 500] m. The figure's original input is identified independently of
any neural-network checkpoint.

In Supplementary Fig. 3, panels a–d are an illustration generated with seed 7;
panel e uses the separately archived trace map 0370. `sizes` and `centers` are
in metres, `normals` are unit vectors and `polys` contains the displayed disk
vertices with the original 2.5 diameter multiplier. The `panel_*_curve_*_xy`
arrays contain the exact plotted coordinates. Polar-angle densities are per
radian while their plotted horizontal coordinates are degrees. `mask` is the
native fracture mask; `panel_e_display` is the displayed 1−mask image.

Supplementary Fig. 4 uses recipe 6036 and native fracture image 1912. Its nine
1200 × 1200 float64 maps are deterministic functions of one native boolean
mask and the recorded PCG64 state. The source data preserve these native
inputs, the original construction algorithm, all constants and every full
map's byte hash. No map is downsampled or rounded. To reconstruct the maps,
run from this directory:

```bash
python reconstruct_sfig04.py --out /path/to/sfig04_maps.npz
```

Omit `--out` to verify all maps in memory. NumPy 2.4.6 and SciPy 1.15.3 are the
reference versions. The generated NPZ contains panels a–i as `W`, `Ft`, `Fc`,
`Gp`, `S0`, `Sf_channelized`, `injector_term`, `score` and `M`, followed by
`x_m` and `y_m`. All maps except `M` are dimensionless; `M` is log10(k / m²).
Coordinates are in metres. Reconstructing these source inputs reproduces all
nine original map hashes and both coordinate hashes; rendering them with the
original script reproduces the manuscript PNG byte for byte.

In Supplementary Fig. 8, each panel's candidate, path and selected coordinates
are in metres. The location CSV retains candidate indices and names, including
the fixed injector and producer in every count. Nested selections use the
original index lists; plotting-coordinate checks cover all six panels.
