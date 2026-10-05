# Data formats

## Units and grids

| Quantity | Symbol | Unit | Grid |
|---|---|---|---|
| Cooling | `Delta T = T_init - T` | K | 160 x 128 output grid |
| Pressure change | `Delta p = p_w - p_w,init` | MPa | 160 x 128 output grid |
| Log permeability | `m = log10 k`, `k` in m^2 | dec | 128 x 128 reservoir grid |

The output grid covers `x` in [-400, 400] m (128 columns) and `y` in [-500, 500] m (160 rows, row 0 at
y = -500 m). Rows 16-143 (|y| <= 400 m) are the 800 x 800 m reservoir; the other rows are the cap and bed layers.
The permeability grid covers the reservoir square [-400, 400]^2 m. The injector is at (0, -250) m; the producer
position varies by scenario. The eleven stored times are 7,000; 63,000; 511,000; 4,095,000; 32,183,000;
79,583,000; 126,983,000; 174,383,000; 221,783,000; 269,183,000 and 316,000,000 s. Time indices 4, 5, 6, 7 and 10
are 1.02, 2.52, 4.02, 5.53 and 10.01 years.

## Reservoir cases (`reservoirs/cases/<label>.npz`)

The released files are byte-identical to the `multiwell_opt17` case files of the paper runs (SHA-256 in
`geouno/resources/cases.json` and `protocols/case_manifest.json`). Labels: `test_01`-`test_24` (standard test
split) and `pilot_01` (validation reservoir used for the pilot check).

| Key | Shape, dtype | Meaning |
|---|---|---|
| `scenario`, `case_id` | str | corpus scenario and member identifiers |
| `truth` | (11, 2, 160, 128) f64 | FE `Delta T`, `Delta p` at the eleven stored times |
| `logk_truth` | (128, 128) f32 | true log10 permeability |
| `prior_mean` | (128, 128) f32 | random prior mean `M0` of the inverse prior (training mean + smooth 0.5-dec field) |
| `point_xy` | (17, 2) f64 | monitoring locations (m); rows 0 and 1 are the injector and producer |
| `weights` | (17, 160, 128) f64 | normalized 6-m averaging blocks on the output grid |
| `k_weights` | (17, 128, 128) f64 | the same blocks on the permeability grid |
| `observed` | (11, 2, 17) f64 | block temperature/pressure plus noise (0.5 K, 0.25 MPa); only times 0-6 are assimilated |
| `k_observed` | (17,) f64 | block log10 permeability plus 0.1-dec noise |
| `qoi_weights` | (2, 160, 128) f64 | 6-m blocks at the producer (temperature target) and injector (pressure target) |

Noise is coordinate-keyed: the error of a reading depends only on the reservoir's `observation_seed`, the channel
and the location, so shared readings are identical across layouts. Other monitoring configurations are generated
from these files with `geouno prepare-cases` (`geouno.monitoring.prepare_case`).

## Base inputs (`reservoirs/base_inputs/<label>.npz`)

`base`: (11, 13, 160, 128) float32, the standardized network input of the reservoir at each stored time, as built
by the original data loader. Inversion replaces channel 0 by the candidate permeability.

| Channel | Content | Used by GeoUNO |
|---|---|---|
| 0 | `(m - M_mean) / M_std` (m on the output grid; cap/bed rows copy the nearest reservoir row) | yes |
| 1 | confidence `C` (zero in inversion inputs) | no |
| 2, 3, 4 | reservoir mask, `x/400`, `y/500` | rebuilt from the grid |
| 5, 6 | distance to producer / injector, divided by 400 m | yes |
| 7, 8 | injection rate `q` [kg/s] and temperature `T_inj` [K], standardized | yes |
| 9 | `S_max` (sampling amplitude of the characterization prior) | no |
| 10 | absolute producer bottom-hole pressure [MPa], standardized with `pp_norm` | yes |
| 11 | normalized log time `tau` | yes |
| 12 | depth [km], standardized with `depth_norm` | yes |

`geouno.inputs.PhysicalInputEncoder` converts this to GeoUNO's six fields (`m + 13.5`, mask, `x/400`, `y/500`,
producer and injector distances) and five controls (`q/3.5`, `(T_inj - 303)/10`, `(p_prod - 20)/10`,
`(log10 t - 6)/2`, `(depth - 3.06)/0.6`). The normalization constants are in `geouno/resources/stats.json`.

## Inverse prior (`prior/prior_factor_64eb46fb.npy`)

Float32 (4096, 4096) lower-triangular Cholesky factor `L` of the latent covariance on the 64 x 64 grid. A prior
draw is `m = M0 + I(L z)`, `z ~ N(0, I)`, with bilinear interpolation `I` to 128 x 128 (align corners). It is fitted
from 1,024 training fields only (`geouno refit-prior`).

## Calibration (`geouno/resources/calibration.json`)

Validation-fitted surrogate-error second moments (120 validation realizations, 289 blocks), per stored time:
`block_local_variance[t][c]` (diagonal block variance, channel `c`), `pressure_common_variance[t]` (spatially shared
pressure component), `field_variance[t][c]` and `qoi_variance[t][c][d]` (producer/injector blocks). The likelihood
adds instrument variance (0.5 K)^2 or (0.25 MPa)^2 to the local variance.

## Monitoring design (`geouno/resources/design.json`)

Nested candidate indices for each geometry (`multiwell`, `vertical`, `inclined`) and rule (`optimized`,
`space_filling`), from 2 to 33 locations, chosen on four validation reservoirs before any test inference.

## NUTS results (`nuts/test_0X/`)

`chain_c/latent_samples.npz` (`y`: 1 x 1024 x 4096 whitened coordinates), `chain_c/diagnostics.json`
(Pyro diagnostics, step sizes, timings), `diagnostics.json` (four-chain R-hat/ESS gate), `forecast.npz` and
`scores.json` (256 balanced draws at ten years). The merged `latent_samples.npz` of the original return is omitted
because it is the concatenation of the four chain files.

## FE corpus (Zenodo, on publication)

The database will be released with the paper as a "pack" directory, the format read by `geouno.training`:

| File | Shape | Content |
|---|---|---|
| `meta.csv` | 95,966 rows | `case_id`, `scen`, `split` (train/val/test/holdout), `row`, `real` (member label; `nom` = nominal run) |
| `real.npy` | (N, 128, 128) f32 | realized log10 permeability |
| `fields.npy` | (N, 11, 2, 160, 128) f32 | `Delta T`, `Delta p` at the eleven stored times |
| `scal.npy` | (N, 9) f32 | `q`, `T_inj`, `S_max`, producer x, producer y, mean confidence, drawdown, depth, initial pressure |
| `_ref_times.npy` | (11,) | stored times in seconds |
| `stats.json` | | training normalization (`fields_mean/std`, `M_mean/std`, `p_res_std_mpa`, `pp_norm`, `depth_norm`, ...) |

Split sizes: 79,409 training, 6,808 validation, 9,232 test and 517 extreme-pressure-holdout records (72,939,
6,275, 8,621 and 448 actual realizations; 6,585 / 542 / 618 / 69 scenarios). The absolute producer pressure is
`p0 - 0.00981 * y_prod - drawdown` [MPa].

## Licence

The model weights, reservoir cases, NUTS results, prior-fit inputs and Source Data released here are licensed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please cite the paper when using them.
