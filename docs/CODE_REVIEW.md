# Code review notes (release preparation, 5 October 2026)

The scientific code path of the paper was reviewed while transcribing it into this package: model, input
encoding, static conditioning, posterior, MAP, ES-MDA, NUTS and its diagnostics, forecast scoring, monitoring
geometry and noise, A-optimal design, POD fitting, training objective and schedule, and the prior fit. The
transcription is verified bitwise against the archived code (`provenance/verify_equivalence.py`).

## Verified mathematics

* Static conditioning: the symmetric root `S = I - R R^T` with `c = 1/(s(s + sigma))`, `s^2 = lambda + sigma^2`
  satisfies `S S^T = I - H^T (H H^T + sigma^2 I)^{-1} H` because `2c - c^2 lambda = 1/(lambda + sigma^2)`
  (unit test against the dense formula).
* `A_k` uses separable 1-D linear interpolation with aligned corners, which equals the bilinear interpolation used
  to map latent fields to permeability, so static data and permeability maps share one observation operator.
* The pressure covariance (diagonal plus a rank-one spatially shared component) and its Sherman-Morrison quadratic
  form agree with a dense inverse (unit test).
* The design gain `reduction(...)` equals the drop of `trace(Y S Y^T)` after `update(...)` for every candidate
  (unit test); normalizing `Y` gives the equal-weight relative-variance objective of Methods Eq. 6.
* Empirical CRPS (V-statistic form), MMD (biased V-statistic, zero for identical ensembles) and the ES-MDA
  inflation (eight factors of 8, reciprocals sum to one) were checked.
* Training schedule: 11 warm-up, 55 transfer and 66 fine-tuning epochs; 4,309,756 trainable parameters.

No error that changes a reported number was found in the reviewed code beyond the static-only member order that
the 2 October 2026 audit had already corrected.

## Findings

1. **Stale THM template in the manuscript audit snapshot (documentation).**
   `AVF-Nature-communication---Yulong-Jonah/scientific_audit_20261002/source_snapshot/OL-geothermal-final-scheme/
   THM_template_final.i` (SHA-256 `70d80c53...`) is a July 2026 version: it still builds explicit lower-dimensional
   fracture blocks and uses `epv_coeff = 0.58`, `P_coeff = 1.88e-11`. The paper describes no explicit fracture
   elements and `A_P = 7.916e-12`, `A_eps = 0.541`, which is the r4/r5 template `a87cf842...` that the
   design-benchmark FE runs pinned (`OL-geothermal-final-scheme` `main`). The FE results are unaffected; the snapshot (and any
   line numbers cited from it in `SOURCE_ATTRIBUTION.md`) should be updated to `a87cf842...`.
2. **Generator version gap (provenance).** The cluster copy of `generate_dataset_v4.py` pinned by the
   design-benchmark FE runs (`9261f5e1...`, `/mnt/home/yliu5/projects/v4r5/`) is not in any repository; the latest
   version committed to `OL-geothermal-final-scheme` is `b8eea882...`. It should be retrieved from the cluster and archived.
3. **Superseded model presented as current (documentation).** `OL-geothermal-final-scheme/geouno/README.md` still
   calls the grid-output model `cf278e64...` the "fixed manuscript checkpoint" and reports a failed NUTS case that
   belongs to that model; `geouno/artifacts/nuts_chains` and `artifacts/inference/cases` also belong to it. A note
   pointing to this repository would prevent confusion when that repository is made public.
4. **Cached mutable return value (robustness).** The original `history_matching_dense.network_points` is
   `lru_cache`-decorated and returns the cached list itself; a caller that mutated it would corrupt later calls.
   No paper code path mutates it; `geouno.monitoring.candidate_points` returns a copy.
5. **Platform round-off in recomputed geometry (portability).** Recomputing the frozen monitoring candidates on a
   different machine reproduces 83 of 84 geometries exactly and the remaining one to the last floating-point digit
   (`np.hypot` differs between C libraries). Masks, noise keys and therefore observations are unaffected; case
   preparation uses the frozen points for the registered reservoirs.
6. **FP32 versus FP64 rescoring (precision note).** For the corrected static-only forecasts, the Source Data
   rescored log-permeability RMSE in FP64 while `scores.json` uses the FP32 path; values differ at about 3e-8 dec
   and no printed digit changes.
7. **Redundant work (efficiency).** The original `inverse_backend.Problem` built the default covariance, data and
   static conditioning before overwriting or discarding them. The merged `geouno.inverse.Problem` builds each once
   and computes static conditioning only when it is used.
8. **Global side effects (usability).** The original backends set `torch.set_num_threads` inside their
   constructors. Here thread count is an explicit argument; note that CPU results depend at round-off level on it.
9. **Licensing.** ES-MDA uses Equinor's `iterative_ensemble_smoother` (GPL-3.0). It is an optional extra and is not
   vendored, so the package itself can be MIT-licensed.

## Not resolved by code review

The paper and audit list the remaining scientific limitations (no FE re-simulation of posterior members, test-set
exposure during development, shared upstream fracture images across splits, unequal training budgets, NUTS
depth-cap saturation, unrecorded MOOSE build). They are reproduced in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
