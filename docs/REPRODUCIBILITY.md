# Reproducing the paper

All commands assume `pip install -e ".[all]"` and `geouno download` (add `--group nuts` / `--group prior` where
noted). Times refer to the hardware stated in the paper unless marked otherwise.

## 1. Numbers in the tables (seconds)

```bash
geouno paper-numbers
```

Recomputes Table 1, Table 2 (both cohorts, including NUTS), Supplementary Table 1 and the abstract numbers from
`reference/source_data` and checks all 80 printed values. Expected: `{"checked": 80, "mismatches": 0}`.

## 2. Registered inference tasks

The study registered 1,128 inference tasks before inference: for each of 24 test reservoirs, 45 monitoring
configurations (validation-designed budgets of 2-33 locations in three access geometries: 25 configurations;
space-filling controls with 5, 9, 17 and 33 locations: 12; six observation-channel variants; two shorter durations),
four noise variants on the first six reservoirs, and three-start MAP for the 17-location multiwell configuration. Prior and statically conditioned
reference ensembles accompany each reservoir.

```bash
geouno run --config multiwell_opt17 --case test_01 --method esmda --threads 4   # one task
python scripts/run_all_tasks.py --out runs/paper --threads 4 --shard 0/8        # all tasks, resumable
python scripts/run_all_tasks.py --out runs/paper --summary                      # agreement with the paper runs
```

| Method | Seed | Settings |
|---|---|---|
| ES-MDA | `observation_seed + 19000` | 256 members, 8 updates, `alpha_i = 8`, all singular values, no localization |
| static-only (`multiwell_opt9_k`) | `observation_seed + 19000` | the ES-MDA initial ensemble, analytically conditioned |
| MAP | `observation_seed + 29000` | starts: origin + 2 normal draws; L-BFGS-B `maxiter=1000, maxfun=20000, maxls=40, gtol=1e-4, ftol=1e-10, maxcor=10` |
| prior / static_prior | `81071` | 256 draws; static_prior conditions on the 17 local permeability values |
| NUTS (`test_01`-`test_06`) | `observation_seed + 19000 + chain` | 4 chains, 512 warm-up + 1,024 draws, target acceptance 0.9, max tree depth 7 |

`observation_seed = 610000 + i` for `test_{i+1:02d}`. Likelihood: dynamic readings at stored times 0-6 (through
4.02 years), noise 0.5 K / 0.25 MPa plus the validation-fitted surrogate-error covariance; static readings with
0.1 dec noise; forecasts at 10.01 years. The paper ran ES-MDA and MAP on four CPU threads (10.2 and 10.4 min per
reservoir on average) and NUTS on one NVIDIA RTX A4000 (14.55 h per reservoir for four sequential chains).

![Monitoring comparisons (paper Fig. 6)](images/monitoring.png)

*Paper Fig. 6: candidate geometries and designed versus space-filling layouts (a-d), error ratios versus number of
locations (e-g, i-k), monitoring duration (h, l) and observation types (m). Values come from the tasks above.*

### Expected agreement

`geouno run` and `scripts/run_all_tasks.py --summary` report relative differences of the manuscript metrics against
`reference/paper_task_scores.json`. The network runs in FP32, and CPU round-off depends on the processor and the
number of threads; ES-MDA and MAP iterate on the network outputs and therefore amplify round-off slightly. Agreement
to a few parts in 10^4 or better is expected; bitwise agreement is not. Section 5 lists what was measured for this
release.

## 3. NUTS

```bash
geouno download --group nuts
geouno nuts-verify            # recompute R-hat/ESS from all 4 x 1,024 draws and rescore 256 balanced draws
geouno nuts-chain --case test_01 --chain 0 --device cuda:0 --out runs/nuts   # a new chain (hours)
geouno nuts-merge --case test_01 --dir runs/nuts
```

All six reservoirs meet the registered gate (max R-hat 1.022, min bulk ESS 306 for `test_04`, no divergences). Every
retained transition reached the depth-7 cap. A strict cross-device smoke test of the frozen bundle failed only its
FP32 tolerance, which is documented in `nuts/DEVIATION_smoke_20261001.md`. The other 18 reservoirs (72 chains) are
being sampled and are not part of this release.

## 4. Inverse prior, monitoring inputs and training

```bash
geouno download --group prior && geouno refit-prior --out runs/prior    # marginal SD 0.976 dec
geouno prepare-cases --check --out runs/cases                           # 49 configurations x 25 reservoirs
```

Monitoring placement was optimized on four validation reservoirs; the released `design.json` freezes the result.
Re-deriving it needs the three other validation reservoirs from the FE data release. Training and Table 1 evaluation
need the FE corpus ([TRAINING.md](TRAINING.md)).

## 5. Verification of this release

RESULTS_PLACEHOLDER

## 6. Limitations

* Synthetic two-dimensional reservoir sections; no field validation; posterior members are not re-simulated with FE.
* The test set was inspected during model development; 141 of 249 test fracture-image identifiers also occur in
  training, so the benchmark measures new characterizations and operating configurations rather than unseen geology.
* GeoUNO and the baselines have different training budgets and objectives.
* NUTS transitions saturate the maximum tree depth; inference costs of MAP/ES-MDA (CPU) and NUTS (GPU) use different
  resources.
* The FE solver build and part of the upstream DFN/FE lineage are not fully recorded (see `data_generation/`).
