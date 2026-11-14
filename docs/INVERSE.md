# Inverse problems: the paper and the code

`Problem` exposes the paper's physical observations `d`, predictions `g(xi)` and
full error covariance `R`. MAP and NUTS use the direct covariance-weighted
objective. ES-MDA passes these physical quantities to the existing Equinor
package. There is no observation-whitening step in `Problem`.

## Symbols and shapes

| Paper symbol | Code | Meaning and shape |
|---|---|---|
| `m = log10 k` | `problem.permeability(xi)` | Log permeability, dec, `[members, 128, 128]` |
| `M0` | `problem.mean` | Fixed reservoir prior mean, dec, `[128, 128]` |
| `L_*` | `problem.backend.factor` | Fitted prior covariance factor, `[4096, 4096]` |
| `I` (interpolation) | Bilinear interpolation in `permeability` | Maps the `64 × 64` latent grid to the reservoir grid |
| `x = z` | Output of `prior.transform` | Original prior coordinates, dimension 4096 |
| `xi` (`ξ`) | Inputs to `permeability`, `forward`, `potential` | Coordinates of the conditioned prior, dimension 4096 |
| `g(xi)` | `problem.forward(xi)` | Physical block predictions, `[members, nobs]` |
| `d` | `problem.d` | Physical observations, `[nobs]` |
| `R` | `problem.R` | Full observation-error covariance, `[nobs, nobs]` |
| `(g-d).T R^(-1) (g-d)` | `problem.data_misfit(predictions)` | Covariance-weighted squared mismatch, one value per member |
| `J(xi)` | `problem.potential(xi)` | Scalar negative log posterior for one `[4096]` vector |

The channels are cooling `Delta T = T_init - T` in K and pressure change
`Delta p = p - p_init` in MPa. See [data formats](DATA.md) for grids and times.

## 1. Condition the permeability prior once

The manuscript's prior and conditional parameterization are

```text
z ~ N(0, I),                         m(z) = M0 + I L_* z
d_k = A_k m + e_k,                  e_k ~ N(0, sigma_k² I)
x | d_k ~ N(mu_x_given_y, Sigma_x_given_y),   x = z
x = mu_x_given_y + S_k xi,           xi ~ N(0, I)
S_k S_k.T = Sigma_x_given_y
m(xi) = mu_k + B_k xi
mu_k = M0 + I L_* mu_x_given_y,       B_k = I L_* S_k.
```

`I` denotes spatial interpolation in the field expressions and the identity
matrix in Gaussian covariances. `A_k` averages local log permeability;
`sigma_k = 0.1 dec`. [`prior.condition`](../geouno/inverse/prior.py) computes the
conditional mean and a compact symmetric covariance root; `prior.transform`
applies it. `problem.permeability(xi)` then applies the prior factor and
interpolation without explicitly constructing `B_k`.

All inverse methods update **`xi`**. Without static measurements, `xi = x = z`.
With static conditioning, the dynamic likelihood uses temperature and pressure;
the permeability measurements have already been incorporated into the prior.

## 2. Assemble `d`, `g` and `R` in the same order

[`Problem`](../geouno/inverse/problem.py) uses **time, channel, location** order,
with location varying fastest. Default channels `(0, 1)` put cooling before
pressure at each selected time. For `P` locations, the covariance blocks are

```text
R_T,t = (sigma_T² + v_T,t) I_P
R_p,t = (sigma_p² + v_p,t) I_P + q_t 1_P 1_P.T.
```

The prescribed errors are `sigma_T = 0.5 K` and `sigma_p = 0.25 MPa`. Calibration
provides local surrogate-error variances through `block_local_variance[t][channel]`
and shared pressure-error variance through `pressure_common_variance[t]`.
These stored quantities are already **variances**. The common pressure term
contributes to both diagonal and off-diagonal entries.

`R` is block diagonal across times and channels and remains fixed throughout
inference. Temperature blocks have units K² and pressure blocks MPa². The paper's
main comparison uses `17 × 7 × 2 = 238` dynamic observations through 4.02 years,
plus seventeen static permeability measurements handled by prior conditioning.

## 3. ES-MDA calls the existing package with physical covariance

The manuscript's update is

```text
xi_j^(ell+1) = xi_j^ell
    + C_xi_g (C_gg + alpha_ell R)^(-1)
      [d + epsilon_j^ell - g(xi_j^ell)]
epsilon_j^ell ~ N(0, alpha_ell R),     sum_ell 1/alpha_ell = 1.
```

The sample covariances use the **`Ne - 1`** normalization:

```text
C_xi_g = sum_j (xi_j - mean_xi) (g_j - mean_g).T / (Ne - 1)
C_gg   = sum_j (g_j  - mean_g)  (g_j - mean_g).T / (Ne - 1).
```

[`esmda.py`](../geouno/inverse/esmda.py) retains Equinor's
`iterative_ensemble_smoother` 1.1.0 and passes physical data directly:

```python
solver = PhysicalESMDA(
    covariance=problem.R.detach().cpu().numpy(),
    observations=problem.d.detach().cpu().numpy(),
    alpha=steps,
    seed=seed + 1,
)
g = problem.forward_numpy(xi.T)
solver.prepare_assimilation(Y=g.T, truncation=1.0)
xi = solver.assimilate_batch(X=xi)
```

The library uses members in columns: `xi` is `[4096, Ne]` and `g.T` is
`[nobs, Ne]`. `Problem` uses members in rows. The paper uses 256 members,
eight updates with `alpha = 8`, all singular values retained and no localization.
Initial members use `default_rng(seed)` with draw shape `[4096, Ne]`;
observation perturbations use `seed + 1`.

The small [`PhysicalESMDA` adapter](../geouno/inverse/_esmda_reference.py)
corrects only a dense-covariance sampling issue in version 1.1.0: the library
stores an upper factor `U`, with `R = U.T @ U`, but samples with `U @ eta`.
The adapter uses `sqrt(alpha) U.T @ eta`, so the perturbations have covariance
`alpha R`. Gain calculation and assimilation remain in the package. Its internal
matrix factorizations are numerical linear algebra; the supplied `d`, `g` and
`R` stay in the physical coordinates used by the paper. The old identity-covariance
path was unaffected by this factor-orientation issue.

## 4. MAP and NUTS use the direct posterior formula

```text
pi(xi | d) proportional to exp(-J(xi))
r = g(xi) - d
J(xi) = 0.5 xi.T xi + 0.5 r.T R^(-1) r.
```

`problem.data_misfit` evaluates `r.T R^(-1) r` directly by solving `R w = r`:

```python
predictions = problem.forward(xi[None])[0]
r = predictions - problem.d
w = torch.linalg.solve(problem.R, r)
J = 0.5 * (xi.square().sum() + r @ w)
```

The code solves the linear system rather than constructing an explicit inverse.
There are no transformed observations or predictions. The log-determinant of
fixed `R` is constant in `xi` and is omitted.

Both [`map.py`](../geouno/inverse/map.py) and
[`nuts.py`](../geouno/inverse/nuts.py) call `problem.potential`. MAP uses three
starts (zero and two standard-normal draws), L-BFGS-B, at most 1,000 iterations
per start, `gtol=1e-4` and `ftol=1e-10`; the lowest objective wins.
The paper's NUTS protocol uses four chains, 512 warm-up steps and 1,024 retained
draws per chain, target acceptance 0.9 and maximum tree depth seven. Forecasts
use 64 evenly spaced draws from each chain.

## API migration and saved results

| Earlier API | Current API |
|---|---|
| `forward(...)`: whitened predictions | `forward(...)`: physical `g(xi)` |
| `data`: whitened observations | `d`: physical observations |
| `cholesky` and implicit observation covariance | Explicit physical `R` |
| Squared difference of whitened predictions and observations | `data_misfit(predictions)`, or `potential(xi)` |
| Identity covariance passed to ES-MDA | Physical `R` passed to the package |

`Problem` has no `whiten`, `standardized_residual` or `R_cholesky` interface.
Custom MAP/NUTS problems provide a differentiable `potential(xi)`; custom ES-MDA
problems provide physical `d`, `R`, predictions and `data_misfit`.

NPZ files and chain diagnostics retain the historical key **`y`**, containing
**`xi`**. It is unrelated to the paper's static-observation notation `y = d_k`.
The diagnostic field `mean_whitened_misfit` also retains its legacy name for
file compatibility; its value is now evaluated directly as the ensemble mean
of `r.T R^(-1) r / nobs`.

The direct `nuts()` API keeps its original defaults of 512 draws and depth six.
Use `PAPER_PROTOCOL`, or explicit `samples=1024, max_depth=7`, for the paper;
`geouno nuts-chain` already passes those settings. Data, seeds and saved-sample
meanings are retained. Different linear-algebra operation orders can introduce
small floating-point differences, so compare fresh runs with the saved results.

## Validate and reproduce

```bash
python -m pytest -q tests/test_inverse.py
geouno run --config multiwell_opt17 --case test_01 --method esmda --threads 4
geouno run --config multiwell_opt17 --case test_01 --method map --threads 4
```

Tests check covariance ordering, posterior values and gradients, and ES-MDA
against the direct covariance formula and the former equivalent formulation.
The paper seeds are `observation_seed + 19000` for ES-MDA and
`observation_seed + 29000` for MAP. See the [reproduction guide](REPRODUCIBILITY.md)
for batch commands and NUTS. A saved-table check does not rerun inference;
one reservoir does not establish reproduction of the full 24-reservoir comparison.

The paper's 24-reservoir mean ES-MDA field RMSEs are `0.683 dec`, `6.621 K` and
`0.519 MPa`. Average predicted fields across members before computing each
reservoir's spatial RMSE, then average the 24 errors equally. Near-well RMSE
uses cells within 6 m; uncertainty intervals use quantiles of per-member block
averages. Forecasts target 10.01 years.
