# Third-party software

GeoUNO's own code is MIT-licensed. It uses the following packages as installed dependencies; none of their
source code is vendored in this repository.

| Package | Used for | Licence |
|---|---|---|
| [PyTorch](https://pytorch.org) | neural operator, automatic differentiation | BSD-3-Clause |
| [NumPy](https://numpy.org), [SciPy](https://scipy.org) | arrays, L-BFGS-B (MAP), special functions | BSD-3-Clause |
| [Pyro](https://pyro.ai) | NUTS kernel (`pyro.infer.NUTS`, `MCMC`) | Apache-2.0 |
| [ArviZ](https://www.arviz.org) | rank-normalized R-hat and bulk/tail ESS | Apache-2.0 |
| [iterative_ensemble_smoother](https://github.com/equinor/iterative_ensemble_smoother) (Equinor) | ES-MDA update | **GPL-3.0** (optional extra `geouno[esmda]`) |
| [MOOSE](https://mooseframework.inl.gov) / PorousFlow | FE reference simulations that produced the database (upstream of this release, not distributed) | LGPL-2.1 |
| DFN.lab | discrete fracture networks underlying the permeability fields (upstream of this release, not distributed) | see its own terms of use |

Methods are cited in the paper: ES-MDA (Emerick and Reynolds, 2013), NUTS (Hoffman and Gelman, 2014),
rank-normalized R-hat (Vehtari et al., 2021), MMD (Gretton et al., 2012), POD-DeepONet (Lu et al., 2022),
goal-oriented A-optimal design (Attia et al., 2018), Peaceman well model (Peaceman, 1983).
