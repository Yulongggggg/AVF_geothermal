# Third-party software

The GeoUNO Python package is MIT-licensed. It uses the following packages as installed dependencies; none of their
source code is vendored in this repository.

| Package | Used for | Licence |
|---|---|---|
| [PyTorch](https://pytorch.org) | neural operator, automatic differentiation | BSD-3-Clause |
| [NumPy](https://numpy.org), [SciPy](https://scipy.org) | arrays, L-BFGS-B (MAP), special functions | BSD-3-Clause |
| [Pyro](https://pyro.ai) | NUTS kernel (`pyro.infer.NUTS`, `MCMC`) | Apache-2.0 |
| [ArviZ](https://www.arviz.org) | rank-normalized R-hat and bulk/tail ESS | Apache-2.0 |
| [iterative_ensemble_smoother](https://github.com/equinor/iterative_ensemble_smoother) (Equinor) | ES-MDA update | **GPL-3.0** (optional extra `geouno[esmda]`) |
| [MOOSE](https://mooseframework.inl.gov) / PorousFlow | FE reference solver; external dependency of the [finite-element example](finite_element/README.md) | LGPL-2.1 |
| DFN.lab | discrete fracture networks underlying the permeability fields (upstream of this release, not distributed) | see its own terms of use |

Methods are cited in the paper: ES-MDA (Emerick and Reynolds, 2013), NUTS (Hoffman and Gelman, 2014),
rank-normalized R-hat (Vehtari et al., 2021), MMD (Gretton et al., 2012), POD-DeepONet (Lu et al., 2022),
goal-oriented A-optimal design (Attia et al., 2018), Peaceman well model (Peaceman, 1983).

## Finite-element example

`finite_element/` is a separate **LGPL-2.1-only** source-code component, outside the MIT-licensed
GeoUNO Python package. It includes the original `cavity` application, realization-generation and
packing routines, and the example's execution scripts. The licence is retained in
[`finite_element/LICENSE`](finite_element/LICENSE) and
[`finite_element/application/LICENSE`](finite_element/application/LICENSE).
MOOSE itself is an external dependency, pinned to commit
[`381947fe984564cf250e888a4689a8710902152e`](https://github.com/idaholab/moose/tree/381947fe984564cf250e888a4689a8710902152e).
The supplied mesh, fracture image, numerical input and reference arrays are the authors' data under
CC BY 4.0. Source identities and the reconstruction procedure are documented in the
[finite-element guide](finite_element/README.md).

## CO₂ benchmark

The CO₂ experiments use the public dataset accompanying Wen et al. (2022), obtained from the
[U-FNO repository](https://github.com/gegewen/ufno/tree/8315fd7b5bd75282b7efe42ee6b8de86543d13cc).
That repository carries a [CC BY-NC-ND 4.0 licence](https://github.com/gegewen/ufno/blob/8315fd7b5bd75282b7efe42ee6b8de86543d13cc/LICENSE);
its README supplies the original download link and does not specify a separate dataset licence.
Benchmark inputs and reference fields, including the reference fields in the gallery, are third-party
material and are excluded from GeoUNO's code and data licences. Obtain the full dataset from the provider
and follow its terms. The benchmark training implementations in `geouno/co2` are the study authors' code.

## Supplementary comparison programs

`comparisons/pcno/` is a separate **GPL-3.0-only** component, outside the MIT-licensed GeoUNO package.
It contains the study's two-dimensional adaptation and retains Zhenqian Xue's official PCNO v1.0 source
at Code Ocean commit `dc4cd4417ed7f97715ffb47496690ce60f7665a3`, with the original GPL text in
[`comparisons/pcno/LICENSE`](comparisons/pcno/LICENSE). The original four-dimensional source files
are retained unchanged in `comparisons/pcno/upstream/`; they are not imported by GeoUNO.

`comparisons/pno/` is the study authors' MIT-licensed implementation of the conditional PNO comparison,
following Bülte, Scholl and Kutyniok (2025). The [method repository](https://github.com/cbuelt/pfno)
states MIT licensing in its README; its source is not vendored here. The adaptation and execution
protocols for both comparisons are described in [`comparisons/README.md`](comparisons/README.md).
