# FE database generation (archival)

This directory archives the MOOSE PorousFlow input template and the Python generator that create the coupled
thermo-hydro-mechanical (THM) simulation records of the paper. It is provided for transparency; the generated
database will be released on Zenodo, so regenerating it is not needed to use or verify GeoUNO.

The files are byte-identical copies from `OL-geothermal-final-scheme` @ `229a6b7` (`main`); hashes are in
`SHA256SUMS`. Comments are partly in Chinese. Paths in the files point to the original workstation and must be
overridden (see below).

## Physics

`THM_template_final.i` (SHA-256 `a87cf842...`) solves quasi-static momentum, energy and water-mass balance in an
800 x 800 m vertical section with 5 m effective thickness, flanked by cap and bed layers: Biot effective stress,
IAPWS-IF97 water (MOOSE `Water97FluidProperties`), thermo-poroelastic porosity with
`A_P = (alpha_B - phi_ref)/K_s = 7.916e-12 Pa^-1`, `A_eps = alpha_B - phi_ref = 0.541`, Kozeny-Carman permeability,
a point-source injector at (0, -250) m and a Peaceman producer at scenario-specific bottom-hole pressure, Newton
with MUMPS and adaptive time stepping to 3.16e8 s. Natural fractures enter only through the static equivalent
permeability field; no lower-dimensional fracture elements are generated. The template uses only standard MOOSE
framework and module objects (PorousFlow, FluidProperties, solid mechanics). Supplementary Section "Thermo-hydro-
mechanical reference model" of the paper lists the equations and parameters.

The design-benchmark FE runs of the paper (operating-design validation) pinned exactly this template and
`generate_dataset_v3.py` (`9e0acc29...`), physics version `v4-prodbhp-variable-r5`; two anchor cases reproduced the
archived corpus fields to field RMSE of about 1e-6 K and 3e-7 MPa (gate 0.05 K / 0.01 MPa) before production runs.

## Generator

| File | Role |
|---|---|
| `generate_dataset_v4.py` | time-resolved generator (`--mode main|sweep|truth`, sharded, resumable); r5 physics with sampled producer bottom-hole pressure |
| `generate_dataset_v3.py` | pipeline library imported by v4 (scenario seeding, field construction, screening, case writing, collection) |
| `build_prior_field.py` | mean log-permeability `M` from DFN trace images (heat-map structure recipe) |
| `confidence_model.py` | spatial confidence `C` and its mapping to the local standard deviation `S` |
| `injectivity_screen.py` | Darcy-proxy injectivity screen before FE solves |
| `generate_uq_dataset.py` | field I/O helpers (MOOSE `PiecewiseMultilinear` permeability files, resampling) |

Per scenario the generator samples geology, confidence, controls and permeability members
`m = M + D_S L z` (`L L^T = R_l`, squared-exponential correlation, `l = 120` m), writes the MOOSE inputs, runs the
solver and stores `Delta T` and `Delta p` at eleven times plus well curves.

## Requirements and paths

* A MOOSE application with the PorousFlow, FluidProperties and solid-mechanics modules (the corpus used an
  application binary named `cavity-opt` built with all modules), MPI, Python with NumPy, SciPy, Pillow and
  `netCDF4`.
* DFN trace images `nf_slice_XXXX.png` and the matching Gmsh meshes `mesh_XXXX.msh` (generated upstream with
  DFN.lab; to be distributed with the data release).
* Set `GEO_IMG_DIR`, `GEO_MESH_DIR`, `GEO_BIN` (MOOSE executable) and `GEO_MPIEXEC` before running; the hard-coded
  defaults in `generate_dataset_v3.py`, `build_prior_field.py` and `generate_uq_dataset.py` refer to the original
  workstation.

## Provenance limits

* The corpus combines several generation campaigns and physics versions recorded in its statistics
  (`v4-prodbhp-variable-r5` and the depth family `v4-coussy-depth-r6d`). Per-record generator versions will be
  documented in the data release.
* The cluster copy of `generate_dataset_v4.py` pinned by the design-benchmark runs (`9261f5e1...`) is not in the
  public archive; the copy here is the latest committed version (`b8eea882...`). The template and the v3 pipeline
  library match the pinned hashes exactly.
* The exact MOOSE build (framework commit, compiler, dependencies) of the full corpus was not recorded.
