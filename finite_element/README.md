# Finite-element example

This component reproduces the FE realization used as `test_01` in the released inverse problems:
`v4m02656_r012`, row 39801 of the simulation pack. It includes the actual triangular mesh,
well completion, coupled THM input, permeability input and saved FE fields and well curves.
The model uses MOOSE PorousFlow, Water97 fluid properties, thermo-poroelastic porosity and
Kozeny–Carman reservoir permeability. The mesh has 52,845 nodes and 105,260 elements.

The calculation spans ten years. Its reference archive contains ten intermediate temperature/pressure
snapshots, the final fields, and 33 samples of seven well quantities. Combining `snap_fields` and
`fields` reproduces the corresponding eleven-time pack row exactly. Field channels are cooling in K
and pressure change in MPa on the 160 × 128 output grid.

## Solver and environment

The recorded calculation uses MOOSE commit
[`381947fe984564cf250e888a4689a8710902152e`](https://github.com/idaholab/moose/tree/381947fe984564cf250e888a4689a8710902152e)
with `moose-dev=2026.06.16`, PETSc 3.25.2, libMesh `2026.06.05_ab36c00` and MPICH 5.0.1.
The original `cavity` application source is in `application/`. Its local repository did not have a
commit, so `provenance.json` identifies each source file by SHA-256. The application registers the
standard MOOSE modules; its included finite-volume constraint is unused by this THM input.
MOOSE itself remains an external dependency.

Use the matching MOOSE build environment, then build the supplied application against that framework:

```bash
git clone https://github.com/idaholab/moose.git moose
git -C moose checkout 381947fe984564cf250e888a4689a8710902152e
git -C moose submodule update --init --recursive
export MOOSE_DIR="$(pwd)/moose"
make -C finite_element/application -j8
```

The Python collector requires NumPy, SciPy, Pillow and netCDF4. `environment.json` records the versions
used in the completed example. The supplied application must be built with its compatible MOOSE/MPI
dependencies. The example reported below used the existing original executable; it did not require a
new framework build.

## Run the stored case

From the repository root, with the solver environment active:

```bash
python finite_element/run.py \
  --executable /path/to/cavity-opt --ranks 6 --out fe-test-01
```

For an existing Slurm allocation, add `--launcher "srun --mpi=pmi2"`. All new inputs and outputs go
to the new `--out` directory. The command verifies the supplied file hashes, runs the solver, applies
the original collector and writes `run.json` with per-array comparisons against `reference/fields.npz`.
`--prepare-only` writes the isolated input and prints the command without starting the solver.

The recorded six-process CPU rerun through `run.py` took 338 seconds, including collection. Every stored array, including all
snapshots, final fields and well curves, matched the original byte for byte; the complete compressed
archive also had the same SHA-256. See `reference/reproduction.json`. This verification covers this
one realization; the 95,966-record corpus is distributed separately.

## Reconstruct the permeability input

`case/permeability.i` contains the original algorithm's 400 × 400 permeability input in m². It was
reconstructed using the original characterization seed, screening draw and sampling covariance.
The reconstructed characterization and latent draw match the stored case exactly, and the solver
rerun reproduces its FE outputs. The original permeability text was removed by the production
collector, so its historical file checksum is unavailable.

The fixed-case run above needs no additional data. To repeat the seed reconstruction, obtain the
separate FE sampling factor (134,217,856 bytes):

```bash
geouno download --group fe
geouno verify --group fe
python finite_element/reconstruct.py \
  --factor artifacts/finite_element/fe_sampling_cholesky_fdc8c5e.npy \
  --out reconstructed-permeability.i
```

The example path assumes the default artifact directory in a source checkout. If `GEOUNO_DATA` is
set, use that directory instead. This factor belongs to FE realization generation and differs from
the inverse-prior factor. Its SHA-256 is
`fdc8c5e1ed73f3e25b6564ad679beb6d50c738da439a9631dbe1f39c06cacf11`.

`generation/` retains the original numerical routines used by these entry points. `packing/` records
the original snapshot assembly and realized-permeability packing routines. Historical comments and
documentation strings were removed, and filesystem defaults were made relative; `provenance.json`
records the original source hashes and verifies unchanged executable syntax before those path
substitutions. The full corpus-generation workflow also requires its scenario assets, sampling factor
and production manifests; it is separate from this fixed-case example.

To reproduce the published model training after packing, use the released `stats.json`. It preserves
normalization fitted to the preceding pack's training split; every contributing case remains in the
current training split. The packing routines calculate new statistics for their input corpus, so those
new values do not replace the frozen normalization used by the reported models.

## Licences

Source code in this component, including the supplied MOOSE application, is distributed under
[LGPL 2.1](LICENSE). The mesh, fracture image, numerical input and reference data are the authors'
data under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The GeoUNO model package's MIT
licence does not replace the LGPL terms of this component.
