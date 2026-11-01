# Source provenance and numerical verification

The application model is the refined `resid64_p1` model, seed 20260915, with checkpoint SHA-256
`f39894ff83e04f47b2a9098f0e387540e1390cf12b9d7f8726dc815dffd16dbb`.
Its three training stages use 99, 44 and 22 epochs. Main Table 1 compares the models after the common
99-epoch stage; Supplementary Table 1 reports the refined GeoUNO results and the earlier baseline runs.

## Implementations

The archived research modules and their origins are listed in
[`provenance/SOURCES.json`](../provenance/SOURCES.json). The original final GeoUNO model and trainer are in
[`final_geouno_20261015`](../provenance/original_sources/final_geouno_20261015).
The portable package retains their model arithmetic, losses, learning-rate schedules and validation criteria.
Filesystem paths and entry points are configurable.

```bash
python provenance/verify_final_sources.py
pytest -q tests/test_training_resume.py tests/test_baselines.py
```

The source comparison checks eight model and training definitions. Restart handling is tested separately:
the best checkpoint survives interruption, and resumes require matching data and protocol metadata.
Baseline implementations accept the original research checkpoints with strict state loading.

## Data and experiment identity

The checkpoint, prior factor, calibration, observation layouts and reservoir cases are identified by
[`release_manifest.json`](../geouno/resources/release_manifest.json),
[`cases.json`](../geouno/resources/cases.json) and the packaged configuration files.
Original source files and experiment outputs retain their recorded hashes.

Source Data contains scenario/time scores, seed-level comparison values, all 24 inference reservoirs,
monitoring results, operating-design results and computational costs. The
[Source Data index](../geouno/resources/source_data/README.md) maps these files to the current manuscript.
`geouno paper-numbers` recalculates the supported table entries from those records.

Model predictions and inverse experiments are reproduced with the commands in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). Code, model and data licences are specified in
[LICENSE](../LICENSE), [THIRD_PARTY.md](../THIRD_PARTY.md) and [DATA.md](DATA.md#licence).
