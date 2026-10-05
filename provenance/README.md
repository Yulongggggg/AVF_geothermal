# Provenance material

| Path | Content |
|---|---|
| `original_sources/` | the original research modules from which `geouno` was transcribed |
| `SOURCES.json` | origin (repository, commit, path) and SHA-256 of each original file, and whether the hash equals one pinned by the paper's protocols or runtime records |
| `nuts_bundle_runtime/` | portable runtime of the frozen NUTS bundle (`bundle.json`, `src/`), the reference implementation for the equivalence check |
| `verify_equivalence.py` | bitwise comparison of `geouno` with the archived code on the released data |
| `equivalence_report.json` | result of the last run of `verify_equivalence.py` |
| `assemble_release.py` | how the release assets, resources, protocols and reference scores were built |

See [docs/PROVENANCE.md](../docs/PROVENANCE.md).
