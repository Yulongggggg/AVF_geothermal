# Changelog

## 1.0.0 (2026-10-05)

First public release accompanying the manuscript *Joint operator learning for geothermal forecasting and
inference under uncertainty* (revision of 5 October 2026).

* `geouno` package: GeoUNO operator (POD1024 + 64 learned coordinate functions per field), physical input
  encoding, static Gaussian conditioning, posterior, MAP, ES-MDA, NUTS, forecast scores, monitoring geometries and
  A-optimal design, POD fitting and transfer training, inverse-prior fit; `geouno` command-line interface.
* Release assets `artifacts-v1`: checkpoint `ed043489...`, inverse-prior factor, 24 test reservoirs and one
  validation pilot, six-reservoir NUTS chains, prior-fit inputs, transfer parent `cf278e64...`.
* Frozen protocols, per-task reference scores (1,206 records) and Source Data of the manuscript tables.
* Provenance: original sources with hashes, archived NUTS runtime, bitwise equivalence verifier, release assembly.
* Static-only forecasts use the corrected member order (2 October 2026 audit).
