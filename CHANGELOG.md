# Changelog

## Unreleased (2026-10-08)

* Require a matching completion record and all four output hashes before skipping a batch task.
* Reject invalid shard and monitoring-configuration inputs instead of silently doing no work.
* Make NUTS verification check artifact hashes, report explicit pass/fail comparisons and return failure on
  mismatches or failed convergence diagnostics.
* Clarify public-access status, asset-free table checks and the 1,153 monitoring case files produced by the
  current registered configurations.

## Review corrections (2026-10-07)

* Use the geothermal operator name while retaining checkpoint/import compatibility.
* Package Source Data in installed wheels and update Table 1 to all three independent GeoUNO seeds.
* Validate extracted artifacts, MAP restart identity and training-resume source/runtime identity.
* Separate inference and forecasting time and evaluation counts.

## 1.0.0 (2026-10-05)

Initial code and artifact snapshot accompanying the manuscript *Joint operator learning for geothermal forecasting and
inference under uncertainty* (revision of 5 October 2026).

* `geouno` package: GeoUNO operator (POD1024 + 64 learned coordinate functions per field), physical input
  encoding, static Gaussian conditioning, posterior, MAP, ES-MDA, NUTS, forecast scores, monitoring geometries and
  A-optimal design, POD fitting and transfer training, inverse-prior fit; `geouno` command-line interface.
* Release assets `artifacts-v1`: checkpoint `ed043489...`, inverse-prior factor, 24 test reservoirs and one
  validation pilot, six-reservoir NUTS chains, prior-fit inputs, transfer parent `cf278e64...`.
* Frozen protocols, per-task reference scores (1,206 records) and Source Data of the manuscript tables.
* Provenance: original sources with hashes, archived NUTS runtime, bitwise equivalence verifier, release assembly.
* Static-only forecasts use the corrected member order (2 October 2026 audit).
