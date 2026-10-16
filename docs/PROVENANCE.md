# Final-model release provenance

The 15 October 2026 update corresponds to the simplified `resid64_p1` paper model and its saved application
experiments. The application checkpoint is final seed 20260915, SHA-256
`f39894ff83e04f47b2a9098f0e387540e1390cf12b9d7f8726dc815dffd16dbb`.

## Numerical implementation

The original final `model.py`, `train.py` and `evaluate.py` are preserved under
[`original_sources/final_geouno_20261015`](../provenance/original_sources/final_geouno_20261015).
The portable package changes imports, filesystem paths, CLI arguments and the constructor's default variant
(from the research default to the selected `resid64_p1`). It preserves the model arithmetic, least-squares
solve, training losses, learning-rate schedules and stage selection.

```bash
python provenance/verify_final_sources.py
```

This checks nine definitions against the original source using their syntax trees, normalizing the public
class name, constructor default and docstrings. It executes neither training nor prediction. Strict checkpoint
loading separately confirms tensor compatibility and 3,520,755 parameters.

## Data and scores

- New calibration, optimized layouts, monitoring configurations and reservoir observation arrays come from
  `operator_final_20261015`.
- Final NUTS artifacts contain six reservoirs × four completed chains and their saved diagnostics/forecasts.
- The geological truth, prior means and base FE inputs are unchanged; observation locations use the new design.
- `paper-numbers` recomputes 170 printed entries with zero mismatches from saved records. It validates the
  three final GeoUNO seeds and their scenario/time/member identities. The final application seed is included.
- Baseline rows retain their approved results. The common 99-epoch appendix table includes all six models, each with three completed seeds. CO₂ tables contain all three completed seeds at each POD rank.
- File hashes are in `geouno/resources/source_data/SHA256SUMS` and `release_manifest.json`.

The final large assets are stored on the Git LFS branch `artifacts-final-20261015`; the unchanged prior-fit
inputs remain in the earlier `artifacts-v1` release. Downloads use each asset's manifest URL and SHA-256.

## Validation scope

This release update uses saved experimental outputs. Its validation comprises source comparison, strict model
loading, artifact integrity, observation-array regeneration, numerical aggregation, portable-resource checks
and unit tests that do not execute neural predictions. It makes no new cross-device prediction-equivalence
claim. The dated [7 October review](REVIEW_20261007.md) and `verify_equivalence.py` apply to the earlier code
revision and its artifacts.

The original prior and inference algorithms remain in `geouno/inverse`; monitoring geometry, coordinate-keyed
noise and the common observation model remain in `geouno/monitoring`. Their scientific definitions are unchanged.
Component licences are listed in [THIRD_PARTY.md](../THIRD_PARTY.md).
