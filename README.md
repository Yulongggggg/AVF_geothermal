# GeoUNO model and inference assets

Git LFS assets for the final resid64_p1 model and saved application results.

Checkpoint: f39894ff83e04f47b2a9098f0e387540e1390cf12b9d7f8726dc815dffd16dbb.

The 24 `nuts_test_*_final_20261015.tar` files contain the four original NUTS chains for each
reservoir, with 512 warm-up iterations and 1,024 retained draws per chain, together with
the saved forecasts, scores and diagnostics. Each archive contains 20 files; the four
individual chain arrays provide the complete draws. The original test_16 diagnostics
are retained with its reported convergence flag.

Use Git LFS to download this branch, then `geouno download --from-dir /path/to/this/checkout`.
The code and detailed manifest are on the release-v1.0.0 branch. Weights and data: CC BY 4.0.
