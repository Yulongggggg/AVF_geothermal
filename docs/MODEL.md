# The final GeoUNO operator

`geouno.model.GeoUNO` is the `resid64_p1` model used in the 15 October 2026 results.
The application checkpoint is `geouno_resid64_p1_f39894ff.pt` (seed 20260915).

## Inputs and outputs

Six fields on the 160 × 128 grid encode `log10(k) + 13.5`, the reservoir mask, `x/400`, `y/500`,
and producer/injector distances divided by 400 m. Five scalars encode `q/3.5`, `(T_inj − 303)/10`,
`(p_prod − 20)/10`, `(log10(t) − 6)/2` and `(depth − 3.06)/0.6`, with physical units kg/s, K, MPa, s and km.
The outputs are cooling `ΔT` in K and pressure change `Δp` in MPa, on the reference grid or at query coordinates.

## Architecture

Four scalar-conditioned encoder blocks (32, 64, 128 and 192 channels), two transformer layers and separate
field decoders retain layerwise FiLM modulation. Each field combines 1,024 fixed training-fitted POD functions
with 64 learned coordinate functions. POD coefficients are projections of the centered decoder output after
subtracting its mean shape. Learned coefficients come from a least-squares projection of the residual after
subtracting the POD contribution. The coordinate functions are normalized on the reference grid; the solve
adds `1e-6 I` to the Gram matrix and uses FP64 before returning FP32 coefficients.

The pressure-level input has 657 components in this order: mean-pooled features (192), max-pooled features
(192), scalars (5), pressure-grid features (256), and P1 well-permeability summaries (12). The level MLP is
657 → 256 → 128 → 1. P1 uses three Gaussian radii (6, 24, 96 m) around each well and two statistics per radius.
There is no coefficient-correction MLP or P0 regional feature extractor. The model has **3,520,755 trainable
parameters**, with POD functions held as fixed buffers.

See [`final.py`](../geouno/model/final.py) for the exact residual projection, and
[`geouno.py`](../geouno/model/geouno.py) for the shared coefficient–basis query implementation.

## Training and loading

The three Table 1 seeds are 20260915, 20260925 and 20260926. Each uses 99 main epochs, 44 temperature-only
epochs and 22 joint epochs; application experiments use seed 20260915 from this same comparison.
[TRAINING.md](TRAINING.md) gives the complete protocol.

```python
from geouno.model import load_geouno
net = load_geouno('artifacts/model/geouno_resid64_p1_f39894ff.pt')
fields = net(field, scalar)                         # [B, 2, 160, 128]
values = net(field, scalar, query=xy)               # [B, 2, N]
out, parts = net(field, scalar, return_parts=True)  # evaluation mode
```

`load_geouno` verifies SHA-256 and loads every tensor strictly. Final evaluation uses FP32 with TF32 disabled;
the inverse backend uses FP64 likelihood, covariance and latent coordinates. The backend applies the fixed
pressure pool and the model's observation-basis calculation.
