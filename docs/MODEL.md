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

The three Supplementary Table 1 seeds are 20260915, 20260925 and 20260926. Each uses 99 main epochs, 44 temperature-only
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

## Predicting from your own permeability map

[`examples/predict_fields.py`](../examples/predict_fields.py) reads a 128 × 128 log-permeability map and
physical controls from an NPZ file. Its input keys are:

| Key | Shape | Units or meaning |
|---|---|---|
| `logk` | (128, 128) | `log10(k)` for permeability in m², over x and y in [−400, 400] m |
| `producer_xy_m` | (2,) | Producer x, y in metres; the injector is fixed at (0, −250) m |
| `q_kg_s` | scalar | Prescribed circulation rate, kg/s |
| `T_inj_K` | scalar | Injection temperature, K |
| `p_prod_MPa` | scalar | Absolute producer bottom-hole pressure, MPa |
| `depth_km` | scalar | Reservoir-center depth, km |
| `times_s` | (number of times,) | Positive forecast times, seconds |

For example, to combine a permeability map saved as `my_logk.npy` with specified operating controls:

```python
import numpy as np

np.savez("my_reservoir.npz", logk=np.load("my_logk.npy"), producer_xy_m=[0., 250.],
         q_kg_s=3.5, T_inj_K=303., p_prod_MPa=20., depth_km=3.06,
         times_s=[32183000., 126983000., 316000000.])
```

From the repository root, run:

```bash
geouno download
python examples/predict_fields.py --input my_reservoir.npz --out my_forecast.npz --threads 4
```

The output contains `cooling_K` and `pressure_change_MPa`, each with shape
`(number of times, 160, 128)`, together with times and grid coordinates. These are changes from the
initial fields, with the cooling sign convention defined above. The example uses the paper's reservoir
geometry and trained application checkpoint; [DATA.md](DATA.md) describes the grid and controls.
