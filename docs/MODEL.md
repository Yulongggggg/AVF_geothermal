# The GeoUNO operator

`geouno.model.GeoUNOBasisHead` (checkpoint `geouno_pod1024_ed043489.pt`, SHA-256
`ed04348969fe27903bc9bd084781164f65c610c7e46da0ff75bee60553fdc18e`).

## Inputs and outputs

* Six fields on the 160 x 128 grid: `m + 13.5` with `m = log10 k`, reservoir mask, `x/400`, `y/500`,
  producer distance / 400 m, injector distance / 400 m.
* Five controls: `q/3.5`, `(T_inj - 303)/10`, `(p_prod - 20)/10`, `(log10 t - 6)/2`, `(depth - 3.06)/0.6`, with
  injection rate `q` [kg/s], injection temperature [K], absolute producer bottom-hole pressure [MPa], time [s] and
  depth [km].
* Output: `Delta T` [K] and `Delta p` [MPa] on the 160 x 128 grid, or at arbitrary query points
  `(x/400, y/500)` via `model(field, scalar, query=xy)`.

Mean field and confidence of the characterization determine the permeability draws; they are not network inputs.

## Architecture

```
controls --MLP--> c (128)
fields --CondBlock x4 (32, 64, 128, 192 ch, avg-pool between)--> 2-layer transformer on the 20 x 16 grid
       --> separate temperature and pressure decoders (CondBlocks with skips, back to 160 x 128)
       --> 1x1 heads: raw T, raw p fields
features     = [pooled encoder (384), well summaries (12), controls (5), 4x4 pressure grid (256), regional readout (112)]
coefficients a = <raw - mean(raw) - mu, POD> + delta(features)      (1,024 POD + 64 learned per field; delta: MLP on 769 features)
levels       b_T = s_T * spatial mean of raw T;  b_p = s_p * MLP(features)
u_l(x)       = b_l + s_l' [ mu_l(x) + sum_r a_lr psi_lr(x) ]          (s_T' = s_T; s_p' = pressure residual scale)
```

* `CondBlock`: two 3x3 convolutions with group normalization, FiLM modulation by `c` (tanh-bounded gain), residual.
* POD functions: 1,024 per field, fitted on training snapshots only (randomized SVD), orthonormal under the
  pixel-mean inner product; bilinear interpolation extends them to query points.
* Learned functions: 64 per field from coordinate MLPs (temperature with Fourier features of 1-32 cycles),
  standardized over the sensor grid.
* Pressure level: an MLP on pooled encoder features, twelve well-neighbourhood permeability summaries (Gaussian
  radii 6, 24, 96 m around each well), the controls, a 4 x 4 pooled pressure grid and a learned regional readout.
* 4,309,756 trainable parameters; the POD basis (2 x 1,024 x 160 x 128) is a fixed buffer.

Exact definitions: `geouno/model/geouno.py` and its parents in `geology.py`, `joint.py`, `direct.py` and
`blocks.py`. Supplementary Section "Network architecture and training objective" of the paper gives the
equations.

## Training lineage

The selected model inherits every tensor of an earlier grid-output predictor (`cf278e64...`, 253 epochs) and
completes 132 transfer epochs: 11 coefficient-head warm-up epochs, 55 joint-transfer epochs and 66 fine-tuning
epochs from the validation-selected transfer checkpoint, with loss = equal-scenario paired RMSE + 0.2 x MMD^2.
The checkpoint of the final epoch (131, also the best validation score) is used for every application. See
[TRAINING.md](TRAINING.md) and `protocols/model_selection.json`.

## Numerical settings

All reported inference uses FP32 network evaluation with FP64 likelihood, covariance, latent variables and
gradients; TF32 disabled; PyTorch's fused attention fast path disabled; and the fixed 5 x 4 average pool in place
of the equivalent adaptive pool. `geouno.model.load_geouno` and `geouno.backend.Backend` apply these settings.
CPU results depend at round-off level on the thread count and processor.

## Loading

```python
from geouno.model import load_geouno
net = load_geouno('artifacts/model/geouno_pod1024_ed043489.pt')   # verifies SHA-256, loads with weights_only=True
fields = net(field, scalar)                                        # [B, 2, 160, 128]
values = net(field, scalar, query=xy)                              # [B, 2, N]
out, parts = net(field, scalar, return_parts=True)                 # coefficients, basis, mean shape, levels
```
