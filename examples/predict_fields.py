"""Predict cooling and pressure change from a permeability map and physical controls."""

import argparse
from pathlib import Path

import numpy as np
import torch

from geouno import artifacts
from geouno.inputs import field_on_grid
from geouno.model import configure_inference, load_geouno


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="NPZ with the physical inputs listed in docs/MODEL.md")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", help="Directory containing the downloaded model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as values:
        logk = np.asarray(values["logk"], dtype=np.float32)
        producer = np.asarray(values["producer_xy_m"], dtype=np.float32)
        times = np.asarray(values["times_s"], dtype=np.float64)
        controls = np.asarray(
            [values[name].item() for name in ("q_kg_s", "T_inj_K", "p_prod_MPa", "depth_km")], dtype=np.float32
        )
    if logk.shape != (128, 128) or producer.shape != (2,) or times.ndim != 1 or not len(times):
        raise ValueError("Expected logk[128,128], producer_xy_m[2] and a nonempty times_s vector")
    if not all(np.isfinite(value).all() for value in (logk, producer, times, controls)):
        raise ValueError("Every physical input must be finite")
    if np.any(times <= 0) or controls[0] <= 0 or controls[3] <= 0:
        raise ValueError("Times, circulation rate and depth must be positive")
    configure_inference(args.threads)
    net = load_geouno(
        artifacts.path("model/geouno_resid64_p1_f39894ff.pt", args.data), device=args.device
    )
    y, x = torch.meshgrid(
        torch.linspace(-500, 500, 160, device=args.device),
        torch.linspace(-400, 400, 128, device=args.device),
        indexing="ij",
    )
    permeability = torch.as_tensor(logk, device=args.device)[None, None]
    field = torch.stack(
        [
            field_on_grid(permeability)[0, 0] + 13.5,
            (y.abs() <= 400).float(),
            x / 400,
            y / 500,
            torch.hypot(x - float(producer[0]), y - float(producer[1])) / 400,
            torch.hypot(x, y + 250) / 400,
        ]
    )[None]
    q, tinj, pressure, depth = controls
    predictions = []
    with torch.inference_mode():
        for time in times:
            scalar = torch.tensor(
                [[q / 3.5, (tinj - 303) / 10, (pressure - 20) / 10, (np.log10(time) - 6) / 2,
                  (depth - 3.06) / 0.6]], dtype=torch.float32, device=args.device
            )
            predictions.append(net(field, scalar)[0].cpu().numpy())
    predictions = np.stack(predictions)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, cooling_K=predictions[:, 0], pressure_change_MPa=predictions[:, 1], times_s=times,
        x_m=x[0].cpu().numpy(), y_m=y[:, 0].cpu().numpy()
    )
    print(f"Saved {len(times)} joint field predictions to {out}")


if __name__ == "__main__":
    main()
