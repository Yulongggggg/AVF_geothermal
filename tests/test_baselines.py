"""Architecture dimensions, training gradients and baseline protocol isolation."""

import json

import pytest
import torch

from geouno.baselines.models import BASELINES, build
from geouno.baselines import protocol
from geouno.baselines.table_metrics import summarize
from geouno.training.loss import objective
from geouno.training.synthetic import make_synthetic_pack


PARAMETERS = {
    "deeponet": 1422082,
    "fno": 3152578,
    "ufno": 3936258,
    "unet": 1546146,
    "resnet": 593794,
    "attention_unet": 2413698,
}


@pytest.mark.model_execution
@pytest.mark.parametrize("kind", BASELINES)
def test_published_baseline_dimensions_and_training_gradient(kind):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(20260915)
        model = build(kind, [11.32, 7.99], 1.6)
        assert sum(p.numel() for p in model.parameters()) == PARAMETERS[kind]
        fields = torch.randn(2, 6, 160, 128)
        controls = torch.randn(2, 5)
        prediction = model(fields, controls)
        assert prediction.shape == (2, 2, 160, 128)
        assert torch.isfinite(prediction).all()
        loss, _ = objective(
            prediction,
            torch.randn_like(prediction),
            [2],
            model.scales,
            model.pressure_residual_scale,
            weight=0.2,
            kind="rmse",
        )
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert sum(float(p.grad.abs().sum()) for p in model.parameters()) > 0
    finally:
        torch.set_num_threads(previous)


def test_baseline_protocol_separates_models_and_rejects_changed_data(tmp_path):
    pack = make_synthetic_pack(tmp_path / "pack", train=1, val=1, members=1)
    protocol.configure(pack, tmp_path / "runs")
    a, a_digest = protocol.freeze("unet", 20260915, smoke=True)
    b, b_digest = protocol.freeze("attention_unet", 20260915, smoke=True)
    assert a != b and a_digest != b_digest
    assert protocol.freeze("unet", 20260915, smoke=True) == (a, a_digest)
    stats = json.loads((pack / "stats.json").read_text())
    stats["fields_std"][0] *= 2
    (pack / "stats.json").write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="stats_sha256"):
        protocol.freeze("unet", 20260915, smoke=True)


def test_relative_l2_pools_time_before_averaging_scenarios():
    records = []
    for case, multiplier in (("a", 1), ("b", 2)):
        for ti, error, truth in ((0, 3, 4), (10, 4, 3)):
            row = dict(scenario=case, time_index=ti)
            for field in ("T", "P"):
                row.update(
                    {
                        field + "_field_rmse": error * multiplier,
                        field + "_truth_rms": truth,
                        field + "_nearwell_rmse": 2 * error * multiplier,
                    }
                )
            records.append(row)
    values = summarize(records)
    assert values["all_T_relative"] == pytest.approx(1.5)
    assert values["ten_T_relative"] == pytest.approx(2)
    assert values["all_T_field_rmse"] == pytest.approx(5.25)
    assert values["all_T_nearwell_rmse"] == pytest.approx(10.5)
