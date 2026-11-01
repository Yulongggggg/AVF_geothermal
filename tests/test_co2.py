"""CO2 masking, physical pressure units and the widened joint-field model."""

import json

import numpy as np
import pytest
import torch
from torch import nn

from geouno.co2.data import PRESSURE_OFFSET_BAR, PRESSURE_SCALE_BAR, Samples, normalize_pressure
from geouno.co2.evaluate import evaluate
from geouno.co2.models import PAPER_MODELS, UFNOSimpleHead
from geouno.co2.train import masked_rmse


def test_masked_training_loss_and_pooled_physical_error():
    target = torch.ones(2, 2, 2, 2)
    mask = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]], [[[1.0, 1.0], [1.0, 1.0]]]])
    prediction = target + torch.tensor([2.0, 4.0])[:, None, None, None]
    prediction[0, :, 0, 1] = 1000.0  # outside the reservoir
    assert masked_rmse(prediction, target, mask).tolist() == pytest.approx([3.0, 3.0])
    fields = torch.zeros(2, 8, 2, 2)
    fields[:, 1:2] = mask
    fields[:, 6:8] = prediction

    class PredictionFromInputs(nn.Module):
        def forward(self, field, scalar):
            return field[:, 6:8]

    data = [(fields[i], torch.zeros(6), target[i], i, 0) for i in range(2)]
    result = evaluate(PredictionFromInputs(), data, "cpu", 2, [1.0, 1.0], 0)
    expected = np.sqrt((4.0 + 4 * 16.0) / 5)
    assert result["rmse"] == pytest.approx([expected, expected])
    assert result["rel_l2"] == pytest.approx([3.0, 3.0])
    assert result["physical_rmse"] == pytest.approx([expected, expected * PRESSURE_SCALE_BAR])
    assert result["relative_l2_pressure_target"] == "offset-normalized pressure"


def test_pressure_normalization_and_case_time_mapping(tmp_path):
    physical = np.array([PRESSURE_OFFSET_BAR, PRESSURE_OFFSET_BAR + PRESSURE_SCALE_BAR])
    assert normalize_pressure(physical) == pytest.approx([0.0, 1.0])
    np.save(tmp_path / "test_fields.npy", np.zeros((2, 8, 96, 200), np.float32))
    np.save(tmp_path / "test_scalars.npy", np.arange(10, dtype=np.float32).reshape(2, 5))
    target = np.zeros((2, 24, 2, 96, 200), np.float32)
    target[1, 23] = 9
    np.save(tmp_path / "test_targets.npy", target)
    np.save(tmp_path / "time_grid.npy", np.logspace(0, 2, 24, dtype=np.float32))
    stats = {"fields_std": [1.0, 2.0], "dp_test_normalized": True}
    (tmp_path / "stats.json").write_text(json.dumps(stats))
    samples = Samples(tmp_path, "test", times=[23, 0], cases=[1, 0])
    assert samples.index == [(1, 23), (1, 0), (0, 23), (0, 0)]
    field, scalar, value, case, ti = samples[0]
    assert field.shape == (8, 96, 200)
    assert scalar.tolist() == pytest.approx([5, 6, 7, 8, 9, 1])
    assert (case, ti) == (1, 23) and torch.all(value == 9)
    stats["dp_test_normalized"] = False
    (tmp_path / "stats.json").write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="normalization"):
        Samples(tmp_path, "test")


@pytest.mark.model_execution
def test_co2_joint_model_uses_widened_inputs_and_training_gradients():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(19)
        net = UFNOSimpleHead([0.18, 1.35], 1.15, pod_rank=4, learned_rank=4)
        field = torch.randn(2, 8, 96, 200)
        field[:, 1] = 1
        prediction = net(field, torch.randn(2, 6))
        assert prediction.shape == (2, 2, 96, 200)
        assert torch.isfinite(prediction).all()
        masked_rmse(prediction, torch.randn_like(prediction), field[:, 1:2]).sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        assert net.enc[0].conv1.weight.grad[:, 6:8].abs().sum() > 0
        assert len(PAPER_MODELS) == 8 and "attention_unet" not in PAPER_MODELS
    finally:
        torch.set_num_threads(previous)
