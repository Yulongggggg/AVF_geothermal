"""Conditional sampling and checkpoint persistence for supplementary comparisons."""

from pathlib import Path

import numpy as np
import pytest
import torch

from comparisons.pcno.model import PCNOSpatialBackbone, load as load_pcno
from comparisons.pno.model import Predictor, energy_per_sample, load as load_pno
from geouno.training.synthetic import make_synthetic_pack


def test_comparison_checkpoint_roundtrips_preserve_fields_and_input_gradients(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    models = [
        (PCNOSpatialBackbone([11.0, 8.0], 1.6, width=(8, 8), modes=(2, 2)), load_pcno, 6),
        (Predictor(), load_pno, 8),
    ]
    for index, (model, load, channels) in enumerate(models):
        model.eval()
        path = tmp_path / f"{index}.pt"
        state = dict(model=model.state_dict())
        state.update(model_config=model.config) if index == 0 else state.update(kind="pno")
        torch.save(state, path)
        restored = load(path)
        field = torch.randn(1, channels, 32, 32)
        controls = torch.randn(1, 5)
        result = []
        for net in (model, restored):
            x = field.clone().requires_grad_()
            torch.manual_seed(91)
            output = net(x, controls)
            gradient = torch.autograd.grad(output.square().mean(), x)[0]
            result.append((output, gradient, torch.get_rng_state()))
        assert all(torch.equal(a, b) for a, b in zip(*result))


def test_pno_energy_score_uses_off_diagonal_draw_pairs():
    torch.manual_seed(2)
    pred = torch.randn(2, 4, 2, 3, 5, requires_grad=True)
    truth = torch.randn(2, 2, 3, 5)
    dimension = truth[0].numel()
    expected = []
    for p, y in zip(pred, truth):
        first = sum(torch.linalg.vector_norm(draw - y) for draw in p) / 4
        pair = sum(torch.linalg.vector_norm(p[i] - p[j]) for i in range(4) for j in range(4) if i != j) / 12
        expected.append((first - 0.5 * pair) / dimension**0.5)
    expected = torch.stack(expected)
    actual = energy_per_sample(pred, truth)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), pred, retain_graph=True)[0], torch.autograd.grad(expected.sum(), pred)[0]
    )


def pno_pack(directory):
    pack = make_synthetic_pack(directory, train=1, val=1, members=1)
    n = len(np.load(pack / "real.npy", mmap_mode="r"))
    inputs = np.empty((n, 2, 128, 128), dtype=np.float32)
    inputs[:, 0] = -13.2
    inputs[:, 1] = 0.7
    np.save(pack / "inputs.npy", inputs)
    np.save(pack / "index.npy", np.arange(n))
    return pack


class SmallStochasticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, field, scalar):
        return self.weight * (field[:, :2] + torch.randn_like(field[:, :2]) * 0.01)


def test_pno_first_selected_checkpoint_survives_interrupted_optimizer_write(tmp_path, monkeypatch):
    from comparisons.pno import train

    pack = pno_pack(tmp_path / "pack")
    monkeypatch.setattr(train, "Predictor", SmallStochasticModel)
    scores = iter([1.0, 2.0, 2.0])
    monkeypatch.setattr(train, "validation", lambda *args: {"selection_score": next(scores)})
    original = train.atomic_checkpoint

    def stop(value, path):
        if Path(path).name == "last.pt":
            raise InterruptedError("after selected checkpoint")
        original(value, path)

    monkeypatch.setattr(train, "atomic_checkpoint", stop)
    args = ["--pack", str(pack), "--out", str(tmp_path / "run"), "--device", "cpu", "--smoke"]
    with pytest.raises(InterruptedError):
        train.main(args)
    best = tmp_path / "run/smoke/pno_s20260913/best.pt"
    original_bytes = best.read_bytes()
    monkeypatch.setattr(train, "atomic_checkpoint", original)
    train.main(args)
    assert best.read_bytes() == original_bytes
    train.main(args)
    assert best.read_bytes() == original_bytes
