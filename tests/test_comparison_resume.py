"""Comparison runs retain selected weights and reject changed arrays or caches."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from geouno.baselines import protocol as baseline_protocol
from geouno.co2 import protocol as co2_protocol
from geouno.training.synthetic import make_synthetic_pack


@pytest.mark.parametrize("filename", ["real.npy", "fields.npy", "scal.npy"])
def test_baseline_freeze_rejects_replaced_array(tmp_path, filename):
    pack = make_synthetic_pack(tmp_path / "pack", train=1, val=1, members=1)
    baseline_protocol.configure(pack, tmp_path / "runs")
    baseline_protocol.freeze("unet", 20260915, True, "cpu")
    (pack / filename).write_bytes(b"replaced array")
    with pytest.raises(ValueError, match="pack_arrays"):
        baseline_protocol.freeze("unet", 20260915, True, "cpu")


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_co2_freeze_rejects_replaced_prepared_array(tmp_path, split):
    for name in ("stats.json", "time_grid.npy"):
        (tmp_path / name).write_text("{}")
    for dataset in ("train", "val", "test"):
        for kind in ("fields", "scalars", "targets"):
            (tmp_path / f"{dataset}_{kind}.npy").write_bytes(b"original array")
    args = SimpleNamespace(prepared=tmp_path, model="unet", seed=1, epochs=20, smoke=True,
                           batch=2, lr=3e-4, device="cpu", workers=0)
    first = co2_protocol.freeze(args, tmp_path / "run", None)
    assert co2_protocol.freeze(args, tmp_path / "run", None) == first
    (tmp_path / f"{split}_targets.npy").write_bytes(b"replaced target array")
    with pytest.raises(ValueError, match="prepared_arrays"):
        co2_protocol.freeze(args, tmp_path / "run", None)


class ZeroFields(torch.nn.Module):
    def forward(self, field, scalar):
        return field.new_zeros((len(field), 2, 160, 128))


@pytest.mark.parametrize("evaluator", ["evaluate", "table_metrics"])
@pytest.mark.parametrize("change", ["array", "source", "legacy"])
def test_baseline_evaluation_rejects_changed_inputs(tmp_path, monkeypatch, evaluator, change):
    module = importlib.import_module("geouno.baselines." + evaluator)
    pack = make_synthetic_pack(tmp_path / "pack", train=1, val=1, members=1)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_config": {"architecture": "fair_physical_baseline"}}, checkpoint)
    monkeypatch.setattr(module, "load", lambda _: ZeroFields())
    if evaluator == "evaluate":
        monkeypatch.setattr(module, "load_scope", lambda: {
            row: dict(high_pressure_training_p90=False, above_generator_envelope=False) for row in range(5)
        })
        path = tmp_path / "val.json"

        def run():
            return module.evaluate(checkpoint, "val", tmp_path, pack, "cpu", True)
    else:
        path = tmp_path / "table.json"

        def run():
            return module.evaluate(checkpoint, pack, path, "cpu", True, "val")
    result = run()
    assert result["status"] == "complete"
    if change == "array":
        values = np.load(pack / "fields.npy", mmap_mode="r+")
        values[:] += 1
        values.flush()
    elif change == "source":
        original_sha = module.sha
        monkeypatch.setattr(module, "sha", lambda p: "0" * 64 if Path(p).name == "models.py" else original_sha(p))
    else:
        result.pop("identity")
        path.write_text(json.dumps(result))
    monkeypatch.setattr(module, "load", lambda _: pytest.fail("Reject cache before evaluating weights"))
    with pytest.raises(ValueError, match="different inputs"):
        run()


@pytest.mark.parametrize("change", ["raw", "output", "legacy"])
def test_co2_conversion_marker_checks_sources_and_output(tmp_path, monkeypatch, change):
    module = importlib.import_module("geouno.co2.prepare")
    raw, prepared = tmp_path / "raw", tmp_path / "prepared"
    raw.mkdir()
    for name in ("sg_test_a.pt", "sg_test_u.pt", "dP_test_u.pt"):
        (raw / name).write_bytes(b"source")
    calls = []

    def convert(split, n, units):
        calls.append(split)
        for kind in ("fields", "scalars", "targets"):
            (prepared / f"{split}_{kind}.npy").write_bytes(b"converted array")
        return dict(n=n)

    monkeypatch.setattr(module, "convert", convert)
    args = ["--raw", str(raw), "--out", str(prepared), "--splits", "test"]
    module.main(args)
    module.main(args)
    assert calls == ["test"]
    if change == "raw":
        (raw / "dP_test_u.pt").write_bytes(b"new raw pressure")
    elif change == "output":
        (prepared / "test_targets.npy").write_bytes(b"new prepared pressure")
    else:
        (prepared / "test_conversion.json").write_text(json.dumps(dict(source_pressure_units="bar")))
    with pytest.raises(ValueError, match="changed"):
        module.main(args)
    assert calls == ["test"]


class TinyModel(torch.nn.Module):
    def __init__(self, co2=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.config = {}
        self.scales = [1., 1.]
        self.pressure_residual_scale = 1.
        self.co2 = co2

    def forward(self, field, scalar):
        shape = (len(field), 2, 2, 2) if self.co2 else (len(field), 1)
        return self.weight.expand(shape)


def stop_before_first_optimizer_checkpoint(module, monkeypatch):
    original = module.atomic_checkpoint

    def interrupted(value, path):
        if Path(path).name == "last.pt":
            raise InterruptedError("after selected weights, before optimizer checkpoint")
        return original(value, path)

    monkeypatch.setattr(module, "atomic_checkpoint", interrupted)
    return original


def test_baseline_first_epoch_selected_checkpoint_survives_interruption(tmp_path, monkeypatch):
    module = importlib.import_module("geouno.baselines.train")
    full_evaluate = importlib.import_module("geouno.baselines.evaluate")
    out = tmp_path / "run"
    (tmp_path / "stats.json").write_text("{}")
    (tmp_path / "meta.csv").write_text("row,scen,split\n0,train,train\n1,val,val\n")
    monkeypatch.setattr(module, "freeze", lambda *args: (out, "protocol"))
    monkeypatch.setattr(module, "build", lambda *args: TinyModel())

    class Data:
        def __init__(self, pack, split, *args):
            class Pairs(list):
                names = [split]
                selected = {split: [0]}
            self.pairs = Pairs([0])

        def __len__(self):
            return 1

        def __getitem__(self, item):
            return torch.ones(1, 1), torch.zeros(1, 1), torch.zeros(1, 1), 1

    monkeypatch.setattr(module, "ScenarioPairs", Data)
    monkeypatch.setattr(module, "objective", lambda pred, *a, **k:
                        (pred.square().mean(), {"loss": pred.square().mean()}))
    validation_scores = iter([1., .5, 1., .9, .9])

    def evaluate(*args, **kwargs):
        value = next(validation_scores)
        return dict(summary={key: dict(T_rmse=value, P_rmse=value) for key in ("10", "all_evaluated_times")})

    monkeypatch.setattr(module, "evaluate", evaluate)
    monkeypatch.setattr(full_evaluate, "evaluate", lambda *a, **k: {})
    original = stop_before_first_optimizer_checkpoint(module, monkeypatch)
    args = ["--model", "unet", "--pack", str(tmp_path), "--out", str(tmp_path), "--cpu", "--smoke"]
    with pytest.raises(InterruptedError):
        module.main(args)
    before = (out / "best.pt").read_bytes()
    assert not (out / "last.pt").exists()
    monkeypatch.setattr(module, "atomic_checkpoint", original)
    module.main(args)
    assert (out / "best.pt").read_bytes() == before
    last = torch.load(out / "last.pt", weights_only=False)
    assert last["best"] == .5


def test_co2_first_epoch_selected_checkpoint_survives_interruption(tmp_path, monkeypatch):
    module = importlib.import_module("geouno.co2.train")
    (tmp_path / "stats.json").write_text(json.dumps(dict(fields_std=[1., 1.], p_res_std=1.)))
    monkeypatch.setattr(module, "freeze", lambda *args: "protocol")
    monkeypatch.setattr(module, "build", lambda *args: TinyModel(co2=True))

    class Data:
        def __init__(self, prepared, split, **kwargs):
            self.split = split

        def __len__(self):
            return 4

        def __getitem__(self, item):
            return torch.ones(8, 2, 2), torch.zeros(6), torch.zeros(2, 2, 2), item, 0

    monkeypatch.setattr(module, "Samples", Data)
    validation_scores = iter([.25, .45])

    def evaluate(net, data, *args):
        value = .1 if data.split == "test" else next(validation_scores)
        return dict(rmse_scaled=[value, value])

    monkeypatch.setattr(module, "evaluate", evaluate)
    original = stop_before_first_optimizer_checkpoint(module, monkeypatch)
    args = ["--model", "unet", "--prepared", str(tmp_path), "--out", str(tmp_path), "--seed", "1",
            "--device", "cpu", "--workers", "0", "--batch", "2", "--smoke"]
    out = tmp_path / "smoke/unet_s1"
    with pytest.raises(InterruptedError):
        module.main(args)
    before = (out / "best.pt").read_bytes()
    assert not (out / "last.pt").exists()
    monkeypatch.setattr(module, "atomic_checkpoint", original)
    module.main(args)
    assert (out / "best.pt").read_bytes() == before
    assert json.loads((out / "complete.json").read_text())["best_val"] == .5
