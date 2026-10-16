"""Final-stage checkpoint selection and scenario grouping; no model execution."""

import copy
import numpy as np
from geouno.training.data import ScenarioPairs
from geouno.training.synthetic import make_synthetic_pack
from geouno.training.train import admissible


def metrics(t=1.0, p=1.0, tail=1.0):
    row = dict(T_rmse=t, P_rmse=p, inj_P90=tail, prod_P90=tail, P_scenario_P90=tail)
    return dict(summary={str(k): copy.deepcopy(row) for k in (4, 7, 10)} | {"all_evaluated_times": copy.deepcopy(row)})


def test_temperature_stage_preserves_pressure():
    start = metrics()
    assert admissible(metrics(0.9, 1.0), start, "temperature")[1]
    assert not admissible(metrics(0.9, 1.01), start, "temperature")[1]
    assert not admissible(metrics(1.01, 1.0), start, "temperature")[1]


def test_joint_stage_controls_each_field_and_pressure_tails():
    start = metrics()
    assert admissible(metrics(0.9, 0.9), start, "joint")[1]
    assert not admissible(metrics(0.5, 1.01), start, "joint")[1]
    assert not admissible(metrics(0.9, 0.9, 1.03), start, "joint")[1]


def test_dataset_groups_members_by_scenario(tmp_path):
    pack = make_synthetic_pack(tmp_path / "pack", train=4, val=2, members=3)
    data = ScenarioPairs(pack, "train")
    field, scalar, truth, n = data[(0, 10)]
    assert len(data) == 4 and n == 3
    assert field.shape == (3, 6, 160, 128) and scalar.shape == (3, 5) and truth.shape == (3, 2, 160, 128)
    assert np.isfinite(field).all()
