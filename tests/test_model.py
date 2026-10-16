"""Architecture checks on a randomly initialized GeoUNO (no artifacts needed)."""
import json

import numpy as np
import pytest
import torch

from geouno.artifacts import resource
from geouno.inputs import PhysicalInputEncoder, field_on_grid
from geouno.model import TRAINABLE_PARAMETERS, GeoUNOBasisHead, fixed_pressure_pool
from geouno.model.geouno import coordinates


pytestmark = pytest.mark.model_execution

@pytest.fixture(scope='module')
def net():
    torch.manual_seed(0)
    model = GeoUNOBasisHead([11.3, 8.0], 1.6, pod_rank=1024).eval()
    with torch.no_grad():
        model.pod.normal_()
        model.mean_function.normal_()
    return model


def _inputs(batch=2):
    g = torch.Generator().manual_seed(1)
    yy, xx = torch.meshgrid(torch.linspace(-500, 500, 160), torch.linspace(-400, 400, 128), indexing='ij')
    m = torch.randn(batch, 1, 160, 128, generator=g) * .5 + .1
    mask = (yy.abs() <= 400).float().expand(batch, 1, -1, -1)
    rp = torch.hypot(xx - 50, yy - 200)[None, None].expand(batch, 1, -1, -1) / 400
    ri = torch.hypot(xx, yy + 250)[None, None].expand(batch, 1, -1, -1) / 400
    field = torch.cat([m, mask, (xx / 400).expand(batch, 1, -1, -1), (yy / 500).expand(batch, 1, -1, -1), rp, ri], 1)
    return field, torch.randn(batch, 5, generator=g) * .3


def test_trainable_parameter_count_matches_paper(net):
    assert sum(p.numel() for p in net.parameters()) == TRAINABLE_PARAMETERS == 3_520_755


def test_output_is_exact_basis_expansion(net):
    field, scalar = _inputs()
    with torch.no_grad():
        out, parts = net(field, scalar, return_parts=True)
        expanded = torch.einsum('bcr,crn->bcn', parts['coefficients'], parts['basis']) + parts['mean'][None]
        rebuilt = torch.stack(((expanded[:, 0] + parts['constant_coefficients'][:, 0:1]) * net.scales[0],
                               expanded[:, 1] * net.pressure_residual_scale
                               + parts['constant_coefficients'][:, 1:2] * net.scales[1]), 1).reshape_as(out)
    assert out.shape == (2, 2, 160, 128)
    assert torch.equal(out, rebuilt)


def test_queries_at_sensor_nodes_reproduce_grid_values(net):
    field, scalar = _inputs(1)
    idx = torch.tensor([0, 127, 20479, 999, 10240, 8192])
    with torch.no_grad():
        grid = net(field, scalar).flatten(2)[:, :, idx]
        query = net(field, scalar, query=net.sensor_coordinates[idx])
    torch.testing.assert_close(query, grid, rtol=1e-4, atol=5e-4)
    with torch.no_grad():
        assert net(field, scalar, query=coordinates(height=31, width=27)).shape == (1, 2, 31 * 27)


def test_fixed_pool_is_equivalent_to_adaptive_pool():
    torch.manual_seed(2)
    model = GeoUNOBasisHead([11.3, 8.0], 1.6, pod_rank=8).eval()
    h = torch.randn(3, 192, 20, 16)
    adaptive = model.pressure_grid(h)
    fixed = fixed_pressure_pool(model).pressure_grid(h)
    torch.testing.assert_close(adaptive, fixed)


def test_input_encoder_recovers_physical_controls():
    stats = json.loads(resource('stats.json').read_text())
    enc = PhysicalInputEncoder(stats)
    x = torch.zeros(1, 13, 160, 128)
    q, tinj, pp, depth, ti = 2.5, 300., 25., 3.2, 10
    x[:, 7] = (q - stats['scal_mean'][0]) / stats['scal_std'][0]
    x[:, 8] = (tinj - stats['scal_mean'][1]) / stats['scal_std'][1]
    x[:, 10] = (pp - stats['pp_norm'][0]) / stats['pp_norm'][1]
    x[:, 11] = stats['tau_norm'][ti]
    x[:, 12] = (depth - stats['depth_norm'][0]) / stats['depth_norm'][1]
    _, cond = enc(x)
    expected = [q / 3.5, (tinj - 303) / 10, (pp - 20) / 10, (np.log10(stats['times_s'][ti]) - 6) / 2,
                (depth - 3.06) / .6]
    np.testing.assert_allclose(cond[0].numpy(), expected, rtol=1e-5, atol=1e-5)


def test_field_on_grid_copies_edge_rows_into_cap_and_bed():
    t = torch.arange(128.).repeat(128, 1).T[None, None]   # value = row index on the 128-row reservoir grid
    out = field_on_grid(t)[0, 0]
    assert torch.allclose(out[0], out[15]) and torch.allclose(out[-1], out[-16])
    assert torch.allclose(out[0], t[0, 0, 0]) and torch.allclose(out[-1], t[0, 0, -1])
