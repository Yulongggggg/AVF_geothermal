"""Numerical settings validated by four-GPU diagnostic 104311.

The pooling equivalence requires this architecture's canonical 20x16 feature grid.
Neural parameters, data, losses and optimization schedules are preserved.
"""
import os
import torch

DDP_SETTINGS = dict(bucket_cap_mb=1000, find_unused_parameters=True)
NUMERICS = dict(deterministic_algorithms=True, cudnn_benchmark=False,
                cudnn_deterministic=True, cudnn_tf32=False, matmul_tf32=False,
                float32_matmul_precision='highest', cublas_workspace_config=':4096:8',
                pressure_pool='fixed_5x4_stride_5x4_on_20x16', ddp=DDP_SETTINGS)


def configure_runtime():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


class CanonicalPressurePool(torch.nn.AvgPool2d):
    def __init__(self):
        super().__init__((5, 4), (5, 4))

    def forward(self, field):
        assert field.shape[-2:] == (20, 16), 'Fixed pooling requires canonical feature grid'
        return super().forward(field)


def fixed_pressure_pool(net):
    assert isinstance(net.pressure_grid[2], torch.nn.AdaptiveAvgPool2d)
    assert net.pressure_grid[2].output_size in (4, (4, 4))
    net.pressure_grid[2] = CanonicalPressurePool()
    return net
