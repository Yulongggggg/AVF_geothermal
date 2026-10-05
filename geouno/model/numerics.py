"""Numerical settings used for every reported GeoUNO inference.

Source: ``operator_parallel_20260927_v2/runtime.py`` (``CanonicalPressurePool``,
``fixed_pressure_pool``, ``configure_runtime``) and the backend settings of
``operator_final_20260930/backend.py``.

* ``AdaptiveAvgPool2d((4, 4))`` on the canonical 20 x 16 feature grid is replaced by the
  equivalent fixed ``AvgPool2d((5, 4), (5, 4))``. Training validated this equivalence;
  all paper inference used the fixed pool.
* TF32 is disabled and PyTorch's fused multi-head-attention fast path is switched off,
  so evaluation-mode attention uses the same kernels as the reported runs.
"""
import os

import torch

DETERMINISTIC_SETTINGS = dict(
    deterministic_algorithms=True, cudnn_benchmark=False, cudnn_deterministic=True,
    cudnn_tf32=False, matmul_tf32=False, float32_matmul_precision='highest',
    cublas_workspace_config=':4096:8', pressure_pool='fixed_5x4_stride_5x4_on_20x16')


class CanonicalPressurePool(torch.nn.AvgPool2d):
    def __init__(self):
        super().__init__((5, 4), (5, 4))

    def forward(self, field):
        assert field.shape[-2:] == (20, 16), 'Fixed pooling requires canonical feature grid'
        return super().forward(field)


def fixed_pressure_pool(net):
    """Swap the adaptive 4 x 4 pressure-grid pool for its fixed-kernel equivalent (in place)."""
    assert isinstance(net.pressure_grid[2], torch.nn.AdaptiveAvgPool2d)
    assert net.pressure_grid[2].output_size in (4, (4, 4))
    net.pressure_grid[2] = CanonicalPressurePool()
    return net


def configure_inference(threads=None):
    """Apply the precision settings of the reported inference runs (process-wide).

    ``threads`` sets ``torch.set_num_threads``. CPU results depend slightly on the thread
    count; the archived reference checks used 2 threads for the NUTS bundle and 4 for
    MAP/ES-MDA.
    """
    if threads is not None:
        torch.set_num_threads(int(threads))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)


def configure_deterministic_training():
    """Settings used for distributed training (deterministic kernels, no TF32)."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
