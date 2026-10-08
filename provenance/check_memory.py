"""Optional activation recomputation for CPU verification on memory-limited accounts.

This helper changes only verification instances, never the package or archived source.
Batch shapes, model state, floating-point operations and comparison tolerances stay fixed.
"""
import ctypes
import gc

from torch.utils.checkpoint import checkpoint


def release_unused():
    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        pass  # malloc_trim is an optional glibc allocator hint.


def recompute_activations(operator):
    def wrap(forward):
        return lambda *args: checkpoint(forward, *args, use_reentrant=False)

    for block in (*operator.enc, *operator.temperature_decoder,
                  *operator.pressure_decoder, operator.global_mix, operator.geology_readout):
        block.forward = wrap(block.forward)
    operator.branch = wrap(operator.branch)
