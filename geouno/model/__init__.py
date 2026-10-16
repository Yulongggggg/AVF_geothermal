"""GeoUNO neural operator: architecture and checkpoint loading."""

import hashlib
from pathlib import Path

import torch

from .geouno import CoordinateTrunk, build_from_checkpoint, coordinates
from .final import FinalGeoUNO

from .geothermal import GeothermalJointOperator, load_parent
from .geology import GeologyJointOperator  # Historical import compatibility.
from .numerics import fixed_pressure_pool, configure_inference

GeoUNOBasisHead = FinalGeoUNO
GeoUNO = FinalGeoUNO

#: SHA-256 of the checkpoint evaluated in the paper (POD1024 + 64 learned functions).
CHECKPOINT_SHA256 = "f39894ff83e04f47b2a9098f0e387540e1390cf12b9d7f8726dc815dffd16dbb"
#: Historical parent hash retained for compatibility with legacy training utilities.
PARENT_SHA256 = "cf278e646fbe53db11ec4a03f0545b6354ef81daa9d8c2282ebaf24704ccff9b"
#: Trainable parameters of the selected model (the POD basis is a fixed buffer).
TRAINABLE_PARAMETERS = 3_520_755


def sha256(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_geouno(path, device="cpu", expected_sha256=CHECKPOINT_SHA256, fixed_pool=True):
    """Load the GeoUNO checkpoint for inference.

    The file hash is checked against ``expected_sha256`` (pass ``None`` to skip, e.g. for
    a checkpoint you trained yourself). ``fixed_pool=True`` applies the fixed-kernel
    pressure pool used for every reported inference.
    """
    if expected_sha256 is not None:
        actual = sha256(path)
        if actual != expected_sha256:
            raise ValueError(f"{path}: SHA-256 {actual} does not match the expected {expected_sha256}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    net = build_from_checkpoint(state)
    if fixed_pool:
        fixed_pressure_pool(net)
    return net.to(device).eval()


__all__ = [
    "GeoUNO",
    "FinalGeoUNO",
    "GeoUNOBasisHead",
    "CoordinateTrunk",
    "GeothermalJointOperator",
    "GeologyJointOperator",
    "coordinates",
    "build_from_checkpoint",
    "load_geouno",
    "load_parent",
    "fixed_pressure_pool",
    "configure_inference",
    "sha256",
    "CHECKPOINT_SHA256",
    "PARENT_SHA256",
    "TRAINABLE_PARAMETERS",
]
