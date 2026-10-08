"""Compatibility imports for checkpoints and code using the historical module name.

Use :mod:`geouno.model.geothermal` and ``GeothermalJointOperator`` in new code.
Geology feature names still describe permeability inputs, not the physical operator.
"""
from .geothermal import GeothermalJointOperator, GeologyGate, RegionalReadout, geology_maps, load_parent

GeologyJointOperator = GeothermalJointOperator

__all__ = ['GeologyJointOperator', 'GeothermalJointOperator', 'GeologyGate', 'RegionalReadout',
           'geology_maps', 'load_parent']
