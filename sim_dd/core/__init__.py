"""
Solver-independent core of the drift-diffusion simulator.

Nothing in this package imports DEVSIM. ``Device`` describes the device,
``DDSolver`` drives the solve through a back-end interface, and results come
back as plain-Python :class:`SweepResults`.
"""

from .device import Device, Contact, make_oled1, make_oled2, default_oled_mesh_nm
from .dd_solver import DDSolver, SweepConfig, SweepResults

__all__ = [
    "Device",
    "Contact",
    "make_oled1",
    "make_oled2",
    "default_oled_mesh_nm",
    "DDSolver",
    "SweepConfig",
    "SweepResults",
]
