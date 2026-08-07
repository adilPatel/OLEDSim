"""
DEVSIM-dependent back end for the drift-diffusion solver.

Everything in this sub-package talks directly to DEVSIM. The rest of the
program (``core.Device``, ``core.DDSolver``, ``output_tools``) is kept free of
DEVSIM imports so that this whole package can be replaced by a custom solver
back end later -- see ``devsim_backend.backend.DevsimBackend`` for the
interface a replacement needs to implement.
"""

from .backend import DevsimBackend

__all__ = ["DevsimBackend"]
