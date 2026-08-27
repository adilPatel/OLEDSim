"""
Device-independent core of the drift-diffusion PINN (DDNet).

Nothing here holds a device-specific number (layer widths, HOMO/LUMO, contact
densities, loss weights, ...); those live in the scripts under
``sim_pinn.devices`` and are passed in.

Modules
-------
``networks``      the shared FCNN backbone and the autodiff derivative helper.
``scaling``       physical constants and the nondimensionalization.
``poisson``       phi-Net, its boundary ansatz, the Poisson residual.
``densities``     n-Net/p-Net, the log parametrisation, currents,
                  recombination and the continuity residuals.
``boundaries``    thermionic injecting contacts and their flux balance.
``loss_weights``  the weight rules: fixed, or adaptive (inverse-Dirichlet,
                  SoftAdapt).
``pinn_dd``       ``PINNProblem``, which bundles the three networks together.
``training``      the Adam training loop.
``reporting``     scoring against a reference, current reporting, plotting.
"""

from .boundaries import ThermionicContact, injection_current, reduced_field
from .densities import LogDensityNet
from .loss_weights import (
    DEFAULT_LOSS_WEIGHTS,
    LOSS_TERMS,
    FixedWeights,
    InverseDirichletWeights,
    SoftAdaptWeights,
    LossWeights,
    make_weights,
)
from .networks import FCNN, d_dx
from .pinn_dd import PINNProblem, build_networks, build_problem
from .poisson import POISSON_RESIDUAL_FLOOR, PhiNet
from .reporting import (
    evaluate,
    plot,
    plot_weights,
    relative_L1,
    report_currents,
    report_thermionic,
)
from .scaling import EPS_0, K_B, Q, compute_scaling
from .training import train

__all__ = [
    # networks / architecture
    "FCNN",
    "PhiNet",
    "LogDensityNet",
    "d_dx",
    # scaling
    "Q",
    "K_B",
    "EPS_0",
    "compute_scaling",
    # problem
    "PINNProblem",
    "build_networks",
    "build_problem",
    "POISSON_RESIDUAL_FLOOR",
    # boundaries
    "ThermionicContact",
    "injection_current",
    "reduced_field",
    # loss weights
    "LossWeights",
    "FixedWeights",
    "InverseDirichletWeights",
    "SoftAdaptWeights",
    "make_weights",
    "DEFAULT_LOSS_WEIGHTS",
    "LOSS_TERMS",
    # training and reporting
    "train",
    "evaluate",
    "relative_L1",
    "report_currents",
    "report_thermionic",
    "plot",
    "plot_weights",
]
