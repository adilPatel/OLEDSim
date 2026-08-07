### Overview of the PINN Models Used

This document describes the physical models used for the physics-informed
neural network (PINN) used, in addition to implementation details. For now,
only a Poisson solver rather than the full set of drift diffusion equations.

Two scripts make up the current demo:

| file | role |
|---|---|
| `devsim_reference.py` | Solves the device with the DEVSIM drift-diffusion back end and saves the 2.5 V solution to `devsim_reference_2.5V.npz`. |
| `poisson_demo.py` | Trains the Poisson PINN (phi-Net) against that reference and plots both potentials together. |

Run `devsim_reference.py` first; `poisson_demo.py` reads the `.npz` it writes.

## The device

A 100 nm undoped organic active layer with two Ohmic contacts, built from the
same `core`/`devsim_backend` machinery as oled1. It is **OLED1's contact set
with one change**: the bottom contact's electron density is reduced from
1e25 to **1e17 cm^-3**.

| | top (anode) | bot (cathode) |
|---|---|---|
| electron density | 3.2e-10 | **1e17** (OLED1: 1e25) |
| hole density | 2.0e12 | 7.3e-30 |
| voltage offset | Vbi = 2.0 V | 0 |

# Why reduce the bottom electron density

The Debye length screening a contact goes as `1/sqrt(n)`, so the pinned
density sets the width of the space-charge layer and therefore how much
curvature the potential has:

| `n_bot` (cm^-3) | `L_D` (nm) | `L/L_D` |
|---|---|---|
| 1e25 (OLED1) | 0.0008 | 132000 |
| 1e20 | 0.239 | 418 |
| **1e17 (used here)** | **7.57** | **13.2** |

At 1e25 the screening layer is five orders of magnitude narrower than the
device — effectively a discontinuity, which no smooth network can represent
and which even the finite-volume solver needs a very fine mesh to resolve. At
1e17 the layer is a few nm wide: steep but resolvable, and still physically
reasonable for an organic semiconductor (well below the 1e27 DOS).

# Bias convention

`Vbi = 2.0 V` is applied as the anode's `voltage_offset`, per the convention
in `core/device.py`. The PINN comparison is done at an applied bias of
**2.5 V**, so the net drop across the device is `2.5 - 2.0 = 0.5 V` — forward
bias nearly flattening the built-in field.

Note that the Boltzmann potentials implied by the pinned densities
(1.13 V for this contact pair) do *not* equal the specified `Vbi`. That is
expected: in this framework `voltage_offset` sets the potential boundary
condition independently of the density pins, so `Vbi` is imposed directly
rather than derived.

## Drift Diffusion Equations

# Poisson Equation

The Poisson Equation used is scaled according to the extrinsic Debye length
`L_D = sqrt(eps*kT/(q^2*N))` to ensure that the equation has O(1) residual
terms. The voltage used is scaled according to quasi-Fermi levels
`V_hat = ±q(V-phi)/kT`. The Poisson Equation is thus:

    d^2V_hat/dx_hat^2 = n_i/N(exp(V_hat)-exp(-V_hat)+(N_A-N_d)/n_i)

with Boltzmann carrier densities `n = n_i*exp(V_hat)` and
`p = n_i*exp(-V_hat)`.

# Form used in this demo

The Boltzmann form above closes the equation by *assuming* equilibrium
statistics, `n = n_i exp(V_hat)`. That assumption does not hold at 2.5 V
forward bias, where the carrier densities are set by injection and transport,
not by a local equilibrium with the potential. This demo therefore keeps the
same scaling but takes `n` and `p` as supplied quantities:

    lambda^2 * phi_hat''(x_hat) = n_hat - p_hat

with `lambda = L_D/ell`, `ell` the device length, and `n_hat = n/C_tilde`.
This is `eps*phi'' = q*(n - p - C)` scaled, with `C = 0` because the layer is
**undoped** — unlike the doped p-n junction of `poisson_pinn_doped.py`, the
entire space charge here is mobile carriers.

The density scale `C_tilde = 1e17 cm^-3` is the bottom contact's electron
density — the largest carrier density present, so it sets the screening
length. There is no doping to scale by.

Sign convention: the same as `sim_dd` and the DDNet paper's Eq. 1, where a net
*electron* excess gives positive curvature. `poisson_pinn_doped.py` writes
`lambda^2*phi'' = -C` for the depletion case, which is the identical equation
with `n = p = 0` and `C` the fixed dopant charge.

# TODO: Carrier densities

A logarithmic formulation is used for the carrier densities n,p owing to their 
very large changes in concentrations. where n_hat, p_hat = -log(n_hat,p_hat/C_tilde).
The network's internals operate in the log space, and the output is exponentiated
to compute the residuals. 

# Boundary conditions

Both contacts are Ohmic, so `phi` is pinned at both ends (Dirichlet). The two
values are read from the DEVSIM solution at the contact nodes — these are the
conditions DEVSIM itself imposed (applied bias at the anode, 0 at the cathode),
so the two problems are identical rather than merely similar:

    phi(0) = +0.500 V  ->  phi_hat = +19.31
    phi(L) =  0.000 V  ->  phi_hat =   0.00

TODO: Contact carrier densities
The contact densities are also fixed at both ends (Dirichlet), with the values
specified in devsim_reference.py.



## PINN Architecture

Firstly, an Ansatz network is constructed for the drift-diffusion equations.
Then the weights are initialised using Glorot-Xavier. Then points will be
randomly sampled in the region, and the residuals are computed with automatic
differentiation. It's trained with ADAM.

As implemented in `forward_demo.py`, following the same phase structure
(A–E) as `poisson_pinn_doped.py`:

TODO: n and p networks
The electron and hole networks n,p are also FCNNs but operate in log space as 
described previously. They possess the same overall structure besides the 
exponentiation.

- **Network.** Fully-connected, 4 layers of 64 units, `tanh` activations.
  `tanh` is required rather than preferred: the residual needs a *second*
  derivative through automatic differentiation, and ReLU's second derivative
  is identically zero. No log-space trick is needed for `phi` — its dynamic
  range is modest (~0.5 V), unlike `n` and `p`.
- **Initialisation.** Glorot/Xavier normal, biases zero.
- **Input normalisation.** `x_hat` is mapped from `[0,1]` to `[-1,1]` before
  the first layer, where `tanh` is centred.
- **Collocation.** 8192 interior points, resampled every epoch.
- **Optimiser.** Adam, `lr = 1e-2`, decayed 10x at epochs 2000/4000/5500/7000.
- **Precision.** float64 throughout — the densities span ~26 decades and
  float32 loses the small end entirely.

# Measured settings

[Add the output details here...]

## Next steps [REMOVE THIS SECTION AFTER IMPLEMENTING THE N AND P NETS]

This demo deliberately solves only the Poisson equation, with `n` and `p`
supplied from DEVSIM. Adding the n-Net and p-Net makes the system nonlinear
and coupled, at which point the logarithmic output parametrisation DDNet uses
(`-log(n/C)`, exponentiated through a hard constraint) becomes necessary: the
densities here already span 26 decades, and a network outputting them directly
would capture only the largest 3-4, as DDNet's Supplementary Fig. 2 shows.
