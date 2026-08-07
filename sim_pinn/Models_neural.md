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

# The carrier densities are supplied, not solved

Because `C = 0`, Poisson has no source at all unless `n` and `p` are given.
The depletion approximation used in `poisson_pinn_doped.py` is unavailable
here for exactly that reason — there is no dopant charge to fall back on when
the carriers are neglected.

Since the n-Net and p-Net are to be provided separately, this demo takes
`n(x)` and `p(x)` from the converged DEVSIM solution **at the same bias** and
treats them as fixed, known source terms. That makes the Poisson problem
linear in `phi` and gives an unambiguous reference: with the *exact* `n` and
`p` imposed, a correct phi-Net must reproduce DEVSIM's potential.

The DEVSIM profiles live on a non-uniform 125-node mesh while the PINN samples
collocation points anywhere in the domain, so the source is interpolated
(`_interp_tensor`, piecewise linear via `torch.searchsorted`). The
interpolation is done in **linear** space, not log space: it is the difference
`n - p` that enters Poisson, and that difference changes sign near the anode
where holes dominate, which a log interpolation cannot represent.

# Boundary conditions

Both contacts are Ohmic, so `phi` is pinned at both ends (Dirichlet). The two
values are read from the DEVSIM solution at the contact nodes — these are the
conditions DEVSIM itself imposed (applied bias at the anode, 0 at the cathode),
so the two problems are identical rather than merely similar:

    phi(0) = +0.500 V  ->  phi_hat = +19.31
    phi(L) =  0.000 V  ->  phi_hat =   0.00

## Consistency check: does DEVSIM satisfy the scaled equation?

Before training anything, the scaled equation was checked directly against the
DEVSIM solution: evaluate `lambda^2*phi_hat''` on DEVSIM's own mesh (with a
non-uniform three-point stencil) and compare against `n_hat - p_hat`.

    median ratio (LHS/RHS) = 1.0000
    relative L1 of residual = 1.2e-12

This confirms the scaling, the sign convention, the permittivity, and the
`C = 0` assumption are all correct, and that the PINN is being asked to solve
precisely the equation DEVSIM solved. Worth doing first: had the scaling been
wrong, the PINN would have converged faithfully to the wrong answer.

## PINN Architecture

Firstly, an Ansatz network is constructed for the drift-diffusion equations.
Then the weights are initialised using Glorot-Xavier. Then points will be
randomly sampled in the region, and the residuals are computed with automatic
differentiation. It's trained with ADAM.

As implemented in `poisson_demo.py`, following the same phase structure
(A–E) as `poisson_pinn_doped.py`:

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

Two defaults differ from `poisson_pinn_doped.py`, both measured rather than
assumed.

**Collocation count `N_INT = 8192`** (vs 2048): the source term here is an
interpolated DEVSIM profile rather than a smooth analytic `tanh`, so denser
sampling resolves it better.

| `N_INT` | relative L1 |
|---|---|
| 2048 | 1.74 % |
| 8192 | 1.21 % |

**Boundary weight `W_BC = 1.0`.** This is not a placeholder — raising it makes
the solve markedly *worse*. The Dirichlet values are O(19) in scaled units
(`0.5 V / 0.0259 V = 19.3`), so the squared boundary term starts around 1e2,
and any upweighting lets it dominate the loss and starve the interior
residual:

| `W_BC` | relative L1 |
|---|---|
| **1** | **1.74 %** |
| 10 | 35.3 % |
| 100 | 64.7 % |
| 1000 | 78.6 % |

(measured at `N_INT = 2048`.) The hard-constraint alternative — multiplying
the network output by a factor vanishing at both contacts, as DDNet does for
its carrier outputs — would remove the trade-off entirely and is the natural
next step if the boundary error ever becomes the limit. Here it is already
~1e-9, so it is not.

## Results

At 2.5 V, against the DEVSIM drift-diffusion solution:

| metric | value |
|---|---|
| relative L1 error | **1.21 %** |
| max absolute error | 3.9 mV |
| boundary residual | ~1e-8 |
| final training loss | 4.1e-04 |

The relative L1 error is the metric used throughout the DDNet paper
(Supplementary Eq. 3), so these numbers are directly comparable with it — and
comfortably inside the "below 2%" the paper reports for its 1D forward
solutions.

`poisson_demo.png` has four panels: (a) both potentials overlaid — the
requested comparison, (b) their pointwise difference, (c) the DEVSIM carrier
densities used as the source term (log axis: they span ~26 decades), and
(d) the training loss.

# Remaining error

The pointwise difference in panel (b) is a single smooth bulge of one sign,
peaking around 4 mV mid-device, rather than random scatter. That is the
signature of a small, nearly constant error in `phi''` integrating up twice —
a Poisson solution is pinned only by its boundary values, so a slight drift in
curvature is cheap in the interior residual.

It is *not* caused by the boundary term being underweighted (see the `W_BC`
table above — that was tested and rejected). Increasing collocation density
reduces it, which points at how well the interpolated source term is resolved.
The most promising next steps are the hard-constraint ansatz and a
scale-normalised residual.

## Next steps

This demo deliberately solves only the Poisson equation, with `n` and `p`
supplied from DEVSIM. Adding the n-Net and p-Net makes the system nonlinear
and coupled, at which point the logarithmic output parametrisation DDNet uses
(`-log(n/C)`, exponentiated through a hard constraint) becomes necessary: the
densities here already span 26 decades, and a network outputting them directly
would capture only the largest 3-4, as DDNet's Supplementary Fig. 2 shows.
