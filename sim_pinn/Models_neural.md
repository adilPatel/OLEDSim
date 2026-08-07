### Overview of the PINN Models Used

This document describes the physical models used for the physics-informed
neural network (PINN), in addition to implementation details. The full
drift-diffusion system is now solved: all three DDNet subnetworks (phi-Net,
n-Net and p-Net) are trained together against the coupled Poisson + continuity
equations.

Two scripts make up the current demo:

| file | role |
|---|---|
| `devsim_reference.py` | Solves the device with the DEVSIM drift-diffusion back end and saves the 2.5 V solution to `devsim_reference_2.5V.npz`. |
| `forward_demo.py` | Trains the coupled DDNet (phi-Net + n-Net + p-Net) and compares all three fields against that reference. |

Run `devsim_reference.py` first; `forward_demo.py` reads the `.npz` it writes.

**The reference is a scoring target, not a training input.** In the earlier
Poisson-only stage `n` and `p` were interpolated from DEVSIM and fed to the
network as a fixed source term. They are now unknowns produced by their own
networks, so the only things taken from the `.npz` are the contact boundary
values and the profiles used to compute the error afterwards. Nothing in the
interior loss references DEVSIM — this is a forward PINN solve, not a
regression fit.

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

The transport parameters were irrelevant while only Poisson was being solved,
but the continuity equations depend on them. `devsim_reference.py` overrides
neither mobility nor the densities of states, so these are
`core.device.Device`'s defaults, and the PINN reads the same numbers:

| parameter | value | note |
|---|---|---|
| `mu_n`, `mu_p` | 1e-6 cm²/V·s | equal, so both scaled mobilities are 1 |
| `NC300`, `NV300` | 1e27 cm^-3 | |
| LUMO / HOMO | -0.4 / -3.0 eV | `EG = 2.6 eV` |
| `nie` | 1.551e5 cm^-3 | `sqrt(NC*NV)*exp(-EG/(2*Ut))` |
| `eps_r` | 4.0 | |
| `gammar` | 1 | Langevin prefactor |

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
not by a local equilibrium with the potential. The densities are therefore
solved for by their own networks, and Poisson is written

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

# Continuity equations

The two carrier continuity equations close the system. Scaled per DDNet
Supplementary Table 1 (lengths by `ell`, potential by `Ut`, densities by
`C_tilde`, mobilities by `mu_tilde = max(mu_n, mu_p)`):

    Jn_hat' =  R_hat,     Jn_hat =  mu_n_hat*(n_hat' - n_hat*phi_hat')
    Jp_hat' = -R_hat,     Jp_hat = -mu_p_hat*(p_hat' + p_hat*phi_hat')

The opposite signs on `R_hat` are physical: a recombination event removes one
electron *and* one hole, so it is a sink for both, and the sign difference
reflects the opposite charges carried.

Note that the thermal voltage present in the unscaled currents
(`Jn = q*mu_n*(Ut*n' - n*phi')`) has disappeared. That is precisely what
scaling `phi` by `Ut` buys, and it has a practical consequence worth stating
because the naive guess is wrong: the scaled currents are **not** small.
Evaluated on the DEVSIM reference, `Jn_hat ~ 3.9`, comparable to the Poisson
source `|n_hat - p_hat| ~ 0.26`, so no relative reweighting of the continuity
and Poisson residuals is needed (see *Loss weights* below).

# Recombination

`R` is Langevin recombination, matching `sim_dd`'s `CreateLangevin`
(`os_physics.py`) exactly so the PINN and DEVSIM cannot differ on the physics:

    R = gammar * (q/eps) * (n*p - nie^2) * (mu_n + mu_p)

with `gammar = 1`. `nie` is computed the same way DEVSIM's node model does,
`nie = sqrt(NC*NV)*exp(-EG/(2*Ut))`, which gives `1.551e5 cm^-3` for this
device's `EG = 2.6 eV` and `NC = NV = 1e27 cm^-3`. At `T = 300 K` the
temperature-dependent corrections in DEVSIM's expression (`EGALPH`, `DEG`) all
vanish, so the Python constant reproduces the node model exactly.

The `nie^2` term is ~2.4e-24 in scaled units and utterly negligible against
`n_hat*p_hat` over most of the device, but it is kept: it is what makes `R`
vanish at equilibrium, and dropping it would put a small spurious recombination
everywhere the carriers are depleted.

Transport parameters are `core.device.Device`'s defaults, which is what
`devsim_reference.py` leaves them at — it overrides neither mobility nor the
densities of states: `mu_n = mu_p = 1e-6 cm^2/V-s`, `NC300 = NV300 = 1e27 cm^-3`.
Since the two mobilities are equal, both scaled mobilities come out at exactly
1, but the division by `mu_tilde` is kept explicit in the code so unequal
mobilities stay correct.

# Carrier densities: the logarithmic formulation

A logarithmic formulation is used for the carrier densities `n`, `p` owing to
the very large range of concentrations. Following DDNet Sec. 4.2, each network
outputs the compressed variable

    u_n = -log(n_hat),    u_p = -log(p_hat)

(natural log) and the density is recovered through a **hard constraint**,

    n_hat = exp(-u_n),    p_hat = exp(-u_p)

so the network's internals operate in log space and the output is
exponentiated before the residuals are computed.

Two clarifications on the formulation as originally written here. The
compressed variable is `-log(n_hat)`, i.e. `-log(n/C_tilde)` — the scaling by
`C_tilde` happens *inside* the log, and `n_hat` is already the scaled density,
so writing `n_hat = -log(n_hat/C_tilde)` would divide by `C_tilde` twice and
also reuse one symbol for two different quantities. The two are kept
distinct here: `n_hat` is the scaled density, `u_n` is what the network emits.

The negative sign matters and is not cosmetic. Scaling by `C_tilde = 1e17`
shifts the density range to `(0, 1]`, so `log(n_hat)` is negative throughout
and `-log(n_hat)` is positive — DDNet makes the same point for its `C_tilde =
1e16` example. The hard constraint gives two things beyond range compression:

- **Positivity.** `exp(-u) > 0` identically, so a negative density is not
  merely penalised but unrepresentable. This matters because the Langevin term
  is bilinear in `n` and `p`; a transient negative density during early
  training would flip the sign of the recombination term.
- **Dynamic range.** The densities span ~26 decades, which is ~60 in `u`. In
  `u` that is an O(10) network output, well within what a tanh network
  resolves. DDNet's Supplementary Fig. 2 shows a network outputting the density
  directly captures only the largest 3-4 decades.

**Derivatives use the chain rule on `u`, not on `exp(-u)`.** The continuity
residual needs `n_hat'`, which is computed as

    n_hat' = -u_n' * exp(-u_n) = -u_n' * n_hat

rather than by differentiating `exp(-u_n)` directly. The two are algebraically
identical, but this form keeps the large exponential factored out of the
autograd graph: `u_n'` is O(1) and the exponential enters as a single
multiplication, instead of as the gradient of an `exp()` spanning tens of
decades.

# Boundary conditions

Both contacts are Ohmic, so `phi` is pinned at both ends (Dirichlet). The two
values are read from the DEVSIM solution at the contact nodes — these are the
conditions DEVSIM itself imposed (applied bias at the anode, 0 at the cathode),
so the two problems are identical rather than merely similar:

    phi(0) = +0.500 V  ->  phi_hat = +19.31
    phi(L) =  0.000 V  ->  phi_hat =   0.00

# Contact carrier densities — majority carrier only

DEVSIM pins *both* carrier densities at *both* Ohmic contacts, with the values
specified in `devsim_reference.py`. Imposing all four of those pins on the
networks does not work, and the reason is a property of the reference data
rather than a modelling preference.

**Only the majority pin at each contact is representable by a smooth network.
The minority pin is a genuine discontinuity, not a resolvable boundary layer:**

| contact | carrier | pinned value | value at the *adjacent* node (0.125 nm away) | jump |
|---|---|---|---|---|
| left (anode) | n (minority) | 3.2e-10 | 4.80e14 | ~24 decades |
| right (cathode) | p (minority) | 7.3e-30 | 2.00e9 | ~38 decades |

There is no intermediate structure — the interior trend at the right contact
heads toward ~1e9 while the pin sits at 7.3e-30. Excluding just those two
nodes, the interior is smooth and well conditioned:

| variable | interior range |
|---|---|
| `-log(n_hat)` | 0.015 .. 5.34 |
| `-log(p_hat)` | 10.8 .. 17.7 |

This is the same pathology that motivated softening the bottom contact from
1e25 to 1e17 for `phi`, and it has the same resolution. Forcing the network to
fit a step of tens of decades cannot succeed, and the attempt actively corrupts
the interior fit as the boundary term comes to dominate the loss.

So **the majority carrier is pinned at each contact and the minority carrier is
left free**, determined by the continuity equations (`MAJORITY_ONLY_BC` in
`forward_demo.py`, which retains the four-pin path for comparison):

    p(0) = 2.0e12 cm^-3    (anode, majority)   — pinned
    n(L) = 1.0e17 cm^-3    (cathode, majority) — pinned
    n(0), p(L)                                 — free

Physically this is the right call: the minority density at an injecting contact
is set by transport, its precise value there is irrelevant to the current
(which the majority carrier carries), and it is unresolved by the mesh anyway.

The density boundary terms are imposed in **log space**, matching `u` directly
rather than the density. That keeps the boundary term commensurate with the
network's own output scale; matching densities would make it a comparison
between numbers of order 1e-14, contributing essentially no gradient.



## PINN Architecture

Firstly, an Ansatz network is constructed for the drift-diffusion equations.
Then the weights are initialised using Glorot-Xavier. Then points are randomly
sampled in the region, and the residuals are computed with automatic
differentiation. It's trained with ADAM.

As implemented in `forward_demo.py`, following the same phase structure
(A–E) as `poisson_pinn_doped.py`.

# The three subnetworks

DDNet's Fig. 1 architecture: three coupled FCNNs sharing one spatial input.
The electron and hole networks are also FCNNs but operate in log space as
described previously; they have the same overall structure as phi-Net apart
from the exponentiation. In the code the shared backbone is factored into an
`FCNN` class, and the *output parametrisation* is what distinguishes the
subnetworks — `PhiNet` takes the backbone output directly, `LogDensityNet`
applies the hard constraint.

| | phi-Net | n-Net | p-Net |
|---|---|---|---|
| output | `phi_hat` | `u_n = -log(n_hat)` | `u_p = -log(p_hat)` |
| hard constraint | none | `n_hat = exp(-u_n)` | `p_hat = exp(-u_p)` |
| shape | 4 x 64, tanh | 4 x 64, tanh | 4 x 64, tanh |

DDNet uses the same depth (4 layers) and width (64 neurons) for all three
subnetworks (Supp. Sec. 3), which is what is used here. Total: 38,019
trainable parameters.

- **Network.** Fully-connected, 4 layers of 64 units, `tanh` activations.
  `tanh` is required rather than preferred: the residual needs a *second*
  derivative through automatic differentiation, and ReLU's second derivative
  is identically zero. No log-space trick is needed for `phi` — its dynamic
  range is modest (~0.5 V, i.e. `phi_hat` in [0, 19.3]), unlike `n` and `p`.
- **Initialisation.** Glorot/Xavier normal, biases zero.
- **Input normalisation.** `x_hat` is mapped from `[0,1]` to `[-1,1]` before
  the first layer, where `tanh` is centred.
- **Log-space output offset.** The n-Net and p-Net add a constant offset to
  `u`, set from the interior mean of the reference profile in log space. Xavier
  init gives an output near zero, i.e. `density_hat ~ 1`; the true interior
  hole density is `~e^-11.9`, so without the offset the p-Net starts ~5 decades
  too high and the bilinear Langevin term is correspondingly wrong at step 0.
  This uses the reference only to pick a *starting point* for the optimiser —
  the same role as any initialisation heuristic, not a training target — and
  the contact nodes are excluded so the discontinuous minority pins do not skew
  it. Measured offsets: `u_n = +1.64`, `u_p = +11.85`.
- **Collocation.** 4096 interior points — DDNet's own `2^12` (Sec. 4.3) —
  resampled every epoch. The Poisson-only stage used 8192, which measured
  better than 2048 *there* (1.21% vs 1.74% relative L1), but that rationale
  does not carry over: the gain came from resolving an interpolated DEVSIM
  profile used as a fixed source term, and the source is now a pair of smooth
  learned functions. Cost is linear in this number (measured: 0.094 / 0.181 /
  0.297 s per epoch at 2048 / 4096 / 8192, for three networks with second
  derivatives), so the halving is what makes a 20000-epoch CPU run practical.
- **Optimiser.** Adam, `lr = 1e-3`, decayed 3.3x at epochs
  5000/10000/14000/17000, for 20000 epochs. Both the rate and the epoch count
  changed from the Poisson-only stage (`lr = 1e-2`, 8000 epochs): the coupled
  system is nonlinear — the Langevin term is bilinear in `n` and `p`, and the
  drift term couples each density to `phi'` — so steps that were safe when
  `phi` was the only unknown against a *fixed* source now let the densities
  overshoot in log space, where an overshoot of a few units is a few decades in
  the density.
- **Precision.** float64 throughout. This is a change from the Poisson-only
  stage, which could use float32 because the 26-decade range lived only in the
  *inputs* and was differenced away immediately. The densities are now
  unknowns, `u_p` reaches ~17.7 in the interior, and `n_hat*p_hat` in the
  Langevin term underflows float32 — the log parametrisation keeps the network
  *outputs* O(1), but the residuals are still evaluated on exponentials.
  float64 forces CPU, since the MPS Metal backend does not implement it at all;
  `USE_GPU` and float64 are mutually exclusive and the code raises rather than
  silently mixing them.

# Loss

Following DDNet Eq. 11, the total loss sums interior PDE residuals and boundary
residuals, each a mean-squared error:

    L_tot = L_poisson + W_CONT*(L_Jn + L_Jp) + W_JTOT*L_Jtot + W_BC*L_BC

# Loss weights

Evaluating each term on the DEVSIM reference profiles gives the natural scales:

| quantity | scale on the reference |
|---|---|
| Poisson source `\|n_hat - p_hat\|` | 2.6e-1 |
| total current `\|Jn_hat + Jp_hat\|` | 3.9 |
| recombination `\|R_hat\|` | 6.1e-4 |

Read naively this says everything is already commensurate and no weighting is
needed — and it is worth noting the obvious guess is wrong in the *other*
direction too: the scaled currents are not small, since scaling `phi` by `Ut`
removes the thermal voltage from the current expression, leaving `Jn_hat ~ 3.9`.

**But those global scales are dominated by electrons, and following them
uncritically is a mistake.** The first full run weighted both continuity
equations equally at 1.0 and the p-Net flattened near the cathode instead of
following DEVSIM's roll-off (3.91e11 vs 1.65e10 at 99 nm). The cause is scale,
not missing physics, and both halves of that were measured on the reference by
perturbing `p` by 10x beyond 80 nm:

| residual | unperturbed | `p` 10x too high |
|---|---|---|
| Poisson | 9.998e-02 | 9.998e-02 (**unchanged**) |
| hole continuity | 1.218e-05 | 3.374e-02 (**~2800x**) |

So Poisson genuinely cannot see `p` there (`p/(n-p) ~ 1e-6`), while hole
continuity can — the constraint exists, it is just far too small to matter in
an unweighted sum, since `|Jp_hat|/|Jn_hat| ~ 1e-5` near the cathode puts the
hole term at ~1e-5 against a total loss of ~2.5e-2.

The two continuity equations therefore get **separate** weights,
`W_CONT_N = 1.0` and `W_CONT_P = 1e3`. The hole weight is set from the error it
has to make visible rather than by trial: a 10x-too-high `p` beyond 80 nm gives
`mean r_p^2 = 4.7e-4`, so at `1e3` it contributes 0.47 to the loss — about 20x
the Poisson term, a penalty large enough that the optimiser cannot ignore it.
Note that a small `cont_p` early in training is *not* evidence the weight is
inert: the p-Net starts close to satisfying hole continuity and drifts later.

`W_BC = 1.0` was measured in the Poisson-only stage: raising it made the solve
markedly worse (1.74% → 35.3% at 10, → 64.7% at 100, → 78.6% at 1000), because
the Dirichlet values are O(19) in scaled units so the squared boundary term
already starts ~1e2 and any upweighting starves the interior residual.

# The total-current constraint

One term is not in DDNet: `L_Jtot`, the variance of `Jn_hat + Jp_hat` across
the collocation batch.

In 1D steady state with no external generation, adding the two continuity
equations gives `(Jn + Jp)' = 0` exactly — the total current is constant across
the device. That is *implied* by the two continuity residuals, so the term is
formally redundant. But only formally: the continuity residuals are **local**
conditions constraining the derivative of each current separately, and nothing
couples a residual at one collocation point to one at another. A solution can
have both continuity residuals small in a mean-squared sense while the total
current still drifts across the domain — and `Jn + Jp` is the single number the
device is actually characterised by.

Penalising the spread directly ties the whole domain to one value. It is
implemented as a variance rather than a deviation from a target, which is the
translation-invariant way to say "constant" without having to know what the
constant is — the terminal current is an *output* of the solve, not an input.

# Evaluation

The relative L1 error of DDNet Supp. Eq. 3 is used throughout. Two adjustments
for the densities:

- **Scored in log space.** A linear-space L1 on a quantity spanning decades
  reports only how well the largest values were fitted and says nothing about
  the rest — exactly the failure mode DDNet's Supplementary Fig. 2 illustrates.
  Both are reported, but the log-space figure is the meaningful one.
- **Interior nodes only**, excluding the two contact nodes. Those carry the
  minority-carrier discontinuities the networks are deliberately not asked to
  fit, so including them would score the model against a target it was never
  given — and in log space two points off by tens of decades would swamp the
  other 123.

A physics check independent of DEVSIM is also reported: the spread of
`Jn + Jp` across the device. That tests whether the learned solution is
self-consistent, not whether it matches a reference.

# Measured settings

## Run 1 — equal continuity weights (`W_CONT_N = W_CONT_P = 1.0`)

20000 epochs, ~60 min on CPU (8 threads), float64, 4096 collocation points.

| epoch | total | poisson | cont | Jtot | bc | lr |
|---|---|---|---|---|---|---|
| 0 | 1.870e+02 | 3.765e-02 | 9.466e-04 | 1.960e-05 | 1.869e+02 | 1e-3 |
| 2000 | 1.475e-01 | 1.337e-01 | 4.831e-04 | 4.493e-06 | 1.334e-02 | 1e-3 |
| 4000 | 1.264e-01 | 5.709e-02 | 4.131e-02 | 3.877e-03 | 2.414e-02 | 1e-3 |
| 6000 | 6.157e-02 | 5.675e-02 | 1.946e-04 | 2.269e-07 | 4.628e-03 | 3e-4 |
| 10000 | 3.810e-02 | 3.544e-02 | 2.688e-04 | 5.378e-06 | 2.385e-03 | 9e-5 |
| 14000 | 2.996e-02 | 2.821e-02 | 2.331e-04 | 4.521e-07 | 1.519e-03 | 2.7e-5 |
| 19999 | 2.484e-02 | 2.343e-02 | 2.786e-04 | 7.871e-07 | 1.128e-03 | 8.1e-6 |

Accuracy against DEVSIM at 2.5 V:

| quantity | relative L1 |
|---|---|
| `phi` | **1.568 %** (max abs error 6.97e-3 V) |
| `n` (log10, interior) | **0.482 %** |
| `p` (log10, interior) | **1.862 %** |
| `n` (linear, interior) | 10.01 % |
| `p` (linear, interior) | 19.72 % |

Current density: PINN `Jn+Jp` = +1.7375e-04 A/cm², constant to **0.08 %**
across the device; DEVSIM's is +1.6256e-04 A/cm² (finite-differenced from the
stored profiles), so the terminal current agrees to ~6.9 %.

All three log-space errors are below DDNet's own <2 % benchmark. Two
observations shaped what came next:

- **The spread between the log and linear metrics is the point of the log
  parametrisation, not a defect.** `p` at 1.86 % in log space but 19.7 % in
  linear space means the profile is right across its whole range while the few
  largest values carry most of the linear error — precisely the distinction
  DDNet's Supplementary Fig. 2 draws.
- **The p-Net flattens near the cathode**, missing DEVSIM's sharp roll-off
  (3.91e11 vs 1.65e10 at 99 nm, a 24x overshoot). This is what motivated the
  per-equation continuity weights described above; it is a loss-scale artefact,
  not a limitation of the architecture.

Poisson dominates the converged loss (2.34e-2 of 2.48e-2, i.e. 94 %),
consistent with the source stiffness noted earlier: `n-p` climbs from 0.088 at
4 nm to 0.985 at 99.9 nm, forcing `phi_hat''` to span 0.42 .. 169.

## Run 2 — per-equation continuity weights (`W_CONT_P = 1e3`)

Identical in every other respect. This is the configuration in the code.

| epoch | total | poisson | cont_n | cont_p | Jtot | bc | lr |
|---|---|---|---|---|---|---|---|
| 0 | 1.870e+02 | 3.765e-02 | 9.464e-04 | 2.433e-07 | 1.960e-05 | 1.869e+02 | 1e-3 |
| 2000 | 1.548e-01 | 1.297e-01 | 7.519e-03 | 2.018e-07 | 1.230e-04 | 1.722e-02 | 1e-3 |
| 6000 | 6.180e-02 | 5.683e-02 | 2.061e-04 | 7.884e-08 | 2.884e-07 | 4.682e-03 | 3e-4 |
| 10000 | 3.854e-02 | 3.519e-02 | 7.163e-04 | 4.086e-08 | 1.912e-05 | 2.571e-03 | 9e-5 |
| 14000 | 2.999e-02 | 2.808e-02 | 3.077e-04 | 1.649e-08 | 6.728e-06 | 1.581e-03 | 2.7e-5 |
| 19999 | 2.488e-02 | 2.346e-02 | 2.757e-04 | 7.242e-09 | 7.416e-07 | 1.131e-03 | 8.1e-6 |

Accuracy against DEVSIM at 2.5 V, with run 1 alongside:

| quantity | run 1 | **run 2** | change |
|---|---|---|---|
| `phi` rel. L1 | 1.568 % | **1.577 %** | unchanged |
| `n` rel. L1 (log10) | 0.482 % | **0.484 %** | unchanged |
| `p` rel. L1 (log10) | 1.862 % | **0.874 %** | **2.1x better** |
| `p` rel. L1 (linear) | 19.72 % | **7.415 %** | **2.7x better** |
| `Jn+Jp` spread | 0.08 % | **0.08 %** | unchanged |

The hole weight did exactly what it was sized to do and nothing else: `p`
improved by a factor of 2-3 while `phi`, `n` and the current constancy are
untouched. The cathode overshoot at 94 nm went from 4.18e11 (3.5x too high)
to 1.504e11 against DEVSIM's 1.198e11 (1.25x).

Note `cont_p` is now ~7e-9 rather than larger: the weight did not inflate the
term, it kept the p-Net in the region where the term stays small. The 1e3
weight costs nothing in the converged loss (7e-9 x 1e3 = 7e-6, negligible
against Poisson's 2.3e-2) — it matters during training, not at the optimum.

**Remaining discrepancy, and why it is expected.** The p-Net still departs from
DEVSIM over the final few nm, heading toward ~1e11 where DEVSIM falls to
1.65e10 at 99 nm. That is the minority-carrier pin deliberately not imposed
(see *Contact carrier densities* above): with no BC there and `Jp` some five
orders below `Jn`, nothing in the loss pulls `p` down to a value that DEVSIM
itself only reaches by way of a 38-decade discontinuity. This is a consequence
of a deliberate modelling choice, not a fitting failure.

**Where the remaining error is.** Poisson is 94 % of the converged loss
(2.35e-2 of 2.49e-2), consistent with the source stiffness: `n-p` climbs from
0.088 at 4 nm to 0.985 at 99.9 nm, forcing `phi_hat''` to span 0.42 .. 169. The
`phi` error peaks mid-device (~7e-3 V near 50 nm, panel b) rather than at the
contacts, which is the signature of a curvature the network is smoothing rather
than a boundary being missed.

## Next steps

The obvious lever for the remaining `phi` error is **non-uniform collocation
weighted toward the cathode**: uniform sampling under-resolves the region
carrying nearly all the curvature. Increasing `W_CONT_P` further is *not*
indicated — `cont_p` is already at the level where the hole equation is
satisfied, and the residual `p` discrepancy is a missing boundary condition
rather than an under-weighted residual.
