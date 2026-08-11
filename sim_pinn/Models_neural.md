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
- **Collocation distribution.** Symmetric `Beta(0.5, 0.5)` (the arcsine
  distribution) rather than uniform — see *Collocation sampling* below.
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

# Collocation sampling

DDNet samples collocation points uniformly. That is a poor match for this
device, because the solution's structure is concentrated at the two contacts.
Measured on the DEVSIM reference, the fraction of each field's *total
variation* lying in the outer 10 % at each end:

| field | `x < 0.1` (anode) | `x > 0.9` (cathode) |
|---|---|---|
| `u_n = -log(n_hat)` | **62.5 %** | 13.2 % |
| `u_p = -log(p_hat)` | 0.8 % | **69.0 %** |
| `\|phi_hat''\|` | 3.6 % | 26.3 % |

Two thirds of each density's structure sits in a single 10 % band, and **the
two carriers need opposite ends** — `n` rises steeply at the anode where it is
injected, `p` falls steeply at the cathode. Uniform sampling puts only 20 % of
points in those two bands combined, so the layers that dominate the solution
are the least resolved part of the domain.

> **Superseded — see the run 3 post-mortem.** The measurements in this section
> are correct, but the inference drawn from them is not: they locate where the
> *fields vary*, which is not where the *residual is hard to satisfy*. Beta
> sampling was measured and regressed every metric, and `BETA_CONCENTRATION` is
> back to `1.0` (uniform). The section is kept because the reasoning and its
> refutation are both worth having on record.

The collocation points are therefore drawn from a symmetric `Beta(a, a)` on
`[0,1]` with `a = 0.5`, whose density is U-shaped and clusters at both ends at
once. A one-sided bias toward the cathode would fix the holes and starve the
electrons; the symmetric form serves both.

| distribution | fraction in outer 10 % |
|---|---|
| uniform (`a = 1`) | 20.0 % |
| `Beta(0.7, 0.7)` | 30.4 % |
| **`Beta(0.5, 0.5)`** | **41.0 %** |
| `Beta(0.4, 0.4)` | 48.0 % |
| `Beta(0.3, 0.3)` | 56.5 % |

`a = 0.5` is the **arcsine distribution** — the limiting density of Chebyshev
nodes and the standard choice for boundary-layer clustering. It is picked from
the geometry of the problem class rather than tuned to this device, which is
what makes it generalisable: another device with contact layers at both ends
gets the right treatment with no retuning. Setting `a = 1` recovers the
previous uniform behaviour exactly.

**Implementation.** Sampled via the closed form `sin^2(pi*U/2) ~ Beta(1/2,1/2)`
for `U ~ Uniform(0,1)`, verified against a reference Beta sampler (quantiles
agree to 3-4 digits). This is preferred over `torch.distributions.Beta` because
it is a single elementwise op on the `torch.rand` call already being made: it
stays on-device, honours the float64 default (torch's Beta returns float32),
and adds no host-to-device transfer to a function that runs every epoch.
Measured cost is unchanged at 0.15 s/epoch.

The `Beta(a<1)` density diverges at 0 and 1, so points can land arbitrarily
close to the contacts — which is the intent — but the endpoints themselves are
clamped out, since `x` exactly 0 or 1 would duplicate the Dirichlet points
imposed separately in `boundary_loss()`.

**Interaction with the loss weights.** Non-uniform sampling changes what
`mean(r^2)` estimates: it is no longer the domain average but an average
weighted by the sampling density. That is precisely the desired effect here,
and it overlaps with what `W_CONT_P` was doing by hand — on a synthetic cathode
layer, switching uniform → `Beta(0.5,0.5)` raises `mean(r^2)` there by **5.7x**
on its own. The difference is that the sampling achieves it from the geometry
of the problem, whereas `W_CONT_P` is a constant reverse-engineered from one
observed failure. See *Loss weights* below.

Note that `L_Jtot` is unaffected by this reasoning: it is the *variance* of
`Jn+Jp` across the batch, a statement of constancy rather than a domain
average, and so remains valid under any sampling distribution.

# Loss

Following DDNet Eq. 11, the total loss sums interior PDE residuals and boundary
residuals, each a mean-squared error:

    L_tot = L_poisson + W_CONT_N*L_Jn + W_CONT_P*L_Jp + W_JTOT*L_Jtot + W_BC*L_BC

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
`W_CONT_N = 1.0` and `W_CONT_P = 1e3`.

## What the weight does, arithmetically

A weight is nothing more than a scalar multiplying one mean-squared residual in
the sum. Writing `r_p = Jp_hat' + R_hat` for the hole continuity residual at a
collocation point, the term is `L_Jp = mean(r_p^2)`, and the total is

    L_tot = L_poisson + 1*L_Jn + 1000*L_Jp + 1*L_Jtot + 1*L_BC

The optimiser descends `dL_tot/dtheta`, so multiplying a term by 1000
multiplies its gradient contribution by 1000. **Nothing about the term itself
changes** — not its minimum, not where it vanishes. Only how much the optimiser
*cares* about it relative to the terms it competes with.

Concretely, with `p` about 10x too high beyond 80 nm, `mean(r_p^2) = 4.7e-4`:

| weight | contribution to `L_tot` | as a share of the ~2.5e-2 total |
|---|---|---|
| `1` | 4.7e-4 | **1.9 %** — cheaper to improve Poisson instead, so it is ignored |
| `1e3` | 0.47 | **~20x the entire Poisson term** — now the cheapest thing to fix |

So the weight is a pure rescaling of one gradient channel. At convergence
`cont_p` reached 7.2e-9, contributing `7.2e-9 x 1e3 = 7.2e-6` — negligible. It
mattered during descent, not at the solution. Relatedly, a small `cont_p` early
in training is *not* evidence the weight is inert: the p-Net starts close to
satisfying hole continuity and drifts later.

**The weakness of this technique** is that `1e3` was reverse-engineered from one
observed failure, on one device, at one bias. It is a constant fitted to a
symptom, and there is no reason to expect it to transfer to a different device.

This section previously proposed Beta sampling as the better fix, on the grounds
that it addresses the *cause* — too few collocation points where the solution
varies fastest — from the geometry of the problem class. That was measured in
run 3 and **it regressed every metric**, so the claim is withdrawn: too few
points where the solution *varies* was not the cause. The genuine principled
replacement, for this weight and for `W_POISSON` together, is adaptive weighting
driven by gradient statistics rather than any constant chosen in advance (see
*Next steps*).

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

## Run 3 — Beta(0.5, 0.5) collocation sampling

Non-uniform collocation, as described in *Collocation sampling*. Everything else
is held at run 2's settings, including `W_CONT_P = 1e3`, so the sampling change
is measured as a single variable. 20000 epochs, 52 min on CPU, 0.155 s/epoch —
sampling cost is unchanged, as predicted.

**This run made every metric worse. The hypothesis in *Collocation sampling*
above is wrong for this device, and the reason is instructive.**

| epoch | total | poisson | cont_n | cont_p | Jtot | bc | lr |
|---|---|---|---|---|---|---|---|
| 0 | 1.870e+02 | 3.767e-02 | 1.243e-03 | 2.487e-07 | 2.606e-05 | 1.869e+02 | 1e-3 |
| 2000 | 2.240e-01 | 1.929e-01 | 2.421e-03 | 1.377e-07 | 4.989e-06 | 2.848e-02 | 1e-3 |
| 6000 | 1.104e-01 | 9.566e-02 | 3.768e-04 | 3.695e-08 | 9.078e-07 | 1.434e-02 | 3e-4 |
| 10000 | 7.242e-02 | 6.270e-02 | 1.981e-03 | 2.283e-08 | 1.750e-04 | 7.547e-03 | 9e-5 |
| 14000 | 5.521e-02 | 4.896e-02 | 2.469e-04 | 1.432e-08 | 1.450e-07 | 5.989e-03 | 2.7e-5 |
| 19999 | 4.424e-02 | 3.956e-02 | 1.889e-04 | 1.015e-08 | 8.163e-08 | 4.482e-03 | 8.1e-6 |

| quantity | run 2 (uniform) | **run 3 (Beta)** | change |
|---|---|---|---|
| `phi` rel. L1 | 1.577 % | **5.534 %** | **3.5x worse** |
| `phi` max abs error | 6.97e-3 V | **2.33e-2 V** | 3.3x worse |
| `n` rel. L1 (log10) | 0.484 % | **0.738 %** | 1.5x worse |
| `p` rel. L1 (log10) | 0.874 % | **1.531 %** | 1.8x worse |
| `Jn+Jp` spread | 0.08 % | **0.06 %** | unchanged |
| converged Poisson term | 2.35e-2 | **3.96e-2** | 1.7x worse |

The loss is worse at *every* logged epoch, not just at the end, so this is not a
matter of needing longer — the run is on a worse trajectory from epoch 2000.

# Why it failed: the diagnosis was right, the inference from it was not

The *Collocation sampling* section above justifies `Beta(0.5,0.5)` by where the
**total variation of the densities** lives. That measurement is correct. The
error was treating it as a proxy for where the **loss** needs resolution, and
those are different places. Two measurements pin it down.

First, what the sampling actually changes (200k samples, `|phi_hat''|`
interpolated from the reference):

| distribution | mean `\|phi_hat''\|` at sampled points | fraction of points in mid-device `[0.2, 0.8]` |
|---|---|---|
| uniform | 41.0 | **59.9 %** |
| `Beta(0.5,0.5)` | 48.3 | **41.0 %** |

Beta buys an 18 % increase in the curvature it sees, and pays for it by removing
a third of the mid-device points.

Second, that mid-device region is exactly where run 2's error already was. Run 2
noted this and it was recorded above without its consequence being drawn: *"the
`phi` error peaks mid-device (~7e-3 V near 50 nm, panel b) rather than at the
contacts, which is the signature of a curvature the network is smoothing rather
than a boundary being missed."* Run 3's pointwise profile confirms the
prediction that follows — the contacts are now fitted well and the error has
migrated into the thinned-out middle:

| x [nm] | `phi` PINN | `phi` DEVSIM | error |
|---|---|---|---|
| 4 | +0.46744 | +0.46603 | 1.4e-3 |
| **48** | **+0.17179** | **+0.15043** | **2.1e-2** |
| **58** | **+0.12054** | **+0.09728** | **2.3e-2** |
| 94 | -0.00145 | -0.00585 | 4.4e-3 |

So the sampling change worked as designed and the design was aimed at the wrong
target. **Contact resolution was not the binding constraint; mid-device
curvature was.** Poisson stayed 89 % of the converged loss (3.96e-2 of 4.42e-2)
— the same diagnosis as runs 1 and 2, now with a larger value.

The general lesson, worth keeping because it applies to the next idea as well:
*where a field varies most* and *where the residual is hardest to satisfy* are
not the same set, and only the second one justifies spending collocation points.
The densities' variation is at the contacts, but their log parametrisation
already handles that — `u_n` and `u_p` are smooth and O(10) there. Poisson has
no such reparametrisation, and its difficulty is spread across the device.

Note also that `cont_p` fell to 1.0e-8 (run 2: 7.2e-9) and `Jtot` spread stayed
at 0.06 %, so the claim that Beta sampling would *partly substitute* for
`W_CONT_P` is untested by this run rather than confirmed: the hole equation was
comfortably satisfied in both.

**Status: reverted.** `BETA_CONCENTRATION` is restored to `1.0` (uniform) as the
default, with the Beta path retained since the implementation is sound and the
argument may hold for a device whose difficulty really is at the contacts.

## Run 4 — `W_POISSON = 10`, uniform sampling

Run 3's post-mortem says the binding constraint is the Poisson residual and that
it is not a sampling problem. The direct test of that claim is to leave sampling
uniform and simply weight Poisson up. Everything else is held at run 2's
settings (`W_CONT_P = 1e3`, `BETA = 1.0`, 4096 points, 20000 epochs); the loss is

    L_tot = 10*L_poisson + L_Jn + 1e3*L_Jp + L_Jtot + L_BC

20000 epochs, 70 min (sharing a CPU with the run 5 variant; ~52 min solo).

**This is the best configuration measured so far, by a wide margin.**

The `poisson` column below is the *raw, unweighted* `mean(r_poisson^2)`, so it is
directly comparable with runs 1-3. The `total` column is **not** comparable, as
it contains the 10x factor.

| epoch | total | poisson (raw) | cont_n | cont_p | Jtot | bc | lr |
|---|---|---|---|---|---|---|---|
| 0 | 1.905e+02 | 3.966e-02 | 7.511e-03 | 2.396e-07 | 2.858e-04 | 1.901e+02 | 1e-3 |
| 2000 | 3.470e-01 | 2.219e-02 | 6.768e-03 | 8.794e-08 | 7.146e-05 | 1.182e-01 | 1e-3 |
| 6000 | 1.023e-01 | 7.349e-03 | 2.575e-03 | 3.768e-08 | 1.106e-05 | 2.617e-02 | 3e-4 |
| 10000 | 3.255e-02 | 2.398e-03 | 9.075e-04 | 1.527e-08 | 1.994e-05 | 7.623e-03 | 9e-5 |
| 14000 | 1.814e-02 | 1.523e-03 | 2.179e-04 | 5.800e-09 | 1.765e-06 | 2.685e-03 | 2.7e-5 |
| 18000 | 1.154e-02 | 9.779e-04 | 2.171e-04 | 3.038e-09 | 7.437e-08 | 1.536e-03 | 8.1e-6 |
| 19999 | 9.053e-03 | **7.541e-04** | 3.040e-04 | 2.427e-09 | 3.352e-07 | 1.205e-03 | 8.1e-6 |

Accuracy against DEVSIM at 2.5 V, against the previous best:

| quantity | run 2 (best prior) | run 3 (Beta) | **run 4 (`W_POISSON=10`)** | vs run 2 |
|---|---|---|---|---|
| `phi` rel. L1 | 1.577 % | 5.534 % | **0.478 %** | **3.3x better** |
| `n` rel. L1 (log10) | 0.484 % | 0.738 % | **0.316 %** | **1.5x better** |
| `p` rel. L1 (log10) | 0.874 % | 1.531 % | **0.886 %** | unchanged |
| `n` rel. L1 (linear) | ~10 % | 14.641 % | **4.015 %** | **2.5x better** |
| `p` rel. L1 (linear) | 7.415 % | 18.045 % | **3.520 %** | **2.1x better** |
| raw Poisson term | 2.35e-2 | 3.96e-2 | **7.54e-4** | **31x lower** |
| `Jn+Jp` spread | 0.08 % | 0.06 % | 0.22 % | slightly worse |
| **terminal current vs DEVSIM** | 6.9 % | — | **0.26 %** | **26x better** |

The terminal current is the headline. Runs 1-3 all landed ~7 % away from
DEVSIM's +1.6256e-04 A/cm²; run 4 gives +1.6298e-04 A/cm², i.e. **0.26 %**. That
is the number the device is actually characterised by, and it was the quantity
least improved by everything tried before.

# Why this worked, and why it is not merely "tuning a weight"

The obvious objection is that this is the same reverse-engineered-constant
technique criticised under *Loss weights*, and it is worth being precise about
why it is not the same thing.

`W_CONT_P = 1e3` was fitted to a *symptom*: the p-Net flattened, so the hole
term was multiplied until it stopped. `W_POISSON = 10` follows from a
*measurement* that was already in this document across three runs — Poisson was
94 %, 94 % and 89 % of the converged loss respectively, while being the term
whose residual is intrinsically hardest (`phi_hat''` spans 0.83..169). A term
that dominates the loss but is still the least converged is under-weighted
relative to its difficulty, not over-weighted by its share.

Two checks that it is a genuinely better solution and not a redistribution of
error:

- **The BC term improved too**, 1.14e-2 at epoch 8000 down to 1.21e-3 at
  convergence, rather than being starved to buy the Poisson reduction. Had the
  weight simply traded one term for another, this is where it would show.
- **The accuracy metrics moved with the residual.** A lower residual is only
  meaningful if the DEVSIM-scored error follows, and `phi`, `n` and both linear
  density errors all improved together. (Run 3 is the cautionary case: a
  plausible loss-side story that pointed the wrong way.)

`W_BC = 1.0` remains correct — this is not a contradiction of the earlier
finding that raising `W_BC` is harmful. Raising `W_BC` upweights a term that is
already easy to satisfy (a two-point Dirichlet match); raising `W_POISSON`
upweights the one that is not.

**Caveat, stated plainly:** `10` itself is not derived, only its direction is.
The principled version of this is adaptive weighting that sets the balance from
the gradient statistics during training rather than from a constant chosen in
advance — see *Next steps*.

The one metric that did not improve is the `Jn+Jp` spread (0.08 % → 0.22 %),
which is expected: `W_JTOT` was left at 1.0 while Poisson went to 10, so the
constancy constraint is now relatively 10x weaker. At 0.22 % it remains a
well-satisfied constraint, and the terminal current it produces is 26x closer to
DEVSIM, so nothing was actually lost.

The `p` log error is unchanged at ~0.886 %, consistent with the run 2 finding
that the remaining hole discrepancy is the *missing minority boundary condition*
at the cathode — a modelling choice that no loss weight can address.

## Run 5 — Fourier-feature input embedding (negative result)

Run 3's post-mortem established that the difficulty is the Poisson residual.
Run 4 addressed that by reweighting. Run 5 tests the other candidate
explanation, which was that the difficulty is one of *representation*: a tanh
FCNN under Xavier init emits output slopes of order 1-10, and the boundary
layers in `u_n`, `u_p` reach slopes of ~540 in scaled units, so the optimiser
must inflate first-layer weights by ~2 decades to reach them.

The standard fix is a fixed random Fourier-feature embedding (Tancik et al.
2020; applied to stiff PINNs by Wang, Wang & Perdikaris 2021), which supplies
those frequencies at initialisation instead:

    x -> [sin(2*pi*B*x), cos(2*pi*B*x)],    B ~ N(0, sigma^2),  sigma = 1.0

32 frequencies, so the first layer takes 64 inputs; 50,115 parameters against
the baseline's 38,019. Uniform sampling, `W_POISSON = 1`, everything else at run
2's settings. 20000 epochs, 86 min.

**It is worse than the plain FCNN on every accuracy metric.**

| quantity | run 2 (plain FCNN) | **run 5 (Fourier)** | run 4 (best) |
|---|---|---|---|
| `phi` rel. L1 | 1.577 % | **6.956 %** | 0.478 % |
| `n` rel. L1 (log10) | 0.484 % | **0.967 %** | 0.316 % |
| `p` rel. L1 (log10) | 0.874 % | **2.746 %** | 0.886 % |
| `p` rel. L1 (linear) | 7.415 % | **41.170 %** | 3.520 % |
| raw Poisson term | 2.35e-2 | **1.05e-2** | 7.54e-4 |
| terminal current vs DEVSIM | 6.9 % | **33.0 %** | 0.26 % |

The terminal current is the clearest failure: +1.0900e-04 A/cm² against DEVSIM's
+1.6256e-04, a 33 % shortfall, by far the worst of any run.

# Why the embedding hurt, and what the ~540 slopes actually mean

Note first the diagnostic trap this run walked into. Run 5's raw Poisson term
(1.05e-2) is *better* than run 2's (2.35e-2), while its `phi` error is **4.4x
worse**. A lower residual on the sampled collocation points bought a worse
solution — the same lesson as run 3, and the reason accuracy is always scored
against DEVSIM here rather than inferred from the loss.

The mechanism is visible from epoch 0: `cont_n` starts at **2.36e4**, against
1.24e-3 for the plain FCNN — seven orders of magnitude worse. The continuity
residual contains `n_hat' = -u_n'*n_hat`, so a high-frequency `u_n` at
initialisation produces enormous spurious currents. The run spends most of its
budget recovering from its own starting point (`cont_n` is still 1.46 at epoch
2000, and the `bc` term only reaches 1e-5 by relaxing everything else).

The deeper error is in the premise. The ~540 slopes are real but they are
confined to the outer **0.5 nm** at each contact — the last 1-2 mesh cells,
immediately adjacent to the minority-carrier pins the networks are *deliberately
not asked to fit* (see *Contact carrier densities*). They are the reference
approaching a discontinuity, not structure the network needs to represent. Away
from those cells the fields are smooth, and `phi` in particular is a gentle ramp
whose difficulty is large-but-**low-frequency** curvature (`phi_hat''` up to 169,
but no oscillation anywhere in the domain).

Fourier features are the right tool for genuinely high-frequency targets. This
device has none. The embedding added spectral capacity the problem cannot use
and destroyed the smooth-function prior that made the plain tanh FCNN a good fit
in the first place.

**Recorded as a dead end.** The `FourierFCNN` class is not added to
`forward_demo.py`; it lived only in the experiment harness.

## Run 6 — `W_POISSON = 30` (sweep)

Run 4 established the direction but not the magnitude. This is the next sweep
point, identical in every other respect. 20000 epochs, 50 min.

| epoch | total | poisson (raw) | cont_n | cont_p | Jtot | bc | lr |
|---|---|---|---|---|---|---|---|
| 2000 | 4.082e-01 | 6.353e-03 | 2.936e-02 | 6.757e-08 | 1.269e-03 | 1.869e-01 | 1e-3 |
| 6000 | 1.100e-01 | 2.195e-03 | 3.529e-03 | 3.671e-08 | 1.527e-05 | 4.062e-02 | 3e-4 |
| 10000 | 5.570e-02 | 7.074e-04 | **1.745e-02** | 1.710e-08 | 2.051e-04 | 1.680e-02 | 9e-5 |
| 14000 | 4.155e-02 | 4.651e-04 | **1.823e-02** | 6.488e-09 | 5.901e-05 | 9.300e-03 | 2.7e-5 |
| 18000 | 1.385e-02 | 3.608e-04 | 7.525e-04 | 3.411e-09 | 1.791e-06 | 2.267e-03 | 8.1e-6 |
| 19999 | 1.080e-02 | **2.815e-04** | 6.492e-04 | 2.689e-09 | 1.185e-06 | 1.702e-03 | 8.1e-6 |

| quantity | run 4 (`W_POISSON=10`) | **run 6 (`=30`)** |
|---|---|---|
| `phi` rel. L1 | 0.478 % | **0.416 %** |
| `n` rel. L1 (log10) | 0.316 % | **0.311 %** |
| `p` rel. L1 (log10) | 0.886 % | 0.913 % |
| `n` rel. L1 (linear) | 4.015 % | 4.247 % |
| `p` rel. L1 (linear) | 3.520 % | **3.277 %** |
| raw Poisson term | 7.54e-4 | **2.82e-4** |
| `Jn+Jp` spread | 0.22 % | **0.05 %** |
| terminal current vs DEVSIM | 0.26 % | **0.67 %** |

**The two are close to indistinguishable.** `phi` is 13 % better at 30 and the
current constancy recovers to 0.05 % (better than every earlier run), but the
terminal current is 2.6x further from DEVSIM (0.67 % vs 0.26 %) and `p` in log
space is marginally worse. Neither dominates; the curve is flat between 10 and
30, which is itself the useful finding — the result is **not** sensitive to the
exact value, so `W_POISSON = 10` is not a knife-edge constant.

Worth noting for anyone reading the loss traces: `cont_n` shows large transient
excursions at 30 (1.75e-2 and 1.82e-2 at epochs 10000 and 14000, against
9.08e-4 for run 4 at epoch 10000) before settling to 6.49e-4. Upweighting
Poisson does destabilise electron continuity in mid-training, and the effect is
stronger at 30 than at 10. It resolved here, but it is the mechanism that will
eventually bound the weight from above — which is what the `W_POISSON = 100`
sweep point is testing.

## Runs 7-8 and the complete `W_POISSON` sweep

Runs 4 and 6 established the direction and showed 10 and 30 were near-tied, so
the sweep was completed at 3 and 100. All five points are identical apart from
the weight: uniform sampling, `W_CONT_P = 1e3`, 4096 points, 20000 epochs.

| `W_POISSON` | `phi` rel. L1 | `n` (log10) | `p` (log10) | `n` (lin) | `p` (lin) | raw Poisson | `Jn+Jp` spread | current vs DEVSIM |
|---|---|---|---|---|---|---|---|---|
| 1 (run 2) | 1.577 % | 0.484 % | **0.874 %** | ~10 % | 7.415 % | 2.35e-2 | 0.08 % | 6.9 % |
| 3 (run 7) | 0.620 % | 0.325 % | 0.898 % | 4.442 % | 3.697 % | 7.35e-4 | 0.14 % | **0.44 %** |
| 10 (run 4) | 0.478 % | 0.316 % | 0.886 % | 4.015 % | 3.520 % | 7.54e-4 | 0.22 % | 0.26 % |
| 30 (run 6) | 0.416 % | 0.311 % | 0.913 % | 4.247 % | 3.277 % | 2.82e-4 | 0.05 % | 0.67 % |
| **100 (run 8)** | **0.222 %** | **0.307 %** | 0.920 % | **3.770 %** | **3.169 %** | **3.48e-5** | **0.02 %** | 0.78 % |

**`W_POISSON = 100` is the best configuration measured.** It wins `phi` (0.222 %,
a **7.1x** improvement on run 2 and 2.2x on the next best), `n` in both metrics,
`p` in linear space, the raw Poisson residual (**675x** below run 2), and the
current constancy (0.02 %).

Every metric except two improves monotonically with the weight across the whole
sweep. The exceptions are worth stating rather than burying:

- **`p` in log space is flat-to-slightly-worse** (0.874 % → 0.920 %). This is the
  minority-carrier boundary condition again, and it is the one quantity no loss
  weight touches — exactly as run 4 predicted and for the reason given under
  *Contact carrier densities*. The variation across the whole sweep is 5 %
  relative, i.e. noise against the 3-7x moves elsewhere.
- **The terminal current is non-monotonic**: 6.9 % → 0.44 % → 0.26 % → 0.67 % →
  0.78 %, with the best value at `W = 10`. All of 3-100 are within 1 % and the
  ordering among them is not obviously meaningful at a single seed, but it is
  the one respect in which 100 is not the best point, and it is the quantity the
  device is characterised by. See the caveat below.

# The mid-training `cont_n` transients were a red herring

Runs 6 and 8 both show large `cont_n` excursions in mid-training — peaks of
7.67e-1 and 6.72e-1 respectively, against 6.77e-3 for run 4 — which at epoch
8000 looked like the weight destabilising electron continuity and bounding
`W_POISSON` from above. Measured at matched epochs, the trend was clean:

| `W_POISSON` | Poisson @8000 | `cont_n` @8000 | `cont_n` peak |
|---|---|---|---|
| 3 | 3.84e-3 | 7.27e-3 | 3.94e-2 |
| 10 | 4.35e-3 | 2.77e-3 | 6.77e-3 |
| 30 | 1.27e-3 | 7.03e-3 | 7.67e-1 |
| 100 | 1.85e-4 | 2.05e-2 | 6.72e-1 |

**That reading was wrong.** The excursions all occur during the `lr = 1e-3` and
`3e-4` phases and vanish once the schedule decays: by epoch 18000, `W = 100` has
`cont_n = 9.35e-5`, the *lowest* of any run, and it converges to 9.89e-5. The
transients are an interaction between the weight and the early learning rate,
not a stability ceiling — a larger `W_POISSON` makes the loss surface stiffer,
so the same step size overshoots, and the decay fixes it.

The general point: **a mid-training loss trace is not evidence about the
converged solution**, particularly under a decaying schedule. This is the third
time in this series that a plausible loss-side inference pointed the wrong way
(run 3's sampling argument, run 5's lower-Poisson-worse-`phi`, and now this), and
the discipline that caught all three is the same — score against DEVSIM, at
convergence, and treat the residual as a description of the mechanism only.

# Caveats on adopting 100

Stated plainly, because the sweep is a single seed and the improvements are
real but the extrapolation is not established:

1. **The sweep has not turned over.** 100 is the best point measured, not a
   measured optimum — the curve is still improving at the edge of the range. The
   honest statement is "≥100 is better than ≤30", and 300/1000 are untested.
2. **The terminal current disagrees with the other metrics**, peaking at
   `W = 10`. With `phi` improving 2x from 10 to 100 while the current worsens
   0.5 %, the balance favours 100, but a device-engineering use that cares only
   about terminal current has a defensible reason to prefer 10.
3. **Single seed.** None of these runs is repeated, and the `p` log-space
   differences (0.874-0.920 %) are almost certainly within seed noise. The `phi`
   and Poisson trends are far too large to be noise; the fine ordering is not.

`W_POISSON = 100` is adopted in `forward_demo.py` on the strength of the `phi`,
`n`, Poisson and current-constancy improvements, all of which are large and
monotonic across a 100x range of the weight.

## Next steps

Done, and recorded above:

- ~~Run 3 to convergence~~ — it **regressed**; reverted to uniform sampling.
- ~~Sweep `W_POISSON`~~ — five points, 1 to 100. `W_POISSON = 100` adopted.
- ~~Fourier-feature embedding~~ — tested (run 5) and **worse on every metric**;
  a dead end for this device, and not added to `forward_demo.py`.

Open, in rough priority order:

1. **Extend the sweep to 300 and 1000.** The curve had not turned over at 100,
   so the optimum is not yet bracketed. This is the cheapest outstanding item
   and it decides whether 100 is a resting point or just the edge of the range
   that happened to be tested. Watch the terminal current specifically: it is
   already drifting the wrong way (0.26 % at 10 → 0.78 % at 100) while `phi`
   improves, so the two metrics may cross.
2. **Repeat the best configuration at 2-3 seeds.** Everything above is a single
   seed. The large trends (`phi` 7.1x, Poisson 675x) are far outside plausible
   seed noise, but the fine ordering between 10/30/100 on `p` and on the
   terminal current is not, and it is currently being read as signal.
3. **Replace the hand-set weights with adaptive weighting.** `W_POISSON` and
   `W_CONT_P` are both constants chosen in advance, and the sweep above shows
   how much the answer depends on one of them. The learning-rate-annealing
   scheme of Wang, Teng & Perdikaris (2021) sets each weight from the running
   ratio of gradient magnitudes, deriving both from the training dynamics and
   removing the two magic numbers together. This is the principled end-point of
   what runs 4-8 establish empirically, and it subsumes items 1 and 4.
4. **Lower `W_CONT_P`** (1, 1e1, 1e2). Still *untested* rather than answered:
   `cont_p` has been ~1e-9 to 1e-8 at convergence in every run to date, so the
   hole equation is comfortably satisfied throughout and no run has probed
   whether the weight is still doing work.
5. **Revisit the lr schedule now that `W_POISSON` is large.** The mid-training
   `cont_n` excursions at `W >= 30` are a weight/step-size interaction that the
   decay eventually cleans up; a lower initial lr, or a warmup, might avoid
   spending several thousand epochs recovering from them.
6. **Soften the cathode hole density** in `devsim_reference_oled1.py`, the way
   the bottom electron density was softened for `phi`. The residual `p`
   discrepancy in the last few nm is a *missing boundary condition*, and the
   sweep is now positive evidence for that reading rather than just an argument:
   `p` in log space sat at 0.874-0.920 % across a 100x range of `W_POISSON`
   while `phi` moved 7x. No loss weight touches it, because it is not a
   weighting problem.
