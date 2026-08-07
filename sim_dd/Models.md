### Overview of the Physical Models Used

This document describes some of the physical models employed by this program.

## Charge Injection and Boundary Models

# Ohmic Contacts

An Ohmic contact pins the electron and hole densities at the contact node to
fixed values (Dirichlet boundary conditions for `ElectronContinuityEquation`/
`HoleContinuityEquation`), assuming the electrode can supply or absorb
carriers freely. `Potential` is likewise pinned (Dirichlet) to the applied
bias plus a built-in offset (`V_offset`) that encodes equilibrium band
bending (e.g. the built-in voltage at the anode, 0 at the cathode).

The pinned densities can be supplied directly (`electron_density`,
`hole_density`, cm^-3) or derived from the electrode's `work_function` (eV)
against the semiconductor's HOMO/LUMO, assuming Boltzmann statistics
(`Contact.compute_densities_from_work_function`, `core/device.py`):

    electron_density = NC * exp(-(LUMO - work_function) / kT)
    hole_density     = NV * exp(-(work_function - HOMO) / kT)

i.e. the injected density falls off exponentially with how far the work
function sits below the LUMO (electrons) or above the HOMO (holes) -- the
same barrier quantity used by the thermionic model below.

**Implementation** (`devsim_backend`): `CreateOSPotentialOnlyContact` builds
the `Potential - (bias - V_offset)` residual as a `contact_node_model`.
`CreateOSDriftDiffusionContact` builds `Electrons - electron_density` /
`Holes - hole_density` residuals the same way, attached via
`contact_equation(..., node_model=..., edge_current_model=Jn/Jp)` so DEVSIM
still reports the bulk drift-diffusion current through the contact for the
I-V sweep, even though the residual itself is a simple density pin.

# Thermionic Contacts

A thermionic contact (`Contact(contact_type="thermionic")`) replaces the Ohmic
Dirichlet density pin with a *field-dependent* thermionic injection current
(Emtage-O'Dwyer / Scott-Malliaras), so the contact's carrier densities become
solved unknowns rather than fixed parameters.

## Physics

- Coulomb capture radius: `r_c = q^2/(4*pi*eps*kT)` (~15.9 nm for F8BT).
- Reduced electric field: `f = q*E*r_c/kT`.
- For brevity: `psi = 1/f + 1/sqrt(f) - (1/f)*sqrt(1 + 2*sqrt(f))`.
- Prefactor `C_{n,p} = 16*pi*eps*mu*N*(kT/q)^2`, which implicitly incorporates
  the Richardson constant (`N` = `NC` for electrons, `NV` for holes).
- Zero-field surface recombination velocity:
  `S(0) = 16*pi*eps*mu*(kT)^2/q^3` (using `mu_n` or `mu_p`).
- Field-dependent recombination velocity: `S(E) = S(0)*(1/psi^2 - f)/4`.
- `phi_{n,p}{t,b}` is the intrinsic injection barrier for electrons/holes at
  the top/bottom contact, derived from the work function exactly as in the
  Ohmic case: `phi_n = LUMO - work_function`, `phi_p = work_function - HOMO`.

The field-dependent thermionic currents are then:

    Jn (bottom) =  C_n*exp(-phi_nb/kT)*exp(sqrt(f)) - q*n*S(E)
    Jn (top)    = -C_n*exp(-phi_nt/kT)*exp(sqrt(f)) + q*n*S(E)
    Jp (bottom) = -C_p*exp(-phi_pb/kT)*exp(sqrt(f)) + q*p*S(E)
    Jp (top)    =  C_p*exp(-phi_pt/kT)*exp(sqrt(f)) - q*p*S(E)

### Deviations from the literal specification

Three deliberate changes were made, each verified numerically. They are
algebraically exact rewrites or bounded regularisations, not physics changes.

**1. `psi` is evaluated in a cancellation-free form.** As written, `psi`
subtracts two terms that each diverge as `1/f` to leave a result of order
`1/2` -- catastrophic cancellation. Checked against 60-digit arithmetic, the
literal form loses all significance below `f ~ 1e-12` and returns *exactly
zero* at `f ~ 1e-16`, which makes the `1/psi^2` in `S(E)` a division by zero.
Since `f = 0` is the equilibrium condition (zero bias, flat bands), the
literal form fails on the very first solve. Multiplying by the conjugate
gives, exactly,

    psi = 1/(1 + sqrt(f) + sqrt(1 + 2*sqrt(f)))

which is cancellation-free, accurate to machine precision over 30+ decades,
and finite at `f = 0` (`psi = 1/2`, hence `S(E) = S(0)` -- the correct
zero-field limit).

**2. The current is factored as a density difference.** Because
`C = q*N*S(0)` identically (verified: both equal 4.1736e4 for OLED2's
electrons), the current can be written

    J = q*S(E)*(n_inj - n),   n_inj = N*exp(-phi/kT)*exp(sqrt(f))*S(0)/S(E)

This is the same quantity, but the residual then vanishes *identically* at
equilibrium by construction rather than through the cancellation of two large
numbers, and it is manifestly linear in the contact's own unknown `n` with a
strictly negative slope `-q*S(E)`. Setting `f = 0` gives
`n_inj = N*exp(-phi/kT)`, i.e. exactly the density the Ohmic contact would
pin to for the same work function -- so the two contact models agree at
equilibrium, as they must.

**3. `f` uses the field magnitude and is clamped to a small positive floor.**
The model contains `exp(sqrt(f))`, whose derivative `exp(sqrt(f))/(2*sqrt(f))`
diverges as `f -> 0`. This is an integrable singularity in the physics but a
hard divide-by-zero for DEVSIM's symbolic differentiation (confirmed: it
raises a fatal FPE at exactly `f = 0`). `f` is therefore clamped below at
`f_reduced_floor = 1e-8`, which corresponds to a field of ~1.6e-4 V/cm --
some nine orders of magnitude below any real device field -- and perturbs
both `S(E)/S(0)` and `exp(sqrt(f))` by ~2e-4.

The field *magnitude* is used rather than a signed projection. The sign of the
potential drop across a contact edge is not a reliable indicator of which way
the field aids injection: measured on OLED2, the top contact's
`P@n1 - P@n0` is positive at equilibrium (+2.3e-3 V) but negative under
forward bias (-4.6e-3 V at 4 V), while the bottom contact's is negative
throughout, so a signed form would clamp to the floor exactly where injection
is strongest. The magnitude is regularised as `sqrt(dV^2 + 1e-30)` rather than
`abs(dV)`, since DEVSIM's derivative of `abs(x)` at `x=0` involves `sgn(0)`
and raises an FPE.

**On the `x_c = r_c/4` sampling point.** The specification evaluates `n`/`p` a
distance `x_c` (~3.97 nm for F8BT) from the contact. This is *not* done here,
for a structural reason: `ContactEquation::AssembleNodeEquation` places every
Jacobian entry at the contact node's own row and column, so a density sampled
at a distant node could only enter as a *lagged*, frozen quantity with no
Jacobian contribution -- precisely the kind of inconsistent Jacobian that
previously drove this solve onto spurious branches. It was measured instead
whether the distinction matters: over the 4 nm to `x_c`, the majority-carrier
(hole) density at the anode varies by 8% and the field by 2%. The model is
therefore evaluated at the contact node, trading a few percent against a fully
consistent, analytically differentiable Jacobian.

## Implementation

`ReducedFieldExpression` and `BuildInjectionExpressions` (`common_physics.py`)
assemble the expressions above as DEVSIM expression strings.
`CreateInjectionContact` registers them, and `CreateOSThermionicContact`
(`os_physics.py`) wires them into the contact equations.
`SetOSParameters` precomputes `r_c`, `r_c_over_Vt`, `S0n` and `S0p` in Python,
since none depend on a solution variable.

Two DEVSIM-specific constraints shape the implementation:

**The residual is a region `edge_model`, not a contact node model.** The model
needs the electric field, which is intrinsically an edge quantity (a potential
*difference*). A contact node model has no edge context -- referencing
`Potential@n0`/`EdgeInverseLength` from one makes DEVSIM evaluate the whole
expression to `invalid`. There is also no edge-to-node projection in DEVSIM
(`edge_average_model` goes the other way). Decisively,
`ContactEquation::AssembleEdgeEquation` looks up `emodel:var@n0` *and*
`emodel:var@n1` and contributes Jacobian entries for both ends of the edge,
so an edge model can express the residual's dependence on the neighbouring
node's potential, which a node model cannot. A `contact_edge_model` must
*not* be used -- that segfaults during contact assembly -- and derivatives
must be registered for both `@n0` and `@n1` of every solved variable, or
DEVSIM's assembly loop hangs.

Note also that inside an edge model the thermal voltage is `V_t_edge`; `V_t`
is a node model, and referencing it makes the expression evaluate to
`invalid`.

**The residual is a flux balance, not the injection current alone.** DEVSIM
permutes the contact node's bulk continuity row away and replaces it with
`edge_model`, so the residual is written

    (bulk drift-diffusion current) - (injection current) = 0

A residual of "injection = 0" would instead assert that the contact injects
nothing, which is trivially satisfiable by collapsing the carrier density --
the spurious zero-current branch that defeated earlier versions of this model.
This also differs from the earlier difference-based attempt (which failed) in
one decisive respect: there, *both* terms could go to zero together, so Newton
could satisfy the difference by shrinking the contact density. Here the
injection term is `q*S(E)*(n_inj - n)` with `n_inj` strictly positive and
density-independent, so as `n -> 0` it tends to `q*S(E)*n_inj > 0` rather than
to zero. The degenerate root is not admissible.

`edge_current_model` is set to the bulk current `Jn`/`Jp` so the reported
terminal current is directly comparable with the opposite contact's -- which
is what makes the continuity check below meaningful.

## How this ties into the solver

It is tempting to picture this as "compute the injected density `n_inj`, then
use it as the boundary condition." That is *not* what happens, and the actual
mechanism is worth spelling out, because it is what makes the contact behave
like a real electrode instead of a disguised Ohmic pin.

**There are three unknowns at a contact node -- `Potential`, `Electrons`,
`Holes` -- and only one of them still gets a textbook Dirichlet condition.**

**`Potential` is unchanged.** `CreateOSThermionicContact` calls
`CreateOSPotentialOnlyContact` first, exactly as the Ohmic contact does, and
that pins `Potential - (bias - V_offset) = 0`. The electrode is still treated
as a perfect conductor at a fixed voltage. Nothing about the injection model
touches this.

**`Electrons`/`Holes` get no boundary condition in the Dirichlet sense at
all.** There is no separate step that fixes a value and lets current be
whatever it turns out to be. Instead, DEVSIM's *continuity equation itself* --
the row that would otherwise say "divergence of bulk current equals net
generation" -- is replaced at that one node. For any ordinary interior node,
the electron row (`CreateECE`) sums the drift-diffusion current `Jn_const`
over every edge touching the node (with a sign depending on which end the
node is -- this is discrete divergence in a finite-volume scheme), and sets
that equal to the generation term:

    divergence(Jn_const) - ElectronGeneration = 0

A contact node has only one edge (it is a boundary node), so its "divergence"
is just that single edge's contribution. `contact_equation(...,
edge_model=Jn_inj)` does two things there: it deletes the ordinary bulk row
DEVSIM would otherwise install, and replaces it with the divergence of
`Jn_inj` -- the balance expression from the previous section -- instead of
`Jn_const`. So the contact's row reads:

    Jn_bulk(the one edge into the device) - Jn_injection(Electrons@n0, field) = 0

**This is a Robin (mixed) condition, not a Dirichlet one.** It relates the
density and its flux to each other, rather than fixing either independently:

- Ohmic: `Electrons@n0 - electron_density = 0` -- the *value* is fixed; current
  is whatever falls out.
- Thermionic: `Jn_bulk - Jn_injection = 0` -- neither the value nor the current
  is fixed on its own; `Electrons@n0` is whichever number makes the two
  currents equal.

Physically this is the right condition for a real electrode: it doesn't fix a
density, it fixes an *emission law*, and the density is a consequence of
balancing that law against the current the bulk is demanding.

**`Electrons@n0` is solved for exactly like every other node's density --
never computed separately and substituted in.** Newton carries one unknown
per node per variable; `Electrons@n0` is one entry in the same vector as
`Electrons@n1`, `Electrons@n2`, etc., and the row above is one equation in the
same matrix as every bulk row. Each Newton iteration updates all of them
together. There is no step where `n_inj` is computed first and then imposed;
`n_inj` is only a fixed reference number (computed once, in Python, from the
barrier and field) that appears *inside* the residual -- it is the density at
which the injection current alone would vanish, not the density the solver
uses.

That last distinction is also why the two contact models agree at `V = 0`:
at zero field, `n_inj` reduces to exactly `NC*exp(-phi/kT)` -- the same value
the Ohmic contact would have pinned to -- so at equilibrium the Robin
condition and the Dirichlet pin land on the same density. They only diverge
once current flows and the balance pulls `Electrons@n0` away from `n_inj`.

## Verification

Current continuity now holds. Across the OLED2 sweep, comparing the current
reported at the thermionic anode with that at the Ohmic cathode
(`|I_top + I_bot| / max(|I_top|, |I_bot|)`), and separately checking that
`Jn + Jp` is constant across every mesh edge:

| V | I_top (A) | I_bot (A) | rel. continuity error |
|------|---------------|----------------|------|
| 1.6  | 1.464301e-03  | -1.464301e-03  | 9.7e-08 |
| 2.0  | 2.948587e-01  | -2.948587e-01  | 2.0e-09 |
| 3.0  | 3.385784e+00  | -3.385784e+00  | 5.5e-12 |
| 4.0  | 9.571286e+00  | -9.571286e+00  | 5.4e-12 |
| 4.5  | 1.379419e+01  | -1.379419e+01  | 1.6e-11 |

The full 0.1-4.5 V sweep completes with no convergence failure.

Two caveats, both understood and bounded:

**Below turn-on (`Vbi = 2.0 V`), currents fall to 1e-19-1e-21 A** and the
ratio becomes meaningless -- it is noise divided by noise, some 1e20 times
smaller than the current at 4.5 V and well below the sweep's own
double-precision noise floor (~1.4e-15 A). The all-Ohmic OLED1 shows exactly
the same pattern in this regime, so it is not specific to the thermionic
contact.

**Between roughly 1.2 and 1.5 V the residual error is mesh discretisation
error.** It is not a solver tolerance artefact: tightening `relative_error` to
1e-14 (converging to `RelError ~1e-15`) leaves the error unchanged, as does
varying `absolute_error` over 15 orders of magnitude. Refining the mesh does
fix it, which is the defining signature of discretisation error:

| V | 125 nodes | 247 nodes | improvement |
|------|----------|----------|--------|
| 1.20 | 1.36e-01 | 3.15e-05 | 4300x |
| 1.30 | 1.43e-03 | 1.37e-06 | 1000x |
| 1.40 | 1.93e-05 | 2.46e-07 | 79x |

The default 125-node mesh under-resolves the injection layer near turn-on (the
currents themselves also shift materially there), so **for quantitative work
near turn-on, refine the mesh at the thermionic contact and confirm the
current has converged.** Above ~1.6 V the default mesh is already adequate.


## Gummel preconditioning

For devices with a thermionic contact, `DevsimBackend.solve_equilibrium`
runs a decoupled **Gummel iteration** before the coupled Newton solve:

1. Solve `PotentialEquation` alone, with carriers held fixed.
2. Solve `ElectronContinuityEquation` alone; damp the update.
3. Solve `HoleContinuityEquation` alone; damp the update.
4. Repeat until the potential stops changing.

The damping is geometric (in log space):
`new = previous^(1-alpha) * solved^alpha`, with `alpha = 0.2` by default.
Carrier densities span tens of orders of magnitude, so damping the logarithm
keeps the update proportionate across the whole device; damping the value
directly would be dominated by the largest densities.

DEVSIM cannot disable an equation in place, so each sub-solve is set up by
deleting every equation and re-creating only the one being solved.
`_snapshot_equations` captures each equation's full definition (via
`get_equation_command` / `get_contact_equation_command`) after
`setup_drift_diffusion`, which is enough to reconstruct any of them exactly;
`_restrict_equations` and `_restore_equations` do the deleting and
rebuilding. The coupled system is always restored afterwards, including on
failure, and a preconditioner failure is non-fatal -- the coupled solve then
just runs from wherever the iteration reached.

This runs **only** when a thermionic contact is present. Ohmic-only devices
(e.g. OLED1) go straight to the coupled solve, unchanged (verified: OLED1's
`_has_thermionic` stays `False` and the preconditioner is never invoked).

**It remains load-bearing.** With the injection model above in place, the
preconditioner was disabled to check whether it was still needed: OLED2's
equilibrium solve then fails outright with a convergence failure. With it, the
largest ratio between adjacent electron-density nodes in the first 20 nm is
2.05 -- an ordinary smooth profile -- and the coupled Newton solve holds that
state through the whole bias sweep.
