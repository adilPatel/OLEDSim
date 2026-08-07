# Copyright 2013 Devsim LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from devsim import *
from .model_create import *

def SetUniversalParameters(device, region):
    universal = {
        'q' :          1.6e-19,          #, 'coul'),
      'k' :          1.3806503e-23,    #, 'J/K'),
      'Permittivity_0' :  8.85e-14     #, 'F/cm^2')
    }
    for k, v in universal.items():
        set_parameter(device=device, region=region, name=k, value=v)


def GetContactBiasName(contact):
    return "{0}_bias".format(contact)

def GetContactNodeModelName(contact):
    return "{0}nodemodel".format(contact)


def CreateVT(device, region, variables):
    '''
      Calculates the thermal voltage, based on the temperature.
      V_t : node model
      V_t_edge : edge model from arithmetic mean
    '''
    CreateNodeModel(device, region, 'V_t', "k*T/q")
    CreateArithmeticMean(device, region, 'V_t', 'V_t_edge')
    if 'T' in variables:
        CreateArithmeticMeanDerivative(device, region, 'V_t', 'V_t_edge', 'T')


def CreateDensityOfStates(device, region, variables):
    '''
      Set up models for density of states.
      Neglects Bandgap narrowing.
    '''
    eq = (
        ('NC', 'NC300 * (T/300)^1.5', ('T',)),
      ('NV', 'NV300 * (T/300)^1.5', ('T',)),
      ('NTOT', 'Donors + Acceptors', ()),
      # Band Gap Narrowing
      ('DEG', '0', ()),
      #('DEG', 'V0.BGN * (log(NTOT/N0.BGN) + ((log(NTOT/N0.BGN)^2 + CON.BGN)^(0.5)))', ()),
      ('EG', 'EG300 + EGALPH*((300^2)/(300+EGBETA) - (T^2)/(T+EGBETA)) - DEG', ('T')),
      ('NIE', '((NC * NV)^0.5) * exp(-EG/(2*V_t))*exp(DEG)', ('T')),
      ('EC', '-Potential - Affinity - DEG/2', ('Potential',)),
      ('EV', 'EC - EG + DEG/2', ('Potential', 'T')),
      ('EI', '0.5 * (EC + EV + V_t*log(NC/NV))', ('Potential', 'T')),
    )

    for (model, equation, variable_list) in eq:
        CreateNodeModel(device, region, model, equation)
        vset = set(variable_list)
        for v in variables:
            if v in vset:
                CreateNodeModelDerivative(device, region, model, equation, v)


def CreateQuasiFermiLevels(device, region, electron_model, hole_model, variables):
    '''
    Creates the models for the quasi-Fermi levels.  Assuming Boltzmann statistics.
    '''
    eq = (
        ('EFN', 'EC + V_t * log(%s/NC)' % electron_model, ('Potential', 'Electrons')),
      ('EFP', 'EV - V_t * log(%s/NV)' % hole_model, ('Potential', 'Holes')),
    )
    for (model, equation, variable_list) in eq:
        CreateNodeModel(device, region, model, equation)
        vset = set(variable_list)
        for v in variables:
            if v in vset:
                CreateNodeModelDerivative(device, region, model, equation, v)


def CreateEField(device, region):
    '''
      Creates the EField and DField.
    '''
    edge_average_model(device=device, region=region, node_model="Potential",
                       edge_model="EField", average_type="negative_gradient")
    edge_average_model(device=device, region=region, node_model="Potential",
                       edge_model="EField", average_type="negative_gradient", derivative="Potential")

def CreateDField(device, region):
    CreateEdgeModel(device, region, "DField", "Permittivity * EField")
    CreateEdgeModel(device, region, "DField:Potential@n0", "Permittivity * EField:Potential@n0")
    CreateEdgeModel(device, region, "DField:Potential@n1", "Permittivity * EField:Potential@n1")


def ReducedFieldExpression(node_suffix):
    '''
      Returns the reduced electric field ``f = q*E*r_c/kT`` at a contact, as a
      DEVSIM *expression string* to be inlined into the injection residual.

      ``f`` is the reduced field magnitude at this contact, made dimensionless
      by the Coulomb capture radius ``r_c = q^2/(4*pi*eps*kT)``. It must be
      >= 0, since the model takes sqrt(f).

      The field MAGNITUDE is used, not a signed projection. The sign of the
      potential drop across a contact edge is not a reliable indicator of
      which way the field aids injection: measured on OLED2, the top contact's
      ``P@n1 - P@n0`` is positive at equilibrium (+2.3e-3 V) but negative
      under forward bias (-4.6e-3 V at 4 V), while the bottom contact's is
      negative throughout. A signed form would therefore clamp to the floor
      exactly where injection is strongest. The Emtage-O'Dwyer /
      Scott-Malliaras model is in any case formulated in terms of the field
      magnitude that lowers the image-force saddle point.

      The magnitude is regularised as ``sqrt(dV^2 + tiny)`` rather than
      ``abs(dV)``: DEVSIM's derivative of ``abs(x)`` at x=0 involves sgn(0)
      and raises a divide-by-zero FPE, and dV=0 is reachable.

      ``node_suffix`` ("n0" for the top/low-x contact, "n1" for the
      bottom/high-x contact) selects which end of the edge is the contact; it
      no longer affects the field, but is kept for the caller's clarity.

      Returned as an expression string, not as a model, so that it can be
      inlined into the injection *contact node model* that uses it. Building
      it as a separate contact_edge_model and referencing that from a
      contact_node_model would straddle two DEVSIM namespaces -- the same
      mixing that was previously found to segfault during contact assembly
      (see Models.md) -- and DEVSIM only couples a contact node model's
      Jacobian back to the contact node's own unknowns anyway.
    '''
    if node_suffix not in ("n0", "n1"):
        raise ValueError("node_suffix must be 'n0' or 'n1', got %r" % (node_suffix,))

    # Field magnitude across the contact's own edge. EdgeInverseLength is 1/dx.
    # Regularised magnitude (see docstring) rather than abs().
    raw = ("(((Potential@n0 - Potential@n1)^2 + 1e-30)^0.5)"
           "*EdgeInverseLength")

    # f = max(f_floor, q*r_c*E/(k*T)). r_c_over_Vt == r_c/(kT/q) is
    # precomputed by SetOSParameters, so this is just a scale factor.
    #
    # The clamp is to a small POSITIVE floor, not to zero. The model contains
    # exp(sqrt(f)), whose derivative exp(sqrt(f))/(2*sqrt(f)) diverges as
    # f -> 0 -- an integrable singularity in the physics, but a hard
    # divide-by-zero for DEVSIM's symbolic differentiation, which was
    # confirmed to raise a fatal FPE at exactly f = 0. Since f = 0 is the
    # equilibrium condition (zero bias, flat bands), clamping to zero would
    # fail on the very first solve.
    #
    # f_reduced_floor = 1e-8 corresponds to a field of ~1.6e-4 V/cm, i.e.
    # ~9 orders of magnitude below any real device field, and perturbs both
    # S(E)/S(0) and exp(sqrt(f)) by ~2e-4 -- physically irrelevant -- while
    # keeping the derivative bounded at ~5e3.
    return "max(f_reduced_floor, {s}*({raw}))".format(s="r_c_over_Vt", raw=raw)


def BuildInjectionExpressions(f_expr, carrier, dos_param, barrier_param,
                              s_zero_param):
    '''
      Returns the injection-current expression for one carrier, in terms of a
      reduced-field expression ``f_expr``.

      Implements the field-dependent (Emtage-O'Dwyer / Scott-Malliaras)
      thermionic injection model documented in Models.md:

        psi   = 1/f + 1/sqrt(f) - (1/f)*sqrt(1 + 2*sqrt(f))
        S(E)  = S(0) * (1/psi^2 - f)/4
        J     = C*exp(-phi/kT)*exp(sqrt(f)) - q*n*S(E)

      with C = 16*pi*eps*mu*N*(kT/q)^2 and S(0) = 16*pi*eps*mu*(kT)^2/q^3,
      so that C == q*N*S(0) identically.

      TWO REWRITES are applied, both algebraically exact, for numerical
      reasons that were verified against 60-digit arithmetic:

      1. psi is evaluated as ``1/(1 + sqrt(f) + sqrt(1+2*sqrt(f)))``. The
         literal form above subtracts two terms that each diverge as 1/f,
         to leave a result of order 1/2 -- catastrophic cancellation. In
         double precision the literal form loses all significance below
         f ~ 1e-12 and returns *exactly zero* at f ~ 1e-16, which would make
         the 1/psi^2 in S(E) a division by zero. Since f = 0 is precisely the
         equilibrium starting condition (zero bias, flat bands), the literal
         form fails on the very first solve. The rewrite follows from
         multiplying by the conjugate: (1+s)^2 - (1+2s) = s^2 = f. It is
         cancellation-free, exact to machine precision over 30+ decades, and
         finite at f = 0 (psi = 1/2, hence S(E) = S(0), the correct
         zero-field limit) with no regularisation floor needed.

      2. The current is factored as ``q*S(E)*(n_inj - n)`` rather than as the
         difference of two independently-computed terms, where

           n_inj = N*exp(-phi/kT) * exp(sqrt(f)) * S(0)/S(E)

         is the density the contact is trying to establish. This is the same
         quantity (using C == q*N*S(0)), but it makes the residual vanish
         *identically* at equilibrium by construction, rather than by the
         cancellation of two large numbers. It also exposes the residual's
         structure to Newton: it is linear in the contact's own unknown n
         with a strictly negative slope -q*S(E), which is what keeps the
         solve off the spurious zero-current branch.

      Sign convention: the returned expression is positive when carriers flow
      from the electrode into the semiconductor.
    '''
    # psi, in the cancellation-free form (rewrite 1 above).
    sqrt_f = "({0})^0.5".format(f_expr)
    psi = "1/(1 + {s} + (1 + 2*{s})^0.5)".format(s=sqrt_f)

    # S(E)/S(0) = (1/psi^2 - f)/4, finite and equal to 1 at f = 0.
    s_ratio = "((1/({psi})^2 - ({f}))/4)".format(psi=psi, f=f_expr)

    # S(E) itself.
    s_field = "({s0}*{r})".format(s0=s_zero_param, r=s_ratio)

    # The density the contact drives towards (rewrite 2 above):
    #   n_inj = N*exp(-phi/kT)*exp(sqrt(f))/(S(E)/S(0))
    # NOTE: V_t_edge, not V_t. V_t is a NODE model; this is an edge model, so
    # it must use the edge-averaged thermal voltage. Referencing V_t here
    # makes DEVSIM evaluate the whole expression to "invalid".
    n_inj = "({N}*exp(-{phi}/V_t_edge)*exp({s})/{r})".format(
        N=dos_param, phi=barrier_param, s=sqrt_f, r=s_ratio)

    # J = q*S(E)*(n_inj - n), positive = injection into the semiconductor.
    return "q*{S}*({ninj} - {c})".format(S=s_field, ninj=n_inj, c=carrier)


def CreateInjectionContact(device, region, contact, carrier, dos_param,
                           barrier_param, s_zero_param, node_suffix, sign,
                           bulk_current,
                           variables=("Potential", "Electrons", "Holes")):
    '''
      Field-dependent thermionic injection residual for one carrier at one
      contact. Returns the name of a *region edge model*, which the caller
      wires into contact_equation as that carrier's ``edge_model``.

      ``sign`` is +1 or -1 and encodes the direction the continuity equation
      counts current at this contact, following the per-contact current
      expressions in Models.md (electrons inject in the +x sense at the
      bottom contact and the -x sense at the top contact, and vice versa for
      holes).

      WHY A REGION EDGE MODEL, not a contact node model:

      The residual needs the electric field, which is intrinsically an edge
      quantity (a potential *difference*). A contact node model has no edge
      context -- referencing Potential@n0/EdgeInverseLength from one makes
      DEVSIM evaluate the expression to "invalid", which was confirmed
      empirically. There is also no edge->node projection in DEVSIM
      (``edge_average_model`` goes the other way).

      More importantly, ContactEquation::AssembleNodeEquation places every
      Jacobian entry at the contact node's OWN row/column, so a node model
      simply cannot express the residual's dependence on the neighbouring
      node's potential. ContactEquation::AssembleEdgeEquation, by contrast,
      looks up ``emodel:var@n0`` and ``emodel:var@n1`` and contributes
      entries for both ends of the edge -- giving a Jacobian that is exactly
      consistent with the residual, field dependence included.

      Per the constraints established for this contact model (see Models.md):
      the residual must be an ordinary REGION edge model, not a
      contact_edge_model (which segfaults during contact assembly), and
      derivatives must be registered for BOTH ``@n0`` and ``@n1`` of EVERY
      solved variable, even where one side is symbolically zero, or DEVSIM's
      assembly loop hangs.

      ``bulk_current`` is the drift-diffusion current edge model (Jn/Jp). The
      residual returned is the FLUX BALANCE

          bulk_current - sign*J_injection

      rather than the injection current alone. DEVSIM permutes the contact
      node's bulk continuity row away and replaces it with this edge model,
      so a residual of "injection = 0" would assert that the contact injects
      nothing -- trivially satisfiable by collapsing the carrier density, and
      exactly the spurious zero-current branch documented in Models.md.
      Writing the balance instead makes the physical solution the only root.

      This differs from the earlier difference-based attempt (also recorded in
      Models.md as failing) in one decisive respect: there, BOTH terms could
      go to zero together, so Newton could satisfy the difference by
      shrinking the contact density. Here the injection term is
      ``q*S(E)*(n_inj - n)`` with ``n_inj`` a strictly positive,
      density-independent quantity, so as n -> 0 the injection term tends to
      the strictly positive ``q*S(E)*n_inj`` rather than to zero. The
      degenerate root is therefore not admissible.
    '''
    f_expr = ReducedFieldExpression(node_suffix)
    # Evaluate the carrier density at the contact's own end of the edge.
    carrier_at = "{0}@{1}".format(carrier, node_suffix)

    core = BuildInjectionExpressions(f_expr, carrier_at, dos_param,
                                     barrier_param, s_zero_param)

    name = "{0}_{1}_injection".format(contact, carrier)
    prefix = "" if sign > 0 else "-"
    expr = "({bulk}) - {sgn}({core})".format(
        bulk=bulk_current, sgn=prefix, core=core)

    CreateEdgeModel(device, region, name, expr)
    # Both-sided derivatives for every solved variable (see docstring).
    for var in variables:
        CreateEdgeModelDerivatives(device, region, name, expr, var)
    return name


def CreateECE(device, region, Jn):
    '''
      Electron Continuity Equation using specified equation for Jn
    '''
    NCharge = "q * Electrons"
    CreateNodeModel(device, region, "NCharge", NCharge)
    CreateNodeModelDerivative(device, region, "NCharge", NCharge, "Electrons")

    equation(device=device, region=region, name="ElectronContinuityEquation", variable_name="Electrons",
             time_node_model = "NCharge",
             edge_model=Jn, variable_update="positive", node_model="ElectronGeneration")

def CreateHCE(device, region, Jp):
    '''
      Hole Continuity Equation using specified equation for Jp
    '''
    PCharge = "-q * Holes"
    CreateNodeModel(device, region, "PCharge", PCharge)
    CreateNodeModelDerivative(device, region, "PCharge", PCharge, "Holes")

    equation(device=device, region=region, name="HoleContinuityEquation", variable_name="Holes",
             time_node_model = "PCharge",
             edge_model=Jp, variable_update="positive", node_model="HoleGeneration")

def CreatePE(device, region):
    '''
      Create Poisson Equation assuming the Electrons and Holes as solution variables
    '''
    pne = "-q*kahan3(Holes, -Electrons, NetDoping)"
    CreateNodeModel(device, region,           "PotentialNodeCharge", pne)
    CreateNodeModelDerivative(device, region, "PotentialNodeCharge", pne, "Electrons")
    CreateNodeModelDerivative(device, region, "PotentialNodeCharge", pne, "Holes")

    equation(device=device, region=region, name="PotentialEquation", variable_name="Potential",
             node_model="PotentialNodeCharge", edge_model="DField",
             time_node_model="", variable_update="log_damp")


def CreateBernoulliString(Potential="Potential", scaling_variable="V_t", sign=-1):
    '''
    Creates the Bernoulli function for Scharfetter Gummel
    sign -1 for potential
    sign +1 for energy
    scaling variable should be V_t
    Potential should be scaled by V_t in V
    Ec, Ev should scaled by V_t in eV

    returns the Bernoulli expression and its argument
    Caller should understand that B(-x) = B(x) + x
    '''

    tdict = {
        "Potential" : Potential,
      "V_t"       : scaling_variable
    }
    if sign == -1:
        vdiff="(%(Potential)s@n0 - %(Potential)s@n1)/%(V_t)s" % tdict
    elif sign == 1:
        vdiff="(%(Potential)s@n1 - %(Potential)s@n0)/%(V_t)s" % tdict
    else:
        raise NameError("Invalid Sign %s" % sign)

    Bern01 = "B(%s)" % vdiff
    return (Bern01, vdiff)


def CreateElectronCurrent(device, region, mu_n, Potential="Potential", sign=-1, ElectronCurrent="ElectronCurrent", V_t="V_t_edge"):
    '''
    Electron current
    mu_n = mobility name
    Potential is the driving potential
    '''
    EnsureEdgeFromNodeModelExists(device, region, "Potential")
    EnsureEdgeFromNodeModelExists(device, region, "Electrons")
    EnsureEdgeFromNodeModelExists(device, region, "Holes")
    if Potential == "Potential":
        (Bern01, vdiff) = CreateBernoulliString(scaling_variable=V_t, Potential=Potential, sign=sign)
    else:
        raise NameError("Implement proper call")

    tdict = {
        'Bern01' : Bern01,
      'vdiff'  : vdiff,
      'mu_n'   : mu_n,
      'V_t'    : V_t
    }

    Jn = "q*%(mu_n)s*EdgeInverseLength*%(V_t)s*kahan3(Electrons@n1*%(Bern01)s,  Electrons@n1*%(vdiff)s,  -Electrons@n0*%(Bern01)s)" % tdict

    CreateEdgeModel(device, region, ElectronCurrent, Jn)
    for i in ("Electrons", "Potential", "Holes"):
        CreateEdgeModelDerivatives(device, region, ElectronCurrent, Jn, i)

def CreateHoleCurrent(device, region, mu_p, Potential="Potential", sign=-1, HoleCurrent="HoleCurrent", V_t="V_t_edge"):
    '''
    Hole current
    '''
    EnsureEdgeFromNodeModelExists(device, region, "Potential")
    EnsureEdgeFromNodeModelExists(device, region, "Electrons")
    EnsureEdgeFromNodeModelExists(device, region, "Holes")
    if Potential == "Potential":
        (Bern01, vdiff) = CreateBernoulliString(scaling_variable=V_t, Potential=Potential, sign=sign)
    else:
        raise NameError("Implement proper call for " + Potential)

    tdict = {
        'Bern01' : Bern01,
      'vdiff'  : vdiff,
      'mu_p'   : mu_p,
      'V_t'    : V_t
    }

    Jp ="-q*%(mu_p)s*EdgeInverseLength*%(V_t)s*kahan3(Holes@n1*%(Bern01)s, -Holes@n0*%(Bern01)s, -Holes@n0*%(vdiff)s)" % tdict
    CreateEdgeModel(device, region, HoleCurrent, Jp)
    for i in ("Holes", "Potential", "Electrons"):
        CreateEdgeModelDerivatives(device, region, HoleCurrent, Jp, i)
