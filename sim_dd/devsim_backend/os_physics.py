import math

from devsim import *
from .model_create import *
from .common_physics import *

def SetOSParameters(device, region, homo, lumo, nc300, nv300,
                     mu_n=1.0e-6, mu_p=1.0e-6, relative_permittivity=4.0,
                     f_reduced_floor=1.0e-8):
    '''
      Sets Organic Semiconductor device parameters on the specified region.

      ``homo``/``lumo`` (eV, relative to vacuum) and ``nc300``/``nv300``
      (conduction/valence band effective densities of states, cm^-3) come
      from the ``Device`` description. ``Affinity`` (electron affinity) is
      the depth of the LUMO below vacuum, and ``EG300`` (energy gap) is the
      HOMO-LUMO separation.

      ``mu_n``/``mu_p`` (cm^2/V-s) are the constant electron/hole mobilities.
      ``relative_permittivity`` is the material's dimensionless relative
      permittivity, multiplied by the vacuum permittivity parameter
      (Permittivity_0) to get Permittivity.

      ``f_reduced_floor`` (dimensionless) is the lower clamp on the reduced
      field f used by the thermionic contact model. The model contains
      exp(sqrt(f)), whose derivative diverges as f -> 0, so f is never
      allowed to reach exactly zero (which is the equilibrium condition).
      See CreateReducedField in common_physics.py. It is a no-op for devices
      that only use Ohmic contacts.
    '''
    SetUniversalParameters(device, region)

    affinity = -lumo
    eg300 = lumo - homo

    par = {
        'Permittivity'     : relative_permittivity*get_parameter(device=device, region=region, name='Permittivity_0'),
      'NC300'       : nc300,  # Conduction band effective density of states (1/cm^3)
      'NV300'       : nv300, # Valence band effective density of states (1/cm^3)
      'EG300'       : eg300,    # Energy gap (eV)
      'EGALPH'      : 2.73e-4, # Energy gap temperature coefficient (eV/K)
      'EGBETA'      : 0      , # Energy gap reference temperature (K)
      'Affinity'    : affinity      , # Electron affinity (eV)

      'MUN'      : mu_n     , # Constant mobility (cm^2/V-s)
      'MUP'      : mu_p      , # Constant mobility (cm^2/V-s)

      # Langevin
      "gammar" : 1,

      # SRH
      "taun" : 1e-5,
      "taup" : 1e-5,
      "n1" : 1e10,
      "p1" : 1e10,

      # Field-dependent thermionic contacts (CreateOSThermionicContact).
      "f_reduced_floor" : f_reduced_floor,
    }

    for k, v in par.items():
        set_parameter(device=device, region=region, name=k, value=v)

    # Derived constants for the field-dependent injection model, computed here
    # (after q/k/Permittivity are set above) rather than as node/edge models,
    # since none of them depend on a solution variable.
    q = get_parameter(device=device, region=region, name='q')
    kb = get_parameter(device=device, region=region, name='k')
    permittivity = get_parameter(device=device, region=region, name='Permittivity')
    temperature = get_parameter(device=device, name='T')
    kT = kb * temperature
    kT_over_q = kT / q

    # Coulomb capture radius r_c = q^2/(4*pi*eps*kT): the distance at which the
    # image-charge attraction equals the thermal energy. ~15.9 nm for F8BT.
    r_c = q * q / (4.0 * math.pi * permittivity * kT)
    set_parameter(device=device, region=region, name='r_c', value=r_c)

    # The reduced field is f = q*E*r_c/kT = E * (r_c / (kT/q)), so this is the
    # single scale factor converting a field (V/cm) into f.
    set_parameter(device=device, region=region, name='r_c_over_Vt',
                  value=r_c / kT_over_q)

    # Zero-field surface recombination velocity S(0) = 16*pi*eps*mu*(kT)^2/q^3,
    # per carrier. This implicitly carries the Richardson constant: the
    # prefactor C = 16*pi*eps*mu*N*(kT/q)^2 satisfies C == q*N*S(0) exactly,
    # which is what makes the injection residual vanish at equilibrium.
    common = 16.0 * math.pi * permittivity * kT * kT / (q ** 3)
    set_parameter(device=device, region=region, name='S0n', value=common * mu_n)
    set_parameter(device=device, region=region, name='S0p', value=common * mu_p)

def CreateLangevin(device, region, variables):
    '''
      Langevin recombination model in terms of generation.
    '''
    ULANG="gammar * q/Permittivity * (Electrons*Holes - NIE^2) * (MUN + MUP)"
    Gn = "-q * ULANG"
    Gp = "+q * ULANG"
    CreateNodeModel(device, region, "ULANG", ULANG)
    CreateNodeModel(device, region, "ElectronGeneration", Gn)
    CreateNodeModel(device, region, "HoleGeneration", Gp)
    for i in ("Electrons", "Holes", "T"):
        if i in variables:
            CreateNodeModelDerivative(device, region, "ULANG", ULANG, i)
            CreateNodeModelDerivative(device, region, "ElectronGeneration", Gn, i)
            CreateNodeModelDerivative(device, region, "HoleGeneration", Gp, i)
            
def CreateOSConstantMobility(device, region):
    '''
      Creates constant electron/hole mobility edge models from the MUN/MUP
      parameters, and the corresponding drift-diffusion currents Jn/Jp.
      Returns the edge model names, in the same form as CreateAroraMobilityLF,
      so it can be passed as **opts into CreateOSDriftDiffusion(Contact).
    '''
    # Step 1: put the constant mobility parameters onto edge models. MUN/MUP
    # don't depend on position or any solution variable, but CreateElectronCurrent/
    # CreateHoleCurrent expect an edge model name (they multiply it against
    # other edge quantities like EdgeInverseLength), so we can't pass the
    # parameter names "MUN"/"MUP" straight through -- they need to exist as
    # edge models first. No derivative is needed since MUN/MUP are constants.
    CreateEdgeModel(device, region, "mu_n_const", "MUN")
    CreateEdgeModel(device, region, "mu_p_const", "MUP")

    # Step 2: build the Scharfetter-Gummel current edge models Jn/Jp using
    # the constant mobility edge models from Step 1, mirroring exactly how
    # CreateAroraMobilityLF builds Jn_arora_lf/Jp_arora_lf from mu_arora_n_lf/
    # mu_arora_p_lf.
    CreateElectronCurrent(device, region, mu_n='mu_n_const', Potential="Potential", sign=-1, ElectronCurrent="Jn_const", V_t="V_t_edge")
    CreateHoleCurrent(device, region, mu_p='mu_p_const', Potential="Potential", sign=-1, HoleCurrent="Jp_const", V_t="V_t_edge")

    # Step 3: return the model names so the caller can forward them on to
    # CreateOSDriftDiffusion/CreateOSDriftDiffusionContact via **opts, same
    # convention as CreateAroraMobilityLF/CreateHFMobility.
    return {
        'mu_n' : 'mu_n_const',
        'mu_p' : 'mu_p_const',
        'Jn'   : 'Jn_const',
        'Jp'   : 'Jp_const',
    }

def CreateOSPotentialOnly(device, region):
    '''
      Creates the physical models for an Organic Semiconductor region for
      equilibrium (potential-only) simulation. Mirrors CreateSiliconPotentialOnly,
      but does not call SetSiliconParameters, since SetOSParameters has
      already set the OS-specific parameters (MUN, MUP, gammar, etc.) and we
      don't want to overwrite them with silicon values.
    '''
    # Step 1: set up the thermal voltage and density-of-states node models
    # (NC, NV, NIE, EC, EV, ...), which only depend on Potential at this stage.
    variables = ("Potential",)
    CreateVT(device, region, variables)
    CreateDensityOfStates(device, region, variables)

    # Step 2: build the equilibrium (intrinsic) carrier densities and the
    # resulting charge, assuming Boltzmann statistics. NetDoping is fixed at
    # 0 for now (set in diode_os.py), so IntrinsicCharge currently reduces to
    # Holes - Electrons at equilibrium; this will matter once doped OS
    # devices are implemented.
    for i in (
        ("IntrinsicElectrons",       "NIE*exp(Potential/V_t)"),
         ("IntrinsicHoles",           "NIE^2/IntrinsicElectrons"),
         ("IntrinsicCharge",          "kahan3(IntrinsicHoles, -IntrinsicElectrons, NetDoping)"),
         ("PotentialIntrinsicCharge", "-q * IntrinsicCharge")
    ):
        n = i[0]
        e = i[1]
        CreateNodeModel(device, region, n, e)
        CreateNodeModelDerivative(device, region, n, e, 'Potential')

    # Step 3: quasi-Fermi levels, EField/DField -- reused directly from
    # new_physics.py since they're generic and not silicon-specific.
    CreateQuasiFermiLevels(device, region, 'IntrinsicElectrons', 'IntrinsicHoles', variables)

    CreateEField(device, region)
    CreateDField(device, region)

    # Step 4: the bulk Poisson equation, driven by the intrinsic charge
    # computed above (since Electrons/Holes are not yet solution variables
    # at this equilibrium-only stage).
    equation(device=device, region=region, name="PotentialEquation", variable_name="Potential",
             node_model="PotentialIntrinsicCharge", edge_model="DField", variable_update="log_damp")


def CreateOSPotentialOnlyContact(device, region, contact, V_offset=0.0, is_circuit=False):
    '''
      Creates the potential equation at an Ohmic contact for an Organic
      Semiconductor region, neglecting band bending.

      V_offset shifts the equilibrium Potential at this contact relative to
      the applied bias -- use it to encode a built-in voltage (e.g. Vbi at
      the anode, 0 at the cathode) so the device starts with a non-flat
      band profile even at zero applied bias.
    '''
    # Step 1: make sure the bulk contact charge model exists. At this stage
    # Electrons/Holes are not yet solution variables (this runs during the
    # potential-only equilibrium solve, same as CreateSiliconPotentialOnlyContact),
    # so the charge must be expressed in terms of IntrinsicCharge -- the
    # Boltzmann-derived intrinsic densities from CreateOSPotentialOnly --
    # not the Electrons/Holes node solutions, which don't exist yet.
    if not InNodeModelList(device, region, "contactcharge_node"):
        CreateNodeModel(device, region, "contactcharge_node", "q*IntrinsicCharge")

    # Step 2: build the Dirichlet boundary condition residual for Potential.
    # The contact pins Potential to (bias - V_offset): V_offset encodes the
    # equilibrium band bending at this electrode (e.g. the built-in voltage
    # at the anode) so that zero applied bias does not imply flat-band.
    contact_model = "Potential - ({0} - {1})".format(GetContactBiasName(contact), V_offset)

    # Step 3: register the residual and its derivative w.r.t. Potential.
    # The derivative is the constant 1, since contact_model is linear in
    # Potential, so there's no need to symbolically differentiate it.
    contact_model_name = GetContactNodeModelName(contact)
    CreateContactNodeModel(device, contact, contact_model_name, contact_model)
    CreateContactNodeModel(device, contact, "{0}:{1}".format(contact_model_name, "Potential"), "1")
    if is_circuit:
        # If this contact is tied to an external circuit node, the residual
        # also depends on the bias variable itself (which is now a circuit
        # unknown rather than a fixed parameter), so register that derivative too.
        CreateContactNodeModel(device, contact, "{0}:{1}".format(contact_model_name, GetContactBiasName(contact)), "-1")

    # Step 4: tell DEVSIM to use this residual as the boundary condition for
    # PotentialEquation at this contact's nodes, while still letting charge
    # accumulate via contactcharge_node/DField so the contact's contribution
    # to the overall charge balance with the bulk is correctly accounted for.
    if is_circuit:
        contact_equation(device=device, contact=contact, name="PotentialEquation",
                         node_model=contact_model_name, edge_model="",
                         node_charge_model="contactcharge_node", edge_charge_model="DField",
                         node_current_model="", edge_current_model="", circuit_node=GetContactBiasName(contact))
    else:
        contact_equation(device=device, contact=contact, name="PotentialEquation",
                         node_model=contact_model_name, edge_model="",
                         node_charge_model="contactcharge_node", edge_charge_model="DField",
                         node_current_model="", edge_current_model="")


def CreateOSDriftDiffusionContact(device, region, contact, Jn, Jp, electron_density, hole_density, V_offset=0.0, is_circuit=False):
    '''
      Ohmic injection contact for drift diffusion in an Organic Semiconductor
      region. Pins Electrons and Holes to fixed contact densities (Dirichlet
      boundary conditions), rather than equilibrium values derived from NetDoping.

      electron_density/hole_density are the fixed densities (#/cm^3) this
      contact injects -- pass different values per contact (e.g. an anode
      with high hole/low electron density vs. a cathode with the reverse)
      so the two contacts are not electrically identical at zero bias.
    '''
    # Step 1: pin Potential at the contact, same as the potential-only case.
    CreateOSPotentialOnlyContact(device, region, contact, V_offset=V_offset, is_circuit=is_circuit)

    # Step 2: build the Dirichlet residuals for Electrons and Holes, using
    # this contact's own injection densities. Unlike the silicon contact, we
    # don't derive these from NetDoping/NIE -- we just pin them directly to
    # the densities passed in, since Ohmic injection assumes the contact can
    # supply/absorb carriers freely at those fixed densities.
    contact_electrons_model = "Electrons - {0}".format(electron_density)
    contact_holes_model = "Holes - {0}".format(hole_density)
    contact_electrons_name = "{0}nodeelectrons".format(contact)
    contact_holes_name = "{0}nodeholes".format(contact)

    # Step 3: register each residual and its derivative w.r.t. its own
    # variable. As with Potential, these are linear, so the derivative is
    # just the constant 1.
    CreateContactNodeModel(device, contact, contact_electrons_name, contact_electrons_model)
    CreateContactNodeModel(device, contact, "{0}:{1}".format(contact_electrons_name, "Electrons"), "1")

    CreateContactNodeModel(device, contact, contact_holes_name, contact_holes_model)
    CreateContactNodeModel(device, contact, "{0}:{1}".format(contact_holes_name, "Holes"), "1")

    # Step 4: use these residuals as the boundary condition for the electron
    # and hole continuity equations at the contact, while still letting the
    # current models (Jn/Jp) carry current across the contact edge into the
    # bulk -- this is how DEVSIM computes the terminal current for the I-V curve.
    if is_circuit:
        contact_equation(device=device, contact=contact, name="ElectronContinuityEquation",
                         node_model=contact_electrons_name,
                         edge_current_model=Jn, circuit_node=GetContactBiasName(contact))

        contact_equation(device=device, contact=contact, name="HoleContinuityEquation",
                         node_model=contact_holes_name,
                         edge_current_model=Jp, circuit_node=GetContactBiasName(contact))
    else:
        contact_equation(device=device, contact=contact, name="ElectronContinuityEquation",
                         node_model=contact_electrons_name,
                         edge_current_model=Jn)

        contact_equation(device=device, contact=contact, name="HoleContinuityEquation",
                         node_model=contact_holes_name,
                         edge_current_model=Jp)


def CreateOSThermionicContact(device, region, contact, Jn, Jp, electron_barrier, hole_barrier,
                               node_suffix, V_offset=0.0, is_circuit=False):
    '''
      Field-dependent thermionic injection contact (Emtage-O'Dwyer /
      Scott-Malliaras) for drift diffusion in an Organic Semiconductor
      region. Replaces CreateOSDriftDiffusionContact's Dirichlet density
      pins with an injection-current residual for each carrier, so
      Electrons/Holes become solved unknowns at this contact rather than
      pinned parameters. See Models.md for the governing equations.

      ``electron_barrier``/``hole_barrier`` are the intrinsic injection
      barriers phi_n/phi_p (eV) at this contact.

      ``node_suffix`` ("n0" for the top/low-x contact, "n1" for the
      bottom/high-x contact) orients the field and fixes the sign with which
      each carrier's injection enters its continuity equation. Per the
      per-contact current expressions in Models.md, electron injection
      enters with +1 at the bottom contact and -1 at the top contact, and
      hole injection with the opposite sign at each -- because "into the
      semiconductor" is +x at the top contact and -x at the bottom one,
      while electrons and holes carry current in opposite senses.
    '''
    # Step 1: Potential contact equation, same as the Ohmic case. Only the
    # carrier continuity equations change; the electrode is still a good
    # conductor, so Potential stays Dirichlet-pinned.
    CreateOSPotentialOnlyContact(device, region, contact, V_offset=V_offset, is_circuit=is_circuit)

    # Step 2: register this contact's barriers as region parameters, so the
    # injection expressions can reference them by name.
    phi_n_param = "{0}_phi_n".format(contact)
    phi_p_param = "{0}_phi_p".format(contact)
    set_parameter(device=device, region=region, name=phi_n_param, value=electron_barrier)
    set_parameter(device=device, region=region, name=phi_p_param, value=hole_barrier)

    # Step 3: build the per-carrier injection residuals. The signs follow the
    # four current expressions in Models.md.
    if node_suffix == "n0":       # top contact
        n_sign, p_sign = -1, +1
    else:                          # bottom contact
        n_sign, p_sign = +1, -1

    # Each residual is the FLUX BALANCE at the contact:
    #
    #     (bulk current leaving the contact edge) - (injected current) = 0
    #
    # not the injection current on its own. This distinction is the whole
    # point. DEVSIM permutes the contact node's bulk continuity row away and
    # replaces it with edge_model, so a residual of "injection = 0" would say
    # the contact injects nothing -- which is trivially satisfiable by
    # collapsing the carrier density, and is exactly the spurious
    # zero-current branch documented in Models.md. Writing the balance makes
    # the physical solution the only root: the contact must supply precisely
    # the current the bulk carries away.
    Jn_inj = CreateInjectionContact(
        device, region, contact, carrier="Electrons",
        dos_param="NC300", barrier_param=phi_n_param, s_zero_param="S0n",
        node_suffix=node_suffix, sign=n_sign, bulk_current=Jn,
    )
    Jp_inj = CreateInjectionContact(
        device, region, contact, carrier="Holes",
        dos_param="NV300", barrier_param=phi_p_param, s_zero_param="S0p",
        node_suffix=node_suffix, sign=p_sign, bulk_current=Jp,
    )

    # Step 4: wire each residual in as the contact's carrier equation.
    #
    # edge_model is the Jacobian-driving residual (the injection current);
    # edge_current_model is what get_contact_current/PrintCurrents report for
    # the I-V sweep. Only edge_model is assembled into the matrix in
    # non-circuit mode -- edge_current_model is assembled only when a
    # circuit_node is present -- so the two kwargs serve different purposes
    # and both must be given.
    #
    # The reported terminal current is the BULK drift-diffusion current (Jn/Jp)
    # arriving at the contact, not the injection expression. At a converged
    # solution the residual forces the two to be equal, so reporting the bulk
    # current makes the terminal current directly comparable with the current
    # at the opposite contact -- which is exactly the continuity check.
    if is_circuit:
        contact_equation(device=device, contact=contact, name="ElectronContinuityEquation",
                         edge_model=Jn_inj, edge_current_model=Jn,
                         circuit_node=GetContactBiasName(contact))
        contact_equation(device=device, contact=contact, name="HoleContinuityEquation",
                         edge_model=Jp_inj, edge_current_model=Jp,
                         circuit_node=GetContactBiasName(contact))
    else:
        contact_equation(device=device, contact=contact, name="ElectronContinuityEquation",
                         edge_model=Jn_inj, edge_current_model=Jn)
        contact_equation(device=device, contact=contact, name="HoleContinuityEquation",
                         edge_model=Jp_inj, edge_current_model=Jp)


def CreateOSDriftDiffusion(device, region, Jn='Jn', Jp='Jp'):
    '''
      Instantiate all equations for drift diffusion simulation
    '''
    CreateDensityOfStates(device, region, ("Potential",))
    CreateQuasiFermiLevels(device, region, "Electrons", "Holes", ("Electrons", "Holes", "Potential"))
    CreatePE(device, region)
    CreateLangevin(device, region, ("Electrons", "Holes", "Potential"))
    CreateECE(device, region, Jn)
    CreateHCE(device, region, Jp)

