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
from .common_physics import *

def SetSiliconParameters(device, region):
    '''
      Sets Silicon device parameters on the specified region.
    '''

    SetUniversalParameters(device, region)

##D. B. M. Klaassen, J. W. Slotboom, and H. C. de Graaff, "Unified apparent bandgap narrowing in n- and p-type Silicon," Solid-State Electronics, vol. 35, no. 2, pp. 125-29, 1992.
    par = {
        'Permittivity'     : 11.1*get_parameter(device=device, region=region, name='Permittivity_0'),
      'NC300'       : 2.8e19,  # '1/cm^3'
      'NV300'       : 3.1e19, # '1/cm^3'
      'EG300'       : 1.12,    # 'eV'
      'EGALPH'      : 2.73e-4, # 'eV/K'
      'EGBETA'      : 0      , # 'K'
      'Affinity'    : 4.05      , # 'K'
      # Canali model
      'BETAN0'      : 2.57e-2, # '1'
      'BETANE'      : 0.66,    # '1'
      'BETAP0'      : 0.46,    # '1'
      'BETAPE'      : 0.17,    # '1'
      'VSATN0'      : 1.43e9,
      'VSATNE'      : -0.87,
      'VSATP0'      : 1.62e8,
      'VSATPE'      : -0.52,
      # Arora model
      'MUMN'  : 88,
      'MUMEN' : -0.57,
      'MU0N'  : 7.4e8,
      'MU0EN' : -2.33,
      'NREFN' : 1.26e17,
      'NREFNE' : 2.4,
      'ALPHA0N' : 0.88,
      'ALPHAEN' : -0.146,
      'MUMP'  : 54.3,
      'MUMEP' : -0.57,
      'MU0P'  : 1.36e8,
      'MU0EP' : -2.23,
      'NREFP' : 2.35e17,
      'NREFPE' : 2.4,
      'ALPHA0P' : 0.88,
      'ALPHAEP' : -0.146,
      # SRH
      "taun" : 1e-5,
      "taup" : 1e-5,
      "n1" : 1e10,
      "p1" : 1e10,
      # TEMP
      "T" : 300
    }

    for k, v in par.items():
        set_parameter(device=device, region=region, name=k, value=v)

def CreateSiliconPotentialOnly(device, region):
    '''
      Creates the physical models for a Silicon region for equilibrium simulation.
    '''

    variables = ("Potential",)
    CreateVT(device, region, variables)
    CreateDensityOfStates(device, region, variables)

    SetSiliconParameters(device, region)

    # require NetDoping
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

    CreateQuasiFermiLevels(device, region, 'IntrinsicElectrons', 'IntrinsicHoles', variables)

    CreateEField(device, region)
    CreateDField(device, region)

    equation(device=device, region=region, name="PotentialEquation", variable_name="Potential",
             node_model="PotentialIntrinsicCharge", edge_model="DField", variable_update="log_damp")

def CreateSiliconPotentialOnlyContact(device, region, contact, is_circuit=False):
    '''
      Creates the potential equation at the contact
      if is_circuit is true, than use node given by GetContactBiasName
    '''
    if not InNodeModelList(device, region, "contactcharge_node"):
        CreateNodeModel(device, region, "contactcharge_node", "q*IntrinsicCharge")

    celec_model = "(1e-10 + 0.5*abs(NetDoping+(NetDoping^2 + 4 * NIE^2)^(0.5)))"
    chole_model = "(1e-10 + 0.5*abs(-NetDoping+(NetDoping^2 + 4 * NIE^2)^(0.5)))"
    contact_model = "Potential -{0} + ifelse(NetDoping > 0, \
    -V_t*log({1}/NIE), \
    V_t*log({2}/NIE))".format(GetContactBiasName(contact), celec_model, chole_model)

    contact_model_name = GetContactNodeModelName(contact)
    CreateContactNodeModel(device, contact, contact_model_name, contact_model)
    CreateContactNodeModel(device, contact, "{0}:{1}".format(contact_model_name,"Potential"), "1")
    if is_circuit:
        CreateContactNodeModel(device, contact, "{0}:{1}".format(contact_model_name,GetContactBiasName(contact)), "-1")

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


def CreateSRH(device, region, variables):
    '''
      Shockley Read hall recombination model in terms of generation.
    '''
    USRH="(Electrons*Holes - NIE^2)/(taup*(Electrons + n1) + taun*(Holes + p1))"
    Gn = "-q * USRH"
    Gp = "+q * USRH"
    CreateNodeModel(device, region, "USRH", USRH)
    CreateNodeModel(device, region, "ElectronGeneration", Gn)
    CreateNodeModel(device, region, "HoleGeneration", Gp)
    for i in ("Electrons", "Holes", "T"):
        if i in variables:
            CreateNodeModelDerivative(device, region, "USRH", USRH, i)
            CreateNodeModelDerivative(device, region, "ElectronGeneration", Gn, i)
            CreateNodeModelDerivative(device, region, "HoleGeneration", Gp, i)

def CreateSiliconDriftDiffusion(device, region, mu_n="mu_n", mu_p="mu_p", Jn='Jn', Jp='Jp'):
    '''
      Instantiate all equations for drift diffusion simulation
    '''
    CreateDensityOfStates(device, region, ("Potential",))
    CreateQuasiFermiLevels(device, region, "Electrons", "Holes", ("Electrons", "Holes", "Potential"))
    CreatePE(device, region)
    CreateSRH(device, region, ("Electrons", "Holes", "Potential"))
    CreateECE(device, region, Jn)
    CreateHCE(device, region, Jp)


def CreateSiliconDriftDiffusionContact(device, region, contact, Jn, Jp, is_circuit=False):
    '''
      Restrict electrons and holes to their equilibrium values
      Integrates current into circuit
    '''
    CreateSiliconPotentialOnlyContact(device, region, contact, is_circuit)

    celec_model = "(1e-10 + 0.5*abs(NetDoping+(NetDoping^2 + 4 * NIE^2)^(0.5)))"
    chole_model = "(1e-10 + 0.5*abs(-NetDoping+(NetDoping^2 + 4 * NIE^2)^(0.5)))"
    contact_electrons_model = "Electrons - ifelse(NetDoping > 0, {0}, NIE^2/{1})".format(celec_model, chole_model)
    contact_holes_model = "Holes - ifelse(NetDoping < 0, +{1}, +NIE^2/{0})".format(celec_model, chole_model)
    contact_electrons_name = "{0}nodeelectrons".format(contact)
    contact_holes_name = "{0}nodeholes".format(contact)

    CreateContactNodeModel(device, contact, contact_electrons_name, contact_electrons_model)
    CreateContactNodeModel(device, contact, "{0}:{1}".format(contact_electrons_name, "Electrons"), "1")

    CreateContactNodeModel(device, contact, contact_holes_name, contact_holes_model)
    CreateContactNodeModel(device, contact, "{0}:{1}".format(contact_holes_name, "Holes"), "1")

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


def CreateAroraMobilityLF(device, region):
    '''
      Creates node mobility models and then averages them on edge
      Uses model from Muller and Kamins
      Add T derivative dependence later
    '''
    models = (
        ('Tn', 'T/300'),
      ('mu_arora_n_node',
            'MUMN * pow(Tn, MUMEN) + (MU0N * pow(T, MU0EN))/(1 + pow((NTOT/(NREFN*pow(Tn, NREFNE))), ALPHA0N*pow(Tn, ALPHAEN)))'),
      ('mu_arora_p_node',
            'MUMP * pow(Tn, MUMEP) + (MU0P * pow(T, MU0EP))/(1 + pow((NTOT/(NREFP*pow(Tn, NREFPE))), ALPHA0P*pow(Tn, ALPHAEP)))')
    )

    for k, v in models:
        CreateNodeModel(device, region, k, v)
    CreateArithmeticMean(device, region, 'mu_arora_n_node', 'mu_arora_n_lf')
    CreateArithmeticMean(device, region, 'mu_arora_p_node', 'mu_arora_p_lf')
    CreateElectronCurrent(device, region, mu_n = 'mu_arora_n_lf', Potential="Potential", sign=-1, ElectronCurrent="Jn_arora_lf", V_t="V_t_edge")
    CreateHoleCurrent(device, region, mu_p = 'mu_arora_p_lf', Potential="Potential", sign=-1, HoleCurrent="Jp_arora_lf", V_t="V_t_edge")
    return {
        'mu_n' : 'mu_arora_n_lf',
      'mu_p' : 'mu_arora_p_lf',
      'Jn'   : 'Jn_arora_lf',
      'Jp'   : 'Jp_arora_lf',
    }


def CreateHFMobility(device, region, mu_n, mu_p, Jn, Jp):
    '''
      Add T derivatives when debugged
      use parameters to set model flags
      Caughey Thomas
    '''

    tdict = {
        'Jn' : Jn,
      'mu_n' : mu_n,
      'Jp' : Jp,
      'mu_p' : mu_p
    }
    tlist = (
        ("vsat_n", "VSATN0 * pow(T, VSATNE)" % tdict, ('T')),
      ("beta_n", "BETAN0 * pow(T, BETANE)" % tdict, ('T')),
      ("Epar_n",
            "ifelse((%(Jn)s * EField) > 0, abs(EField), 1e-15)" % tdict, ('Potential')),
      ("mu_n", "%(mu_n)s * pow(1 + pow((%(mu_n)s*Epar_n/vsat_n), beta_n), -1/beta_n)"
            % tdict, ('Electrons', 'Holes', 'Potential', 'T')),
      ("vsat_p", "VSATP0 * pow(T, VSATPE)" % tdict, ('T')),
      ("beta_p", "BETAP0 * pow(T, BETAPE)" % tdict, ('T')),
      ("Epar_p",
            "ifelse((%(Jp)s * EField) > 0, abs(EField), 1e-15)" % tdict, ('Potential')),
      ("mu_p", "%(mu_p)s * pow(1 + pow(%(mu_p)s*Epar_p/vsat_p, beta_p), -1/beta_p)"
            % tdict, ('Electrons', 'Holes', 'Potential', 'T')),
    )

    variable_list = ('Electrons', 'Holes', 'Potential')
    for (model, equation, variables) in tlist:
        CreateEdgeModel(device, region, model, equation)
        for v in variable_list:
            if v in variables:
                CreateEdgeModelDerivatives(device, region, model, equation, v)

    CreateElectronCurrent(device, region, mu_n='mu_n', Potential="Potential", sign=-1, ElectronCurrent="Jn", V_t="V_t_edge")
    CreateHoleCurrent(    device, region, mu_p='mu_p', Potential="Potential", sign=-1, HoleCurrent="Jp", V_t="V_t_edge")
    return {
        'mu_n' : 'mu_n',
      'mu_p' : 'mu_p',
      'Jn'   : 'Jn',
      'Jp'   : 'Jp',
    }
