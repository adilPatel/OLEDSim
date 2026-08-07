"""
Solver-independent device description.

Nothing in this module imports DEVSIM. A ``Device`` is a plain container for
the geometry (mesh node positions), the two contacts, and the built-in
voltage. A back end (e.g. ``devsim_backend``) consumes this data to build its
own internal representation.
"""

from dataclasses import dataclass, field
from math import exp
from typing import Optional

# Boltzmann constant in eV/K, for thermal voltage (kT/q) calculations.
_K_BOLTZMANN_EV = 8.617333262e-5


@dataclass
class Contact:
    """A single electrode.

    ``voltage_offset`` encodes the equilibrium band bending at this electrode
    (e.g. the built-in voltage at the anode, 0 at the cathode) so the device
    starts with a built-in field even at zero applied bias.

    ``electron_density`` / ``hole_density`` are the fixed Ohmic injection
    densities (#/cm^3) this contact pins its carriers to. The two contacts are
    deliberately not mirror images of each other -- real electrodes differ in
    barrier height, not just which carrier they favour.

    Alternatively, a contact can be described by its ``work_function`` (eV),
    with the injection densities derived from it and the semiconductor
    HOMO/LUMO (see :meth:`compute_densities_from_work_function`). Exactly one
    of the two descriptions must be given: either both ``electron_density``
    and ``hole_density``, or ``work_function`` alone. Mixing (or giving
    neither) is an error. Cross-contact consistency -- both contacts of a
    device must use the *same* description -- is enforced by
    :meth:`Device.__post_init__`, since only ``Device`` sees every contact.

    ``contact_type`` selects the boundary condition used for the carrier
    continuity equations: ``"ohmic"`` (default) pins Electrons/Holes to
    fixed densities, as above. ``"thermionic"`` replaces that pin with a
    field-dependent thermionic injection current (Emtage-O'Dwyer /
    Scott-Malliaras, see Models.md), so Electrons/Holes become solved
    unknowns at the contact rather than fixed parameters. Thermionic
    contacts require ``work_function`` (the barrier is derived from it, same
    as the Ohmic work-function path) and must not be given
    ``electron_density``/``hole_density``.
    """

    name: str
    tag: str
    material: str = "metal"
    voltage_offset: float = 0.0
    electron_density: Optional[float] = None
    hole_density: Optional[float] = None
    work_function: Optional[float] = None
    contact_type: str = "ohmic"

    def __post_init__(self):
        if self.contact_type not in ("ohmic", "thermionic"):
            raise ValueError(
                "Contact '{0}': contact_type must be 'ohmic' or 'thermionic', "
                "got '{1}'.".format(self.name, self.contact_type)
            )

        densities_given = self.electron_density is not None or self.hole_density is not None

        if self.contact_type == "thermionic":
            if self.work_function is None:
                raise ValueError(
                    "Contact '{0}': contact_type='thermionic' requires "
                    "work_function (the barrier is derived from it).".format(self.name)
                )
            if densities_given:
                raise ValueError(
                    "Contact '{0}': contact_type='thermionic' derives its "
                    "carrier densities from the solved thermionic emission "
                    "current -- do not pass electron_density/hole_density "
                    "(they are not pinned parameters for this contact "
                    "type).".format(self.name)
                )
            return

        if self.work_function is not None:
            if densities_given:
                raise ValueError(
                    "Contact '{0}': specify either electron_density/hole_density "
                    "or work_function, not both.".format(self.name)
                )
        elif not densities_given:
            raise ValueError(
                "Contact '{0}': must specify either electron_density/hole_density "
                "or work_function.".format(self.name)
            )
        elif self.electron_density is None or self.hole_density is None:
            raise ValueError(
                "Contact '{0}': electron_density and hole_density must both be "
                "given together.".format(self.name)
            )

    @property
    def uses_work_function(self):
        return self.work_function is not None

    def compute_densities_from_work_function(self, homo, lumo, nc300, nv300, temperature):
        """Derive (electron_density, hole_density) from ``work_function`` and
        the semiconductor ``homo``/``lumo`` (eV), conduction/valence band
        effective densities of states ``nc300``/``nv300`` (cm^-3), and
        ``temperature`` (K).

        Boltzmann injection densities: the electron/hole barrier is how far
        the Fermi level (work_function) sits below the LUMO / above the HOMO,
        so density falls off exponentially as that barrier grows.
        """
        kT = _K_BOLTZMANN_EV * temperature
        self.electron_density = nc300 * exp(-(lumo - self.work_function) / kT)
        self.hole_density = nv300 * exp(-(self.work_function - homo) / kT)
        return self.electron_density, self.hole_density


def compute_built_in_voltage(top_work_function, bot_work_function):
    """Built-in voltage from the two contacts' work functions (eV).

    Vbi is the equilibrium potential difference set up by the electrodes'
    differing work functions. Currently just their difference; a placeholder
    for later Fermi-level pinning (where Vbi would saturate once the
    semiconductor's surface states pin the interface Fermi level rather than
    following the metal work function directly).
    """
    return abs(top_work_function - bot_work_function)


def geometric_refine_to_zero(outer, min_spacing):
    """Node positions from ``outer`` down to 0, halving the spacing each step.

    Geometric refinement toward a contact: successively halved spacing in the
    outermost region resolves the steep carrier-density transition at the
    electrode without stalling the Newton solve.
    """
    positions = [outer]
    spacing = outer * 0.5
    x = outer
    while spacing > min_spacing:
        x -= spacing
        positions.append(round(x, 8))
        spacing /= 2
    positions.append(0.0)
    return sorted(set(positions))


def default_oled_mesh_nm(length_nm=100.0):
    """The refined 1-D mesh used by oled1: fine geometric grading in the
    outer 1 nm at each contact, 0.5 nm across the 0-10/90-100 nm boundary
    layers, and 1.0 nm through the 10-90 nm bulk.
    """
    near_top = geometric_refine_to_zero(1.0, 0.1)
    near_bot = sorted(length_nm - x for x in near_top)
    positions_nm = (
        near_top
        + [1.5 + 0.5 * i for i in range(17)]
        + [10.0 + i for i in range(80)]
        + [90.0 + 0.5 * i for i in range(18)]
        + near_bot
    )
    return sorted(set(round(p, 8) for p in positions_nm))


@dataclass
class Device:
    """A drift-diffusion device, independent of any particular solver.

    ``mesh_positions_nm`` is a sorted list of node positions (nm). ``contacts``
    maps each contact name to a :class:`Contact`. ``built_in_voltage`` is the
    equilibrium potential difference between the contacts.
    """

    name: str
    mesh_positions_nm: list = field(default_factory=list)
    contacts: dict = field(default_factory=dict)
    built_in_voltage: float = 0.0
    material: str = "Organic"
    temperature: float = 300.0
    # Semiconductor HOMO/LUMO (eV), passed on to the back end (os_physics).
    homo: Optional[float] = None
    lumo: Optional[float] = None
    # Conduction/valence band effective densities of states (cm^-3), passed
    # on to the back end (os_physics) and to Contact's work-function stub.
    nc300: float = 1e27
    nv300: float = 1e27
    # Constant electron/hole mobilities (cm^2/V-s), passed on to the back end
    # (os_physics.SetOSParameters). Defaults match the values formerly
    # hardcoded there, so existing devices (make_oled1) are unaffected.
    mu_n: float = 1.0e-6
    mu_p: float = 1.0e-6
    # Relative permittivity (dimensionless), passed on to the back end
    # (os_physics.SetOSParameters), which multiplies it by the vacuum
    # permittivity parameter. Default matches the value formerly hardcoded
    # there.
    relative_permittivity: float = 4.0
    # Name of the contact whose bias is swept during the I-V sweep.
    bias_contact: str = "top"

    def __post_init__(self):
        if self.contacts:
            uses_wf = {name: c.uses_work_function for name, c in self.contacts.items()}
            if len(set(uses_wf.values())) > 1:
                raise ValueError(
                    "Device '{0}': all contacts must use the same description -- "
                    "either work_function or electron_density/hole_density, not a "
                    "mix ({1}).".format(self.name, uses_wf)
                )
            if any(uses_wf.values()):
                for contact in self.contacts.values():
                    contact.compute_densities_from_work_function(
                        self.homo, self.lumo, self.nc300, self.nv300, self.temperature
                    )

                top = self.contacts[self.contacts_by_position()[0]]
                bot = self.contacts[self.contacts_by_position()[-1]]
                self.built_in_voltage = compute_built_in_voltage(
                    top.work_function, bot.work_function
                )
                # The contact with the deeper (more negative) work function --
                # the better hole injector -- is pinned to Vbi; the other to 0,
                # matching make_oled1's anode-gets-Vbi/cathode-gets-0 convention.
                if top.work_function <= bot.work_function:
                    top.voltage_offset = self.built_in_voltage
                    bot.voltage_offset = 0.0
                else:
                    top.voltage_offset = 0.0
                    bot.voltage_offset = self.built_in_voltage

    # -- convenience accessors used by the back end ----------------------------

    @property
    def top_contact(self):
        return self.contacts[self.contacts_by_position()[0]]

    @property
    def bottom_contact(self):
        return self.contacts[self.contacts_by_position()[-1]]

    def contacts_by_position(self):
        """Contact names ordered top (x=0) then bottom (x=max)."""
        # top tag sits at the first node, bot tag at the last.
        top = next(n for n, c in self.contacts.items() if c.tag == "top")
        bot = next(n for n, c in self.contacts.items() if c.tag == "bot")
        return [top, bot]

    @property
    def length_nm(self):
        return self.mesh_positions_nm[-1] - self.mesh_positions_nm[0]


def make_oled1():
    """Build the OLED1 device (matching the original oled1.py definition).

    Placeholder contact densities standing in for two electrodes with different
    injection barriers: "top" is an ITO-like anode (good hole injector, poor
    electron injector), "bot" is a low-work-function metal cathode (the
    reverse). ``Vbi`` is pinned at the anode and 0 at the cathode.
    """
    Vbi = 2.5
    # LUMO/HOMO (eV, relative to vacuum) matching the values formerly
    # hardcoded as Affinity/EG300 in devsim_backend.os_physics.
    lumo = -0.4
    homo = lumo - 2.6
    contacts = {
        "top": Contact(
            name="top", tag="top", material="metal",
            voltage_offset=Vbi,
            electron_density=3.2e-10, hole_density=2.0e12,
        ),
        "bot": Contact(
            name="bot", tag="bot", material="metal",
            voltage_offset=0.0,
            electron_density=1.0e25, hole_density=7.3e-30,
        ),
    }
    return Device(
        name="OLED1",
        mesh_positions_nm=default_oled_mesh_nm(100.0),
        contacts=contacts,
        built_in_voltage=Vbi,
        material="Organic",
        temperature=300.0,
        homo=homo,
        lumo=lumo,
        bias_contact="top",
    )


def make_oled2():
    """Build the OLED2 device: a 100 nm F8BT-based organic diode.

    "top" is a 5.5 eV work-function anode using the thermionic-emission
    contact model (Schottky barrier lowering + Richardson emission); "bot"
    is a 3.3 eV work-function cathode using the Ohmic contact model. Both
    contacts are described by work_function (required for the "same
    description" consistency check in Device.__post_init__, and for the
    thermionic contact regardless), so built_in_voltage/voltage_offset are
    derived automatically, same as make_oled1's work-function path would be.
    """
    # HOMO/LUMO given in the spec as depths below vacuum (eV); this module's
    # convention (matching make_oled1) is negative = below vacuum.
    lumo = -3.3
    homo = -5.9
    contacts = {
        "top": Contact(
            name="top", tag="top", material="metal",
            work_function=-5.3, contact_type="thermionic",
        ),
        "bot": Contact(
            name="bot", tag="bot", material="metal",
            work_function=-3.3, contact_type="ohmic",
        ),
    }
    return Device(
        name="OLED2",
        mesh_positions_nm=default_oled_mesh_nm(100.0),
        contacts=contacts,
        material="Organic",
        temperature=300.0,
        homo=homo,
        lumo=lumo,
        # Literature-typical organic-semiconductor DOS (~1e19-1e21 cm^-3);
        # not given in the OLED2 spec, so a representative value is used
        # here rather than Device's default of 1e27, which is unphysically
        # large and (combined with F8BT's higher mobility below) overflows
        # the Langevin recombination model during the equilibrium solve.
        nc300=1e21,
        nv300=1e21,
        mu_n=4e-3,
        mu_p=1e-3,
        relative_permittivity=3.5,
        bias_contact="top",
    )
