"""
DEVSIM implementation of the drift-diffusion back-end interface.

``DDSolver`` (in ``core.dd_solver``) drives a device through an equilibrium
solve and a bias sweep purely through the small method surface defined here.
None of those methods leak DEVSIM types out to the caller: they take/return
plain Python floats and lists. To target a different solver later, write a
class with the same methods and hand it to ``DDSolver`` instead -- nothing in
``core`` needs to change.

The physics itself is unchanged; this class is a thin orchestration layer over
the existing model-creation functions in ``os_physics`` / ``common_physics``.
"""

import math

from devsim import (
    set_parameter,
    solve,
    get_contact_list,
    get_node_model_values,
    write_devices,
    edge_average_model,
    get_edge_model_values,
    get_parameter,
    set_node_values,
    get_equation_list,
    get_equation_command,
    delete_equation,
    equation,
    get_contact_equation_list,
    get_contact_equation_command,
    delete_contact_equation,
    contact_equation,
)

from .os_physics import (
    SetOSParameters,
    CreateOSPotentialOnly,
    CreateOSPotentialOnlyContact,
    CreateOSConstantMobility,
    CreateOSDriftDiffusion,
    CreateOSDriftDiffusionContact,
    CreateOSThermionicContact,
)
from .common_physics import GetContactBiasName
from .model_create import CreateSolution, CreateNodeModel
from .ramp2 import PrintCurrents
from . import mesh as mesh_builder


class DevsimBackend:
    """Drives DEVSIM for an organic-semiconductor drift-diffusion problem.

    One instance corresponds to one DEVSIM device. Construction is cheap; the
    device/mesh is only built when :meth:`build_device` is called, so a
    ``DevsimBackend`` can be created before its ``Device`` is fully described.
    """

    # Equation names, used by the Gummel preconditioner to solve subsets of
    # the coupled system.
    _POTENTIAL_EQ = "PotentialEquation"
    _ELECTRON_EQ = "ElectronContinuityEquation"
    _HOLE_EQ = "HoleContinuityEquation"

    def __init__(self, device_name="OLED1", region="MyRegion"):
        self.device_name = device_name
        self.region = region
        # Populated by build_device(); cached so the sweep/extraction methods
        # don't have to be handed the Device each call.
        self._mobility_opts = None
        self._contact_offsets = {}
        self._bias_contact = None
        # Set by setup_drift_diffusion(): whether any contact uses the
        # thermionic model, and hence whether the Gummel preconditioner
        # (which only helps that case) should run before the coupled solve.
        self._has_thermionic = False
        # Full definitions of every region/contact equation, captured after
        # setup_drift_diffusion() so the Gummel preconditioner can delete and
        # restore them.
        self._equation_snapshot = None

    # ---- device construction -------------------------------------------------

    def build_device(self, device):
        """Create the DEVSIM mesh + device from a solver-independent Device.

        ``device`` is a ``core.device.Device``; only its plain-data attributes
        (mesh positions, contact definitions, built-in voltage) are read here.
        """
        mesh_builder.create_mesh(self.device_name, self.region, device)
        self._contact_offsets = {
            name: contact.voltage_offset for name, contact in device.contacts.items()
        }
        self._bias_contact = device.bias_contact

    # ---- equilibrium ---------------------------------------------------------

    def setup_equilibrium(self, device):
        """Set parameters and build the potential-only equilibrium problem."""
        set_parameter(name="T", value=device.temperature)
        SetOSParameters(
            self.device_name, self.region,
            device.homo, device.lumo, device.nc300, device.nv300,
            mu_n=device.mu_n, mu_p=device.mu_p,
            relative_permittivity=device.relative_permittivity,
        )

        # NetDoping (N_A and N_D are 0 for the undoped OLED for now).
        CreateNodeModel(self.device_name, self.region, "Acceptors", "0.0")
        CreateNodeModel(self.device_name, self.region, "Donors", "0.0")
        CreateNodeModel(self.device_name, self.region, "NetDoping", "Donors-Acceptors")

        CreateSolution(self.device_name, self.region, "Potential")
        CreateOSPotentialOnly(self.device_name, self.region)

        for name in get_contact_list(device=self.device_name):
            set_parameter(device=self.device_name, name=GetContactBiasName(name), value=0.0)
            CreateOSPotentialOnlyContact(
                self.device_name, self.region, name,
                V_offset=self._contact_offsets[name],
            )

    def solve_potential_only(self):
        """Initial DC (potential-only) solution."""
        solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=30)

    def setup_drift_diffusion(self, device):
        """Add the electron/hole solution variables and DD equations."""
        CreateSolution(self.device_name, self.region, "Electrons")
        CreateSolution(self.device_name, self.region, "Holes")

        self._mobility_opts = CreateOSConstantMobility(self.device_name, self.region)

        set_node_values(device=self.device_name, region=self.region,
                        name="Electrons", init_from="IntrinsicElectrons")
        set_node_values(device=self.device_name, region=self.region,
                        name="Holes", init_from="IntrinsicHoles")

        opts = self._mobility_opts
        CreateOSDriftDiffusion(self.device_name, self.region, Jn=opts["Jn"], Jp=opts["Jp"])

        for name in get_contact_list(device=self.device_name):
            contact = device.contacts[name]
            if contact.contact_type == "thermionic":
                self._has_thermionic = True
                # "top" sits at the lowest-x mesh node (n0 of its boundary
                # edge), "bot" at the highest-x node (n1) -- see mesh.py's
                # create_mesh, which always places top_contact/bottom_contact
                # this way.
                node_suffix = "n0" if contact.tag == "top" else "n1"
                CreateOSThermionicContact(
                    self.device_name, self.region, name,
                    Jn=opts["Jn"], Jp=opts["Jp"],
                    electron_barrier=device.lumo - contact.work_function,
                    hole_barrier=contact.work_function - device.homo,
                    node_suffix=node_suffix,
                    V_offset=contact.voltage_offset,
                )
            else:
                CreateOSDriftDiffusionContact(
                    self.device_name, self.region, name,
                    Jn=opts["Jn"], Jp=opts["Jp"],
                    electron_density=contact.electron_density,
                    hole_density=contact.hole_density,
                    V_offset=contact.voltage_offset,
                )

        if self._has_thermionic:
            self._equation_snapshot = self._snapshot_equations()

    # ---- Gummel preconditioner ----------------------------------------------
    #
    # Only used for devices with a thermionic contact. See Models.md: the
    # coupled Newton solve can settle onto a spurious near-zero-current
    # solution branch for those devices. Solving the equations in a decoupled
    # (Gummel) sequence first produces a smooth, physically sensible
    # equilibrium state, which is then handed to the coupled solve as a
    # starting point.

    def _snapshot_equations(self):
        """Capture the full definition of every region/contact equation."""
        snap = {"region": {}, "contact": {}}
        for name in get_equation_list(device=self.device_name, region=self.region):
            snap["region"][name] = get_equation_command(
                device=self.device_name, region=self.region, name=name)
        for contact in get_contact_list(device=self.device_name):
            for name in get_contact_equation_list(device=self.device_name, contact=contact):
                snap["contact"][(contact, name)] = get_contact_equation_command(
                    device=self.device_name, contact=contact, name=name)
        return snap

    def _restrict_equations(self, keep_names):
        """Delete every equation, then re-create only those in keep_names.

        DEVSIM has no way to disable an equation in place, so the Gummel
        sub-solves are set up by deleting all equations and restoring the
        subset being solved. The snapshot taken in setup_drift_diffusion()
        holds enough to recreate any of them exactly.
        """
        snap = self._equation_snapshot
        for contact, name in list(snap["contact"].keys()):
            if name in get_contact_equation_list(device=self.device_name, contact=contact):
                delete_contact_equation(device=self.device_name, contact=contact, name=name)
        for name in list(snap["region"].keys()):
            if name in get_equation_list(device=self.device_name, region=self.region):
                delete_equation(device=self.device_name, region=self.region, name=name)
        for name in keep_names:
            equation(**snap["region"][name])
        for (contact, name), cmd in snap["contact"].items():
            if name in keep_names:
                contact_equation(**cmd)

    def _restore_equations(self):
        """Restore the full coupled system."""
        self._restrict_equations(list(self._equation_snapshot["region"].keys()))

    def _damp_density(self, name, previous, alpha):
        """Geometric (log-space) damping of a carrier density update.

        new = previous^(1-alpha) * solved^alpha, elementwise. Densities span
        many orders of magnitude, so damping the logarithm (rather than the
        value) keeps the update proportionate everywhere in the device.
        """
        current = get_node_model_values(device=self.device_name, region=self.region, name=name)
        damped = [
            math.exp((1.0 - alpha) * math.log(p) + alpha * math.log(c))
            if (p > 0.0 and c > 0.0) else c
            for p, c in zip(previous, current)
        ]
        set_node_values(device=self.device_name, region=self.region, name=name, values=damped)

    def gummel_precondition(self, alpha=0.2, max_outer=400, tolerance=1e-6):
        """Decoupled Poisson/electron/hole iteration, as a starting point.

        Solves each equation in turn with the others held fixed, damping the
        carrier updates, until the potential stops changing. Returns True if
        it converged. A failure here is not fatal: the caller falls back to
        the coupled solve from whatever state was reached.
        """
        previous_potential = None
        for _ in range(max_outer):
            self._restrict_equations([self._POTENTIAL_EQ])
            try:
                solve(type="dc", absolute_error=1e12, relative_error=1e-10,
                      maximum_iterations=100)
            except Exception:
                return False

            previous_electrons = list(get_node_model_values(
                device=self.device_name, region=self.region, name="Electrons"))
            self._restrict_equations([self._ELECTRON_EQ])
            try:
                solve(type="dc", absolute_error=1e13, relative_error=1e-10,
                      maximum_iterations=100)
            except Exception:
                return False
            self._damp_density("Electrons", previous_electrons, alpha)

            previous_holes = list(get_node_model_values(
                device=self.device_name, region=self.region, name="Holes"))
            self._restrict_equations([self._HOLE_EQ])
            try:
                solve(type="dc", absolute_error=1e13, relative_error=1e-10,
                      maximum_iterations=100)
            except Exception:
                return False
            self._damp_density("Holes", previous_holes, alpha)

            potential = get_node_model_values(
                device=self.device_name, region=self.region, name="Potential")
            if previous_potential is not None:
                delta = max(abs(a - b) for a, b in zip(potential, previous_potential))
                if delta < tolerance:
                    return True
            previous_potential = list(potential)
        return False

    def solve_equilibrium(self):
        """Full drift-diffusion solve at zero applied bias.

        For devices with a thermionic contact, a decoupled Gummel iteration
        runs first to produce a smooth starting state (see
        :meth:`gummel_precondition` and Models.md). The coupled Newton solve
        below then runs from that state, and is what the result actually
        comes from either way.
        """
        if self._has_thermionic and self._equation_snapshot is not None:
            try:
                self.gummel_precondition()
            finally:
                # Always put the coupled system back, even if the
                # preconditioner raised or bailed out part-way through.
                self._restore_equations()

        # 150 (not DEVSIM's default 50): devices with a near-zero injection
        # barrier at one contact (e.g. OLED2's cathode, whose work function
        # sits exactly at the LUMO) start far from equilibrium and need
        # substantially more Newton iterations to converge, even though they
        # do converge steadily rather than diverging. Devices that already
        # converge quickly (e.g. OLED1, ~5 iterations) are unaffected by the
        # higher ceiling.
        solve(type="dc", absolute_error=1e12, relative_error=1e-9, maximum_iterations=150)

    # ---- bias sweep ----------------------------------------------------------

    def apply_bias(self, voltage):
        """Set the swept bias on the driven contact."""
        set_parameter(device=self.device_name,
                      name=GetContactBiasName(self._bias_contact), value=voltage)

    def solve_step(self):
        """Solve at the currently applied bias."""
        solve(type="dc", absolute_error=1e13, relative_error=1e-8, maximum_iterations=80)

    # ---- result extraction (all return plain floats/lists) -------------------

    def terminal_current(self, contact):
        """Return (voltage, total_current) at a contact, in DEVSIM units (A)."""
        voltage, total_current = PrintCurrents(self.device_name, contact)
        return voltage, total_current

    def node_values(self, name):
        """Return a node model's values as a plain list of floats."""
        return list(get_node_model_values(device=self.device_name, region=self.region, name=name))

    def edge_values(self, name):
        """Return an edge model's values as a plain list of floats."""
        return list(get_edge_model_values(device=self.device_name, region=self.region, name=name))

    def edge_midpoints(self):
        """Return the edge midpoint positions (cm) as a plain list."""
        edge_average_model(device=self.device_name, region=self.region,
                           node_model="x", edge_model="xmid")
        return list(get_edge_model_values(device=self.device_name, region=self.region, name="xmid"))

    def region_parameter(self, name):
        """Return a scalar region parameter."""
        return get_parameter(device=self.device_name, region=self.region, name=name)

    def positions_cm(self):
        """Node positions along the device (cm)."""
        return self.node_values("x")

    def write_tecplot(self, path):
        write_devices(file=path, type="tecplot")

    @property
    def contacts(self):
        return list(get_contact_list(device=self.device_name))
