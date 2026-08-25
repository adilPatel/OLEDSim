"""
devsim_reference_oled1.py

Generate the DEVSIM drift-diffusion reference solution that the coupled
drift-diffusion PINN (sim_pinn/forward_demo.py) is validated against.

This test case is called "oled1" within diode_pinn_1d, but it is NOT the same
device as sim_dd's OLED1 (tests/diode_1d/oled1): it is a 100 nm undoped organic
layer with two Ohmic contacts, built from the same ``core``/``devsim_backend``
machinery as sim_dd's OLED1, with one deliberate change carried over from that
device -- the bottom contact's electron density is reduced from 1e25 to
1e17 cm^-3.

Why reduce it
-------------
The Debye length that screens a contact goes as 1/sqrt(n), so the pinned
density sets the width of the space-charge layer, and therefore how much
curvature the solution has:

    n_bot = 1e25 cm^-3  ->  L_D = 0.0008 nm  ->  L/L_D = 132000
    n_bot = 1e17 cm^-3  ->  L_D = 7.57   nm  ->  L/L_D = 13.2

At 1e25 the screening layer is five orders of magnitude narrower than the
device: essentially a discontinuity, which no smooth network can represent and
which even the finite-volume solver needs a very fine mesh to resolve. At 1e17
the layer is a few nm wide -- steep, but resolvable -- while remaining
physically reasonable for an organic semiconductor (well below the 1e27 DOS).

The sweep runs from 2.0 V to 5.0 V so the PINN can be compared against the
DEVSIM solution at several biases above the built-in voltage. Vbi is 2.0 V,
applied as the anode's voltage_offset, following the convention in
core/device.py.

Running this writes one ``oled1_devsim_reference_<V>V.npz`` per target bias
(in this same folder) containing the node positions, the
Potential/Electrons/Holes profiles at that bias (and at equilibrium), and the
terminal current there; plus ``oled1_devsim_iv.npz`` with the full sweep's
current-voltage curve. ``sim_pinn/oled1_forward.py`` loads these.

Note on current units
---------------------
The mesh is one-dimensional with no cross-sectional area applied, so DEVSIM's
terminal current is per unit area: the values recorded here as ``current_A``
are A/cm^2, directly comparable with the PINN's ``report_currents()`` output.
"""

import os
import sys

import numpy as np

# Make the sim_dd package importable when run as a plain script. This file
# lives two directories below the repo root (tests/diode_pinn_1d/oled1/), so
# three levels up gets back to the root, matching sim_pinn/forward_demo.py's
# own path-resolution depth from tests/diode_pinn_1d/oled1/ to sim_pinn/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_ROOT, "sim_dd"))

from core import DDSolver, SweepConfig  # noqa: E402
from core.device import Contact, Device, default_oled_mesh_nm  # noqa: E402
from devsim_backend import DevsimBackend  # noqa: E402


# Bias sweep and the built-in voltage. The sweep starts below SWEEP_START and
# ramps up in SWEEP_STEP increments; the step is chosen so every target bias
# is landed on exactly.
SWEEP_START = 0.1
SWEEP_STOP = 5.0
SWEEP_STEP = 0.1
BUILT_IN_VOLTAGE = 2.0

# Biases at which a full profile is written out for the PINN to be trained
# and scored against.
TARGET_BIASES = (2.5, 3.0, 3.3, 3.4, 3.5, 3.6, 3.7, 4.0, 4.5)

# Bottom-contact electron density, reduced from OLED1's 1e25 (see module
# docstring). This is the one parameter changed from make_oled1.
N_BOT_ELECTRONS = 1.0e17

IV_NPZ = os.path.join(_HERE, "oled1_devsim_iv.npz")


def reference_npz(bias):
    """Path of the profile file for one target bias."""
    return os.path.join(_HERE, "oled1_devsim_reference_{0:.1f}V.npz".format(bias))


def make_device():
    """OLED1's 100 nm organic diode with a softened bottom contact.

    Contact densities are OLED1's, except ``bot``'s electron density (see
    N_BOT_ELECTRONS). ``top`` is the ITO-like anode pinned to Vbi; ``bot`` is
    the cathode at 0, matching make_oled1's convention.
    """
    lumo = -0.4
    homo = lumo - 2.6
    contacts = {
        "top": Contact(
            name="top", tag="top", material="metal",
            voltage_offset=BUILT_IN_VOLTAGE,
            electron_density=3.2e-10, hole_density=2.0e12,
        ),
        "bot": Contact(
            name="bot", tag="bot", material="metal",
            voltage_offset=0.0,
            electron_density=N_BOT_ELECTRONS, hole_density=7.3e-30,
        ),
    }
    return Device(
        name="PINNREF",
        mesh_positions_nm=default_oled_mesh_nm(100.0),
        contacts=contacts,
        built_in_voltage=BUILT_IN_VOLTAGE,
        material="Organic",
        temperature=300.0,
        homo=homo,
        lumo=lumo,
        bias_contact="top",
    )


def main():
    device = make_device()
    backend = DevsimBackend(device_name=device.name, region="MyRegion")
    # The ramp has to start from low bias and walk up: each step is the initial
    # guess for the next, and the Newton solve will not converge if dropped
    # straight in at 2 V. SWEEP_STOP is nudged past 5.0 V so that point is
    # recorded too (SweepConfig.voltages() stops strictly below `stop`).
    solver = DDSolver(
        device, backend,
        sweep=SweepConfig(start=SWEEP_START, stop=SWEEP_STOP + 0.5 * SWEEP_STEP,
                          step=SWEEP_STEP),
    )

    results = solver.run()

    voltages = np.array(results.voltages)
    x_nm = np.array(results.x_nm)
    # 1D mesh with no area applied, so DEVSIM's terminal current in A is
    # already a current density in A/cm^2. top_currents is stored in mA.
    currents = np.array(results.top_currents) * 1e-3

    # Equilibrium (first recorded point, V = 0), shared by every output file.
    potential_eq = np.array(results.profile("Potential")[0])

    # Full IV curve over the requested 2-5 V window (the sub-2 V ramp is only
    # there to get the Newton solve there, and is not part of the deliverable).
    in_window = (voltages >= 2.0 - 1e-9) & (voltages <= 5.0 + 1e-9)
    np.savez(
        IV_NPZ,
        voltages=voltages[in_window],
        current_A=currents[in_window],
        voltages_full=voltages,
        current_A_full=currents,
        built_in_voltage=BUILT_IN_VOLTAGE,
        n_bot_electrons=N_BOT_ELECTRONS,
    )

    print()
    print("=" * 66)
    print("DEVSIM reference sweep, {0:.1f} - {1:.1f} V".format(2.0, 5.0))
    print("=" * 66)
    print("Built-in voltage     : {0:.4f} V".format(BUILT_IN_VOLTAGE))
    print("Nodes                : {0}".format(len(x_nm)))
    print()
    print("  {0:>8s}  {1:>14s}".format("V (V)", "J (A/cm^2)"))
    for v, j in zip(voltages[in_window], currents[in_window]):
        print("  {0:8.2f}  {1:14.6e}".format(v, j))
    print()
    print("IV curve written to {0}".format(IV_NPZ))

    for target in TARGET_BIASES:
        index = int(np.argmin(np.abs(voltages - target)))
        if abs(voltages[index] - target) > 1e-6:
            raise RuntimeError(
                "No recorded bias point at {0} V; closest was {1} V. Adjust the "
                "sweep step so it lands on the target.".format(target, voltages[index])
            )

        potential = np.array(results.profile("Potential")[index])
        electrons = np.array(results.profile("Electrons")[index])
        holes = np.array(results.profile("Holes")[index])

        out_npz = reference_npz(target)
        np.savez(
            out_npz,
            x_nm=x_nm,
            bias=voltages[index],
            current_A=currents[index],
            potential=potential,
            electrons=electrons,
            holes=holes,
            potential_eq=potential_eq,
            built_in_voltage=BUILT_IN_VOLTAGE,
            n_bot_electrons=N_BOT_ELECTRONS,
        )

        print()
        print("-" * 66)
        print("Reference profile at {0:.4f} V".format(voltages[index]))
        print("-" * 66)
        print("Terminal current     : {0:.6e} A/cm^2".format(currents[index]))
        print("Potential range      : {0:+.4f} .. {1:+.4f} V".format(
            potential.min(), potential.max()))
        print("Electrons range      : {0:.4e} .. {1:.4e} cm^-3".format(
            electrons.min(), electrons.max()))
        print("Holes range          : {0:.4e} .. {1:.4e} cm^-3".format(
            holes.min(), holes.max()))
        print("Contact potentials   : phi(0) = {0:+.4f} V, phi(L) = {1:+.4f} V".format(
            potential[0], potential[-1]))
        print("Written to {0}".format(out_npz))


if __name__ == "__main__":
    main()
