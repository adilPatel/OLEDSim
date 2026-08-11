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

The sweep runs to 2.5 V so the PINN can be compared against the DEVSIM
solution at that bias. Vbi is 2.0 V, applied as the anode's voltage_offset,
following the convention in core/device.py.

Running this writes ``oled1_devsim_reference_2.5V.npz`` (in this same folder)
containing the node positions and the Potential/Electrons/Holes profiles at
2.5 V (and at equilibrium), which sim_pinn/forward_demo.py loads.
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


# Target bias for the PINN comparison, and the built-in voltage.
TARGET_BIAS = 2.5
BUILT_IN_VOLTAGE = 2.0

# Bottom-contact electron density, reduced from OLED1's 1e25 (see module
# docstring). This is the one parameter changed from make_oled1.
N_BOT_ELECTRONS = 1.0e17

OUTPUT_NPZ = os.path.join(_HERE, "oled1_devsim_reference_2.5V.npz")


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
    # Sweep past TARGET_BIAS so that bias point is actually recorded. The step
    # is chosen to land exactly on 2.5 V.
    solver = DDSolver(
        device, backend,
        sweep=SweepConfig(start=0.1, stop=TARGET_BIAS + 0.05, step=0.1),
    )

    results = solver.run()

    voltages = np.array(results.voltages)
    x_nm = np.array(results.x_nm)

    # Locate the recorded point closest to the target bias.
    index = int(np.argmin(np.abs(voltages - TARGET_BIAS)))
    if abs(voltages[index] - TARGET_BIAS) > 1e-6:
        raise RuntimeError(
            "No recorded bias point at {0} V; closest was {1} V. Adjust the "
            "sweep step so it lands on the target.".format(TARGET_BIAS, voltages[index])
        )

    potential = np.array(results.profile("Potential")[index])
    electrons = np.array(results.profile("Electrons")[index])
    holes = np.array(results.profile("Holes")[index])

    # Equilibrium (first recorded point, V = 0) for reference.
    potential_eq = np.array(results.profile("Potential")[0])

    np.savez(
        OUTPUT_NPZ,
        x_nm=x_nm,
        bias=voltages[index],
        potential=potential,
        electrons=electrons,
        holes=holes,
        potential_eq=potential_eq,
        built_in_voltage=BUILT_IN_VOLTAGE,
        n_bot_electrons=N_BOT_ELECTRONS,
    )

    print()
    print("=" * 66)
    print("DEVSIM reference solution")
    print("=" * 66)
    print("Recorded bias        : {0:.4f} V".format(voltages[index]))
    print("Built-in voltage     : {0:.4f} V".format(BUILT_IN_VOLTAGE))
    print("Nodes                : {0}".format(len(x_nm)))
    print("Potential range      : {0:+.4f} .. {1:+.4f} V".format(
        potential.min(), potential.max()))
    print("Electrons range      : {0:.4e} .. {1:.4e} cm^-3".format(
        electrons.min(), electrons.max()))
    print("Holes range          : {0:.4e} .. {1:.4e} cm^-3".format(
        holes.min(), holes.max()))
    print("Contact potentials   : phi(0) = {0:+.4f} V, phi(L) = {1:+.4f} V".format(
        potential[0], potential[-1]))
    print()
    print("Written to {0}".format(OUTPUT_NPZ))


if __name__ == "__main__":
    main()
