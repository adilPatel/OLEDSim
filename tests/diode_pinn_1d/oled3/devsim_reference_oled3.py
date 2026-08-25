"""
devsim_reference_oled3.py


"""

import os
import sys

import numpy as np

# Make the sim_dd package importable when run as a plain script. This file
# lives two directories below the repo root (tests/diode_pinn_1d/oled3/), so
# three levels up gets back to the root, matching sim_pinn/forward_demo.py's
# own path-resolution depth from tests/diode_pinn_1d/oled3/ to sim_pinn/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_ROOT, "sim_dd"))

from core import DDSolver, SweepConfig  # noqa: E402
from core.device import Contact, Device, default_oled_mesh_nm  # noqa: E402
from devsim_backend import DevsimBackend  # noqa: E402


# Target bias for the PINN comparison, and the built-in voltage.
TARGET_BIAS = 2.2


OUTPUT_NPZ = os.path.join(_HERE, "oled3_devsim_reference_2.2V.npz")

def make_device():
    """Build the OLED3 device: a 100 nm F8BT-based organic diode.

    "top" is a 5.5 eV work-function Ohmic contact; 
    "bot" is a 3.7 eV Ohmic contact. 
    """
    # HOMO/LUMO given in the spec as depths below vacuum (eV); this module's
    # convention (matching make_oled1) is negative = below vacuum.
    lumo = -3.3
    homo = -5.9
    contacts = {
        "top": Contact(
            name="top", tag="top", material="metal",
            work_function=-5.5, contact_type="ohmic",
        ),
        "bot": Contact(
            name="bot", tag="bot", material="metal",
            work_function=-3.7, contact_type="ohmic",
        ),
    }
    return Device(
        name="PINNREF3",
        mesh_positions_nm=default_oled_mesh_nm(100.0),
        contacts=contacts,
        material="Organic",
        temperature=300.0,
        homo=homo,
        lumo=lumo,
        nc300=1e21,
        nv300=1e21,
        mu_n=1e-3,
        mu_p=1e-3,
        relative_permittivity=3.5,
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

    # Vbi is derived from the two work functions by Device.__post_init__
    # (compute_built_in_voltage), not specified here, so it is read back off
    # the built device rather than held as a module constant.
    built_in_voltage = device.built_in_voltage

    np.savez(
        OUTPUT_NPZ,
        x_nm=x_nm,
        bias=voltages[index],
        potential=potential,
        electrons=electrons,
        holes=holes,
        potential_eq=potential_eq,
        built_in_voltage=built_in_voltage,
    )

    print()
    print("=" * 66)
    print("DEVSIM reference solution")
    print("=" * 66)
    print("Recorded bias        : {0:.4f} V".format(voltages[index]))
    print("Built-in voltage     : {0:.4f} V".format(built_in_voltage))
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
