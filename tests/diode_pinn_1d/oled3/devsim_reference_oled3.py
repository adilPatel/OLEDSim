"""
devsim_reference_oled3.py

Generate the DEVSIM drift-diffusion reference solutions that the coupled
drift-diffusion PINN (sim_pinn/devices/oled3_forward.py) is validated
against.

OLED3 is a *symmetric* 100 nm F8BT diode: the top contact's work function
(-5.5 eV) sits 0.4 eV above the HOMO and the bottom contact's (-3.7 eV) sits
0.4 eV below the LUMO, so both Ohmic pins are equal and the solution
satisfies n(x) = p(L - x).

Running this writes one ``oled3_devsim_reference_<V>V.npz`` per target bias
(in this same folder) containing the node positions, the
Potential/Electrons/Holes profiles at that bias (and at equilibrium), and
the terminal current there; plus ``oled3_devsim_iv.npz`` with the full
sweep's current-voltage curve.

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
# lives two directories below the repo root (tests/diode_pinn_1d/oled3/), so
# three levels up gets back to the root, matching sim_pinn/forward_demo.py's
# own path-resolution depth from tests/diode_pinn_1d/oled3/ to sim_pinn/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_ROOT, "sim_dd"))

from core import DDSolver, SweepConfig  # noqa: E402
from core.device import Contact, Device, default_oled_mesh_nm  # noqa: E402
from devsim_backend import DevsimBackend  # noqa: E402


# Biases at which a full profile is written out for the PINN to be trained
# and scored against.
TARGET_BIASES = (2.0, 2.2, 2.4, 2.6, 2.8, 3.0)

# Bias sweep. The ramp starts well below the first target and walks up in
# SWEEP_STEP increments -- each solved point is the initial guess for the
# next, so the Newton solve cannot be dropped straight in at 2 V. The step
# divides every target exactly, so each one is landed on.
SWEEP_START = 0.1
SWEEP_STOP = max(TARGET_BIASES)
SWEEP_STEP = 0.1


def reference_npz(bias):
    """Path of the profile file for one target bias."""
    return os.path.join(_HERE, "oled3_devsim_reference_{0:.1f}V.npz".format(bias))


# Full sweep's current-voltage curve, read by core.plot_iv.
IV_NPZ = os.path.join(_HERE, "oled3_devsim_iv.npz")

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
    # SweepConfig.voltages() stops strictly below `stop`, so the top of the
    # range is nudged past SWEEP_STOP for that point to be recorded too.
    solver = DDSolver(
        device, backend,
        sweep=SweepConfig(start=SWEEP_START,
                          stop=SWEEP_STOP + 0.5 * SWEEP_STEP,
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

    # Vbi is derived from the two work functions by Device.__post_init__
    # (compute_built_in_voltage), not specified here, so it is read back off
    # the built device rather than held as a module constant.
    built_in_voltage = device.built_in_voltage

    # Full IV curve over the target window. The sub-2 V ramp is only there to
    # walk the Newton solve up, so it is kept separately: `voltages`/
    # `current_A` are the deliverable window, `*_full` the whole sweep that
    # core.plot_iv draws its reference line from.
    in_window = ((voltages >= min(TARGET_BIASES) - 1e-9)
                 & (voltages <= max(TARGET_BIASES) + 1e-9))
    np.savez(
        IV_NPZ,
        voltages=voltages[in_window],
        current_A=currents[in_window],
        voltages_full=voltages,
        current_A_full=currents,
        built_in_voltage=built_in_voltage,
    )

    print()
    print("=" * 66)
    print("DEVSIM reference sweep, {0:.1f} - {1:.1f} V".format(
        min(TARGET_BIASES), max(TARGET_BIASES)))
    print("=" * 66)
    print("Built-in voltage     : {0:.4f} V".format(built_in_voltage))
    print("Nodes                : {0}".format(len(x_nm)))
    print()
    print("  {0:>8s}  {1:>14s}".format("V (V)", "J (A/cm^2)"))
    for v, j in zip(voltages[in_window], currents[in_window]):
        print("  {0:8.2f}  {1:14.6e}".format(v, j))
    print()
    print("IV curve written to {0}".format(IV_NPZ))

    for target in TARGET_BIASES:
        # Locate the recorded point closest to this target bias.
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
            built_in_voltage=built_in_voltage,
        )

        print()
        print("-" * 66)
        print("Recorded bias        : {0:.4f} V".format(voltages[index]))
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
