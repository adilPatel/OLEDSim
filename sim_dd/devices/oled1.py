"""
OLED1 front end.

A 1-D undoped organic diode: build the device, solve equilibrium, sweep the
applied bias, then save validation tables and diagnostic plots. All the heavy
lifting lives in the modular back end (``core`` + ``devsim_backend`` +
``output_tools``); this file only wires them together and describes *this*
device.
"""

import os
import sys

# Make the sim_dd package importable when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import DDSolver, SweepConfig, make_oled1
from devsim_backend import DevsimBackend
import output_tools


def main():
    device = make_oled1()
    backend = DevsimBackend(device_name=device.name, region="MyRegion")
    solver = DDSolver(device, backend, sweep=SweepConfig(start=0.1, stop=5.0, step=0.02))

    results = solver.run()

    # Validation tables (tests/diode_1d/<device>/), per tests/diode_1d/format.md.
    output_tools.save_validation(results, device.name)

    # Tecplot dump of the final solution.
    backend.write_tecplot("diode_os.tec")

    # Diagnostic figures at the final bias point + the I-V curve.
    x_cm = backend.positions_cm()
    output_tools.plot_current(backend, x_cm)
    output_tools.plot_mobility(backend, x_cm)
    output_tools.plot_langevin(backend, x_cm)
    output_tools.plot_iv(results)

    # Interactive slider plots to inspect profiles across the sweep.
    output_tools.density_slider(results, x_cm)
    output_tools.potential_slider(results, x_cm)
    output_tools.current_slider(results)
    output_tools.show()


if __name__ == "__main__":
    main()
