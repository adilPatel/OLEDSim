## Program Architecture

The drift-diffusion solver is abstracted away from the OLED front-end files so
the solver and mesher can be swapped from DEVSIM to a custom solver in future.
The DEVSIM-dependent code is isolated behind a small back-end interface; the
rest of the program never imports DEVSIM.

### Layout

```
sim_dd/
  core/                 # solver-independent. NEVER imports devsim.
    device.py           #   Device, Contact, mesh helpers, make_oled1()
    dd_solver.py        #   DDSolver, SweepConfig, SweepResults
  devsim_backend/       # everything DEVSIM-dependent lives here
    backend.py          #   DevsimBackend -- implements the back-end interface
    mesh.py             #   builds a DEVSIM 1-D device from a Device
    model_create.py     #   (moved) low-level DEVSIM model builders
    common_physics.py   #   (moved) generic DD physics
    os_physics.py       #   (moved) organic-semiconductor physics
    ramp2.py            #   (moved) current extraction helpers
    new_physics.py, inorganic_physics.py  # (moved) silicon reference physics
  output_tools.py       # plotting + saving (validation tables, sliders, .eps)
  devices/
    oled1.py            # thin front end: wires the above together
```

### Class: Device (`core/device.py`)
Solver-independent device description. Holds the mesh as a floating-point array
of node positions (nm), the top/bottom `Contact`s (each carrying its tag,
material, Ohmic injection densities, and voltage offset), and the built-in
voltage. No DEVSIM types.

### Class: DDSolver (`core/dd_solver.py`)
Owns a `Device`, a `SweepConfig`, and a *back-end* object, and drives the whole
solve purely through the back-end method surface (`build_device`,
`setup_equilibrium`, `solve_potential_only`, `setup_drift_diffusion`,
`solve_equilibrium`, `apply_bias`, `solve_step`, `terminal_current`,
`node_values`, ...). It initialises the solver, performs the equilibrium
solution, then runs the full bias sweep, storing per-bias-point profiles in a
plain-Python `SweepResults`. Swapping solvers = passing a different back end;
nothing in `core` changes.

### Back end: DevsimBackend (`devsim_backend/backend.py`)
The DEVSIM implementation of the back-end interface, a thin orchestration layer
over the moved physics modules. A replacement solver only needs a class with
the same methods.

### Module: OutputTools (`output_tools.py`)
Functions for plotting `SweepResults` and saving them into the `tests/` files
using the formats in `tests/diode_1d/format.md`. Keeps front ends like
`oled1.py` short. The saving/slider functions are solver-agnostic; the
edge-model diagnostic plots read through the back end.