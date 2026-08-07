"""
Solver-independent drift-diffusion driver.

``DDSolver`` owns a :class:`~core.device.Device`, a bias sweep, and a *back end*
object that actually talks to a solver. It never imports DEVSIM: it only calls
the back-end method surface (``build_device``, ``setup_equilibrium``,
``solve_equilibrium``, ``apply_bias``, ``solve_step``, ``terminal_current``,
``node_values``...). Swapping solvers is a matter of passing a different back
end -- see ``devsim_backend.DevsimBackend``.
"""

from dataclasses import dataclass, field


@dataclass
class SweepConfig:
    """Applied-bias sweep on the driven contact.

    Defaults reproduce the original oled1 ramp: start at 0.1 V, step 0.02 V,
    stop below 5.0 V. A V=0 equilibrium point is always recorded first.
    """

    start: float = 0.1
    stop: float = 5.0
    step: float = 0.02

    def voltages(self):
        v = self.start
        out = []
        while v < self.stop:
            out.append(round(v, 10))
            v += self.step
        return out


@dataclass
class SweepResults:
    """Results of an equilibrium solve plus a bias sweep.

    Everything is stored as plain Python so it survives independently of any
    solver. ``profiles`` holds one dict per recorded bias point, each mapping a
    quantity name ("Electrons", "Holes", "Potential", ...) to a per-node list.
    ``edge_profiles`` is the same idea for per-edge quantities ("Jn_const",
    "Jp_const"), aligned with ``x_edge_nm`` rather than ``x_nm``.
    ``voltages`` and ``top_currents`` / ``bot_currents`` are aligned with it.
    """

    x_nm: list = field(default_factory=list)
    x_edge_nm: list = field(default_factory=list)
    voltages: list = field(default_factory=list)
    top_currents: list = field(default_factory=list)   # mA
    bot_currents: list = field(default_factory=list)   # mA
    profiles: list = field(default_factory=list)       # list of {name: [values]}, one per node
    edge_profiles: list = field(default_factory=list)  # list of {name: [values]}, one per edge
    # Scalar metadata captured once, after the sweep.
    mun: float = 0.0
    mup: float = 0.0
    band_gap: float = 0.0
    built_in_voltage: float = 0.0

    def profile(self, name):
        """All per-node arrays for ``name``, one per recorded bias point."""
        return [p[name] for p in self.profiles]

    def edge_profile(self, name):
        """All per-edge arrays for ``name``, one per recorded bias point."""
        return [p[name] for p in self.edge_profiles]


# Node quantities captured at every recorded bias point.
_PROFILE_QUANTITIES = ("Electrons", "Holes", "Donors", "Acceptors", "Potential")
# Edge quantities captured at every recorded bias point.
_EDGE_PROFILE_QUANTITIES = ("Jn_const", "Jp_const")


class DDSolver:
    """Drives a :class:`Device` through equilibrium and a bias sweep."""

    def __init__(self, device, backend, sweep=None):
        self.device = device
        self.backend = backend
        self.sweep = sweep or SweepConfig()
        self.results = SweepResults(built_in_voltage=device.built_in_voltage)

    # ---- setup ---------------------------------------------------------------

    def initialise(self):
        """Build the device and set up equilibrium + drift-diffusion problems."""
        self.backend.build_device(self.device)
        self.backend.setup_equilibrium(self.device)
        self.backend.solve_potential_only()
        self.backend.setup_drift_diffusion(self.device)

    def solve_equilibrium(self):
        """Solve at zero applied bias and record it as the first point."""
        self.backend.solve_equilibrium()
        self._record_point()

    # ---- sweep ---------------------------------------------------------------

    def run_sweep(self):
        """Ramp the bias contact through the configured sweep, recording each
        converged point.
        """
        for v in self.sweep.voltages():
            print("Solving at V = {0:.2f}".format(v))
            self.backend.apply_bias(v)
            self.backend.solve_step()
            self._record_point()
        self._capture_metadata()
        return self.results

    def run(self):
        """Full pipeline: initialise, equilibrium, sweep. Returns results."""
        self.initialise()
        self.solve_equilibrium()
        return self.run_sweep()

    # ---- recording -----------------------------------------------------------

    def _record_point(self):
        top_voltage, top_current = self.backend.terminal_current("top")
        _, bot_current = self.backend.terminal_current("bot")
        self.results.voltages.append(top_voltage)
        self.results.top_currents.append(1000 * top_current)
        self.results.bot_currents.append(1000 * bot_current)
        self.results.profiles.append(
            {name: self.backend.node_values(name) for name in _PROFILE_QUANTITIES}
        )
        self.results.edge_profiles.append(
            {name: self.backend.edge_values(name) for name in _EDGE_PROFILE_QUANTITIES}
        )

    def _capture_metadata(self):
        r = self.results
        r.x_nm = [xi * 1e7 for xi in self.backend.positions_cm()]
        r.x_edge_nm = [xi * 1e7 for xi in self.backend.edge_midpoints()]
        r.mun = self.backend.region_parameter("MUN")
        r.mup = self.backend.region_parameter("MUP")
        r.band_gap = self.backend.region_parameter("EG300")
