"""
Plotting and result-saving helpers for the drift-diffusion solver.

These functions consume a solver-independent ``core.dd_solver.SweepResults``
(and, for the edge-model diagnostic plots, the back end that produced it). They
exist to keep front-end scripts like ``devices/oled1.py`` short.

None of the *validation* / *saving* functions import DEVSIM. The diagnostic
edge-model plots read edge quantities through the back end's ``edge_values`` /
``edge_midpoints`` methods, so they too stay solver-agnostic.
"""

import os

import matplotlib
import matplotlib.pyplot
from matplotlib.widgets import Slider


# ---- validation / saving (format defined in tests/diode_1d/format.md) --------

def validation_dir_for(device_name, base=None):
    """Return (and create) the tests/diode_1d/<device> output directory."""
    if base is None:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "tests", "diode_1d")
    out = os.path.join(base, device_name.lower())
    os.makedirs(out, exist_ok=True)
    return out


def save_profiles(results, device_name, out_dir=None):
    """Write the per-voltage profile table (x, v, potential, n, p, muN, muP)."""
    if out_dir is None:
        out_dir = validation_dir_for(device_name)
    path = os.path.join(out_dir, "{0}_devsim_profiles.txt".format(device_name.lower()))

    electrons = results.profile("Electrons")
    holes = results.profile("Holes")
    potential = results.profile("Potential")

    with open(path, "w") as f:
        f.write("{0}\n".format(device_name))
        f.write("{0}\n".format(results.band_gap))
        f.write("{0}\n".format(results.built_in_voltage))
        f.write("\n")
        f.write("x (nm) | v (V) | potential (V) | n (cm^-3) | p (cm^-3) | "
                "muN (cm^2/Vs) | muP (cm^2/Vs)\n")
        for i, v_step in enumerate(results.voltages):
            if i > 0:
                f.write("\n")
            for xi, ni, pi, poti in zip(results.x_nm, electrons[i], holes[i], potential[i]):
                f.write("{0:e}\t{1:e}\t{2:e}\t{3:e}\t{4:e}\t{5:e}\t{6:e}\n".format(
                    xi, v_step, poti, ni, pi, results.mun, results.mup))
    return path


def save_figures(results, device_name, out_dir=None):
    """Write the I-V figure table (V-Vbi, J in mA/cm^2)."""
    if out_dir is None:
        out_dir = validation_dir_for(device_name)
    path = os.path.join(out_dir, "{0}_devsim_figures.txt".format(device_name.lower()))

    with open(path, "w") as f:
        f.write("{0}\n".format(device_name))
        f.write("{0}\n".format(results.built_in_voltage))
        f.write("\n")
        f.write("V-Vbi (V) | J (mA/cm^2)\n")
        for v_step, j_step in zip(results.voltages, results.top_currents):
            f.write("{0:e}\t{1:e}\n".format(v_step - results.built_in_voltage, j_step))
    return path


def save_validation(results, device_name, out_dir=None):
    """Write both the profile and figure validation tables."""
    return (save_profiles(results, device_name, out_dir),
            save_figures(results, device_name, out_dir))


# ---- interactive slider plots ------------------------------------------------

def density_slider(results, x_cm):
    """Interactive semilog plot of the carrier/doping densities vs position,
    with a slider over the swept bias points. Returns the Figure.
    """
    fig, ax = matplotlib.pyplot.subplots()
    matplotlib.pyplot.subplots_adjust(bottom=0.25)

    histories = {
        "Electrons": results.profile("Electrons"),
        "Holes": results.profile("Holes"),
        "Donors": results.profile("Donors"),
        "Acceptors": results.profile("Acceptors"),
    }
    lines = {}
    for name, history in histories.items():
        (line,) = ax.semilogy(x_cm, history[0], label=name)
        lines[name] = line
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("Density (#/cm^3)")
    ax.set_xlim(min(x_cm), max(x_cm))
    all_values = [v for history in histories.values() for frame in history for v in frame]
    ax.set_ylim(min(v for v in all_values if v > 0), max(all_values) * 10)
    ax.legend()
    ax.set_title("V = {0:.2f} V".format(results.voltages[0]))

    slider_ax = matplotlib.pyplot.axes((0.2, 0.1, 0.6, 0.03))
    slider = Slider(slider_ax, "Voltage index", 0, len(results.voltages) - 1,
                    valinit=0, valstep=1)

    def update(_val):
        idx = int(slider.val)
        for name, history in histories.items():
            lines[name].set_ydata(history[idx])
        ax.set_title("V = {0:.2f} V".format(results.voltages[idx]))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    # Keep the slider alive past this function's scope.
    fig._voltage_slider = slider
    return fig


def current_slider(results):
    """Interactive semilog plot of electron/hole current density vs position,
    with a slider over the swept bias points. Uses the edge-quantity profiles
    (results.edge_profile/results.x_edge_nm) captured across the sweep, unlike
    plot_current (a static plot of the backend's *final* bias point only).
    Returns the Figure.
    """
    fig, ax = matplotlib.pyplot.subplots()
    matplotlib.pyplot.subplots_adjust(bottom=0.25)

    x_cm = [xi * 1e-7 for xi in results.x_edge_nm]
    histories = {
        "Jn_const": results.edge_profile("Jn_const"),
        "Jp_const": results.edge_profile("Jp_const"),
    }
    lines = {}
    for name, history in histories.items():
        (line,) = ax.semilogy(x_cm, [abs(v) for v in history[0]], label=name)
        lines[name] = line
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("|J| (A/cm^2)")
    ax.set_xlim(min(x_cm), max(x_cm))
    all_values = [abs(v) for history in histories.values() for frame in history for v in frame]
    positive_values = [v for v in all_values if v > 0]
    if positive_values:
        ax.set_ylim(min(positive_values), max(positive_values) * 10)
    ax.legend()
    ax.set_title("V = {0:.2f} V".format(results.voltages[0]))

    slider_ax = matplotlib.pyplot.axes((0.2, 0.1, 0.6, 0.03))
    slider = Slider(slider_ax, "Voltage index", 0, len(results.voltages) - 1,
                    valinit=0, valstep=1)

    def update(_val):
        idx = int(slider.val)
        for name, history in histories.items():
            lines[name].set_ydata([abs(v) for v in history[idx]])
        ax.set_title("V = {0:.2f} V".format(results.voltages[idx]))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    fig._voltage_slider = slider
    return fig


def potential_slider(results, x_cm):
    """Interactive plot of Potential vs position, with a bias slider."""
    fig, ax = matplotlib.pyplot.subplots()
    matplotlib.pyplot.subplots_adjust(bottom=0.25)

    history = results.profile("Potential")
    (line,) = ax.plot(x_cm, history[0])
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("Potential (V)")
    ax.set_xlim(min(x_cm), max(x_cm))
    all_values = [v for frame in history for v in frame]
    ax.set_ylim(min(all_values), max(all_values))
    ax.set_title("V = {0:.2f} V".format(results.voltages[0]))

    slider_ax = matplotlib.pyplot.axes((0.2, 0.1, 0.6, 0.03))
    slider = Slider(slider_ax, "Voltage index", 0, len(results.voltages) - 1,
                    valinit=0, valstep=1)

    def update(_val):
        idx = int(slider.val)
        line.set_ydata(history[idx])
        ax.set_title("V = {0:.2f} V".format(results.voltages[idx]))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    fig._voltage_slider = slider
    return fig


# ---- static diagnostic figures (final bias point) ----------------------------

def _edge_pair_plot(backend, x_cm, edge_models, ylabel, path):
    """Plot a pair of edge models vs the edge midpoints and save to ``path``."""
    matplotlib.pyplot.figure()
    xmid = backend.edge_midpoints()
    y = backend.edge_values(edge_models[0])
    ymin, ymax = min(y), max(y)
    for name in edge_models:
        y = backend.edge_values(name)
        if min(y) < ymin:
            ymin = min(y)
        elif max(y) > ymax:
            ymax = max(y)
        matplotlib.pyplot.plot(xmid, y)
    matplotlib.pyplot.xlabel("x (cm)")
    matplotlib.pyplot.ylabel(ylabel)
    matplotlib.pyplot.legend(edge_models)
    matplotlib.pyplot.axis((min(x_cm), max(x_cm), 0.5 * ymin, 2 * ymax))
    matplotlib.pyplot.savefig(path)
    return ymin, ymax


def plot_current(backend, x_cm, path="diode_os_current.eps"):
    """Electron/hole current density along the device (final bias point)."""
    return _edge_pair_plot(backend, x_cm, ("Jn_const", "Jp_const"),
                           "J (A/cm^2)", path)


def plot_mobility(backend, x_cm, path="diode_os_mobility.eps"):
    """Electron/hole mobility along the device (final bias point)."""
    return _edge_pair_plot(backend, x_cm, ("mu_n_const", "mu_p_const"),
                           "J (A/cm^2)", path)


def plot_langevin(backend, x_cm, path="ULANG.eps"):
    """Langevin recombination density along the device (final bias point)."""
    matplotlib.pyplot.figure()
    ymin, ymax = 10, 10
    y = backend.node_values("ULANG")
    if max(y) > ymax:
        ymax = max(y)
    matplotlib.pyplot.semilogy(x_cm, y)
    matplotlib.pyplot.xlabel("x (cm)")
    matplotlib.pyplot.ylabel("Density (#/cm^3)")
    matplotlib.pyplot.legend(("ULANG",))
    matplotlib.pyplot.axis((min(x_cm), max(x_cm), ymin, ymax * 10))
    matplotlib.pyplot.savefig(path)


def plot_iv(results, path="diode_os_iv.eps"):
    """Terminal |I|-V curve on a semilog axis."""
    matplotlib.pyplot.figure()
    matplotlib.pyplot.semilogy(results.voltages,
                               [abs(i) for i in results.top_currents], label="top")
    matplotlib.pyplot.xlabel("V (V)")
    matplotlib.pyplot.ylabel("I (mA)")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig(path)


def show():
    matplotlib.pyplot.show()
