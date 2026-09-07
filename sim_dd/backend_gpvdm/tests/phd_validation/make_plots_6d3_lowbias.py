#!/usr/bin/env python3
"""Low-bias density profiles for the 6.3 eV (ohmic-anode) devices.

One figure per mu0n, showing n(x) and p(x) at 1.8 / 1.9 / 2.0 / 2.1 / 2.2 V:
solid lines from oled_sim, dashed lines from the reference TCAD data.

This is the sub-turn-on regime.  With both contacts ohmic the built-in voltage
is V_bi = 6.3 - 2.9 = 3.4 V, so every bias plotted here sits well below it and
the profiles are set by the equilibrium band bending rather than by injection.
That is where essentially all of the 6d3 disagreement lives; above V_bi the two
codes agree to a few percent (see make_plots.py's summary).

Run from a venv that has matplotlib:  <venv>/bin/python make_plots_6d3_lowbias.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = "6d3"
ANODE_WF = 6.3
MU = [("0d001", r"$10^{-3}$"), ("0d01", r"$10^{-2}$"), ("0d1", r"$10^{-1}$")]
BIASES = [1.8, 1.9, 2.0, 2.1, 2.2]

# validated categorical palette (dataviz six-checks, light surface): all PASS at
# 5 slots.  A 5-step sequential ramp of one hue was tried first and fails the
# normal-vision floor - adjacent steps are not separable - so identity here is
# categorical, reinforced by the solid/dashed split for source.
COLORS = ["#3b5bdb", "#c2410c", "#0ca678", "#9333ea", "#a16207"]
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#1a1a1a", "#4a4a4a", "#dcdcdc"


def load_reference(label, carrier):
    """Reference file -> {V: (x_nm, density_cm^-3)}; all-NaN biases dropped."""
    path = os.path.join(HERE, "devices_%s" % FOLDER,
                        "Densities_%s_%s_%s_data.txt" % (FOLDER, label, carrier))
    if not os.path.exists(path):
        return {}
    x, v, d = [], [], []
    for line in open(path):
        if line.startswith("#"):
            continue
        a = line.split()
        if len(a) < 3:
            continue
        x.append(float(a[0])); v.append(float(a[1]))
        try:
            d.append(float(a[2]))
        except ValueError:
            d.append(np.nan)
    x, v, d = np.array(x), np.array(v), np.array(d)
    out = {}
    for bias in BIASES:
        m = np.isclose(v, bias, atol=1e-6)
        if not m.any():
            continue
        xs, ds = x[m], d[m]
        if np.all(np.isnan(ds)):
            continue
        o = np.argsort(xs)
        out[bias] = (xs[o], ds[o])
    return out


def load_sim(label):
    """device_profiles -> {V: (x_nm, n_cm^-3, p_cm^-3)}."""
    path = os.path.join(HERE, "devices_%s" % FOLDER,
                        "device_profiles_%s_%s.dat" % (FOLDER, label))
    if not os.path.exists(path):
        return {}
    blocks, cur, V = {}, [], None
    for line in open(path):
        s = line.strip()
        if s.startswith("# V ="):
            if V is not None and cur:
                blocks[V] = np.array(cur)
            V = float(s.split("=")[1]); cur = []
        elif s and not s.startswith("#"):
            cur.append([float(t) for t in s.split()])
    if V is not None and cur:
        blocks[V] = np.array(cur)

    out = {}
    for bias in BIASES:
        key = min(blocks, key=lambda k: abs(k - bias))
        if abs(key - bias) > 0.051:      # the sweep step is 0.1 V
            continue
        a = blocks[key]
        out[bias] = (a[:, 0] * 1e9, a[:, 1] * 1e-6, a[:, 2] * 1e-6)
    return out


def panel(ax, sim, ref, col, title):
    for c, bias in zip(COLORS, BIASES):
        if bias in sim:
            xs = sim[bias][0]
            ax.semilogy(xs, sim[bias][col], "-", color=c, lw=2.0, zorder=3)
        if bias in ref:
            rx, rd = ref[bias]
            ax.semilogy(rx, rd, "--", color=c, lw=1.7, dashes=(5, 2.5), zorder=4)
    ax.set_title(title, fontsize=11, color=INK, pad=8)
    ax.set_xlabel("Position  $x$  (nm)", fontsize=10, color=INK2)
    ax.set_xlim(0, 100)
    ax.grid(True, which="major", color=GRID, lw=0.7, zorder=0)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.5, zorder=0)
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def main():
    for label, mutex in MU:
        sim = load_sim(label)
        refn, refp = load_reference(label, "n"), load_reference(label, "p")
        if not sim:
            print("no simulation output for", label)
            continue

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
        fig.patch.set_facecolor(SURFACE)
        panel(axes[0], sim, refn, 1, "Free electrons  $n(x)$")
        panel(axes[1], sim, refp, 2, "Free holes  $p(x)$")
        axes[0].set_ylabel(r"Carrier density  (cm$^{-3}$)", fontsize=10, color=INK2)

        # y-range from the device interior only: the contact spikes and the
        # depletion tails would otherwise flatten everything of interest
        vals = []
        for d, c in ((refn, 1), (refp, 2)):
            for _, r in d.items():
                m = (r[0] >= 5) & (r[0] <= 95)
                g = r[1][m]
                g = g[np.isfinite(g) & (g > 0)]
                if g.size:
                    vals.append(g)
            for b in sim:
                xs = sim[b][0]
                m = (xs >= 5) & (xs <= 95)
                g = sim[b][c][m]
                g = g[g > 0]
                if g.size:
                    vals.append(g)
        allv = np.concatenate(vals)
        lo = 10 ** np.floor(np.log10(allv.min()))
        hi = 10 ** np.ceil(np.log10(allv.max()))
        axes[0].set_ylim(max(lo, hi / 1e10), hi)

        handles = [Line2D([], [], color=c, lw=2.0, label="%.1f V" % b)
                   for c, b in zip(COLORS, BIASES)]
        handles += [Line2D([], [], color=INK2, lw=2.0, label="This work (oled_sim)"),
                    Line2D([], [], color=INK2, lw=1.7, ls="--", dashes=(5, 2.5),
                           label="Reference TCAD")]
        fig.legend(handles=handles, fontsize=9.5, frameon=False, labelcolor=INK2,
                   ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.012))

        fig.suptitle(r"F8BT 100 nm EGDM,  anode %.1f eV (ohmic),  $\mu_{0n}$ = %s cm$^2$/Vs"
                     "\n" r"sub-turn-on detail ($V_{bi}$ = 3.4 V)"
                     % (ANODE_WF, mutex), fontsize=12.5, color=INK, y=0.995)
        fig.tight_layout(rect=[0, 0.075, 1, 0.90])
        out = os.path.join(HERE, "devices_%s" % FOLDER,
                           "densities_%s_%s_lowbias.png" % (FOLDER, label))
        fig.savefig(out, dpi=200, facecolor=SURFACE)
        plt.close(fig)
        print("wrote", os.path.relpath(out, HERE))


if __name__ == "__main__":
    main()
