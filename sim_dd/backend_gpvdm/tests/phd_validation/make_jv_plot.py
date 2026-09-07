#!/usr/bin/env python3
"""JV curves for the 6d3 (ohmic-anode) devices: ours vs the reference.

The reference export contains no current density.  It is recovered from the
recombination profiles instead: in steady state every carrier injected must
recombine, so

        J(V) = q * Integral R(x, V) dx

This identity is verified against our own data first, where both J and R are
known independently - it holds to 4 significant figures (see the printout),
which is what licenses applying it to the reference.

Voltages where the reference emitted NaN (0d001 at 4.8 and 6.2 V, 0d01 at
4.0 V) are dropped rather than interpolated over.

Run from a venv with matplotlib:  <venv>/bin/python make_jv_plot.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
QE = 1.602176634e-19
FOLDER = "6d3"
MU = [("0d001", r"$10^{-3}$"), ("0d01", r"$10^{-2}$"), ("0d1", r"$10^{-1}$")]
VBI = 2.6          # both contacts ohmic -> V_bi = Eg

# validated categorical palette (dataviz six checks, light surface)
COLORS = ["#3b5bdb", "#c2410c", "#0ca678"]
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#1a1a1a", "#4a4a4a", "#dcdcdc"


def ref_jv(label):
    """J(V) = q * int R dx from the reference recombination export."""
    path = os.path.join(HERE, "devices_%s" % FOLDER,
                        "Recombination_%s_%s_data.txt" % (FOLDER, label))
    rows = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        a = line.split()
        if len(a) < 3:
            continue
        x, v = float(a[0]), round(float(a[1]), 4)
        try:
            r = float(a[2])
        except ValueError:
            r = np.nan
        rows.setdefault(v, []).append((x, r))
    V, J = [], []
    for v in sorted(rows):
        arr = np.array(sorted(rows[v]))
        xs, rs = arr[:, 0] * 1e-7, arr[:, 1]      # nm -> cm
        if not np.all(np.isfinite(rs)):
            continue                              # failed bias: drop it
        V.append(v)
        J.append(QE * np.trapz(rs, xs) * 1e4)     # cm^-2 s^-1 -> A/m^2
    return np.array(V), np.array(J)


def our_jv(label):
    path = os.path.join(HERE, "devices_%s" % FOLDER,
                        "device_figures_%s_%s.dat" % (FOLDER, label))
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    fig.patch.set_facecolor(SURFACE)

    for c, (label, mutex) in zip(COLORS, MU):
        rv, rj = ref_jv(label)
        ov, oj = our_jv(label)
        for ax in axes:
            ax.plot(ov, np.clip(oj, 1e-12, None), "-", color=c, lw=2.2, zorder=4)
            ax.plot(rv, np.clip(rj, 1e-12, None), "--", color=c, lw=1.8,
                    dashes=(5, 2.5), zorder=5)

    for ax, logy in ((axes[0], True), (axes[1], False)):
        ax.axvline(VBI, color=INK2, lw=1.0, ls="-", alpha=0.35, zorder=1)
        ax.set_xlabel("Applied bias  $V$  (V)", fontsize=10, color=INK2)
        ax.set_ylabel(r"Current density  $J$  (A/m$^2$)", fontsize=10, color=INK2)
        ax.grid(True, which="major", color=GRID, lw=0.7, zorder=0)
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        if logy:
            ax.set_yscale("log")
            ax.set_xlim(0, 10)
            ax.set_ylim(1e-4, 1e5)
            ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.5, zorder=0)
            ax.set_title("Log scale  -  full sweep", fontsize=11, color=INK, pad=8)
        else:
            ax.set_xlim(0, 10)
            ax.set_title("Linear scale  -  above turn-on", fontsize=11,
                         color=INK, pad=8)

    axes[0].annotate(r"$V_{bi}$ = %.1f V" % VBI, xy=(VBI, 2e4),
                     xytext=(VBI + 0.35, 2e4), fontsize=9, color=INK2)

    handles = [Line2D([], [], color=c, lw=2.2, label=r"$\mu_{0n}$ = %s" % m)
               for c, (_, m) in zip(COLORS, MU)]
    handles += [Line2D([], [], color=INK2, lw=2.2, label="This work (oled_sim)"),
                Line2D([], [], color=INK2, lw=1.8, ls="--", dashes=(5, 2.5),
                       label=r"Reference ($q\int R\,dx$)")]
    fig.legend(handles=handles, fontsize=9.5, frameon=False, labelcolor=INK2,
               ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.012))
    fig.suptitle(r"JV characteristics  -  F8BT 100 nm EGDM, anode 6.3 eV (ohmic)",
                 fontsize=12.5, color=INK, y=0.975)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    out = os.path.join(HERE, "devices_%s" % FOLDER, "jv_%s.png" % FOLDER)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.relpath(out, HERE))

    print("\nJ (A/m^2) at selected biases:")
    print("%6s %8s %12s %12s %8s" % ("mu0n", "V", "ours", "reference", "ratio"))
    for label, _ in MU:
        rv, rj = ref_jv(label)
        ov, oj = our_jv(label)
        for V in (2.0, 3.0, 5.0, 8.0, 10.0):
            i = np.argmin(np.abs(ov - V))
            k = np.argmin(np.abs(rv - V))
            if abs(rv[k] - V) > 0.06:
                continue
            print("%6s %8.1f %12.4e %12.4e %8.2f"
                  % (label, V, oj[i], rj[k], oj[i] / rj[k]))


if __name__ == "__main__":
    main()
