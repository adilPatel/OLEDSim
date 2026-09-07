#!/usr/bin/env python3
"""Mobility and carrier density profiles for the 6d3 device at 2 V.

Two panels - electrons and holes.  Each shows the EGDM mobility from this
simulator together with the carrier density that produced it, plus the
reference TCAD density for the same carrier.

Why a second y-axis here: mu and n are different quantities in different
units, and normally a dual axis is a bad idea because the alignment of the
two scales is arbitrary and invents a correlation.  In this case the
correlation IS the subject - EGDM's g1(n) makes mu an explicit function of
the local density - so the pairing is causal rather than coincidental.  The
scales are still arbitrary, so each axis is labelled in its series colour
and the mobility axis is annotated with what g1 does across the plotted
density range, rather than leaving the reader to infer a slope by eye.

Run from a venv with matplotlib:  <venv>/bin/python make_mobility_plot.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

FOLDER = "6d3"
LABEL = "0d001"
MU0N_TEXT = r"$10^{-3}$"
BIAS = 2.0

# two-slot categorical palette, validated (dataviz six checks, light surface)
C_MU = "#3b5bdb"      # mobility
C_DEN = "#c2410c"     # density
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#1a1a1a", "#4a4a4a", "#dcdcdc"


def load_reference(carrier, bias):
    path = os.path.join(HERE, "devices_%s" % FOLDER,
                        "Densities_%s_%s_%s_data.txt" % (FOLDER, LABEL, carrier))
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
            d.append(np.nan)          # the tool emits NaN for failed biases
    x, v, d = np.array(x), np.array(v), np.array(d)
    m = np.isclose(v, bias, atol=1e-6)
    o = np.argsort(x[m])
    return x[m][o], d[m][o]


def load_sim(bias):
    """-> x_nm, n_cm3, p_cm3, mun, mup  (mobilities in m^2/Vs)"""
    path = os.path.join(HERE, "devices_%s" % FOLDER,
                        "device_profiles_%s_%s.dat" % (FOLDER, LABEL))
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
    a = blocks[min(blocks, key=lambda z: abs(z - bias))]
    # x n p phi E R mun mup nt pt Jn Jp
    return a[:, 0] * 1e9, a[:, 1] * 1e-6, a[:, 2] * 1e-6, a[:, 6], a[:, 7]


def panel(ax, x, mu, dens, rx, rd, carrier):
    sym = "n" if carrier == "e" else "p"
    name = "Electrons" if carrier == "e" else "Holes"

    # left axis: mobility
    ax.semilogy(x, mu, "-", color=C_MU, lw=2.2, zorder=4)
    ax.set_ylabel(r"Mobility  $\mu_%s$  (m$^2$/Vs)" % sym,
                  fontsize=10, color=C_MU)
    ax.tick_params(axis="y", colors=C_MU, labelsize=9)
    ax.tick_params(axis="x", colors=INK2, labelsize=9)
    ax.set_xlabel("Position  $x$  (nm)", fontsize=10, color=INK2)
    ax.set_xlim(0, 100)
    ax.set_title("%s  -  $\\mu_%s$ and $%s(x)$" % (name, sym, sym),
                 fontsize=11, color=INK, pad=8)
    ax.grid(True, which="major", color=GRID, lw=0.7, zorder=0)
    ax.set_facecolor(SURFACE)
    for sp in ("top",):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(C_MU)
    ax.spines["bottom"].set_color(GRID)

    # right axis: densities, simulated and reference
    ax2 = ax.twinx()
    ax2.semilogy(x, dens, "-", color=C_DEN, lw=2.2, zorder=3)
    ax2.semilogy(rx, rd, "--", color=C_DEN, lw=1.8, dashes=(5, 2.5), zorder=3)
    ax2.set_ylabel(r"Carrier density  $%s$  (cm$^{-3}$)" % sym,
                   fontsize=10, color=C_DEN)
    ax2.tick_params(axis="y", colors=C_DEN, labelsize=9)
    ax2.spines["right"].set_color(C_DEN)
    ax2.spines["top"].set_visible(False)
    ax2.set_facecolor("none")

    # keep the density axis showing the informative range, not the
    # contact spikes and depletion tails
    good = np.concatenate([dens[dens > 0], rd[np.isfinite(rd) & (rd > 0)]])
    hi = 10 ** np.ceil(np.log10(good.max()))
    ax2.set_ylim(hi / 1e10, hi)
    return ax2


def main():
    x, n, p, mun, mup = load_sim(BIAS)
    rxn, rdn = load_reference("n", BIAS)
    rxp, rdp = load_reference("p", BIAS)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    fig.patch.set_facecolor(SURFACE)
    panel(axes[0], x, mun, n, rxn, rdn, "e")
    panel(axes[1], x, mup, p, rxp, rdp, "h")

    handles = [
        Line2D([], [], color=C_MU, lw=2.2, label="Mobility (this work)"),
        Line2D([], [], color=C_DEN, lw=2.2, label="Density (this work)"),
        Line2D([], [], color=C_DEN, lw=1.8, ls="--", dashes=(5, 2.5),
               label="Density (reference TCAD)"),
    ]
    fig.legend(handles=handles, fontsize=9.5, frameon=False, labelcolor=INK2,
               ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.012))

    fig.suptitle(r"F8BT 100 nm EGDM at 2.0 V  -  anode 6.3 eV (ohmic), "
                 r"$\mu_{0n}$ = %s cm$^2$/Vs" % MU0N_TEXT,
                 fontsize=12.5, color=INK, y=0.975)
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    out = os.path.join(HERE, "devices_%s" % FOLDER,
                       "mobility_%s_%s_2V.png" % (FOLDER, LABEL))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.relpath(out, HERE))

    # numbers behind the picture
    m = (x >= 10) & (x <= 95)
    print("\nbulk medians (x = 10-95 nm) at %.1f V:" % BIAS)
    print("  mu_n = %.3e m2/Vs   n = %.3e cm^-3   ref n = %.3e"
          % (np.median(mun[m]), np.median(n[m]),
             np.median(np.interp(x[m], rxn, rdn))))
    print("  mu_p = %.3e m2/Vs   p = %.3e cm^-3   ref p = %.3e"
          % (np.median(mup[m]), np.median(p[m]),
             np.median(np.interp(x[m], rxp, rdp))))
    print("  mu_n range across device: %.3e .. %.3e (%.0fx)"
          % (mun.min(), mun.max(), mun.max() / mun.min()))
    print("  mu_p range across device: %.3e .. %.3e (%.0fx)"
          % (mup.min(), mup.max(), mup.max() / mup.min()))


if __name__ == "__main__":
    main()
