#!/usr/bin/env python3
"""Compare computed EGDM mobilities against the reference export (5d5, 5 V).

`Mobilities_5d5_0d001_5V_data.txt` is the only mobility export in the whole
dataset, and it is on the same cell-centred mesh we use (0.5 .. 99.5 nm), so
this is a node-for-node comparison with no interpolation.

Two panels, electrons and holes.  Each shows our mu(x), the reference mu(x),
and - on a second axis - the carrier density that drives g1(n), since the
whole point of EGDM is that mu is a function of the local density.

Note the file is at 5 V, a bias where the 0d001 DENSITY export is entirely
NaN: the reference tool converged there but failed to write the densities.
The mobility export survived, which is what makes this comparison possible.

Run from a venv with matplotlib:  <venv>/bin/python make_mobility_compare.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

FOLDER, LABEL, BIAS = "5d5", "0d001", 5.0

# validated two-slot categorical palette (dataviz six checks, light surface)
C_SIM = "#3b5bdb"
C_REF = "#c2410c"
C_DEN = "#0ca678"
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#1a1a1a", "#4a4a4a", "#dcdcdc"


def load_ref_mobility():
    path = os.path.join(HERE, "devices_%s" % FOLDER,
                        "Mobilities_%s_%s_5V_data.txt" % (FOLDER, LABEL))
    d = np.array([[float(t) for t in l.split()]
                  for l in open(path) if not l.startswith("#")])
    return d[:, 0], d[:, 1], d[:, 2]        # x_nm, muN, muP  [cm^2/Vs]


def load_sim(bias):
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
    # x n p phi E R mun mup nt pt Jn Jp   (mobilities m^2/Vs -> cm^2/Vs)
    return (a[:, 0] * 1e9, a[:, 1] * 1e-6, a[:, 2] * 1e-6,
            a[:, 6] * 1e4, a[:, 7] * 1e4)


def panel(ax, x, mu_sim, mu_ref, dens, carrier):
    sym = "n" if carrier == "e" else "p"
    name = "Electrons" if carrier == "e" else "Holes"

    ax.semilogy(x, mu_sim, "-", color=C_SIM, lw=2.2, zorder=4)
    ax.semilogy(x, mu_ref, "--", color=C_REF, lw=1.9, dashes=(5, 2.5), zorder=5)
    ax.set_ylabel(r"Mobility  $\mu_%s$  (cm$^2$/Vs)" % sym, fontsize=10, color=INK2)
    ax.set_xlabel("Position  $x$  (nm)", fontsize=10, color=INK2)
    ax.set_xlim(0, 100)

    m = (x >= 20) & (x <= 90)
    ratio = np.median(mu_sim[m] / mu_ref[m])
    ax.set_title("%s  -  bulk $\\mu$ ratio (ours/ref) = %.2f" % (name, ratio),
                 fontsize=11, color=INK, pad=8)
    ax.grid(True, which="major", color=GRID, lw=0.7, zorder=0)
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)

    ax2 = ax.twinx()
    ax2.semilogy(x, np.clip(dens, 1e-30, None), ":", color=C_DEN, lw=1.8, zorder=2)
    ax2.set_ylabel(r"$%s(x)$  (cm$^{-3}$)" % sym, fontsize=10, color=C_DEN)
    ax2.tick_params(axis="y", colors=C_DEN, labelsize=9)
    ax2.spines["right"].set_color(C_DEN)
    ax2.spines["top"].set_visible(False)
    ax2.set_facecolor("none")
    good = dens[dens > 0]
    if good.size:
        hi = 10 ** np.ceil(np.log10(good.max()))
        ax2.set_ylim(hi / 1e12, hi)


def main():
    rx, rmn, rmp = load_ref_mobility()
    sx, n, p, smn, smp = load_sim(BIAS)
    assert np.max(np.abs(sx - rx)) < 1e-9, "grids differ"

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    fig.patch.set_facecolor(SURFACE)
    panel(axes[0], sx, smn, rmn, n, "e")
    panel(axes[1], sx, smp, rmp, p, "h")

    handles = [
        Line2D([], [], color=C_SIM, lw=2.2, label="This work (oled_sim)"),
        Line2D([], [], color=C_REF, lw=1.9, ls="--", dashes=(5, 2.5),
               label="Reference TCAD"),
        Line2D([], [], color=C_DEN, lw=1.8, ls=":", label="Carrier density (ours)"),
    ]
    fig.legend(handles=handles, fontsize=9.5, frameon=False, labelcolor=INK2,
               ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.012))
    fig.suptitle(r"EGDM mobility vs reference at 5.0 V  -  anode 5.5 eV, "
                 r"$\mu_{0n}$ = $10^{-3}$ cm$^2$/Vs",
                 fontsize=12.5, color=INK, y=0.975)
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    out = os.path.join(HERE, "devices_%s" % FOLDER,
                       "mobility_compare_%s_%s_5V.png" % (FOLDER, LABEL))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.relpath(out, HERE))

    m = (sx >= 20) & (sx <= 90)
    print("\nbulk (x = 20-90 nm) mobility ratio ours/reference:")
    print("  electrons: %.3f   holes: %.3f"
          % (np.median(smn[m] / rmn[m]), np.median(smp[m] / rmp[m])))
    print("\nreference muN floor in the depleted anode half (x < 35 nm):")
    print("  ref %.4e vs our g1->1 base %.4e  ->  %.3f"
          % (rmn[:35].min(), smn[:35].min(), rmn[:35].min() / smn[:35].min()))


if __name__ == "__main__":
    main()
