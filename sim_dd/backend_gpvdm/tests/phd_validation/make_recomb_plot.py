#!/usr/bin/env python3
"""Recombination profiles for the three 6d3 (ohmic-anode) devices.

Three panels, one per mu0n, each overlaying this simulator's R(x) on the
reference TCAD profile at the same bias.

R is the sharpest test of the model: R = gamma*q*(mu_n+mu_p)/eps * (np - n0p0)
combines the densities AND the EGDM mobilities, so a discrepancy here that is
larger than the density discrepancy alone points at the mobility rather than
at the carrier populations.

Bias: 5 V by default, overridable as argv[1].  The reference emits NaN for
whole voltage columns that failed to converge (0d001 at 4.8 and 6.2 V, 0d01 at
4.0 V); 2 V and 5 V are both clean in all three files.

2 V is the interesting one: it is below the built-in voltage, where the carrier
densities disagree by orders of magnitude.  At 5 V the densities already agree,
so R there mostly tests the recombination prefactor rather than the profile.

Run from a venv with matplotlib:  <venv>/bin/python make_recomb_plot.py [bias]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

FOLDER = "6d3"
BIAS = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
MU = [("0d001", r"$10^{-3}$"), ("0d01", r"$10^{-2}$"), ("0d1", r"$10^{-1}$")]

# two-slot categorical palette, validated (dataviz six checks, light surface):
# lightness band / chroma floor / CVD separation / normal-vision floor /
# contrast all PASS.  Source is additionally encoded solid vs dashed.
C_SIM = "#3b5bdb"
C_REF = "#c2410c"
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#1a1a1a", "#4a4a4a", "#dcdcdc"


def load_reference(label, bias):
    """-> x_nm, R_cm3s1 at `bias`; NaN entries preserved so gaps show."""
    path = os.path.join(HERE, "devices_%s" % FOLDER,
                        "Recombination_%s_%s_data.txt" % (FOLDER, label))
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


def load_sim(label, bias):
    """-> x_nm, R_cm3s1  (device_profiles stores R in m^-3 s^-1)."""
    path = os.path.join(HERE, "devices_%s" % FOLDER,
                        "device_profiles_%s_%s.dat" % (FOLDER, label))
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
    return a[:, 0] * 1e9, a[:, 5] * 1e-6


def main():
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    lo, hi = np.inf, 0.0
    stats = []
    for ax, (label, mutex) in zip(axes, MU):
        sx, sR = load_sim(label, BIAS)
        rx, rR = load_reference(label, BIAS)

        ax.semilogy(sx, np.clip(sR, 1e-300, None), "-", color=C_SIM, lw=2.2, zorder=4)
        ax.semilogy(rx, np.clip(rR, 1e-300, None), "--", color=C_REF, lw=1.9,
                    dashes=(5, 2.5), zorder=3)

        ax.set_title(r"$\mu_{0n}$ = %s cm$^2$/Vs" % mutex,
                     fontsize=11, color=INK, pad=8)
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

        # y-range from the device interior: the contact edges carry
        # near-zero R and would otherwise stretch the axis pointlessly
        good = np.concatenate([
            sR[(sx >= 5) & (sx <= 95) & (sR > 0)],
            rR[(rx >= 5) & (rx <= 95) & np.isfinite(rR) & (rR > 0)]])
        lo = min(lo, good.min()); hi = max(hi, good.max())

        # integrated recombination: what the current must supply
        m = np.isfinite(rR)
        stats.append((label,
                      np.trapz(sR, sx * 1e-7),          # cm^-2 s^-1
                      np.trapz(rR[m], rx[m] * 1e-7)))

    axes[0].set_ylabel(r"Recombination rate  $R$  (cm$^{-3}$s$^{-1}$)",
                       fontsize=10, color=INK2)
    top = 10 ** np.ceil(np.log10(hi))
    axes[0].set_ylim(max(top / 1e8, 10 ** np.floor(np.log10(lo))), top)

    handles = [
        Line2D([], [], color=C_SIM, lw=2.2, label="This work (oled_sim)"),
        Line2D([], [], color=C_REF, lw=1.9, ls="--", dashes=(5, 2.5),
               label="Reference TCAD"),
    ]
    fig.legend(handles=handles, fontsize=9.5, frameon=False, labelcolor=INK2,
               ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.012))
    fig.suptitle(r"Recombination profile at %.1f V  -  F8BT 100 nm EGDM, "
                 r"anode 6.3 eV (ohmic)" % BIAS,
                 fontsize=12.5, color=INK, y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    out = os.path.join(HERE, "devices_%s" % FOLDER,
                       "recombination_%s_%.0fV.png" % (FOLDER, BIAS))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.relpath(out, HERE))

    print("\nintegrated recombination at %.1f V (cm^-2 s^-1):" % BIAS)
    print("%8s %14s %14s %8s" % ("mu0n", "this work", "reference", "ratio"))
    for label, si, ri in stats:
        print("%8s %14.4e %14.4e %8.3f" % (label, si, ri, si / ri))


if __name__ == "__main__":
    main()
