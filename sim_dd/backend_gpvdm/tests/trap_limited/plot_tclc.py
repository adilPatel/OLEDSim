#!/usr/bin/env python3
"""plot_tclc.py - log-log J-V of the trap-limited device and its trap-free
control, with the Mark-Helfrich and Mott-Gurney power laws overlaid.

Run verify_tclc.py first (or this script will run the decks itself).
"""
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OLED_SIM = os.path.join(BACKEND_DIR, "oled_sim")

KT = 1.380649e-23 * 300 / 1.602176634e-19
ET = 0.075
L, MU, EPSR = 200e-9, 1e-9, 3.0
FIT_VMIN = 5.0


def jv(out_dir, deck):
    path = os.path.join(BACKEND_DIR, out_dir, "device_figures.dat")
    if not os.path.exists(path):
        subprocess.run([OLED_SIM, os.path.join(SCRIPT_DIR, deck)],
                       cwd=BACKEND_DIR, capture_output=True, text=True)
    d = np.loadtxt(path)
    V, J = d[:, 0], d[:, 1]
    m = (V > 0) & (J > 0)
    return V[m], J[m]


def main():
    Vf, Jf = jv("tests/trap_limited/out_trapfree", "tlc_trapfree.ini")
    Vt, Jt = jv("tests/trap_limited/out_traps", "tlc_traps.ini")

    l = ET / KT
    fig, ax = plt.subplots(figsize=(7.5, 6))

    ax.loglog(Vf, Jf, "-", color="tab:red", lw=2,
              label="trap-free (control)")
    ax.loglog(Vt, Jt, "-", color="black", lw=2,
              label=f"exponential e-traps, $E_t$={ET} eV")

    # power-law guides, anchored at the top of the fitted window
    for V, J, p, col, lab in [
            (Vf, Jf, 2.0, "tab:red", r"Mott-Gurney  $J\propto V^{2}$"),
            (Vt, Jt, l + 1, "black",
             rf"Mark-Helfrich  $J\propto V^{{l+1}}$, $l+1$={l+1:.2f}")]:
        m = V > FIT_VMIN
        Va = V[m][-1]
        Ja = J[m][-1]
        Vg = np.array([FIT_VMIN * 0.8, V[-1] * 1.15])
        ax.loglog(Vg, Ja * (Vg / Va) ** p, "--", color=col, lw=1.2,
                  alpha=0.75, label=lab)

    ax.set_xlabel("voltage [V]")
    ax.set_ylabel(r"current density $J$ [A/m$^2$]")
    ax.set_title("Trap-limited vs trap-free space-charge-limited current\n"
                 f"200 nm electron-only device, "
                 r"$N_{trap}$=3$\times10^{24}$ m$^{-3}$, "
                 f"$T_t$={ET/8.617333e-5:.0f} K")
    # The sub-turn-on decades (V < ~1.5V, where the device is still fighting
    # the built-in potential and J spans 10 orders of magnitude) compress the
    # power-law region into invisibility; clip to the region the fits cover.
    ax.set_xlim(1.5, 16)
    lo = min(Jt[Vt > 1.5].min(), Jf[Vf > 1.5].min())
    hi = max(Jt.max(), Jf.max())
    ax.set_ylim(lo * 0.4, hi * 4)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, "tclc_jv.png")
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
