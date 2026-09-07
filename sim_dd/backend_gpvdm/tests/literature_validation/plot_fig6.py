#!/usr/bin/env python3
# plot_fig6.py - reproduce Knapp et al., J. Appl. Phys. 108, 054504 (2010),
# Fig. 6: log-log J-V of a single-carrier device with EGDM transport, with
# and without an exponential trap distribution.
#
# Device parameters (from the paper, see oled_fig6.ini for the derivations):
#   L = 129nm, T = 293K, phiB = 0.3eV, N0 = 1e27 m^-3, sigma = 0.07eV,
#   mu0 = 1e-9 m^2/Vs, Vbi = 0.9V, Ntrap = 1e24 m^-3, T_trap = 2100K.
#
# Both devices are simulated.  The trapped deck uses quasi-equilibrium traps
# (trap_kinetics = equilibrium), as Knapp et al. do: the traps share the free
# holes' quasi-Fermi level, so trapped charge appears in Poisson only, the
# drift-diffusion currents carry free carriers alone, and there is no
# trap-assisted recombination.  This also removes the ill-conditioning that
# made the SRH-kinetics form stall - the trap rows become the identity
# xpt = xp instead of a rate balance whose deep-band derivatives underflow.
#
# Usage (run from anywhere):
#   python3 plot_fig6.py

import os
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OLED_SIM = os.path.join(BACKEND_DIR, "oled_sim")

CASES = [
    # (deck, output dir, label, colour, linestyle)
    ("oled_fig6_trapfree.ini", "output_fig6_trapfree", "trapfree", "tab:red", "-"),
    ("oled_fig6.ini", "output_fig6_traps", "traps", "black", "--"),
]


def run(deck):
    path = os.path.join(SCRIPT_DIR, deck)
    r = subprocess.run([OLED_SIM, path], cwd=BACKEND_DIR,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:])
        print(r.stderr, file=sys.stderr)
        sys.exit(f"oled_sim failed on {deck}")


def read_jv(out_dir):
    path = os.path.join(BACKEND_DIR, out_dir, "device_figures.dat")
    d = np.loadtxt(path, comments="#")
    return d[:, 0], d[:, 1]


def main():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for csv, lab, col in [("knapp_fig6_trapfree_digitised.csv",
                           "Knapp Fig. 6 trap-free (digitised)", "tab:red"),
                          ("knapp_fig6_traps_digitised.csv",
                           "Knapp Fig. 6 traps (digitised)", "black")]:
        path = os.path.join(SCRIPT_DIR, csv)
        if not os.path.exists(path):
            continue
        ref = np.loadtxt(path, delimiter=",", skiprows=1)
        ax.loglog(ref[:, 0], ref[:, 1], "-", color=col, linewidth=4.0,
                  alpha=0.25, label=lab, zorder=1)

    for deck, out_dir, label, colour, ls in CASES:
        print(f"running {deck} ...")
        run(deck)
        V, J = read_jv(out_dir)
        # log-log: keep the forward-bias, positive-current branch
        m = (V > 0) & (J > 0)
        ax.loglog(V[m], J[m], ls, color=colour, linewidth=2, label=label)
        print(f"  {label}: {m.sum()} points, "
              f"J({V[m][-1]:.1f}V) = {J[m][-1]:.3e} A/m^2")

    ax.set_xlabel("voltage [V]")
    ax.set_ylabel("current density J [A/m$^2$]")
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(1e-12, 1e9)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    ax.set_title("Knapp Fig. 6 - EGDM single-carrier device, 129nm, 293K\n"
                 "reduced-field clamp $qaF/\\sigma \\leq 2$; "
                 "quasi-equilibrium traps", fontsize=11)
    fig.tight_layout()

    out_png = os.path.join(SCRIPT_DIR, "fig6_reproduction.png")
    fig.savefig(out_png, dpi=150)
    print(f"plot written to {out_png}")


if __name__ == "__main__":
    main()
