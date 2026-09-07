#!/usr/bin/env python3
# plot_fig4.py - reproduce Knapp et al., J. Appl. Phys. 108, 054504 (2010),
# Fig. 4: J-V of a single-carrier EGDM device at four injection barriers.
#
# Device (see oled_fig4.ini for the full derivation):
#   N0 = 2.44e26 m^-3, mu0 = 1.1e-16 m^2/Vs, sigma/kT = 6, L = 22 nm,
#   no traps, T = 293K (only the ratio sigma/kT is quoted in the paper).
#
# Barriers: Delta = 0, 0.33, 0.67, 1.0 eV.  Delta = 0 uses an ohmic-np
# contact pinned to the Gaussian DOS half-filling (Nc/2); the other three
# use thermionic (Scott-Malliaras) contacts at the stated barrier.  Both
# contacts carry the same barrier so there is no built-in potential, which
# is what the figure shows (smooth power laws, no turn-on knee).
#
# J is read from the BULK, not from the contact flux.  At 0.01V across 22nm
# the terminal current is a small difference between two large opposing
# fluxes and the reported contact value is noisy and mesh-sensitive, while
# the interior Jp is flat to 4-5 digits and converges under refinement
# (3.06e-3 / 3.12e-3 / 3.15e-3 at 100 / 200 / 400 points).
#
# Usage:  python3 plot_fig4.py

import os
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OLED_SIM = os.path.join(BACKEND_DIR, "oled_sim")
BASE = os.path.join(SCRIPT_DIR, "oled_fig4.ini")

NC_HALF = 1.22e26          # Gaussian DOS half-filling = zero-barrier density

# (Delta [eV], colour, marker) - colours follow the paper's figure
CASES = [
    (0.00, "#2e2a8f", "v"),
    (0.33, "black",   "o"),
    (0.67, "#9a9a9a", "^"),
    (1.00, "#d62728", "s"),
]


def deck_for(delta, out_dir):
    """Rewrite the two contact blocks for this barrier."""
    txt = open(BASE).read()
    if delta == 0.0:
        contacts = ("model = ohmic-np\nmajority = hole\nnp = %g" % NC_HALF)
    else:
        contacts = ("model = thermionic\nmajority = hole\nbarrier = %g" % delta)

    out, mode = [], None
    for ln in txt.splitlines():
        s = ln.strip()
        if s.startswith("[contact_anode]") or s.startswith("[contact_cathode]"):
            out.append(ln)
            out.append(contacts)
            mode = "skip"
            continue
        if mode == "skip":
            # drop the old model/majority/np/barrier lines of that block
            if s.startswith(("model", "majority", "np ", "np=", "barrier")):
                continue
            mode = None
        if s.startswith("output_dir"):
            out.append("output_dir = %s" % out_dir)
            continue
        out.append(ln)

    path = os.path.join(SCRIPT_DIR, "_fig4_%s.ini" % str(delta).replace(".", "p"))
    open(path, "w").write("\n".join(out) + "\n")
    return path


def bulk_jv(out_dir):
    """J per voltage, averaged over the interior nodes (see header)."""
    path = os.path.join(BACKEND_DIR, out_dir, "device_profiles.dat")
    V, J = [], []
    v, jp = None, []
    for line in open(path):
        if line.startswith("# V ="):
            if v is not None and jp:
                V.append(v); J.append(np.median(jp))
            v = float(line.split("=")[1]); jp = []
        elif line.strip() and not line.startswith("#"):
            c = line.split()
            if len(c) >= 12:
                jp.append(float(c[11]))
    if v is not None and jp:
        V.append(v); J.append(np.median(jp))
    return np.array(V), np.array(J)


def main():
    fig, ax = plt.subplots(figsize=(7.5, 6))

    # digitised reference (see digitise_fig4.py), drawn first as a thick
    # translucent band so the simulated curves overlay it
    for delta, colour, marker in CASES:
        ref = os.path.join(SCRIPT_DIR,
                           "knapp_fig4_d%s_digitised.csv"
                           % ("%.2f" % delta).replace(".", "p"))
        if not os.path.exists(ref):
            continue
        d = np.loadtxt(ref, delimiter=",", skiprows=1)
        ax.loglog(d[:, 0], d[:, 1], "-", color=colour, linewidth=5.0,
                  alpha=0.22, zorder=1,
                  label="_nolegend_" if delta else "Knapp Fig. 4 (digitised)")

    for delta, colour, marker in CASES:
        # if the ohmic/image-force CSV exists (run_fig4_ohmic.py), use it and
        # skip the in-deck thermionic run entirely
        oh = os.path.join(SCRIPT_DIR, "fig4_ohmic_d%s.csv"
                          % ("%.2f" % delta).replace(".", "p"))
        if os.path.exists(oh):
            d = np.loadtxt(oh, delimiter=",", skiprows=1)
            V, J = d[:, 0], d[:, 1]
            m = (V >= 0.01) & (J > 0)
            ax.loglog(V[m], J[m], "-", color=colour, linewidth=1.6, zorder=2)
            ax.loglog(V[m], J[m], marker, color=colour, mfc="none",
                      markersize=7, linestyle="none",
                      label="$\\Delta$ %g eV" % delta, zorder=3)
            print("Delta=%.2f eV: J(0.01V)=%.3e  J(1V)=%.3e A/m^2"
                  % (delta, J[m][0], J[m][-1]))
            continue

        out_dir = "out_fig4_%s" % str(delta).replace(".", "p")
        deck = deck_for(delta, out_dir)
        r = subprocess.run([OLED_SIM, deck], cwd=BACKEND_DIR,
                           capture_output=True, text=True)
        os.remove(deck)
        if r.returncode != 0:
            print(r.stdout[-800:]); sys.exit("oled_sim failed for Delta=%s" % delta)

        # ohmic contact at the image-force-lowered barrier, if available
        # (run_fig4_ohmic.py); otherwise the in-deck thermionic result
        oh = os.path.join(SCRIPT_DIR, "fig4_ohmic_d%s.csv"
                          % ("%.2f" % delta).replace(".", "p"))
        if os.path.exists(oh):
            d = np.loadtxt(oh, delimiter=",", skiprows=1)
            V, J = d[:, 0], d[:, 1]
        else:
            V, J = bulk_jv(out_dir)
        m = (V >= 0.01) & (J > 0)
        ax.loglog(V[m], J[m], "-", color=colour, linewidth=1.6, zorder=2)
        # markers on a log-spaced subset, as in the paper
        idx = np.unique(np.searchsorted(V[m], np.logspace(-2, 0, 22)))
        idx = idx[idx < m.sum()]
        if m.sum() < 40:
            idx = np.arange(m.sum())
        ax.loglog(V[m][idx], J[m][idx], marker, color=colour, mfc="none",
                  markersize=7, linestyle="none",
                  label="$\\Delta$ %g eV" % delta, zorder=3)
        print("Delta=%.2f eV: J(0.01V)=%.3e  J(0.1V)=%.3e  J(1V)=%.3e A/m^2"
              % (delta, J[m][0], J[m][np.argmin(abs(V[m] - 0.1))], J[m][-1]))

    ax.set_xlabel("voltage [V]", fontsize=12)
    ax.set_ylabel("current density J [A/m$^2$]", fontsize=12)
    ax.set_xlim(1e-2, 1e0)
    ax.set_ylim(1e-13, 1e1)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    ax.set_title("Knapp Fig. 4 - single-carrier EGDM device\n"
                 "$L$=22nm, $N_0$=2.44e26 m$^{-3}$, "
                 "$\\sigma/kT$=6, $\\mu_0$=1.1e-16 m$^2$/Vs\n"
                 "thick pale = digitised paper, thin+markers = this work\n"
                 "ohmic contacts at image-force-lowered barriers",
                 fontsize=10.5)
    fig.tight_layout()
    out_png = os.path.join(SCRIPT_DIR, "fig4_reproduction.png")
    fig.savefig(out_png, dpi=150)
    print("plot written to", out_png)


if __name__ == "__main__":
    main()
