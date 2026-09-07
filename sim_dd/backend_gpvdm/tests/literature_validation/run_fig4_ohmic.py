#!/usr/bin/env python3
# run_fig4_ohmic.py - Knapp Fig. 4 with OHMIC contacts at image-force-lowered
# barriers.
#
# The paper does not say a thermionic (Scott-Malliaras) injection model was
# used - only that image-force barrier lowering was taken into account.  So
# the quoted Delta is treated as the INTRINSIC barrier, the Schottky lowering
#
#     dphi = sqrt(q^3 F / (4 pi eps))        [J], /q for eV
#
# is computed from the local field, and the contact is then an ohmic one
# pinned to the Gaussian DOS occupancy at the LOWERED barrier:
#
#     np = Nc * Int g(E) f(E, EF = -(Delta - dphi)) dE
#
# dphi is field- and therefore bias-dependent (0.015 eV at 0.01V rising to
# 0.148 eV at 1V across 22nm), while a deck carries a single fixed np.  Each
# bias point is therefore run as its own single-point simulation with np
# evaluated at that point's average field V/L, and the J-V is stitched from
# the results.  Both contacts carry the same barrier, so V_bi = 0 as the
# figure requires.
#
# Usage:  python3 run_fig4_ohmic.py

import math
import os
import subprocess
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OLED_SIM = os.path.join(BACKEND_DIR, "oled_sim")
BASE = os.path.join(SCRIPT_DIR, "oled_fig4.ini")

Qe = 1.602176634e-19
EPS0 = 8.8541878128e-12
KB = 1.380649e-23
EPSR = float(os.environ.get('FIG4_EPSR','3.0'))
L = 22e-9
T = 300.0
KT = KB * T / Qe
SIGMA = 6.0 * KT
NC = 2.44e26

DELTAS = [0.0, 0.33, 0.67, 1.00]
VOLTS = np.logspace(-2, 0, 25)


def gauss_np(barrier):
    """Gaussian DOS occupancy at EF = -barrier (mirrors dos.c)."""
    NE, span = 3000, 12.0
    lo, hi = -span * SIGMA, span * SIGMA
    dE = (hi - lo) / NE
    s = 0.0
    for e in range(NE):
        E = lo + (e + 0.5) * dE
        g = math.exp(-(E * E) / (2 * SIGMA * SIGMA)) / (SIGMA * math.sqrt(2 * math.pi))
        a = (E + barrier) / KT
        if a > 40.0:
            f = math.exp(-a)
        elif a < -40.0:
            f = 1.0
        else:
            f = 1.0 / (1.0 + math.exp(a))
        s += g * f * dE
    return NC * s


def lowering(F):
    """Schottky image-force barrier lowering [eV]."""
    return math.sqrt(Qe ** 3 * F / (4.0 * math.pi * EPSR * EPS0)) / Qe


def run_point(V, npval, out_dir):
    txt = open(BASE).read()
    head = txt[:txt.index("[contact_anode]")]
    head = head.replace("output_dir = output_fig4", "output_dir = " + out_dir)
    # single bias point: sweep straight to V in one step
    head = head.replace("voltage_stop = 1.0", "voltage_stop = %.6g" % V)
    head = head.replace("voltage_step = 0.002", "voltage_step = %.6g" % (V / 40.0))
    blk = ("model = ohmic-np\nmajority = hole\nnp = %.6e\n" % npval)
    deck = (head + "[contact_anode]\n" + blk + "\n[contact_cathode]\n" + blk +
            "\n[outcoupling]\nenabled = 0\n")
    path = os.path.join(SCRIPT_DIR, "_f4o.ini")
    open(path, "w").write(deck)
    r = subprocess.run([OLED_SIM, path], cwd=BACKEND_DIR,
                       capture_output=True, text=True)
    os.remove(path)
    if r.returncode != 0:
        return float("nan")
    # bulk current: median interior Jp of the final block (see plot_fig4.py)
    prof = os.path.join(BACKEND_DIR, out_dir, "device_profiles.dat")
    jp, cur = [], []
    for line in open(prof):
        if line.startswith("# V ="):
            if cur:
                jp = cur
            cur = []
        elif line.strip() and not line.startswith("#"):
            c = line.split()
            if len(c) >= 12:
                cur.append(float(c[11]))
    if cur:
        jp = cur
    if not jp:
        return float("nan")
    n = len(jp)
    return float(np.median(jp[n // 5: 4 * n // 5]))


def main():
    out = {}
    for D in DELTAS:
        V_ok, J_ok = [], []
        for V in VOLTS:
            F = V / L
            beff = max(D - lowering(F), 0.0)
            npv = gauss_np(beff)
            J = run_point(V, npv, "out_f4o")
            if J == J and J > 0:
                V_ok.append(V); J_ok.append(J)
        out[D] = (np.array(V_ok), np.array(J_ok))
        print("Delta=%.2f: %2d/%d points, J(1V)=%.3e"
              % (D, len(V_ok), len(VOLTS), J_ok[-1] if J_ok else float("nan")))
        np.savetxt(os.path.join(SCRIPT_DIR, "fig4_ohmic_d%s.csv"
                                % ("%.2f" % D).replace(".", "p")),
                   np.c_[out[D][0], out[D][1]], delimiter=",",
                   header="V,J", comments="")


if __name__ == "__main__":
    main()
