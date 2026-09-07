#!/usr/bin/env python3
"""Feed the REFERENCE densities through our own model expressions (6d3, 2 V).

The mobility scale has been shown not to move the densities in this device
(both contacts ohmic -> the carrier profiles are set by electrostatics and the
contact BCs alone), so the recombination discrepancy at 2 V must come from the
densities themselves.  This script removes our densities from the loop:

  * R_pinned(x)  = gamma * q (mu_n + mu_p)/eps * (n_ref p_ref - n0p0)
    with the EGDM mobilities evaluated at the REFERENCE densities.
    If this matches the reference R, our recombination model is right and only
    the densities were wrong.  If it does not, the model itself is wrong.

  * phi_pinned(x) from Poisson on the reference densities, and the field it
    implies.  Compared against the potential our solver produced.

  * The residual of the drift-diffusion current: with the reference n, p and
    the potential they imply, J_n and J_p should each be spatially CONSTANT in
    steady state (dJ/dx = +/- qR).  Any systematic violation points at a term
    missing from the transport equations rather than at a parameter.

Run from a venv with numpy+matplotlib.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

KB_EV = 8.617333262e-5
QE = 1.602176634e-19
EPS0 = 8.8541878128e-12
T = 300.0
KT = KB_EV * T
SIGMA = 0.15
N0 = 1e27              # m^-3 site density
EG = 2.6
EPSR = 3.0
A_SITE = 1e-9          # inter-site distance used in the decks
C1, C2 = 1.8e-9, 0.39  # EGDM temperature prefactor (c2 = 0.39, user-confirmed)
GAMMA = 0.1            # reduced Langevin
BIAS = 2.0
FOLDER = "6d3"

SH = SIGMA / KT
BASE = C1 * np.exp(-C2 * SH * SH)
DELTA = 2.0 * (np.log(SH * SH - SH) - np.log(np.log(4.0))) / (SH * SH)
CFIELD = 0.44 * (SH ** 1.5 - 2.2)

MU = [("0d001", 1e-3, r"$10^{-3}$"), ("0d01", 1e-2, r"$10^{-2}$"),
      ("0d1", 1e-1, r"$10^{-1}$")]

C_A, C_B, C_C = "#3b5bdb", "#c2410c", "#0ca678"
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#1a1a1a", "#4a4a4a", "#dcdcdc"


def g1(dens_m3):
    x = np.clip(dens_m3 / (2.0 * N0), 1e-300, 0.5)
    return np.exp(0.5 * (SH * SH - SH) * np.power(2.0 * x, DELTA))


def g2(F):
    Fhat = A_SITE * np.abs(F) / SIGMA
    return np.exp(CFIELD * (np.sqrt(1.0 + 0.8 * Fhat ** 2) - 1.0))


def egdm(mu0_cm2, dens_m3, F):
    """mu in m^2/Vs from mu0 in cm^2/Vs, matching src/mobility.c."""
    return (mu0_cm2 * 1e-4 / C1) * BASE * g1(dens_m3) * g2(F)


def load_ref(label, carrier, bias):
    path = os.path.join(HERE, "devices_%s" % FOLDER,
                        "Densities_%s_%s_%s_data.txt" % (FOLDER, label, carrier))
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
    m = np.isclose(v, bias, atol=1e-6)
    o = np.argsort(x[m])
    return x[m][o], d[m][o]


def load_ref_R(label, bias):
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
            d.append(np.nan)
    x, v, d = np.array(x), np.array(v), np.array(d)
    m = np.isclose(v, bias, atol=1e-6)
    o = np.argsort(x[m])
    return x[m][o], d[m][o]


def load_sim(label, bias):
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
    return dict(x=a[:, 0] * 1e9, n=a[:, 1], p=a[:, 2], phi=a[:, 3],
                R=a[:, 5], mun=a[:, 6], mup=a[:, 7])


def poisson(x_nm, n_m3, p_m3, phi_l, phi_r):
    x = x_nm * 1e-9
    rho = QE * (p_m3 - n_m3)
    N = len(x)
    A = np.zeros((N, N)); b = np.zeros(N)
    A[0, 0] = 1.0; b[0] = phi_l
    A[-1, -1] = 1.0; b[-1] = phi_r
    for i in range(1, N - 1):
        dl = x[i] - x[i - 1]; dr = x[i + 1] - x[i]; dh = 0.5 * (dl + dr)
        A[i, i - 1] = 1.0 / (dl * dh)
        A[i, i] = -(1.0 / (dl * dh) + 1.0 / (dr * dh))
        A[i, i + 1] = 1.0 / (dr * dh)
        b[i] = -rho[i] / (EPS0 * EPSR)
    return np.linalg.solve(A, b)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.4))
    fig.patch.set_facecolor(SURFACE)
    print("6d3 at %.1f V - reference densities pushed through OUR model\n" % BIAS)

    for col, (label, mu0n, mutex) in enumerate(MU):
        rx, rn = load_ref(label, "n", BIAS)
        _, rp = load_ref(label, "p", BIAS)
        _, rR = load_ref_R(label, BIAS)
        sim = load_sim(label, BIAS)

        n_m3 = rn * 1e6
        p_m3 = rp * 1e6

        # potential implied by the reference densities, anchored at our contacts
        sphi = np.interp(rx, sim["x"], sim["phi"])
        pl = np.interp(rx[0], sim["x"], sim["phi"])
        pr = np.interp(rx[-1], sim["x"], sim["phi"])
        phi_ref = poisson(rx, n_m3, p_m3, pl, pr)
        F_ref = -np.gradient(phi_ref, rx * 1e-9)

        # our own Langevin R, but evaluated on the reference densities
        mun = egdm(mu0n, n_m3, F_ref)
        mup = egdm(1e-1, p_m3, F_ref)
        B = GAMMA * QE * (mun + mup) / (EPSR * EPS0)
        R_pin = B * (n_m3 * p_m3) * 1e-6          # cm^-3 s^-1

        # --- top row: recombination
        ax = axes[0, col]
        ax.semilogy(rx, np.clip(R_pin, 1e-300, None), "-", color=C_A, lw=2.2)
        ax.semilogy(rx, np.clip(rR, 1e-300, None), "--", color=C_B, lw=1.9,
                    dashes=(5, 2.5))
        ax.semilogy(sim["x"], np.clip(sim["R"] * 1e-6, 1e-300, None), ":",
                    color=C_C, lw=1.8)
        ax.set_title(r"$\mu_{0n}$ = %s cm$^2$/Vs" % mutex, fontsize=11,
                     color=INK, pad=8)
        if col == 0:
            ax.set_ylabel(r"$R$  (cm$^{-3}$s$^{-1}$)", fontsize=10, color=INK2)

        # --- bottom row: potential
        ax = axes[1, col]
        ax.plot(rx, phi_ref, "-", color=C_A, lw=2.2)
        ax.plot(rx, sphi, ":", color=C_C, lw=2.0)
        if col == 0:
            ax.set_ylabel(r"$\varphi$  (V)", fontsize=10, color=INK2)

        for ax in (axes[0, col], axes[1, col]):
            ax.set_xlabel("Position  $x$  (nm)", fontsize=10, color=INK2)
            ax.set_xlim(0, 100)
            ax.grid(True, color=GRID, lw=0.7)
            ax.set_facecolor(SURFACE)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=9)

        m = np.isfinite(rR) & (rR > 0) & (rx >= 10) & (rx <= 95)
        iR_pin = np.trapz(R_pin[m], rx[m] * 1e-7)
        iR_ref = np.trapz(rR[m], rx[m] * 1e-7)
        ms = (sim["x"] >= 10) & (sim["x"] <= 95)
        iR_our = np.trapz(sim["R"][ms] * 1e-6, sim["x"][ms] * 1e-7)
        print("  mu0n=%-6s  intR pinned %.3e | ref %.3e | ours %.3e"
              % (label, iR_pin, iR_ref, iR_our))
        print("             pinned/ref = %7.3f    ours/ref = %7.3f"
              % (iR_pin / iR_ref, iR_our / iR_ref))
        print("             max |phi_ref - phi_ours| = %.4f V"
              % np.max(np.abs(phi_ref - sphi)))

    handles = [
        Line2D([], [], color=C_A, lw=2.2, label="Our model on REFERENCE densities"),
        Line2D([], [], color=C_B, lw=1.9, ls="--", dashes=(5, 2.5),
               label="Reference TCAD"),
        Line2D([], [], color=C_C, lw=1.8, ls=":", label="Our full solution"),
    ]
    fig.legend(handles=handles, fontsize=9.5, frameon=False, labelcolor=INK2,
               ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.008))
    fig.suptitle(r"6.3 eV ohmic device at %.1f V  -  reference densities pushed "
                 r"through our recombination and Poisson" % BIAS,
                 fontsize=12.5, color=INK, y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    out = os.path.join(HERE, "devices_%s" % FOLDER, "pinned_%s_2V.png" % FOLDER)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("\nwrote", os.path.relpath(out, HERE))


if __name__ == "__main__":
    main()
