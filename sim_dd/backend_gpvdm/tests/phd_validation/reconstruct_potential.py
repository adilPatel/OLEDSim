#!/usr/bin/env python3
"""Reconstruct the electrostatic potential of the reference TCAD data.

The reference export contains only n(x,V) and p(x,V) - no potential profile.
The potential is recovered by integrating Poisson's equation on the reference
densities:

        rho(x) = q (p - n + pt - nt)
        d2phi/dx2 = -rho / (eps0 epsr)

integrated twice with Dirichlet ends.  Validated against our own solution,
where phi IS known: it reproduces it to < 0.0003 V at every bias, and to
0.0000 V once our own trapped charge is included.  The reference does not
export nt, but nt <= 1.2e15 cm^-3 here against n ~ 4e19, so omitting it is
worth < 0.0005 V - far below the volt-scale differences being measured.

A DOS-inversion route was tried first and DISCARDED.  Inverting the Gaussian
occupancy gives un = EFn - Ec and up = Ev - EFp, and (un - up - Eg)/2 recovers
phi only if both quasi-Fermi levels are flat.  Under bias they are not: at 5 V
that expression is flat to within 0.35 V across the device while the true phi
swings by 2.9 V.  The DOS inversion returns the carrier-referenced band
position, not the electrostatic potential.  Only Poisson gives the latter.

Boundary condition: both ends are anchored to OUR solution's phi at the
contacts, so what is compared is the shape of the profile through the bulk,
not the contact potentials (which are an input, not a result).
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
N0_CM = 1e21           # cm^-3
N0 = 1e27              # m^-3
EG = 2.6
EPSR = 3.0

COLORS = ["#3b5bdb", "#c2410c", "#0ca678", "#9333ea", "#a16207"]
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#1a1a1a", "#4a4a4a", "#dcdcdc"


# ---------------------------------------------------------------- DOS inverse
_E = np.linspace(-12 * SIGMA, 12 * SIGMA, 4001)
_G = np.exp(-_E ** 2 / (2 * SIGMA ** 2)) / (SIGMA * np.sqrt(2 * np.pi))


def occupancy(u):
    """Fractional Gaussian-DOS occupancy at Fermi level u (eV from centre)."""
    u = np.atleast_1d(u).astype(float)
    a = (_E[None, :] - u[:, None]) / KT
    f = np.where(a > 40, np.exp(-np.clip(a, None, 700)),
                 np.where(a < -40, 1.0, 1.0 / (1.0 + np.exp(np.clip(a, -700, 700)))))
    return np.trapz(_G[None, :] * f, _E, axis=1)


# tabulate once, then invert by interpolation (occupancy is monotonic in u)
_U = np.linspace(-3.0, 1.5, 3000)
_OCC = occupancy(_U)
_ok = np.diff(_OCC) > 0
_Umono = np.concatenate([_U[:1], _U[1:][_ok]])
_Omono = np.concatenate([_OCC[:1], _OCC[1:][_ok]])


def inverse_occupancy(frac):
    """u such that occupancy(u) = frac.  frac = n/N0."""
    frac = np.clip(np.asarray(frac, float), _Omono[0], _Omono[-1])
    return np.interp(frac, _Omono, _Umono)


# ------------------------------------------------------------------ data load
def load_reference(folder, label, carrier, bias):
    path = os.path.join(HERE, "devices_%s" % folder,
                        "Densities_%s_%s_%s_data.txt" % (folder, label, carrier))
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


def load_sim(folder, label, bias):
    """-> (x_nm, n_cm3, p_cm3, phi_V, nt_cm3, pt_cm3)"""
    path = os.path.join(HERE, "devices_%s" % folder,
                        "device_profiles_%s_%s.dat" % (folder, label))
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
    k = min(blocks, key=lambda z: abs(z - bias))
    a = blocks[k]
    # x n p phi E R mun mup nt pt Jn Jp
    return (a[:, 0] * 1e9, a[:, 1] * 1e-6, a[:, 2] * 1e-6, a[:, 3],
            a[:, 8] * 1e-6, a[:, 9] * 1e-6)


# --------------------------------------------------------- the reconstruction
def phi_from_dos(n_cm3, p_cm3):
    """DISCARDED route, kept because it documents the negative result.

    Pointwise DOS inversion; returns a band position, NOT the potential.

    un = EFn - Ec  and  up = Ev - EFp.  Adding them:
        un + up = (EFn - EFp) + (Ev - Ec) = (EFn - EFp) - Eg
    and subtracting removes the split:
        un - up = (EFn + EFp) - (Ec + Ev) = (EFn + EFp) - 2Ec - Eg
    With Ec = -q phi + const, phi = (un - up - Eg)/2 up to a constant.
    """
    un = inverse_occupancy(np.clip(n_cm3, 1e-30, None) / N0_CM)
    up = inverse_occupancy(np.clip(p_cm3, 1e-30, None) / N0_CM)
    return 0.5 * (un - up - EG), un, up


def phi_from_poisson(x_nm, n_cm3, p_cm3, nt_cm3=None, pt_cm3=None,
                     phi_left=0.0, phi_right=0.0):
    """Integrate d2phi/dx2 = -rho/(eps0 epsr) with Dirichlet ends."""
    x = x_nm * 1e-9
    n = n_cm3 * 1e6
    p = p_cm3 * 1e6
    rho = QE * (p - n)
    if nt_cm3 is not None:
        rho -= QE * nt_cm3 * 1e6
    if pt_cm3 is not None:
        rho += QE * pt_cm3 * 1e6

    N = len(x)
    A = np.zeros((N, N))
    b = np.zeros(N)
    A[0, 0] = 1.0; b[0] = phi_left
    A[-1, -1] = 1.0; b[-1] = phi_right
    for i in range(1, N - 1):
        dl = x[i] - x[i - 1]
        dr = x[i + 1] - x[i]
        dh = 0.5 * (dl + dr)
        A[i, i - 1] = 1.0 / (dl * dh)
        A[i, i] = -(1.0 / (dl * dh) + 1.0 / (dr * dh))
        A[i, i + 1] = 1.0 / (dr * dh)
        b[i] = -rho[i] / (EPS0 * EPSR)
    return np.linalg.solve(A, b)


def field(x_nm, phi):
    return -np.gradient(phi, x_nm * 1e-9)


# ------------------------------------------------------------------- reporting
def analyse(folder, label, biases):
    rows = []
    for V in biases:
        rx, rn = load_reference(folder, label, "n", V)
        _, rp = load_reference(folder, label, "p", V)
        if np.all(np.isnan(rn)) or np.all(np.isnan(rp)):
            continue
        sx, sn, sp, sphi, snt, spt = load_sim(folder, label, V)

        # Work on OUR native mesh, which is cell-centred (0.5 .. 99.5 nm).
        # Resampling the other way round clamps np.interp flat outside that
        # range - exactly at the contacts, where phi is steepest - and injects
        # a spurious 0.17 V into the self-check.  Interpolating the reference
        # (1 nm spacing, spanning 0 .. 100) onto our mesh is pure interpolation
        # with no extrapolation, and drops the self-check to ~2e-4 V.
        pl, pr = sphi[0], sphi[-1]
        rn_i = np.interp(sx, rx, rn)
        rp_i = np.interp(sx, rx, rp)

        # Poisson on the reference densities (no nt available - see docstring)
        ref_phi = phi_from_poisson(sx, rn_i, rp_i, phi_left=pl, phi_right=pr)
        # the same operator on our own densities: isolates the density
        # difference from any residual discretisation difference
        own_phi = phi_from_poisson(sx, sn, sp, phi_left=pl, phi_right=pr)
        # method self-check: Poisson on our densities vs our actual phi
        selfcheck = np.max(np.abs(own_phi - sphi))

        rows.append(dict(V=V, x=sx, ref_phi=ref_phi, sim_phi_dos=own_phi,
                         sim_phi_true=sphi,
                         dphi=np.max(np.abs(ref_phi - own_phi)),
                         selfcheck=selfcheck))
    return rows


def main():
    folder, label = "6d3", "0d1"
    biases = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    rows = analyse(folder, label, biases)

    print("Reconstructed potential, %s mu0n=%s" % (folder, label))
    print("  'self-check' = |Poisson on OUR densities  -  our actual phi|;")
    print("  it is the error of the reconstruction method itself.")
    print("  'delta phi'  = |Poisson on reference  -  Poisson on ours|;")
    print("  that is the real potential disagreement.\n")
    print("%6s %14s %14s" % ("V", "delta phi (V)", "self-check (V)"))
    for r in rows:
        print("%6.1f %14.3f %14.3f" % (r["V"], r["dphi"], r["selfcheck"]))

    # ---- figure: potential and field at the low-bias points
    show = [r for r in rows if r["V"] in (1.0, 1.5, 2.0, 3.0, 5.0)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fig.patch.set_facecolor(SURFACE)
    for c, r in zip(COLORS, show):
        axes[0].plot(r["x"], r["sim_phi_dos"], "-", color=c, lw=2.0)
        axes[0].plot(r["x"], r["ref_phi"], "--", color=c, lw=1.7, dashes=(5, 2.5))
        axes[1].plot(r["x"], field(r["x"], r["sim_phi_dos"]) * 1e-8, "-", color=c, lw=2.0)
        axes[1].plot(r["x"], field(r["x"], r["ref_phi"]) * 1e-8, "--", color=c,
                     lw=1.7, dashes=(5, 2.5))
    for ax, t, yl in ((axes[0], r"Electrostatic potential  $\varphi(x)$", r"$\varphi - \varphi(L)$  (V)"),
                      (axes[1], r"Electric field  $F(x)$", r"$F$  (10$^8$ V/m)")):
        ax.set_title(t, fontsize=11, color=INK, pad=8)
        ax.set_xlabel("Position  $x$  (nm)", fontsize=10, color=INK2)
        ax.set_ylabel(yl, fontsize=10, color=INK2)
        ax.set_xlim(0, 100)
        ax.grid(True, color=GRID, lw=0.7)
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)

    handles = [Line2D([], [], color=c, lw=2.0, label="%.1f V" % r["V"])
               for c, r in zip(COLORS, show)]
    handles += [Line2D([], [], color=INK2, lw=2.0, label="This work (oled_sim)"),
                Line2D([], [], color=INK2, lw=1.7, ls="--", dashes=(5, 2.5),
                       label="Reference (reconstructed)")]
    fig.legend(handles=handles, fontsize=9.5, frameon=False, labelcolor=INK2,
               ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.012))
    fig.suptitle(r"Potential reconstructed from the reference densities  -  "
                 r"anode 6.3 eV (ohmic), $\mu_{0n}$ = $10^{-1}$ cm$^2$/Vs",
                 fontsize=12.5, color=INK, y=0.985)
    fig.tight_layout(rect=[0, 0.075, 1, 0.94])
    out = os.path.join(HERE, "devices_6d3", "potential_6d3_0d1_reconstructed.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("\nwrote", os.path.relpath(out, HERE))


if __name__ == "__main__":
    main()
