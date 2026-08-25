"""
scaling.py

Physical constants and the nondimensionalization that turns a device's
physical parameters into the scaled quantities every residual is written in
(DDNet Supplementary Table 1).

Scaled system solved by the networks:

    lambda^2 * phi_hat'' = n_hat - p_hat
    Jn_hat'              =  R_hat
    Jp_hat'              = -R_hat
    Jn_hat               =  mu_n_hat*(n_hat' - n_hat*phi_hat')
    Jp_hat               = -mu_p_hat*(p_hat' + p_hat*phi_hat')

Lengths are scaled by the device length ell, potential by the thermal voltage
Ut, densities by c_tilde and mobilities by mu_tilde = max(mu_n, mu_p), giving
lambda = L_D/ell with L_D = sqrt(eps*Ut/(q*c_tilde)). See Models_neural.md for
the derivation.
"""

import numpy as np

# Physical constants, matching sim_dd/devsim_backend/common_physics.py
# (SetUniversalParameters).
Q     = 1.6e-19            # C
K_B   = 1.3806503e-23      # J/K
EPS_0 = 8.85e-14           # F/cm


def compute_scaling(*, eps_r, T, mu_n, mu_p, homo, lumo, nc300, nv300,
                    gammar, c_tilde, ell):
    """Nondimensionalize the device parameters.

    Parameters
    ----------
    eps_r : relative permittivity of the organic layer.
    T : temperature, K.
    mu_n, mu_p : electron/hole mobilities, cm^2/V-s.
    homo, lumo : HOMO/LUMO levels, eV.
    nc300, nv300 : conduction-/valence-band effective density of states, cm^-3.
    gammar : Langevin recombination prefactor.
    c_tilde : density scale, cm^-3 (typically the largest contact density).
    ell : device length, cm.

    Returns a dict of derived quantities, consumed both by the residuals and
    by the reporting code that converts predictions back to physical units.
    """
    eps_org = eps_r * EPS_0
    Ut = K_B * T / Q  # thermal voltage, V

    eg = lumo - homo
    nie = np.sqrt(nc300 * nv300) * np.exp(-eg / (2.0 * Ut))

    mu_tilde = max(mu_n, mu_p)
    mu_n_hat = mu_n / mu_tilde
    mu_p_hat = mu_p / mu_tilde

    lam_D = np.sqrt(eps_org * Ut / (Q * c_tilde))   # Debye length, cm
    lam = lam_D / ell                                # scaled Debye parameter

    nie_hat = nie / c_tilde
    R_tilde = mu_tilde * Ut * c_tilde / (ell ** 2)
    langevin_prefactor = gammar * (Q / eps_org) * (mu_n + mu_p) * (c_tilde ** 2) / R_tilde

    j_scale = Q * mu_tilde * Ut * c_tilde / ell

    # --- Field-dependent thermionic injection constants -------------------
    # Mirrors SetOSParameters in sim_dd/devsim_backend/os_physics.py, which
    # precomputes these for the same reason: none depend on a solution
    # variable. Unused by purely Ohmic devices.
    #
    # Coulomb capture radius r_c = q^2/(4*pi*eps*kT): the distance at which
    # the image-charge attraction equals the thermal energy. kT is in Joules,
    # so kT = q*Ut.
    kT_J = Q * Ut
    r_c = Q * Q / (4.0 * np.pi * eps_org * kT_J)
    # The reduced field is f = q*E*r_c/kT = E*(r_c/Ut). Scaled, the field is
    # E = -(Ut/ell)*phi_hat', so f = (r_c/ell)*|phi_hat'| -- this one number
    # is the whole conversion from the scaled potential gradient to f.
    r_c_over_ell = r_c / ell

    # Zero-field surface recombination velocity S(0) = 16*pi*eps*mu*(kT)^2/q^3
    # (cm/s), per carrier. This implicitly carries the Richardson constant:
    # the prefactor C = 16*pi*eps*mu*N*(kT/q)^2 satisfies C == q*N*S(0)
    # exactly, which is what makes the injection residual vanish at
    # equilibrium.
    s0_common = 16.0 * np.pi * eps_org * kT_J * kT_J / (Q ** 3)
    s0n = s0_common * mu_n
    s0p = s0_common * mu_p
    # J_inj = q*S(0)*(...)*c_tilde against j_scale = q*mu_tilde*Ut*c_tilde/ell,
    # so S(0) enters the scaled residual as the dimensionless velocity
    # S(0)*ell/(mu_tilde*Ut).
    v_scale = mu_tilde * Ut / ell          # cm/s
    s0n_hat = s0n / v_scale
    s0p_hat = s0p / v_scale

    # Effective DOS scaled by c_tilde, so n_inj comes out in the units the
    # density networks work in.
    nc_hat = nc300 / c_tilde
    nv_hat = nv300 / c_tilde

    return {
        "eps_org": eps_org,
        "Ut": Ut,
        "eg": eg,
        "nie": nie,
        "nie_hat": nie_hat,
        "mu_tilde": mu_tilde,
        "mu_n_hat": mu_n_hat,
        "mu_p_hat": mu_p_hat,
        "lam_D": lam_D,
        "lam": lam,
        "langevin_prefactor": langevin_prefactor,
        "j_scale": j_scale,
        "c_tilde": c_tilde,
        "ell": ell,
        "mu_n": mu_n,
        "mu_p": mu_p,
        # Thermionic injection.
        "r_c": r_c,
        "r_c_over_ell": r_c_over_ell,
        "s0n": s0n,
        "s0p": s0p,
        "s0n_hat": s0n_hat,
        "s0p_hat": s0p_hat,
        "nc_hat": nc_hat,
        "nv_hat": nv_hat,
        "nc300": nc300,
        "nv300": nv300,
    }
