"""
scaling.py

Physical constants and the nondimensionalization every residual is written in
(DDNet Supplementary Table 1).

Lengths are scaled by the device length ell, potential by the thermal voltage
Ut, densities by c_tilde and mobilities by mu_tilde = max(mu_n, mu_p), giving
the scaled system

    lambda^2 * phi_hat'' = n_hat - p_hat
    Jn_hat'              =  R_hat
    Jp_hat'              = -R_hat
    Jn_hat               =  mu_n_hat*(n_hat' - n_hat*phi_hat')
    Jp_hat               = -mu_p_hat*(p_hat' + p_hat*phi_hat')

with lambda = L_D/ell and L_D = sqrt(eps*Ut/(q*c_tilde)). Full derivation in
Models_neural.md.
"""

import numpy as np

# Universal constants, matching common_physics.py's SetUniversalParameters.
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

    Returns a dict of derived quantities, consumed by the residuals and by the
    reporting code that converts predictions back to physical units.
    """
    eps_org = eps_r * EPS_0
    Ut = K_B * T / Q  # thermal voltage, V

    # Intrinsic density nie = sqrt(Nc*Nv)*exp(-Eg/2kT), as in
    # common_physics.py's CreateDensityOfStates.
    eg = lumo - homo
    nie = np.sqrt(nc300 * nv300) * np.exp(-eg / (2.0 * Ut))

    # Mobilities scaled by the larger of the two, so one of them is 1.
    mu_tilde = max(mu_n, mu_p)
    mu_n_hat = mu_n / mu_tilde
    mu_p_hat = mu_p / mu_tilde

    lam_D = np.sqrt(eps_org * Ut / (Q * c_tilde))   # Debye length, cm
    lam = lam_D / ell                                # scaled Debye parameter

    nie_hat = nie / c_tilde
    # Langevin prefactor gammar*(q/eps)*(mu_n + mu_p), scaled: densities enter
    # as a product (c_tilde^2) and the whole rate is divided by R_tilde.
    R_tilde = mu_tilde * Ut * c_tilde / (ell ** 2)
    langevin_prefactor = gammar * (Q / eps_org) * (mu_n + mu_p) * (c_tilde ** 2) / R_tilde

    j_scale = Q * mu_tilde * Ut * c_tilde / ell

    # --- Field-dependent thermionic injection constants -------------------
    # Precomputed here, as SetOSParameters does, since none depend on a
    # solution variable. Unused by purely Ohmic devices.

    # Coulomb capture radius r_c = q^2/(4*pi*eps*kT), the distance at which
    # image-charge attraction equals the thermal energy. kT in Joules = q*Ut.
    kT_J = Q * Ut
    r_c = Q * Q / (4.0 * np.pi * eps_org * kT_J)
    # Reduced field f = q*E*r_c/kT. Scaled, E = -(Ut/ell)*phi_hat', so
    # f = (r_c/ell)*|phi_hat'| -- this single factor is the whole conversion.
    r_c_over_ell = r_c / ell

    # Zero-field surface recombination velocity
    # S(0) = 16*pi*eps*mu*(kT)^2/q^3 (cm/s), per carrier. This carries the
    # Richardson constant implicitly: C = 16*pi*eps*mu*N*(kT/q)^2 satisfies
    # C == q*N*S(0) exactly, which is what makes the injection residual vanish
    # at equilibrium.
    s0_common = 16.0 * np.pi * eps_org * kT_J * kT_J / (Q ** 3)
    s0n = s0_common * mu_n
    s0p = s0_common * mu_p
    # S enters the scaled residual as the dimensionless S(0)*ell/(mu_tilde*Ut).
    v_scale = mu_tilde * Ut / ell          # cm/s
    s0n_hat = s0n / v_scale
    s0p_hat = s0p / v_scale

    # Effective DOS in the density networks' own units, so n_inj comes out
    # directly comparable to their output.
    nc_hat = nc300 / c_tilde
    nv_hat = nv300 / c_tilde

    return {
        # Bulk scaling, read by the residuals.
        "eps_org": eps_org, "Ut": Ut, "eg": eg,
        "nie": nie, "nie_hat": nie_hat,
        "mu_tilde": mu_tilde, "mu_n_hat": mu_n_hat, "mu_p_hat": mu_p_hat,
        "lam_D": lam_D, "lam": lam,
        "langevin_prefactor": langevin_prefactor,
        # Unit conversions, read by reporting.
        "j_scale": j_scale, "c_tilde": c_tilde, "ell": ell,
        "mu_n": mu_n, "mu_p": mu_p,
        # Thermionic injection, read by boundaries.py.
        "r_c": r_c, "r_c_over_ell": r_c_over_ell,
        "s0n": s0n, "s0p": s0p, "s0n_hat": s0n_hat, "s0p_hat": s0p_hat,
        "nc_hat": nc_hat, "nv_hat": nv_hat, "nc300": nc300, "nv300": nv300,
    }
