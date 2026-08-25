"""
oled2_forward_ohmic4.py

Diagnostic: Ohmic anode with BOTH carrier densities pinned at x = 0.

Why n(0) is pinnable
--------------------
Electrons are the *majority* carrier at both contacts in this device:

    n(0) = 1.5154e+16 cm^-3      p(0) = 1.8230e+11 cm^-3      n/p = 8.3e+04

and n(0) is smooth into the bulk (n(1)/n(0) = 1.063), so it is representable
by a smooth network despite being the anode value. Leaving it free would leave
the dominant charge at that contact unconstrained.

The only genuine discontinuity is p(L) at the cathode: p(L-1)/p(L) = 1.07e+23,
holes falling to 2.1e-23 cm^-3 right at the contact. That one stays free.

So this configuration pins three of the four density boundary values:

    n(0) = 1.5154e+16   pinned (majority, smooth)
    p(0) = 1.8230e+11   pinned (minority but smooth)
    n(L) = 1.0000e+21   pinned (majority)
    p(L)                FREE   (discontinuous -- unrepresentable)

Implemented with majority_only_bc=False, which enables the minority pins, and
u_p_bc_right=None to suppress the p(L) one specifically.

Everything else is imported unchanged from oled2_forward.py.

Usage
-----
    python -m sim_pinn.devices.oled2_forward_ohmic4 [poisson_weight ...]   # default: 10
"""

import os
import sys

import numpy as np
import torch

from sim_pinn import core
from sim_pinn.devices import oled2_forward as base


# Normalise the Poisson residual pointwise by the local source magnitude
# (core.poisson.poisson_residual), so the bulk contributes as much signal as
# the contact layer. Here the normaliser is a network output rather than
# reference data, and is detached inside poisson_residual.
NORMALIZE = os.environ.get("POISSON_NORMALIZE", "1") == "1"

# Impose the contact potentials through phi-Net's architecture rather than as
# a loss penalty (core.PhiNet). Required alongside NORMALIZE: with the
# normalised residual, phi'' = 0 is a degenerate minimum that gradient descent
# cannot escape at any Poisson weight, since the residual is then -1 at every
# collocation point regardless of phi. The ansatz makes a constant phi
# unrepresentable.
HARD_PHI_BC = os.environ.get("HARD_PHI_BC", "1") == "1"

# Electron-continuity weight. Poisson cannot determine phi on its own once the
# densities are free: for any smooth phi there is an n = lam^2*phi'' + p that
# satisfies it exactly, and because lam^2 = 5.0e-07 is tiny, absorbing even a
# 0.28 V error in phi costs n well under 1% of its local value. Continuity is
# the equation that breaks that degeneracy -- it fixes n given phi through
# drift-diffusion -- so its weight is the direct lever on phi accuracy, more so
# than any Poisson weight. Left at the default 1.0 unless overridden.
W_CONT_N = float(os.environ.get("W_CONT_N", "0")) or None


def build(poisson_weight, seed=0, w_cont_n=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    weights = dict(base.LOSS_WEIGHTS)
    weights["poisson"] = float(poisson_weight)
    if w_cont_n is not None:
        weights["cont_n"] = float(w_cont_n)

    phi_net, n_net, p_net = core.build_networks(
        width=base.WIDTH, depth=base.DEPTH,
        u_n_offset=base._u_n_init, u_p_offset=base._u_p_init,
        device=base.device,
        phi_bc=((base.phi_bc_left, base.phi_bc_right) if HARD_PHI_BC else None),
    )

    n_bc_left = float(base.REF_N_HAT[0])
    p_bc_left = float(base.REF_P_HAT[0])
    n_bc_right = float(base.REF_N_HAT[-1])

    problem = core.build_problem(
        phi_net=phi_net, n_net=n_net, p_net=p_net,
        scaling=base.SCALING, device=base.device, dtype=base.dtype,
        x_left=base.X_LEFT, x_right=base.X_RIGHT,
        phi_bc_left=base.phi_bc_left, phi_bc_right=base.phi_bc_right,
        # False enables the minority pins; p(L) is then suppressed by passing
        # u_p_bc_right=None below.
        majority_only_bc=False,
        thermionic_left=None,
        u_p_bc_left=-np.log(p_bc_left),     # anode holes
        u_n_bc_left=-np.log(n_bc_left),     # anode electrons  <-- new
        u_n_bc_right=-np.log(n_bc_right),   # cathode electrons
        u_p_bc_right=None,                  # cathode holes: discontinuous
        loss_weights=weights,
        beta_concentration=base.BETA_CONCENTRATION,
        normalize_poisson=NORMALIZE,
    )
    return problem


def run(poisson_weight, w_cont_n=None):
    print()
    print("#" * 70)
    print("# Ohmic anode, n(0) AND p(0) pinned; w_poisson = {0:g}, "
          "w_cont_n = {1:g}".format(
              poisson_weight,
              w_cont_n if w_cont_n is not None
              else base.LOSS_WEIGHTS["cont_n"]))
    print("#" * 70)

    problem = build(poisson_weight, w_cont_n=w_cont_n)

    print("Pins: n(0) = {0:.4e}, p(0) = {1:.4e}, n(L) = {2:.4e} cm^-3; "
          "p(L) free".format(base.REF_ELECTRONS[0], base.REF_HOLES[0],
                             base.REF_ELECTRONS[-1]))
    print("      phi(0)_hat = {0:+.4f}, phi(L)_hat = {1:+.4f}".format(
        base.phi_bc_left, base.phi_bc_right))
    print("Poisson residual: {0}".format(
        "NORMALISED  r = (lam^2*phi'' - (n-p)) / |n-p|" if NORMALIZE
        else "absolute    r = lam^2*phi'' - (n-p)"))
    print("phi contact BC  : {0}".format(
        "HARD (ansatz, exact by construction)" if HARD_PHI_BC
        else "soft (loss penalty)"))

    history_epochs, history_losses = core.train(
        problem, epochs=base.EPOCHS, n_int=base.N_INT, lr=base.LEARNING_RATE,
        milestones=base.MILESTONES, gamma=base.GAMMA,
        log_every=base.LOG_EVERY, print_every=base.PRINT_EVERY,
        convergence_threshold=base.CONVERGENCE_THRESHOLD,
    )

    res = core.evaluate(
        problem, x_hat=base.REF_X_HAT, phi_true_V=base.REF_POTENTIAL,
        n_true=base.REF_ELECTRONS, p_true=base.REF_HOLES, bias=base.REF_BIAS,
    )
    core.report_currents(
        problem, x_hat=base.REF_X_HAT, x_cm=base.REF_X_NM * 1e-7,
        ref_potential_V=base.REF_POTENTIAL, ref_electrons=base.REF_ELECTRONS,
        ref_holes=base.REF_HOLES,
    )

    png = os.path.join(
        base._TEST_DIR,
        "oled2_forward_ohmic4{0}{1}_poisson{2:g}{3}.png".format(
            "_normalized" if NORMALIZE else "",
            "_hardbc" if HARD_PHI_BC else "", poisson_weight,
            "" if w_cont_n is None else "_contn{0:g}".format(w_cont_n)))
    try:
        core.plot(res, history_epochs, history_losses,
                   base.REF_X_NM, base.REF_BIAS, png)
    except ImportError:
        pass
    return res


def main():
    # Two sweep modes:
    #   python -m sim_pinn.devices.oled2_forward_ohmic4 10 100   -> sweep w_poisson
    #   CONT_N_SWEEP="5 10 50 100" python ...        -> sweep w_cont_n, with
    #                                                   w_poisson fixed (W_POISSON,
    #                                                   default 1: the best found)
    sweep = os.environ.get("CONT_N_SWEEP", "").split()
    if sweep:
        w_poisson = float(os.environ.get("W_POISSON", "1"))
        out = [(w_poisson, w, run(w_poisson, w_cont_n=w))
               for w in (float(s) for s in sweep)]
    else:
        weights = [float(w) for w in sys.argv[1:]] or [10.0]
        out = [(w, W_CONT_N, run(w, w_cont_n=W_CONT_N)) for w in weights]

    print()
    print("=" * 70)
    print("Both-carriers-pinned anode: summary")
    print("=" * 70)
    print("  {0:>10}  {1:>10}  {2:>12}  {3:>12}  {4:>12}".format(
        "w_poisson", "w_cont_n", "phi relL1", "n relL1(log)", "p relL1(log)"))
    for wp, wc, res in out:
        print("  {0:>10g}  {1:>10g}  {2:>11.3f}%  {3:>11.3f}%  {4:>11.3f}%".format(
            wp, wc if wc is not None else base.LOSS_WEIGHTS["cont_n"],
            res["rel_phi"] * 100, res["rel_n_log"] * 100,
            res["rel_p_log"] * 100))


if __name__ == "__main__":
    main()
