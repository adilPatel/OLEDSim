"""
oled2_forward_ohmic.py

Diagnostic variant of oled2_forward.py: the thermionic anode is replaced by an
**Ohmic** anode whose hole density is pinned to the DEVSIM reference value at
x = 0. Written to isolate the terms of the coupled solve.

Why
---
The thermionic flux balance is orders of magnitude smaller than the other loss
terms at weight 1.0 (see LOSS_WEIGHTS in oled2_forward.py), so it can be
driven to the trivial zero-flux solution rather than to a genuine balance,
leaving the device with no working anode.

Removing the thermionic term entirely -- pinning p(0) to the number DEVSIM
reports instead of deriving it from a barrier -- separates that from the rest:
can phi-Net and the density nets solve this device *at all* when the anode is
handed to them exactly right? If this variant still misses the reference, the
problem is in the coupled Poisson/continuity solve (the stiff source near the
contacts), not in the contact model. If it succeeds, the thermionic weighting
is the whole story.

Everything else -- device parameters, scaling, architecture, schedule -- is
imported unchanged from oled2_forward.py so the only variable is the anode
treatment and the Poisson weight.

Boundary conditions here
------------------------
  x = 0 (anode, Ohmic)   : phi pinned; p(0) pinned to the reference
                           (holes are the majority carrier at the anode);
                           n(0) left free (minority, discontinuous).
  x = L (cathode, Ohmic) : phi pinned; n(L) pinned to the reference
                           (electrons majority); p(L) left free.

Both contact potentials are pinned to the reference values, identical to the
thermionic run -- phi is Dirichlet at a contact regardless of contact type.

Usage
-----
    python -m sim_pinn.devices.oled2_forward_ohmic [poisson_weight]

Defaults to sweeping 10 and 100 if no weight is given.
"""

import os
import sys

import numpy as np
import torch

from sim_pinn import core

# Import the baseline configuration verbatim. Importing oled2_forward runs its
# module-level setup (scaling, reference load, prints) but NOT training --
# main() is guarded by __name__ == "__main__" there, so this is safe.
from sim_pinn.devices import oled2_forward as base


def build(poisson_weight, seed=0):
    """Build a fresh Ohmic-anode problem with the given Poisson weight.

    Networks are re-initialised per run from the same seed so the only
    difference between sweep points is the weight.
    """
    # Re-seed per build so a sweep varies only in the weights: torch's RNG
    # drives the Xavier init and the collocation draw.
    torch.manual_seed(seed)
    np.random.seed(seed)

    weights = dict(base.LOSS_WEIGHTS)
    weights["poisson"] = float(poisson_weight)

    phi_net, n_net, p_net = core.build_networks(
        width=base.WIDTH, depth=base.DEPTH,
        u_n_offset=base._u_n_init, u_p_offset=base._u_p_init,
        device=base.device,
    )

    # Reference contact densities, scaled. p at the anode, n at the cathode --
    # the majority carrier at each Ohmic contact.
    p_bc_left = float(base.REF_P_HAT[0])
    n_bc_right = float(base.REF_N_HAT[-1])

    problem = core.build_problem(
        phi_net=phi_net, n_net=n_net, p_net=p_net,
        scaling=base.SCALING, device=base.device, dtype=base.dtype,
        x_left=base.X_LEFT, x_right=base.X_RIGHT,
        # Identical contact potentials to the thermionic run.
        phi_bc_left=base.phi_bc_left, phi_bc_right=base.phi_bc_right,
        majority_only_bc=True,
        # Anode is now Ohmic: no ThermionicContact, hole density pinned.
        thermionic_left=None,
        u_p_bc_left=-np.log(p_bc_left),
        # Cathode unchanged from the baseline.
        u_n_bc_right=-np.log(n_bc_right),
        loss_weights=weights,
        beta_concentration=base.BETA_CONCENTRATION,
    )
    return problem, p_bc_left


def run(poisson_weight):
    """Train and score one Poisson weight; returns evaluate()'s dict."""
    print()
    print("#" * 70)
    print("# Ohmic anode, Poisson weight = {0:g}".format(poisson_weight))
    print("#" * 70)

    problem, p_bc_left = build(poisson_weight)

    print("Anode (x=0, Ohmic): phi_hat = {0:+.4f}, p(0) = {1:.4e} cm^-3 pinned "
          "(n(0) free)".format(base.phi_bc_left, p_bc_left * base.C_TILDE))
    print("Cathode (x=L, Ohmic): phi_hat = {0:+.4f}, n(L) = {1:.4e} cm^-3 pinned "
          "(p(L) free)".format(base.phi_bc_right, base.REF_ELECTRONS[-1]))
    print("Loss weights: {0}".format(
        problem.loss_weights.format_weights()))

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
        "oled2_forward_ohmic_poisson{0:g}.png".format(poisson_weight))
    try:
        core.plot(res, history_epochs, history_losses,
                   base.REF_X_NM, base.REF_BIAS, png)
    except ImportError:
        pass

    return res


def main():
    """Sweep the Poisson weight over the command-line arguments."""
    weights = [float(w) for w in sys.argv[1:]] or [10.0, 100.0]
    summary = []
    for w in weights:
        res = run(w)
        summary.append((w, res))

    print()
    print("=" * 70)
    print("Ohmic-anode Poisson weight sweep")
    print("=" * 70)
    print("  {0:>10}  {1:>12}  {2:>12}  {3:>12}".format(
        "w_poisson", "phi relL1", "n relL1(log)", "p relL1(log)"))
    for w, res in summary:
        print("  {0:>10g}  {1:>11.3f}%  {2:>11.3f}%  {3:>11.3f}%".format(
            w, res["rel_phi"] * 100,
            res["rel_n_log"] * 100, res["rel_p_log"] * 100))


if __name__ == "__main__":
    main()
