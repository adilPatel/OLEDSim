"""
oled2_poisson_only.py

Diagnostic: solve Poisson ALONE, with the charge density taken from the
DEVSIM reference rather than from the density networks.

This removes every coupling in the problem. There is no n-Net, no p-Net, no
continuity equation, no recombination, no current constancy and no contact
model -- just phi-Net against

    lambda^2 * phi_hat'' - (n_hat(x) - p_hat(x)) = 0

where n_hat(x) and p_hat(x) are the reference profiles interpolated to the
collocation points, plus the two Dirichlet potential pins.

What it isolates
----------------
When the coupled solve produces a wrong potential, two explanations remain:

  (a) phi-Net cannot represent this potential. The problem is stiff
      (lambda^2 = 5.0e-07, L/L_D = 1413), so the residual is dominated by the
      source term and a near-discontinuity may satisfy it cheaply at finitely
      many collocation points. If so, this run fails too.

  (b) phi-Net is fine, and the wrong potential is driven by wrong densities
      fed back through the coupling. If so, this run succeeds -- and the fault
      is on the density / continuity side, not in Poisson.

Handing phi-Net the exact reference charge is the strongest possible version
of the problem: if it cannot solve that, no amount of loss weighting on the
coupled system will help. Since it takes the reference solution as an input it
is a diagnostic, not a validation -- it cannot demonstrate that the solver
works, only locate the fault.

Why the residual is normalised
------------------------------
Under the absolute residual (NORMALIZE=False) the fault is *conditioning*, not
a misplaced objective minimum. The residual's natural magnitude spans five
orders of magnitude across the device:

    bulk (x < 90 nm)     : |n-p| ~ 1e-05 .. 1e-03
    cathode (x > 99 nm)  : |n-p| ~ 1e-01 .. 1e+00

Under uniform sampling essentially all collocation points land in the flat
bulk, where any smooth potential scores well, so the mean-squared loss is set
by whichever handful land in the last nanometre. That leaves too little
separation between the true solution and a trivial flat-phi collapse to pull
the optimizer to the physical answer.

Debye rescaling (X = x/lam_D) does NOT help: substituting gives
d2phi/dX2 = lam^2 * d2phi/dxhat2, algebraically the identical residual. lam^2
is not an artifact of the non-dimensionalisation; it is the physical fact that
the device is 1413 Debye lengths long.

NORMALIZE=True divides the residual pointwise by the local source magnitude,
so every collocation point contributes a relative error and the bulk stops
being free.

Usage
-----
    python -m sim_pinn.devices.oled2_poisson_only [epochs]
    POISSON_NORMALIZE=0 python -m sim_pinn.devices.oled2_poisson_only   # original formulation
"""

import os
import sys

import numpy as np
import torch

from sim_pinn import core
from sim_pinn.devices import oled2_forward as base


EPOCHS = 20000
MILESTONES = [5000, 10000, 14000, 17000]
GAMMA = 0.3
LR = 1e-3
N_INT = 4096
PRINT_EVERY = 2000

# Residual formulation. False reproduces the original absolute residual;
# True divides by the local source magnitude (see docstring).
NORMALIZE = os.environ.get("POISSON_NORMALIZE", "1") == "1"

# Floor on the normalising scale. The source |n_hat - p_hat| never falls below
# ~1.5e-05 in the interior of this device, so this floor is inactive here; it
# exists to keep the division safe if the source passes through zero (which it
# would in a device with a compensated region).
RESIDUAL_FLOOR = 1.0e-12

# With a normalised residual both terms are O(1), so the boundary term needs
# no relative weight; under the absolute residual the interior term is many
# decades smaller and the pins have to be weighted up to compete.
BC_WEIGHT = base.LOSS_WEIGHTS["bc"] if not NORMALIZE else 1.0

PNG = os.path.join(
    base._TEST_DIR,
    "oled2_poisson_only_normalized.png" if NORMALIZE else "oled2_poisson_only.png")


def reference_charge_interpolator():
    """Interpolate the reference charge density onto arbitrary x_hat.

    Returns a callable x_hat -> (n_hat, p_hat) as torch tensors. Linear
    interpolation is done in LOG space, since both densities span many
    decades and linear-space interpolation would be dominated by the largest
    values.
    """
    x_ref = base.REF_X_HAT.astype(np.float64)
    log_n = np.log(base.REF_N_HAT.astype(np.float64))
    log_p = np.log(base.REF_P_HAT.astype(np.float64))

    # Reference arrays copied onto the accelerator once, so the per-epoch
    # interpolation below never touches host memory.
    x_t = torch.as_tensor(x_ref, dtype=base.dtype, device=base.device)
    ln_t = torch.as_tensor(log_n, dtype=base.dtype, device=base.device)
    lp_t = torch.as_tensor(log_p, dtype=base.dtype, device=base.device)

    def interp(x_query):
        xq = x_query.reshape(-1)
        # searchsorted finds, for each query point, the index where it would
        # slot into the sorted reference grid -- the right-hand bracket.
        # .contiguous() lays the query out in memory as searchsorted requires;
        # .clamp keeps the index in range so idx-1 and idx are always valid.
        idx = torch.searchsorted(x_t, xq.contiguous()).clamp(1, len(x_t) - 1)
        x0, x1 = x_t[idx - 1], x_t[idx]
        w = ((xq - x0) / (x1 - x0)).clamp(0.0, 1.0)
        ln = ln_t[idx - 1] * (1 - w) + ln_t[idx] * w
        lp = lp_t[idx - 1] * (1 - w) + lp_t[idx] * w
        # Back out of log space, reshaped to the (N, 1) column the residual
        # expects.
        return torch.exp(ln).reshape(-1, 1), torch.exp(lp).reshape(-1, 1)

    return interp


def main():
    """Train phi-Net alone against the reference charge, then score and plot.

    Takes an optional epoch count as the first command-line argument.
    """
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else EPOCHS

    # Fix both RNGs so the weight init and collocation draw are reproducible.
    torch.manual_seed(0)
    np.random.seed(0)

    print()
    print("#" * 70)
    print("# Poisson-only: phi-Net vs reference charge density")
    print("#" * 70)

    lam = base.SCALING["lam"]
    print("lambda     = {0:.4e}   (lambda^2 = {1:.4e})".format(lam, lam ** 2))
    print("phi pins   : phi_hat(0) = {0:+.4f}, phi_hat(1) = {1:+.4f}".format(
        base.phi_bc_left, base.phi_bc_right))
    print("charge     : n_hat, p_hat interpolated from the DEVSIM reference")
    print("epochs     = {0}".format(epochs))
    print("residual   : {0}".format(
        "NORMALISED  r = (lam^2*phi'' - (n-p)) / |n-p|" if NORMALIZE
        else "absolute    r = lam^2*phi'' - (n-p)"))
    print("bc weight  = {0:g}".format(BC_WEIGHT))

    phi_net = core.PhiNet(width=base.WIDTH, depth=base.DEPTH).to(base.device)
    interp = reference_charge_interpolator()

    # The two contact points and their target values, built once on-device.
    x_bc = torch.tensor([[base.X_LEFT], [base.X_RIGHT]],
                        device=base.device, dtype=base.dtype)
    phi_bc = torch.tensor([[base.phi_bc_left], [base.phi_bc_right]],
                          device=base.device, dtype=base.dtype)

    # .parameters() yields the module's trainable tensors; Adam updates them
    # from the .grad that backward() fills in.
    opt = torch.optim.Adam(phi_net.parameters(), lr=LR)
    # MultiStepLR scales the learning rate by gamma at each milestone epoch.
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=MILESTONES, gamma=GAMMA)

    hist_e, hist_l = [], []

    print()
    print("Training phi-Net alone...")
    for epoch in range(epochs):
        # backward() accumulates into .grad, so last epoch's gradients must be
        # cleared; set_to_none frees them outright rather than zero-filling.
        opt.zero_grad(set_to_none=True)

        # Uniform collocation draw, straight onto the device.
        # requires_grad_(True) marks x for differentiation, so the two
        # autograd.grad calls below have a recorded graph to walk.
        x = torch.rand(N_INT, 1, device=base.device,
                       dtype=base.dtype).requires_grad_(True)
        # detach() here: the charge is fixed reference data, so the
        # interpolation must not become a path autograd differentiates.
        n_hat, p_hat = interp(x.detach())

        # phi'' by differentiating twice. grad_outputs=ones seeds the
        # vector-Jacobian product so each collocation point gets its own
        # derivative; create_graph=True keeps the first derivative itself
        # differentiable, which is what makes the second call possible.
        phi = phi_net(x)
        dphi = torch.autograd.grad(phi, x, torch.ones_like(phi),
                                   create_graph=True)[0]
        d2phi = torch.autograd.grad(dphi, x, torch.ones_like(dphi),
                                    create_graph=True)[0]

        # lambda^2*phi'' - (n - p), the residual this script exists to test.
        r = lam ** 2 * d2phi - (n_hat - p_hat)
        if NORMALIZE:
            # Pointwise normalisation: divide by the local magnitude of the
            # terms being balanced, so each collocation point contributes a
            # *relative* error rather than an absolute one. See the module
            # docstring for why this is the operative fix.
            # clamp(min=) is an elementwise floor keeping the divisor off
            # zero. No detach() needed, unlike core.poisson_residual: here the
            # charge is reference data and carries no gradient anyway.
            scale = torch.clamp(torch.abs(n_hat - p_hat), min=RESIDUAL_FLOOR)
            r = r / scale
        L_poisson = torch.mean(r ** 2)
        L_bc = torch.mean((phi_net(x_bc) - phi_bc) ** 2)
        loss = L_poisson + BC_WEIGHT * L_bc

        # backward() writes d(loss)/d(param) into each .grad and frees the
        # graph; step() applies the update, then the LR schedule advances.
        loss.backward()
        opt.step()
        sched.step()

        if epoch % 20 == 0:
            hist_e.append(epoch)
            # .item() pulls one value to the CPU, forcing the device to finish
            # its queued work -- so it is sampled, not read every epoch.
            hist_l.append(loss.item())
        if epoch % PRINT_EVERY == 0 or epoch == epochs - 1:
            print("  epoch {0:6d} | total {1:.3e} | poisson {2:.3e} | "
                  "bc {3:.3e} | lr {4:.1e}".format(
                      epoch, loss.item(), L_poisson.item(), L_bc.item(),
                      sched.get_last_lr()[0]))

    # --- evaluate against the reference potential ---
    x_eval = torch.as_tensor(base.REF_X_HAT.reshape(-1, 1),
                             dtype=base.dtype, device=base.device)
    # no_grad(): nothing here is differentiated, so skip graph recording.
    # cpu() moves the result to host memory and numpy() views it as an array,
    # which is only legal off-graph and on the CPU.
    with torch.no_grad():
        phi_pred_V = phi_net(x_eval).cpu().numpy().flatten() * base.Ut

    rel_phi = core.relative_L1(phi_pred_V, base.REF_POTENTIAL)
    max_abs = np.max(np.abs(phi_pred_V - base.REF_POTENTIAL))

    print()
    print("=" * 70)
    print("Poisson-only vs DEVSIM reference potential")
    print("=" * 70)
    print("  phi relative L1   : {0:.4e}  ({1:.3f} %)".format(rel_phi, rel_phi * 100))
    print("  phi max abs error : {0:.4e} V".format(max_abs))

    # Plot: potential, error, charge density used, loss curve.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    ax[0, 0].plot(base.REF_X_NM, base.REF_POTENTIAL, lw=3, alpha=.6,
                  label="reference (drift-diffusion)")
    ax[0, 0].plot(base.REF_X_NM, phi_pred_V, "r--", lw=2,
                  label=r"Poisson-only ($\phi$-Net)")
    ax[0, 0].set_xlabel("x [nm]"); ax[0, 0].set_ylabel(r"$\phi$ [V]")
    ax[0, 0].set_title(r"(a) Poisson-only potential (rel. $L_1$ = {0:.2f}%)".format(
        rel_phi * 100))
    ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)

    ax[0, 1].plot(base.REF_X_NM, phi_pred_V - base.REF_POTENTIAL, color="purple")
    ax[0, 1].set_xlabel("x [nm]")
    ax[0, 1].set_ylabel(r"$\phi_{\rm PINN} - \phi_{\rm ref}$ [V]")
    ax[0, 1].set_title("(b) Pointwise difference"); ax[0, 1].grid(alpha=.3)

    ax[1, 0].semilogy(base.REF_X_NM, base.REF_ELECTRONS, lw=3, alpha=.6, label="n (reference)")
    ax[1, 0].semilogy(base.REF_X_NM, base.REF_HOLES, lw=3, alpha=.6, label="p (reference)")
    ax[1, 0].set_xlabel("x [nm]"); ax[1, 0].set_ylabel(r"density [cm$^{-3}$]")
    ax[1, 0].set_title("(c) Charge density supplied to Poisson (exact)")
    ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)

    ax[1, 1].semilogy(hist_e, hist_l, color="darkorange")
    ax[1, 1].set_xlabel("epoch"); ax[1, 1].set_ylabel(r"$\mathcal{L}$")
    ax[1, 1].set_title("(d) Training loss"); ax[1, 1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(PNG, dpi=150)
    print("Saved plot to {0}".format(PNG))


if __name__ == "__main__":
    main()
