"""
reporting.py

Scoring a trained problem: the comparison against a reference solution, the
terminal current and its constancy, and the injection balance at any
thermionic contact. All output is in physical units, converted back through
the scaling dict.
"""

import numpy as np
import torch

from .boundaries import contact_fields, injection_density
from .scaling import Q


def relative_L1(pred, true):
    """Relative L1 error, as used throughout the DDNet paper (Supp. Eq. 3)."""
    return np.sum(np.abs(pred - true)) / np.sum(np.abs(true))


def evaluate(problem, *, x_hat, phi_true_V, n_true, p_true, bias=None,
             interior=slice(1, -1)):
    """Compare the trained networks against a reference solution.

    x_hat : scaled evaluation coordinates, shape (N,).
    phi_true_V, n_true, p_true : reference potential (V) and densities
        (cm^-3) at those coordinates.
    interior : slice excluding contact nodes, where minority-carrier pins may
        be discontinuities the networks are deliberately not fit to.
    """
    scaling = problem.scaling
    Ut = scaling["Ut"]
    c_tilde = scaling["c_tilde"]

    x_eval_t = torch.as_tensor(
        x_hat.reshape(-1, 1), dtype=problem.dtype, device=problem.device)
    # No autodiff needed: only the network values are read here.
    with torch.no_grad():
        phi_pred_hat = problem.phi_net(x_eval_t).cpu().numpy().flatten()
        n_pred_hat = problem.n_net.density(x_eval_t).cpu().numpy().flatten()
        p_pred_hat = problem.p_net.density(x_eval_t).cpu().numpy().flatten()

    phi_pred_V = phi_pred_hat.astype(np.float64) * Ut
    n_pred = n_pred_hat.astype(np.float64) * c_tilde
    p_pred = p_pred_hat.astype(np.float64) * c_tilde

    rel_phi = relative_L1(phi_pred_V, phi_true_V)
    # Densities scored in log space, the variable the networks actually learn.
    rel_n_log = relative_L1(np.log10(n_pred[interior]), np.log10(n_true[interior]))
    rel_p_log = relative_L1(np.log10(p_pred[interior]), np.log10(p_true[interior]))
    rel_n_lin = relative_L1(n_pred[interior], n_true[interior])
    rel_p_lin = relative_L1(p_pred[interior], p_true[interior])

    max_abs_phi = np.max(np.abs(phi_pred_V - phi_true_V))

    print()
    print("=" * 70)
    header = "Comparison against reference" if bias is None else \
        "Comparison against reference at {0:.2f} V".format(bias)
    print(header)
    print("=" * 70)
    print("  phi  relative L1        : {0:.4e}  ({1:.3f} %)".format(rel_phi, rel_phi * 100))
    print("  phi  max abs error      : {0:.4e} V".format(max_abs_phi))
    print("  n    relative L1 (log10): {0:.4e}  ({1:.3f} %)".format(rel_n_log, rel_n_log * 100))
    print("  p    relative L1 (log10): {0:.4e}  ({1:.3f} %)".format(rel_p_log, rel_p_log * 100))
    print("  n    relative L1 (lin)  : {0:.4e}  ({1:.3f} %)".format(rel_n_lin, rel_n_lin * 100))
    print("  p    relative L1 (lin)  : {0:.4e}  ({1:.3f} %)".format(rel_p_lin, rel_p_lin * 100))
    print("  (densities scored on interior nodes only; contact pins excluded)")

    return {
        "phi_pred": phi_pred_V, "phi_true": phi_true_V,
        "n_pred": n_pred, "n_true": n_true,
        "p_pred": p_pred, "p_true": p_true,
        "rel_phi": rel_phi, "rel_n_log": rel_n_log, "rel_p_log": rel_p_log,
    }


def report_currents(problem, *, x_hat, x_cm=None, ref_potential_V=None,
                    ref_electrons=None, ref_holes=None, interior=slice(1, -1)):
    """Report the predicted terminal current and how constant it is.

    Jn + Jp should be independent of x, so its spread across the device is a
    self-consistency check independent of any reference comparison. If x_cm
    and the reference profiles are supplied, the reference current is
    finite-differenced for comparison.
    """
    x_t = torch.as_tensor(
        x_hat.reshape(-1, 1), dtype=problem.dtype, device=problem.device
    ).requires_grad_(True)

    _, dphi, _, n_hat, p_hat, dn, dp = problem.fields(x_t)
    Jn, Jp = problem.scaled_currents(dphi, n_hat, p_hat, dn, dp)

    j_scale = problem.scaling["j_scale"]
    # detach() before .numpy(): fields() built a graph via create_graph=True,
    # and these values are only being reported.
    Jn_np = Jn.detach().cpu().numpy().flatten() * j_scale
    Jp_np = Jp.detach().cpu().numpy().flatten() * j_scale
    Jtot = Jn_np + Jp_np

    print()
    print("Current density (A/cm^2):")
    print("  PINN   Jn+Jp : mean {0:+.4e}, spread {1:.3e} ({2:.2f} % of mean)".format(
        Jtot[interior].mean(), np.ptp(Jtot[interior]),
        100 * np.ptp(Jtot[interior]) / abs(Jtot[interior].mean())))

    if x_cm is not None and ref_potential_V is not None \
            and ref_electrons is not None and ref_holes is not None:
        mu_n = problem.scaling["mu_n"]
        mu_p = problem.scaling["mu_p"]
        Ut = problem.scaling["Ut"]
        dphi_ref = np.gradient(ref_potential_V, x_cm)
        Jn_ref = Q * mu_n * (Ut * np.gradient(ref_electrons, x_cm) - ref_electrons * dphi_ref)
        Jp_ref = -Q * mu_p * (Ut * np.gradient(ref_holes, x_cm) + ref_holes * dphi_ref)
        print("  Reference Jn+Jp : mean {0:+.4e}  (finite-differenced from stored profiles)".format(
            (Jn_ref + Jp_ref)[interior].mean()))

    return Jn_np, Jp_np, Jtot


def report_thermionic(problem):
    """Report the injection balance at each thermionic contact.

    Prints, per contact and per carrier, the bulk drift-diffusion current
    arriving at the contact, the current the electrode injects, and their
    mismatch -- the flux balance the training residual imposes, read back in
    physical units, so it is a self-consistency check independent of any
    reference. Also prints the reduced field f and n_inj, the density the
    contact drives towards.
    """
    contacts = [(name, c) for name, c in
                (("left", problem.thermionic_left),
                 ("right", problem.thermionic_right))
                if c is not None]
    if not contacts:
        return {}

    s = problem.scaling
    j_scale = s["j_scale"]
    c_tilde = s["c_tilde"]
    Ut = s["Ut"]

    print()
    print("Thermionic contacts (injection balance, A/cm^2):")

    out = {}
    for name, contact in contacts:
        Jn, Jp, Jn_inj, Jp_inj, f, n_hat, p_hat = contact_fields(
            contact, phi_net=problem.phi_net, n_net=problem.n_net,
            p_net=problem.p_net, scaling=s, device=problem.device,
            dtype=problem.dtype)
        # n_inj recomputed for reporting only; injection_current folds it into
        # the difference it returns.
        n_inj = injection_density(f, dos_hat=s["nc_hat"],
                                  barrier_eV=contact.phi_n, Ut=Ut)
        p_inj = injection_density(f, dos_hat=s["nv_hat"],
                                  barrier_eV=contact.phi_p, Ut=Ut)

        # detach(): contact_fields builds a graph, and these are read values.
        val = lambda t: float(t.detach().cpu().reshape(-1)[0])  # noqa: E731
        rec = {
            "f": val(f),
            "Jn_bulk": val(Jn) * j_scale,
            "Jp_bulk": val(Jp) * j_scale,
            "Jn_inj": contact.n_sign * val(Jn_inj) * j_scale,
            "Jp_inj": contact.p_sign * val(Jp_inj) * j_scale,
            "n_contact": val(n_hat) * c_tilde,
            "p_contact": val(p_hat) * c_tilde,
            "n_inj": val(n_inj) * c_tilde,
            "p_inj": val(p_inj) * c_tilde,
        }
        out[name] = rec

        print("  {0} contact (x = {1:.1f}, phi_n = {2:.3f} eV, phi_p = {3:.3f} eV), "
              "f = {4:.4e}".format(name, contact.x, contact.phi_n,
                                   contact.phi_p, rec["f"]))
        for carrier in ("n", "p"):
            bulk = rec["J{0}_bulk".format(carrier)]
            inj = rec["J{0}_inj".format(carrier)]
            denom = max(abs(bulk), abs(inj))
            mismatch = abs(bulk - inj) / denom if denom > 0 else 0.0
            print("    J{0}: bulk {1:+.4e} | injected {2:+.4e} | mismatch {3:.2f} %".format(
                carrier, bulk, inj, 100 * mismatch))
        print("    n({0}) = {1:.4e} cm^-3 (contact drives towards {2:.4e})".format(
            contact.x, rec["n_contact"], rec["n_inj"]))
        print("    p({0}) = {1:.4e} cm^-3 (contact drives towards {2:.4e})".format(
            contact.x, rec["p_contact"], rec["p_inj"]))

    return out


# ============================================================
# Plotting
# ============================================================

def plot(res, history_epochs, history_losses, x_nm, bias, filename):
    """Plot potential, both carrier densities, and the training loss.

    ``res`` is the dict returned by ``evaluate``.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Potential: PINN against reference.
    ax = axes[0, 0]
    ax.plot(x_nm, res["phi_true"], "-", lw=3, alpha=0.45, color="tab:blue",
            label="reference (drift-diffusion)")
    ax.plot(x_nm, res["phi_pred"], "--", lw=1.8, color="tab:red",
            label="PINN ($\\phi$-Net)")
    ax.set_xlabel(r"$x$ [nm]")
    ax.set_ylabel(r"$\phi$ [V]")
    ax.set_title("(a) Electrostatic potential at {0:.1f} V "
                 "(rel. $L_1$ = {1:.2f}%)".format(bias, res["rel_phi"] * 100))
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Pointwise difference in phi.
    ax = axes[0, 1]
    ax.plot(x_nm, res["phi_pred"] - res["phi_true"], color="tab:purple", lw=1.3)
    ax.set_xlabel(r"$x$ [nm]")
    ax.set_ylabel(r"$\phi_{\rm PINN} - \phi_{\rm ref}$ [V]")
    ax.set_title("(b) Pointwise difference")
    ax.grid(alpha=0.3)

    # (c) Carrier densities, log axis. The reference curves exclude the
    # contact nodes: the minority pins there are discontinuities the networks
    # are not asked to fit.
    ax = axes[1, 0]
    ax.semilogy(x_nm[1:-1], res["n_true"][1:-1], "-", lw=3, alpha=0.45,
                color="tab:blue", label="n (reference)")
    ax.semilogy(x_nm[1:-1], res["p_true"][1:-1], "-", lw=3, alpha=0.45,
                color="tab:green", label="p (reference)")
    ax.semilogy(x_nm, res["n_pred"], "--", lw=1.6, color="tab:red",
                label="n (PINN, n-Net)")
    ax.semilogy(x_nm, res["p_pred"], "--", lw=1.6, color="tab:orange",
                label="p (PINN, p-Net)")
    ax.set_xlabel(r"$x$ [nm]")
    ax.set_ylabel(r"carrier density [cm$^{-3}$]")
    ax.set_title("(c) Carrier densities (rel. $L_1^{{\\log}}$: "
                 "n {0:.2f}%, p {1:.2f}%)".format(
                     res["rel_n_log"] * 100, res["rel_p_log"] * 100))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # (d) Training loss.
    ax = axes[1, 1]
    ax.semilogy(history_epochs, history_losses, color="tab:orange", lw=1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\mathcal{L}_{\rm tot}$")
    ax.set_title("(d) Training loss")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print("\nSaved plot to {0}".format(filename))
    return fig


def plot_weights(weighting, filename, bias=None):
    """Plot an adaptive rule's weights, targets and gradient spreads.

    Three stacked panels sharing the epoch axis, all log-y because the
    quantities span many decades:

      (a) the live weight lambda_i actually multiplying each loss term;
      (b) the instantaneous target lambda_hat_i = std_max/std_i, before the
          running average -- the gap between (b) and (a) is how hard a weight
          is still being pulled;
      (c) the gradient standard deviation std_i behind the target.

    Terms skipped by the update record NaN, so they simply do not draw and a
    gap in a trace is visibly a gap rather than a stale value.
    """
    import matplotlib.pyplot as plt

    epochs = weighting.history_epochs
    # Only plot terms that actually adapted; a term that never produced a
    # gradient std is all-NaN and would add an empty line and a misleading
    # legend entry.
    active = [t for t in weighting.terms
              if any(v == v for v in weighting.history_std[t])]

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    panels = [
        (weighting.history, r"$\lambda_i$", "(a) Adaptive loss weights"),
        (weighting.history_target, r"$\hat{\lambda}_i$",
         r"(b) Instantaneous target $\hat{\lambda}_i = "
         r"\mathrm{std}_{\max}/\mathrm{std}_i$ (before smoothing)"),
        (weighting.history_std, r"$\mathrm{std}(\nabla_\theta \mathcal{L}_i)$",
         r"(c) Gradient standard deviation"),
    ]

    for ax, (data, ylabel, title) in zip(axes, panels):
        for t in active:
            ax.semilogy(epochs, data[t], lw=1.4, label=t)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")

    axes[0].legend(fontsize=8, ncol=len(active))
    axes[-1].set_xlabel("epoch")

    if bias is not None:
        fig.suptitle("Inverse-Dirichlet weight evolution at "
                     "{0:.1f} V".format(bias))
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()

    fig.savefig(filename, dpi=150)
    print("Saved weight-evolution plot to {0}".format(filename))
    return fig
