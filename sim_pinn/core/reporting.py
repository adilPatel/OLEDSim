"""
reporting.py

Scoring a trained problem: comparison against a reference solution, the
terminal current and its constancy, the injection balance at any thermionic
contact, and the figures for each. All output is in physical units, converted
back through the scaling dict.
"""

import numpy as np
import torch

from .boundaries import contact_fields, injection_density
from .scaling import Q


def relative_L1(pred, true):
    """Relative L1 error, as used throughout the DDNet paper (Supp. Eq. 3)."""
    return np.sum(np.abs(pred - true)) / np.sum(np.abs(true))


def evaluate(problem, *, x_hat, phi_true_V, n_true, p_true, bias=None,
             x_cm=None, interior=slice(1, -1)):
    """Compare the trained networks against a reference solution.

    x_hat : scaled evaluation coordinates, shape (N,).
    phi_true_V, n_true, p_true : reference potential (V) and densities (cm^-3)
        at those coordinates.
    bias : applied bias in volts, for the printed header only.
    x_cm : evaluation coordinates in cm. Given, the reference currents are
        finite-differenced from the reference profiles so Jn(x)/Jp(x) can be
        plotted against the prediction.
    interior : slice excluding contact nodes, where minority-carrier pins may
        be discontinuities the networks are deliberately not fit to.

    Prints the error table and returns the predicted and reference profiles
    together with the three headline errors.
    """
    scaling = problem.scaling
    Ut = scaling["Ut"]
    c_tilde = scaling["c_tilde"]

    # as_tensor converts the numpy array without copying where it can; the
    # (-1, 1) reshape makes it the column of points the networks expect, and
    # dtype/device are matched to the trained model.
    x_eval_t = torch.as_tensor(
        x_hat.reshape(-1, 1), dtype=problem.dtype, device=problem.device)
    # no_grad() switches off graph recording for everything inside: only
    # network values are read here, no residual is differentiated, so this
    # saves the memory a graph over all evaluation points would take.
    with torch.no_grad():
        # .cpu() brings each result back to host memory, .numpy() views it as
        # an array (legal only off-graph and on the CPU -- both hold here),
        # .flatten() drops the trailing size-1 axis.
        phi_pred_hat = problem.phi_net(x_eval_t).cpu().numpy().flatten()
        n_pred_hat = problem.n_net.density(x_eval_t).cpu().numpy().flatten()
        p_pred_hat = problem.p_net.density(x_eval_t).cpu().numpy().flatten()

    # Undo the scaling: phi by Ut, both densities by c_tilde.
    phi_pred_V = phi_pred_hat.astype(np.float64) * Ut
    n_pred = n_pred_hat.astype(np.float64) * c_tilde
    p_pred = p_pred_hat.astype(np.float64) * c_tilde

    rel_phi = relative_L1(phi_pred_V, phi_true_V)
    # Densities scored in log space, the variable the networks actually learn,
    # and again linearly, which the large values dominate.
    rel_n_log = relative_L1(np.log10(n_pred[interior]), np.log10(n_true[interior]))
    rel_p_log = relative_L1(np.log10(p_pred[interior]), np.log10(p_true[interior]))
    rel_n_lin = relative_L1(n_pred[interior], n_true[interior])
    rel_p_lin = relative_L1(p_pred[interior], p_true[interior])

    max_abs_phi = np.max(np.abs(phi_pred_V - phi_true_V))

    # Current profiles, for the Jn(x)/Jp(x) panels. These need derivatives, so
    # they are taken outside the no_grad() block above via problem.fields(),
    # which is the same path training uses -- the plotted currents are exactly
    # the ones the residuals are built from, not a finite-difference rederivation.
    x_grad_t = torch.as_tensor(
        x_hat.reshape(-1, 1), dtype=problem.dtype,
        device=problem.device).requires_grad_(True)
    _, dphi_g, _, n_hat_g, p_hat_g, dn_g, dp_g = problem.fields(x_grad_t)
    Jn_t, Jp_t = problem.scaled_currents(
        dphi_g, n_hat_g, p_hat_g, dn_g, dp_g, x_grad_t)
    # detach() before numpy(): fields() built a graph with create_graph=True
    # and these are only being reported.
    j_scale = scaling["j_scale"]
    Jn_pred = Jn_t.detach().cpu().numpy().flatten().astype(np.float64) * j_scale
    Jp_pred = Jp_t.detach().cpu().numpy().flatten().astype(np.float64) * j_scale

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

    out = {
        # Profiles, prediction and reference side by side, for plot().
        "phi_pred": phi_pred_V, "phi_true": phi_true_V,
        "n_pred": n_pred, "n_true": n_true,
        "p_pred": p_pred, "p_true": p_true,
        "Jn_pred": Jn_pred, "Jp_pred": Jp_pred,
        # Headline errors, also used in the figure titles.
        "rel_phi": rel_phi, "rel_n_log": rel_n_log, "rel_p_log": rel_p_log,
    }

    # Reference currents, finite-differenced from the stored profiles on the
    # same grid, so the plotted comparison is like for like.
    if x_cm is not None:
        mu_n, mu_p = scaling["mu_n"], scaling["mu_p"]
        dphi_ref = np.gradient(phi_true_V, x_cm)
        out["Jn_true"] = Q * mu_n * (
            Ut * np.gradient(n_true, x_cm) - n_true * dphi_ref)
        out["Jp_true"] = -Q * mu_p * (
            Ut * np.gradient(p_true, x_cm) + p_true * dphi_ref)
    return out


def report_currents(problem, *, x_hat, x_cm=None, ref_potential_V=None,
                    ref_electrons=None, ref_holes=None, interior=slice(1, -1)):
    """Report the predicted terminal current and how constant it is.

    x_hat : scaled evaluation coordinates.
    x_cm, ref_potential_V, ref_electrons, ref_holes : optional reference
        profiles; given all four, the reference current is finite-differenced
        for comparison.
    interior : slice excluding contact nodes.

    Jn + Jp must be independent of x, so its spread across the device is a
    self-consistency check needing no reference. Returns (Jn, Jp, Jtot) in
    A/cm^2.
    """
    # requires_grad_(True), and no no_grad() block: unlike evaluate() above,
    # the currents contain derivatives, so the graph is needed here.
    x_t = torch.as_tensor(
        x_hat.reshape(-1, 1), dtype=problem.dtype, device=problem.device
    ).requires_grad_(True)

    _, dphi, _, n_hat, p_hat, dn, dp = problem.fields(x_t)
    Jn, Jp = problem.scaled_currents(dphi, n_hat, p_hat, dn, dp, x_t)

    j_scale = problem.scaling["j_scale"]
    # detach() before .numpy(): fields() built a graph via create_graph=True,
    # and numpy() refuses a tensor that still carries one. These values are
    # only being reported, so dropping the graph loses nothing.
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
        # Same drift-diffusion expressions, unscaled, on the stored profiles.
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

    Prints, per contact and per carrier, the bulk current arriving at the
    contact, the current the electrode injects and their mismatch -- the flux
    balance the training residual imposes, read back in physical units, so it
    is a self-consistency check needing no reference. Also prints the reduced
    field f and n_inj, the density the contact drives towards.

    Returns a dict of contact name -> those quantities; empty for a purely
    Ohmic device.
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

        # Each of these is a 1-element tensor still attached to a graph:
        # detach() drops the graph, cpu() moves it to host memory, and
        # reshape(-1)[0] takes the single value out as a float.
        val = lambda t: float(t.detach().cpu().reshape(-1)[0])  # noqa: E731
        rec = {
            "f": val(f),
            # Currents in A/cm^2. The injection terms carry the contact's sign
            # so they are directly comparable with the bulk ones.
            "Jn_bulk": val(Jn) * j_scale,
            "Jp_bulk": val(Jp) * j_scale,
            "Jn_inj": contact.n_sign * val(Jn_inj) * j_scale,
            "Jp_inj": contact.p_sign * val(Jp_inj) * j_scale,
            # Densities at the contact, and the values it drives them towards.
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
            # Mismatch relative to the larger of the two, so it stays finite
            # when either side is near zero.
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

# Vertical range of the carrier-density panel, cm^-3. The contact pins sit
# ~30 decades below the bulk, so the axis is clipped to the band the device
# physics occupies rather than autoscaled to the pins.
DENSITY_PLOT_YLIM = (1.0e10, 1.0e15)


def plot(res, history_epochs, history_losses, x_nm, bias, filename):
    """Six-panel summary: potential, its error, densities, loss, Jn(x), Jp(x).

    res : the dict returned by ``evaluate``.
    history_epochs, history_losses : the loss curve from ``train``.
    x_nm : evaluation coordinates in nm, for the x-axis.
    bias : applied bias in volts, for the titles.
    filename : path to write the PNG to.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 13.5))

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

    # (b) Pointwise difference in phi, which (a) is too coarse to show.
    ax = axes[0, 1]
    ax.plot(x_nm, res["phi_pred"] - res["phi_true"], color="tab:purple", lw=1.3)
    ax.set_xlabel(r"$x$ [nm]")
    ax.set_ylabel(r"$\phi_{\rm PINN} - \phi_{\rm ref}$ [V]")
    ax.set_title("(b) Pointwise difference")
    ax.grid(alpha=0.3)

    # (c) Carrier densities, log axis. The reference curves drop the contact
    # nodes, where the minority pins are discontinuities the networks are not
    # asked to fit; the predictions are drawn across the full domain.
    ax = axes[1, 0]
    ax.semilogy(x_nm[1:-1], res["n_true"][1:-1], "-", lw=3, alpha=0.45,
                color="tab:blue", label="n (reference)")
    ax.semilogy(x_nm[1:-1], res["p_true"][1:-1], "-", lw=3, alpha=0.45,
                color="tab:green", label="p (reference)")
    ax.semilogy(x_nm, res["n_pred"], "--", lw=1.6, color="tab:red",
                label="n (PINN, n-Net)")
    ax.semilogy(x_nm, res["p_pred"], "--", lw=1.6, color="tab:orange",
                label="p (PINN, p-Net)")
    # Clip the view to the decades the physics lives in. The minority pins
    # run to ~1e-16 and the predictions can undershoot further still; on a
    # full-range axis that compresses the 1e10-1e15 band the profiles and
    # their boundary layers actually occupy into a few pixels.
    ax.set_ylim(DENSITY_PLOT_YLIM)
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

    # (e), (f) Current profiles. Linear y: the point is whether each current is
    # FLAT and at the right level, which a log axis would hide. In 1D steady
    # state Jn + Jp is independent of x, so any slope here is a residual error.
    for ax, key, name, colour in ((axes[2, 0], "Jn", "J_n", "tab:red"),
                                  (axes[2, 1], "Jp", "J_p", "tab:orange")):
        pred = res.get(key + "_pred")
        true = res.get(key + "_true")
        if pred is None:
            ax.set_visible(False)
            continue
        if true is not None:
            # Contact nodes dropped: the reference current is finite-differenced
            # across the minority-pin discontinuity there and is meaningless.
            ax.plot(x_nm[1:-1], true[1:-1], "-", lw=3, alpha=0.45,
                    color="tab:blue", label="reference")
        ax.plot(x_nm, pred, "--", lw=1.6, color=colour, label="PINN")
        ax.set_xlabel(r"$x$ [nm]")
        ax.set_ylabel(r"$" + name + r"$ [A cm$^{-2}$]")
        # Spread of the prediction over the interior, the same self-consistency
        # measure report_currents() prints.
        interior = pred[1:-1]
        spread = (100 * np.ptp(interior) / abs(np.mean(interior))
                  if np.mean(interior) else float("nan"))
        ax.set_title("({0}) ${1}(x)$  (PINN mean {2:+.3e}, spread {3:.2f} %)".format(
            "e" if key == "Jn" else "f", name, np.mean(interior), spread))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print("\nSaved plot to {0}".format(filename))
    return fig


def plot_weights(weighting, filename, bias=None, clip_decades=6.0):
    """Plot each loss term's relative loss and relative weight, one panel each.

    weighting : a LossWeights that recorded history (needs history_loss);
        returns None if it did not.
    filename : path to write the PNG to.
    bias : applied bias in volts, used in the title. Omitted if None.
    clip_decades : half-height of the y-axis, in decades either side of 1.0.
        Values outside are drawn but do not rescale the view; None autoscales.

    Both curves are divided by their own first recorded value, so they start
    at 1.0 and share a dimensionless axis: each panel then shows which way a
    term's weight moved as its loss fell or stalled.
    """
    import matplotlib.pyplot as plt

    epochs = weighting.history_epochs
    hist_loss = getattr(weighting, "history_loss", None)
    if hist_loss is None:                      # rule records no losses
        return None

    # Skip terms that never recorded a value (all-NaN); they would draw an
    # empty panel. `v == v` is False only for NaN.
    active = [t for t in weighting.terms
              if any(v == v for v in hist_loss[t])]
    if not active:
        return None

    def _relative(series):
        """Series divided by its first finite non-zero entry.

        series : list of floats, possibly starting with NaN or 0.0 for a term
            inactive early on. Returns all-NaN if there is no usable base.
        """
        base = next((v for v in series if v == v and v != 0.0), None)
        if base is None:
            return [float("nan")] * len(series)
        return [v / base for v in series]

    n = len(active)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.8 * n + 1.2), sharex=True)
    if n == 1:
        axes = [axes]                          # keep the zip below uniform

    for ax, t in zip(axes, active):
        ax.semilogy(epochs, _relative(hist_loss[t]), lw=1.4, color="tab:blue",
                    label=r"relative loss $\mathcal{L}_i/\mathcal{L}_i(0)$")
        ax.semilogy(epochs, _relative(weighting.history[t]), lw=1.4,
                    color="tab:red",
                    label=r"relative weight $\lambda_i/\lambda_i(0)$")
        ax.axhline(1.0, color="0.5", lw=0.8, ls=":")   # the common starting point
        if clip_decades is not None:
            ax.set_ylim(10.0 ** -clip_decades, 10.0 ** clip_decades)
        ax.set_ylabel(t, fontsize=11)
        ax.grid(alpha=0.3, which="both")

    axes[0].legend(fontsize=9, ncol=2, loc="upper right")
    axes[-1].set_xlabel("epoch")

    name = getattr(weighting, "scheme_name", "Adaptive")
    if bias is not None:
        fig.suptitle("{0}: loss and weight per term at {1:.1f} V".format(
            name, bias))
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    else:
        fig.tight_layout()

    fig.savefig(filename, dpi=150)
    print("Saved weight-evolution plot to {0}".format(filename))
    return fig


def plot_iv(iv_npz, summaries, filename, v_min=None, built_in_voltage=None):
    """Semilog J-V curve: DEVSIM reference line, PINN terminal currents as points.

    iv_npz : path to the DEVSIM sweep archive, holding ``voltages_full`` and
        ``current_A_full`` (A/cm^2).
    summaries : the per-bias dicts from ``run_bias``, each with "bias" and
        "j_pinn" (A/cm^2). One marker is drawn per entry.
    filename : path to write the PNG to.
    v_min : left edge of the x-axis. Defaults to 0.1 V below the lowest
        recorded bias, keeping the view on the turn-on region and above rather
        than the many decades of sub-threshold leakage.
    built_in_voltage : if given, marked with a vertical line as the turn-on.

    Currents are plotted in mA/cm^2, the usual unit for a device J-V.
    """
    import matplotlib.pyplot as plt

    data = np.load(iv_npz)
    V = data["voltages_full"]
    J = data["current_A_full"] * 1e3            # A/cm^2 -> mA/cm^2

    v_pinn = np.array([s["bias"] for s in summaries])
    j_pinn = np.array([s["j_pinn"] for s in summaries]) * 1e3

    if v_min is None:
        v_min = (v_pinn.min() if len(v_pinn) else V.min()) - 0.1

    # Restrict to the window being plotted before taking limits, so the
    # sub-threshold decades do not set the y-scale.
    win = V >= v_min
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.semilogy(V[win], J[win], "-", lw=2.5, alpha=0.7, color="tab:blue",
                label="DEVSIM reference")
    ax.semilogy(v_pinn, j_pinn, "o", ms=7, mfc="none", mew=1.8,
                color="tab:red", label="PINN (SoftAdapt)")

    if built_in_voltage is not None and built_in_voltage >= v_min:
        ax.axvline(built_in_voltage, color="0.5", lw=1.0, ls=":")
        ax.text(built_in_voltage, ax.get_ylim()[0], r" $V_{\rm bi}$",
                va="bottom", ha="left", fontsize=9, color="0.4")

    ax.set_xlim(v_min, max(V[win].max(), v_pinn.max() + 0.1))
    ax.set_xlabel(r"applied bias $V$ [V]")
    ax.set_ylabel(r"$J_{\rm tot}$ [mA cm$^{-2}$]")
    ax.set_title("Terminal current: PINN against DEVSIM")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print("Saved J-V plot to {0}".format(filename))
    return fig
