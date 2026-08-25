"""
oled1_forward.py

Example: coupled drift-diffusion PINN (DDNet) for OLED1's 100 nm organic
active layer with Ohmic contacts, validated against the DEVSIM
drift-diffusion solver at several applied biases.

Three networks -- phi-Net, n-Net, p-Net -- trained together against the
coupled Poisson + continuity system (DDNet Fig. 1). The reusable model,
loss, scaling and training/eval/plot logic live in ``sim_pinn.core``; this
script supplies OLED1's device parameters, boundary conditions, and run
configuration (loss weights, sampling distribution, epochs, ...).

Device
------
The 100 nm organic layer of ``devsim_reference_oled1.py``: OLED1's contact
set with the bottom contact's electron density reduced from 1e25 to
1e17 cm^-3.

Boundary conditions
--------------------
phi is pinned at both contacts (Dirichlet). The majority carrier density is
pinned at each contact; the minority carrier is left free (see
MAJORITY_ONLY_BC below) since the literal DEVSIM pin there is a
discontinuity a smooth network cannot represent.

Multiple biases
---------------
``devsim_reference_oled1.py`` sweeps 2 - 5 V and writes one reference profile
per bias in BIASES, plus the full IV curve. A separate set of networks is
trained from scratch at each bias -- nothing is transferred between them, so
the biases are independent samples of the same configuration -- and each
writes its own figure. ``main()`` then tabulates the PINN terminal current
against the DEVSIM one at each bias.

Running this script trains at every bias in BIASES; import it and call
``run_bias(V)`` to do just one.

Usage
-----
    python -m sim_pinn.devices.oled1_forward
"""

import os

import numpy as np
import torch

from sim_pinn import core

# Seeds are re-applied per bias in run_bias() so each run starts identically,
# rather than inheriting the RNG state left by the previous one.
SEED = 0

torch.set_default_dtype(torch.float32)

# ============================================================
# Run configuration: device (CPU/GPU)
# ============================================================

USE_GPU = True

if not USE_GPU:
    device = torch.device("cpu")
else:
    if torch.get_default_dtype() == torch.float64:
        raise SystemExit(
            "USE_GPU=True requires float32: the MPS backend has no float64. "
            "Set torch.set_default_dtype(torch.float32) as well, and re-check "
            "the Langevin residual for underflow before trusting the result."
        )
    _accelerator = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else None
    device = torch.device(_accelerator.type) if _accelerator is not None else torch.device("cpu")
dtype = torch.get_default_dtype()
print("Using device:", device)

_HERE = os.path.dirname(os.path.abspath(__file__))
# Repo root: up out of sim_pinn/devices/.
_ROOT = os.path.dirname(os.path.dirname(_HERE))

# DEVSIM references + comparison plots live under tests/diode_pinn_1d/oled1,
# not next to this script (see devsim_reference_oled1.py in that directory).
_TEST_DIR = os.path.join(_ROOT, "tests", "diode_pinn_1d", "oled1")
IV_NPZ = os.path.join(_TEST_DIR, "oled1_devsim_iv.npz")

# Applied biases to train at. Each needs a matching reference file from
# devsim_reference_oled1.py (whose TARGET_BIASES must agree with this list).
BIASES = (3.3, 3.4, 3.5, 3.6, 3.7)


def reference_npz(bias):
    """Path of the DEVSIM reference profile for one bias."""
    return os.path.join(_TEST_DIR, "oled1_devsim_reference_{0:.1f}V.npz".format(bias))


# Figure filenames carry a tag for the configuration that produced them, so a
# run cannot overwrite another configuration's figures: "demo" for the plain
# run, "hdbc" once the hard boundary ansatz is on, "id" once inverse-Dirichlet
# weighting is on top of it.
def plot_tag():
    """Short tag naming the configuration under test."""
    if USE_ADAPTIVE_WEIGHTS:
        return "id"
    if USE_HARD_DENSITY_BC or USE_HARD_PHI_BC:
        return "hdbc"
    return "demo"


def plot_png(bias):
    """Path of the comparison figure for one bias."""
    return os.path.join(_TEST_DIR, "oled1_forward_{0}_{1:.1f}V.png".format(
        plot_tag(), bias))


def weights_png(bias):
    """Path of the adaptive-weight evolution figure for one bias."""
    return os.path.join(_TEST_DIR, "oled1_forward_{0}_weights_{1:.1f}V.png".format(
        plot_tag(), bias))


# ============================================================
# Device / material parameters (OLED1, 100 nm organic layer)
# ============================================================

EPS_R  = 4.0                 # OLED1's relative permittivity (Device default)
TEMP_K = 300.0                # K

# Transport parameters: core.device.Device's defaults (devsim_reference.py's
# make_device() overrides neither mobility nor the densities of states).
MU_N   = 1.0e-6               # electron mobility, cm^2/V-s
MU_P   = 1.0e-6               # hole mobility, cm^2/V-s
NC300  = 1.0e27               # conduction-band effective DOS, cm^-3
NV300  = 1.0e27               # valence-band effective DOS, cm^-3
GAMMAR = 1.0                  # Langevin prefactor ("gammar" in os_physics.py)

# HOMO/LUMO of the reference device (implies the intrinsic density nie via
# core.compute_scaling, following common_physics.py's CreateDensityOfStates
# at T = 300 K: NIE = sqrt(NC*NV) * exp(-EG/(2*Ut))).
LUMO = -0.4
HOMO = LUMO - 2.6             # EG = 2.6 eV

# Density scale: the bottom contact's electron density, the largest carrier
# density in the device. Bias-independent (it is a contact pin), so the same
# scaling serves every run.
C_TILDE = 1.0e17              # cm^-3

ELL = 100.0e-7                # device length, cm (100 nm)

# Domain scaled to x_hat in [0, 1] by the device length (not by the Debye
# length).
X_LEFT, X_RIGHT = 0.0, 1.0

# Which density boundary conditions are actually imposable. At each Ohmic
# contact DEVSIM pins BOTH carrier densities, but only the majority pin is
# representable by a smooth network -- the minority pin is a genuine
# discontinuity in the reference data. So the majority carrier is pinned at
# each contact and the minority carrier is left free, determined by the
# continuity equations.
#
# Left contact (anode): holes are the majority carrier.
# Right contact (cathode): electrons are the majority carrier.
MAJORITY_ONLY_BC = True


# ============================================================
# Network architecture
# ============================================================

WIDTH = 64
DEPTH = 4


# ============================================================
# Loss weights, sampling, training schedule
# ============================================================

# Interior collocation points per step: DDNet's own 2^12 = 4096 (Sec. 4.3).
N_INT = 4096

# Loss weights (see core.DEFAULT_LOSS_WEIGHTS for the same defaults). The two
# continuity residuals get separate weights, and Poisson is weighted well
# above the rest since it is the hardest term to converge.
LOSS_WEIGHTS = dict(core.DEFAULT_LOSS_WEIGHTS)
# LOSS_WEIGHTS["poisson"] = 100.0   # override any entry here as needed

# Collocation sampling distribution: symmetric Beta(a, a) on [0, 1]. a = 1
# recovers uniform sampling; a < 1 clusters points at both contacts.
BETA_CONCENTRATION = 1.0

# ---- Adaptive loss weighting -------------------------------------------
#
# Inverse-Dirichlet weighting (Maddu, Sturm, Muller & Sbalzarini 2022); see
# core.InverseDirichletWeights for the rule and Models_neural.md for the
# derivation.
#
# Motivation, specific to this device: a fixed weight is calibrated to one
# operating point. The `bc` term grows with the applied bias -- the boundary
# residual is squared and the contact value scales with the bias -- so a
# weight balancing it at the bottom of the sweep no longer does at the top.
# Setting the weights from the gradient balance instead tracks that.
#
# Why this scheme and not a gradient-magnitude ratio: every term in this loss
# except Poisson admits a trivial minimiser (a Dirichlet pin met exactly, a
# variance zeroed by any constant current, a flat density profile). A rule
# whose denominator is a mean gradient magnitude rewards a term for reaching
# such a zero and starves Poisson. Using the gradient *standard deviation*
# instead distinguishes a converged term (gradients small and uniform) from a
# stiff one (small on average, very uneven), so a weight grows only in
# proportion to the spread deficit it is derived from.
#
# This balances trainability, not accuracy, so it is scored against DEVSIM
# rather than assumed to be an improvement.
USE_ADAPTIVE_WEIGHTS = True

# Hard boundary ansatz for phi (see core.PhiNet): the contact values are built
# into the architecture, so phi satisfies them identically and the `bc` term
# carries only the density pins. Independent of the weighting scheme -- useful
# on its own, and it removes one exactly-satisfiable term from the loss.
USE_HARD_PHI_BC = True

# Hard boundary ansatz for the carrier densities (see core.LogDensityNet).
# Same trial-function construction as USE_HARD_PHI_BC, applied to
# u = -log(density_hat), and honouring MAJORITY_ONLY_BC: only the majority
# carrier is pinned at each contact (holes at the anode, electrons at the
# cathode), the minority end left free.
#
# Motivation: with soft pins the coupled solve has a degenerate minimum in
# which n and p collapse to nearly equal constants several decades below their
# physical scale. Flat densities zero the continuity and current-constancy
# residuals identically, and n ~ p shrinks the Poisson source enough that a
# straight-line phi satisfies Poisson as well. Pinning n(L) and p(0)
# architecturally removes that solution from the hypothesis space.
USE_HARD_DENSITY_BC = True

# Running-average rate for the weight update. Higher than a magnitude-ratio
# scheme would use (0.5 against 0.1): the gradient std is a much less noisy
# statistic than a raw magnitude ratio, so the weights can track it closely
# without chasing collocation noise.
ID_ALPHA = 0.5

# Which terms adapt. None = every term in ID_INITIAL_WEIGHTS. Unlike a
# reference-anchored scheme there is no distinguished term held fixed: the
# numerator is a max over the terms themselves, so all of them can adapt.
ID_TERMS = None

# Optional term to hold at its initial weight (normally None -- the scheme
# does not need an anchor).
ID_REFERENCE = None

# Optional cap on a single term's target ratio. NOT part of the published
# method -- a local addition, off by default and kept as a diagnostic rather
# than a tuning knob. See core.InverseDirichletWeights.
ID_MAX_RATIO = None

# Renormalise the adapted weights to sum to their count. OFF: this is not the
# paper's rule and it hurts -- it turns absolute amplifications into shares of
# a fixed budget, so a term with a large target divides every other weight
# down, including that of the term setting the reference scale.
ID_NORMALIZE = False

# Recompute the weights every N epochs. Each update costs one extra backward
# pass per adapted term, so N > 1 trades adaptation speed for wall-clock.
ID_UPDATE_EVERY = 10

# Starting weights for the adaptive run. Deliberately all-ones, so the
# trajectory shows what the scheme derives rather than where a hand-set value
# would put it -- the point is to test the rule, not to seed it with the
# answer. With ID_ALPHA = 0.5 the running average forgets the start quickly.
ID_INITIAL_WEIGHTS = {
    "bc": 1.0, "cont_n": 1.0, "cont_p": 1.0, "jtot": 1.0,
    "poisson": 1.0, "thermionic": 1.0,
}

# DDNet trains for 40,000 epochs.
EPOCHS = 20000
MILESTONES = [5000, 10000, 14000, 17000]
GAMMA = 0.3
LEARNING_RATE = 1e-3

# How often to copy the loss back from the accelerator (epochs); every
# .item() is a synchronisation point, so this is sampled rather than read
# every epoch.
LOG_EVERY = 20
PRINT_EVERY = 2000
CONVERGENCE_THRESHOLD = 1e-10


# ============================================================
# Derived scaling
# ============================================================

SCALING = core.compute_scaling(
    eps_r=EPS_R, T=TEMP_K, mu_n=MU_N, mu_p=MU_P, homo=HOMO, lumo=LUMO,
    nc300=NC300, nv300=NV300, gammar=GAMMAR, c_tilde=C_TILDE, ell=ELL,
)

print("Thermal voltage Ut     = {0:.6f} V".format(SCALING["Ut"]))
print("Debye length lambda_D  = {0:.4e} cm  ({1:.4f} nm)".format(
    SCALING["lam_D"], SCALING["lam_D"] * 1e7))
print("Scaled parameter lambda= {0:.4e}".format(SCALING["lam"]))
print("Scaled length L/L_D    = {0:.4f}".format(1.0 / SCALING["lam"]))
print("Intrinsic density nie  = {0:.4e} cm^-3  (nie_hat = {1:.4e})".format(
    SCALING["nie"], SCALING["nie_hat"]))
print("Langevin prefactor     = {0:.4e}".format(SCALING["langevin_prefactor"]))


# ============================================================
# Reference data + boundary conditions
# ============================================================

def load_reference(bias):
    """Load the DEVSIM reference profile at one bias and derive its BCs.

    Returns a dict holding the reference arrays in both physical and scaled
    units, together with the Dirichlet contact values the PINN is given.
    """
    path = reference_npz(bias)
    if not os.path.exists(path):
        raise SystemExit(
            "Missing {0}.\nRun `python devsim_reference_oled1.py` (from {1}) first "
            "to generate the DEVSIM reference solutions.".format(path, _TEST_DIR)
        )

    ref = np.load(path)
    x_nm = ref["x_nm"]
    potential = ref["potential"]        # V
    electrons = ref["electrons"]        # cm^-3
    holes = ref["holes"]                # cm^-3
    ref_bias = float(ref["bias"])
    vbi = float(ref["built_in_voltage"])
    # Terminal current in A/cm^2 (1D mesh, unit area -- see the reference
    # script's module docstring). Older reference files predate this field.
    ref_current = float(ref["current_A"]) if "current_A" in ref else None

    # Scaled reference arrays. Used only for boundary values and for scoring
    # the trained networks -- never as an interior training target.
    Ut = SCALING["Ut"]
    x_hat = x_nm / (ELL * 1e7)                # nm -> scaled [0, 1]
    phi_hat = potential / Ut                  # V  -> scaled by Ut
    n_hat = electrons / C_TILDE               # cm^-3 -> scaled by C_tilde
    p_hat = holes / C_TILDE

    return {
        "bias": ref_bias, "vbi": vbi, "current_A": ref_current,
        "x_nm": x_nm, "potential": potential,
        "electrons": electrons, "holes": holes,
        "x_hat": x_hat, "phi_hat": phi_hat, "n_hat": n_hat, "p_hat": p_hat,
        # Dirichlet contact values, taken from the reference solution at the
        # two contact nodes (applied bias at the anode, 0 at the cathode;
        # Ohmic density pins from the Contact spec).
        "phi_bc_left": float(phi_hat[0]),     # x = 0   (top / anode)
        "phi_bc_right": float(phi_hat[-1]),   # x = L   (bot / cathode)
        "n_bc_left": float(n_hat[0]),
        "n_bc_right": float(n_hat[-1]),
        "p_bc_left": float(p_hat[0]),
        "p_bc_right": float(p_hat[-1]),
    }


def load_iv():
    """Load the DEVSIM IV curve, or None if the sweep has not been run."""
    if not os.path.exists(IV_NPZ):
        return None
    iv = np.load(IV_NPZ)
    return {"voltages": iv["voltages"], "current_A": iv["current_A"]}


def describe_reference(ref):
    """Print the reference profile and the boundary conditions taken from it."""
    print()
    print("Reference from DEVSIM at {0:.2f} V (Vbi = {1:.2f} V)".format(
        ref["bias"], ref["vbi"]))
    print("  phi(0) = {0:+.5f} V  -> phi_hat = {1:+.4f}".format(
        ref["potential"][0], ref["phi_bc_left"]))
    print("  phi(L) = {0:+.5f} V  -> phi_hat = {1:+.4f}".format(
        ref["potential"][-1], ref["phi_bc_right"]))
    print("  n range: {0:.3e} .. {1:.3e} cm^-3".format(
        ref["electrons"].min(), ref["electrons"].max()))
    print("  p range: {0:.3e} .. {1:.3e} cm^-3".format(
        ref["holes"].min(), ref["holes"].max()))
    if ref["current_A"] is not None:
        print("  terminal current: {0:.4e} A/cm^2".format(ref["current_A"]))

    print()
    if MAJORITY_ONLY_BC:
        print("Density BCs: majority carrier only")
        print("  p(0) = {0:.4e} cm^-3   (anode, majority)".format(ref["holes"][0]))
        print("  n(L) = {0:.4e} cm^-3   (cathode, majority)".format(ref["electrons"][-1]))
        print("  n(0), p(L) left free -- discontinuous in the reference (see source)")
    else:
        print("Density BCs: both carriers pinned at both contacts")


# ============================================================
# Build networks + problem
# ============================================================

def build(ref):
    """Construct phi-Net / n-Net / p-Net and the PINN problem for one bias."""
    # Log-space initialisation offsets: one scalar per carrier, shifting the
    # net's starting output to the right order of magnitude. Without them
    # Xavier init gives u ~ 0, i.e. density_hat = exp(0) = 1 = C_tilde for
    # both carriers -- ~5 decades high for holes. Since the residuals depend
    # on u exponentially (n_hat*p_hat = exp(-u_n - u_p) in the Langevin term),
    # that start inflates recombination by ~1e5 and the early gradients chase
    # it instead of the physics. See LogDensityNet.__init__.
    #
    # The interior mean [1:-1] excludes the contact nodes: the minority pins
    # there are genuine discontinuities (p_hat = 7.3e-47 at the cathode) and
    # would drag the mean tens of units away from the interior scale.
    #
    # Caveat: this reads the DEVSIM reference. Defensible for a forward-problem
    # demo scored against that reference -- it is an initialisation heuristic,
    # not a training target, and carries only the mean order of magnitude, no
    # shape -- but unavailable on a device with no reference solution. The
    # generalisable substitute needs no solution: the contact densities are
    # known device parameters, so -log of their geometric mean gives the same
    # scalar.
    u_n_init = float(np.mean(-np.log(ref["n_hat"][1:-1])))
    u_p_init = float(np.mean(-np.log(ref["p_hat"][1:-1])))
    print()
    print("Log-space initialisation offsets (interior mean):")
    print("  u_n offset = {0:+.4f}  -> n_hat ~ {1:.3e}".format(u_n_init, np.exp(-u_n_init)))
    print("  u_p offset = {0:+.4f}  -> p_hat ~ {1:.3e}".format(u_p_init, np.exp(-u_p_init)))

    phi_bc = (ref["phi_bc_left"], ref["phi_bc_right"]) if USE_HARD_PHI_BC else None

    # Density pins, majority carrier only (MAJORITY_ONLY_BC): holes at the
    # anode (x = 0), electrons at the cathode (x = 1). The free end is None,
    # which LogDensityNet turns into the one-sided ansatz.
    u_n_bc = u_p_bc = None
    if USE_HARD_DENSITY_BC:
        u_n_bc = (None, -np.log(ref["n_bc_right"]))
        u_p_bc = (-np.log(ref["p_bc_left"]), None)
        if not MAJORITY_ONLY_BC:
            u_n_bc = (-np.log(ref["n_bc_left"]), u_n_bc[1])
            u_p_bc = (u_p_bc[0], -np.log(ref["p_bc_right"]))

    phi_net, n_net, p_net = core.build_networks(
        width=WIDTH, depth=DEPTH, u_n_offset=u_n_init, u_p_offset=u_p_init,
        device=device, phi_bc=phi_bc, u_n_bc=u_n_bc, u_p_bc=u_p_bc,
    )

    print("Trainable parameters: {0}".format(
        sum(p.numel() for p in list(phi_net.parameters())
            + list(n_net.parameters()) + list(p_net.parameters()))))

    return core.build_problem(
        phi_net=phi_net, n_net=n_net, p_net=p_net, scaling=SCALING,
        device=device, dtype=dtype,
        x_left=X_LEFT, x_right=X_RIGHT,
        phi_bc_left=ref["phi_bc_left"], phi_bc_right=ref["phi_bc_right"],
        majority_only_bc=MAJORITY_ONLY_BC,
        u_p_bc_left=-np.log(ref["p_bc_left"]),      # anode: holes (majority)
        u_n_bc_right=-np.log(ref["n_bc_right"]),    # cathode: electrons (majority)
        u_n_bc_left=-np.log(ref["n_bc_left"]),      # minority pin, only used if
        u_p_bc_right=-np.log(ref["p_bc_right"]),    # MAJORITY_ONLY_BC is False
        loss_weights=LOSS_WEIGHTS,
        beta_concentration=BETA_CONCENTRATION,
    )


# ============================================================
# Train / evaluate / plot
# ============================================================

def make_weights():
    """Build the loss-weight rule for a run.

    Either rule presents the same interface to core.train (a live weights dict
    plus an update hook), so switching between them needs no other change.
    """
    if not USE_ADAPTIVE_WEIGHTS:
        return core.make_weights("fixed", initial_weights=LOSS_WEIGHTS)
    return core.make_weights(
        "inverse_dirichlet", initial_weights=ID_INITIAL_WEIGHTS,
        reference=ID_REFERENCE, alpha=ID_ALPHA, update_every=ID_UPDATE_EVERY,
        terms=ID_TERMS, max_ratio=ID_MAX_RATIO, normalize=ID_NORMALIZE,
    )


def train(problem, weighting):
    return core.train(
        problem, epochs=EPOCHS, n_int=N_INT, lr=LEARNING_RATE,
        milestones=MILESTONES, gamma=GAMMA, log_every=LOG_EVERY,
        print_every=PRINT_EVERY, convergence_threshold=CONVERGENCE_THRESHOLD,
        loss_weights=weighting,
    )


def evaluate(problem, ref):
    return core.evaluate(
        problem, x_hat=ref["x_hat"], phi_true_V=ref["potential"],
        n_true=ref["electrons"], p_true=ref["holes"], bias=ref["bias"],
    )


def report_currents(problem, ref):
    return core.report_currents(
        problem, x_hat=ref["x_hat"], x_cm=ref["x_nm"] * 1e-7,
        ref_potential_V=ref["potential"], ref_electrons=ref["electrons"],
        ref_holes=ref["holes"],
    )


def plot(res, history_epochs, history_losses, ref, filename=None):
    if filename is None:
        filename = plot_png(ref["bias"])
    return core.plot(res, history_epochs, history_losses,
                      ref["x_nm"], ref["bias"], filename)


def run_bias(bias, interior=slice(1, -1)):
    """Train, score and plot one bias from scratch. Returns a summary dict."""
    print()
    print("#" * 70)
    print("# Applied bias {0:.2f} V".format(bias))
    print("#" * 70)

    # Reset the RNG per bias so the three runs differ only in the bias, not in
    # the initialisation or the collocation draw.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ref = load_reference(bias)
    describe_reference(ref)

    problem = build(ref)
    weighting = make_weights()
    history_epochs, history_losses = train(problem, weighting)
    res = evaluate(problem, ref)
    _, _, Jtot = report_currents(problem, ref)
    try:
        plot(res, history_epochs, history_losses, ref)
        # The weight-trajectory figure only exists for a rule that records
        # one; the fixed rule has nothing to plot.
        if USE_ADAPTIVE_WEIGHTS:
            core.plot_weights(weighting, weights_png(ref["bias"]),
                              bias=ref["bias"])
    except ImportError:
        pass

    print()
    print("Final loss weights: {0}".format(weighting.format_weights()))

    j_pinn = float(np.mean(Jtot[interior]))
    j_spread = float(np.ptp(Jtot[interior]))
    return {
        "weighting": weighting,
        "bias": ref["bias"],
        "rel_phi": res["rel_phi"],
        "rel_n_log": res["rel_n_log"],
        "rel_p_log": res["rel_p_log"],
        "max_abs_phi": float(np.max(np.abs(res["phi_pred"] - res["phi_true"]))),
        "j_pinn": j_pinn,
        "j_spread": j_spread,
        "j_devsim": ref["current_A"],
    }


def report_current_comparison(summaries):
    """Tabulate the PINN terminal current against the DEVSIM sweep."""
    print()
    print("=" * 78)
    print("Terminal current: PINN vs DEVSIM")
    print("=" * 78)
    print("  {0:>6s}  {1:>14s}  {2:>14s}  {3:>10s}  {4:>12s}".format(
        "V (V)", "PINN (A/cm^2)", "DEVSIM (A/cm^2)", "ratio", "PINN spread"))
    for s in summaries:
        j_ref = s["j_devsim"]
        ratio = "{0:10.3f}".format(s["j_pinn"] / j_ref) if j_ref else "{0:>10s}".format("n/a")
        ref_txt = "{0:14.6e}".format(j_ref) if j_ref is not None else "{0:>14s}".format("n/a")
        spread = 100 * s["j_spread"] / abs(s["j_pinn"]) if s["j_pinn"] else float("nan")
        print("  {0:6.2f}  {1:14.6e}  {2}  {3}  {4:11.2f} %".format(
            s["bias"], s["j_pinn"], ref_txt, ratio, spread))
    print("  (PINN spread = max-min of Jn+Jp across the device, as % of its mean;")
    print("   the total current must be x-independent, so this is a self-consistency check)")


def report_accuracy_summary(summaries):
    """Tabulate the profile errors across biases."""
    print()
    print("=" * 78)
    print("Accuracy against DEVSIM, per bias")
    print("=" * 78)
    print("  {0:>6s}  {1:>12s}  {2:>12s}  {3:>14s}  {4:>14s}".format(
        "V (V)", "phi relL1", "phi max err", "n relL1 (log)", "p relL1 (log)"))
    for s in summaries:
        print("  {0:6.2f}  {1:11.3f} %  {2:10.4f} V  {3:13.3f} %  {4:13.3f} %".format(
            s["bias"], s["rel_phi"] * 100, s["max_abs_phi"],
            s["rel_n_log"] * 100, s["rel_p_log"] * 100))


def main(biases=BIASES):
    summaries = [run_bias(bias) for bias in biases]
    report_accuracy_summary(summaries)
    report_current_comparison(summaries)
    return summaries


if __name__ == "__main__":
    # Note: training is not run automatically on import; main() (and thus
    # train()) only executes when this script is run directly.
    main()
