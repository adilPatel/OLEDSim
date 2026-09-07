"""
oled3_forward.py

Example: coupled drift-diffusion PINN (DDNet) for OLED3's 100 nm organic
active layer with Ohmic contacts, validated against the DEVSIM
drift-diffusion solver at a single applied bias.

Three networks -- phi-Net, n-Net, p-Net -- trained together against the
coupled Poisson + continuity system (DDNet Fig. 1). The reusable model,
loss, scaling and training/eval/plot logic live in ``sim_pinn.core``; this
script supplies OLED3's device parameters, boundary conditions, and run
configuration (loss weights, sampling distribution, epochs, ...).

Device
------
The 100 nm F8BT layer of ``devsim_reference_oled3.py``: LUMO -3.3 eV,
HOMO -5.9 eV, both effective DOS 1e21 cm^-3, both mobilities 1e-3 cm^2/V-s,
eps_r 3.5. The device is *symmetric*: the top contact's work function
(-5.5 eV) sits 0.4 eV above the HOMO and the bottom contact's (-3.7 eV) sits
0.4 eV below the LUMO, so both Ohmic pins equal N*exp(-0.4/Ut) = 1.95e14
cm^-3 and the reference solution satisfies n(x) = p(L - x).

Contrast with OLED1
-------------------
OLED1's density scale of 1e17 cm^-3 gives a 7.6 nm Debye length, so its
100 nm device is 13 screening lengths long and the potential is set by two
thin contact layers. Here the pin is 1.95e14 cm^-3, giving L_D = 162 nm and
L/L_D = 0.62: the device is shorter than one screening length, the space
charge barely bends the potential, and phi is close to the straight line
from V - Vbi = 0.4 V down to 0. The Poisson stiffness that made phi-Net
OLED1's accuracy bottleneck is therefore absent here.

Boundary conditions
--------------------
phi is pinned at both contacts (Dirichlet). The majority carrier density is
pinned at each contact; the minority carrier is left free (see
MAJORITY_ONLY_BC below) since the literal DEVSIM pin there is a
discontinuity a smooth network cannot represent -- the reference minority
density drops from 1.9e14 to 1.1e-16 cm^-3 across the single contact node.

Multiple biases
---------------
``devsim_reference_oled3.py`` sweeps up to 3.0 V and writes one reference
profile per bias in BIASES, plus the full IV curve. A separate set of
networks is trained from scratch at each bias -- nothing is transferred
between them, so the biases are independent samples of the same
configuration -- and each writes its own figure. ``main()`` then tabulates
the PINN terminal current against the DEVSIM one and draws the semilog J-V.

Running this script trains at every bias in BIASES; import it and call
``run_bias(V)`` to do just one.

Usage
-----
    python -m sim_pinn.devices.oled3_forward
"""

import os

import numpy as np
import torch

from sim_pinn import core

# Seed is re-applied in run_bias() so the run starts identically rather than
# inheriting whatever RNG state the importing process left.
SEED = 0

# ============================================================
# Network architecture
# ============================================================

WIDTH = 64
DEPTH = 4

# set_default_dtype fixes the precision of every tensor created afterwards.
# float32 here because the MPS accelerator has no float64.
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
    # torch.accelerator is the backend-agnostic handle on whatever hardware
    # is present (MPS on Apple silicon, CUDA on an NVIDIA box); falling back
    # to the CPU when there is none. torch.device is just the tag that tells
    # every tensor and module where to live.
    _accelerator = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else None
    device = torch.device(_accelerator.type) if _accelerator is not None else torch.device("cpu")
# The default dtype set above; every tensor built later is matched to it,
# since torch refuses to combine tensors of differing dtype.
dtype = torch.get_default_dtype()
print("Using device:", device)

_HERE = os.path.dirname(os.path.abspath(__file__))
# Repo root: up out of sim_pinn/devices/.
_ROOT = os.path.dirname(os.path.dirname(_HERE))

# DEVSIM reference + comparison plots live under tests/diode_pinn_1d/oled3,
# not next to this script (see devsim_reference_oled3.py in that directory).
_TEST_DIR = os.path.join(_ROOT, "tests", "diode_pinn_1d", "oled3")

# Full sweep's current-voltage curve, for the J-V comparison figure.
IV_NPZ = os.path.join(_TEST_DIR, "oled3_devsim_iv.npz")

# Applied biases to train at. Each needs a matching reference file from
# devsim_reference_oled3.py (whose TARGET_BIASES must agree with this list).
BIASES = (2.0,)


def reference_npz(bias):
    """Path of the DEVSIM reference profile for one bias."""
    return os.path.join(_TEST_DIR, "oled3_devsim_reference_{0:.1f}V.npz".format(bias))


def plot_tag():
    """Short tag naming the configuration under test.

    Figures carry the tag so one configuration cannot overwrite another's
    output: the adaptive rule's own name when weights adapt, "hdbc" for fixed
    weights under the hard boundary ansatz, "demo" for the plain run.
    """
    if USE_ADAPTIVE_WEIGHTS:
        tag = {"inverse_dirichlet": "id", "softadapt": "sa"}.get(
            ADAPTIVE_KIND, ADAPTIVE_KIND)
    elif USE_HARD_DENSITY_BC or USE_HARD_PHI_BC:
        tag = "hdbc"
    else:
        tag = "demo"
    # The parametrisation gets its own suffix so a quasi-Fermi run cannot
    # overwrite the log-form figures it is being compared against.
    return tag + "_qf" if USE_QUASI_FERMI else tag


def plot_png(bias):
    """Path of the comparison figure for one bias."""
    return os.path.join(_TEST_DIR, "oled3_forward_{0}_{1:.1f}V.png".format(
        plot_tag(), bias))


def iv_png():
    """Path of the J-V comparison figure for the whole sweep."""
    return os.path.join(_TEST_DIR, "oled3_forward_{0}_JV.png".format(plot_tag()))


def weights_png(bias):
    """Path of the adaptive-weight evolution figure for one bias."""
    return os.path.join(_TEST_DIR, "oled3_forward_{0}_weights_{1:.1f}V.png".format(
        plot_tag(), bias))


# ============================================================
# Device / material parameters (OLED3, 100 nm organic layer)
# ============================================================

EPS_R  = 3.5                  # OLED3's relative permittivity
TEMP_K = 300.0                # K

# Transport parameters, from devsim_reference_oled3.py's make_device().
# Symmetric device: electrons and holes share both mobility and DOS.
MU_N   = 1.0e-3               # electron mobility, cm^2/V-s
MU_P   = 1.0e-3               # hole mobility, cm^2/V-s
NC300  = 1.0e21               # conduction-band effective DOS, cm^-3
NV300  = 1.0e21               # valence-band effective DOS, cm^-3
GAMMAR = 1.0                  # Langevin prefactor ("gammar" in os_physics.py)

# HOMO/LUMO of the reference device (implies the intrinsic density nie via
# core.compute_scaling, following common_physics.py's CreateDensityOfStates
# at T = 300 K: NIE = sqrt(NC*NV) * exp(-EG/(2*Ut))).
LUMO = -3.3
HOMO = -5.9                   # EG = 2.6 eV

# Injection barrier at both contacts: the top work function (-5.5 eV) sits
# PHI_B above the HOMO, the bottom one (-3.7 eV) sits PHI_B below the LUMO.
PHI_B = 0.4                   # eV

# Density scale: the Ohmic contact pin, the largest carrier density in the
# device. Both contacts share PHI_B and both DOS are equal, so the same value
# N*exp(-PHI_B/Ut) pins holes at the anode and electrons at the cathode.
# Derived from the device parameters rather than read off the reference, and
# bias-independent, so one scaling serves every run.
C_TILDE = NC300 * np.exp(-PHI_B / (core.K_B * TEMP_K / core.Q))   # ~1.95e14 cm^-3

ELL = 100.0e-7                # device length, cm (100 nm)

# Domain scaled to x_hat in [0, 1] by the device length (not by the Debye
# length).
X_LEFT, X_RIGHT = 0.0, 1.0

# Which density boundary conditions are actually imposable. At each Ohmic
# contact DEVSIM pins BOTH carrier densities, but only the majority pin is
# representable by a smooth network -- the minority pin is a genuine
# discontinuity in the reference data (1.9e14 -> 1.1e-16 cm^-3 across one
# node). So the majority carrier is pinned at each contact and the minority
# carrier is left free, determined by the continuity equations.
#
# Left contact (top, anode): holes are the majority carrier.
# Right contact (bot, cathode): electrons are the majority carrier.
MAJORITY_ONLY_BC = False


# ============================================================
# Loss weights, sampling, training schedule
# ============================================================

# Interior collocation points per step. DDNet uses 2^12 = 4096 (Sec. 4.3);
# doubled to 2^13 here because BETA_CONCENTRATION = 0.5 moves points from the
# bulk to the contacts. At 4096 the clustered draw would leave the 10-90 nm
# bulk with ~2400 points against the uniform draw's ~3270; at 8192 the bulk
# gets ~4840 AND the outer 2 nm layers ~1480 (against ~165 uniform), so both
# regions are sampled better than the unclustered baseline rather than traded
# off against each other.
N_INT = 8192

# Loss weights (see core.DEFAULT_LOSS_WEIGHTS for the same defaults). Only
# consulted when USE_ADAPTIVE_WEIGHTS is off.
LOSS_WEIGHTS = dict(core.DEFAULT_LOSS_WEIGHTS)
# LOSS_WEIGHTS["poisson"] = 100.0   # override any entry here as needed

# Collocation sampling distribution: symmetric Beta(a, a) on [0, 1]. a = 1
# recovers uniform sampling; a < 1 clusters points at both contacts. Uniform
# here: at L/L_D = 0.62 there is no thin contact layer to resolve.
BETA_CONCENTRATION = 0.5

# ---- Adaptive loss weighting -------------------------------------------
#
# Why adapt at all: a fixed weight is calibrated to one operating point, and
# the terms of this loss span many decades. Deriving the weights from the loss
# itself tracks that. This balances trainability, not accuracy, so it is
# scored against DEVSIM rather than assumed to be an improvement.
USE_ADAPTIVE_WEIGHTS = True

# Which adaptive rule: "inverse_dirichlet" or "softadapt".
#
# inverse_dirichlet is retained for reproducibility but is NOT recommended on
# this loss: lambda_i = max_j(s_j)/s_i pins the largest-spread term at
# lambda = 1 with no way to raise it, so when that term is the accuracy
# bottleneck it is starved and stays starved. Which term wins that argmax can
# turn on numerical noise near a tie.
#
# softadapt scores each term by its recent *relative* rate of change and takes
# a softmax, so the weights are bounded (sum to 1), no term is singled out by
# an argmax, and a term that has stopped improving because it is finished is
# not amplified.
ADAPTIVE_KIND = "softadapt"

# Softmax temperature. beta > 0 puts more weight on the slowly-improving
# terms. With relative rates s_i is O(1), so beta ~ 1 is the scale-free
# starting point; beta -> 0 gives uniform weights.
SA_BETA = 1.0

# Recompute every N epochs. s_i is then a difference over N epochs, which
# averages out collocation-resampling noise, but couples beta to this value.
SA_UPDATE_EVERY = 10

# Use each term's relative rate s_i / |L_i(t-1)| rather than the raw
# difference. Effectively required here: the losses span many decades, and an
# absolute rate ranks a term at 1e-13 as "barely changing" however fast it is
# actually converging (see core.SoftAdaptWeights).
SA_NORMALIZE = True

# Scale each softmax term by its share of the total loss (the paper's
# "Loss-Weighted SoftAdapt"). Effectively required: the relative rate is
# unbounded above, so without this the rule is captured by whichever term is
# smallest and noisiest. The loss share also does automatically what the
# hand-set LOSS_WEIGHTS do by hand -- bias the optimiser towards the terms
# still carrying real residual.
SA_LOSS_WEIGHTED = True

# Starting weights, deliberately uniform: the softmax overwrites them from the
# second update onward, so these only set the very first epochs.
SA_INITIAL_WEIGHTS = {
    "bc": 1.0, "cont_n": 1.0, "cont_p": 1.0, "jtot": 1.0,
    "poisson": 1.0, "thermionic": 1.0,
}

# ---- Density parametrisation -------------------------------------------
#
# False: the networks emit u = -log(density_hat)  (core.densities, DDNet's
# own form). True: they emit the quasi-Fermi potentials phi_n, phi_p
# (core.densities_qf), with the densities recovered from the Boltzmann
# relations. See Models_neural.md, "Quasi-Fermi parametrisation".
#
# On this device the log form leaves the unpinned minority ends flat: its
# continuity residual is multiplied by n_hat ~ 1e-30 there and carries no
# gradient. The quasi-Fermi current Jn = -mu_n*n*phi_n' keeps a constraint in
# that region, and the ~30-decade contact roll-off becomes a nearly flat
# phi_n rather than a 70-unit swing in u.
USE_QUASI_FERMI = True


# Decay length (scaled by the device length) of the quasi-Fermi two-ended
# boundary shape functions -- see core.QuasiFermiNet. Only used when both ends
# of a density net are pinned, i.e. MAJORITY_ONLY_BC = False.
#
# The linear interpolant a two-ended ansatz normally uses ramps between pins
# that are ~77 units apart here, while the true phi_n leaves its contact value
# within ~0.1 nm and sits near the far pin for the rest of the device. Cancelling
# that ramp needs a bubble amplitude ~5e4, which a tanh network cannot supply;
# 0.002 (0.2 nm) brings the requirement down to ~1e2.
QF_BC_DECAY = 0.0005

# Analytic interior term carried inside the quasi-Fermi bubble (see
# core.QuasiFermiNet). With the localised baseline confined to ~0.25 nm of
# each contact, the whole interior is otherwise left to the network; this
# supplies its leading behaviour analytically.
#
# QF_INTERIOR_LINEAR: take the linear ramp between the two contacts'
# equilibrium quasi-Fermi values, phi +/- log(nie_hat). Derived from the
# contact potentials and the material's nie -- not fitted to the reference.
# The gradient is the physically meaningful part: Jn = -mu_n*n*phi_n', so a
# constant phi_n' is the constant-current solution.
#
# QF_INTERIOR_BUMP: amplitude of an additional 4x(1-x) term, which vanishes at
# both contacts and peaks mid-device. 0 disables it. Measured against the
# reference the required interior correction is a monotone ~11-unit ramp
# rather than a mid-device bump, so the linear term carries it and this is
# available as a separate knob rather than a replacement.
QF_INTERIOR_LINEAR = True
QF_INTERIOR_BUMP = 0.0


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

# Running-average rate for the inverse-Dirichlet weight update. Higher than a
# magnitude-ratio scheme would use (0.5 against 0.1): the gradient std is a
# much less noisy statistic than a raw magnitude ratio, so the weights can
# track it closely without chasing collocation noise.
ID_ALPHA = 0.5

# Which terms adapt. None = every term in the initial-weights dict. Unlike a
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

# Starting weights for the inverse-Dirichlet run, deliberately all-ones so the
# trajectory shows what the scheme derives rather than where a hand-set value
# would put it. With ID_ALPHA = 0.5 the running average forgets the start
# quickly.
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

print("Density scale C_tilde  = {0:.4e} cm^-3".format(C_TILDE))
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

    Returns a dict of the reference arrays in both physical and scaled units,
    the terminal current, and the Dirichlet contact values the PINN is given.
    """
    path = reference_npz(bias)
    if not os.path.exists(path):
        raise SystemExit(
            "Missing {0}.\nRun `python devsim_reference_oled3.py` (from {1}) first "
            "to generate the DEVSIM reference solution.".format(path, _TEST_DIR)
        )

    ref = np.load(path)
    x_nm = ref["x_nm"]
    potential = ref["potential"]        # V
    electrons = ref["electrons"]        # cm^-3
    holes = ref["holes"]                # cm^-3
    ref_bias = float(ref["bias"])
    vbi = float(ref["built_in_voltage"])
    # Terminal current in A/cm^2 (1D mesh, unit area). devsim_reference_oled3.py
    # records no current field, so this is None and the comparison falls back
    # to the finite-differenced profile current in report_currents().
    ref_current = float(ref["current_A"]) if "current_A" in ref else None

    # Scaled reference arrays. Used only for boundary values and for scoring
    # the trained networks -- never as an interior training target.
    Ut = SCALING["Ut"]
    x_hat = x_nm / (ELL * 1e7)                # nm -> scaled [0, 1]
    phi_hat = potential / Ut                  # V  -> scaled by Ut
    n_hat = electrons / C_TILDE               # cm^-3 -> scaled by C_tilde
    p_hat = holes / C_TILDE

    return {
        # Run metadata.
        "bias": ref_bias, "vbi": vbi, "current_A": ref_current,
        "built_in_voltage": vbi,
        # Profiles in physical units, for scoring and plotting.
        "x_nm": x_nm, "potential": potential,
        "electrons": electrons, "holes": holes,
        # The same, scaled -- the units the networks work in.
        "x_hat": x_hat, "phi_hat": phi_hat, "n_hat": n_hat, "p_hat": p_hat,
        # Dirichlet contact values, read off the two contact nodes: applied
        # bias at the anode, 0 at the cathode, Ohmic density pins from the
        # Contact spec.
        "phi_bc_left": float(phi_hat[0]),     # x = 0   (top / anode)
        "phi_bc_right": float(phi_hat[-1]),   # x = L   (bot / cathode)
        "n_bc_left": float(n_hat[0]),
        "n_bc_right": float(n_hat[-1]),
        "p_bc_left": float(p_hat[0]),
        "p_bc_right": float(p_hat[-1]),
    }


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
    # Log-space initialisation offsets, one scalar per carrier, shifting each
    # net's starting output to the right order of magnitude (see
    # LogDensityNet). The interior mean [1:-1] drops the contact nodes, whose
    # minority pins are discontinuities that would drag the mean tens of units
    # off the interior scale.
    #
    # Caveat: this reads the reference. Defensible for a forward-problem demo
    # scored against it -- an initialisation heuristic carrying only the mean
    # order of magnitude, no shape -- but unavailable without a reference. The
    # generalisable substitute needs none: the contact densities are known
    # device parameters, so -log of their geometric mean gives the same scalar.
    if USE_QUASI_FERMI:
        # Interior mean of the quasi-Fermi levels implied by the reference,
        # phi_n = phi_hat - log(n_hat) + log(nie_hat). Under the localised
        # shape functions the baseline decays to zero away from the contacts,
        # so this offset is what sets the bulk level.
        lnie_i = SCALING["log_nie_hat"]
        # Zero when the analytic interior term already sets the bulk level;
        # otherwise the interior mean of the reference quasi-Fermi levels.
        if QF_INTERIOR_LINEAR:
            u_n_init = u_p_init = 0.0
        else:
            u_n_init = float(np.mean(
                ref["phi_hat"][1:-1] - np.log(ref["n_hat"][1:-1]) + lnie_i))
            u_p_init = float(np.mean(
                ref["phi_hat"][1:-1] + np.log(ref["p_hat"][1:-1]) - lnie_i))
        print()
        print("Quasi-Fermi initialisation offsets (interior mean):")
        print("  phi_n offset = {0:+.4f}   phi_p offset = {1:+.4f}".format(
            u_n_init, u_p_init))
    else:
        u_n_init = float(np.mean(-np.log(ref["n_hat"][1:-1])))
        u_p_init = float(np.mean(-np.log(ref["p_hat"][1:-1])))
        print()
        print("Log-space initialisation offsets (interior mean):")
        print("  u_n offset = {0:+.4f}  -> n_hat ~ {1:.3e}".format(u_n_init, np.exp(-u_n_init)))
        print("  u_p offset = {0:+.4f}  -> p_hat ~ {1:.3e}".format(u_p_init, np.exp(-u_p_init)))

    phi_bc = (ref["phi_bc_left"], ref["phi_bc_right"]) if USE_HARD_PHI_BC else None

    # Contact pin values, in whichever variable the density nets emit. Under
    # the quasi-Fermi form these come from the contact potential and the
    # material's nie (core.ohmic_quasi_fermi_bc), not from the reference
    # densities: at an Ohmic contact the quasi-Fermi levels equal the metal's.
    if USE_QUASI_FERMI:
        lnie = SCALING["log_nie_hat"]
        # Majority pins from the Ohmic equilibrium rule: at a contact whose
        # majority density IS the density scale, density_hat = 1 and the
        # quasi-Fermi level is the contact potential offset by log(nie_hat).
        pin_p_left = core.ohmic_quasi_fermi_bc(
            ref["phi_bc_left"], log_nie_hat=lnie, carrier="p")
        pin_n_right = core.ohmic_quasi_fermi_bc(
            ref["phi_bc_right"], log_nie_hat=lnie, carrier="n")
        # Minority pins, used only when MAJORITY_ONLY_BC is False. The
        # equilibrium rule does not apply here -- it assumes density_hat = 1,
        # true only for the majority carrier -- so these are inverted from the
        # reference's own contact densities:
        #     phi_n = phi_hat - log(n_hat) + log(nie_hat)
        pin_n_left = ref["phi_bc_left"] - np.log(ref["n_bc_left"]) + lnie
        pin_p_right = ref["phi_bc_right"] + np.log(ref["p_bc_right"]) - lnie
        print("Quasi-Fermi contact pins (scaled by Ut):")
        print("  majority: phi_p(anode) = {0:+.4f}   phi_n(cathode) = {1:+.4f}".format(
            pin_p_left, pin_n_right))
        print("  minority: phi_n(anode) = {0:+.4f}   phi_p(cathode) = {1:+.4f}".format(
            pin_n_left, pin_p_right))
    else:
        pin_n_left = -np.log(ref["n_bc_left"])
        pin_n_right = -np.log(ref["n_bc_right"])
        pin_p_left = -np.log(ref["p_bc_left"])
        pin_p_right = -np.log(ref["p_bc_right"])

    # Density pins, majority carrier only (MAJORITY_ONLY_BC): holes at the
    # anode (x = 0), electrons at the cathode (x = 1). The free end is None,
    # which the density net turns into the one-sided ansatz.
    u_n_bc = u_p_bc = None
    if USE_HARD_DENSITY_BC:
        u_n_bc = (None, pin_n_right)
        u_p_bc = (pin_p_left, None)
        if not MAJORITY_ONLY_BC:
            u_n_bc = (pin_n_left, u_n_bc[1])
            u_p_bc = (u_p_bc[0], pin_p_right)

    # Interior term coefficients, per carrier. Both carriers share the slope
    # phi(L) - phi(0): the quasi-Fermi levels track the electrostatic
    # potential across the bulk, offset by +/- log(nie_hat).
    qf_interior = None
    if USE_QUASI_FERMI and (QF_INTERIOR_LINEAR or QF_INTERIOR_BUMP):
        lin = 1.0 if QF_INTERIOR_LINEAR else 0.0
        slope = ref["phi_bc_right"] - ref["phi_bc_left"]
        qf_interior = {
            "n": dict(interior_offset=lin * (ref["phi_bc_left"] + lnie),
                      interior_slope=lin * slope,
                      interior_bump=QF_INTERIOR_BUMP),
            "p": dict(interior_offset=lin * (ref["phi_bc_left"] - lnie),
                      interior_slope=lin * slope,
                      interior_bump=-QF_INTERIOR_BUMP),
        }
        print("Quasi-Fermi interior term:")
        print("  phi_n: {0:+.4f} {1:+.4f}*x   phi_p: {2:+.4f} {3:+.4f}*x".format(
            qf_interior["n"]["interior_offset"], slope,
            qf_interior["p"]["interior_offset"], slope))

    phi_net, n_net, p_net = core.build_networks(
        width=WIDTH, depth=DEPTH, u_n_offset=u_n_init, u_p_offset=u_p_init,
        device=device, phi_bc=phi_bc, u_n_bc=u_n_bc, u_p_bc=u_p_bc,
        quasi_fermi=USE_QUASI_FERMI,
        log_nie_hat=SCALING["log_nie_hat"] if USE_QUASI_FERMI else None,
        bc_decay=QF_BC_DECAY if USE_QUASI_FERMI else None,
        qf_interior=qf_interior,
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
        u_p_bc_left=pin_p_left,        # anode: holes (majority)
        u_n_bc_right=pin_n_right,      # cathode: electrons (majority)
        u_n_bc_left=pin_n_left,        # minority pin, only used if
        u_p_bc_right=pin_p_right,      # MAJORITY_ONLY_BC is False
        loss_weights=LOSS_WEIGHTS,
        beta_concentration=BETA_CONCENTRATION,
        quasi_fermi=USE_QUASI_FERMI,
    )


# ============================================================
# Train / evaluate / plot
# ============================================================

def make_weights():
    """Build the loss-weight rule for a run, per USE_ADAPTIVE_WEIGHTS and
    ADAPTIVE_KIND.

    Every rule presents core.train the same interface -- a live weights dict
    plus an update hook -- so switching between them needs no other change.
    """
    if not USE_ADAPTIVE_WEIGHTS:
        return core.make_weights("fixed", initial_weights=LOSS_WEIGHTS)
    if ADAPTIVE_KIND == "softadapt":
        return core.make_weights(
            "softadapt", initial_weights=SA_INITIAL_WEIGHTS,
            beta=SA_BETA, update_every=SA_UPDATE_EVERY,
            terms=ID_TERMS, normalize=SA_NORMALIZE,
            loss_weighted=SA_LOSS_WEIGHTED,
        )
    return core.make_weights(
        "inverse_dirichlet", initial_weights=ID_INITIAL_WEIGHTS,
        reference=ID_REFERENCE, alpha=ID_ALPHA, update_every=ID_UPDATE_EVERY,
        terms=ID_TERMS, max_ratio=ID_MAX_RATIO, normalize=ID_NORMALIZE,
    )


def train(problem, weighting):
    """Train one problem under this script's schedule."""
    return core.train(
        problem, epochs=EPOCHS, n_int=N_INT, lr=LEARNING_RATE,
        milestones=MILESTONES, gamma=GAMMA, log_every=LOG_EVERY,
        print_every=PRINT_EVERY, convergence_threshold=CONVERGENCE_THRESHOLD,
        loss_weights=weighting,
    )


def evaluate(problem, ref):
    """Score the trained networks against the reference profile."""
    return core.evaluate(
        problem, x_hat=ref["x_hat"], phi_true_V=ref["potential"],
        n_true=ref["electrons"], p_true=ref["holes"], bias=ref["bias"],
        x_cm=ref["x_nm"] * 1e-7,
    )


def report_currents(problem, ref):
    """Report the terminal current, against the finite-differenced reference."""
    return core.report_currents(
        problem, x_hat=ref["x_hat"], x_cm=ref["x_nm"] * 1e-7,
        ref_potential_V=ref["potential"], ref_electrons=ref["electrons"],
        ref_holes=ref["holes"],
    )


def plot(res, history_epochs, history_losses, ref, filename=None):
    """Write the four-panel figure, to plot_png(bias) unless told otherwise."""
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

    # Reset the RNG so the run is reproducible.
    # manual_seed fixes torch's RNG, which drives both the Xavier weight init
    # and the collocation draw; numpy's seed covers the reference handling.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ref = load_reference(bias)
    describe_reference(ref)

    problem = build(ref)
    weighting = make_weights()
    history_epochs, history_losses = train(problem, weighting)
    res = evaluate(problem, ref)
    _, _, Jtot = report_currents(problem, ref)
    rec = report_recombination(problem, ref)
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

    # Terminal current: its interior mean, and the spread that says how
    # x-independent it actually came out.
    j_pinn = float(np.mean(Jtot[interior]))
    j_spread = float(np.ptp(Jtot[interior]))
    return {
        "weighting": weighting,
        "bias": ref["bias"],
        "built_in_voltage": float(ref["built_in_voltage"]),
        # Profile errors, for report_accuracy_summary.
        "rel_phi": res["rel_phi"],
        "rel_n_log": res["rel_n_log"],
        "rel_p_log": res["rel_p_log"],
        "max_abs_phi": float(np.max(np.abs(res["phi_pred"] - res["phi_true"]))),
        # Currents, for report_current_comparison.
        "j_pinn": j_pinn,
        "j_spread": j_spread,
        "j_devsim": ref["current_A"],
        # Recombination accuracy, split by region (see report_recombination).
        "rel_R_layer": rec["rel_R_layer"],
        "rel_R_bulk": rec["rel_R_bulk"],
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


def report_recombination(problem, ref, n_layer_nm=5.0):
    """Score the Langevin recombination profile, and the boundary layers.

    R = gammar*(q/eps)*(mu_n + mu_p)*(n*p - nie^2) is the emission-zone
    observable: it is a *product* of the two densities, so it peaks where the
    carrier populations overlap rather than where either is largest. Bulk-
    dominated density errors can therefore look small while R at the contacts
    is badly wrong, which is what this reports separately.

    n_layer_nm : width of the contact region scored as the "boundary layer".
    """
    x_nm = ref["x_nm"]
    # Predicted densities on the reference nodes, in cm^-3.
    x_t = torch.as_tensor(ref["x_hat"].reshape(-1, 1), dtype=problem.dtype,
                          device=problem.device)
    # no_grad: these are being reported, not differentiated.
    with torch.no_grad():
        n_pred = problem.n_net.density(x_t).cpu().numpy().flatten() * C_TILDE
        p_pred = problem.p_net.density(x_t).cpu().numpy().flatten() * C_TILDE

    # R in physical units, both from the PINN and from the reference profiles.
    pref = GAMMAR * (core.Q / (EPS_R * core.EPS_0)) * (MU_N + MU_P)
    nie = SCALING["nie"]
    R_pred = pref * (n_pred * p_pred - nie ** 2)
    R_ref = pref * (ref["electrons"] * ref["holes"] - nie ** 2)

    # Masks: the two contact layers, and the bulk between them.
    layer = (x_nm <= n_layer_nm) | (x_nm >= x_nm[-1] - n_layer_nm)
    bulk = ~layer

    def rel(mask):
        num = np.sum(np.abs(R_pred[mask] - R_ref[mask]))
        den = np.sum(np.abs(R_ref[mask]))
        return num / den if den else float("nan")

    print()
    print("=" * 70)
    print("Langevin recombination R(x) at {0:.2f} V".format(ref["bias"]))
    print("=" * 70)
    print("  R relative L1, contact layers (<={0:.0f} nm): {1:8.3f} %".format(
        n_layer_nm, rel(layer) * 100))
    print("  R relative L1, bulk                       : {0:8.3f} %".format(
        rel(bulk) * 100))
    print("  R relative L1, whole device               : {0:8.3f} %".format(
        rel(np.ones_like(layer, dtype=bool)) * 100))
    print("  peak R  PINN {0:.4e} at x = {1:6.2f} nm".format(
        R_pred.max(), x_nm[int(np.argmax(R_pred))]))
    print("  peak R  ref  {0:.4e} at x = {1:6.2f} nm".format(
        R_ref.max(), x_nm[int(np.argmax(R_ref))]))
    # Where the emission actually sits: the integrated-R centroid.
    def centroid(R):
        w = np.clip(R, 0.0, None)
        return float(np.sum(w * x_nm) / np.sum(w)) if np.sum(w) else float("nan")
    print("  emission centroid  PINN {0:6.2f} nm   ref {1:6.2f} nm".format(
        centroid(R_pred), centroid(R_ref)))
    return {"rel_R_layer": rel(layer), "rel_R_bulk": rel(bulk)}


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
    """Train at every bias in turn, then tabulate accuracy and current."""
    summaries = [run_bias(bias) for bias in biases]
    report_accuracy_summary(summaries)
    report_current_comparison(summaries)
    try:
        core.plot_iv(IV_NPZ, summaries, iv_png(),
                     built_in_voltage=summaries[0]["built_in_voltage"])
    except (ImportError, FileNotFoundError):
        pass
    return summaries


if __name__ == "__main__":
    # Note: training is not run automatically on import; main() (and thus
    # train()) only executes when this script is run directly.
    main()
