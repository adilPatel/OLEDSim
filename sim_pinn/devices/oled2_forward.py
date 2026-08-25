"""
oled2_forward.py

Example: coupled drift-diffusion PINN (DDNet) for OLED2's 100 nm organic
active layer with thermionic and Ohmic contacts, validated against the DEVSIM
drift-diffusion solver at 3.0 V.

Three networks -- phi-Net, n-Net, p-Net -- trained together against the
coupled Poisson + continuity system (DDNet Fig. 1). The reusable model,
loss, scaling and training/eval/plot logic live in core.py; this script
supplies OLED2's device parameters, boundary conditions, and run
configuration (loss weights, sampling distribution, epochs, ...).

Device
------
The 100 nm organic layer of ``devsim_reference_oled2.py``: OLED2's
exact device parameters with a thermionic top contact and an Ohmic 
bottom contact.

Boundary conditions
--------------------
phi is pinned at both contacts (Dirichlet), at both contact types: the
electrode is a good conductor either way, exactly as DEVSIM's thermionic
contact keeps the Ohmic potential equation.

The electron density is pinned at the bottom (Ohmic) contact, where electrons
are the majority carrier; the hole density there is left free (see
MAJORITY_ONLY_BC below) since the literal DEVSIM pin is a discontinuity a
smooth network cannot represent.

At the top contact both densities are left free: the thermionic model has no
density pin at all. They are fixed instead by the field-dependent injection
flux balance (TOP_CONTACT below, and core.ThermionicContact), which enters
the loss as an extra residual term rather than as a Dirichlet condition.

Running this script builds the problem and prints the derived scaling, but
does not train (call train() / main() explicitly -- see the bottom of this
file).
"""

import os

import numpy as np
import torch

from sim_pinn import core

torch.manual_seed(0)
np.random.seed(0)

torch.set_default_dtype(torch.float64)

# ============================================================
# Run configuration: device (CPU/GPU)
# ============================================================

USE_GPU = False

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

# DEVSIM reference + comparison plot live under tests/diode_pinn_1d/oled2,
# not next to this script (see devsim_reference_oled2.py in that directory).
_TEST_DIR = os.path.join(_ROOT, "tests", "diode_pinn_1d", "oled2")
REFERENCE_NPZ = os.path.join(_TEST_DIR, "oled2_devsim_reference_3.0V.npz")
PLOT_PNG = os.path.join(_TEST_DIR, "oled2_forward_demo.png")

# ============================================================
# Device / material parameters (OLED2, 100 nm organic layer)
# ============================================================

EPS_R  = 3.5                  # OLED2's relative permittivity
TEMP_K = 300.0                # K

# Transport parameters, matching devsim_reference_oled2.py's make_device()
# exactly -- unlike OLED1, that script overrides both the mobilities and the
# densities of states, so these are its values, not core.device.Device's
# defaults. The DOS in particular is the literature-typical organic value
# rather than Device's 1e27, which combined with F8BT's higher mobility
# overflows the Langevin model during the equilibrium solve.
MU_N   = 4.0e-3               # electron mobility, cm^2/V-s
MU_P   = 1.0e-3               # hole mobility, cm^2/V-s
NC300  = 1.0e21               # conduction-band effective DOS, cm^-3
NV300  = 1.0e21               # valence-band effective DOS, cm^-3
GAMMAR = 1.0                  # Langevin prefactor ("gammar" in os_physics.py)

# HOMO/LUMO of the reference device (implies the intrinsic density nie via
# core.compute_scaling, following common_physics.py's CreateDensityOfStates
# at T = 300 K: NIE = sqrt(NC*NV) * exp(-EG/(2*Ut))).
LUMO = -3.3
HOMO = -5.9                   # EG = 2.6 eV

# Contact work functions (eV, negative = below vacuum), from make_device().
# The top contact is thermionic, the bottom Ohmic; both are described by a
# work function, so the injection barriers below are derived rather than
# specified.
WF_TOP = -5.3                 # anode (thermionic)
WF_BOT = -3.3                 # cathode (Ohmic)

# Density scale: it is chosen as 1e21 due to the F8BT's intrinsic DOS.
# TODO: Make this equal to max(1e21, n_bottom) where n is the computed
# bottom contact density.
C_TILDE = 1.0e21              # cm^-3

ELL = 100.0e-7                # device length, cm (100 nm)

# Domain scaled to x_hat in [0, 1] by the device length (not by the Debye
# length).
X_LEFT, X_RIGHT = 0.0, 1.0

# Which density boundary conditions are actually imposable. At the Ohmic
# (bottom) contact DEVSIM pins BOTH carrier densities, but only the majority
# pin is representable by a smooth network -- the minority pin is a genuine
# discontinuity in the reference data. So the majority carrier is pinned there
# and the minority carrier is left free, determined by the continuity
# equations.
#
# This flag only governs Ohmic contacts. The thermionic (top) contact has no
# density pin of either kind: both densities there are unknowns set by the
# injection flux balance, so the flag does not reach it.
#
# Right contact (cathode, Ohmic): electrons are the majority carrier.
MAJORITY_ONLY_BC = True

# The top contact's field-dependent thermionic injection. The two barriers are
# derived from the work function exactly as backend.py does for the DEVSIM
# device: phi_n = LUMO - work_function, phi_p = work_function - HOMO. For
# OLED2's -5.3 eV anode this gives a 2.0 eV electron barrier and a 0.6 eV hole
# barrier -- a good hole injector and a poor electron injector, which is what
# makes it the anode.
PHI_N_TOP = LUMO - WF_TOP     # electron injection barrier at the top, eV
PHI_P_TOP = WF_TOP - HOMO     # hole injection barrier at the top, eV

TOP_CONTACT = core.ThermionicContact(
    x=0.0, side="left", phi_n=PHI_N_TOP, phi_p=PHI_P_TOP,
)

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
# above the rest since it is the hardest term to converge. "thermionic"
# weights the top contact's injection flux balance, the term that stands in
# for the Dirichlet density pins there.
LOSS_WEIGHTS = dict(core.DEFAULT_LOSS_WEIGHTS)
# LOSS_WEIGHTS["poisson"] = 100.0   # override any entry here as needed
#
# NOTE: the thermionic term is expected to need a weight well above 1.0 for
# this device, for the same reason W_CONT_P does. Its natural scale is set by
# the injected densities: with a 0.6 eV hole barrier, p_inj_hat ~ 8.6e-11, so
# the hole flux balance has magnitude ~s0p_hat*p_inj_hat ~ 5e-10 and its
# square ~2e-19 -- invisible in a sum whose other terms are ~1e-2. The
# electron side is smaller still (a 2.0 eV barrier gives n_inj_hat ~ 2.8e-34).
# Left at 1.0 here because the right value should be measured against the
# DEVSIM reference rather than guessed; see Models_neural.md.
# LOSS_WEIGHTS["thermionic"] = 1.0

# Collocation sampling distribution: symmetric Beta(a, a) on [0, 1]. a = 1
# recovers uniform sampling; a < 1 clusters points at both contacts.
BETA_CONCENTRATION = 1.0

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

print()
print("Thermionic injection (top contact):")
print("  Coulomb radius r_c   = {0:.4e} cm  ({1:.4f} nm)".format(
    SCALING["r_c"], SCALING["r_c"] * 1e7))
print("  r_c/ell              = {0:.4e}   (f = this * |phi_hat'|)".format(
    SCALING["r_c_over_ell"]))
print("  S0n, S0p             = {0:.4e}, {1:.4e} cm/s".format(
    SCALING["s0n"], SCALING["s0p"]))
print("  S0n_hat, S0p_hat     = {0:.4e}, {1:.4e}".format(
    SCALING["s0n_hat"], SCALING["s0p_hat"]))
print("  barriers phi_n, phi_p= {0:.3f}, {1:.3f} eV".format(PHI_N_TOP, PHI_P_TOP))
# Zero-field injection densities: N*exp(-phi/Ut), the values an Ohmic contact
# of the same work function would pin to. The field-dependent model raises
# these by exp(sqrt(f))/(S(E)/S(0)) under bias.
print("  n_inj(f=0)           = {0:.4e} cm^-3".format(
    NC300 * np.exp(-PHI_N_TOP / SCALING["Ut"])))
print("  p_inj(f=0)           = {0:.4e} cm^-3".format(
    NV300 * np.exp(-PHI_P_TOP / SCALING["Ut"])))

# ============================================================
# Reference data + boundary conditions
# ============================================================

if not os.path.exists(REFERENCE_NPZ):
    raise SystemExit(
        "Missing {0}.\nRun `python devsim_reference_oled2.py` (from {1}) first "
        "to generate the DEVSIM reference solution.".format(
            REFERENCE_NPZ, _TEST_DIR)
    )

_ref = np.load(REFERENCE_NPZ)
REF_X_NM      = _ref["x_nm"]
REF_POTENTIAL = _ref["potential"]        # V
REF_ELECTRONS = _ref["electrons"]        # cm^-3
REF_HOLES     = _ref["holes"]            # cm^-3
REF_BIAS      = float(_ref["bias"])
VBI           = float(_ref["built_in_voltage"])

# Scaled reference arrays. Used only for boundary values and for scoring the
# trained networks -- never as an interior training target.
Ut = SCALING["Ut"]
REF_X_HAT   = REF_X_NM / (ELL * 1e7)          # nm -> scaled [0, 1]
REF_PHI_HAT = REF_POTENTIAL / Ut              # V  -> scaled by Ut
REF_N_HAT   = REF_ELECTRONS / C_TILDE         # cm^-3 -> scaled by C_tilde
REF_P_HAT   = REF_HOLES / C_TILDE

# Dirichlet contact values, taken from the reference solution at the two
# contact nodes (applied bias at the anode, 0 at the cathode). phi is pinned
# at both contacts; the density is pinned only at the Ohmic cathode, since the
# thermionic anode has no density pin at all.
phi_bc_left  = float(REF_PHI_HAT[0])     # x = 0   (top / anode, thermionic)
phi_bc_right = float(REF_PHI_HAT[-1])    # x = L   (bot / cathode, Ohmic)

n_bc_right = float(REF_N_HAT[-1])
p_bc_right = float(REF_P_HAT[-1])

print()
print("Reference from DEVSIM at {0:.2f} V (Vbi = {1:.2f} V)".format(REF_BIAS, VBI))
print("  phi(0) = {0:+.5f} V  -> phi_hat = {1:+.4f}".format(REF_POTENTIAL[0], phi_bc_left))
print("  phi(L) = {0:+.5f} V  -> phi_hat = {1:+.4f}".format(REF_POTENTIAL[-1], phi_bc_right))
print("  n range: {0:.3e} .. {1:.3e} cm^-3".format(REF_ELECTRONS.min(), REF_ELECTRONS.max()))
print("  p range: {0:.3e} .. {1:.3e} cm^-3".format(REF_HOLES.min(), REF_HOLES.max()))

print()
print("Density BCs:")
print("  x = 0 (anode, thermionic): n(0), p(0) both free -- set by the")
print("        injection flux balance, no Dirichlet pin of either carrier")
if MAJORITY_ONLY_BC:
    print("  x = L (cathode, Ohmic)   : n(L) = {0:.4e} cm^-3 pinned (majority);".format(
        REF_ELECTRONS[-1]))
    print("        p(L) left free -- discontinuous in the reference (see source)")
else:
    print("  x = L (cathode, Ohmic)   : n(L) = {0:.4e}, p(L) = {1:.4e} cm^-3 "
          "both pinned".format(REF_ELECTRONS[-1], REF_HOLES[-1]))


# ============================================================
# Build networks + problem
# ============================================================

# Log-space initialisation offsets, from the interior mean of the reference
# profile (an initialisation heuristic, not a training target); contact nodes
# are excluded since the Ohmic minority pin there is discontinuous.
_u_n_init = float(np.mean(-np.log(REF_N_HAT[1:-1])))
_u_p_init = float(np.mean(-np.log(REF_P_HAT[1:-1])))
print()
print("Log-space initialisation offsets (interior mean):")
print("  u_n offset = {0:+.4f}  -> n_hat ~ {1:.3e}".format(_u_n_init, np.exp(-_u_n_init)))
print("  u_p offset = {0:+.4f}  -> p_hat ~ {1:.3e}".format(_u_p_init, np.exp(-_u_p_init)))

phi_net, n_net, p_net = core.build_networks(
    width=WIDTH, depth=DEPTH, u_n_offset=_u_n_init, u_p_offset=_u_p_init,
    device=device,
)

print("Trainable parameters: {0}".format(
    sum(p.numel() for p in list(phi_net.parameters())
        + list(n_net.parameters()) + list(p_net.parameters()))))

problem = core.build_problem(
    phi_net=phi_net, n_net=n_net, p_net=p_net, scaling=SCALING,
    device=device, dtype=dtype,
    x_left=X_LEFT, x_right=X_RIGHT,
    phi_bc_left=phi_bc_left, phi_bc_right=phi_bc_right,
    majority_only_bc=MAJORITY_ONLY_BC,
    # Anode: thermionic, so no density pins here at all.
    thermionic_left=TOP_CONTACT,
    # Cathode: Ohmic, electrons are the majority carrier.
    u_n_bc_right=-np.log(n_bc_right),
    u_p_bc_right=-np.log(p_bc_right),    # minority pin, only used if
                                          # MAJORITY_ONLY_BC is False
    loss_weights=LOSS_WEIGHTS,
    beta_concentration=BETA_CONCENTRATION,
)


# ============================================================
# Train / evaluate / plot
# ============================================================

def train():
    return core.train(
        problem, epochs=EPOCHS, n_int=N_INT, lr=LEARNING_RATE,
        milestones=MILESTONES, gamma=GAMMA, log_every=LOG_EVERY,
        print_every=PRINT_EVERY, convergence_threshold=CONVERGENCE_THRESHOLD,
    )


def evaluate():
    return core.evaluate(
        problem, x_hat=REF_X_HAT, phi_true_V=REF_POTENTIAL,
        n_true=REF_ELECTRONS, p_true=REF_HOLES, bias=REF_BIAS,
    )


def report_currents():
    ref_x_cm = REF_X_NM * 1e-7
    return core.report_currents(
        problem, x_hat=REF_X_HAT, x_cm=ref_x_cm,
        ref_potential_V=REF_POTENTIAL, ref_electrons=REF_ELECTRONS,
        ref_holes=REF_HOLES,
    )


def report_thermionic():
    return core.report_thermionic(problem)


def plot(res, history_epochs, history_losses, filename=PLOT_PNG):
    return core.plot(res, history_epochs, history_losses, REF_X_NM, REF_BIAS, filename)


def main():
    history_epochs, history_losses = train()
    res = evaluate()
    report_currents()
    report_thermionic()
    try:
        plot(res, history_epochs, history_losses)
    except ImportError:
        pass


if __name__ == "__main__":
    # Note: training is not run automatically on import; main() (and thus
    # train()) only executes when this script is run directly.
    main()