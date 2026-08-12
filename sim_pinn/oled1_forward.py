"""
oled1_forward.py

Example: coupled drift-diffusion PINN (DDNet) for OLED1's 100 nm organic
active layer with Ohmic contacts, validated against the DEVSIM
drift-diffusion solver at 2.5 V.

Three networks -- phi-Net, n-Net, p-Net -- trained together against the
coupled Poisson + continuity system (DDNet Fig. 1). The reusable model,
loss, scaling and training/eval/plot logic live in ddnet.py; this script
supplies OLED1's device parameters, boundary conditions, and run
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

Running this script builds the problem and prints the derived scaling, but
does not train (call train() / main() explicitly -- see the bottom of this
file).
"""

import os

import numpy as np
import torch

import ddnet

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
_ROOT = os.path.dirname(_HERE)

# DEVSIM reference + comparison plot live under tests/diode_pinn_1d/oled1,
# not next to this script (see devsim_reference_oled1.py in that directory).
_TEST_DIR = os.path.join(_ROOT, "tests", "diode_pinn_1d", "oled1")
REFERENCE_NPZ = os.path.join(_TEST_DIR, "oled1_devsim_reference_2.5V.npz")
PLOT_PNG = os.path.join(_TEST_DIR, "oled1_forward_demo.png")


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
# ddnet.compute_scaling, following common_physics.py's CreateDensityOfStates
# at T = 300 K: NIE = sqrt(NC*NV) * exp(-EG/(2*Ut))).
LUMO = -0.4
HOMO = LUMO - 2.6             # EG = 2.6 eV

# Density scale: the bottom contact's electron density, the largest carrier
# density in the device.
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

# Loss weights (see ddnet.DEFAULT_LOSS_WEIGHTS for the same defaults). The two
# continuity residuals get separate weights, and Poisson is weighted well
# above the rest since it is the hardest term to converge.
LOSS_WEIGHTS = dict(ddnet.DEFAULT_LOSS_WEIGHTS)
# LOSS_WEIGHTS["poisson"] = 100.0   # override any entry here as needed

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

SCALING = ddnet.compute_scaling(
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

if not os.path.exists(REFERENCE_NPZ):
    raise SystemExit(
        "Missing {0}.\nRun `python devsim_reference_oled1.py` (from {1}) first "
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
# contact nodes (applied bias at the anode, 0 at the cathode; Ohmic density
# pins from the Contact spec).
phi_bc_left  = float(REF_PHI_HAT[0])     # x = 0   (top / anode)
phi_bc_right = float(REF_PHI_HAT[-1])    # x = L   (bot / cathode)

n_bc_left  = float(REF_N_HAT[0])
n_bc_right = float(REF_N_HAT[-1])
p_bc_left  = float(REF_P_HAT[0])
p_bc_right = float(REF_P_HAT[-1])

print()
print("Reference from DEVSIM at {0:.2f} V (Vbi = {1:.2f} V)".format(REF_BIAS, VBI))
print("  phi(0) = {0:+.5f} V  -> phi_hat = {1:+.4f}".format(REF_POTENTIAL[0], phi_bc_left))
print("  phi(L) = {0:+.5f} V  -> phi_hat = {1:+.4f}".format(REF_POTENTIAL[-1], phi_bc_right))
print("  n range: {0:.3e} .. {1:.3e} cm^-3".format(REF_ELECTRONS.min(), REF_ELECTRONS.max()))
print("  p range: {0:.3e} .. {1:.3e} cm^-3".format(REF_HOLES.min(), REF_HOLES.max()))

print()
if MAJORITY_ONLY_BC:
    print("Density BCs: majority carrier only")
    print("  p(0) = {0:.4e} cm^-3   (anode, majority)".format(REF_HOLES[0]))
    print("  n(L) = {0:.4e} cm^-3   (cathode, majority)".format(REF_ELECTRONS[-1]))
    print("  n(0), p(L) left free -- discontinuous in the reference (see source)")
else:
    print("Density BCs: both carriers pinned at both contacts")


# ============================================================
# Build networks + problem
# ============================================================

# Log-space initialisation offsets, from the interior mean of the reference
# profile (an initialisation heuristic, not a training target); contact
# nodes are excluded since the minority pins there are discontinuous.
_u_n_init = float(np.mean(-np.log(REF_N_HAT[1:-1])))
_u_p_init = float(np.mean(-np.log(REF_P_HAT[1:-1])))
print()
print("Log-space initialisation offsets (interior mean):")
print("  u_n offset = {0:+.4f}  -> n_hat ~ {1:.3e}".format(_u_n_init, np.exp(-_u_n_init)))
print("  u_p offset = {0:+.4f}  -> p_hat ~ {1:.3e}".format(_u_p_init, np.exp(-_u_p_init)))

phi_net, n_net, p_net = ddnet.build_networks(
    width=WIDTH, depth=DEPTH, u_n_offset=_u_n_init, u_p_offset=_u_p_init,
    device=device,
)

print("Trainable parameters: {0}".format(
    sum(p.numel() for p in list(phi_net.parameters())
        + list(n_net.parameters()) + list(p_net.parameters()))))

problem = ddnet.build_problem(
    phi_net=phi_net, n_net=n_net, p_net=p_net, scaling=SCALING,
    device=device, dtype=dtype,
    x_left=X_LEFT, x_right=X_RIGHT,
    phi_bc_left=phi_bc_left, phi_bc_right=phi_bc_right,
    majority_only_bc=MAJORITY_ONLY_BC,
    u_p_bc_left=-np.log(p_bc_left),      # anode: holes (majority)
    u_n_bc_right=-np.log(n_bc_right),    # cathode: electrons (majority)
    u_n_bc_left=-np.log(n_bc_left),      # minority pin, only used if
    u_p_bc_right=-np.log(p_bc_right),    # MAJORITY_ONLY_BC is False
    loss_weights=LOSS_WEIGHTS,
    beta_concentration=BETA_CONCENTRATION,
)


# ============================================================
# Train / evaluate / plot
# ============================================================

def train():
    return ddnet.train(
        problem, epochs=EPOCHS, n_int=N_INT, lr=LEARNING_RATE,
        milestones=MILESTONES, gamma=GAMMA, log_every=LOG_EVERY,
        print_every=PRINT_EVERY, convergence_threshold=CONVERGENCE_THRESHOLD,
    )


def evaluate():
    return ddnet.evaluate(
        problem, x_hat=REF_X_HAT, phi_true_V=REF_POTENTIAL,
        n_true=REF_ELECTRONS, p_true=REF_HOLES, bias=REF_BIAS,
    )


def report_currents():
    ref_x_cm = REF_X_NM * 1e-7
    return ddnet.report_currents(
        problem, x_hat=REF_X_HAT, x_cm=ref_x_cm,
        ref_potential_V=REF_POTENTIAL, ref_electrons=REF_ELECTRONS,
        ref_holes=REF_HOLES,
    )


def plot(res, history_epochs, history_losses, filename=PLOT_PNG):
    return ddnet.plot(res, history_epochs, history_losses, REF_X_NM, REF_BIAS, filename)


def main():
    history_epochs, history_losses = train()
    res = evaluate()
    report_currents()
    try:
        plot(res, history_epochs, history_losses)
    except ImportError:
        pass


if __name__ == "__main__":
    # Note: training is not run automatically on import; main() (and thus
    # train()) only executes when this script is run directly.
    main()
