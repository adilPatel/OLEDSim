"""
forward_demo.py

Coupled drift-diffusion PINN (DDNet) for a 100 nm organic active layer with
Ohmic contacts, validated against the DEVSIM drift-diffusion solver at 2.5 V.

Three networks -- phi-Net, n-Net, p-Net -- trained together against the
coupled Poisson + continuity system (DDNet Fig. 1).

Physics
-------
    eps * phi''   = q*(n - p - C),      C = 0
    Jn'           =  q*R
    Jp'           = -q*R
    Jn            =  q*mu_n*(Ut*n' - n*phi')
    Jp            = -q*mu_p*(Ut*p' + p*phi')

R is Langevin recombination (matches sim_dd's CreateOSLangevin):

    R = gammar * (q/eps) * (n*p - nie^2) * (mu_n + mu_p)

Scaling (DDNet Supplementary Table 1): lengths by device length ell, potential
by thermal voltage Ut, densities by C_tilde, mobilities by
mu_tilde = max(mu_n, mu_p). Scaled system:

    lambda^2 * phi_hat'' = n_hat - p_hat
    Jn_hat'              =  R_hat
    Jp_hat'              = -R_hat
    Jn_hat               =  mu_n_hat*(n_hat' - n_hat*phi_hat')
    Jp_hat               = -mu_p_hat*(p_hat' + p_hat*phi_hat')

with lambda = L_D/ell and L_D = sqrt(eps*Ut/(q*C_tilde)).

Logarithmic parametrisation
---------------------------
n-Net and p-Net output the compressed variable u = -log(density_hat) rather
than density directly (DDNet Sec. 4.2); the density is recovered by the hard
constraint

    n_hat = exp(-u_n),   p_hat = exp(-u_p)

Device
------
The 100 nm organic layer of ``devsim_reference.py``: OLED1's contact set with
the bottom contact's electron density reduced from 1e25 to 1e17 cm^-3.

Boundary conditions
-------------------
phi is pinned at both contacts (Dirichlet). The majority carrier density is
pinned at each contact; the minority carrier is left free (see
MAJORITY_ONLY_BC).

Every phase below is numbered to match the walkthrough document
(DDNet_Forward_Walkthrough.md).
"""

import os

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

# CPU by default. 
USE_GPU = False

torch.set_default_dtype(torch.float64)

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
print("Using device:", device)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# DEVSIM reference + comparison plot live under tests/diode_pinn_1d/oled1,
# not next to this script (see devsim_reference_oled1.py in that directory).
_TEST_DIR = os.path.join(_ROOT, "tests", "diode_pinn_1d", "oled1")
REFERENCE_NPZ = os.path.join(_TEST_DIR, "oled1_devsim_reference_2.5V.npz")


# ============================================================
# PHASE A -- Problem setup (Steps 1-2)
# ============================================================

# --- Step 1: nondimensionalization / scaling constants ---
# Physical constants, matching sim_dd/devsim_backend/common_physics.py
# (SetUniversalParameters).
q       = 1.6e-19            # C
k_B     = 1.3806503e-23      # J/K
eps_0   = 8.85e-14           # F/cm
T       = 300.0              # K
eps_r   = 4.0                # OLED1's relative permittivity (Device default)
eps_org = eps_r * eps_0
Ut      = k_B * T / q        # thermal voltage, V (~0.02589)

# Transport parameters: core.device.Device's defaults (devsim_reference.py's
# make_device() overrides neither mobility nor the densities of states).
mu_n   = 1.0e-6              # electron mobility, cm^2/V-s
mu_p   = 1.0e-6              # hole mobility, cm^2/V-s
NC300  = 1.0e27              # conduction-band effective DOS, cm^-3
NV300  = 1.0e27              # valence-band effective DOS, cm^-3
GAMMAR = 1.0                 # Langevin prefactor ("gammar" in os_physics.py)

# HOMO/LUMO of the reference device, and the implied intrinsic density.
# Follows common_physics.py's CreateDensityOfStates at T = 300 K:
#     NIE = sqrt(NC*NV) * exp(-EG/(2*Ut))
LUMO   = -0.4
HOMO   = LUMO - 2.6
EG     = LUMO - HOMO         # 2.6 eV
nie    = np.sqrt(NC300 * NV300) * np.exp(-EG / (2.0 * Ut))

# Density scale: the bottom contact's electron density, the largest carrier
# density in the device.
C_tilde = 1.0e17             # cm^-3

# Mobility scale, mu_tilde = max(mu_n, mu_p) per DDNet Supplementary Table 1.
mu_tilde = max(mu_n, mu_p)
mu_n_hat = mu_n / mu_tilde
mu_p_hat = mu_p / mu_tilde

ell   = 100.0e-7                                    # device length, cm (100 nm)
lam_D = np.sqrt(eps_org * Ut / (q * C_tilde))       # Debye length, cm
lam   = lam_D / ell                                 # scaled Debye parameter

# Scaled Langevin prefactor. R = gammar*(q/eps)*(n*p - nie^2)*(mu_n+mu_p),
# scaled by R_tilde = mu_tilde*Ut*C_tilde/ell^2 and densities by C_tilde:
#
#     R_hat = [gammar*(q/eps)*(mu_n+mu_p)*C_tilde^2 / R_tilde] * (n_hat*p_hat - nie_hat^2)
nie_hat  = nie / C_tilde
R_tilde  = mu_tilde * Ut * C_tilde / (ell ** 2)
LANGEVIN_PREFACTOR = (
    GAMMAR * (q / eps_org) * (mu_n + mu_p) * (C_tilde ** 2) / R_tilde
)

print("Thermal voltage Ut     = {0:.6f} V".format(Ut))
print("Debye length lambda_D  = {0:.4e} cm  ({1:.4f} nm)".format(lam_D, lam_D * 1e7))
print("Scaled parameter lambda= {0:.4e}".format(lam))
print("Scaled length L/L_D    = {0:.4f}".format(1.0 / lam))
print("Intrinsic density nie  = {0:.4e} cm^-3  (nie_hat = {1:.4e})".format(nie, nie_hat))
print("Langevin prefactor     = {0:.4e}".format(LANGEVIN_PREFACTOR))


# --- Step 2: geometry, reference data, boundary conditions ---
# Domain scaled to x_hat in [0, 1] by the device length (not by L_D).
X_LEFT, X_RIGHT = 0.0, 1.0

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
REF_X_HAT   = REF_X_NM / (ell * 1e7)          # nm -> scaled [0, 1]
REF_PHI_HAT = REF_POTENTIAL / Ut              # V  -> scaled by Ut
REF_N_HAT   = REF_ELECTRONS / C_tilde         # cm^-3 -> scaled by C_tilde
REF_P_HAT   = REF_HOLES / C_tilde

# Dirichlet contact values, taken from the DEVSIM solution at the two contact
# nodes (applied bias at the anode, 0 at the cathode; Ohmic density pins from
# the Contact spec).
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


# ------------------------------------------------------------
# Which density boundary conditions are actually imposable
# ------------------------------------------------------------
# At each Ohmic contact DEVSIM pins BOTH carrier densities, but only the
# majority pin is representable by a smooth network -- the minority pin is a
# genuine discontinuity in the reference data. So the majority carrier is
# pinned at each contact and the minority carrier is left free, determined by
# the continuity equations.
MAJORITY_ONLY_BC = True

# Left contact (anode): holes are the majority carrier. Right contact
# (cathode): electrons are the majority carrier.
print()
if MAJORITY_ONLY_BC:
    print("Density BCs: majority carrier only")
    print("  p(0) = {0:.4e} cm^-3   (anode, majority)".format(REF_HOLES[0]))
    print("  n(L) = {0:.4e} cm^-3   (cathode, majority)".format(REF_ELECTRONS[-1]))
    print("  n(0), p(L) left free -- discontinuous in the reference (see source)")
else:
    print("Density BCs: both carriers pinned at both contacts")


# ============================================================
# PHASE B -- Build the ansatz (Steps 3-4)
# ============================================================

class FCNN(nn.Module):
    """Fully-connected tanh network: the shared DDNet backbone.

    Same depth (4 layers) and width (64 neurons) for phi-Net, n-Net and p-Net
    (Supp. Sec. 3); the output parametrisation is what distinguishes them (see
    PhiNet and LogDensityNet below). tanh is required rather than ReLU since
    the Poisson residual needs a second derivative through autodiff.
    """

    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        # Step 4: initialization -- Xavier/Glorot
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Map [0, 1] -> [-1, 1]: tanh is centred there.
        return self.net(2.0 * x - 1.0)


class PhiNet(nn.Module):
    """phi-Net: the FCNN backbone, output taken directly as phi_hat."""

    def __init__(self, width=64, depth=4):
        super().__init__()
        self.body = FCNN(width, depth)

    def forward(self, x):
        return self.body(x)


class LogDensityNet(nn.Module):
    """n-Net / p-Net: FCNN backbone in logarithmic parametrisation (DDNet Sec. 4.2).

    The network's raw output is the compressed variable

        u(x) = -log(density_hat)         [natural log]

    and the density is recovered through the hard constraint

        density_hat = exp(-u)

    This gives the O(10)-scale output room to represent densities spanning
    ~26 decades, and makes positivity structural (exp(-u) > 0 identically)
    rather than something the optimiser has to learn.

    ``forward`` returns u; ``density`` applies the hard constraint.
    ``continuity_residuals`` differentiates u and applies the chain rule
    rather than differentiating exp(-u) directly, for numerical stability.
    """

    def __init__(self, width=64, depth=4, u_offset=0.0):
        super().__init__()
        self.body = FCNN(width, depth)
        # Additive offset on u so the network starts near the right order of
        # magnitude rather than at density_hat ~ 1 (Xavier init's output ~ 0).
        # A shift of the output, not a constraint on it.
        self.register_buffer("u_offset", torch.tensor(float(u_offset)))

    def forward(self, x):
        """Return the compressed variable u = -log(density_hat)."""
        return self.body(x) + self.u_offset

    def density(self, x):
        """Return density_hat = exp(-u), the hard-constrained physical density."""
        return torch.exp(-self.forward(x))


# Offsets set from the interior mean of the reference profile in log space
# (an initialisation heuristic, not a training target); contact nodes are
# excluded since the minority pins there are discontinuous.
_u_n_init = float(np.mean(-np.log(REF_N_HAT[1:-1])))
_u_p_init = float(np.mean(-np.log(REF_P_HAT[1:-1])))
print()
print("Log-space initialisation offsets (interior mean):")
print("  u_n offset = {0:+.4f}  -> n_hat ~ {1:.3e}".format(_u_n_init, np.exp(-_u_n_init)))
print("  u_p offset = {0:+.4f}  -> p_hat ~ {1:.3e}".format(_u_p_init, np.exp(-_u_p_init)))

phi_net = PhiNet().to(device)
n_net   = LogDensityNet(u_offset=_u_n_init).to(device)
p_net   = LogDensityNet(u_offset=_u_p_init).to(device)

ALL_PARAMS = list(phi_net.parameters()) + list(n_net.parameters()) + list(p_net.parameters())
print("Trainable parameters: {0}".format(sum(p.numel() for p in ALL_PARAMS)))


# ============================================================
# PHASE C -- Encode the physics as a loss (Steps 5-8)
# ============================================================

# Interior collocation points: DDNet's own 2^12 = 4096 (Sec. 4.3).
N_INT = 4096

# Loss weights. The two continuity residuals get separate weights, and Poisson
# is weighted well above the rest since it is the hardest term to converge.
W_BC = 1.0
W_CONT_N = 1.0
W_CONT_P = 1.0e3
W_JTOT = 1.0            # current-continuity constraint (see current_constancy_residual)
W_POISSON = 100.0


# Collocation sampling distribution: symmetric Beta(a, a) on [0, 1].
# a = 1 recovers uniform sampling; a < 1 clusters points at both contacts.
BETA_CONCENTRATION = 1.0


def sample_interior(n):
    """Step 5: random collocation points in the domain (mesh-free).

    Drawn from a symmetric Beta(a, a) with a = BETA_CONCENTRATION. For a = 0.5
    this uses the closed-form arcsine substitution

        U ~ Uniform(0,1)  =>  sin^2(pi*U/2) ~ Beta(1/2, 1/2)

    which stays on-device and in float64; other a != 1 fall back to torch's
    Beta sampler.

    Samples are clamped strictly inside (0, 1) so they cannot land exactly on
    the endpoints, which are imposed separately as Dirichlet points in
    boundary_loss().
    """
    if BETA_CONCENTRATION == 1.0:
        u = torch.rand(n, 1, device=device)
    elif BETA_CONCENTRATION == 0.5:
        u = torch.sin(0.5 * np.pi * torch.rand(n, 1, device=device)) ** 2
    else:
        u = torch.distributions.Beta(
            torch.tensor(BETA_CONCENTRATION, device=device, dtype=torch.get_default_dtype()),
            torch.tensor(BETA_CONCENTRATION, device=device, dtype=torch.get_default_dtype()),
        ).sample((n, 1)).reshape(n, 1)

    eps = torch.finfo(torch.get_default_dtype()).eps
    u = u.clamp(eps, 1.0 - eps)

    x = u * (X_RIGHT - X_LEFT) + X_LEFT
    return x.requires_grad_(True)


def _d_dx(f, x):
    """First derivative df/dx via automatic differentiation.

    ``create_graph=True`` keeps the derivative differentiable, needed for the
    second derivative in Poisson and for backpropagating through it.
    """
    return torch.autograd.grad(
        f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]


def fields(x):
    """Evaluate all three networks and the derivatives the residuals need.

    Returns phi_hat, its first and second derivatives, the two densities, and
    the two density derivatives. Density derivatives use the chain rule on the
    log variable rather than differentiating exp(-u) directly:

        n_hat  = exp(-u_n)
        n_hat' = -u_n' * exp(-u_n) = -u_n' * n_hat
    """
    phi = phi_net(x)
    dphi = _d_dx(phi, x)
    d2phi = _d_dx(dphi, x)

    u_n = n_net(x)
    u_p = p_net(x)
    n_hat = torch.exp(-u_n)
    p_hat = torch.exp(-u_p)

    dn = -_d_dx(u_n, x) * n_hat
    dp = -_d_dx(u_p, x) * p_hat

    return phi, dphi, d2phi, n_hat, p_hat, dn, dp


def poisson_residual(d2phi, n_hat, p_hat):
    """Step 6a: Poisson residual, lambda^2 * phi_hat'' - (n_hat - p_hat) = 0.

    Sign convention matches eps*phi'' = q*(n - p - C) with C = 0 (sim_dd,
    DDNet Eq. 1): a net electron excess gives positive curvature.
    """
    return lam ** 2 * d2phi - (n_hat - p_hat)


def scaled_currents(phi_d, n_hat, p_hat, dn, dp):
    """Scaled electron and hole current densities (DDNet Eq. 10).

        Jn_hat =  mu_n_hat*(n_hat' - n_hat*phi_hat')
        Jp_hat = -mu_p_hat*(p_hat' + p_hat*phi_hat')

    Ut does not appear here (unlike the unscaled Jn = q*mu_n*(Ut*n' - n*phi'))
    because it is absorbed into the phi scaling.
    """
    Jn = mu_n_hat * (dn - n_hat * phi_d)
    Jp = -mu_p_hat * (dp + p_hat * phi_d)
    return Jn, Jp


def langevin_recombination(n_hat, p_hat):
    """Scaled Langevin recombination rate, mirroring
    sim_dd/devsim_backend/os_physics.py's CreateLangevin:

        ULANG = gammar * (q/eps) * (n*p - nie^2) * (mu_n + mu_p)

    with the constant folded into LANGEVIN_PREFACTOR. nie_hat^2 is kept even
    though it is negligible against n_hat*p_hat, since it is what makes R
    vanish at equilibrium.
    """
    return LANGEVIN_PREFACTOR * (n_hat * p_hat - nie_hat ** 2)


def continuity_residuals(x, phi_d, n_hat, p_hat, dn, dp):
    """Step 6b: the two carrier continuity residuals.

        Jn_hat' - R_hat = 0
        Jp_hat' + R_hat = 0

    Opposite signs: a recombination event is a sink for both carriers, but of
    opposite charge (DDNet Eq. 1: div Jn = qR, div Jp = -qR).
    """
    Jn, Jp = scaled_currents(phi_d, n_hat, p_hat, dn, dp)
    R = langevin_recombination(n_hat, p_hat)
    return _d_dx(Jn, x) - R, _d_dx(Jp, x) + R, Jn, Jp


# --- Boundary conditions ---
# Built once on the device since the two contact points never change.
_X_BC = torch.tensor([[X_LEFT], [X_RIGHT]], device=device)
_PHI_BC = torch.tensor([[phi_bc_left], [phi_bc_right]], device=device)

# Majority-carrier density pins, in log space (the variable the networks
# actually output).
_X_BC_LEFT = torch.tensor([[X_LEFT]], device=device)
_X_BC_RIGHT = torch.tensor([[X_RIGHT]], device=device)
_U_P_BC_LEFT = torch.tensor([[-np.log(p_bc_left)]], device=device)    # anode: holes
_U_N_BC_RIGHT = torch.tensor([[-np.log(n_bc_right)]], device=device)  # cathode: electrons

# The minority pins, kept for reference and for the MAJORITY_ONLY_BC=False path.
_U_N_BC_LEFT = torch.tensor([[-np.log(n_bc_left)]], device=device)
_U_P_BC_RIGHT = torch.tensor([[-np.log(p_bc_right)]], device=device)


def boundary_loss():
    """Step 7: Dirichlet boundary residuals at the two contacts.

    phi is pinned at both contacts. The densities are pinned per
    MAJORITY_ONLY_BC: the majority carrier at each contact always, the minority
    carrier only if the (unrepresentable) literal pins are requested.
    """
    L_phi = torch.mean((phi_net(_X_BC) - _PHI_BC) ** 2)

    L_dens = torch.mean((p_net(_X_BC_LEFT) - _U_P_BC_LEFT) ** 2) \
        + torch.mean((n_net(_X_BC_RIGHT) - _U_N_BC_RIGHT) ** 2)

    if not MAJORITY_ONLY_BC:
        L_dens = L_dens \
            + torch.mean((n_net(_X_BC_LEFT) - _U_N_BC_LEFT) ** 2) \
            + torch.mean((p_net(_X_BC_RIGHT) - _U_P_BC_RIGHT) ** 2)

    return L_phi + L_dens


def current_constancy_residual(Jn, Jp):
    """Total-current constancy: in 1D steady state, (Jn + Jp)' = 0 across the
    device. Implemented as the variance of Jn + Jp over the batch -- constant
    without needing to know the constant's value.
    """
    Jtot = Jn + Jp
    return torch.mean((Jtot - Jtot.mean()) ** 2)


def total_loss():
    """Step 8: assemble the total loss.

    Returns the terms as tensors, not floats -- .item() forces a
    synchronisation, so the training loop only calls it on epochs it prints.
    """
    x_int = sample_interior(N_INT)
    phi, dphi, d2phi, n_hat, p_hat, dn, dp = fields(x_int)

    r_poisson = poisson_residual(d2phi, n_hat, p_hat)
    r_n, r_p, Jn, Jp = continuity_residuals(x_int, dphi, n_hat, p_hat, dn, dp)

    L_poisson = torch.mean(r_poisson ** 2)
    L_cont_n = torch.mean(r_n ** 2)
    L_cont_p = torch.mean(r_p ** 2)
    L_jtot = current_constancy_residual(Jn, Jp)
    L_bc = boundary_loss()

    total = (W_POISSON * L_poisson
             + W_CONT_N * L_cont_n
             + W_CONT_P * L_cont_p
             + W_JTOT * L_jtot
             + W_BC * L_bc)
    # cont_n and cont_p reported separately as well as summed since they
    # differ by orders of magnitude (see W_CONT_P).
    return total, L_poisson, L_cont_n, L_cont_p, L_jtot, L_bc


# ============================================================
# PHASE D -- Train (Steps 9-11)
# ============================================================

# DDNet trains for 40,000 epochs.
EPOCHS = 20000
MILESTONES = [5000, 10000, 14000, 17000]
optimizer = torch.optim.Adam(ALL_PARAMS, lr=1e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=MILESTONES, gamma=0.3)

# How often to copy the loss back from the accelerator; every .item() is a
# synchronisation point, so this is sampled rather than read every epoch.
LOG_EVERY = 20


def train():
    """Run the training loop, returning (epochs, losses) for the loss curve."""
    history_epochs = []
    device_losses = []          # kept on-device, drained in one sync at the end
    print("\nTraining coupled drift-diffusion PINN (phi-Net + n-Net + p-Net)...")

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss, l_poi, l_cont_n, l_cont_p, l_jtot, l_bc = total_loss()
        loss.backward()             # backprop through the AD graph (Step 10)
        optimizer.step()
        scheduler.step()

        if epoch % LOG_EVERY == 0 or epoch == EPOCHS - 1:
            history_epochs.append(epoch)
            device_losses.append(loss.detach())

        if epoch % 2000 == 0 or epoch == EPOCHS - 1:
            print("  epoch {0:6d} | total {1:.3e} | poisson {2:.3e} | cont_n {3:.3e} "
                  "| cont_p {4:.3e} | Jtot {5:.3e} | bc {6:.3e} | lr {7:.1e}".format(
                      epoch, loss.item(), l_poi.item(), l_cont_n.item(),
                      l_cont_p.item(), l_jtot.item(), l_bc.item(),
                      scheduler.get_last_lr()[0]))

            # Step 11: early stop on convergence threshold. Checked only on
            # printing epochs so it costs no extra synchronisation.
            if loss.item() < 1e-10:
                print("  Converged at epoch {0}, loss = {1:.3e}".format(
                    epoch, loss.item()))
                break

    history_losses = torch.stack(device_losses).cpu().numpy()
    return history_epochs, history_losses


# ============================================================
# PHASE E -- Evaluate (Step 12)
# ============================================================

def _relative_L1(pred, true):
    """Relative L1 error, as used throughout the DDNet paper (Supp. Eq. 3)."""
    return np.sum(np.abs(pred - true)) / np.sum(np.abs(true))


def evaluate():
    """Compare all three trained networks against the DEVSIM solution."""
    # DEVSIM's own mesh, so the comparison is pointwise exact.
    x_eval_t = torch.as_tensor(
        REF_X_HAT.reshape(-1, 1), dtype=torch.get_default_dtype(), device=device)
    with torch.no_grad():
        phi_pred_hat = phi_net(x_eval_t).cpu().numpy().flatten()
        n_pred_hat = n_net.density(x_eval_t).cpu().numpy().flatten()
        p_pred_hat = p_net.density(x_eval_t).cpu().numpy().flatten()

    phi_pred_V = phi_pred_hat.astype(np.float64) * Ut
    n_pred = n_pred_hat.astype(np.float64) * C_tilde
    p_pred = p_pred_hat.astype(np.float64) * C_tilde

    phi_true_V = REF_POTENTIAL
    n_true = REF_ELECTRONS
    p_true = REF_HOLES

    # Interior slice, excluding the two contact nodes: the minority pins there
    # are discontinuities the networks are deliberately not fit to (see
    # MAJORITY_ONLY_BC).
    interior = slice(1, -1)

    rel_phi = _relative_L1(phi_pred_V, phi_true_V)
    # Densities scored in log space, the variable the networks actually learn.
    rel_n_log = _relative_L1(np.log10(n_pred[interior]), np.log10(n_true[interior]))
    rel_p_log = _relative_L1(np.log10(p_pred[interior]), np.log10(p_true[interior]))
    rel_n_lin = _relative_L1(n_pred[interior], n_true[interior])
    rel_p_lin = _relative_L1(p_pred[interior], p_true[interior])

    max_abs_phi = np.max(np.abs(phi_pred_V - phi_true_V))

    print()
    print("=" * 70)
    print("Comparison against DEVSIM at {0:.2f} V".format(REF_BIAS))
    print("=" * 70)
    print("  phi  relative L1        : {0:.4e}  ({1:.3f} %)".format(rel_phi, rel_phi * 100))
    print("  phi  max abs error      : {0:.4e} V".format(max_abs_phi))
    print("  n    relative L1 (log10): {0:.4e}  ({1:.3f} %)".format(rel_n_log, rel_n_log * 100))
    print("  p    relative L1 (log10): {0:.4e}  ({1:.3f} %)".format(rel_p_log, rel_p_log * 100))
    print("  n    relative L1 (lin)  : {0:.4e}  ({1:.3f} %)".format(rel_n_lin, rel_n_lin * 100))
    print("  p    relative L1 (lin)  : {0:.4e}  ({1:.3f} %)".format(rel_p_lin, rel_p_lin * 100))
    print("  (densities scored on interior nodes only; contact pins excluded)")

    print()
    print("Sample comparison:")
    print("  {0:>8s}  {1:>11s} {2:>11s}   {3:>10s} {4:>10s}   {5:>10s} {6:>10s}".format(
        "x [nm]", "phi_pinn", "phi_devsim", "n_pinn", "n_devsim", "p_pinn", "p_devsim"))
    for i in range(0, len(REF_X_NM), max(1, len(REF_X_NM) // 12)):
        print("  {0:8.2f}  {1:+11.5f} {2:+11.5f}   {3:10.3e} {4:10.3e}   {5:10.3e} {6:10.3e}".format(
            REF_X_NM[i], phi_pred_V[i], phi_true_V[i],
            n_pred[i], n_true[i], p_pred[i], p_true[i]))

    return {
        "phi_pred": phi_pred_V, "phi_true": phi_true_V,
        "n_pred": n_pred, "n_true": n_true,
        "p_pred": p_pred, "p_true": p_true,
        "rel_phi": rel_phi, "rel_n_log": rel_n_log, "rel_p_log": rel_p_log,
    }


def report_currents():
    """Report the predicted terminal current and how constant it is.

    Jn + Jp should be independent of x (see current_constancy_residual); its
    spread across the device is a self-consistency check independent of the
    DEVSIM comparison.
    """
    x_t = torch.as_tensor(
        REF_X_HAT.reshape(-1, 1), dtype=torch.get_default_dtype(), device=device
    ).requires_grad_(True)

    _, dphi, _, n_hat, p_hat, dn, dp = fields(x_t)
    Jn, Jp = scaled_currents(dphi, n_hat, p_hat, dn, dp)

    # Undo the current scaling: J = q*mu_tilde*Ut*C_tilde/ell * J_hat.
    J_scale = q * mu_tilde * Ut * C_tilde / ell
    Jn_np = Jn.detach().cpu().numpy().flatten() * J_scale
    Jp_np = Jp.detach().cpu().numpy().flatten() * J_scale
    Jtot = Jn_np + Jp_np

    # DEVSIM's own current, from its stored profiles, by the same formula.
    ref_x_cm = REF_X_NM * 1e-7
    dphi_ref = np.gradient(REF_POTENTIAL, ref_x_cm)
    Jn_ref = q * mu_n * (Ut * np.gradient(REF_ELECTRONS, ref_x_cm) - REF_ELECTRONS * dphi_ref)
    Jp_ref = -q * mu_p * (Ut * np.gradient(REF_HOLES, ref_x_cm) + REF_HOLES * dphi_ref)

    interior = slice(1, -1)
    print()
    print("Current density (A/cm^2):")
    print("  PINN   Jn+Jp : mean {0:+.4e}, spread {1:.3e} ({2:.2f} % of mean)".format(
        Jtot[interior].mean(), np.ptp(Jtot[interior]),
        100 * np.ptp(Jtot[interior]) / abs(Jtot[interior].mean())))
    print("  DEVSIM Jn+Jp : mean {0:+.4e}  (finite-differenced from stored profiles)".format(
        (Jn_ref + Jp_ref)[interior].mean()))

    return Jn_np, Jp_np, Jtot


_PLOT_PNG = os.path.join(_TEST_DIR, "oled1_forward_demo.png")


def plot(res, history_epochs, history_losses, filename=_PLOT_PNG):
    """Plot potential, both carrier densities, and the training loss."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Potential: PINN against DEVSIM.
    ax = axes[0, 0]
    ax.plot(REF_X_NM, res["phi_true"], "-", lw=3, alpha=0.45, color="tab:blue",
            label="DEVSIM (drift-diffusion)")
    ax.plot(REF_X_NM, res["phi_pred"], "--", lw=1.8, color="tab:red",
            label="PINN ($\\phi$-Net)")
    ax.set_xlabel(r"$x$ [nm]")
    ax.set_ylabel(r"$\phi$ [V]")
    ax.set_title("(a) Electrostatic potential at {0:.1f} V "
                 "(rel. $L_1$ = {1:.2f}%)".format(REF_BIAS, res["rel_phi"] * 100))
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Pointwise difference in phi.
    ax = axes[0, 1]
    ax.plot(REF_X_NM, res["phi_pred"] - res["phi_true"], color="tab:purple", lw=1.3)
    ax.set_xlabel(r"$x$ [nm]")
    ax.set_ylabel(r"$\phi_{\rm PINN} - \phi_{\rm DEVSIM}$ [V]")
    ax.set_title("(b) Pointwise difference")
    ax.grid(alpha=0.3)

    # (c) Predicted carrier densities, log axis. DEVSIM curves exclude the
    # contact nodes: the minority pins there are discontinuities the networks
    # are not asked to fit.
    ax = axes[1, 0]
    ax.semilogy(REF_X_NM[1:-1], res["n_true"][1:-1], "-", lw=3, alpha=0.45,
                color="tab:blue", label="n (DEVSIM)")
    ax.semilogy(REF_X_NM[1:-1], res["p_true"][1:-1], "-", lw=3, alpha=0.45,
                color="tab:green", label="p (DEVSIM)")
    ax.semilogy(REF_X_NM, res["n_pred"], "--", lw=1.6, color="tab:red",
                label="n (PINN, n-Net)")
    ax.semilogy(REF_X_NM, res["p_pred"], "--", lw=1.6, color="tab:orange",
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


def main():
    history_epochs, history_losses = train()
    res = evaluate()
    report_currents()
    try:
        plot(res, history_epochs, history_losses)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
