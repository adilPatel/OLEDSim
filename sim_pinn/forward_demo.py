"""
poisson_demo.py

Poisson-only PINN for a 100 nm organic active layer with Ohmic contacts,
validated against the DEVSIM drift-diffusion solver at 2.5 V.

Purpose: validate the phi-Net autodifferentiation + training-loop machinery in
isolation, BEFORE coupling in the carrier continuity equations for n and p.
This mirrors poisson_pinn_doped.py, but for the *undoped organic* device
rather than a doped silicon p-n junction, and with the carrier densities taken
from DEVSIM rather than eliminated by the depletion approximation.

Physics used
------------
Unlike the depletion approximation of poisson_pinn_doped.py -- where mobile
carriers are neglected and Poisson decouples into a *linear* problem with a
closed-form solution -- an undoped organic layer has no fixed dopant charge at
all. The space charge here is entirely mobile carriers:

    eps * phi'' = q*(n - p - C),      C = 0  (undoped)

So there is nothing to solve against unless n and p are supplied. Since the
n-Net and p-Net are to be provided separately, this demo takes n(x) and p(x)
from the converged DEVSIM solution at the same bias and treats them as fixed,
known source terms. That makes the Poisson problem linear in phi and gives an
unambiguous reference for phi: with the *exact* n and p imposed, a correct
phi-Net must reproduce DEVSIM's potential.

    lambda^2 * phi_hat''(x_hat) = n_hat - p_hat        (scaled, C = 0)

Scaling follows Models_neural.md: lengths by the extrinsic Debye length,
potential by the thermal voltage,

    x_hat = x/L_D,   L_D = sqrt(eps*Ut/(q*N)),   phi_hat = phi/Ut

Device
------
The 100 nm organic layer of ``devsim_reference.py``: OLED1's contact set with
the bottom contact's electron density reduced from 1e25 to 1e17 cm^-3, which
widens the screening layer from 0.0008 nm to 7.6 nm and makes the potential
resolvable by a smooth network (see that module's docstring).

Both contacts are Ohmic, so phi is pinned at both ends (Dirichlet). The DEVSIM
sweep runs to 2.5 V against a 2.0 V built-in voltage, so the net drop across
the device is 0.5 V.

Every phase below is numbered to match the walkthrough document
(DDNet_Forward_Walkthrough.md), restricted to phi-Net only.
"""

import os

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

# CPU by default. MPS was benchmarked (see Models_neural.md) and measured at
# 95.0 s vs CPU's 99.0 s for the full 8000-epoch run -- a ~4% difference, i.e.
# no real speedup. The network here (4 layers x 64 units, ~12k parameters) is
# too small to saturate a GPU, and the second derivative required by the
# Poisson residual doubles the number of autograd kernels via
# create_graph=True, so the per-launch overhead dominates over the actual
# compute on every step. GPU dispatch overhead currently costs about as much
# as it saves.
#
# Set USE_GPU = True to opt in (e.g. once the n-Net/p-Net make the model and
# the collocation batch large enough for a GPU to help).
USE_GPU = False

# Single precision is required on MPS -- the Metal backend does not implement
# float64 at all ("Cannot convert a MPS Tensor to float64 dtype") -- and is
# kept even on CPU so switching USE_GPU does not also change numerical
# behaviour. It is safe for *this* problem despite the carrier densities
# spanning ~26 decades, because the wide range lives in the inputs and is
# differenced away immediately. The scaled unknowns are all modest: phi_hat
# runs 0..19.3 and the source rho_hat = n_hat - p_hat runs -2e-5..1. Although
# p_hat's minimum (7.3e-47) does underflow float32, p is negligible against n
# wherever n dominates, so the difference that actually enters Poisson is
# unaffected: measured max relative error in rho_hat from float32 is 8.9e-08.
#
# This will NOT remain true once the n-Net and p-Net are added, since the
# densities themselves become unknowns spanning those 26 decades. At that
# point either the logarithmic parametrisation absorbs the range (DDNet's
# approach, which keeps the *network* outputs O(1) and would stay float32-
# safe) or the model has to move back to float64, which forces CPU.
torch.set_default_dtype(torch.float32)

if not USE_GPU:
    device = torch.device("cpu")
else:
    _accelerator = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else None
    device = torch.device(_accelerator.type) if _accelerator is not None else torch.device("cpu")
print("Using device:", device)

_HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_NPZ = os.path.join(_HERE, "devsim_reference_2.5V.npz")


# ============================================================
# PHASE A -- Problem setup (Steps 1-2)
# ============================================================

# --- Step 1: nondimensionalization / scaling constants ---
# Physical constants, matching sim_dd/devsim_backend/common_physics.py
# (SetUniversalParameters) so the PINN and DEVSIM cannot differ on constants.
q       = 1.6e-19            # C
k_B     = 1.3806503e-23      # J/K
eps_0   = 8.85e-14           # F/cm
T       = 300.0              # K
eps_r   = 4.0                # OLED1's relative permittivity (Device default)
eps_org = eps_r * eps_0
Ut      = k_B * T / q        # thermal voltage, V (~0.02589)

# Density scale. The bottom contact's electron density is the largest carrier
# density in the device, so it sets the screening length -- the natural choice
# of scale for this problem (there is no doping to scale by, C = 0).
C_tilde = 1.0e17             # cm^-3

ell   = 100.0e-7                                    # device length, cm (100 nm)
lam_D = np.sqrt(eps_org * Ut / (q * C_tilde))       # Debye length, cm
lam   = lam_D / ell                                 # scaled Debye parameter

print("Thermal voltage Ut     = {0:.6f} V".format(Ut))
print("Debye length lambda_D  = {0:.4e} cm  ({1:.4f} nm)".format(lam_D, lam_D * 1e7))
print("Scaled parameter lambda= {0:.4e}".format(lam))
print("Scaled length L/L_D    = {0:.4f}".format(1.0 / lam))


# --- Step 2: geometry, source term, boundary conditions ---
# The domain is scaled to x_hat in [0, 1] by the device length (not by L_D), so
# the geometry is fixed and lambda carries the ratio. This matches the
# convention of poisson_pinn_doped.py, where x_hat spans [-1, 1] via ell.
X_LEFT, X_RIGHT = 0.0, 1.0

if not os.path.exists(REFERENCE_NPZ):
    raise SystemExit(
        "Missing {0}.\nRun `python devsim_reference.py` first to generate the "
        "DEVSIM reference solution.".format(os.path.basename(REFERENCE_NPZ))
    )

_ref = np.load(REFERENCE_NPZ)
REF_X_NM      = _ref["x_nm"]
REF_POTENTIAL = _ref["potential"]        # V
REF_ELECTRONS = _ref["electrons"]        # cm^-3
REF_HOLES     = _ref["holes"]            # cm^-3
REF_BIAS      = float(_ref["bias"])
VBI           = float(_ref["built_in_voltage"])

# Scaled reference arrays.
REF_X_HAT   = REF_X_NM / (ell * 1e7)          # nm -> scaled [0, 1]
REF_PHI_HAT = REF_POTENTIAL / Ut              # V  -> scaled by Ut
REF_N_HAT   = REF_ELECTRONS / C_tilde         # cm^-3 -> scaled by C_tilde
REF_P_HAT   = REF_HOLES / C_tilde

# Dirichlet (Ohmic) contact values for phi_hat, taken from the DEVSIM solution
# at the two contact nodes. These are the boundary conditions DEVSIM itself
# imposed (applied bias at the anode, 0 at the cathode), so using them makes
# the two problems identical rather than merely similar.
phi_bc_left  = float(REF_PHI_HAT[0])     # x = 0   (top / anode)
phi_bc_right = float(REF_PHI_HAT[-1])    # x = L   (bot / cathode)

print()
print("Reference from DEVSIM at {0:.2f} V (Vbi = {1:.2f} V)".format(REF_BIAS, VBI))
print("  phi(0) = {0:+.5f} V  -> phi_hat = {1:+.4f}".format(REF_POTENTIAL[0], phi_bc_left))
print("  phi(L) = {0:+.5f} V  -> phi_hat = {1:+.4f}".format(REF_POTENTIAL[-1], phi_bc_right))
print("  n range: {0:.3e} .. {1:.3e} cm^-3".format(REF_ELECTRONS.min(), REF_ELECTRONS.max()))
print("  p range: {0:.3e} .. {1:.3e} cm^-3".format(REF_HOLES.min(), REF_HOLES.max()))


# Device-resident copies of the reference profiles, built once. These are read
# on every training step, so uploading them here rather than converting from
# numpy inside the loop avoids 8000 host->device copies -- which on MPS would
# cost more than the kernels they feed.
_XP = torch.as_tensor(REF_X_HAT, dtype=torch.get_default_dtype(), device=device)
_FP_N = torch.as_tensor(REF_N_HAT, dtype=torch.get_default_dtype(), device=device)
_FP_P = torch.as_tensor(REF_P_HAT, dtype=torch.get_default_dtype(), device=device)


def _interp_tensor(x, xp, fp):
    """Piecewise-linear interpolation of a reference profile at points ``x``.

    The DEVSIM solution lives on its own non-uniform 125-node mesh, while the
    PINN samples collocation points anywhere in the domain, so the source term
    must be interpolated. Implemented with torch.searchsorted (rather than
    np.interp) so it stays on the accelerator and works directly on the
    sampled tensors; no gradient is needed through it, since n and p are fixed
    data here, not unknowns.

    ``xp``/``fp`` are the pre-uploaded device tensors above.
    """
    flat = x.reshape(-1)
    # Index of the interval containing each point, clamped to stay in range.
    idx = torch.searchsorted(xp, flat.contiguous())
    idx = idx.clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    weight = (flat - x0) / (x1 - x0)
    return (f0 + weight * (f1 - f0)).reshape(x.shape)


def charge_source(x):
    """Scaled net charge (n_hat - p_hat) at ``x``, from the DEVSIM solution.

    This is the right-hand side of the scaled Poisson equation. The doping
    term is absent because the layer is undoped (C = 0), so the entire space
    charge is mobile carriers -- the defining feature of an organic diode as
    opposed to the doped junction of poisson_pinn_doped.py.

    Interpolated in *linear* space rather than log space: it is the difference
    n - p that enters Poisson, and that difference changes sign near the anode
    (where holes dominate), which a log interpolation cannot represent.
    """
    n_hat = _interp_tensor(x, _XP, _FP_N)
    p_hat = _interp_tensor(x, _XP, _FP_P)
    return n_hat - p_hat


# ============================================================
# PHASE B -- Build the ansatz (Steps 3-4)
# ============================================================

class PhiNet(nn.Module):
    """phi-Net: plain FCNN, tanh activations.

    No log-space trick needed -- the potential has a modest dynamic range
    (~0.5 V here), unlike n and p, which span 26 decades and will need the
    logarithmic parametrisation when their networks are added.

    tanh rather than ReLU is required, not merely preferred: the residual
    needs a *second* derivative through automatic differentiation, and ReLU's
    second derivative is identically zero.
    """

    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        # Step 4: initialization -- Xavier/Glorot, standard and automatic
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Map [0, 1] -> [-1, 1] before the first layer: tanh is centred there,
        # and an input range of [0, 1] wastes half its dynamic range.
        return self.net(2.0 * x - 1.0)


phi_net = PhiNet().to(device)


# ============================================================
# PHASE C -- Encode the physics as a loss (Steps 5-8)
# ============================================================

# Interior collocation points. 8192 rather than poisson_pinn_doped's 2048:
# measured, 2048 -> 1.74% relative L1 and 8192 -> 1.21% under otherwise
# identical settings. The source term here is a *interpolated DEVSIM profile*
# rather than a smooth analytic tanh, so denser sampling resolves it better.
N_INT = 8192
N_BC  = 2      # boundary points: just the two contacts in 1D

# Weight on the boundary term, relative to the interior residual.
#
# 1.0 is not a placeholder -- it was measured. Raising it makes the solve
# markedly worse, because the Dirichlet values are O(19) in scaled units
# (phi_hat = phi/Ut, and 0.5 V / 0.0259 V = 19.3), so the squared boundary
# term starts out ~1e2 and any upweighting lets it dominate the loss and
# starve the interior residual:
#
#     W_BC      relative L1
#     1         1.74 %
#     10       35.3  %
#     100      64.7  %
#     1000     78.6  %
#
# The hard-constraint alternative (multiplying the network output by a factor
# vanishing at both contacts) would remove the trade-off entirely and is the
# natural next step if the boundary error ever becomes the limit; here it is
# already ~1e-9, so it is not.
W_BC = 1.0


def sample_interior(n):
    """Step 5: random collocation points in the domain (mesh-free).

    Sampled directly on the target device with torch.rand rather than drawn
    from numpy and copied across: this runs every epoch, so a host->device
    transfer here would be one of the larger costs in the loop.
    """
    x = torch.rand(n, 1, device=device) * (X_RIGHT - X_LEFT) + X_LEFT
    return x.requires_grad_(True)


def poisson_residual(x):
    """Step 6: PDE residual via automatic differentiation.

        lambda^2 * phi_hat'' - (n_hat - p_hat) = 0

    Note the sign convention: this is eps*phi'' = q*(n - p - C) scaled, i.e.
    the same convention as sim_dd (and the DDNet paper's Eq. 1), where a net
    *electron* excess gives positive curvature. poisson_pinn_doped.py wrote
    lambda^2*phi'' = -C for the depletion case, which is the identical
    equation with n = p = 0 and C the fixed dopant charge.
    """
    phi = phi_net(x)
    dphi_dx = torch.autograd.grad(
        phi, x, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
    d2phi_dx2 = torch.autograd.grad(
        dphi_dx, x, grad_outputs=torch.ones_like(dphi_dx), create_graph=True)[0]
    rho = charge_source(x)
    return lam ** 2 * d2phi_dx2 - rho


# The two contact points and their pinned values never change, so they are
# built once on the device rather than re-allocated every epoch.
_X_BC = torch.tensor([[X_LEFT], [X_RIGHT]], device=device)
_PHI_BC = torch.tensor([[phi_bc_left], [phi_bc_right]], device=device)


def boundary_loss():
    """Step 7: Dirichlet (Ohmic) boundary residuals at the two contacts."""
    return torch.mean((phi_net(_X_BC) - _PHI_BC) ** 2)


def total_loss():
    """Step 8: assemble total loss = interior PDE MSE + boundary MSE.

    Returns the three terms as *tensors*, not floats. Calling .item() forces a
    synchronisation with the accelerator, so doing it here would stall the
    pipeline every epoch; the training loop reads the values only on the
    epochs it actually prints or records.
    """
    x_int = sample_interior(N_INT)
    r = poisson_residual(x_int)
    L_int = torch.mean(r ** 2)
    L_bc = boundary_loss()
    return L_int + W_BC * L_bc, L_int, L_bc


# ============================================================
# PHASE D -- Train (Steps 9-11)
# ============================================================

EPOCHS = 8000
MILESTONES = [2000, 4000, 5500, 7000]   # decay points, as in poisson_pinn_doped
optimizer = torch.optim.Adam(phi_net.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=MILESTONES, gamma=0.1)

RESAMPLE_EVERY = 100   # Step 10: resample collocation points periodically


# How often to copy the loss back from the accelerator. Every .item() is a
# synchronisation point that drains the command queue, so reading the loss on
# every one of 8000 epochs would serialise the whole run against the host. The
# loss curve is smooth, so sampling it every 10 epochs loses nothing visible.
LOG_EVERY = 10


def train():
    """Run the training loop, returning (epochs, losses) for the loss curve."""
    history_epochs = []
    history_losses = []
    device_losses = []          # kept on-device, drained in one sync at the end
    print("\nTraining Poisson-only PINN...")

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss, l_int, l_bc = total_loss()
        loss.backward()             # backprop through the AD graph (Step 10)
        optimizer.step()
        scheduler.step()

        if epoch % LOG_EVERY == 0 or epoch == EPOCHS - 1:
            # detach() keeps the value without holding the graph alive; the
            # actual host transfer is deferred to the single stack() below.
            history_epochs.append(epoch)
            device_losses.append(loss.detach())

        if epoch % 1000 == 0 or epoch == EPOCHS - 1:
            # These epochs sync anyway in order to print.
            print("  epoch {0:5d} | total loss {1:.3e} | interior {2:.3e} "
                  "| boundary {3:.3e} | lr {4:.1e}".format(
                      epoch, loss.item(), l_int.item(), l_bc.item(),
                      scheduler.get_last_lr()[0]))

            # Step 11: early stop on convergence threshold. Checked only on
            # printing epochs so it costs no extra synchronisation.
            if loss.item() < 1e-10:
                print("  Converged at epoch {0}, loss = {1:.3e}".format(
                    epoch, loss.item()))
                break

    # One transfer for the whole curve instead of thousands.
    history_losses = torch.stack(device_losses).cpu().numpy()
    return history_epochs, history_losses


# ============================================================
# PHASE E -- Evaluate (Step 12)
# ============================================================

def evaluate():
    """Compare the trained phi-Net against the DEVSIM potential."""
    # Evaluate on DEVSIM's own mesh, so the comparison is pointwise exact and
    # needs no interpolation of the reference.
    x_eval_t = torch.as_tensor(
        REF_X_HAT.reshape(-1, 1), dtype=torch.get_default_dtype(), device=device)
    with torch.no_grad():
        # .cpu() before .numpy(): a device tensor cannot be converted directly.
        phi_pred_hat = phi_net(x_eval_t).cpu().numpy().flatten()

    # Compare in float64 on the host. The network ran in float32 (an MPS
    # requirement), but the *comparison* against DEVSIM's float64 reference
    # costs nothing to do in full precision.
    phi_pred_V = phi_pred_hat.astype(np.float64) * Ut
    phi_true_V = REF_POTENTIAL

    # Relative L1 error, as used throughout the DDNet paper (Supp. Eq. 3).
    rel_L1 = np.sum(np.abs(phi_pred_V - phi_true_V)) / np.sum(np.abs(phi_true_V))
    max_abs = np.max(np.abs(phi_pred_V - phi_true_V))

    print()
    print("Comparison against DEVSIM at {0:.2f} V".format(REF_BIAS))
    print("  relative L1 error : {0:.4e}  ({1:.3f} %)".format(rel_L1, rel_L1 * 100))
    print("  max abs error     : {0:.4e} V".format(max_abs))
    print()
    print("Sample comparison (x [nm], PINN phi [V], DEVSIM phi [V]):")
    for i in range(0, len(REF_X_NM), max(1, len(REF_X_NM) // 12)):
        print("  x={0:7.2f}  phi_pinn={1:+.5f}  phi_devsim={2:+.5f}".format(
            REF_X_NM[i], phi_pred_V[i], phi_true_V[i]))

    return phi_pred_V, phi_true_V, rel_L1


def plot(phi_pred_V, phi_true_V, rel_L1, history_epochs, history_losses,
         filename="poisson_demo.png"):
    """Plot both potentials together, plus the carrier densities and loss."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) The requested comparison: both potentials on the same axes.
    ax = axes[0, 0]
    ax.plot(REF_X_NM, phi_true_V, "-", lw=3, alpha=0.45, color="tab:blue",
            label="DEVSIM (drift-diffusion)")
    ax.plot(REF_X_NM, phi_pred_V, "--", lw=1.8, color="tab:red",
            label="PINN ($\\phi$-Net)")
    ax.set_xlabel(r"$x$ [nm]")
    ax.set_ylabel(r"$\phi$ [V]")
    ax.set_title("(a) Electrostatic potential at {0:.1f} V "
                 "(rel. $L_1$ = {1:.2f}%)".format(REF_BIAS, rel_L1 * 100))
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Pointwise difference.
    ax = axes[0, 1]
    ax.plot(REF_X_NM, phi_pred_V - phi_true_V, color="tab:purple", lw=1.3)
    ax.set_xlabel(r"$x$ [nm]")
    ax.set_ylabel(r"$\phi_{\rm PINN} - \phi_{\rm DEVSIM}$ [V]")
    ax.set_title("(b) Pointwise difference")
    ax.grid(alpha=0.3)

    # (c) The DEVSIM carrier densities used as the source term. Log axis: they
    # span ~26 decades, which is the dynamic range the eventual n-Net/p-Net
    # will have to handle, and which a linear axis would hide entirely.
    ax = axes[1, 0]
    ax.semilogy(REF_X_NM, REF_ELECTRONS, "-", lw=2, color="tab:red", label="n (DEVSIM)")
    ax.semilogy(REF_X_NM, REF_HOLES, "-", lw=2, color="tab:green", label="p (DEVSIM)")
    ax.set_xlabel(r"$x$ [nm]")
    ax.set_ylabel(r"carrier density [cm$^{-3}$]")
    ax.set_title("(c) Source term: DEVSIM carrier densities")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # (d) Training loss. Sampled every LOG_EVERY epochs (see train()), so the
    # epoch numbers are carried explicitly rather than implied by the index.
    ax = axes[1, 1]
    ax.semilogy(history_epochs, history_losses, color="tab:orange", lw=1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\mathcal{L}_{\rm tot}$")
    ax.set_title("(d) Training loss")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print("\nSaved plot to {0}".format(filename))
    fig.show()
    return fig


def main():
    history_epochs, history_losses = train()
    phi_pred_V, phi_true_V, rel_L1 = evaluate()
    try:
        plot(phi_pred_V, phi_true_V, rel_L1, history_epochs, history_losses)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
