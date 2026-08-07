"""
forward_demo.py

Coupled drift-diffusion PINN (DDNet) for a 100 nm organic active layer with
Ohmic contacts, validated against the DEVSIM drift-diffusion solver at 2.5 V.

This is the full three-network DDNet of Fig. 1 of DDNet.pdf: phi-Net, n-Net and
p-Net are trained *together* against the coupled Poisson + continuity system.
It supersedes the Poisson-only stage, in which n and p were interpolated from
the DEVSIM reference and only phi was learned; those profiles are now unknowns
produced by their own networks, and DEVSIM is used purely as an external
reference to score against, never as an input to training.

Physics used
------------
The stationary drift-diffusion system (DDNet Eq. 1), with C = 0 because the
organic layer is undoped -- the entire space charge is mobile carriers:

    eps * phi''   = q*(n - p - C),      C = 0
    Jn'           =  q*R
    Jp'           = -q*R
    Jn            =  q*mu_n*(Ut*n' - n*phi')
    Jp            = -q*mu_p*(Ut*p' + p*phi')

R is Langevin recombination, matching sim_dd's CreateOSLangevin exactly so the
PINN and DEVSIM cannot differ on the physics:

    R = gammar * (q/eps) * (n*p - nie^2) * (mu_n + mu_p)

Scaling follows DDNet Supplementary Table 1: lengths by a characteristic device
length ell, potential by the thermal voltage Ut, all densities by C_tilde, and
mobilities by mu_tilde = max(mu_n, mu_p). In scaled variables the system is

    lambda^2 * phi_hat'' = n_hat - p_hat
    Jn_hat'              =  R_hat
    Jp_hat'              = -R_hat
    Jn_hat               =  mu_n_hat*(n_hat' - n_hat*phi_hat')
    Jp_hat               = -mu_p_hat*(p_hat' + p_hat*phi_hat')

with lambda = L_D/ell and L_D = sqrt(eps*Ut/(q*C_tilde)). Note the scaled
current has no explicit Ut: the thermal voltage is absorbed by scaling phi by
Ut, which is exactly why the scaling is worth doing.

Logarithmic parametrisation
---------------------------
The carrier densities span ~26 decades, so the n-Net and p-Net do not output
densities directly. Following DDNet Sec. 4.2, each outputs the compressed
variable u = -log(density_hat) and the density is recovered through a hard
constraint,

    n_hat = exp(-u_n),   p_hat = exp(-u_p)

so positivity is structural rather than a penalty the optimiser has to learn,
and the network's own output stays O(1) across the whole range. DDNet's
Supplementary Fig. 2 shows the alternative -- a network outputting the density
directly -- captures only the largest 3-4 decades.

Device
------
The 100 nm organic layer of ``devsim_reference.py``: OLED1's contact set with
the bottom contact's electron density reduced from 1e25 to 1e17 cm^-3, which
widens the screening layer from 0.0008 nm to 7.6 nm and makes the potential
resolvable by a smooth network (see that module's docstring).

Boundary conditions
-------------------
phi is pinned at both contacts (Dirichlet), as before. The carrier densities
are pinned only where the pin is *representable*: at each Ohmic contact the
majority carrier is pinned and the minority carrier is left free. This is
forced by the reference data, not a modelling preference -- see MAJORITY_ONLY_BC
below for the measured justification.

Every phase below is numbered to match the walkthrough document
(DDNet_Forward_Walkthrough.md).
"""

import os

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

# CPU by default. MPS was benchmarked for the Poisson-only stage and measured at
# 95.0 s vs CPU's 99.0 s for the full 8000-epoch run -- a ~4% difference, i.e.
# no real speedup. Even with three networks the model is small (~37k parameters
# total) and the residuals need second derivatives through autograd via
# create_graph=True, so per-kernel launch overhead still dominates the actual
# compute. GPU dispatch currently costs about as much as it saves.
USE_GPU = False

# float64 throughout, which is a change from the Poisson-only stage.
#
# That stage could use float32 because the 26-decade carrier range lived only in
# the *inputs* and was differenced away immediately. That is no longer true: the
# densities are now unknowns, and the log-space variables reach u_p ~ 17.7 in
# the interior, so n_hat*p_hat in the Langevin term underflows and the residual
# loses all significance in float32. The log parametrisation keeps the network
# outputs O(1) but the residuals themselves are still evaluated on exponentials.
#
# float64 forces CPU: the MPS Metal backend does not implement float64 at all
# ("Cannot convert a MPS Tensor to float64 dtype"), so USE_GPU and this setting
# are mutually exclusive -- hence the guard below.
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

# Transport parameters. These are core.device.Device's defaults, which is what
# devsim_reference.py's make_device() leaves them at -- it overrides neither
# mobility nor the densities of states.
mu_n   = 1.0e-6              # electron mobility, cm^2/V-s
mu_p   = 1.0e-6              # hole mobility, cm^2/V-s
NC300  = 1.0e27              # conduction-band effective DOS, cm^-3
NV300  = 1.0e27              # valence-band effective DOS, cm^-3
GAMMAR = 1.0                 # Langevin prefactor ("gammar" in os_physics.py)

# HOMO/LUMO of the reference device, and the intrinsic density implied by them.
# NIE follows common_physics.py's CreateDensityOfStates:
#     NIE = sqrt(NC*NV) * exp(-EG/(2*Ut))
# at T = 300 K, where the temperature-dependent corrections (EGALPH, DEG) all
# vanish, so this Python expression reproduces DEVSIM's node model exactly.
LUMO   = -0.4
HOMO   = LUMO - 2.6
EG     = LUMO - HOMO         # 2.6 eV
nie    = np.sqrt(NC300 * NV300) * np.exp(-EG / (2.0 * Ut))

# Density scale. The bottom contact's electron density is the largest carrier
# density in the device, so it sets the screening length -- the natural choice
# of scale for this problem (there is no doping to scale by, C = 0). DDNet
# Supplementary Table 1 asks for C_tilde = O(max[C]); with no doping, the
# largest carrier density plays that role.
C_tilde = 1.0e17             # cm^-3

# Mobility scale, mu_tilde = max(mu_n, mu_p) per DDNet Supplementary Table 1.
# The two are equal here, so both scaled mobilities come out at exactly 1, but
# the division is kept explicit so unequal mobilities stay correct.
mu_tilde = max(mu_n, mu_p)
mu_n_hat = mu_n / mu_tilde
mu_p_hat = mu_p / mu_tilde

ell   = 100.0e-7                                    # device length, cm (100 nm)
lam_D = np.sqrt(eps_org * Ut / (q * C_tilde))       # Debye length, cm
lam   = lam_D / ell                                 # scaled Debye parameter

# Scaled Langevin prefactor. Physically R = gammar*(q/eps)*(n*p - nie^2)*(mu_n+mu_p).
# The continuity equations are scaled by the recombination scale
# R_tilde = mu_tilde*Ut*C_tilde/ell^2 (DDNet Supp. Table 1's lifetime scaling
# ell^2/(mu_tilde*Ut), inverted), and densities by C_tilde, so
#
#     R_hat = R/R_tilde
#           = [gammar*(q/eps)*(mu_n+mu_p)*C_tilde^2 / R_tilde] * (n_hat*p_hat - nie_hat^2)
#
# and the bracket is the constant computed here.
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
# The domain is scaled to x_hat in [0, 1] by the device length (not by L_D), so
# the geometry is fixed and lambda carries the ratio.
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

# Scaled reference arrays. Used ONLY for boundary values and for scoring the
# trained networks -- never as a training target for the interior, which is
# what makes this a forward PINN solve rather than a regression fit.
REF_X_HAT   = REF_X_NM / (ell * 1e7)          # nm -> scaled [0, 1]
REF_PHI_HAT = REF_POTENTIAL / Ut              # V  -> scaled by Ut
REF_N_HAT   = REF_ELECTRONS / C_tilde         # cm^-3 -> scaled by C_tilde
REF_P_HAT   = REF_HOLES / C_tilde

# Dirichlet contact values, taken from the DEVSIM solution at the two contact
# nodes. These are the conditions DEVSIM itself imposed (applied bias at the
# anode, 0 at the cathode; the Ohmic density pins from the Contact spec), so
# using them makes the two problems identical rather than merely similar.
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
# At each Ohmic contact DEVSIM pins BOTH carrier densities. Only the majority
# pin is representable by a smooth network; the minority pin is a genuine
# discontinuity in the reference data, not a resolvable boundary layer:
#
#   left  contact: n(0)   = 3.2e-10, but n(0.125 nm) = 4.8e14
#                  -- a 24-decade jump across one 0.125 nm mesh spacing.
#   right contact: p(L)   = 7.3e-30, while the interior trend heads toward
#                  ~1e9 (p(99.875 nm) = 2.0e9) -- a 38-decade jump across the
#                  final 0.125 nm.
#
# Excluding just those two nodes, the interior is smooth and well conditioned:
# -log(n_hat) spans [0.015, 5.34] and -log(p_hat) spans [10.8, 17.7]. This is
# the same pathology that motivated softening the bottom contact from 1e25 to
# 1e17 for phi, and it has the same resolution: imposing the literal minority
# pin would force the network to fit a step of tens of decades, which it cannot
# do, and the attempt corrupts the interior fit as the boundary term dominates
# the loss.
#
# So the majority carrier is pinned at each contact and the minority carrier is
# left free, determined by the continuity equations. Physically this is the
# right call: the minority density at an injecting contact is set by transport,
# and its precise value there is both irrelevant to the current (which the
# majority carrier carries) and unresolved by the mesh.
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
    """Fully-connected network with tanh activations: the DDNet backbone.

    Shared by all three subnetworks. DDNet uses the same depth (4 layers) and
    width (64 neurons) for phi-Net, n-Net and p-Net (Supp. Sec. 3), so the
    architecture is factored out here and the *output parametrisation* is what
    distinguishes them (see PhiNet and LogDensityNet below).

    tanh rather than ReLU is required, not merely preferred: the Poisson
    residual needs a *second* derivative through automatic differentiation, and
    ReLU's second derivative is identically zero.
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


class PhiNet(nn.Module):
    """phi-Net: the FCNN backbone, output taken directly as phi_hat.

    No log-space trick needed -- the potential has a modest dynamic range
    (~0.5 V here, i.e. phi_hat in [0, 19.3]), unlike n and p.
    """

    def __init__(self, width=64, depth=4):
        super().__init__()
        self.body = FCNN(width, depth)

    def forward(self, x):
        return self.body(x)


class LogDensityNet(nn.Module):
    """n-Net / p-Net: the FCNN backbone in logarithmic parametrisation.

    Implements DDNet Sec. 4.2. The network's raw output is the compressed
    variable

        u(x) = -log(density_hat)         [natural log]

    and the density is recovered through a hard constraint,

        density_hat = exp(-u)

    Two things this buys, both of which matter here:

    * **Dynamic range.** The densities span ~26 decades, which is ~60 in u.
      A network outputting the density directly would have to represent that
      range in its final linear layer; in u it is an O(10) output, well within
      what a tanh network resolves. DDNet's Supplementary Fig. 2 shows the
      direct parametrisation captures only the largest 3-4 decades.

    * **Positivity.** exp(-u) > 0 identically, so a negative density is not
      merely penalised but unrepresentable. That matters because the Langevin
      term is bilinear in n and p, and a transient negative density during
      early training would make the recombination term change sign.

    ``forward`` returns u; ``density`` applies the hard constraint. Callers that
    need both (the continuity residual needs the density and its derivative)
    should take the derivative of u and use the chain rule, which is what
    ``continuity_residuals`` does -- differentiating exp(-u) directly is
    algebraically identical but numerically worse, since it evaluates the
    exponential before differencing rather than after.
    """

    def __init__(self, width=64, depth=4, u_offset=0.0):
        super().__init__()
        self.body = FCNN(width, depth)
        # Additive offset on u, so the network starts near the right order of
        # magnitude instead of having to travel there from u ~ 0. Xavier init
        # gives an output near zero, i.e. density_hat ~ 1; for holes the true
        # interior value is ~e^-14, so without this the p-Net begins ~6 decades
        # too high and the Langevin term is correspondingly wrong at step 0.
        # This is a shift of the *output*, not a constraint on it -- the network
        # can move anywhere from there.
        self.register_buffer("u_offset", torch.tensor(float(u_offset)))

    def forward(self, x):
        """Return the compressed variable u = -log(density_hat)."""
        return self.body(x) + self.u_offset

    def density(self, x):
        """Return density_hat = exp(-u), the hard-constrained physical density."""
        return torch.exp(-self.forward(x))


# Offsets are set from the *interior* mean of the reference profile in log
# space. This uses the reference only to pick a starting point for the
# optimiser -- the same role as any initialisation heuristic -- not as a
# training target, and the contact nodes are excluded so the discontinuous
# minority pins do not skew it.
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
#
# The Poisson-only stage used 8192, which measured better than 2048 there
# (1.21% vs 1.74% relative L1). That rationale does not carry over: the gain
# came from resolving an *interpolated DEVSIM profile* used as a fixed source
# term, and the source is now a pair of smooth learned functions instead.
# Cost is linear in this number (measured: 0.094 / 0.181 / 0.297 s per epoch at
# 2048 / 4096 / 8192, three networks with second derivatives), so the halving
# is what makes a 20000-epoch run practical on CPU.
N_INT = 4096

# Loss weights.
#
# W_BC = 1.0 was measured in the Poisson-only stage: raising it made the solve
# markedly worse (1.74% -> 35.3% at 10, -> 64.7% at 100), because the Dirichlet
# values are O(19) in scaled units so the squared boundary term already starts
# ~1e2 and any upweighting starves the interior residual.
W_BC = 1.0

# The two continuity residuals get *separate* weights. Weighting them together
# was tried first and is wrong, for a reason worth recording since the global
# scales actively mislead here.
#
# Evaluated on the DEVSIM reference, the global scales are
#
#     Poisson source |n_hat - p_hat| ~ 2.6e-1
#     total current  |Jn_hat + Jp_hat| ~ 3.9
#     recombination  |R_hat| ~ 6.1e-4
#
# which suggests everything is already commensurate and no weight is needed.
# But that comparison is dominated by *electrons*. Near the cathode
# |Jp_hat|/|Jn_hat| ~ 1e-5, so the hole continuity residual sits at ~1e-5 while
# the total loss is ~2.5e-2 (Poisson ~2.3e-2). The hole equation is then
# invisible to the optimiser even when it is badly violated in relative terms.
#
# That this is a weighting problem and not missing physics was measured:
# perturbing p by 10x beyond 80 nm on the reference leaves the Poisson residual
# completely unchanged (9.998e-02 -> 9.998e-02, since p/(n-p) ~ 1e-6 there) but
# moves the hole continuity residual from 1.218e-05 to 3.374e-02, a ~2800x
# increase. The constraint exists; it is simply too small to matter in an
# unweighted sum.
#
# W_CONT_P lifts the hole equation to comparable footing. Without it the p-Net
# flattens near the cathode instead of following DEVSIM's roll-off (measured:
# 3.91e11 vs 1.65e10 at 99 nm).
W_CONT_N = 1.0
W_CONT_P = 1.0e3

# Weight on the current-continuity constraint (see current_constancy_residual).
W_JTOT = 1.0


def sample_interior(n):
    """Step 5: random collocation points in the domain (mesh-free).

    Sampled directly on the target device with torch.rand rather than drawn
    from numpy and copied across: this runs every epoch, so a host->device
    transfer here would be one of the larger costs in the loop.
    """
    x = torch.rand(n, 1, device=device) * (X_RIGHT - X_LEFT) + X_LEFT
    return x.requires_grad_(True)


def _d_dx(f, x):
    """First derivative df/dx via automatic differentiation.

    ``create_graph=True`` keeps the derivative itself differentiable, which is
    needed both for the second derivative in Poisson and for backpropagating
    through any residual that contains a derivative.
    """
    return torch.autograd.grad(
        f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]


def fields(x):
    """Evaluate all three networks and the derivatives the residuals need.

    Returns phi_hat, its first and second derivatives, the two densities, and
    the two density derivatives.

    The density derivatives come from the chain rule on the log variable,

        n_hat  = exp(-u_n)
        n_hat' = -u_n' * exp(-u_n) = -u_n' * n_hat

    rather than by differentiating exp(-u_n) directly. The two are
    algebraically identical, but this form keeps the large exponential factored
    out of the autograd graph: u_n' is O(1) and the exponential enters as a
    single multiplication, instead of the gradient of an exp() that can span
    tens of decades.
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
    """Step 6a: Poisson residual.

        lambda^2 * phi_hat'' - (n_hat - p_hat) = 0

    Sign convention: this is eps*phi'' = q*(n - p - C) scaled with C = 0, i.e.
    the same convention as sim_dd and DDNet Eq. 1, where a net *electron*
    excess gives positive curvature.
    """
    return lam ** 2 * d2phi - (n_hat - p_hat)


def scaled_currents(phi_d, n_hat, p_hat, dn, dp):
    """Scaled electron and hole current densities (DDNet Eq. 10).

        Jn_hat =  mu_n_hat*(n_hat' - n_hat*phi_hat')
        Jp_hat = -mu_p_hat*(p_hat' + p_hat*phi_hat')

    The thermal voltage that appears in the unscaled currents
    (Jn = q*mu_n*(Ut*n' - n*phi')) is absent here because phi is scaled by Ut,
    which is precisely the point of that scaling choice.
    """
    Jn = mu_n_hat * (dn - n_hat * phi_d)
    Jp = -mu_p_hat * (dp + p_hat * phi_d)
    return Jn, Jp


def langevin_recombination(n_hat, p_hat):
    """Scaled Langevin net recombination rate.

    Mirrors sim_dd/devsim_backend/os_physics.py's CreateLangevin,

        ULANG = gammar * (q/eps) * (n*p - nie^2) * (mu_n + mu_p)

    with the constant folded into LANGEVIN_PREFACTOR by the scaling above, so
    the PINN and DEVSIM use the identical recombination model.

    nie_hat^2 is ~2.4e-24 here and is utterly negligible against n_hat*p_hat
    over most of the device, but it is kept because it is what makes R vanish at
    equilibrium -- dropping it would put a small spurious recombination
    everywhere the carriers are depleted.
    """
    return LANGEVIN_PREFACTOR * (n_hat * p_hat - nie_hat ** 2)


def continuity_residuals(x, phi_d, n_hat, p_hat, dn, dp):
    """Step 6b: the two carrier continuity residuals.

        Jn_hat' - R_hat = 0
        Jp_hat' + R_hat = 0

    Note the opposite signs: a recombination event removes one electron and one
    hole, so it is a sink for both, and the sign difference reflects the
    opposite charges carried (DDNet Eq. 1: div Jn = qR, div Jp = -qR).
    """
    Jn, Jp = scaled_currents(phi_d, n_hat, p_hat, dn, dp)
    R = langevin_recombination(n_hat, p_hat)
    return _d_dx(Jn, x) - R, _d_dx(Jp, x) + R, Jn, Jp


# --- Boundary conditions ---
# The two contact points never change, so they are built once on the device
# rather than re-allocated every epoch.
_X_BC = torch.tensor([[X_LEFT], [X_RIGHT]], device=device)
_PHI_BC = torch.tensor([[phi_bc_left], [phi_bc_right]], device=device)

# Majority-carrier density pins, in *log* space, since that is the variable the
# networks actually output: matching u directly makes the boundary term
# commensurate with the network's own output scale, whereas matching the
# density would make it a comparison between numbers of order 1e-14.
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
    """Total-current constancy, as an explicit constraint.

    In 1D steady state with no external generation, adding the two continuity
    equations gives (Jn + Jp)' = 0 exactly: the total current is constant across
    the device. That is implied by the two residuals above, so this term is
    formally redundant -- but only formally.

    It is included because the two continuity residuals are *local* conditions
    that constrain the derivative of each current separately, and the quantity
    the device is actually characterised by is the single number Jn + Jp. A
    solution can have both continuity residuals small in a mean-squared sense
    while the total current still drifts across the domain, since nothing
    couples a residual at one collocation point to one at another. Penalising
    the spread of Jn + Jp directly ties the whole domain to one value.

    Implemented as the variance of Jn + Jp over the batch, which is the
    translation-invariant way to say "constant" without having to know what the
    constant is -- the current is an output of the solve, not an input.
    """
    Jtot = Jn + Jp
    return torch.mean((Jtot - Jtot.mean()) ** 2)


def total_loss():
    """Step 8: assemble the total loss.

    Returns the terms as *tensors*, not floats. Calling .item() forces a
    synchronisation with the accelerator, so doing it here would stall the
    pipeline every epoch; the training loop reads the values only on the epochs
    it actually prints or records.
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

    total = (L_poisson
             + W_CONT_N * L_cont_n
             + W_CONT_P * L_cont_p
             + W_JTOT * L_jtot
             + W_BC * L_bc)
    # The two continuity terms are reported separately as well as summed: they
    # differ by orders of magnitude (see W_CONT_P), so a combined figure would
    # just track the electron term and hide whether the hole equation is
    # actually being enforced.
    return total, L_poisson, L_cont_n, L_cont_p, L_jtot, L_bc


# ============================================================
# PHASE D -- Train (Steps 9-11)
# ============================================================

# DDNet trains for 40,000 epochs; the coupled system here needs materially more
# than the 8000 the Poisson-only stage used, since three networks must agree
# with each other rather than one network fitting a fixed source.
EPOCHS = 20000
MILESTONES = [5000, 10000, 14000, 17000]
optimizer = torch.optim.Adam(ALL_PARAMS, lr=1e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=MILESTONES, gamma=0.3)

# lr = 1e-3 rather than the Poisson-only stage's 1e-2. The coupled system is
# nonlinear -- the Langevin term is bilinear in n and p, and the drift term
# couples each density to phi' -- so the large steps that were safe when phi was
# the only unknown against a fixed source now let the densities overshoot in log
# space, where an overshoot of a few units is a few decades in the density.

# How often to copy the loss back from the accelerator. Every .item() is a
# synchronisation point, so reading the loss on every epoch would serialise the
# run against the host. The loss curve is smooth, so sampling loses nothing.
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
    # Evaluate on DEVSIM's own mesh, so the comparison is pointwise exact and
    # needs no interpolation of the reference.
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

    # Interior slice, excluding the two contact nodes. The minority pins there
    # are discontinuities the networks are deliberately not asked to fit (see
    # MAJORITY_ONLY_BC), so including them would score the model against a
    # target it was never given -- and since the error is measured in log space,
    # two points off by tens of decades would swamp the other 123.
    interior = slice(1, -1)

    rel_phi = _relative_L1(phi_pred_V, phi_true_V)
    # Densities are scored in log space, the variable the networks actually
    # learn. A linear-space L1 on a quantity spanning decades reports only how
    # well the largest values were fitted and says nothing about the rest --
    # exactly the failure mode DDNet's Supplementary Fig. 2 illustrates.
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

    The total current Jn + Jp should be independent of x (see
    current_constancy_residual). Its spread across the device is therefore a
    physics check that is completely independent of the DEVSIM comparison: it
    tests whether the learned solution is self-consistent, not whether it
    matches a reference.
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


def plot(res, history_epochs, history_losses, filename="forward_demo.png"):
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

    # (c) Carrier densities, now PREDICTED rather than supplied. Log axis: this
    # is the dynamic range the log parametrisation exists to capture, and a
    # linear axis would hide all of it (DDNet Supp. Fig. 2).
    #
    # The contact nodes are plotted as markers rather than joined to the curve:
    # the minority pins there are the discontinuities the networks are not asked
    # to fit, so drawing them as part of the reference line would suggest a
    # miss where there is deliberately no target.
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
