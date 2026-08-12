"""
ddnet.py

Reusable pieces of the coupled drift-diffusion PINN (DDNet): the network
architectures (phi-Net, n-Net, p-Net), the nondimensionalization/scaling
math, the physics residuals and loss assembly, collocation sampling, and the
train/evaluate/plot loop.

This module holds no device- or example-specific numbers (layer widths,
HOMO/LUMO, contact densities, loss weights, ...); those live in the example
script (e.g. oled1_forward.py) and are passed in via build_problem().

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

Boundary conditions
--------------------
phi is pinned at both contacts (Dirichlet). The majority carrier density is
pinned at each contact; the minority carrier is left free when
majority_only_bc is set (see PINNProblem.boundary_loss).
"""

import numpy as np
import torch
import torch.nn as nn

# Physical constants, matching sim_dd/devsim_backend/common_physics.py
# (SetUniversalParameters). Not device-specific, so they live here rather
# than in the example script.
Q     = 1.6e-19            # C
K_B   = 1.3806503e-23      # J/K
EPS_0 = 8.85e-14           # F/cm


# ============================================================
# Network architectures
# ============================================================

class FCNN(nn.Module):
    """Fully-connected tanh network: the shared DDNet backbone.

    Same depth/width for phi-Net, n-Net and p-Net (Supp. Sec. 3); the output
    parametrisation is what distinguishes them (see PhiNet and LogDensityNet
    below). tanh is required rather than ReLU since the Poisson residual
    needs a second derivative through autodiff.
    """

    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        # Xavier/Glorot initialization.
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


# ============================================================
# Scaling / nondimensionalization
# ============================================================

def compute_scaling(*, eps_r, T, mu_n, mu_p, homo, lumo, nc300, nv300,
                     gammar, c_tilde, ell):
    """Nondimensionalize the device parameters (DDNet Supplementary Table 1).

    Parameters
    ----------
    eps_r : relative permittivity of the organic layer.
    T : temperature, K.
    mu_n, mu_p : electron/hole mobilities, cm^2/V-s.
    homo, lumo : HOMO/LUMO levels, eV.
    nc300, nv300 : conduction-/valence-band effective density of states, cm^-3.
    gammar : Langevin recombination prefactor.
    c_tilde : density scale, cm^-3 (typically the largest contact density).
    ell : device length, cm.

    Returns a dict of derived scaling quantities (Ut, eps_org, nie, lambda,
    mobility ratios, the Langevin prefactor, etc.), used both to build the
    PINNProblem and to convert reference/predicted data to and from physical
    units.
    """
    eps_org = eps_r * EPS_0
    Ut = K_B * T / Q  # thermal voltage, V

    eg = lumo - homo
    nie = np.sqrt(nc300 * nv300) * np.exp(-eg / (2.0 * Ut))

    mu_tilde = max(mu_n, mu_p)
    mu_n_hat = mu_n / mu_tilde
    mu_p_hat = mu_p / mu_tilde

    lam_D = np.sqrt(eps_org * Ut / (Q * c_tilde))   # Debye length, cm
    lam = lam_D / ell                                # scaled Debye parameter

    nie_hat = nie / c_tilde
    R_tilde = mu_tilde * Ut * c_tilde / (ell ** 2)
    langevin_prefactor = gammar * (Q / eps_org) * (mu_n + mu_p) * (c_tilde ** 2) / R_tilde

    j_scale = Q * mu_tilde * Ut * c_tilde / ell

    return {
        "eps_org": eps_org,
        "Ut": Ut,
        "eg": eg,
        "nie": nie,
        "nie_hat": nie_hat,
        "mu_tilde": mu_tilde,
        "mu_n_hat": mu_n_hat,
        "mu_p_hat": mu_p_hat,
        "lam_D": lam_D,
        "lam": lam,
        "langevin_prefactor": langevin_prefactor,
        "j_scale": j_scale,
        "c_tilde": c_tilde,
        "ell": ell,
        "mu_n": mu_n,
        "mu_p": mu_p,
    }


# ============================================================
# Problem state: networks + scaling + boundary conditions
# ============================================================

class PINNProblem:
    """Bundles the three networks, scaling constants, and boundary data
    needed to evaluate the physics residuals and total loss.

    Not a `Device` abstraction -- just the state that the loss/training
    functions need, assembled once by build_problem() from parameters
    supplied by the calling script.
    """

    def __init__(self, *, phi_net, n_net, p_net, scaling, device, dtype,
                 x_left, x_right, phi_bc_left, phi_bc_right,
                 majority_only_bc, u_p_bc_left, u_n_bc_right,
                 u_n_bc_left=None, u_p_bc_right=None,
                 loss_weights=None, beta_concentration=1.0):
        self.phi_net = phi_net
        self.n_net = n_net
        self.p_net = p_net
        self.scaling = scaling
        self.device = device
        self.dtype = dtype

        self.x_left = x_left
        self.x_right = x_right
        self.majority_only_bc = majority_only_bc
        self.beta_concentration = beta_concentration

        self.loss_weights = dict(DEFAULT_LOSS_WEIGHTS)
        if loss_weights:
            self.loss_weights.update(loss_weights)

        self._x_bc = torch.tensor([[x_left], [x_right]], device=device, dtype=dtype)
        self._phi_bc = torch.tensor([[phi_bc_left], [phi_bc_right]], device=device, dtype=dtype)

        self._x_bc_left = torch.tensor([[x_left]], device=device, dtype=dtype)
        self._x_bc_right = torch.tensor([[x_right]], device=device, dtype=dtype)
        self._u_p_bc_left = torch.tensor([[u_p_bc_left]], device=device, dtype=dtype)
        self._u_n_bc_right = torch.tensor([[u_n_bc_right]], device=device, dtype=dtype)

        # Minority pins, only needed when majority_only_bc is False.
        self._u_n_bc_left = None if u_n_bc_left is None else torch.tensor(
            [[u_n_bc_left]], device=device, dtype=dtype)
        self._u_p_bc_right = None if u_p_bc_right is None else torch.tensor(
            [[u_p_bc_right]], device=device, dtype=dtype)

    @property
    def all_params(self):
        return (list(self.phi_net.parameters())
                + list(self.n_net.parameters())
                + list(self.p_net.parameters()))

    # --- collocation sampling ---

    def sample_interior(self, n):
        """Random collocation points in the domain (mesh-free).

        Drawn from a symmetric Beta(a, a) with a = self.beta_concentration.
        a = 1 recovers uniform sampling; a < 1 clusters points at both
        contacts. For a = 0.5 this uses the closed-form arcsine substitution

            U ~ Uniform(0,1)  =>  sin^2(pi*U/2) ~ Beta(1/2, 1/2)

        which stays on-device and in the working dtype; other a != 1 fall
        back to torch's Beta sampler.

        Samples are clamped strictly inside (0, 1) so they cannot land
        exactly on the endpoints, which are imposed separately as Dirichlet
        points in boundary_loss().
        """
        a = self.beta_concentration
        if a == 1.0:
            u = torch.rand(n, 1, device=self.device, dtype=self.dtype)
        elif a == 0.5:
            u = torch.sin(0.5 * np.pi * torch.rand(n, 1, device=self.device, dtype=self.dtype)) ** 2
        else:
            u = torch.distributions.Beta(
                torch.tensor(a, device=self.device, dtype=self.dtype),
                torch.tensor(a, device=self.device, dtype=self.dtype),
            ).sample((n, 1)).reshape(n, 1)

        eps = torch.finfo(self.dtype).eps
        u = u.clamp(eps, 1.0 - eps)

        x = u * (self.x_right - self.x_left) + self.x_left
        return x.requires_grad_(True)

    # --- fields and residuals ---

    def fields(self, x):
        """Evaluate all three networks and the derivatives the residuals need.

        Returns phi_hat, its first and second derivatives, the two densities,
        and the two density derivatives. Density derivatives use the chain
        rule on the log variable rather than differentiating exp(-u)
        directly:

            n_hat  = exp(-u_n)
            n_hat' = -u_n' * exp(-u_n) = -u_n' * n_hat
        """
        phi = self.phi_net(x)
        dphi = _d_dx(phi, x)
        d2phi = _d_dx(dphi, x)

        u_n = self.n_net(x)
        u_p = self.p_net(x)
        n_hat = torch.exp(-u_n)
        p_hat = torch.exp(-u_p)

        dn = -_d_dx(u_n, x) * n_hat
        dp = -_d_dx(u_p, x) * p_hat

        return phi, dphi, d2phi, n_hat, p_hat, dn, dp

    def poisson_residual(self, d2phi, n_hat, p_hat):
        """lambda^2 * phi_hat'' - (n_hat - p_hat) = 0.

        Sign convention matches eps*phi'' = q*(n - p - C) with C = 0 (sim_dd,
        DDNet Eq. 1): a net electron excess gives positive curvature.
        """
        lam = self.scaling["lam"]
        return lam ** 2 * d2phi - (n_hat - p_hat)

    def scaled_currents(self, phi_d, n_hat, p_hat, dn, dp):
        """Scaled electron and hole current densities (DDNet Eq. 10).

            Jn_hat =  mu_n_hat*(n_hat' - n_hat*phi_hat')
            Jp_hat = -mu_p_hat*(p_hat' + p_hat*phi_hat')

        Ut does not appear here (unlike the unscaled Jn = q*mu_n*(Ut*n' -
        n*phi')) because it is absorbed into the phi scaling.
        """
        mu_n_hat = self.scaling["mu_n_hat"]
        mu_p_hat = self.scaling["mu_p_hat"]
        Jn = mu_n_hat * (dn - n_hat * phi_d)
        Jp = -mu_p_hat * (dp + p_hat * phi_d)
        return Jn, Jp

    def langevin_recombination(self, n_hat, p_hat):
        """Scaled Langevin recombination rate, mirroring
        sim_dd/devsim_backend/os_physics.py's CreateLangevin:

            ULANG = gammar * (q/eps) * (n*p - nie^2) * (mu_n + mu_p)

        with the constant folded into scaling["langevin_prefactor"]. nie_hat^2
        is kept even though it is negligible against n_hat*p_hat, since it is
        what makes R vanish at equilibrium.
        """
        prefactor = self.scaling["langevin_prefactor"]
        nie_hat = self.scaling["nie_hat"]
        return prefactor * (n_hat * p_hat - nie_hat ** 2)

    def continuity_residuals(self, x, phi_d, n_hat, p_hat, dn, dp):
        """The two carrier continuity residuals.

            Jn_hat' - R_hat = 0
            Jp_hat' + R_hat = 0

        Opposite signs: a recombination event is a sink for both carriers,
        but of opposite charge (DDNet Eq. 1: div Jn = qR, div Jp = -qR).
        """
        Jn, Jp = self.scaled_currents(phi_d, n_hat, p_hat, dn, dp)
        R = self.langevin_recombination(n_hat, p_hat)
        return _d_dx(Jn, x) - R, _d_dx(Jp, x) + R, Jn, Jp

    def boundary_loss(self):
        """Dirichlet boundary residuals at the two contacts.

        phi is pinned at both contacts. The densities are pinned per
        majority_only_bc: the majority carrier at each contact always, the
        minority carrier only if the (unrepresentable) literal pins are
        requested.
        """
        L_phi = torch.mean((self.phi_net(self._x_bc) - self._phi_bc) ** 2)

        L_dens = torch.mean((self.p_net(self._x_bc_left) - self._u_p_bc_left) ** 2) \
            + torch.mean((self.n_net(self._x_bc_right) - self._u_n_bc_right) ** 2)

        if not self.majority_only_bc:
            L_dens = L_dens \
                + torch.mean((self.n_net(self._x_bc_left) - self._u_n_bc_left) ** 2) \
                + torch.mean((self.p_net(self._x_bc_right) - self._u_p_bc_right) ** 2)

        return L_phi + L_dens

    def current_constancy_residual(self, Jn, Jp):
        """Total-current constancy: in 1D steady state, (Jn + Jp)' = 0 across
        the device. Implemented as the variance of Jn + Jp over the batch --
        constant without needing to know the constant's value.
        """
        Jtot = Jn + Jp
        return torch.mean((Jtot - Jtot.mean()) ** 2)

    def total_loss(self, n_int):
        """Sample interior points and assemble the total physics-informed loss.

        Returns the terms as tensors, not floats -- .item() forces a
        synchronisation, so the training loop should only call it on epochs
        it prints.
        """
        x_int = self.sample_interior(n_int)
        phi, dphi, d2phi, n_hat, p_hat, dn, dp = self.fields(x_int)

        r_poisson = self.poisson_residual(d2phi, n_hat, p_hat)
        r_n, r_p, Jn, Jp = self.continuity_residuals(x_int, dphi, n_hat, p_hat, dn, dp)

        L_poisson = torch.mean(r_poisson ** 2)
        L_cont_n = torch.mean(r_n ** 2)
        L_cont_p = torch.mean(r_p ** 2)
        L_jtot = self.current_constancy_residual(Jn, Jp)
        L_bc = self.boundary_loss()

        w = self.loss_weights
        total = (w["poisson"] * L_poisson
                 + w["cont_n"] * L_cont_n
                 + w["cont_p"] * L_cont_p
                 + w["jtot"] * L_jtot
                 + w["bc"] * L_bc)
        # cont_n and cont_p reported separately as well as summed since they
        # can differ by orders of magnitude depending on the weights.
        return total, L_poisson, L_cont_n, L_cont_p, L_jtot, L_bc


# Default loss weights. The two continuity residuals get separate weights,
# and Poisson is typically weighted well above the rest since it is the
# hardest term to converge.
DEFAULT_LOSS_WEIGHTS = {
    "bc": 1.0,
    "cont_n": 1.0,
    "cont_p": 1.0e3,
    "jtot": 1.0,
    "poisson": 100.0,
}


def _d_dx(f, x):
    """First derivative df/dx via automatic differentiation.

    ``create_graph=True`` keeps the derivative differentiable, needed for the
    second derivative in Poisson and for backpropagating through it.
    """
    return torch.autograd.grad(
        f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]


# ============================================================
# Model construction
# ============================================================

def build_networks(*, width, depth, u_n_offset, p_hat_ref=None, n_hat_ref=None,
                    u_p_offset=None, device):
    """Construct phi-Net, n-Net and p-Net.

    u_n_offset / u_p_offset are the log-space initialisation offsets (an
    initialisation heuristic, not a training target): typically the interior
    mean of -log(density_hat) from a reference profile, excluding contact
    nodes where minority pins are discontinuous. Pass them directly, or
    derive them elsewhere and pass through.
    """
    phi_net = PhiNet(width=width, depth=depth).to(device)
    n_net = LogDensityNet(width=width, depth=depth, u_offset=u_n_offset).to(device)
    p_net = LogDensityNet(width=width, depth=depth, u_offset=u_p_offset).to(device)
    return phi_net, n_net, p_net


def build_problem(*, phi_net, n_net, p_net, scaling, device, dtype,
                   x_left, x_right, phi_bc_left, phi_bc_right,
                   majority_only_bc, u_p_bc_left, u_n_bc_right,
                   u_n_bc_left=None, u_p_bc_right=None,
                   loss_weights=None, beta_concentration=1.0):
    """Factory wrapping PINNProblem's constructor. Kept as a free function so
    call sites read as "build the problem" rather than instantiating a class
    directly, and to leave room for validation/derived setup later.
    """
    return PINNProblem(
        phi_net=phi_net, n_net=n_net, p_net=p_net, scaling=scaling,
        device=device, dtype=dtype, x_left=x_left, x_right=x_right,
        phi_bc_left=phi_bc_left, phi_bc_right=phi_bc_right,
        majority_only_bc=majority_only_bc,
        u_p_bc_left=u_p_bc_left, u_n_bc_right=u_n_bc_right,
        u_n_bc_left=u_n_bc_left, u_p_bc_right=u_p_bc_right,
        loss_weights=loss_weights, beta_concentration=beta_concentration,
    )


# ============================================================
# Training
# ============================================================

def train(problem, *, epochs, n_int, lr=1e-3, milestones=None, gamma=0.3,
          log_every=20, print_every=2000, convergence_threshold=1e-10):
    """Run the training loop, returning (epochs, losses) for the loss curve."""
    optimizer = torch.optim.Adam(problem.all_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones or [], gamma=gamma)

    history_epochs = []
    device_losses = []          # kept on-device, drained in one sync at the end
    print("\nTraining coupled drift-diffusion PINN (phi-Net + n-Net + p-Net)...")

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, l_poi, l_cont_n, l_cont_p, l_jtot, l_bc = problem.total_loss(n_int)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            history_epochs.append(epoch)
            device_losses.append(loss.detach())

        if epoch % print_every == 0 or epoch == epochs - 1:
            print("  epoch {0:6d} | total {1:.3e} | poisson {2:.3e} | cont_n {3:.3e} "
                  "| cont_p {4:.3e} | Jtot {5:.3e} | bc {6:.3e} | lr {7:.1e}".format(
                      epoch, loss.item(), l_poi.item(), l_cont_n.item(),
                      l_cont_p.item(), l_jtot.item(), l_bc.item(),
                      scheduler.get_last_lr()[0]))

            if loss.item() < convergence_threshold:
                print("  Converged at epoch {0}, loss = {1:.3e}".format(
                    epoch, loss.item()))
                break

    history_losses = torch.stack(device_losses).cpu().numpy()
    return history_epochs, history_losses


# ============================================================
# Evaluation
# ============================================================

def relative_L1(pred, true):
    """Relative L1 error, as used throughout the DDNet paper (Supp. Eq. 3)."""
    return np.sum(np.abs(pred - true)) / np.sum(np.abs(true))


def evaluate(problem, *, x_hat, phi_true_V, n_true, p_true, bias=None,
             interior=slice(1, -1)):
    """Compare the trained networks against a reference solution.

    x_hat : scaled evaluation coordinates, shape (N,).
    phi_true_V, n_true, p_true : reference potential (V) and densities
        (cm^-3) at those coordinates, in physical units.
    interior : slice excluding contact nodes, where minority-carrier pins may
        be discontinuities the networks are deliberately not fit to.
    """
    scaling = problem.scaling
    Ut = scaling["Ut"]
    c_tilde = scaling["c_tilde"]

    x_eval_t = torch.as_tensor(
        x_hat.reshape(-1, 1), dtype=problem.dtype, device=problem.device)
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

    Jn + Jp should be independent of x (see PINNProblem.current_constancy_residual);
    its spread across the device is a self-consistency check independent of
    any reference comparison.

    If x_cm and the reference profiles are supplied, also finite-differences
    the reference current for comparison.
    """
    x_t = torch.as_tensor(
        x_hat.reshape(-1, 1), dtype=problem.dtype, device=problem.device
    ).requires_grad_(True)

    _, dphi, _, n_hat, p_hat, dn, dp = problem.fields(x_t)
    Jn, Jp = problem.scaled_currents(dphi, n_hat, p_hat, dn, dp)

    j_scale = problem.scaling["j_scale"]
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


# ============================================================
# Plotting
# ============================================================

def plot(res, history_epochs, history_losses, x_nm, bias, filename):
    """Plot potential, both carrier densities, and the training loss."""
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

    # (c) Predicted carrier densities, log axis. Reference curves exclude the
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
