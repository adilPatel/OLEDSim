"""
pinn_dd.py

``PINNProblem``: the three networks, the scaling constants and the boundary
data, bundled into the object the residuals and the training loop act on.

Not a device abstraction -- just the state the loss needs, assembled once by
``build_problem()`` from parameters a device script supplies. The physics
lives in the modules this one calls: ``poisson`` for phi-Net and the Poisson
residual, ``densities`` for n-Net/p-Net and the transport residuals,
``boundaries`` for the thermionic contacts, ``loss_weights`` for the
multipliers.
"""

import numpy as np
import torch

from . import boundaries, densities, poisson
from .loss_weights import DEFAULT_LOSS_WEIGHTS, FixedWeights, LossWeights
from .networks import d_dx


class PINNProblem:
    """The coupled drift-diffusion PINN: phi-Net + n-Net + p-Net.

    Parameters
    ----------
    phi_net, n_net, p_net : the three subnetworks.
    scaling : the dict from ``compute_scaling``.
    device, dtype : torch placement for the sampled and boundary tensors.
    x_left, x_right : domain endpoints in scaled coordinates.
    phi_bc_left, phi_bc_right : Dirichlet potential values (scaled).
    majority_only_bc : pin only the majority carrier at each Ohmic contact.
        The minority pin is a genuine discontinuity in the reference data,
        which a smooth network cannot represent.
    u_p_bc_left, u_n_bc_right : majority pins, in u = -log(density_hat).
    u_n_bc_left, u_p_bc_right : minority pins, used only when
        majority_only_bc is False.
    thermionic_left, thermionic_right : optional ``ThermionicContact``. A side
        given one has its density pins dropped -- those densities are unknowns
        fixed by the injection flux balance instead.
    loss_weights : a ``LossWeights``, or a dict of overrides on
        DEFAULT_LOSS_WEIGHTS (wrapped in ``FixedWeights``).
    beta_concentration : shape of the collocation sampling distribution.
    normalize_poisson, poisson_residual_floor : passed to the Poisson residual.
    """

    def __init__(self, *, phi_net, n_net, p_net, scaling, device, dtype,
                 x_left, x_right, phi_bc_left, phi_bc_right,
                 majority_only_bc, u_p_bc_left=None, u_n_bc_right=None,
                 u_n_bc_left=None, u_p_bc_right=None,
                 thermionic_left=None, thermionic_right=None,
                 loss_weights=None, beta_concentration=1.0,
                 normalize_poisson=False,
                 poisson_residual_floor=poisson.POISSON_RESIDUAL_FLOOR):
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
        self.normalize_poisson = normalize_poisson
        self.poisson_residual_floor = poisson_residual_floor

        # A thermionic contact replaces that side's density pins entirely, so
        # drop them before they are turned into tensors below.
        self.thermionic_left = thermionic_left
        self.thermionic_right = thermionic_right
        if thermionic_left is not None:
            u_p_bc_left = u_n_bc_left = None
        if thermionic_right is not None:
            u_n_bc_right = u_p_bc_right = None

        self.loss_weights = self._as_loss_weights(loss_weights)

        # Boundary points and targets, built once: both contacts together for
        # phi, and separately per side for the density pins. Allocated on the
        # target device up front so the training loop never copies them from
        # the CPU; shape (N, 1) because the networks take a column of points.
        self._x_bc = torch.tensor([[x_left], [x_right]], device=device, dtype=dtype)
        self._phi_bc = torch.tensor([[phi_bc_left], [phi_bc_right]],
                                    device=device, dtype=dtype)
        self._x_bc_left = torch.tensor([[x_left]], device=device, dtype=dtype)
        self._x_bc_right = torch.tensor([[x_right]], device=device, dtype=dtype)

        def _pin(value):
            return None if value is None else torch.tensor(
                [[value]], device=device, dtype=dtype)

        # Majority pins (None wherever that side is thermionic).
        self._u_p_bc_left = _pin(u_p_bc_left)
        self._u_n_bc_right = _pin(u_n_bc_right)
        # Minority pins, imposed only when majority_only_bc is False.
        self._u_n_bc_left = _pin(u_n_bc_left)
        self._u_p_bc_right = _pin(u_p_bc_right)

    @staticmethod
    def _as_loss_weights(loss_weights):
        """Accept either a LossWeights object or a dict of overrides."""
        if isinstance(loss_weights, LossWeights):
            return loss_weights
        weights = dict(DEFAULT_LOSS_WEIGHTS)
        if loss_weights:
            weights.update(loss_weights)
        return FixedWeights(weights)

    @property
    def all_params(self):
        """Every trainable parameter, the set the optimizer and the adaptive
        weight rules both work over."""
        return (list(self.phi_net.parameters())
                + list(self.n_net.parameters())
                + list(self.p_net.parameters()))

    @property
    def thermionic_contacts(self):
        """The device's thermionic contacts, empty for a purely Ohmic one."""
        return [c for c in (self.thermionic_left, self.thermionic_right)
                if c is not None]

    # --- collocation sampling ---

    def sample_interior(self, n):
        """n random collocation points in the domain (mesh-free).

        Drawn from a symmetric Beta(a, a) with a = self.beta_concentration:
        a = 1 is uniform, a < 1 clusters points at both contacts.
        """
        a = self.beta_concentration
        if a == 1.0:
            # torch.rand draws uniform [0, 1) directly on the target device,
            # avoiding a CPU->GPU copy every epoch.
            u = torch.rand(n, 1, device=self.device, dtype=self.dtype)
        elif a == 0.5:
            # Closed-form arcsine substitution, U ~ Uniform(0,1) =>
            # sin^2(pi*U/2) ~ Beta(1/2, 1/2). Stays on-device and in dtype.
            u = torch.sin(0.5 * np.pi * torch.rand(
                n, 1, device=self.device, dtype=self.dtype)) ** 2
        else:
            # torch.distributions.Beta is the general sampler; .sample() draws
            # without recording a graph (these are inputs, not quantities to
            # differentiate), and .reshape puts them in a (n, 1) column.
            u = torch.distributions.Beta(
                torch.tensor(a, device=self.device, dtype=self.dtype),
                torch.tensor(a, device=self.device, dtype=self.dtype),
            ).sample((n, 1)).reshape(n, 1)

        # Keep samples strictly inside (0, 1): the endpoints are imposed
        # separately as Dirichlet points. torch.finfo gives the smallest
        # representable step for this dtype, so the nudge is the least that
        # actually moves the value in float32 as well as float64.
        eps = torch.finfo(self.dtype).eps
        u = u.clamp(eps, 1.0 - eps)

        x = u * (self.x_right - self.x_left) + self.x_left
        # requires_grad_(True): the residuals differentiate with respect to x,
        # so autograd must record every operation applied to it from here on.
        return x.requires_grad_(True)

    # --- fields and residuals ---

    def fields(self, x):
        """Evaluate all three networks and the derivatives the residuals need.

        Returns (phi, phi', phi'', n_hat, p_hat, n_hat', p_hat').
        """
        phi = self.phi_net(x)
        dphi = d_dx(phi, x)
        d2phi = d_dx(dphi, x)          # Poisson needs the second derivative
        n_hat, p_hat, dn, dp = densities.evaluate_densities(
            self.n_net, self.p_net, x)
        return phi, dphi, d2phi, n_hat, p_hat, dn, dp

    def scaled_currents(self, dphi, n_hat, p_hat, dn, dp):
        """Scaled Jn, Jp at the given fields (see densities.scaled_currents)."""
        return densities.scaled_currents(
            dphi, n_hat, p_hat, dn, dp,
            mu_n_hat=self.scaling["mu_n_hat"],
            mu_p_hat=self.scaling["mu_p_hat"])

    def continuity_residuals(self, x, dphi, n_hat, p_hat, dn, dp):
        """The two continuity residuals, plus the currents they were built
        from -- which the current-constancy term also needs."""
        Jn, Jp = self.scaled_currents(dphi, n_hat, p_hat, dn, dp)
        R = densities.langevin_recombination(
            n_hat, p_hat,
            prefactor=self.scaling["langevin_prefactor"],
            nie_hat=self.scaling["nie_hat"])
        r_n, r_p = densities.continuity_residuals(x, Jn, Jp, R)
        return r_n, r_p, Jn, Jp

    # --- loss terms ---

    def boundary_loss(self):
        """Summed Dirichlet residuals at the two contacts.

        phi is pinned at both contacts regardless of contact type -- the
        electrode is a good conductor either way. Densities are pinned only at
        Ohmic contacts, and there per majority_only_bc; architecturally pinned
        ends contribute zero, since the constraint already holds.
        """
        L = poisson.dirichlet_loss(self.phi_net, self._x_bc, self._phi_bc)

        n_hard_L, n_hard_R = self.n_net.hard_ends
        p_hard_L, p_hard_R = self.p_net.hard_ends

        # Majority pins: holes at the left contact, electrons at the right.
        L = L + densities.pin_loss(
            self.p_net, self._x_bc_left, self._u_p_bc_left, p_hard_L)
        L = L + densities.pin_loss(
            self.n_net, self._x_bc_right, self._u_n_bc_right, n_hard_R)

        # Minority pins, the reverse carrier at each contact.
        if not self.majority_only_bc:
            L = L + densities.pin_loss(
                self.n_net, self._x_bc_left, self._u_n_bc_left, n_hard_L)
            L = L + densities.pin_loss(
                self.p_net, self._x_bc_right, self._u_p_bc_right, p_hard_R)

        return L

    def thermionic_loss(self):
        """Mean-squared injection flux balance over the thermionic contacts;
        zero for a purely Ohmic device."""
        return boundaries.thermionic_loss(
            self.thermionic_contacts,
            phi_net=self.phi_net, n_net=self.n_net, p_net=self.p_net,
            scaling=self.scaling, device=self.device, dtype=self.dtype)

    def loss_terms(self, n_int):
        """Sample n_int interior points and evaluate every unweighted term.

        Returns a dict keyed by ``LOSS_TERMS``, holding tensors rather than
        floats -- .item() forces a synchronisation, so the training loop reads
        them only on epochs it prints.
        """
        x_int = self.sample_interior(n_int)
        _, dphi, d2phi, n_hat, p_hat, dn, dp = self.fields(x_int)
        r_n, r_p, Jn, Jp = self.continuity_residuals(
            x_int, dphi, n_hat, p_hat, dn, dp)

        return {
            # Electrostatics.
            "poisson": poisson.poisson_loss(
                d2phi, n_hat, p_hat, lam=self.scaling["lam"],
                normalize=self.normalize_poisson,
                floor=self.poisson_residual_floor),
            # Transport, one per carrier. Kept separate rather than summed:
            # they can differ by orders of magnitude and carry own weights.
            # Mean squared residual over the batch, one scalar tensor each.
            "cont_n": torch.mean(r_n ** 2),
            "cont_p": torch.mean(r_p ** 2),
            "jtot": densities.current_constancy_residual(Jn, Jp),
            # Contacts: Dirichlet pins, and the flux balance replacing them at
            # a thermionic contact.
            "bc": self.boundary_loss(),
            "thermionic": self.thermionic_loss(),
        }

    def total_loss(self, n_int):
        """The weighted total loss and the unweighted terms behind it.

        ``self.loss_weights.weights`` is read live, so an adaptive rule that
        mutates it in place is picked up here without further plumbing.
        """
        terms = self.loss_terms(n_int)
        w = self.loss_weights.weights
        # Scaling a tensor by a python float keeps it in the graph, so the
        # weighted sum stays differentiable and .backward() reaches every term.
        total = sum(w[name] * L for name, L in terms.items())
        return total, terms


def build_problem(*, phi_net, n_net, p_net, scaling, device, dtype,
                  x_left, x_right, phi_bc_left, phi_bc_right,
                  majority_only_bc, u_p_bc_left=None, u_n_bc_right=None,
                  u_n_bc_left=None, u_p_bc_right=None,
                  thermionic_left=None, thermionic_right=None,
                  loss_weights=None, beta_concentration=1.0,
                  normalize_poisson=False,
                  poisson_residual_floor=poisson.POISSON_RESIDUAL_FLOOR):
    """Factory wrapping ``PINNProblem``'s constructor, so call sites read as
    "build the problem" and there is room for validation later. Arguments are
    PINNProblem's; see its docstring.
    """
    return PINNProblem(
        phi_net=phi_net, n_net=n_net, p_net=p_net, scaling=scaling,
        device=device, dtype=dtype, x_left=x_left, x_right=x_right,
        phi_bc_left=phi_bc_left, phi_bc_right=phi_bc_right,
        majority_only_bc=majority_only_bc,
        u_p_bc_left=u_p_bc_left, u_n_bc_right=u_n_bc_right,
        u_n_bc_left=u_n_bc_left, u_p_bc_right=u_p_bc_right,
        thermionic_left=thermionic_left, thermionic_right=thermionic_right,
        loss_weights=loss_weights, beta_concentration=beta_concentration,
        normalize_poisson=normalize_poisson,
        poisson_residual_floor=poisson_residual_floor,
    )


def build_networks(*, width, depth, u_n_offset, u_p_offset=None, device,
                   phi_bc=None, u_n_bc=None, u_p_bc=None):
    """Construct phi-Net, n-Net and p-Net.

    width, depth : shared backbone geometry.
    u_n_offset, u_p_offset : log-space initialisation offsets, typically the
        interior mean of -log(density_hat) from a reference profile.
    phi_bc : (phi_hat_left, phi_hat_right) to switch phi-Net to the hard
        ansatz. Pass the same values to ``build_problem`` so the reporting
        agrees; the redundant penalty is then dropped automatically.
    u_n_bc, u_p_bc : the same for the density pins, each (u_left, u_right)
        with either entry None to leave that end free -- the normal case under
        majority_only_bc.
    """
    phi_net = poisson.PhiNet(width=width, depth=depth, phi_bc=phi_bc).to(device)
    n_net = densities.LogDensityNet(width=width, depth=depth,
                                    u_offset=u_n_offset, u_bc=u_n_bc).to(device)
    p_net = densities.LogDensityNet(width=width, depth=depth,
                                    u_offset=u_p_offset, u_bc=u_p_bc).to(device)
    return phi_net, n_net, p_net
