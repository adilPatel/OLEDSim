"""
pinn_dd.py

``PINNProblem``: the three networks, the scaling constants and the boundary
data, bundled into the object the residuals and training loop act on.

Not a device abstraction -- just the state the loss needs, assembled once by
``build_problem()`` from parameters supplied by a device script. The physics
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
    u_p_bc_left, u_n_bc_right : the majority pins, in u = -log(density_hat).
    u_n_bc_left, u_p_bc_right : the minority pins, used only when
        majority_only_bc is False.
    thermionic_left, thermionic_right : optional ``ThermionicContact``. A side
        given one has its density pins dropped -- the densities there are
        unknowns fixed by the injection flux balance instead.
    loss_weights : a ``LossWeights`` instance, or a plain dict of overrides on
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
        # the corresponding Dirichlet values must not also be imposed.
        self.thermionic_left = thermionic_left
        self.thermionic_right = thermionic_right
        if thermionic_left is not None:
            u_p_bc_left = u_n_bc_left = None
        if thermionic_right is not None:
            u_n_bc_right = u_p_bc_right = None

        self.loss_weights = self._as_loss_weights(loss_weights)

        self._x_bc = torch.tensor([[x_left], [x_right]], device=device, dtype=dtype)
        self._phi_bc = torch.tensor([[phi_bc_left], [phi_bc_right]],
                                    device=device, dtype=dtype)

        self._x_bc_left = torch.tensor([[x_left]], device=device, dtype=dtype)
        self._x_bc_right = torch.tensor([[x_right]], device=device, dtype=dtype)

        def _pin(value):
            return None if value is None else torch.tensor(
                [[value]], device=device, dtype=dtype)

        # Majority pins at each Ohmic contact (None where that side is
        # thermionic).
        self._u_p_bc_left = _pin(u_p_bc_left)
        self._u_n_bc_right = _pin(u_n_bc_right)
        # Minority pins, only imposed when majority_only_bc is False.
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
        """Random collocation points in the domain (mesh-free).

        Drawn from a symmetric Beta(a, a) with a = self.beta_concentration:
        a = 1 is uniform, a < 1 clusters points at both contacts. a = 0.5 uses
        the closed-form arcsine substitution

            U ~ Uniform(0,1)  =>  sin^2(pi*U/2) ~ Beta(1/2, 1/2)

        which stays on-device and in the working dtype; other a != 1 fall back
        to torch's Beta sampler.

        Samples are clamped strictly inside (0, 1) so they cannot land on the
        endpoints, which are imposed separately as Dirichlet points.
        """
        a = self.beta_concentration
        if a == 1.0:
            u = torch.rand(n, 1, device=self.device, dtype=self.dtype)
        elif a == 0.5:
            u = torch.sin(0.5 * np.pi * torch.rand(
                n, 1, device=self.device, dtype=self.dtype)) ** 2
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

        Returns phi_hat, its first and second derivatives, the two densities
        and their derivatives.
        """
        phi = self.phi_net(x)
        dphi = d_dx(phi, x)
        d2phi = d_dx(dphi, x)
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
        from (which the current-constancy term also needs)."""
        Jn, Jp = self.scaled_currents(dphi, n_hat, p_hat, dn, dp)
        R = densities.langevin_recombination(
            n_hat, p_hat,
            prefactor=self.scaling["langevin_prefactor"],
            nie_hat=self.scaling["nie_hat"])
        r_n, r_p = densities.continuity_residuals(x, Jn, Jp, R)
        return r_n, r_p, Jn, Jp

    # --- loss terms ---

    def boundary_loss(self):
        """Dirichlet residuals at the two contacts.

        phi is pinned at both contacts regardless of contact type -- the
        electrode is a good conductor either way, exactly as
        CreateOSThermionicContact keeps CreateOSPotentialOnlyContact.

        Densities are pinned only at Ohmic contacts, and there per
        majority_only_bc. Architecturally pinned ends are skipped: they hold
        identically, so their penalty would contribute only numerical noise.
        """
        L = poisson.dirichlet_loss(self.phi_net, self._x_bc, self._phi_bc)

        n_hard_L, n_hard_R = self.n_net.hard_ends
        p_hard_L, p_hard_R = self.p_net.hard_ends

        # Majority pins: holes at the left contact, electrons at the right.
        L = L + densities.pin_loss(
            self.p_net, self._x_bc_left, self._u_p_bc_left, p_hard_L)
        L = L + densities.pin_loss(
            self.n_net, self._x_bc_right, self._u_n_bc_right, n_hard_R)

        # Minority pins.
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
        """Sample interior points and evaluate every unweighted loss term.

        Returns a dict keyed by the names in ``LOSS_TERMS``, as tensors rather
        than floats -- .item() forces a synchronisation, so the training loop
        should only read them on epochs it prints.
        """
        x_int = self.sample_interior(n_int)
        _, dphi, d2phi, n_hat, p_hat, dn, dp = self.fields(x_int)

        r_n, r_p, Jn, Jp = self.continuity_residuals(
            x_int, dphi, n_hat, p_hat, dn, dp)

        return {
            "poisson": poisson.poisson_loss(
                d2phi, n_hat, p_hat, lam=self.scaling["lam"],
                normalize=self.normalize_poisson,
                floor=self.poisson_residual_floor),
            # cont_n and cont_p are kept separate rather than summed: they can
            # differ by orders of magnitude and carry separate weights.
            "cont_n": torch.mean(r_n ** 2),
            "cont_p": torch.mean(r_p ** 2),
            "jtot": densities.current_constancy_residual(Jn, Jp),
            "bc": self.boundary_loss(),
            "thermionic": self.thermionic_loss(),
        }

    def total_loss(self, n_int):
        """The weighted total loss and the unweighted terms it came from.

        ``self.loss_weights.weights`` is read live, so an adaptive rule that
        mutates it in place is picked up here without further plumbing.
        """
        terms = self.loss_terms(n_int)
        w = self.loss_weights.weights
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
    "build the problem" and there is room for validation later."""
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

    u_n_offset / u_p_offset are the log-space initialisation offsets (a
    heuristic, not a training target): typically the interior mean of
    -log(density_hat) from a reference profile, excluding contact nodes where
    minority pins are discontinuous.

    phi_bc, if given as (phi_hat_left, phi_hat_right), switches phi-Net to the
    hard boundary ansatz; pass the same values to ``build_problem`` so the
    reporting stays consistent -- the redundant penalty is then dropped
    automatically. u_n_bc / u_p_bc do the same for the density pins, each a
    pair (u_left, u_right) in which either entry may be None to leave that end
    free, which is the normal case under majority_only_bc.
    """
    phi_net = poisson.PhiNet(width=width, depth=depth, phi_bc=phi_bc).to(device)
    n_net = densities.LogDensityNet(width=width, depth=depth,
                                    u_offset=u_n_offset, u_bc=u_n_bc).to(device)
    p_net = densities.LogDensityNet(width=width, depth=depth,
                                    u_offset=u_p_offset, u_bc=u_p_bc).to(device)
    return phi_net, n_net, p_net
