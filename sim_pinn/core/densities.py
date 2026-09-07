"""
densities.py

n-Net and p-Net: the log-density parametrisation, its hard boundary ansatz,
and everything built on the two densities -- currents, Langevin recombination,
the continuity residuals and the density pin losses.

    Jn_hat  =  mu_n_hat*(n_hat' - n_hat*phi_hat')
    Jp_hat  = -mu_p_hat*(p_hat' + p_hat*phi_hat')
    Jn_hat' - R_hat = 0
    Jp_hat' + R_hat = 0
"""

import torch
import torch.nn as nn

from .networks import FCNN, d_dx


class LogDensityNet(nn.Module):
    """n-Net / p-Net: FCNN backbone in logarithmic parametrisation.

    width, depth : backbone geometry.
    u_offset : additive shift on u, setting the starting order of magnitude
        (soft form only).
    u_bc : None for the soft form, or (u_L, u_R) in compressed units for the
        hard one; either entry may be None to leave that end free.

    The raw output is the compressed variable u(x) = -log(density_hat), with
    the density recovered by the hard constraint density_hat = exp(-u)
    (``density``). Two reasons for the compression:

    * **Dynamic range.** The densities span ~26 decades across these devices.
      An MLP whose output is O(1) after Xavier init cannot emit 1e-47 and 1.0
      from the same weights; in log space that range is u ~ [0, 106].
    * **Positivity is structural.** exp(-u) > 0 for any u, so a negative
      density -- which would break the Langevin term and log-scale scoring --
      is unrepresentable rather than merely discouraged.

    Boundary treatment
    ------------------
    Soft: the pins are imposed by a penalty (``pin_loss``).

    Hard: the pin is built in, in the same trial-function form PhiNet uses but
    applied to u rather than the density:

        u(x) = u_L*(1-x) + u_R*x + x*(1-x)*N(x)      [both ends pinned]
        u(x) = u_L       + x*N(x)                    [left end only]
        u(x) = u_R       + (1-x)*N(x)                [right end only]

    The multiplier vanishes at each pinned end, so the pin holds for any N.
    The one-sided forms are the normal case: only the majority carrier is
    pinned at each contact, the minority pin being a genuine discontinuity a
    smooth network cannot represent. Working in log space is deliberate --
    u_L*(1-x) + u_R*x is a *geometric* interpolation of the density, the right
    baseline for a profile spanning decades.

    The hard form is not merely tidier. With soft pins the coupled solve has a
    degenerate minimum where n and p collapse to nearly equal constants
    several decades low: flat densities zero the continuity and
    current-constancy residuals *identically*, and n ~ p shrinks the Poisson
    source enough that a straight-line phi satisfies it too. No weighting
    recovers from that -- scaling an identically zero residual leaves zero --
    whereas pinning the majority end removes the solution from the hypothesis
    space.
    """

    def __init__(self, width=64, depth=4, u_offset=0.0, u_bc=None):
        super().__init__()
        self.body = FCNN(width, depth)
        # Xavier init leaves the backbone near 0, i.e. density_hat = 1
        # (= c_tilde) for both carriers, several decades high for the minority
        # one. That is costly rather than merely slow, since the residuals
        # depend on u exponentially through n_hat*p_hat = exp(-u_n - u_p) in
        # the Langevin term. A shift of the output, not a constraint on it --
        # contrast u_bc below.
        #
        # register_buffer: stored on the module and moved by .to(device), but
        # NOT trainable -- the optimizer never sees it, so the offset stays
        # fixed while the backbone learns around it.
        self.register_buffer("u_offset", torch.tensor(float(u_offset)))
        self.u_bc = u_bc
        if u_bc is not None:
            u_L, u_R = u_bc
            if u_L is None and u_R is None:
                raise ValueError(
                    "u_bc given but both ends are None; pass u_bc=None for a "
                    "fully soft density network.")
            # Register only the ends actually pinned; the other stays free.
            # Buffers again: fixed boundary data that must follow the module
            # onto the GPU. as_tensor wraps a value without copying it if it
            # is already a tensor.
            if u_L is not None:
                self.register_buffer("u_L", torch.as_tensor(float(u_L)))
            if u_R is not None:
                self.register_buffer("u_R", torch.as_tensor(float(u_R)))

    def forward(self, x):
        """Return the compressed variable u = -log(density_hat)."""
        if self.u_bc is None:
            return self.body(x) + self.u_offset

        n_out = self.body(x)
        u_L, u_R = self.u_bc
        # The offset is deliberately absent from the hard forms: the pinned
        # endpoints already set the scale.
        if u_L is not None and u_R is not None:
            return self.u_L * (1.0 - x) + self.u_R * x + x * (1.0 - x) * n_out
        if u_L is not None:
            # Only x = 0 pinned: the multiplier x vanishes there, and grows
            # freely towards x = 1.
            return self.u_L + x * n_out
        return self.u_R + (1.0 - x) * n_out      # only x = 1 pinned

    def density(self, x):
        """density_hat = exp(-u), the hard-constrained physical density."""
        # torch.exp, not math.exp: this has to stay a tensor op for autograd
        # to differentiate through it.
        return torch.exp(-self.forward(x))

    @property
    def hard_ends(self):
        """(left_pinned, right_pinned): which ends are architectural, so the
        matching penalty terms are redundant. Both False when soft."""
        if self.u_bc is None:
            return False, False
        u_L, u_R = self.u_bc
        return u_L is not None, u_R is not None


def evaluate_densities(n_net, p_net, x):
    """Both densities and their derivatives at x.

    Differentiates the log variable and applies the chain rule, rather than
    differentiating exp(-u) directly:

        n_hat  = exp(-u_n)
        n_hat' = -u_n' * exp(-u_n) = -u_n' * n_hat
    """
    # Calling a module runs its forward(). x must carry requires_grad for the
    # d_dx calls below to have a recorded graph to differentiate.
    u_n = n_net(x)
    u_p = p_net(x)
    n_hat = torch.exp(-u_n)
    p_hat = torch.exp(-u_p)
    dn = -d_dx(u_n, x) * n_hat
    dp = -d_dx(u_p, x) * p_hat
    return n_hat, p_hat, dn, dp


def scaled_currents(dphi, n_hat, p_hat, dn, dp, *, mu_n_hat, mu_p_hat):
    """Scaled current densities, diffusion minus drift per carrier:

        Jn_hat =  mu_n_hat*(n_hat' - n_hat*phi_hat')
        Jp_hat = -mu_p_hat*(p_hat' + p_hat*phi_hat')

    Ut is absent (unlike the unscaled Jn = q*mu_n*(Ut*n' - n*phi')) because it
    is absorbed into the phi scaling.
    """
    Jn = mu_n_hat * (dn - n_hat * dphi)
    Jp = -mu_p_hat * (dp + p_hat * dphi)
    return Jn, Jp


def langevin_recombination(n_hat, p_hat, *, prefactor, nie_hat):
    """Scaled Langevin recombination, mirroring os_physics.py's CreateLangevin:

        ULANG = gammar * (q/eps) * (n*p - nie^2) * (mu_n + mu_p)

    with the constant folded into ``prefactor``. nie_hat^2 is kept despite
    being negligible against n_hat*p_hat: it is what makes R vanish at
    equilibrium.
    """
    return prefactor * (n_hat * p_hat - nie_hat ** 2)


def continuity_residuals(x, Jn, Jp, R):
    """The two carrier continuity residuals:

        Jn_hat' - R_hat = 0
        Jp_hat' + R_hat = 0

    Opposite signs: recombination is a sink for both carriers, but of opposite
    charge (div Jn = qR, div Jp = -qR).
    """
    return d_dx(Jn, x) - R, d_dx(Jp, x) + R


def current_constancy_residual(Jn, Jp):
    """Total-current constancy: in 1D steady state (Jn + Jp)' = 0.

    Taken as the variance of Jn + Jp over the batch, which enforces constancy
    without needing to know the constant's value.

    Why this is not independent of the continuity residuals
    -------------------------------------------------------
    Summing them, the recombination cancels:

        r_n + r_p = (Jn' - R) + (Jp' + R) = (Jn + Jp)'

    so this term's integrand *is* their sum, and it is zero automatically when
    both are zero pointwise. It adds no new physics. What it changes is the
    weighting, because the losses are means of squares:

        mean((r_n + r_p)^2) = mean(r_n^2) + mean(r_p^2) + 2*mean(r_n*r_p)

    i.e. it contributes only the cross term, re-penalising the *correlated*
    part of the two errors. Residuals that are large but anti-correlated --
    both carriers mis-transported while the total current is preserved -- are
    invisible to it.

    It is kept for two reasons. First, it is a different operator: the
    continuity residuals differentiate Jn and Jp, this one takes a variance
    over the batch and needs no derivative, which makes it comparatively more
    sensitive to a slow drift in Jn + Jp than to a ripple.
    Second, it is cheap and needs no reference solution, so the spread of
    Jn + Jp is a self-consistency check available during training.
    """
    Jtot = Jn + Jp
    # Variance over the batch. Jtot.mean() stays in the graph deliberately --
    # it is part of the quantity being minimised, so no detach() here.
    return torch.mean((Jtot - Jtot.mean()) ** 2)


def pin_loss(net, x_bc, u_bc, skip):
    """Mean-squared mismatch of one density pin.

    net, x_bc, u_bc : the network, the contact coordinate, the target u.
    skip : that endpoint's entry from ``net.hard_ends``.

    Zero when the end is architecturally pinned (the constraint already holds)
    or when no pin was requested there, as at a thermionic contact.
    """
    if u_bc is None or skip:
        # 0-dim tensor rather than 0.0, so the caller can sum it with the
        # other loss terms; device/dtype matched to x_bc, as torch requires
        # for any operation combining two tensors.
        return torch.zeros((), device=x_bc.device, dtype=x_bc.dtype)
    return torch.mean((net(x_bc) - u_bc) ** 2)
