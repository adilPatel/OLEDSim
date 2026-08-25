"""
densities.py

n-Net and p-Net: the log-density parametrisation, its hard boundary ansatz,
and everything built on the two densities -- currents, Langevin
recombination, the continuity residuals, and the density pin losses.

Scaled equations:

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

    The raw output is the compressed variable u(x) = -log(density_hat), and
    the density is recovered by the hard constraint density_hat = exp(-u)
    (``density``). Two reasons for the compression:

    * **Dynamic range.** The densities span ~26 decades across these devices.
      An MLP whose output is O(1) after Xavier init cannot emit 1e-47 and 1.0
      from the same weights; in log space that range is u ~ [0, 106].
    * **Positivity is structural.** exp(-u) > 0 for any u, so a negative
      density -- which would make the Langevin term and log-scale scoring
      meaningless -- is unrepresentable rather than merely discouraged.

    Boundary treatment
    ------------------
    Soft (u_bc=None): the pins are imposed by a penalty (``pin_loss``).

    Hard (u_bc given): the pin is built into the architecture, in the same
    trial-function form PhiNet uses but applied to u rather than the density:

        u(x) = u_L*(1-x) + u_R*x + x*(1-x)*N(x)      [both ends pinned]
        u(x) = u_L       + x*N(x)                    [left end only]
        u(x) = u_R       + (1-x)*N(x)                [right end only]

    The multiplier vanishes at each pinned endpoint, so the pin holds for any
    N. The one-sided forms are the normal case here: only the majority carrier
    is pinned at each contact (the minority pin is a genuine discontinuity a
    smooth network cannot represent), so n-Net is typically pinned at the
    cathode only and p-Net at the anode only.

    Applying the ansatz in log space is deliberate -- the linear interpolant
    u_L*(1-x) + u_R*x is a *geometric* interpolation of the density, the right
    baseline for a profile spanning decades.

    Why the hard form matters: with soft pins the coupled solve has a
    degenerate minimum in which n and p collapse to nearly equal constants
    several decades below their physical scale. Flat densities zero the
    continuity and current-constancy residuals *identically*, and n ~ p
    shrinks the Poisson source enough that a straight-line phi satisfies it
    too, so no loss weighting can recover -- multiplying an identically zero
    residual by any weight leaves zero. Pinning the majority end removes that
    solution from the hypothesis space.

    u_bc is given in compressed units (u = -log(density_hat)).
    """

    def __init__(self, width=64, depth=4, u_offset=0.0, u_bc=None):
        super().__init__()
        self.body = FCNN(width, depth)
        # Additive offset on u so a soft network starts near the right order
        # of magnitude. After Xavier init FCNN outputs ~0, i.e. density_hat =
        # 1 (= c_tilde) for BOTH carriers, which for holes is several decades
        # high. That is expensive rather than merely slow: the residuals
        # depend on u exponentially through Langevin recombination
        # (n_hat*p_hat = exp(-u_n - u_p)), so the early gradients chase a
        # spurious source term. A shift of the output, not a constraint on it
        # -- contrast u_bc, which is a hard constraint.
        self.register_buffer("u_offset", torch.tensor(float(u_offset)))
        # u_bc = (u_L, u_R); either entry may be None to leave that end free.
        self.u_bc = u_bc
        if u_bc is not None:
            u_L, u_R = u_bc
            if u_L is None and u_R is None:
                raise ValueError(
                    "u_bc given but both ends are None; pass u_bc=None for a "
                    "fully soft density network.")
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
        # The offset is deliberately NOT added in the hard forms: the pinned
        # endpoint values already set the scale.
        if u_L is not None and u_R is not None:
            return self.u_L * (1.0 - x) + self.u_R * x + x * (1.0 - x) * n_out
        if u_L is not None:
            # Only x = 0 pinned: the multiplier x vanishes there and grows
            # freely towards x = 1.
            return self.u_L + x * n_out
        # Only x = 1 pinned.
        return self.u_R + (1.0 - x) * n_out

    def density(self, x):
        """density_hat = exp(-u), the hard-constrained physical density."""
        return torch.exp(-self.forward(x))

    @property
    def hard_ends(self):
        """(left_pinned, right_pinned): which endpoints are architectural, so
        the matching penalty terms are redundant. Both False when soft."""
        if self.u_bc is None:
            return False, False
        u_L, u_R = self.u_bc
        return u_L is not None, u_R is not None


def evaluate_densities(n_net, p_net, x):
    """Both densities and their derivatives at x.

    The derivatives use the chain rule on the log variable rather than
    differentiating exp(-u) directly:

        n_hat  = exp(-u_n)
        n_hat' = -u_n' * exp(-u_n) = -u_n' * n_hat
    """
    u_n = n_net(x)
    u_p = p_net(x)
    n_hat = torch.exp(-u_n)
    p_hat = torch.exp(-u_p)
    dn = -d_dx(u_n, x) * n_hat
    dp = -d_dx(u_p, x) * p_hat
    return n_hat, p_hat, dn, dp


def scaled_currents(dphi, n_hat, p_hat, dn, dp, *, mu_n_hat, mu_p_hat):
    """Scaled electron and hole current densities:

        Jn_hat =  mu_n_hat*(n_hat' - n_hat*phi_hat')
        Jp_hat = -mu_p_hat*(p_hat' + p_hat*phi_hat')

    Ut does not appear (unlike the unscaled Jn = q*mu_n*(Ut*n' - n*phi'))
    because it is absorbed into the phi scaling.
    """
    Jn = mu_n_hat * (dn - n_hat * dphi)
    Jp = -mu_p_hat * (dp + p_hat * dphi)
    return Jn, Jp


def langevin_recombination(n_hat, p_hat, *, prefactor, nie_hat):
    """Scaled Langevin recombination, mirroring os_physics.py's CreateLangevin:

        ULANG = gammar * (q/eps) * (n*p - nie^2) * (mu_n + mu_p)

    with the constant folded into ``prefactor``. nie_hat^2 is kept even though
    it is negligible against n_hat*p_hat, since it is what makes R vanish at
    equilibrium.
    """
    return prefactor * (n_hat * p_hat - nie_hat ** 2)


def continuity_residuals(x, Jn, Jp, R):
    """The two carrier continuity residuals:

        Jn_hat' - R_hat = 0
        Jp_hat' + R_hat = 0

    Opposite signs: a recombination event is a sink for both carriers, but of
    opposite charge (div Jn = qR, div Jp = -qR).
    """
    return d_dx(Jn, x) - R, d_dx(Jp, x) + R


def current_constancy_residual(Jn, Jp):
    """Total-current constancy: in 1D steady state (Jn + Jp)' = 0 across the
    device. Implemented as the variance of Jn + Jp over the batch, which
    enforces constancy without needing to know the constant's value.
    """
    Jtot = Jn + Jp
    return torch.mean((Jtot - Jtot.mean()) ** 2)


def pin_loss(net, x_bc, u_bc, skip):
    """Mean-squared mismatch of one density pin.

    ``skip`` is that endpoint's entry from ``net.hard_ends``: an
    architecturally pinned end satisfies the constraint identically, so its
    penalty is dropped. A pin of None (no pin requested at that end, e.g. a
    thermionic contact) is likewise zero.
    """
    if u_bc is None or skip:
        return torch.zeros((), device=x_bc.device, dtype=x_bc.dtype)
    return torch.mean((net(x_bc) - u_bc) ** 2)
