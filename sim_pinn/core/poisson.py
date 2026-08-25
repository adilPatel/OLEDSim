"""
poisson.py

phi-Net and the Poisson equation: the network that carries the electrostatic
potential, its hard boundary ansatz, and the residual and loss terms built
from it.

Scaled equation:

    lambda^2 * phi_hat'' - (n_hat - p_hat) = 0
"""

import torch
import torch.nn as nn

from .networks import FCNN

# Floor on the normalised Poisson residual's denominator. |n_hat - p_hat|
# stays well above this in the interior of the devices here, so it is
# inactive; it exists to keep the division finite where the source passes
# through zero, as it does in a compensated or ambipolar region.
POISSON_RESIDUAL_FLOOR = 1.0e-12


class PhiNet(nn.Module):
    """phi-Net: the FCNN backbone, optionally wrapped in a Dirichlet ansatz.

    Soft (phi_bc=None): the backbone output is phi_hat directly and the
    contact values are imposed by a penalty term (see ``dirichlet_loss``).
    This is the original DDNet formulation.

    Hard (phi_bc=(phi_L, phi_R)): the contact values are built into the
    architecture as the trial function

        phi_hat(x) = phi_L*(1-x) + phi_R*x + x*(1-x)*N(x)

    on x in [0, 1]. The first two terms are the straight line through the two
    contact values; the third vanishes at both endpoints, so phi_hat(0) =
    phi_L and phi_hat(1) = phi_R hold identically for ANY network output N.

    Why the hard form matters, rather than merely being tidier: with the
    normalised residual (see ``poisson_residual``) the soft formulation has a
    degenerate minimum at phi'' = 0, where the residual equals -1 at every
    collocation point regardless of phi's value or slope, so phi receives no
    useful gradient. Escaping it requires temporarily *increasing* the Poisson
    term to build curvature, which gradient descent will not do -- so no
    choice of loss weight fixes it. The ansatz removes the constant solution
    from the hypothesis space: two distinct pinned endpoints cannot both be
    met by a constant.

    phi_bc is given in scaled units (phi_hat), matching the network's output.
    """

    def __init__(self, width=64, depth=4, phi_bc=None):
        super().__init__()
        self.body = FCNN(width, depth)
        self.phi_bc = phi_bc
        if phi_bc is not None:
            phi_L, phi_R = phi_bc
            # Buffers, not parameters: fixed boundary data, but they must
            # follow the module across .to(device) / dtype changes.
            self.register_buffer("phi_L", torch.as_tensor(float(phi_L)))
            self.register_buffer("phi_R", torch.as_tensor(float(phi_R)))

    def forward(self, x):
        n_out = self.body(x)
        if self.phi_bc is None:
            return n_out
        # x is already the scaled coordinate on [0, 1], so (1-x) and x are the
        # two linear shape functions and x*(1-x) the bubble that vanishes at
        # both contacts.
        return self.phi_L * (1.0 - x) + self.phi_R * x + x * (1.0 - x) * n_out

    @property
    def is_hard(self):
        """True when the Dirichlet values are architectural, so the matching
        penalty term is redundant."""
        return self.phi_bc is not None


def poisson_residual(d2phi, n_hat, p_hat, *, lam, normalize=False,
                     floor=POISSON_RESIDUAL_FLOOR):
    """lambda^2 * phi_hat'' - (n_hat - p_hat).

    Sign convention matches eps*phi'' = q*(n - p - C) with C = 0: a net
    electron excess gives positive curvature.

    With ``normalize``, the residual is divided pointwise by the local source
    magnitude, so each collocation point contributes a *relative* rather than
    absolute error:

        r = (lambda^2*phi'' - (n - p)) / max(|n - p|, floor)

    The source spans several orders of magnitude across the device, so under a
    plain absolute residual the bulk contributes essentially nothing to the
    loss and the potential there is left unconstrained; normalising weights
    every point equally regardless of the local source magnitude.

    The denominator is detached: it scales the residual but must not carry
    gradient, or the optimizer could shrink the loss by inflating |n - p|
    rather than by satisfying the equation.
    """
    r = lam ** 2 * d2phi - (n_hat - p_hat)
    if normalize:
        scale = torch.clamp(torch.abs(n_hat - p_hat), min=floor)
        r = r / scale.detach()
    return r


def poisson_loss(d2phi, n_hat, p_hat, *, lam, normalize=False,
                 floor=POISSON_RESIDUAL_FLOOR):
    """Mean-squared Poisson residual over the collocation batch."""
    r = poisson_residual(d2phi, n_hat, p_hat, lam=lam, normalize=normalize,
                         floor=floor)
    return torch.mean(r ** 2)


def dirichlet_loss(phi_net, x_bc, phi_bc):
    """Mean-squared mismatch of phi against its pinned contact values.

    Returns a literal zero tensor for a hard-ansatz phi-Net, which satisfies
    both values identically -- keeping the penalty there would add only
    numerical noise to the reported loss.
    """
    if phi_net.is_hard:
        return torch.zeros((), device=x_bc.device, dtype=x_bc.dtype)
    return torch.mean((phi_net(x_bc) - phi_bc) ** 2)
