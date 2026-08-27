"""
poisson.py

phi-Net and the Poisson equation: the network carrying the electrostatic
potential, its hard boundary ansatz, and the residual and loss built from it.

    lambda^2 * phi_hat'' - (n_hat - p_hat) = 0
"""

import torch
import torch.nn as nn

from .networks import FCNN

# Floor on the normalised residual's denominator, to keep the division finite
# where the source passes through zero (as it does in a compensated or
# ambipolar region). Inactive on the devices here.
POISSON_RESIDUAL_FLOOR = 1.0e-12


class PhiNet(nn.Module):
    """phi-Net: the FCNN backbone, optionally wrapped in a Dirichlet ansatz.

    width, depth : backbone geometry.
    phi_bc : None for the soft form, or (phi_L, phi_R) in scaled units
        (phi_hat) for the hard one.

    Soft: the backbone output is phi_hat directly, and the contact values are
    imposed by a penalty term (``dirichlet_loss``). The original DDNet form.

    Hard: the contact values are built into the architecture as

        phi_hat(x) = phi_L*(1-x) + phi_R*x + x*(1-x)*N(x)

    on x in [0, 1] -- a straight line through the two contact values plus a
    bubble vanishing at both ends, so the pins hold identically for ANY N.

    The hard form is not merely tidier. With the normalised residual (see
    ``poisson_residual``) phi'' = 0 is a degenerate minimum: the residual is
    then -1 at every collocation point regardless of phi's value or slope, so
    phi gets no useful gradient, and escaping needs the Poisson term to
    *increase* first. No loss weight fixes that; the ansatz instead makes a
    constant phi unrepresentable, since two distinct pins cannot both be met.
    """

    def __init__(self, width=64, depth=4, phi_bc=None):
        super().__init__()
        self.body = FCNN(width, depth)
        self.phi_bc = phi_bc
        if phi_bc is not None:
            phi_L, phi_R = phi_bc
            # register_buffer stores a tensor on the module WITHOUT making it
            # trainable: the optimizer ignores it, but it still moves with the
            # module on .to(device) and dtype changes. Exactly right for fixed
            # boundary data -- a plain attribute would stay on the CPU and
            # break the forward pass on GPU.
            self.register_buffer("phi_L", torch.as_tensor(float(phi_L)))
            self.register_buffer("phi_R", torch.as_tensor(float(phi_R)))

    def forward(self, x):
        n_out = self.body(x)
        if self.phi_bc is None:
            return n_out
        # x is already scaled to [0, 1], so (1-x) and x are the linear shape
        # functions and x*(1-x) the bubble that vanishes at both contacts.
        return self.phi_L * (1.0 - x) + self.phi_R * x + x * (1.0 - x) * n_out

    @property
    def is_hard(self):
        """True when the Dirichlet values are architectural, so the matching
        penalty term is redundant."""
        return self.phi_bc is not None


def poisson_residual(d2phi, n_hat, p_hat, *, lam, normalize=False,
                     floor=POISSON_RESIDUAL_FLOOR):
    """lambda^2 * phi_hat'' - (n_hat - p_hat).

    d2phi, n_hat, p_hat : the second derivative and both densities, scaled.
    lam : the scaled Debye parameter.
    normalize, floor : see below.

    Sign convention follows eps*phi'' = q*(n - p - C) with C = 0: a net
    electron excess gives positive curvature.

    With ``normalize``, each point contributes a *relative* rather than
    absolute error,

        r = (lambda^2*phi'' - (n - p)) / max(|n - p|, floor)

    The source spans several orders of magnitude across the device, so under
    the absolute residual the bulk contributes almost nothing to the loss and
    the potential there is left unconstrained; dividing by the local magnitude
    weights every point equally.
    """
    r = lam ** 2 * d2phi - (n_hat - p_hat)
    if normalize:
        # torch.clamp(min=floor) is an elementwise lower bound, keeping the
        # divisor away from zero.
        scale = torch.clamp(torch.abs(n_hat - p_hat), min=floor)
        # detach() returns the same values cut out of the autograd graph, so
        # no gradient flows back through the denominator. Load-bearing here:
        # without it the optimizer could shrink the loss by inflating |n - p|
        # rather than by satisfying the equation.
        r = r / scale.detach()
    return r


def poisson_loss(d2phi, n_hat, p_hat, *, lam, normalize=False,
                 floor=POISSON_RESIDUAL_FLOOR):
    """Mean-squared Poisson residual over the collocation batch."""
    r = poisson_residual(d2phi, n_hat, p_hat, lam=lam, normalize=normalize,
                         floor=floor)
    return torch.mean(r ** 2)   # mean over the collocation batch


def dirichlet_loss(phi_net, x_bc, phi_bc):
    """Mean-squared mismatch of phi against its pinned contact values.

    phi_net : the network under test.
    x_bc, phi_bc : the contact coordinates and their target phi_hat.

    A hard-ansatz phi-Net meets both values identically, so its penalty would
    contribute only numerical noise; return a literal zero instead.
    """
    if phi_net.is_hard:
        # torch.zeros(()) is a 0-dim (scalar) tensor, not a python float, so
        # it can still be added to the other loss terms. device/dtype are
        # copied from x_bc: torch refuses to combine tensors that differ in
        # either.
        return torch.zeros((), device=x_bc.device, dtype=x_bc.dtype)
    return torch.mean((phi_net(x_bc) - phi_bc) ** 2)
