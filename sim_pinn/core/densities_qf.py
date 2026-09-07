"""
densities_qf.py

n-Net and p-Net in the quasi-Fermi parametrisation: the networks emit the
quasi-Fermi potentials rather than the log densities. Mirrors densities.py --
same function names, same call signatures, same returned quantities -- so
PINNProblem can switch between the two by choosing a module.

    n_hat  = nie_hat*exp(phi_hat - phi_n_hat)
    p_hat  = nie_hat*exp(phi_p_hat - phi_hat)
    Jn_hat = -mu_n_hat*n_hat*phi_n_hat'
    Jp_hat = -mu_p_hat*p_hat*phi_p_hat'
    R_hat  = prefactor*nie_hat^2*(exp(phi_p_hat - phi_n_hat) - 1)

The two continuity residuals are unchanged and are reused from densities.py.
Motivation and derivation: Models_neural.md, "Quasi-Fermi parametrisation".
"""

import numpy as np
import torch
import torch.nn as nn

from .networks import FCNN, d_dx

# Ceiling on any exponent before torch.exp. float32 overflows above ~88; the
# converged solution needs at most |phi_hat - phi_n_hat| ~ 35 on these
# devices, so 60 leaves headroom while capping the transient a randomly
# initialised network can produce on the first epochs.
EXP_CLAMP = 60.0


class QuasiFermiNet(nn.Module):
    """n-Net / p-Net emitting a quasi-Fermi potential.

    width, depth : backbone geometry.
    phi_net : the potential network, needed to form the density from the
        Boltzmann relation. Held as a plain attribute, NOT a submodule, so
        phi-Net's parameters are not registered twice with the optimizer.
    carrier : "n" or "p", fixing the sign of the Boltzmann exponent.
    log_nie_hat : log(nie/c_tilde), from compute_scaling.
    qf_bc : None for the soft form, or (qf_L, qf_R) for the hard one; either
        entry may be None to leave that end free.

    Why quasi-Fermi rather than u = -log(density_hat)
    -------------------------------------------------
    Both are smooth reparametrisations of a positive density, but they place
    the difficulty differently. Across an Ohmic contact layer the minority
    density falls ~30 decades, which u must follow directly; the quasi-Fermi
    potential stays nearly flat there and the collapse is carried by the
    exponential of (phi_hat - phi_n_hat) instead. A tanh MLP represents the
    flat function far more readily than the 30-decade one.

    The transport residual changes character too. With Jn = -mu_n*n*phi_n',
    Jn is the (constant) terminal current, so where n_hat is small phi_n_hat'
    must be correspondingly large: the equation still constrains the network
    in the region where the log form's residual is multiplied by n_hat ~ 1e-30
    and carries no gradient signal at all.

    Boundary treatment
    ------------------
    One end pinned -- the majority_only_bc case -- uses LogDensityNet's form:

        qf(x) = qf_L + x*N(x)          [left end only]
        qf(x) = qf_R + (1-x)*N(x)      [right end only]

    Both ends pinned uses a *localised* pair of shape functions rather than
    the linear interpolant, selected by ``bc_decay``:

        s_L = exp(-x/d),  s_R = exp(-(1-x)/d),  b = (1-s_L)*(1-s_R)
        qf(x) = qf_L*s_L + qf_R*s_R + b*[interior(x) + N(x)]

    Each s vanishes away from its own contact, so the pins still hold
    exactly, but the baseline decays to zero within a few d of each contact
    instead of ramping linearly across the device.

    ``interior`` is an optional analytic term carried inside the bubble, so
    it cannot disturb the pins. It is what the constant ``u_offset`` becomes
    when the bulk profile is not flat: with the localised baseline the whole
    interior is left to the network, and on these devices the quasi-Fermi
    level varies almost linearly across it -- that gradient is what carries
    the current, since Jn = -mu_n*n*phi_n'. Supplying it analytically leaves
    the network only the departure from it.

    interior_slope, interior_offset give the linear form
    ``interior_offset + interior_slope*x``; interior_bump adds a
    x(1-x)-shaped term that vanishes at both ends and peaks mid-device.

    Why not the linear interpolant here
    -----------------------------------
    The minority pin is ~77 units from its partner (phi_n runs +42.6 at the
    anode to -34.7 at the cathode), while the true interior profile sits at
    -22 .. -35 -- i.e. it leaves the contact value within the first 0.1 nm and
    stays near the far pin thereafter. Against a linear baseline the bubble
    must therefore cancel ~65 units of ramp across the whole device while
    itself vanishing at both ends, which requires N ~ -5e4 at the first
    interior node: unreachable for a tanh network whose output is O(1), and
    the reason the linear form collapses to a 33-decade sigmoid instead.
    With d = 0.002 the same profile needs |N| <~ 100.
    """

    def __init__(self, width=64, depth=4, *, phi_net, carrier, log_nie_hat,
                 qf_bc=None, bc_decay=None, u_offset=0.0,
                 interior_offset=0.0, interior_slope=0.0, interior_bump=0.0):
        super().__init__()
        self.body = FCNN(width, depth)
        # Decay length (scaled) of the two-ended shape functions; None keeps
        # the linear interpolant. Only consulted when both ends are pinned.
        self.bc_decay = bc_decay
        # Analytic interior term, carried inside the bubble so the pins are
        # untouched. Buffers: fixed data that must move with .to(device).
        self.register_buffer("interior_offset",
                             torch.tensor(float(interior_offset)))
        self.register_buffer("interior_slope",
                             torch.tensor(float(interior_slope)))
        self.register_buffer("interior_bump",
                             torch.tensor(float(interior_bump)))
        # Additive shift on the network output, as LogDensityNet's u_offset:
        # the localised baseline decays to 0, so the bulk level has to come
        # from the network, and starting it near the right value costs
        # nothing. Buffer, not parameter -- fixed, but moved by .to(device).
        self.register_buffer("u_offset", torch.tensor(float(u_offset)))
        if carrier not in ("n", "p"):
            raise ValueError("carrier must be 'n' or 'p', got {0!r}".format(carrier))
        self.carrier = carrier
        # object.__setattr__ bypasses nn.Module's __setattr__, which would
        # register phi_net as a child module -- that would double-count its
        # parameters in .parameters() and hand the optimizer two copies.
        object.__setattr__(self, "phi_net", phi_net)
        self.register_buffer("log_nie_hat", torch.as_tensor(float(log_nie_hat)))
        self.qf_bc = qf_bc
        if qf_bc is not None:
            qf_L, qf_R = qf_bc
            if qf_L is None and qf_R is None:
                raise ValueError(
                    "qf_bc given but both ends are None; pass qf_bc=None for a "
                    "fully soft network.")
            # Buffers: fixed boundary data that must follow the module onto
            # the GPU, but that the optimizer must not treat as trainable.
            if qf_L is not None:
                self.register_buffer("qf_L", torch.as_tensor(float(qf_L)))
            if qf_R is not None:
                self.register_buffer("qf_R", torch.as_tensor(float(qf_R)))

    def forward(self, x):
        """Return the scaled quasi-Fermi potential phi_n_hat or phi_p_hat."""
        n_out = self.body(x) + self.u_offset
        if self.qf_bc is None:
            return n_out

        qf_L, qf_R = self.qf_bc
        if qf_L is not None and qf_R is not None:
            if self.bc_decay is None:
                return (self.qf_L * (1.0 - x) + self.qf_R * x
                        + x * (1.0 - x) * n_out)
            # Localised shape functions: each is 1 at its own contact and
            # decays over bc_decay, so the pins hold exactly while the
            # baseline dies away from them instead of ramping across.
            s_L = torch.exp(-x / self.bc_decay)
            s_R = torch.exp(-(1.0 - x) / self.bc_decay)
            # interior + network, both damped by the same bubble: the linear
            # ramp carries the bulk quasi-Fermi gradient, the bump adds a
            # mid-device contribution vanishing at both ends.
            inner = (n_out + self.interior_offset + self.interior_slope * x
                     + self.interior_bump * (4.0 * x * (1.0 - x)))
            return (self.qf_L * s_L + self.qf_R * s_R
                    + (1.0 - s_L) * (1.0 - s_R) * inner)
        if qf_L is not None:
            return self.qf_L + x * n_out
        return self.qf_R + (1.0 - x) * n_out

    def log_density(self, x, phi=None):
        """log(density_hat), the exponent before it is exponentiated.

        n: log_nie_hat + phi_hat - phi_n_hat
        p: log_nie_hat + phi_p_hat - phi_hat

        phi may be passed in when the caller has already evaluated phi-Net on
        these points, avoiding a second forward pass.
        """
        if phi is None:
            phi = self.phi_net(x)
        qf = self.forward(x)
        signed = (phi - qf) if self.carrier == "n" else (qf - phi)
        return self.log_nie_hat + signed

    def density(self, x, phi=None):
        """density_hat = exp(log_nie_hat +/- (phi_hat - qf)).

        Same signature as LogDensityNet.density when called with x alone, so
        reporting.py works against either parametrisation unchanged.
        """
        # clamp before exp: float32 overflows above ~88, and an untrained
        # network can emit far more than that on the first epochs.
        return torch.exp(torch.clamp(self.log_density(x, phi), max=EXP_CLAMP))

    @property
    def hard_ends(self):
        """(left_pinned, right_pinned): which ends are architectural, so the
        matching penalty terms are redundant. Both False when soft."""
        if self.qf_bc is None:
            return False, False
        qf_L, qf_R = self.qf_bc
        return qf_L is not None, qf_R is not None


def ohmic_quasi_fermi_bc(phi_bc_hat, *, log_nie_hat, carrier):
    """The quasi-Fermi value pinned at an Ohmic contact.

    At an Ohmic contact the semiconductor is in equilibrium with the metal, so
    both quasi-Fermi levels equal the metal's, i.e. the applied potential
    there. Inverting the Boltzmann relation at a contact whose majority
    density is the density scale (density_hat = 1):

        n: phi_n_hat = phi_hat + log_nie_hat
        p: phi_p_hat = phi_hat - log_nie_hat

    Derived from the contact potential and the material's nie alone -- no
    reference solution is read.
    """
    return (phi_bc_hat + log_nie_hat if carrier == "n"
            else phi_bc_hat - log_nie_hat)


def evaluate_densities(n_net, p_net, x, phi=None):
    """Both densities and their derivatives at x.

    Differentiates the exponent and applies the chain rule, rather than
    differentiating exp() directly:

        n_hat  = exp(log n_hat)
        n_hat' = (log n_hat)' * n_hat = (phi_hat' - phi_n_hat') * n_hat
    """
    log_n = n_net.log_density(x, phi)
    log_p = p_net.log_density(x, phi)
    n_hat = torch.exp(torch.clamp(log_n, max=EXP_CLAMP))
    p_hat = torch.exp(torch.clamp(log_p, max=EXP_CLAMP))
    dn = d_dx(log_n, x) * n_hat
    dp = d_dx(log_p, x) * p_hat
    return n_hat, p_hat, dn, dp


def scaled_currents(n_net, p_net, x, n_hat, p_hat, *, mu_n_hat, mu_p_hat):
    """Scaled current densities in quasi-Fermi form:

        Jn_hat = -mu_n_hat*n_hat*phi_n_hat'
        Jp_hat = -mu_p_hat*p_hat*phi_p_hat'

    Drift and diffusion have cancelled analytically: substituting
    n_hat' = n_hat*(phi_hat' - phi_n_hat') into mu_n_hat*(n_hat' - n_hat*phi_hat')
    leaves only the quasi-Fermi gradient. Both carriers take the same sign
    here, the opposite drift/diffusion signs having been absorbed by the
    opposite signs in their Boltzmann exponents.
    """
    Jn = -mu_n_hat * n_hat * d_dx(n_net(x), x)
    Jp = -mu_p_hat * p_hat * d_dx(p_net(x), x)
    return Jn, Jp


def langevin_recombination(n_net, p_net, x, *, prefactor, nie_hat):
    """Scaled Langevin recombination in quasi-Fermi form:

        R_hat = prefactor*nie_hat^2*(exp(phi_p_hat - phi_n_hat) - 1)

    Equivalent to prefactor*(n_hat*p_hat - nie_hat^2), since
    n_hat*p_hat = nie_hat^2*exp(phi_p_hat - phi_n_hat), but better
    conditioned: the -1 that makes R vanish at equilibrium is subtracted
    against an O(1) exponential instead of against nie_hat^2 ~ 1e-31, which
    in float32 is lost entirely beside n_hat*p_hat ~ 1.

    expm1(z) = exp(z) - 1 computed without the cancellation that loses all
    precision as z -> 0, which is exactly the equilibrium limit.
    """
    split = p_net(x) - n_net(x)
    return prefactor * nie_hat ** 2 * torch.expm1(
        torch.clamp(split, max=EXP_CLAMP))


def pin_loss(net, x_bc, qf_bc, skip):
    """Mean-squared mismatch of one quasi-Fermi pin.

    Zero when the end is architecturally pinned (the constraint already holds)
    or when no pin was requested there.
    """
    if qf_bc is None or skip:
        # 0-dim tensor rather than 0.0, so the caller can sum it with the
        # other loss terms; device/dtype matched to x_bc as torch requires.
        return torch.zeros((), device=x_bc.device, dtype=x_bc.dtype)
    return torch.mean((net(x_bc) - qf_bc) ** 2)
