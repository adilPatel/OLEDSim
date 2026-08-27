"""
boundaries.py

Field-dependent (Emtage-O'Dwyer / Scott-Malliaras) injecting contacts: the
contact descriptor, the injection current, and the flux-balance residual that
replaces the Dirichlet density pins there.

The PINN counterpart of CreateOSThermionicContact. Rather than pinning the two
carrier densities, the contact imposes, per carrier,

    J_bulk_hat - sign * J_inj_hat = 0

at the contact node, leaving both densities free unknowns set by that balance.
"""

import torch

from .densities import evaluate_densities, scaled_currents
from .networks import d_dx

# Lower clamp on the reduced field f. The model contains exp(sqrt(f)), whose
# derivative exp(sqrt(f))/(2*sqrt(f)) diverges as f -> 0 -- integrable in the
# physics, but a divide-by-zero for autodiff, and f = 0 is exactly flat band.
# Matches f_reduced_floor in SetOSParameters; far below any real device field.
F_REDUCED_FLOOR = 1.0e-8


class ThermionicContact:
    """One field-dependent injecting contact.

    Parameters
    ----------
    x : contact position in scaled coordinates (0.0 or 1.0).
    side : "left" (low-x) or "right" (high-x). Sets the sign with which each
        carrier's injection enters its continuity equation, as node_suffix
        does in sim_dd: "into the semiconductor" is +x on the left and -x on
        the right, while electrons and holes carry current in opposite senses.
    phi_n, phi_p : electron/hole injection barriers, eV, derived from the work
        function as the Ohmic contact does it: phi_n = LUMO - wf,
        phi_p = wf - HOMO.

    The literature evaluates n/p a distance x_c = r_c/4 from the contact; as in
    sim_dd, the model is evaluated at the contact node instead.
    """

    def __init__(self, *, x, side, phi_n, phi_p):
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right', got {0!r}".format(side))
        self.x = float(x)
        self.side = side
        self.phi_n = float(phi_n)
        self.phi_p = float(phi_p)
        self.n_sign, self.p_sign = (-1.0, +1.0) if side == "left" else (+1.0, -1.0)


def reduced_field(dphi_hat, r_c_over_ell, floor=F_REDUCED_FLOOR):
    """Reduced field f = q*E*r_c/kT from the scaled potential gradient.

    With phi = Ut*phi_hat and x = ell*x_hat the field is E = -(Ut/ell)*phi_hat',
    so f = |E|*r_c/Ut = (r_c/ell)*|phi_hat'|.

    The field *magnitude* is used, not a signed projection: the sign of the
    potential drop across a contact edge does not reliably indicate which way
    the field aids injection, so a signed form would clamp to the floor
    exactly where injection is strongest.
    """
    # sqrt(dphi^2 + tiny) rather than torch.abs, since d|x|/dx is undefined
    # at 0 and autograd would hand back a NaN or an arbitrary subgradient
    # there. The tiny offset keeps the sqrt differentiable at dphi = 0.
    mag = torch.sqrt(dphi_hat ** 2 + 1.0e-30)
    # clamp(min=floor): elementwise lower bound, applied before the sqrt(f) in
    # _s_ratio, whose derivative diverges as f -> 0.
    return torch.clamp(r_c_over_ell * mag, min=floor)


def _s_ratio(f, sqrt_f):
    """S(E)/S(0) = (1/psi^2 - f)/4, in the cancellation-free form.

    psi is evaluated as 1/(1 + sqrt(f) + sqrt(1 + 2*sqrt(f))), not the literal
    1/f + 1/sqrt(f) - (1/f)*sqrt(1 + 2*sqrt(f)). The literal form subtracts two
    terms each diverging as 1/f to leave a result of order 1/2 -- catastrophic
    cancellation, which returns exactly zero around f ~ 1e-16 and makes the
    1/psi^2 a division by zero. The rewrite follows from multiplying by the
    conjugate ((1+s)^2 - (1+2s) = s^2 = f), is exact, and is finite at f = 0
    (psi = 1/2, hence S(E) = S(0), the correct zero-field limit).
    """
    psi = 1.0 / (1.0 + sqrt_f + torch.sqrt(1.0 + 2.0 * sqrt_f))
    return (1.0 / psi ** 2 - f) / 4.0


def injection_density(f, *, dos_hat, barrier_eV, Ut):
    """The density the contact drives towards:

        n_inj = N*exp(-phi/kT)*exp(sqrt(f))*S(0)/S(E)

    At f -> 0 this reduces to N*exp(-phi/kT), exactly what an Ohmic contact of
    the same work function would pin to -- the closest analogue of that pin,
    and useful for reporting. The barrier is in eV, so phi/kT is phi_eV/Ut.
    """
    sqrt_f = torch.sqrt(f)
    return dos_hat * torch.exp(-barrier_eV / Ut + sqrt_f) / _s_ratio(f, sqrt_f)


def injection_current(*, f, density_hat, dos_hat, barrier_eV, s0_hat, Ut):
    """Scaled injection current for one carrier, q*S(E)*(n_inj - n).

    f : reduced field at the contact.
    density_hat : the carrier's own density there.
    dos_hat, barrier_eV, s0_hat : that carrier's effective DOS, barrier and
        zero-field velocity.
    Ut : thermal voltage.

    Mirrors BuildInjectionExpressions. Factored as S(E)*(n_inj - n) rather
    than as a difference of two independently computed currents, using
    C == q*N*S(0): the residual then vanishes *identically* at equilibrium by
    construction instead of through the cancellation of two large numbers, and
    is manifestly linear in the contact's density with a negative slope.

    Returned in units of j_scale; positive means injection into the
    semiconductor.
    """
    sqrt_f = torch.sqrt(f)
    s_ratio = _s_ratio(f, sqrt_f)
    s_field_hat = s0_hat * s_ratio                                # S(E)
    n_inj_hat = dos_hat * torch.exp(-barrier_eV / Ut + sqrt_f) / s_ratio
    return s_field_hat * (n_inj_hat - density_hat)


def contact_fields(contact, *, phi_net, n_net, p_net, scaling, device, dtype):
    """Evaluate everything the flux balance needs at one contact node.

    Returns (Jn, Jp, Jn_inj, Jp_inj, f, n_hat, p_hat), all scaled.

    Unlike the Dirichlet pins this needs derivatives at the contact -- both
    the field and the bulk current involve gradients -- so the point is a
    fresh requires_grad_ tensor rather than a cached BC tensor.
    """
    # requires_grad_(True) marks x as something to differentiate with respect
    # to, so autograd records everything computed from it. Without it d_dx
    # below has no graph and errors. Trailing underscore = in-place.
    x = torch.tensor([[contact.x]], device=device, dtype=dtype).requires_grad_(True)

    # Bulk side: the drift-diffusion current arriving at the contact.
    phi = phi_net(x)
    dphi = d_dx(phi, x)
    n_hat, p_hat, dn, dp = evaluate_densities(n_net, p_net, x)
    Jn, Jp = scaled_currents(dphi, n_hat, p_hat, dn, dp,
                             mu_n_hat=scaling["mu_n_hat"],
                             mu_p_hat=scaling["mu_p_hat"])

    # Electrode side: what the contact injects at this field.
    Ut = scaling["Ut"]
    f = reduced_field(dphi, scaling["r_c_over_ell"])
    Jn_inj = injection_current(
        f=f, density_hat=n_hat, dos_hat=scaling["nc_hat"],
        barrier_eV=contact.phi_n, s0_hat=scaling["s0n_hat"], Ut=Ut)
    Jp_inj = injection_current(
        f=f, density_hat=p_hat, dos_hat=scaling["nv_hat"],
        barrier_eV=contact.phi_p, s0_hat=scaling["s0p_hat"], Ut=Ut)

    return Jn, Jp, Jn_inj, Jp_inj, f, n_hat, p_hat


def thermionic_residuals(contact, *, phi_net, n_net, p_net, scaling, device, dtype):
    """The two flux balances at one contact:

        Jn_hat - n_sign * Jn_inj_hat = 0
        Jp_hat - p_sign * Jp_inj_hat = 0

    The *balance*, not the injection current alone (as in
    CreateInjectionContact): a residual of "injection = 0" asserts the contact
    injects nothing, trivially satisfiable by collapsing the density. The
    balance makes the physical solution the only root, since
    q*S(E)*(n_inj - n) tends to the strictly positive q*S(E)*n_inj as n -> 0.
    """
    Jn, Jp, Jn_inj, Jp_inj, _, _, _ = contact_fields(
        contact, phi_net=phi_net, n_net=n_net, p_net=p_net,
        scaling=scaling, device=device, dtype=dtype)
    return Jn - contact.n_sign * Jn_inj, Jp - contact.p_sign * Jp_inj


def thermionic_loss(contacts, *, phi_net, n_net, p_net, scaling, device, dtype):
    """Mean-squared flux-balance residual summed over the thermionic contacts.

    Returns a zero tensor -- keeping the graph intact -- for a device with none.
    """
    terms = []
    for contact in contacts:
        if contact is None:
            continue
        r_n, r_p = thermionic_residuals(
            contact, phi_net=phi_net, n_net=n_net, p_net=p_net,
            scaling=scaling, device=device, dtype=dtype)
        terms.append(torch.mean(r_n ** 2) + torch.mean(r_p ** 2))

    if not terms:
        # 0-dim tensor, not 0.0: the caller adds this into the total loss, and
        # a python float there would silently drop out of the autograd graph.
        return torch.zeros((), device=device, dtype=dtype)
    # torch.stack joins the per-contact scalars into one tensor so .sum()
    # keeps them all connected to the graph.
    return torch.stack(terms).sum()
