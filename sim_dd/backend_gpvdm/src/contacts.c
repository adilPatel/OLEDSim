// contacts.c - contact boundary conditions.
//
// Ported concept from gpvdm (lib/initial.c + plugins/newton/newton.c): each
// contact is a Dirichlet ghost node holding fixed carrier densities and a
// potential phi_eq (+ applied voltage).  The equilibrium contact potential
// is found by inverting the DOS of the adjoining layer for the requested
// majority density, referencing everything to the equilibrium Fermi level
// EF = 0 - gpvdm's V_y0/V_y1 built-in potentials arise the same way.
//
// Contact models:
//  * CONTACT_OHMIC_NP        majority density given directly (gpvdm's np)
//  * CONTACT_OHMIC_BARRIER   majority density = D(EF = -phiB), i.e. the
//    layer's own DOS evaluated at the barrier (Nc*exp(-phiB/kT) for
//    Boltzmann statistics, the Gaussian integral for STATS_GAUSSIAN)
//  * CONTACT_THERMIONIC      thermionic injection with image-force
//    barrier lowering after J. C. Scott and G. G. Malliaras, Chem. Phys.
//    Lett. 299, 115 (1999): with the reduced field f = q F r_c / kT
//    (r_c = q^2 / 4 pi eps kT the Coulomb radius),
//        psi(f) = 1/f + f^-1/2 - (1/f) * sqrt(1 + 2 sqrt(f))
//        n_inj  = 4 psi^2 * D(EF = -phiB + sqrt(f) kT)
//    which reduces to plain barrier-limited injection as f -> 0 (4 psi^2 -> 1).
//    The exp(sqrt(f)) factor is exactly the Schottky image-force lowering,
//    since sqrt(f)*kT == sqrt(q F / 4 pi eps) identically.
//
//    The contact field F is taken from the current Newton state, so n_inj
//    depends on phi at the adjacent node.  That dependence is differentiated
//    and handed to the Newton assembly via dn_ghost_dphi / dp_ghost_dphi
//    (see contact_ghost_dphi below), rather than being lagged: with
//    r_c ~ 16nm the reduced field reaches f ~ 30-60 in a 100nm device, where
//    dln(n_inj)/dln(F) is order unity and the coupling is too strong to omit
//    from the Jacobian.

#include "oled.h"

static void contact_densities(struct contact *c, const struct layer *l, double T, double F_into_device)
{
	double kT_eV = KB * T / Qe;
	double n_maj = 0.0;
	const struct dos_side *dmaj = (c->majority == MAJORITY_ELECTRON) ? &l->dosn : &l->dosp;

	// Barrier-limited contacts place the metal Fermi level phiB below the
	// transport level; the injected density is then whatever the material's
	// own DOS holds at that Fermi position:
	//
	//     n_maj = Int N(E) f(E, EF = -phiB + dphi) dE
	//
	// evaluated through dos_get, so it follows the layer's `statistics`
	// setting.  For Boltzmann this reduces to the familiar
	// Nc*exp(-phiB/kT); for a Gaussian DOS it is several decades larger,
	// because the occupied states sit ~sigma^2/kT below the DOS centre.
	// Using Nc*exp(-phiB/kT) unconditionally (as this did previously) is
	// only correct for Boltzmann statistics and silently contradicts a
	// Gaussian-DOS material.
	switch (c->model)
	{
		case CONTACT_OHMIC_NP:
			n_maj = c->np;
			break;
		case CONTACT_OHMIC_BARRIER:
		{
			double den, dden, w;
			dos_get_n(dmaj, -c->barrier, T, &den, &dden, &w);
			n_maj = den;
			break;
		}
		case CONTACT_THERMIONIC:
		{
			double eps = l->epsr * EPS0;
			double rc = Qe * Qe / (4.0 * OLED_PI * eps * KB * T);
			double F = (F_into_device > 0.0) ? F_into_device : 0.0;
			double f = Qe * F * rc / (KB * T);
			double psi2 = 0.25;   // f->0 limit: psi -> 1/2
			double lower = 0.0;
			double den, dden, w;
			if (f > 1e-9)
			{
				double psi = 1.0 / f + 1.0 / sqrt(f) - (1.0 / f) * sqrt(1.0 + 2.0 * sqrt(f));
				psi2 = psi * psi;
				lower = sqrt(f);
			}
			// image-force lowering raises the Fermi level by sqrt(f)*kT
			dos_get_n(dmaj, -c->barrier + lower * kT_eV, T, &den, &dden, &w);
			n_maj = 4.0 * psi2 * den;
			break;
		}
	}

	// Minority carrier at an OHMIC contact: mass action.
	//
	//     n_min = N0n * N0p * exp(-Eg/kT) / n_maj
	//
	// It deliberately does NOT go through dos_get for a Gaussian material.
	// The Gaussian DOS evaluated at u = -Eg includes the disorder-broadened
	// tail, a factor exp(sigma^2/2k^2T^2) ~ 2e7 larger at sigma = 0.15 eV,
	// which would put many spurious minority carriers at the contact.  The
	// Boltzmann mass-action form is the physically intended "the contact is
	// in equilibrium, so np = ni^2" statement.
	//
	// Thermionic contacts are untouched: there the minority is set by the
	// injection barrier, not by equilibrium with the majority.
	int ohmic = (c->model == CONTACT_OHMIC_NP || c->model == CONTACT_OHMIC_BARRIER);
	double np_eq = l->dosn.Nc * l->dosp.Nc * exp(-l->Eg / kT_eV);

	// majority Fermi position -> contact potential; minority from EF = 0
	if (c->majority == MAJORITY_ELECTRON)
	{
		double u = dos_u_from_n(&l->dosn, n_maj, T);     // EF - Ec at contact
		// Ec = -phi - Xi = -u  ->  phi = u - Xi
		c->phi_eq = u - l->Xi;
		c->n_ghost = n_maj;
		if (ohmic && n_maj > 0.0)
			c->p_ghost = np_eq / n_maj;
		else
		{
			double up = -l->Eg - u;                      // Ev - EF
			double p, dp, w;
			dos_get_p(&l->dosp, up, T, &p, &dp, &w);
			c->p_ghost = p;
		}
	}
	else
	{
		double up = dos_u_from_n(&l->dosp, n_maj, T);    // Ev - EF at contact
		// Ev = -phi - Xi - Eg = up  ->  phi = -up - Xi - Eg
		c->phi_eq = -up - l->Xi - l->Eg;
		c->p_ghost = n_maj;
		if (ohmic && n_maj > 0.0)
			c->n_ghost = np_eq / n_maj;   // denominator = the hole majority
		else
		{
			double u = -l->Eg - up;                      // EF - Ec
			double n, dn, w;
			dos_get_n(&l->dosn, u, T, &n, &dn, &w);
			c->n_ghost = n;
		}
	}
}

// d(majority ghost density) / d(phi at the adjacent node), for the Newton
// Jacobian.  Only CONTACT_THERMIONIC has any field dependence; the other two
// models give a constant ghost density, so the derivative is zero.
//
// Chain:  n_maj = 4 psi(f)^2 Neff exp(-phiB/kT) exp(sqrt f),  f = q rc F / kT
//   d n_maj / dF   = n_maj * [ 2 psi'(f)/psi(f) + 1/(2 sqrt f) ] * (q rc / kT)
//   psi'(f)        = -1/f^2 - 1/(2 f^3/2) + sqrt(1+2 sqrt f)/f^2
//                    - 1/(2 f^3/2 sqrt(1+2 sqrt f))
// and F = -(phi_node - phi_ghost)/dy  =>  dF/dphi_node = -1/dy, with a sign
// flip when the majority carrier is repelled rather than aided by +F (the
// same F_in convention used in contacts_update).
static double contact_dnmaj_dphi(const struct contact *c, const struct layer *l,
                                 double T, double F_into_device, double dy,
                                 double sign_F_in)
{
	double eps, rc, f, s, r, psi, dpsi_df, dnmaj_dF, kT_eV;

	if (c->model != CONTACT_THERMIONIC)
		return 0.0;
	if (!(F_into_device > 0.0))
		return 0.0;              // clamped at F<=0, so d/dphi is zero there

	kT_eV = KB * T / Qe;
	eps = l->epsr * EPS0;
	rc = Qe * Qe / (4.0 * OLED_PI * eps * KB * T);
	f = Qe * F_into_device * rc / (KB * T);
	if (f <= 1e-9)
		return 0.0;

	s = sqrt(f);
	r = sqrt(1.0 + 2.0 * s);
	psi = 1.0 / f + 1.0 / s - (1.0 / f) * r;
	dpsi_df = -1.0 / (f * f) - 0.5 / (f * s) + r / (f * f) - 1.0 / (2.0 * f * s * r);

	{
		// n_maj = 4 psi(f)^2 * D(u),  u = -phiB + sqrt(f) kT
		//   dn/df = 8 psi psi' D  +  4 psi^2 D'(u) * kT/(2 sqrt f)
		const struct dos_side *dmaj = (c->majority == MAJORITY_ELECTRON) ? &l->dosn : &l->dosp;
		double den, dden, w, dn_df;
		dos_get_n(dmaj, -c->barrier + s * kT_eV, T, &den, &dden, &w);
		dn_df = 8.0 * psi * dpsi_df * den
		      + 4.0 * psi * psi * dden * kT_eV / (2.0 * s);
		dnmaj_dF = dn_df * (Qe * rc / (KB * T));
	}

	// dF_in/dphi_node = sign_F_in * (-1/dy)
	return dnmaj_dF * sign_F_in * (-1.0 / dy);
}

void contacts_setup(struct device *dev)
{
	contact_densities(&dev->anode, &dev->layers[0], dev->T, 0.0);
	contact_densities(&dev->cathode, &dev->layers[dev->nlayers - 1], dev->T, 0.0);
	dev->anode.dnmaj_dphi = 0.0;
	dev->cathode.dnmaj_dphi = 0.0;
}

// Refresh field-dependent injection between Newton iterations.  Only the
// ghost densities move; phi_eq stays pinned at its equilibrium value so the
// built-in potential is not disturbed.
void contacts_update(struct device *dev)
{
	int N = dev->N;
	double phi_eq_a = dev->anode.phi_eq;
	double phi_eq_c = dev->cathode.phi_eq;

	if (dev->anode.model == CONTACT_THERMIONIC)
	{
		double Va = dev->apply_voltage_to_anode ? dev->Vapplied : 0.0;
		double dy = dev->x[1] - dev->x[0];
		double F = -(dev->phi[0] - (phi_eq_a + Va)) / dy;   // -dphi/dx at x=0
		// hole injection at x=0 is aided by F pointing into the device (+x)
		double sgn = (dev->anode.majority == MAJORITY_HOLE) ? 1.0 : -1.0;
		double F_in = sgn * F;
		contact_densities(&dev->anode, &dev->layers[0], dev->T, F_in);
		dev->anode.phi_eq = phi_eq_a;
		dev->anode.dnmaj_dphi = contact_dnmaj_dphi(&dev->anode, &dev->layers[0],
		                                           dev->T, F_in, dy, sgn);
	}
	if (dev->cathode.model == CONTACT_THERMIONIC)
	{
		double Vc = dev->apply_voltage_to_anode ? 0.0 : dev->Vapplied;
		double dy = dev->x[N - 1] - dev->x[N - 2];
		double F = -((phi_eq_c + Vc) - dev->phi[N - 1]) / dy; // -dphi/dx at x=L
		// electron injection at x=L is aided by F pointing along -x
		double sgn = (dev->cathode.majority == MAJORITY_ELECTRON) ? 1.0 : -1.0;
		double F_in = sgn * F;
		contact_densities(&dev->cathode, &dev->layers[dev->nlayers - 1], dev->T, F_in);
		dev->cathode.phi_eq = phi_eq_c;
		// at x=L the field difference is (phi_ghost - phi_node), so dF/dphi_node
		// carries the opposite sign to the anode case: F = -(phi_g - phi_N-1)/dy
		dev->cathode.dnmaj_dphi = -contact_dnmaj_dphi(&dev->cathode,
		                                              &dev->layers[dev->nlayers - 1],
		                                              dev->T, F_in, dy, sgn);
	}
}
