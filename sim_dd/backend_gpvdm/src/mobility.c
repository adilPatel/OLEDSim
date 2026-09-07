// mobility.c - carrier mobility models.
//
// gpvdm's base model is a constant mobility per material
// (lib/newton_update.c: update_material_arrays -> get_n_muy).  Following the
// simulator spec (and Knapp et al., J. Appl. Phys. 108, 054504 (2010)), two
// further models are provided and the interface takes temperature, local
// field and local carrier density so that additional models (e.g. improved
// EGDM parametrisations, ECDM) can be added as new cases:
//
//  * MOB_CONSTANT       mu = mu0
//  * MOB_POOLE_FRENKEL  mu = mu0 * exp(gamma * sqrt(F))
//  * MOB_EGDM           extended Gaussian disorder model, Pasveer et al.,
//                       PRL 94, 206601 (2005) (the mobility model used by
//                       Knapp et al. 2010): temperature-, density- and
//                       field-dependent hopping mobility in a Gaussian DOS
//                       of width sigma with inter-site distance a.
//
// The field/density dependence is applied lagged (recomputed from the
// current Newton state each iteration, not differentiated in the Jacobian),
// the standard approach that keeps the Jacobian identical to the
// constant-mobility case.

#include "oled.h"

static double egdm_mu(const struct mobility_side *m, double T, double F, double n)
{
	double sigma = m->egdm_sigma * Qe;            // J
	double sh = sigma / (KB * T);                 // sigma-hat
	double a = m->egdm_a;
	double Nt = (m->egdm_Nt > 0.0) ? m->egdm_Nt : 1.0 / (a * a * a);
	// Temperature prefactor mu0(T) = mu0 * c1 * exp(-c2 * sigma_hat^2), with
	// c1 = 1.8e-9 and c2 published as 0.42 (Pasveer et al.).  c2 is by far the
	// more sensitive of the two: sigma_hat^2 = 33.7 at sigma = 0.15 eV / 300 K,
	// so a 0.03 change in c2 scales the mobility by e^(0.03*33.7) = 2.7x.
	// It is settable per layer via egdm_c2 (default 0.39).
	double c1 = 1.8e-9;
	double c2 = m->egdm_c2;
	double mu0T = m->mu0 * c1 * exp(-c2 * sh * sh);
	double mu = mu0T;

	// density enhancement g1(n) - Pasveer eq. (4)
	if (n > 0.0)
	{
		double xmax = (m->egdm_x_clamp > 0.0) ? m->egdm_x_clamp : 0.5;
		double delta = 2.0 * (log(sh * sh - sh) - log(log(4.0))) / (sh * sh);
		double x = n / (2.0 * Nt);
		if (x > xmax) x = xmax;                    // parametrisation limit n <= Nt
		mu *= exp(0.5 * (sh * sh - sh) * pow(2.0 * x, delta));
	}

	// field enhancement g2(F) - Pasveer eq. (5):
	//
	//     ln g2 = C (sqrt(1 + 0.8 Fhat^2) - 1),   Fhat = q a F / sigma,
	//     C     = 0.44 (sh^1.5 - 2.2)
	//
	// Fhat is the REDUCED FIELD: a dimensionless ratio, not a charge.  It is
	// the energy an elementary charge q gains crossing one site of spacing a
	// in field F, measured against the disorder width sigma.  (The q in the
	// numerator is the elementary charge and is of course constant; only
	// F varies.)  egdm_f_clamp is the ceiling on Fhat, not on any charge.
	//
	// Pasveer et al. fit this for Fhat up to ~2.  Beyond that it is an
	// extrapolation: ln g2 becomes asymptotically LINEAR in Fhat with slope
	// C sqrt(0.8) ~ 0.95, so g2 ~ exp(0.95 Fhat) and, since Fhat is
	// proportional to the applied bias, the current picks up an
	// exponential-in-V factor.  In a 129nm device at sigma=0.07eV that drives
	// the log-log J-V slope to ~11, which is unphysical - trap-free SCLC
	// cannot exceed 2 (Mott-Gurney).
	//
	// egdm_f_clamp = Fhat_max holds g2 FIXED at its value at the clamp for
	// any larger field.  Freezing g2 is the assumption that the field
	// enhancement saturates once the parametrisation stops being valid; a
	// linear continuation does not help, because the fit is already
	// asymptotically linear.  This is an extra condition NOT present in
	// Pasveer, applied only outside their fitted range; inside that range the
	// expression is exact and untouched.
	//
	// egdm_f_clamp = 0 disables the limit and recovers raw Pasveer.
	if (F > 0.0)
	{
		double C = 0.44 * (pow(sh, 1.5) - 2.2);
		double Fhat = Qe * a * F / sigma;
		double Fhat_max = m->egdm_f_clamp;

		if (Fhat_max > 0.0 && Fhat > Fhat_max)
			Fhat = Fhat_max;

		mu *= exp(C * (sqrt(1.0 + 0.8 * Fhat * Fhat) - 1.0));
	}
	return mu;
}

double mobility_get(const struct mobility_side *m, double T, double F, double carrier_density)
{
	switch (m->model)
	{
		case MOB_CONSTANT:
			return m->mu0;
		case MOB_POOLE_FRENKEL:
			return m->mu0 * exp(m->pf_gamma * sqrt(fabs(F)));
		case MOB_EGDM:
			return egdm_mu(m, T, fabs(F), carrier_density);
	}
	return m->mu0;
}
