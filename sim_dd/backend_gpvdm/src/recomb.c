// recomb.c - bimolecular recombination coefficient models.
//
// gpvdm's newton plugin computes, per node (plugins/newton/newton.c):
//   Rfree   = B*(n*p - neq*peq)                          free-to-free
//   Rauger  = (Cn*n + Cp*p)*(n*p - neq*peq)              if auger_enabled
//   Rss_srh = (n*p - neq*peq)/(tau_p*(n+n1)+tau_n*(p+p1)) if ss_srh_enabled
// plus dynamic SRH recombination through the trap bands (handled in
// newton.c / dos.c here).  Those expressions are reproduced verbatim in
// newton.c; this file only supplies the bimolecular coefficient B, so a new
// bimolecular mechanism means a new case here:
//
//  * RECOMB_CONSTANT_B  B given directly (gpvdm's material parameter)
//  * RECOMB_LANGEVIN    B = pre * q*(mun+mup)/eps  (Langevin), using the
//                       local (possibly field/density-dependent) mobilities

#include "oled.h"

double recomb_get_B(const struct recomb_config *r, double mun, double mup, double epsr)
{
	switch (r->model)
	{
		case RECOMB_CONSTANT_B:
			return r->B;
		case RECOMB_LANGEVIN:
			return r->langevin_pre * Qe * (mun + mup) / (epsr * EPS0);
	}
	return r->B;
}
