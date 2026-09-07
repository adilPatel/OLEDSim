// newton.c - steady-state drift-diffusion Newton solver.
//
// This is a faithful port of gpvdm's newton plugin (plugins/newton/newton.c
// fill_matrix/solve_electrical + lib/newton_update.c update_y_array) reduced
// to 1D steady state:
//
//  * Unknowns per node: phi, xn, xp (+ trap quasi-Fermi levels per band).
//    n = N(xn + Xi), p = P(xp - Xi - Eg): identical variable convention.
//  * Scharfetter-Gummel fluxes with Bernoulli functions of
//    xi = 3 q (Ec_i - Ec_i-1) / (wn_i + wn_i-1), which reduces to the
//    textbook q dV/kT for Boltzmann statistics (w = 3kT/2) and generalises
//    the Einstein relation D = mu (2/3) w / q for Fermi-Dirac, exactly as
//    gpvdm does.
//  * Recombination: Rfree = B(np - neq peq), Auger, steady-state SRH and
//    dynamic multi-band SRH traps with the same r1..r4 rate structure and
//    the same trap-band Newton unknowns as gpvdm.
//  * Dirichlet ghost-node contacts (densities fixed, ghost phi carries the
//    applied voltage), gpvdm's V_y0/V_y1 scheme.
//  * The damped update  v += du / (1 + |du /(clamp kT/q)|)  is gpvdm's
//    electrical clamp (plugins/newton/update.c).
//
// Differences from gpvdm: the UMFPACK sparse solve is replaced by a banded
// LU over an interleaved variable ordering, and the Jacobian is fully
// analytic (gpvdm differentiates its DOS lookup tables numerically).

#include "oled.h"

#define VAR_PHI 0
#define VAR_XN 1
#define VAR_XP 2

struct bandmat
{
	int n, m, kl, ku;
	double *AB;
	double *b;
	double *rowmax;
};

static void bm_add(struct bandmat *M, int row, int col, double v)
{
	M->AB[(size_t)(M->kl + M->ku + row - col) * M->n + col] += v;
}

// Traps in quasi-equilibrium with the free carriers: no SRH kinetics, no
// trap-assisted recombination, trapped charge in Poisson only.
static int traps_equilibrium(const struct layer *l)
{
	return l->recomb.trap_kinetics == TRAP_QUASI_EQUILIBRIUM;
}

static int e_band_active(const struct layer *l, int b)
{
	return (l->dosn.trap_type != TRAPS_NONE) && (b < l->dosn.nbands) && (l->dosn.band[b].Nb > 0.0);
}

static int h_band_active(const struct layer *l, int b)
{
	return (l->dosp.trap_type != TRAPS_NONE) && (b < l->dosp.nbands) && (l->dosp.band[b].Nb > 0.0);
}

// Refresh all derived per-node arrays from the current Newton state
// (gpvdm lib/newton_update.c: update_y_array).
void update_arrays(struct device *dev)
{
	int i, b;
	int N = dev->N;
	double T = dev->T;

	for (i = 0; i < N; i++)
	{
		struct layer *l = &dev->layers[dev->lay[i]];
		double n, p, dn, dp, wn, wp;

		dos_get_n(&l->dosn, dev->xn[i] + dev->Xi[i], T, &n, &dn, &wn);
		dos_get_p(&l->dosp, dev->xp[i] - dev->Xi[i] - dev->Eg[i], T, &p, &dp, &wp);
		dev->n[i] = n; dev->dn[i] = dn; dev->wn[i] = wn;
		dev->p[i] = p; dev->dp[i] = dp; dev->wp[i] = wp;

		// lagged local field for field/density-dependent mobility
		// freeze_mobility: 1 = equilibrium (mu0), 2 = hold current values
		if (dev->freeze_mobility == 1)
		{
			dev->mun[i] = l->mun.mu0;
			dev->mup[i] = l->mup.mu0;
		}
		else if (dev->freeze_mobility == 2)
		{
			/* keep dev->mun/mup as they are */
		}
		else
		{
			double F, mn, mp;
			if (i == 0)
				F = -(dev->phi[1] - dev->phi[0]) / (dev->x[1] - dev->x[0]);
			else if (i == N - 1)
				F = -(dev->phi[N - 1] - dev->phi[N - 2]) / (dev->x[N - 1] - dev->x[N - 2]);
			else
				F = -(dev->phi[i + 1] - dev->phi[i - 1]) / (dev->x[i + 1] - dev->x[i - 1]);
			mn = mobility_get(&l->mun, T, F, n);
			mp = mobility_get(&l->mup, T, F, p);
			// geometric under-relaxation of the lagged update
			if (dev->mun[i] > 0.0) mn = sqrt(mn * dev->mun[i]);
			if (dev->mup[i] > 0.0) mp = sqrt(mp * dev->mup[i]);
			dev->mun[i] = mn;
			dev->mup[i] = mp;
		}

		dev->nt_all[i] = 0.0;
		dev->pt_all[i] = 0.0;
		for (b = 0; b < dev->nbands; b++)
		{
			// Under quasi-equilibrium the trap level tracks the free carrier
			// quasi-Fermi level; keep the stored xt/xpt in step so the state
			// written to file and used as the next initial guess is coherent.
			if (l->recomb.trap_kinetics == TRAP_QUASI_EQUILIBRIUM)
			{
				dev->xt[b][i] = dev->xn[i];
				dev->xpt[b][i] = dev->xp[i];
			}

			if (e_band_active(l, b))
			{
				trap_get_n(&l->dosn, l, b, dev->xt[b][i] + dev->Xi[i], T,
				           &dev->nt[b][i], &dev->srh_n_r1[b][i], &dev->srh_n_r2[b][i],
				           &dev->srh_n_r3[b][i], &dev->srh_n_r4[b][i],
				           &dev->dnt[b][i], &dev->dsrh_n_r1[b][i], &dev->dsrh_n_r2[b][i],
				           &dev->dsrh_n_r3[b][i], &dev->dsrh_n_r4[b][i]);
				dev->nt_all[i] += dev->nt[b][i];
			}
			else if (dev->nbands > 0)
				dev->nt[b][i] = 0.0;

			if (h_band_active(l, b))
			{
				trap_get_p(&l->dosp, l, b, dev->xpt[b][i] - dev->Xi[i] - dev->Eg[i], T,
				           &dev->pt[b][i], &dev->srh_p_r1[b][i], &dev->srh_p_r2[b][i],
				           &dev->srh_p_r3[b][i], &dev->srh_p_r4[b][i],
				           &dev->dpt[b][i], &dev->dsrh_p_r1[b][i], &dev->dsrh_p_r2[b][i],
				           &dev->dsrh_p_r3[b][i], &dev->dsrh_p_r4[b][i]);
				dev->pt_all[i] += dev->pt[b][i];
			}
			else if (dev->nbands > 0)
				dev->pt[b][i] = 0.0;
		}
	}
}

// One Newton iteration: assemble the banded Jacobian and residual, solve,
// apply the clamped update.  Returns max |du|.
static double newton_iteration(struct device *dev, struct sim_config *cfg, int with_recomb)
{
	int N = dev->N;
	int NB = dev->nbands;
	int m = 3 + 2 * NB;
	int n = N * m;
	int kl = 2 * m - 1, ku = 2 * m - 1;
	int i, b;
	double T = dev->T;
	double Va = dev->apply_voltage_to_anode ? dev->Vapplied : 0.0;
	double Vc = dev->apply_voltage_to_anode ? 0.0 : dev->Vapplied;
	double phi_ga = dev->anode.phi_eq + Va;      // ghost potentials
	double phi_gc = dev->cathode.phi_eq + Vc;
	double max_du = 0.0;

	struct bandmat M;
	M.n = n; M.m = m; M.kl = kl; M.ku = ku;
	M.AB = xmalloc(sizeof(double) * (size_t)(2 * kl + ku + 1) * n);
	M.b = xmalloc(sizeof(double) * n);

	// ghost-node DOS values (energy w for the generalised Einstein relation)
	double wn_ga, wp_ga, wn_gc, wp_gc, tmp1, tmp2;
	{
		struct layer *l0 = &dev->layers[0];
		struct layer *l1 = &dev->layers[dev->nlayers - 1];
		double u;
		u = dos_u_from_n(&l0->dosn, dev->anode.n_ghost, T);
		dos_get_n(&l0->dosn, u, T, &tmp1, &tmp2, &wn_ga);
		u = dos_u_from_n(&l0->dosp, dev->anode.p_ghost, T);
		dos_get_p(&l0->dosp, u, T, &tmp1, &tmp2, &wp_ga);
		u = dos_u_from_n(&l1->dosn, dev->cathode.n_ghost, T);
		dos_get_n(&l1->dosn, u, T, &tmp1, &tmp2, &wn_gc);
		u = dos_u_from_n(&l1->dosp, dev->cathode.p_ghost, T);
		dos_get_p(&l1->dosp, u, T, &tmp1, &tmp2, &wp_gc);
	}

	for (i = 0; i < N; i++)
	{
		struct layer *l = &dev->layers[dev->lay[i]];
		int row_phi = i * m + VAR_PHI;
		int row_e = i * m + VAR_XN;
		int row_h = i * m + VAR_XP;

		// ---- gather left/centre/right node quantities (ghosts at ends) --
		double yc = dev->x[i];
		double yl = (i == 0) ? (dev->x[0] - (dev->x[1] - dev->x[0])) : dev->x[i - 1];
		double yr = (i == N - 1) ? (dev->x[N - 1] + (dev->x[N - 1] - dev->x[N - 2])) : dev->x[i + 1];
		double dyl = yc - yl, dyr = yr - yc, ddh = 0.5 * (dyl + dyr);

		double phic = dev->phi[i];
		double phil = (i == 0) ? phi_ga : dev->phi[i - 1];
		double phir = (i == N - 1) ? phi_gc : dev->phi[i + 1];

		double Ecc = -phic - dev->Xi[i], Evc = Ecc - dev->Eg[i];
		double Ecl, Evl, Ecr, Evr;
		double nc = dev->n[i], pc = dev->p[i], dnc = dev->dn[i], dpc = dev->dp[i];
		double nl, pl, nr, pr, dnl, dpl, dnr, dpr;
		double wnc = dev->wn[i], wpc = dev->wp[i];
		double wnl, wpl, wnr, wpr;
		double munc = dev->mun[i], mupc = dev->mup[i];
		double munl, mupl, munr, mupr;

		if (i == 0)
		{
			Ecl = -phil - dev->Xi[i]; Evl = Ecl - dev->Eg[i];
			nl = dev->anode.n_ghost; pl = dev->anode.p_ghost;
			dnl = 0.0; dpl = 0.0;    // Dirichlet ghost
			wnl = wn_ga; wpl = wp_ga;
			munl = munc; mupl = mupc;
		}
		else
		{
			Ecl = -phil - dev->Xi[i - 1]; Evl = Ecl - dev->Eg[i - 1];
			nl = dev->n[i - 1]; pl = dev->p[i - 1];
			dnl = dev->dn[i - 1]; dpl = dev->dp[i - 1];
			wnl = dev->wn[i - 1]; wpl = dev->wp[i - 1];
			munl = dev->mun[i - 1]; mupl = dev->mup[i - 1];
		}
		if (i == N - 1)
		{
			Ecr = -phir - dev->Xi[i]; Evr = Ecr - dev->Eg[i];
			nr = dev->cathode.n_ghost; pr = dev->cathode.p_ghost;
			dnr = 0.0; dpr = 0.0;
			wnr = wn_gc; wpr = wp_gc;
			munr = munc; mupr = mupc;
		}
		else
		{
			Ecr = -phir - dev->Xi[i + 1]; Evr = Ecr - dev->Eg[i + 1];
			nr = dev->n[i + 1]; pr = dev->p[i + 1];
			dnr = dev->dn[i + 1]; dpr = dev->dp[i + 1];
			wnr = dev->wn[i + 1]; wpr = dev->wp[i + 1];
			munr = dev->mun[i + 1]; mupr = dev->mup[i + 1];
		}

		// generalised Einstein diffusion constants, edge-averaged (gpvdm)
		double Dnc = munc * (2.0 / 3.0) * wnc / Qe;
		double Dpc = mupc * (2.0 / 3.0) * wpc / Qe;
		double Dnl = 0.5 * (munl * (2.0 / 3.0) * wnl / Qe + Dnc);
		double Dnr = 0.5 * (munr * (2.0 / 3.0) * wnr / Qe + Dnc);
		double Dpl = 0.5 * (mupl * (2.0 / 3.0) * wpl / Qe + Dpc);
		double Dpr = 0.5 * (mupr * (2.0 / 3.0) * wpr / Qe + Dpc);

		// ------------------------------- Poisson -------------------------
		double e0 = 0.5 * (((i == 0) ? dev->epsr[i] : dev->epsr[i - 1]) + dev->epsr[i]) * EPS0;
		double e1 = 0.5 * (((i == N - 1) ? dev->epsr[i] : dev->epsr[i + 1]) + dev->epsr[i]) * EPS0;
		double F_phi = (e0 * (phil - phic) / dyl + e1 * (phir - phic) / dyr) / ddh
		             + Qe * (pc - nc + dev->Nad[i] + dev->pt_all[i] - dev->nt_all[i]);

		M.b[row_phi] = -F_phi;
		if (i != 0) bm_add(&M, row_phi, (i - 1) * m + VAR_PHI, e0 / dyl / ddh);
		bm_add(&M, row_phi, row_phi, -e0 / dyl / ddh - e1 / dyr / ddh);
		if (i != N - 1) bm_add(&M, row_phi, (i + 1) * m + VAR_PHI, e1 / dyr / ddh);
		bm_add(&M, row_phi, row_e, -Qe * dnc);
		bm_add(&M, row_phi, row_h, Qe * dpc);
		// Trapped charge in Poisson.  With SRH kinetics the trap quasi-Fermi
		// levels are independent unknowns, so the derivative goes to the trap
		// columns.  Under quasi-equilibrium they are slaved to xn/xp, so by
		// the chain rule d(nt)/d(xn) = dnt and the SAME derivative belongs on
		// the free-carrier column instead - putting it on the trap column
		// there would leave Poisson blind to the trapped charge's response
		// and destroy the quadratic convergence.
		for (b = 0; b < NB; b++)
		{
			if (e_band_active(l, b))
				bm_add(&M, row_phi, traps_equilibrium(l) ? row_e : i * m + 3 + b,
				       -Qe * dev->dnt[b][i]);
			if (h_band_active(l, b))
				bm_add(&M, row_phi, traps_equilibrium(l) ? row_h : i * m + 3 + NB + b,
				       Qe * dev->dpt[b][i]);
		}

		// -------------------------- SG fluxes ----------------------------
		double xil = 3.0 * Qe * (Ecc - Ecl) / (wnc + wnl);
		double xir = 3.0 * Qe * (Ecr - Ecc) / (wnr + wnc);
		double xipl = 3.0 * Qe * (Evc - Evl) / (wpc + wpl);
		double xipr = 3.0 * Qe * (Evr - Evc) / (wpr + wpc);
		double dxil_dphic = -3.0 * Qe / (wnc + wnl);   // dEc/dphi = -1
		double dxil_dphil = 3.0 * Qe / (wnc + wnl);
		double dxir_dphic = 3.0 * Qe / (wnr + wnc);
		double dxir_dphir = -3.0 * Qe / (wnr + wnc);
		double dxipl_dphic = -3.0 * Qe / (wpc + wpl);
		double dxipl_dphil = 3.0 * Qe / (wpc + wpl);
		double dxipr_dphic = 3.0 * Qe / (wpr + wpc);
		double dxipr_dphir = -3.0 * Qe / (wpr + wpc);

		double Jnl = (Dnl / dyl) * (bernoulli_B(-xil) * nc - bernoulli_B(xil) * nl);
		double Jnr = (Dnr / dyr) * (bernoulli_B(-xir) * nr - bernoulli_B(xir) * nc);
		double Jpl = (Dpl / dyl) * (bernoulli_B(-xipl) * pl - bernoulli_B(xipl) * pc);
		double Jpr = (Dpr / dyr) * (bernoulli_B(-xipr) * pc - bernoulli_B(xipr) * pr);

		double dJnl_dxnl = -(Dnl / dyl) * bernoulli_B(xil) * dnl;
		double dJnl_dxnc = (Dnl / dyl) * bernoulli_B(-xil) * dnc;
		double dJnr_dxnc = -(Dnr / dyr) * bernoulli_B(xir) * dnc;
		double dJnr_dxnr = (Dnr / dyr) * bernoulli_B(-xir) * dnr;
		double dJnl_dxi = (Dnl / dyl) * (-bernoulli_dB(-xil) * nc - bernoulli_dB(xil) * nl);
		double dJnr_dxi = (Dnr / dyr) * (bernoulli_dB(-xir) * nr + bernoulli_dB(xir) * nc);
		// note: dJnr/dxir = (Dnr/dyr)(-dB(-xir)*(-1)*nr - dB(xir)*nc)?  Derive:
		// d/dxi [B(-xi)] = -dB(-xi); d/dxi [B(xi)] = dB(xi)
		// dJnr/dxir = (Dnr/dyr)(-dB(-xir)*nr - dB(xir)*nc)
		dJnr_dxi = (Dnr / dyr) * (-bernoulli_dB(-xir) * nr - bernoulli_dB(xir) * nc);

		double dJpl_dxpl = (Dpl / dyl) * bernoulli_B(-xipl) * dpl;
		double dJpl_dxpc = -(Dpl / dyl) * bernoulli_B(xipl) * dpc;
		double dJpr_dxpc = (Dpr / dyr) * bernoulli_B(-xipr) * dpc;
		double dJpr_dxpr = -(Dpr / dyr) * bernoulli_B(xipr) * dpr;
		double dJpl_dxip = (Dpl / dyl) * (-bernoulli_dB(-xipl) * pl - bernoulli_dB(xipl) * pc);
		double dJpr_dxip = (Dpr / dyr) * (-bernoulli_dB(-xipr) * pc - bernoulli_dB(xipr) * pr);

		// Field-dependent (thermionic) contacts: the ghost density is a
		// function of phi at this node, so dJ/dphi picks up an extra term
		// through the ghost that a plain Dirichlet ghost would not have.
		// Only the majority carrier's ghost carries the field dependence.
		double dJnl_dphi_ghost = 0.0, dJpl_dphi_ghost = 0.0;
		double dJnr_dphi_ghost = 0.0, dJpr_dphi_ghost = 0.0;
		if (i == 0 && dev->anode.model == CONTACT_THERMIONIC)
		{
			// Jnl/Jpl depend on nl/pl:  dJnl/dnl = -(Dnl/dyl)*B(xil)
			//                           dJpl/dpl =  (Dpl/dyl)*B(-xipl)
			if (dev->anode.majority == MAJORITY_ELECTRON)
				dJnl_dphi_ghost = -(Dnl / dyl) * bernoulli_B(xil) * dev->anode.dnmaj_dphi;
			else
				dJpl_dphi_ghost = (Dpl / dyl) * bernoulli_B(-xipl) * dev->anode.dnmaj_dphi;
		}
		if (i == N - 1 && dev->cathode.model == CONTACT_THERMIONIC)
		{
			// Jnr/Jpr depend on nr/pr:  dJnr/dnr =  (Dnr/dyr)*B(-xir)
			//                           dJpr/dpr = -(Dpr/dyr)*B(xipr)
			if (dev->cathode.majority == MAJORITY_ELECTRON)
				dJnr_dphi_ghost = (Dnr / dyr) * bernoulli_B(-xir) * dev->cathode.dnmaj_dphi;
			else
				dJpr_dphi_ghost = -(Dpr / dyr) * bernoulli_B(xipr) * dev->cathode.dnmaj_dphi;
		}

		if (i == 0) { dev->Jn[0] = Qe * Jnl; dev->Jp[0] = Qe * Jpl; dev->J_anode = Qe * (Jnl + Jpl); }
		if (i == N - 1) dev->J_cathode = Qe * (Jnr + Jpr);
		dev->Jn[i] = Qe * 0.5 * (Jnl + Jnr);
		dev->Jp[i] = Qe * 0.5 * (Jpl + Jpr);

		// ------------------------- recombination -------------------------
		double Bfree = 0.0, Rfree = 0.0, Rauger = 0.0, Rss = 0.0;
		double dRfree_dxn = 0.0, dRfree_dxp = 0.0;
		double dRauger_dxn = 0.0, dRauger_dxp = 0.0;
		double dRss_dxn = 0.0, dRss_dxp = 0.0;
		double neqpeq = dev->n_eq[i] * dev->p_eq[i];
		struct recomb_config *rc = &l->recomb;

		if (with_recomb)
		{
			Bfree = recomb_get_B(rc, munc, mupc, dev->epsr[i]);
			Rfree = Bfree * (nc * pc - neqpeq);
			dRfree_dxn = Bfree * dnc * pc;
			dRfree_dxp = Bfree * nc * dpc;


			if (rc->auger_enabled)
			{
				double Cn = rc->Cn, Cp = rc->Cp;
				Rauger = (Cn * nc + Cp * pc) * (nc * pc - neqpeq);
				dRauger_dxn = (Cn * dnc) * (nc * pc - neqpeq) + (Cn * nc + Cp * pc) * (dnc * pc);
				dRauger_dxp = (Cp * dpc) * (nc * pc - neqpeq) + (Cn * nc + Cp * pc) * (nc * dpc);
			}
			if (rc->ss_srh_enabled && (rc->tau_n > 0.0 || rc->tau_p > 0.0))
			{
				double den = rc->tau_p * (nc + rc->n1) + rc->tau_n * (pc + rc->p1);
				Rss = (nc * pc - neqpeq) / den;
				dRss_dxn = (dnc * pc) / den - (nc * pc - neqpeq) * rc->tau_p * dnc / (den * den);
				dRss_dxp = (nc * dpc) / den - (nc * pc - neqpeq) * rc->tau_n * dpc / (den * den);
			}

		}

		// Trap-assisted (SRH) recombination.  Under quasi-equilibrium traps
		// this is identically zero: the trapped carriers share the free
		// quasi-Fermi level, exchange no net particles with the bands, and so
		// contribute to Poisson only - the continuity equations and hence the
		// drift-diffusion currents see free carriers alone.
		double Rtrapn = 0.0, Rtrapp = 0.0;
		double dRtrapn_dxn = 0.0, dRtrapp_dxp = 0.0, dRtrapn_dxp = 0.0, dRtrapp_dxn = 0.0;
		for (b = 0; !traps_equilibrium(l) && b < NB; b++)
		{
			if (e_band_active(l, b))
			{
				Rtrapn += nc * dev->srh_n_r1[b][i] - dev->srh_n_r2[b][i];
				dRtrapn_dxn += dnc * dev->srh_n_r1[b][i];
				Rtrapp += pc * dev->srh_n_r3[b][i] - dev->srh_n_r4[b][i];
				dRtrapp_dxp += dpc * dev->srh_n_r3[b][i];
			}
			if (h_band_active(l, b))
			{
				Rtrapp += pc * dev->srh_p_r1[b][i] - dev->srh_p_r2[b][i];
				dRtrapp_dxp += dpc * dev->srh_p_r1[b][i];
				Rtrapn += nc * dev->srh_p_r3[b][i] - dev->srh_p_r4[b][i];
				dRtrapn_dxn += dnc * dev->srh_p_r3[b][i];
			}
		}

		double R_e = Rfree + Rauger + Rss + Rtrapn;
		double R_h = Rfree + Rauger + Rss + Rtrapp;
		dev->Rfree[i] = Rfree;
		dev->Rauger[i] = Rauger;
		dev->Rsrh[i] = Rss + 0.5 * (Rtrapn + Rtrapp);
		dev->Rtot[i] = Rfree + Rauger + Rss + 0.5 * (Rtrapn + Rtrapp);

		// --------------------- electron continuity -----------------------
		double F_e = (Jnr - Jnl) / ddh - R_e;
		M.b[row_e] = -F_e;
		if (i != 0)
		{
			bm_add(&M, row_e, (i - 1) * m + VAR_XN, -dJnl_dxnl / ddh);
			bm_add(&M, row_e, (i - 1) * m + VAR_PHI, -(dJnl_dxi * dxil_dphil) / ddh);
		}
		bm_add(&M, row_e, row_e, (dJnr_dxnc - dJnl_dxnc) / ddh - dRfree_dxn - dRauger_dxn - dRss_dxn - dRtrapn_dxn);
		bm_add(&M, row_e, row_phi, (dJnr_dxi * dxir_dphic - dJnl_dxi * dxil_dphic
		                            + dJnr_dphi_ghost - dJnl_dphi_ghost) / ddh);
		bm_add(&M, row_e, row_h, -dRfree_dxp - dRauger_dxp - dRss_dxp - dRtrapn_dxp);
		if (i != N - 1)
		{
			bm_add(&M, row_e, (i + 1) * m + VAR_XN, dJnr_dxnr / ddh);
			bm_add(&M, row_e, (i + 1) * m + VAR_PHI, (dJnr_dxi * dxir_dphir) / ddh);
		}
		for (b = 0; !traps_equilibrium(l) && b < NB; b++)
		{
			if (e_band_active(l, b))
				bm_add(&M, row_e, i * m + 3 + b, -(nc * dev->dsrh_n_r1[b][i] - dev->dsrh_n_r2[b][i]));
			if (h_band_active(l, b))
				bm_add(&M, row_e, i * m + 3 + NB + b, -(nc * dev->dsrh_p_r3[b][i] - dev->dsrh_p_r4[b][i]));
		}

		// ----------------------- hole continuity -------------------------
		double F_h = (Jpr - Jpl) / ddh + R_h;
		M.b[row_h] = -F_h;
		if (i != 0)
		{
			bm_add(&M, row_h, (i - 1) * m + VAR_XP, -dJpl_dxpl / ddh);
			bm_add(&M, row_h, (i - 1) * m + VAR_PHI, -(dJpl_dxip * dxipl_dphil) / ddh);
		}
		bm_add(&M, row_h, row_h, (dJpr_dxpc - dJpl_dxpc) / ddh + dRfree_dxp + dRauger_dxp + dRss_dxp + dRtrapp_dxp);
		bm_add(&M, row_h, row_phi, (dJpr_dxip * dxipr_dphic - dJpl_dxip * dxipl_dphic
		                            + dJpr_dphi_ghost - dJpl_dphi_ghost) / ddh);
		bm_add(&M, row_h, row_e, dRfree_dxn + dRauger_dxn + dRss_dxn + dRtrapp_dxn);
		if (i != N - 1)
		{
			bm_add(&M, row_h, (i + 1) * m + VAR_XP, dJpr_dxpr / ddh);
			bm_add(&M, row_h, (i + 1) * m + VAR_PHI, (dJpr_dxip * dxipr_dphir) / ddh);
		}
		for (b = 0; !traps_equilibrium(l) && b < NB; b++)
		{
			if (e_band_active(l, b))
				bm_add(&M, row_h, i * m + 3 + b, pc * dev->dsrh_n_r3[b][i] - dev->dsrh_n_r4[b][i]);
			if (h_band_active(l, b))
				bm_add(&M, row_h, i * m + 3 + NB + b, pc * dev->dsrh_p_r1[b][i] - dev->dsrh_p_r2[b][i]);
		}

		// ------------------------ trap band rows -------------------------
		//
		// SRH kinetics: each band's quasi-Fermi level is the unknown that
		// makes capture balance emission.
		//
		// Quasi-equilibrium: the trap level is pinned to the free-carrier
		// quasi-Fermi level of the same sign (xt = xn, xpt = xp), the same
		// identity row used for bands that hold no traps.  Occupancy then
		// follows the free carriers through trap_get_n/p in update_arrays,
		// and the trapped charge reaches the solve via Poisson alone.
		int eq_traps = traps_equilibrium(l);
		for (b = 0; b < NB; b++)
		{
			int row_t = i * m + 3 + b;
			int row_pt = i * m + 3 + NB + b;
			if (e_band_active(l, b) && !eq_traps)
			{
				double Ft = nc * dev->srh_n_r1[b][i] - dev->srh_n_r2[b][i]
				          - pc * dev->srh_n_r3[b][i] + dev->srh_n_r4[b][i];
				M.b[row_t] = -Ft;
				bm_add(&M, row_t, row_t, nc * dev->dsrh_n_r1[b][i] - dev->dsrh_n_r2[b][i]
				                       - pc * dev->dsrh_n_r3[b][i] + dev->dsrh_n_r4[b][i]);
				bm_add(&M, row_t, row_e, dnc * dev->srh_n_r1[b][i]);
				bm_add(&M, row_t, row_h, -dpc * dev->srh_n_r3[b][i]);
			}
			else
			{
				M.b[row_t] = -(dev->xt[b][i] - dev->xn[i]);
				bm_add(&M, row_t, row_t, 1.0);
				bm_add(&M, row_t, row_e, -1.0);
			}
			if (h_band_active(l, b) && !eq_traps)
			{
				double Fpt = pc * dev->srh_p_r1[b][i] - dev->srh_p_r2[b][i]
				           - nc * dev->srh_p_r3[b][i] + dev->srh_p_r4[b][i];
				M.b[row_pt] = -Fpt;
				bm_add(&M, row_pt, row_pt, pc * dev->dsrh_p_r1[b][i] - dev->dsrh_p_r2[b][i]
				                         - nc * dev->dsrh_p_r3[b][i] + dev->dsrh_p_r4[b][i]);
				bm_add(&M, row_pt, row_h, dpc * dev->srh_p_r1[b][i]);
				bm_add(&M, row_pt, row_e, -dnc * dev->srh_p_r3[b][i]);
			}
			else
			{
				M.b[row_pt] = -(dev->xpt[b][i] - dev->xp[i]);
				bm_add(&M, row_pt, row_pt, 1.0);
				bm_add(&M, row_pt, row_h, -1.0);
			}
		}
	}

	// row equilibration then banded solve
	{
		int row, col;
		for (row = 0; row < n; row++)
		{
			double mx = fabs(M.b[row]);
			int c0 = row - kl; if (c0 < 0) c0 = 0;
			int c1 = row + ku; if (c1 > n - 1) c1 = n - 1;
			for (col = c0; col <= c1; col++)
			{
				double a = fabs(M.AB[(size_t)(kl + ku + row - col) * n + col]);
				if (a > mx) mx = a;
			}
			if (mx > 0.0)
			{
				M.b[row] /= mx;
				for (col = c0; col <= c1; col++)
					M.AB[(size_t)(kl + ku + row - col) * n + col] /= mx;
			}
		}
	}

	if (band_lu_solve(M.AB, M.b, n, kl, ku) != 0)
	{
		free(M.AB); free(M.b);
		return -1.0;
	}

	// clamped update (gpvdm plugins/newton/update.c: update_solver_vars).
	// The convergence norm weights quasi-Fermi updates by the local carrier
	// presence: where n or p is a vanishing fraction of a single carrier
	// per m^3 the continuity equation is pure round-off and its update is
	// numerically chaotic but physically meaningless.
	{
		double clamp_kT = cfg->newton_clamp * KB * 300.0 / Qe;
		for (i = 0; i < N; i++)
		{
			double du, w;
			du = M.b[i * m + VAR_PHI];
			dev->phi[i] += du / (1.0 + fabs(du / clamp_kT));
			if (fabs(du) > max_du) max_du = fabs(du);

			du = M.b[i * m + VAR_XN];
			dev->xn[i] += du / (1.0 + fabs(du / clamp_kT));
			w = dev->n[i] / (dev->n[i] + 1.0);
			if (fabs(du) * w > max_du) max_du = fabs(du) * w;

			du = M.b[i * m + VAR_XP];
			dev->xp[i] += du / (1.0 + fabs(du / clamp_kT));
			w = dev->p[i] / (dev->p[i] + 1.0);
			if (fabs(du) * w > max_du) max_du = fabs(du) * w;

			for (b = 0; b < NB; b++)
			{
				du = M.b[i * m + 3 + b];
				dev->xt[b][i] += du / (1.0 + fabs(du / clamp_kT));
				w = dev->n[i] / (dev->n[i] + 1.0);
				if (fabs(du) * w > max_du) max_du = fabs(du) * w;
				du = M.b[i * m + 3 + NB + b];
				dev->xpt[b][i] += du / (1.0 + fabs(du / clamp_kT));
				w = dev->p[i] / (dev->p[i] + 1.0);
				if (fabs(du) * w > max_du) max_du = fabs(du) * w;
			}
		}
	}

	free(M.AB);
	free(M.b);
	return max_du;
}

int newton_solve(struct device *dev, struct sim_config *cfg, int is_equilibrium, int quiet)
{
	int it;
	double err = 1e30;
	int has_lagged = 0;
	int l;

	for (l = 0; l < dev->nlayers; l++)
		if (dev->layers[l].mun.model != MOB_CONSTANT || dev->layers[l].mup.model != MOB_CONSTANT)
			has_lagged = 1;
	// NB: field-dependent contacts are no longer a reason to lag - their
	// dependence on phi is now differentiated into the Jacobian
	// (contact_dnmaj_dphi / dJ*_dphi_ghost), so they stay live throughout.

	// Field/density-dependent mobility is applied Gummel-style: lag it while
	// the solution settles, then hold it fixed so Newton can converge to
	// machine tolerance (the equilibrium solution never depends on mobility).
	dev->freeze_mobility = is_equilibrium ? 1 : 0;

	for (it = 0; it < cfg->max_newton_steps; it++)
	{
		if (!is_equilibrium && has_lagged && dev->freeze_mobility == 0
		    && (it > 25 || err < 1e-4))
			dev->freeze_mobility = 2;

		// The Scott-Malliaras ghost density is refreshed every iteration and
		// its d/dphi is in the Jacobian, so it converges with Newton rather
		// than needing to be frozen.
		contacts_update(dev);
		update_arrays(dev);
		err = newton_iteration(dev, cfg, is_equilibrium ? 0 : 1);
		if (err < 0.0)
		{
			fprintf(stderr, "newton: singular matrix at V=%g\n", dev->Vapplied);
			return -1;
		}
		if (!quiet)
			printf("  newton %3d  max|du|=%8.2e%s\n", it, err,
			       (dev->freeze_mobility == 2) ? " (mobility held)" : "");
		if (err < cfg->newton_tol
		    || (it > 60 && err < cfg->newton_tol_relaxed))
		{
			update_arrays(dev);
			newton_iteration(dev, cfg, is_equilibrium ? 0 : 1); // refresh J/R arrays at solution
			return 0;
		}
	}
	fprintf(stderr, "newton: no convergence at V=%g (err=%g)\n", dev->Vapplied, err);
	return 1;
}
