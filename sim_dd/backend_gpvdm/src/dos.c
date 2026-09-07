// dos.c - density of states models and trap bands.
//
// Mirrors gpvdm's libdos:
//  * Free carriers: Maxwell-Boltzmann analytic, or Fermi-Dirac by numeric
//    integration over a parabolic band (gendosfdgaus.c "fd_look_up_table"),
//    tabulated per material at start-up and interpolated - the in-memory
//    equivalent of gpvdm's cached binary DOS files.  The average carrier
//    energy w feeds the generalised Einstein relation D = mu*(2/3)*w/q
//    exactly as in gpvdm's newton plugin.
//  * Traps: the trap DOS (exponential tail or Gaussian) is discretised into
//    srh_bands bands between srh_start and the band edge; each band becomes
//    a discrete level at its centre with the band-integrated density
//    (gendosfdgaus.c).  The four SRH rate coefficients per band follow
//    gendosfdgaus.c lines 464-467 (electrons) / 477-480 (holes):
//      r1 = vth*sigma_n*Nb*(1-f)                        capture   ( *n )
//      r2 = vth*sigma_n*Nc*exp(Eb/kT)*Nb*f              emission
//      r3 = vth*sigma_p*Nb*f                            capture   ( *p )
//      r4 = vth*sigma_p*Nv*exp((-Eg-Eb)/kT)*Nb*(1-f)    emission
//    with f the Fermi function of the band centre against the trap
//    quasi-Fermi level (the solver unknown).  Because each band is a single
//    level, the rates are evaluated analytically here rather than through
//    gpvdm's interpolation tables - same model, exact derivatives.
//
// The DOS model is selected per material via dos_side.stats / .trap_type,
// so adding e.g. an EGDM-consistent Gauss-Fermi DOS later means adding one
// more case in dos_get_n/dos_get_p and (if needed) a new table builder.

#include "oled.h"

#define FD_TABLE_LEN 4000
#define FD_U_MIN (-2.5)     // eV, EF relative to band edge
#define FD_U_MAX (0.8)
#define FD_E_MAX (1.2)      // eV, integration depth into the band
#define FD_E_POINTS 2400

// Gaussian DOS occupied by Fermi-Dirac statistics:
//
//   n(EF) = Nt * Int g(E) f(E,EF) dE,
//   g(E)  = 1/(sigma sqrt(2 pi)) exp(-E^2 / 2 sigma^2),
//   f     = 1/(1 + exp((E - EF)/kT))
//
// with E measured from the centre of the Gaussian and u = EF - E_centre.
//
// The hole side does NOT apply an explicit (1 - f): callers pass the mirrored
// argument u = Ev - EFp (see dos_get's comment), which turns (1 - f) into f
// under the coordinate flip.  Applying (1 - f) here as well would double-count.
// Verified against an explicit two-band (1 - f) calculation to 9 digits.
//
// Occupancy convention: plain Fermi-Dirac, one carrier per state, no spin
// degeneracy factor.  Some organic-semiconductor treatments instead use
// 1/(1 + (1/2) exp((E - EF)/kT)) for polaron states, which shifts the Fermi
// level by kT ln2; if matching such a reference, that belongs here.
// This is the occupation statistics that belong with the EGDM mobility
// model (Pasveer et al.), which is itself derived for hopping in a
// Gaussian DOS: using a parabolic-band FD integral for the occupation
// while using EGDM for transport describes two different materials.
//
// The equilibrium occupation sits ~sigma^2/kT below the DOS centre, so for
// sigma = 0.15eV at 300K a given Fermi level holds orders of magnitude more
// carriers than the parabolic form predicts.
//
// w_out is the energy fed to the generalised Einstein relation, which the
// solver applies as D = mu*(2/3)*w/q.  For consistency with D/mu =
// (1/q) n / (dn/dEF) the correct choice is w = (3/2) n / (dn/dEF), which
// reduces to (3/2)kT in the non-degenerate limit (recovering D = mu kT/q)
// and rises above it as the Gaussian fills - the enhanced diffusion that
// accompanies energetic disorder.  Note this is NOT the mean carrier
// energy, which is negative for a Gaussian DOS and would give D < 0.
static double gauss_density_integral(double sigma, double u, double T, double *w_out)
{
	const int NE = 3000;
	const double span = 12.0;             // +/- 12 sigma covers the tails
	double lo = -span * sigma, hi = span * sigma;
	double dE = (hi - lo) / NE;
	double kT_eV = KB * T / Qe;
	double sum = 0.0, dsum = 0.0;
	int e;

	if (sigma <= 0.0)
		return 0.0;

	for (e = 0; e < NE; e++)
	{
		double E = lo + (e + 0.5) * dE;
		double g = exp(-(E * E) / (2.0 * sigma * sigma)) / (sigma * sqrt(2.0 * OLED_PI));
		double a = (E - u) / kT_eV;
		double f, dfdu;
		// guard the exponential: f -> 1 for a << 0, f -> 0 for a >> 0
		if (a > 40.0)      { f = exp(-a); dfdu = f / kT_eV; }
		else if (a < -40.0){ f = 1.0;     dfdu = 0.0; }
		else               { double ea = exp(a); f = 1.0 / (1.0 + ea);
		                     dfdu = ea / ((1.0 + ea) * (1.0 + ea) * kT_eV); }
		sum  += g * f * dE;
		dsum += g * dfdu * dE;            // dn/du per eV, before the Nt factor
	}

	if (w_out != NULL)
	{
		// w = (3/2) * n / (dn/dEF), in Joules; both integrals carry the same
		// Nt prefactor so it cancels here.
		double ratio = (dsum > 0.0) ? (sum / dsum) : kT_eV;   // [eV]
		if (ratio < kT_eV) ratio = kT_eV;                     // floor at the MB limit
		*w_out = 1.5 * ratio * Qe;
	}
	return sum;                            // fractional occupancy; scaled by Nc
}

static double fd_density_integral(double meff, double u, double T, double *w_out)
{
	// gendosfdgaus.c free-carrier FD integral: parabolic DOS with
	// effective mass meff*m0, Fermi level u [eV] above the band edge.
	int e;
	double dE = FD_E_MAX / FD_E_POINTS;
	double sum = 0.0, sum_E = 0.0;
	double pre = (1.0 / (2.0 * OLED_PI * OLED_PI)) * pow((2.0 * meff * M0) / (HBAR * HBAR), 1.5);
	double kT_eV = KB * T / Qe;

	for (e = 0; e < FD_E_POINTS; e++)
	{
		double E = (e + 0.5) * dE;
		double rho = sqrt(E * Qe) * pre;
		double f = 1.0 / (1.0 + exp((E - u) / kT_eV));
		sum += rho * Qe * f * dE;
		sum_E += E * Qe * rho * Qe * f * dE;
	}
	if (w_out != NULL)
		*w_out = (sum > 0.0) ? (sum_E / sum) : (1.5 * KB * T);
	return sum;
}

static void fd_table_build(struct fd_table *t, double meff, double T)
{
	int i;
	t->len = FD_TABLE_LEN;
	t->u0 = FD_U_MIN;
	t->du = (FD_U_MAX - FD_U_MIN) / (FD_TABLE_LEN - 1);
	t->n = xmalloc(sizeof(double) * FD_TABLE_LEN);
	t->w = xmalloc(sizeof(double) * FD_TABLE_LEN);
	for (i = 0; i < FD_TABLE_LEN; i++)
	{
		double u = t->u0 + i * t->du;
		t->n[i] = fd_density_integral(meff, u, T, &t->w[i]);
	}
}

// Gaussian tabulation.  The occupied states sit far below the DOS centre,
// so the table has to span further down in u than the parabolic one: the
// range is set from the DOS width itself (+/- span*sigma, with headroom)
// rather than the fixed FD_U_MIN/FD_U_MAX window.
//
// `deep` extends the lower limit so that a caller asking for the MINORITY
// carrier still lands inside the table.  The minority level sits at u = -Eg
// (contacts.c: minority from EF = 0), which for a wide gap is far below
// -12*sigma.  dos_get clamps out-of-range lookups to the table edge, so the
// table must reach -Eg for that lookup to mean anything.
static void gauss_table_build(struct fd_table *t, double sigma, double Nc,
                              double T, double deep)
{
	int i;
	double umin = -(12.0 * sigma + 0.5);
	double umax = 4.0 * sigma + 0.5;
	if (deep > 0.0 && umin > -(deep + 0.5))
		umin = -(deep + 0.5);
	t->len = FD_TABLE_LEN;
	t->u0 = umin;
	t->du = (umax - umin) / (FD_TABLE_LEN - 1);
	t->n = xmalloc(sizeof(double) * FD_TABLE_LEN);
	t->w = xmalloc(sizeof(double) * FD_TABLE_LEN);
	for (i = 0; i < FD_TABLE_LEN; i++)
	{
		double u = t->u0 + i * t->du;
		t->n[i] = Nc * gauss_density_integral(sigma, u, T, &t->w[i]);
	}
}

// carrier density vs u = (EF - Ec) [eV] for electrons; the hole side passes
// u = (Ev - EFp) so the same monotonic-increasing function applies.
static void dos_get(const struct dos_side *d, double u, double T, double *den, double *dden, double *w)
{
	double kT_eV = KB * T / Qe;

	if (d->stats == STATS_BOLTZMANN)
	{
		double n = d->Nc * exp(u / kT_eV);
		*den = n;
		*dden = n / kT_eV;      // d(den)/du, per volt
		*w = 1.5 * KB * T;
		return;
	}

	// Fermi-Dirac (parabolic band) or Gaussian DOS, both from the table
	{
		const struct fd_table *t = &d->fd;
		double pos = (u - t->u0) / t->du;
		int i;
		double fr;
		if (pos < 0.0) pos = 0.0;
		if (pos > t->len - 2) pos = t->len - 2;
		i = (int)pos;
		fr = pos - i;
		*den = t->n[i] + fr * (t->n[i + 1] - t->n[i]);
		*dden = (t->n[i + 1] - t->n[i]) / t->du;
		*w = t->w[i] + fr * (t->w[i + 1] - t->w[i]);
	}
}

void dos_get_n(const struct dos_side *d, double u_eV, double T, double *n, double *dn, double *w)
{
	dos_get(d, u_eV, T, n, dn, w);
}

void dos_get_p(const struct dos_side *d, double u_eV, double T, double *p, double *dp, double *w)
{
	dos_get(d, u_eV, T, p, dp, w);
}

// Invert the DOS: u such that den(u) = n.  (gpvdm get_top_from_n/p)
double dos_u_from_n(const struct dos_side *d, double n, double T)
{
	double kT_eV = KB * T / Qe;
	if (d->stats == STATS_BOLTZMANN)
		return kT_eV * log(n / d->Nc);
	{
		// bracket from the table's own range: the Gaussian table spans a
		// different (wider, deeper) u window than the parabolic one, so the
		// FD_U_MIN/FD_U_MAX constants are not valid bounds for it.
		const struct fd_table *t = &d->fd;
		double lo = t->u0, hi = t->u0 + (t->len - 1) * t->du;

		// A Gaussian DOS holds at most Nc carriers (one per state); asking for
		// more is unphysical and cannot be inverted.  Fail loudly rather than
		// returning a table-edge value that silently poisons the solve.
		if (d->stats == STATS_GAUSSIAN && n >= d->Nc)
		{
			fprintf(stderr,
			        "dos_u_from_n: requested density %.3e m^-3 exceeds the "
			        "Gaussian DOS capacity Nc = %.3e m^-3.\n"
			        "  A Gaussian DOS has a finite number of states; check the "
			        "contact np / Nc settings.\n", n, d->Nc);
			exit(1);
		}
		int it;
		for (it = 0; it < 200; it++)
		{
			double mid = 0.5 * (lo + hi);
			double den, dd, w;
			dos_get(d, mid, T, &den, &dd, &w);
			if (den < n) lo = mid; else hi = mid;
		}
		return 0.5 * (lo + hi);
	}
}

// ------------------------------------------------------------ trap bands ---

static double trap_rho(const struct dos_side *d, double E)
{
	// E in eV, negative into the gap from the band edge (gpvdm convention)
	//
	// For the exponential tail, Ecen shifts the energy the tail is measured
	// FROM (0 = the free-carrier reference, gpvdm's convention).  With a
	// parabolic band that reference is the mobility edge and Ecen = 0 is
	// right.  With a Gaussian transport DOS "E = 0" is the centre of the
	// Gaussian, while the carriers - and hence the electron quasi-Fermi
	// level - sit roughly sigma^2/kT below it, so a tail pinned to the centre
	// has negligible density of states at the Fermi level.
	// Eonset < 0 moves the tail's onset down to the carrier reference so
	// that trap depth is measured from the transport level instead.
	// Eonset = 0 recovers the original gpvdm behaviour exactly.
	if (d->trap_type == TRAPS_EXPONENTIAL)
	{
		double Eshift = E - d->Eonset;
		if (Eshift > 0.0) Eshift = 0.0;   // no trap states above the onset
		return d->Nt * exp(Eshift / d->Et);              // gendosfdgaus dos_exp
	}
	if (d->trap_type == TRAPS_GAUSSIAN)
		return (d->Nt / (d->sigma * sqrt(2.0 * OLED_PI))) *
		       exp(-0.5 * pow((E - d->Ecen) / d->sigma, 2.0));
	return 0.0;
}

void dos_trap_bands_setup(struct dos_side *d)
{
	int b, e;
	const int sub = 400;
	if (d->trap_type == TRAPS_NONE || d->nbands <= 0)
	{
		d->nbands = 0;
		return;
	}
	// Bands span [srh_start, 0] measured from the trap onset, shallowest at
	// the onset (gendosfdgaus).  For the shifted exponential tail the onset
	// is Ecen rather than 0, so the whole window slides with it - otherwise
	// srh_start would truncate the tail before it had decayed.
	double onset = (d->trap_type == TRAPS_EXPONENTIAL) ? d->Eonset : 0.0;
	for (b = 0; b < d->nbands; b++)
	{
		double E0 = onset + d->srh_start * (double)(d->nbands - b) / d->nbands;
		double E1 = onset + d->srh_start * (double)(d->nbands - b - 1) / d->nbands;
		double dE = (E1 - E0) / sub;
		double Nb = 0.0;
		for (e = 0; e < sub; e++)
			Nb += trap_rho(d, E0 + (e + 0.5) * dE) * dE;
		d->band[b].Eb = 0.5 * (E0 + E1);
		d->band[b].Nb = Nb;
	}
}

// Electron-trap band: ut = xt + Xi [eV] is the trap quasi-Fermi argument.
void trap_get_n(const struct dos_side *d, const struct layer *l, int band, double ut_eV, double T,
                double *nt, double *r1, double *r2, double *r3, double *r4,
                double *dnt, double *dr1, double *dr2, double *dr3, double *dr4)
{
	double kT_eV = KB * T / Qe;
	double Eb = d->band[band].Eb;
	double Nb = d->band[band].Nb;
	double f = 1.0 / (1.0 + exp((Eb - ut_eV) / kT_eV));
	double df = f * (1.0 - f) / kT_eV;   // df/dut per volt
	double e_emit = l->dosn.Nc * exp(Eb / kT_eV);
	double h_emit = l->dosp.Nc * exp((-l->Eg - Eb) / kT_eV);

	*nt = Nb * f;
	*dnt = Nb * df;
	*r1 = d->srh_vth * d->srh_sigman * Nb * (1.0 - f);
	*dr1 = -d->srh_vth * d->srh_sigman * Nb * df;
	*r2 = d->srh_vth * d->srh_sigman * e_emit * Nb * f;
	*dr2 = d->srh_vth * d->srh_sigman * e_emit * Nb * df;
	*r3 = d->srh_vth * d->srh_sigmap * Nb * f;
	*dr3 = d->srh_vth * d->srh_sigmap * Nb * df;
	*r4 = d->srh_vth * d->srh_sigmap * h_emit * Nb * (1.0 - f);
	*dr4 = -d->srh_vth * d->srh_sigmap * h_emit * Nb * df;
}

// Hole-trap band, mirrored (gendosfdgaus electrons==FALSE branch):
// ut = xpt - Xi - Eg in the hole frame.
void trap_get_p(const struct dos_side *d, const struct layer *l, int band, double ut_eV, double T,
                double *pt, double *r1, double *r2, double *r3, double *r4,
                double *dpt, double *dr1, double *dr2, double *dr3, double *dr4)
{
	double kT_eV = KB * T / Qe;
	double Eb = d->band[band].Eb;
	double Nb = d->band[band].Nb;
	double f = 1.0 / (1.0 + exp((Eb - ut_eV) / kT_eV));
	double df = f * (1.0 - f) / kT_eV;
	double h_emit = l->dosp.Nc * exp(Eb / kT_eV);
	double e_emit = l->dosn.Nc * exp((-l->Eg - Eb) / kT_eV);

	*pt = Nb * f;
	*dpt = Nb * df;
	*r1 = d->srh_vth * d->srh_sigmap * Nb * (1.0 - f);
	*dr1 = -d->srh_vth * d->srh_sigmap * Nb * df;
	*r2 = d->srh_vth * d->srh_sigmap * h_emit * Nb * f;
	*dr2 = d->srh_vth * d->srh_sigmap * h_emit * Nb * df;
	*r3 = d->srh_vth * d->srh_sigman * Nb * f;
	*dr3 = d->srh_vth * d->srh_sigman * Nb * df;
	*r4 = d->srh_vth * d->srh_sigman * e_emit * Nb * (1.0 - f);
	*dr4 = -d->srh_vth * d->srh_sigman * e_emit * Nb * df;
}

void dos_init_layer(struct layer *l, double T)
{
	// derive effective masses from Nc/Nv (gpvdm stores me/mh; we let the
	// user give Nc/Nv and invert Nc = 2*(m kT / 2 pi hbar^2)^1.5)
	l->dosn.meff = pow(l->dosn.Nc / 2.0, 2.0 / 3.0) * (2.0 * OLED_PI * HBAR * HBAR) / (M0 * KB * T);
	l->dosp.meff = pow(l->dosp.Nc / 2.0, 2.0 / 3.0) * (2.0 * OLED_PI * HBAR * HBAR) / (M0 * KB * T);

	if (l->dosn.stats == STATS_FERMIDIRAC)
		fd_table_build(&l->dosn.fd, l->dosn.meff, T);
	if (l->dosp.stats == STATS_FERMIDIRAC)
		fd_table_build(&l->dosp.fd, l->dosp.meff, T);

	// Gaussian DOS: default the width to the EGDM sigma of the same carrier,
	// so transport and occupation describe one material rather than two.
	if (l->dosn.stats == STATS_GAUSSIAN)
	{
		if (l->dosn.dos_sigma <= 0.0) l->dosn.dos_sigma = l->mun.egdm_sigma;
		gauss_table_build(&l->dosn.fd, l->dosn.dos_sigma, l->dosn.Nc, T, l->Eg);
	}
	if (l->dosp.stats == STATS_GAUSSIAN)
	{
		if (l->dosp.dos_sigma <= 0.0) l->dosp.dos_sigma = l->mup.egdm_sigma;
		gauss_table_build(&l->dosp.fd, l->dosp.dos_sigma, l->dosp.Nc, T, l->Eg);
	}

	dos_trap_bands_setup(&l->dosn);
	dos_trap_bands_setup(&l->dosp);
}
