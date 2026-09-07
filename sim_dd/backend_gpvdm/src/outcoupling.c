// outcoupling.c - transfer-matrix outcoupling of the emission zone.
//
// The optical core is the same port of gpvdm's transfer-matrix light model
// as ../../oled_outcoupling.c (see that file and README.md):
//  * complex index nbar = n - i k, interface r/t coefficients and the
//    coupled forward/backward field equations from
//    gpvdm plugins/light_full/light.c + liblight/light_utils.c,
//  * solved with a banded complex LU instead of UMFPACK.
//
// Integration with the electrical solver: the emission profile is the
// bimolecular recombination profile Rfree(x) inside the chosen electrical
// layer, mapped onto the '*' layer of the optical stack (the exciton module
// is a stub - see exciton.c - so excitons decay where they form).  For each
// wavelength and each dipole position an upward and a downward solve are
// summed incoherently; the outcoupling efficiency is the energy fraction
// escaping through the top of the optical stack:
//     eta = P_top / (P_top + P_bottom + P_absorbed)
// weighted by Rfree(x) and by the emission spectrum.
//
// Limitation (inherited from gpvdm's optical model): normal incidence only,
// no in-plane wavevector integration, so waveguide/plasmon losses are not
// captured.  See the header of ../../oled_outcoupling.c.

#include "oled.h"
#include <complex.h>

#define OC_MAX_LAYERS 32
#define OC_MAX_NK 1024
#define OC_KL 2
#define OC_KU 2

struct oc_layer
{
	char name[STR];
	double thickness;
	int emitter;
	int use_table;
	double n_const, k_const;
	int tpoints;
	double tlam[OC_MAX_NK], tn[OC_MAX_NK], tk[OC_MAX_NK];
};

struct oc_stack
{
	int nlayers;
	struct oc_layer layers[OC_MAX_LAYERS];
	int emitter;
	double total;
};

struct oc_mesh
{
	int N;
	double dy;
	int *lay;
	double *n, *alpha;
	double complex *nbar, *r, *t, *Ep, *En;
};

static double oc_interp(const double *x, const double *y, int n, double xv)
{
	int i;
	if (xv <= x[0]) return y[0];
	if (xv >= x[n - 1]) return y[n - 1];
	for (i = 0; i < n - 1; i++)
		if (xv <= x[i + 1])
			return y[i] + (xv - x[i]) / (x[i + 1] - x[i]) * (y[i + 1] - y[i]);
	return y[n - 1];
}

static int oc_stack_load(struct oc_stack *s, const char *path)
{
	FILE *f = fopen(path, "r");
	char line[512];
	if (f == NULL) { fprintf(stderr, "outcoupling: can't open %s\n", path); return -1; }
	s->nlayers = 0; s->emitter = -1; s->total = 0.0;
	while (fgets(line, sizeof(line), f) != NULL)
	{
		char name[STR], a[STR], b[STR];
		double d;
		char *p = line;
		while (*p == ' ' || *p == '\t') p++;
		if (*p == '#' || *p == '\n' || *p == '\0') continue;
		if (s->nlayers >= OC_MAX_LAYERS) { fclose(f); return -1; }
		{
			struct oc_layer *l = &s->layers[s->nlayers];
			int nf;
			memset(l, 0, sizeof(*l));
			if (*p == '*') { l->emitter = 1; p++; }
			nf = sscanf(p, "%255s %lf %255s %255s", name, &d, a, b);
			if (nf < 3) continue;
			strncpy(l->name, name, STR - 1);
			l->thickness = d;
			if (strncmp(a, "file:", 5) == 0)
			{
				FILE *nk = fopen(a + 5, "r");
				char l2[512];
				if (nk == NULL) { fprintf(stderr, "outcoupling: can't open nk file %s\n", a + 5); fclose(f); return -1; }
				l->use_table = 1;
				while (fgets(l2, sizeof(l2), nk) != NULL && l->tpoints < OC_MAX_NK)
				{
					double lam, nn, kk;
					if (l2[0] == '#') continue;
					if (sscanf(l2, "%lf %lf %lf", &lam, &nn, &kk) == 3)
					{
						l->tlam[l->tpoints] = lam; l->tn[l->tpoints] = nn; l->tk[l->tpoints] = kk;
						l->tpoints++;
					}
				}
				fclose(nk);
			}
			else
			{
				l->n_const = atof(a);
				l->k_const = atof(b);
			}
			if (l->emitter) s->emitter = s->nlayers;
			s->total += d;
			s->nlayers++;
		}
	}
	fclose(f);
	if (s->nlayers < 3 || s->emitter < 0)
	{
		fprintf(stderr, "outcoupling: stack needs >=3 layers and one '*' emitter\n");
		return -1;
	}
	return 0;
}

static void oc_mesh_build(struct oc_mesh *m, const struct oc_stack *s, int points)
{
	int i;
	m->N = points;
	m->dy = s->total / points;
	m->lay = xmalloc(sizeof(int) * points);
	m->n = xmalloc(sizeof(double) * points);
	m->alpha = xmalloc(sizeof(double) * points);
	m->nbar = xmalloc(sizeof(double complex) * points);
	m->r = xmalloc(sizeof(double complex) * points);
	m->t = xmalloc(sizeof(double complex) * points);
	m->Ep = xmalloc(sizeof(double complex) * points);
	m->En = xmalloc(sizeof(double complex) * points);
	for (i = 0; i < points; i++)
	{
		double y = (i + 0.5) * m->dy, edge = 0.0;
		int j, lay = s->nlayers - 1;
		for (j = 0; j < s->nlayers; j++)
		{
			if (y < edge + s->layers[j].thickness) { lay = j; break; }
			edge += s->layers[j].thickness;
		}
		m->lay[i] = lay;
	}
}

// gpvdm liblight/light_utils.c: light_calculate_complex_n_zxl()
static void oc_set_lambda(struct oc_mesh *m, const struct oc_stack *s, double lam)
{
	int y;
	for (y = 0; y < m->N; y++)
	{
		const struct oc_layer *l = &s->layers[m->lay[y]];
		double n, k;
		if (l->use_table)
		{
			n = oc_interp(l->tlam, l->tn, l->tpoints, lam);
			k = oc_interp(l->tlam, l->tk, l->tpoints, lam);
		}
		else { n = l->n_const; k = l->k_const; }
		m->n[y] = n;
		m->alpha[y] = 4.0 * OLED_PI * k / lam;
	}
	for (y = 0; y < m->N; y++)
	{
		double kc = m->alpha[y] * (lam / (4.0 * OLED_PI));
		double nr = (y == m->N - 1) ? m->n[y] : m->n[y + 1];
		double kr = (y == m->N - 1) ? kc : m->alpha[y + 1] * (lam / (4.0 * OLED_PI));
		double complex n0 = m->n[y] - kc * I;
		double complex n1 = nr - kr * I;
		m->nbar[y] = n0;
		m->r[y] = (n0 - n1) / (n0 + n1);
		m->t[y] = (2.0 * n0) / (n0 + n1);
	}
}

static void oc_band_set(double complex *AB, int n, int i, int j, double complex v)
{
	AB[(size_t)(OC_KL + OC_KU + i - j) * n + j] = v;
}

static int oc_band_solve(double complex *AB, double complex *b, int n)
{
	int i, j, k;
	for (k = 0; k < n; k++)
	{
		int last = (k + OC_KL < n - 1) ? k + OC_KL : n - 1;
		int imax = k;
		double amax = cabs(AB[(size_t)(OC_KL + OC_KU) * n + k]);
		for (i = k + 1; i <= last; i++)
		{
			double a = cabs(AB[(size_t)(OC_KL + OC_KU + i - k) * n + k]);
			if (a > amax) { amax = a; imax = i; }
		}
		if (amax == 0.0) return -1;
		if (imax != k)
		{
			int jlast = (k + OC_KL + OC_KU < n - 1) ? k + OC_KL + OC_KU : n - 1;
			for (j = k; j <= jlast; j++)
			{
				double complex t = AB[(size_t)(OC_KL + OC_KU + k - j) * n + j];
				AB[(size_t)(OC_KL + OC_KU + k - j) * n + j] = AB[(size_t)(OC_KL + OC_KU + imax - j) * n + j];
				AB[(size_t)(OC_KL + OC_KU + imax - j) * n + j] = t;
			}
			{ double complex t = b[k]; b[k] = b[imax]; b[imax] = t; }
		}
		for (i = k + 1; i <= last; i++)
		{
			double complex mfac = AB[(size_t)(OC_KL + OC_KU + i - k) * n + k] / AB[(size_t)(OC_KL + OC_KU) * n + k];
			AB[(size_t)(OC_KL + OC_KU + i - k) * n + k] = 0.0;
			{
				int jlast = (k + OC_KL + OC_KU < n - 1) ? k + OC_KL + OC_KU : n - 1;
				for (j = k + 1; j <= jlast; j++)
					AB[(size_t)(OC_KL + OC_KU + i - j) * n + j] -= mfac * AB[(size_t)(OC_KL + OC_KU + k - j) * n + j];
			}
			b[i] -= mfac * b[k];
		}
	}
	for (i = n - 1; i >= 0; i--)
	{
		double complex sum = b[i];
		int jlast = (i + OC_KL + OC_KU < n - 1) ? i + OC_KL + OC_KU : n - 1;
		for (j = i + 1; j <= jlast; j++)
			sum -= AB[(size_t)(OC_KL + OC_KU + i - j) * n + j] * b[j];
		b[i] = sum / AB[(size_t)(OC_KL + OC_KU) * n + i];
	}
	return 0;
}

// gpvdm plugins/light_full/light.c: light_dll_solve_lam_slice() equations
static int oc_tm_solve(struct oc_mesh *m, double lam, const double complex *src_p, const double complex *src_n)
{
	int N = m->N, n = 2 * N, y, ret;
	double dy = m->dy;
	double complex *AB = calloc((size_t)(2 * OC_KL + OC_KU + 1) * n, sizeof(double complex));
	double complex *b = calloc(n, sizeof(double complex));

	for (y = 0; y < N; y++)
	{
		double complex rc = m->r[y], tc = m->t[y];
		double complex rl = (y == 0) ? m->r[y] : m->r[y - 1];
		double complex tl = (y == 0) ? m->t[y] : m->t[y - 1];
		double complex xi_c = ((2 * OLED_PI) / lam) * m->nbar[y];
		double complex xi_r = ((2 * OLED_PI) / lam) * ((y == N - 1) ? m->nbar[y] : m->nbar[y + 1]);
		int rf = 2 * y, rb = 2 * y + 1;

		if (y != 0) oc_band_set(AB, n, rf, 2 * (y - 1), -tl);
		oc_band_set(AB, n, rf, 2 * y, cexp(xi_c * dy * I));
		oc_band_set(AB, n, rf, 2 * y + 1, rl * cexp(-xi_c * dy * I));
		b[rf] = (src_p != NULL) ? src_p[y] : 0.0;

		oc_band_set(AB, n, rb, 2 * y + 1, -tc);
		if (y != N - 1)
		{
			oc_band_set(AB, n, rb, 2 * (y + 1), rc * cexp(xi_r * dy * I));
			oc_band_set(AB, n, rb, 2 * (y + 1) + 1, cexp(-xi_r * dy * I));
		}
		b[rb] = (src_n != NULL) ? src_n[y] : 0.0;
	}

	ret = oc_band_solve(AB, b, n);
	if (ret == 0)
		for (y = 0; y < N; y++) { m->Ep[y] = b[2 * y]; m->En[y] = b[2 * y + 1]; }
	free(AB); free(b);
	return ret;
}

// gpvdm liblight/light_utils.c: light_cal_photon_density_y() Poynting form
static void oc_powers(const struct oc_mesh *m, double *top, double *bottom, double *absorbed)
{
	int y;
	*absorbed = 0.0;
	for (y = 0; y < m->N; y++)
	{
		double complex Et = m->Ep[y] + m->En[y];
		double poynting = 0.5 * EPS0 * C_LIGHT * m->n[y] * (creal(Et) * creal(Et) + cimag(Et) * cimag(Et));
		*absorbed += poynting * m->alpha[y] * m->dy;
	}
	*top = 0.5 * EPS0 * C_LIGHT * m->n[0] * cabs(m->En[0]) * cabs(m->En[0]);
	*bottom = 0.5 * EPS0 * C_LIGHT * m->n[m->N - 1] * cabs(m->Ep[m->N - 1]) * cabs(m->Ep[m->N - 1]);
}

static void oc_eta_at(struct oc_mesh *m, double lam, int ys, double *eta_top, double *eta_bottom, double *eta_abs)
{
	double complex *src = calloc(m->N, sizeof(double complex));
	double tu, bu, au, td, bd, ad, tot;
	src[ys] = 1.0;
	oc_tm_solve(m, lam, NULL, src);
	oc_powers(m, &tu, &bu, &au);
	memset(src, 0, sizeof(double complex) * m->N);
	src[ys] = 1.0;
	oc_tm_solve(m, lam, src, NULL);
	oc_powers(m, &td, &bd, &ad);
	free(src);
	tot = tu + bu + au + td + bd + ad;
	*eta_top = (tu + td) / tot;
	*eta_bottom = (bu + bd) / tot;
	*eta_abs = (au + ad) / tot;
}

// ---------------------------------------------------------------- driver ---

int outcoupling_run(struct device *dev, struct sim_config *cfg)
{
	struct oc_stack s;
	struct oc_mesh m;
	int el = -1, i, li, pi;
	int first = -1, last = -1;
	char path[STR * 2];
	FILE *f;

	if (oc_stack_load(&s, cfg->optical_stack) != 0)
		return -1;
	oc_mesh_build(&m, &s, cfg->mesh_points_optical);

	// electrical layer that emits
	for (i = 0; i < dev->nlayers; i++)
		if (strcmp(dev->layers[i].name, cfg->emission_layer) == 0) el = i;
	if (el < 0)
	{
		fprintf(stderr, "outcoupling: emission_layer '%s' not found in the electrical stack\n", cfg->emission_layer);
		return -1;
	}

	// optical mesh range of the '*' layer
	for (i = 0; i < m.N; i++)
		if (m.lay[i] == s.emitter) { if (first < 0) first = i; last = i; }
	if (first < 0)
	{
		fprintf(stderr, "outcoupling: optical mesh too coarse for the emitter layer\n");
		return -1;
	}

	// emission weights: Rfree(x) inside the emissive electrical layer,
	// resampled by fractional depth onto dipole positions in the optical layer
	{
		int npos = 12;
		double *w = xmalloc(sizeof(double) * npos);
		int *ys = xmalloc(sizeof(int) * npos);
		double wsum = 0.0;
		double x0 = 0.0, xthick = dev->layers[el].thickness;
		double spec_avg_top = 0.0, spec_avg_bot = 0.0, spec_avg_abs = 0.0, spec_w = 0.0;

		for (i = 0; i < el; i++) x0 += dev->layers[i].thickness;

		for (pi = 0; pi < npos; pi++)
		{
			double frac = (pi + 0.5) / npos;
			double xe = x0 + frac * xthick;
			double Rloc = 0.0;
			// nearest electrical node inside the layer
			double best = 1e30;
			for (i = 0; i < dev->N; i++)
				if (dev->lay[i] == el && fabs(dev->x[i] - xe) < best)
				{ best = fabs(dev->x[i] - xe); Rloc = dev->Rfree[i]; }
			if (Rloc < 0.0) Rloc = 0.0;
			w[pi] = Rloc;
			wsum += Rloc;
			ys[pi] = first + (int)((last - first) * frac);
		}
		if (wsum <= 0.0)
		{
			// no recombination yet (e.g. V=0): fall back to a flat profile
			for (pi = 0; pi < npos; pi++) w[pi] = 1.0;
			wsum = npos;
		}

		snprintf(path, sizeof(path), "%s/outcoupling.dat", cfg->output_dir);
		f = fopen(path, "w");
		if (f == NULL) { free(w); free(ys); return -1; }
		fprintf(f, "# transfer-matrix outcoupling (normal incidence)\n");
		fprintf(f, "# emission profile: Rfree(x) of layer '%s' at V=%g\n", cfg->emission_layer, dev->Vapplied);
		fprintf(f, "# lambda_m eta_outcoupled eta_bottom_lost eta_absorbed\n");

		for (li = 0; li < cfg->lam_points; li++)
		{
			double lam = (cfg->lam_points == 1) ? cfg->lam_start
			            : cfg->lam_start + (cfg->lam_stop - cfg->lam_start) * li / (cfg->lam_points - 1);
			double sw = 1.0;
			double et = 0.0, eb = 0.0, ea = 0.0;
			oc_set_lambda(&m, &s, lam);
			for (pi = 0; pi < npos; pi++)
			{
				double t1, b1, a1;
				oc_eta_at(&m, lam, ys[pi], &t1, &b1, &a1);
				et += w[pi] * t1 / wsum;
				eb += w[pi] * b1 / wsum;
				ea += w[pi] * a1 / wsum;
			}
			if (cfg->spectrum_file[0] != '\0')
			{
				// simple two-column spectrum, interpolated
				static double sl[OC_MAX_NK], sv[OC_MAX_NK];
				static int sp = 0, loaded = 0;
				if (!loaded)
				{
					FILE *sf = fopen(cfg->spectrum_file, "r");
					char l2[512];
					if (sf != NULL)
					{
						while (fgets(l2, sizeof(l2), sf) != NULL && sp < OC_MAX_NK)
						{
							double a1, b1;
							if (l2[0] == '#') continue;
							if (sscanf(l2, "%lf %lf", &a1, &b1) == 2) { sl[sp] = a1; sv[sp] = b1; sp++; }
						}
						fclose(sf);
					}
					loaded = 1;
				}
				if (sp >= 2) sw = oc_interp(sl, sv, sp, lam);
			}
			fprintf(f, "%e %f %f %f\n", lam, et, eb, ea);
			spec_avg_top += sw * et; spec_avg_bot += sw * eb; spec_avg_abs += sw * ea;
			spec_w += sw;
		}
		fprintf(f, "# spectrum-averaged: eta_out=%f eta_bottom=%f eta_abs=%f\n",
		        spec_avg_top / spec_w, spec_avg_bot / spec_w, spec_avg_abs / spec_w);
		fclose(f);
		printf("outcoupling: eta_out=%.4f (bottom %.4f, absorbed %.4f) -> %s\n",
		       spec_avg_top / spec_w, spec_avg_bot / spec_w, spec_avg_abs / spec_w, path);
		free(w); free(ys);
	}

	free(m.lay); free(m.n); free(m.alpha); free(m.nbar);
	free(m.r); free(m.t); free(m.Ep); free(m.En);
	return 0;
}
