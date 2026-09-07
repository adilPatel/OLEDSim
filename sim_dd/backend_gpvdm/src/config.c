// config.c - input deck parser.
//
// Plain INI-style text replaces gpvdm's sim.json: [section] headers with
// key = value lines, '#' comments.  Sections:
//   [simulation]          sweep, solver and output settings
//   [layer <name>]        one per device layer, in order from the anode
//   [contact_anode]       x = 0 contact
//   [contact_cathode]     x = L contact
//   [outcoupling]         transfer-matrix outcoupling settings
// See examples/ for annotated decks with every key.

#include "oled.h"

static void strip(char *s)
{
	char *p = strchr(s, '#');
	int n;
	if (p != NULL) *p = '\0';
	n = (int)strlen(s);
	while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' ' || s[n - 1] == '\t'))
		s[--n] = '\0';
}

static int keyis(const char *key, const char *want) { return strcmp(key, want) == 0; }

static void layer_defaults(struct layer *l)
{
	memset(l, 0, sizeof(*l));
	l->points = 0;
	l->Eg = 3.0; l->Xi = 2.0; l->epsr = 3.0; l->Nad = 0.0;
	l->dosn.stats = STATS_BOLTZMANN; l->dosn.Nc = 1e26;
	l->dosp.stats = STATS_BOLTZMANN; l->dosp.Nc = 1e26;
	l->dosn.trap_type = TRAPS_NONE; l->dosp.trap_type = TRAPS_NONE;
	l->dosn.nbands = 0; l->dosp.nbands = 0;
	l->dosn.srh_start = -0.5; l->dosp.srh_start = -0.5;
	l->dosn.srh_sigman = 1e-20; l->dosn.srh_sigmap = 1e-22; l->dosn.srh_vth = 1e5;
	l->dosp.srh_sigman = 1e-22; l->dosp.srh_sigmap = 1e-20; l->dosp.srh_vth = 1e5;
	l->dosn.Et = 0.04; l->dosp.Et = 0.04;
	l->dosn.sigma = 0.1; l->dosp.sigma = 0.1;
	l->dosn.Ecen = -0.5; l->dosp.Ecen = -0.5;
	l->dosn.Eonset = 0.0; l->dosp.Eonset = 0.0;   // gpvdm band-edge reference
	l->mun.model = MOB_CONSTANT; l->mun.mu0 = 1e-8;
	l->mup.model = MOB_CONSTANT; l->mup.mu0 = 1e-8;
	l->mun.egdm_c2 = 0.39; l->mup.egdm_c2 = 0.39;
	l->recomb.model = RECOMB_CONSTANT_B; l->recomb.B = 0.0; l->recomb.langevin_pre = 1.0;
	l->recomb.trap_kinetics = TRAP_KINETIC_SRH;
}

static enum mobility_model parse_mob(const char *v)
{
	if (strcmp(v, "poole_frenkel") == 0) return MOB_POOLE_FRENKEL;
	if (strcmp(v, "egdm") == 0) return MOB_EGDM;
	return MOB_CONSTANT;
}

static enum trap_type parse_traps(const char *v)
{
	if (strcmp(v, "exponential") == 0) return TRAPS_EXPONENTIAL;
	if (strcmp(v, "gaussian") == 0) return TRAPS_GAUSSIAN;
	return TRAPS_NONE;
}

int config_load(const char *path, struct sim_config *cfg, struct device *dev)
{
	FILE *f = fopen(path, "r");
	char line[512], section[STR] = "";
	struct layer *l = NULL;
	struct contact *c = NULL;

	if (f == NULL)
	{
		fprintf(stderr, "can't open deck %s\n", path);
		return -1;
	}

	memset(cfg, 0, sizeof(*cfg));
	memset(dev, 0, sizeof(*dev));
	cfg->T = 300.0;
	cfg->V_start = 0.0; cfg->V_stop = 3.0; cfg->V_step = 0.1;
	cfg->max_newton_steps = 300;
	cfg->newton_tol = 1e-9;
	cfg->newton_tol_relaxed = 1e-6;
	cfg->newton_clamp = 4.0;
	cfg->points_per_layer = 40;
	strcpy(cfg->output_dir, "output");
	cfg->lam_start = 400e-9; cfg->lam_stop = 750e-9; cfg->lam_points = 36;
	cfg->mesh_points_optical = 2000;
	dev->apply_voltage_to_anode = 1;
	dev->anode.model = CONTACT_OHMIC_NP; dev->anode.majority = MAJORITY_HOLE; dev->anode.np = 1e26;
	dev->cathode.model = CONTACT_OHMIC_NP; dev->cathode.majority = MAJORITY_ELECTRON; dev->cathode.np = 1e26;

	while (fgets(line, sizeof(line), f) != NULL)
	{
		char key[STR], val[STR];
		char *eq;
		char *p = line;
		strip(line);
		while (*p == ' ' || *p == '\t') p++;
		if (*p == '\0') continue;

		if (*p == '[')
		{
			char *end = strchr(p, ']');
			if (end == NULL) continue;
			*end = '\0';
			strncpy(section, p + 1, STR - 1);
			l = NULL; c = NULL;
			if (strncmp(section, "layer", 5) == 0)
			{
				if (dev->nlayers >= MAX_LAYERS) { fprintf(stderr, "too many layers\n"); return -1; }
				l = &dev->layers[dev->nlayers++];
				layer_defaults(l);
				{
					const char *nm = section + 5;
					while (*nm == ' ') nm++;
					strncpy(l->name, (*nm != '\0') ? nm : "layer", STR - 1);
				}
			}
			else if (strcmp(section, "contact_anode") == 0) c = &dev->anode;
			else if (strcmp(section, "contact_cathode") == 0) c = &dev->cathode;
			continue;
		}

		eq = strchr(p, '=');
		if (eq == NULL) continue;
		*eq = '\0';
		strncpy(key, p, STR - 1); strip(key);
		{
			char *k = key; int n2 = (int)strlen(k);
			while (n2 > 0 && (k[n2 - 1] == ' ' || k[n2 - 1] == '\t')) k[--n2] = '\0';
		}
		{
			char *v = eq + 1;
			while (*v == ' ' || *v == '\t') v++;
			strncpy(val, v, STR - 1); strip(val);
		}

		if (l != NULL)
		{
			double d = atof(val);
			if (keyis(key, "thickness")) l->thickness = d;
			else if (keyis(key, "points")) l->points = (int)d;
			else if (keyis(key, "Eg")) l->Eg = d;
			else if (keyis(key, "Xi")) l->Xi = d;
			else if (keyis(key, "epsilon_r")) l->epsr = d;
			else if (keyis(key, "Nad")) l->Nad = d;
			else if (keyis(key, "Nc")) l->dosn.Nc = d;
			else if (keyis(key, "Nv")) l->dosp.Nc = d;
			else if (keyis(key, "statistics"))
			{
				enum carrier_stats s = STATS_BOLTZMANN;
				if (strcmp(val, "fermidirac") == 0) s = STATS_FERMIDIRAC;
				else if (strcmp(val, "gaussian") == 0) s = STATS_GAUSSIAN;
				else if (strcmp(val, "boltzmann") != 0)
					fprintf(stderr, "warning: unknown statistics '%s'\n", val);
				l->dosn.stats = s; l->dosp.stats = s;
			}
			else if (keyis(key, "dos_sigma_e")) l->dosn.dos_sigma = d;
			else if (keyis(key, "dos_sigma_h")) l->dosp.dos_sigma = d;
			else if (keyis(key, "mobility_model_e")) l->mun.model = parse_mob(val);
			else if (keyis(key, "mobility_model_h")) l->mup.model = parse_mob(val);
			else if (keyis(key, "mun")) l->mun.mu0 = d;
			else if (keyis(key, "mup")) l->mup.mu0 = d;
			else if (keyis(key, "pf_gamma_e")) l->mun.pf_gamma = d;
			else if (keyis(key, "pf_gamma_h")) l->mup.pf_gamma = d;
			else if (keyis(key, "egdm_sigma_e")) l->mun.egdm_sigma = d;
			else if (keyis(key, "egdm_sigma_h")) l->mup.egdm_sigma = d;
			else if (keyis(key, "egdm_a_e")) l->mun.egdm_a = d;
			else if (keyis(key, "egdm_a_h")) l->mup.egdm_a = d;
			else if (keyis(key, "egdm_c2")) { l->mun.egdm_c2 = d; l->mup.egdm_c2 = d; }
			else if (keyis(key, "egdm_Nt_e")) l->mun.egdm_Nt = d;
			else if (keyis(key, "egdm_Nt_h")) l->mup.egdm_Nt = d;
			else if (keyis(key, "egdm_x_clamp")) { l->mun.egdm_x_clamp = d; l->mup.egdm_x_clamp = d; }
			else if (keyis(key, "egdm_f_clamp")) { l->mun.egdm_f_clamp = d; l->mup.egdm_f_clamp = d; }
			else if (keyis(key, "recomb_model"))
			{
				l->recomb.model = (strcmp(val, "langevin") == 0) ? RECOMB_LANGEVIN : RECOMB_CONSTANT_B;
				// Langevin recombination describes free carriers meeting in
				// the transport bands; the trapped population takes no part.
				// Selecting it therefore also selects quasi-equilibrium traps
				// (Knapp et al. 2010).  An explicit trap_kinetics key placed
				// after this line overrides the choice either way.
				if (l->recomb.model == RECOMB_LANGEVIN)
					l->recomb.trap_kinetics = TRAP_QUASI_EQUILIBRIUM;
			}
			else if (keyis(key, "trap_kinetics"))
				l->recomb.trap_kinetics = (strcmp(val, "equilibrium") == 0)
				                          ? TRAP_QUASI_EQUILIBRIUM : TRAP_KINETIC_SRH;
			else if (keyis(key, "B")) l->recomb.B = d;
			else if (keyis(key, "langevin_pre")) l->recomb.langevin_pre = d;
			else if (keyis(key, "auger_enabled")) l->recomb.auger_enabled = (int)d;
			else if (keyis(key, "Cn")) l->recomb.Cn = d;
			else if (keyis(key, "Cp")) l->recomb.Cp = d;
			else if (keyis(key, "ss_srh_enabled")) l->recomb.ss_srh_enabled = (int)d;
			else if (keyis(key, "tau_n")) l->recomb.tau_n = d;
			else if (keyis(key, "tau_p")) l->recomb.tau_p = d;
			else if (keyis(key, "n1")) l->recomb.n1 = d;
			else if (keyis(key, "p1")) l->recomb.p1 = d;
			else if (keyis(key, "trap_model_e")) l->dosn.trap_type = parse_traps(val);
			else if (keyis(key, "trap_model_h")) l->dosp.trap_type = parse_traps(val);
			else if (keyis(key, "trap_Nt_e")) l->dosn.Nt = d;
			else if (keyis(key, "trap_Nt_h")) l->dosp.Nt = d;
			else if (keyis(key, "trap_Et_e")) l->dosn.Et = d;
			else if (keyis(key, "trap_Et_h")) l->dosp.Et = d;
			else if (keyis(key, "trap_sigma_e")) l->dosn.sigma = d;
			else if (keyis(key, "trap_sigma_h")) l->dosp.sigma = d;
			else if (keyis(key, "trap_Eonset_e")) l->dosn.Eonset = d;
			else if (keyis(key, "trap_Eonset_h")) l->dosp.Eonset = d;
			else if (keyis(key, "trap_Ecen_e")) l->dosn.Ecen = d;
			else if (keyis(key, "trap_Ecen_h")) l->dosp.Ecen = d;
			else if (keyis(key, "srh_bands")) { l->dosn.nbands = (int)d; l->dosp.nbands = (int)d; }
			else if (keyis(key, "srh_start")) { l->dosn.srh_start = d; l->dosp.srh_start = d; }
			else if (keyis(key, "srh_sigman")) { l->dosn.srh_sigman = d; l->dosp.srh_sigman = d; }
			else if (keyis(key, "srh_sigmap")) { l->dosn.srh_sigmap = d; l->dosp.srh_sigmap = d; }
			else if (keyis(key, "srh_sigman_e")) l->dosn.srh_sigman = d;
			else if (keyis(key, "srh_sigmap_e")) l->dosn.srh_sigmap = d;
			else if (keyis(key, "srh_sigman_h")) l->dosp.srh_sigman = d;
			else if (keyis(key, "srh_sigmap_h")) l->dosp.srh_sigmap = d;
			else if (keyis(key, "srh_vth")) { l->dosn.srh_vth = d; l->dosp.srh_vth = d; }
			else fprintf(stderr, "warning: unknown layer key '%s'\n", key);
		}
		else if (c != NULL)
		{
			double d = atof(val);
			if (keyis(key, "model"))
			{
				if (strcmp(val, "ohmic-np") == 0) c->model = CONTACT_OHMIC_NP;
				else if (strcmp(val, "ohmic-barrier") == 0) c->model = CONTACT_OHMIC_BARRIER;
				else if (strcmp(val, "thermionic") == 0) c->model = CONTACT_THERMIONIC;
				else fprintf(stderr, "warning: unknown contact model '%s'\n", val);
			}
			else if (keyis(key, "majority"))
				c->majority = (strcmp(val, "electron") == 0) ? MAJORITY_ELECTRON : MAJORITY_HOLE;
			else if (keyis(key, "np")) c->np = d;
			else if (keyis(key, "barrier")) c->barrier = d;
			else fprintf(stderr, "warning: unknown contact key '%s'\n", key);
		}
		else if (strcmp(section, "simulation") == 0)
		{
			double d = atof(val);
			if (keyis(key, "T")) cfg->T = d;
			else if (keyis(key, "voltage_start")) cfg->V_start = d;
			else if (keyis(key, "voltage_stop")) cfg->V_stop = d;
			else if (keyis(key, "voltage_step")) cfg->V_step = d;
			else if (keyis(key, "max_newton_steps")) cfg->max_newton_steps = (int)d;
			else if (keyis(key, "newton_tol")) cfg->newton_tol = d;
			else if (keyis(key, "newton_tol_relaxed")) cfg->newton_tol_relaxed = d;
			else if (keyis(key, "newton_clamp")) cfg->newton_clamp = d;
			else if (keyis(key, "points_per_layer")) cfg->points_per_layer = (int)d;
			else if (keyis(key, "output_dir")) strncpy(cfg->output_dir, val, STR - 1);
			else if (keyis(key, "apply_voltage_to"))
				dev->apply_voltage_to_anode = (strcmp(val, "cathode") != 0);
			else fprintf(stderr, "warning: unknown simulation key '%s'\n", key);
		}
		else if (strcmp(section, "outcoupling") == 0)
		{
			double d = atof(val);
			if (keyis(key, "enabled")) cfg->outcoupling_enabled = (int)d;
			else if (keyis(key, "optical_stack")) strncpy(cfg->optical_stack, val, STR - 1);
			else if (keyis(key, "emission_layer")) strncpy(cfg->emission_layer, val, STR - 1);
			else if (keyis(key, "lam_start")) cfg->lam_start = d;
			else if (keyis(key, "lam_stop")) cfg->lam_stop = d;
			else if (keyis(key, "lam_points")) cfg->lam_points = (int)d;
			else if (keyis(key, "mesh_points")) cfg->mesh_points_optical = (int)d;
			else if (keyis(key, "spectrum")) strncpy(cfg->spectrum_file, val, STR - 1);
			else fprintf(stderr, "warning: unknown outcoupling key '%s'\n", key);
		}
	}
	fclose(f);

	dev->T = cfg->T;
	if (dev->nlayers == 0)
	{
		fprintf(stderr, "deck has no [layer] sections\n");
		return -1;
	}
	return 0;
}
