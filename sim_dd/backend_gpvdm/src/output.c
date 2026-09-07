// output.c - simulation output files.
//
// Two plain-text files per run (see Simulator.md):
//   device_profiles.dat  position-dependent quantities, one block per
//                        voltage step separated by blank lines:
//                        x n p phi E R mun mup nt pt Jn Jp
//   device_figures.dat   voltage-dependent quantities, one line per step:
//                        V J J_n_anode-ish J_p J_anode J_cathode
// Both are whitespace-separated with '#' headers - trivially parsed with
// numpy.loadtxt / pandas.

#include "oled.h"
#include <sys/stat.h>

static FILE *f_prof = NULL;
static FILE *f_fig = NULL;

void output_open(struct sim_config *cfg)
{
	char path[STR * 2];
	mkdir(cfg->output_dir, 0755);

	snprintf(path, sizeof(path), "%s/device_profiles.dat", cfg->output_dir);
	f_prof = fopen(path, "w");
	if (f_prof == NULL) { fprintf(stderr, "can't write %s\n", path); exit(1); }
	fprintf(f_prof, "# device profiles: one block per voltage, blank-line separated\n");
	fprintf(f_prof, "# block header lines start with '# V ='\n");
	fprintf(f_prof, "# x[m] n[m-3] p[m-3] phi[V] E[V/m] R[m-3s-1] mun[m2/Vs] mup[m2/Vs] nt[m-3] pt[m-3] Jn[A/m2] Jp[A/m2]\n");

	snprintf(path, sizeof(path), "%s/device_figures.dat", cfg->output_dir);
	f_fig = fopen(path, "w");
	if (f_fig == NULL) { fprintf(stderr, "can't write %s\n", path); exit(1); }
	fprintf(f_fig, "# V[V] J[A/m2] J_anode[A/m2] J_cathode[A/m2]\n");
}

void output_profiles(struct device *dev, struct sim_config *cfg)
{
	int i;
	(void)cfg;
	fprintf(f_prof, "# V = %.6f\n", dev->Vapplied);
	for (i = 0; i < dev->N; i++)
	{
		double E;
		if (i == 0)
			E = -(dev->phi[1] - dev->phi[0]) / (dev->x[1] - dev->x[0]);
		else if (i == dev->N - 1)
			E = -(dev->phi[i] - dev->phi[i - 1]) / (dev->x[i] - dev->x[i - 1]);
		else
			E = -(dev->phi[i + 1] - dev->phi[i - 1]) / (dev->x[i + 1] - dev->x[i - 1]);
		fprintf(f_prof, "%e %e %e %e %e %e %e %e %e %e %e %e\n",
		        dev->x[i], dev->n[i], dev->p[i], dev->phi[i], E, dev->Rtot[i],
		        dev->mun[i], dev->mup[i], dev->nt_all[i], dev->pt_all[i],
		        dev->Jn[i], dev->Jp[i]);
	}
	fprintf(f_prof, "\n");
	fflush(f_prof);
}

void output_figures(struct device *dev, struct sim_config *cfg)
{
	(void)cfg;
	fprintf(f_fig, "%f %e %e %e\n", dev->Vapplied,
	        0.5 * (dev->J_anode + dev->J_cathode), dev->J_anode, dev->J_cathode);
	fflush(f_fig);
}

void output_close(void)
{
	if (f_prof != NULL) fclose(f_prof);
	if (f_fig != NULL) fclose(f_fig);
}
