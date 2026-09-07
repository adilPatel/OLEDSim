// main.c - simulation driver.
//
// Stages (each in its own module, per the simulator spec):
//   1. parse the input deck                        (config.c)
//   2. build the mesh + material arrays            (mesh.c, dos.c)
//   3. contact setup / built-in potentials         (contacts.c)
//   4. equilibrium solve (V=0, recombination off)  (newton.c)
//   5. voltage sweep with continuation             (newton.c)
//   6. exciton stage - stub                        (exciton.c)
//   7. transfer-matrix outcoupling                 (outcoupling.c)
//   8. text-file outputs                           (output.c)

#include "oled.h"

int main(int argc, char **argv)
{
	struct sim_config cfg;
	struct device dev;
	int i, b, l;
	double V;
	int nsteps;

	if (argc < 2)
	{
		printf("usage: oled_sim <deck.ini>\n");
		return 1;
	}
	if (config_load(argv[1], &cfg, &dev) != 0)
		return 1;

	printf("oled_sim: %d layer(s), T=%.1fK\n", dev.nlayers, dev.T);
	for (l = 0; l < dev.nlayers; l++)
		dos_init_layer(&dev.layers[l], dev.T);

	mesh_build(&dev, &cfg);
	printf("mesh: %d points over %.1f nm\n", dev.N, dev.L * 1e9);

	contacts_setup(&dev);
	printf("contacts: anode phi_eq=%.3fV n=%.2e p=%.2e | cathode phi_eq=%.3fV n=%.2e p=%.2e\n",
	       dev.anode.phi_eq, dev.anode.n_ghost, dev.anode.p_ghost,
	       dev.cathode.phi_eq, dev.cathode.n_ghost, dev.cathode.p_ghost);
	printf("built-in potential: %.3f V\n", fabs(dev.anode.phi_eq - dev.cathode.phi_eq));

	// initial guess: local charge neutrality (gpvdm lib/initial.c idea),
	// linear blend to the contact potentials near the ends
	for (i = 0; i < dev.N; i++)
	{
		double kT_eV = KB * dev.T / Qe;
		struct layer *la = &dev.layers[dev.lay[i]];
		// Charge-neutral level.  The Boltzmann form 0.5*kT*log(Nv/Nc) is only
		// valid for Boltzmann statistics; for a Gaussian DOS the occupation is
		// offset by ~sigma^2/kT from the DOS centre, so start from the midgap
		// level and let the solve find the rest.  (Getting this wrong is not
		// merely slow - the equilibrium Newton diverges from a bad guess.)
		double neutral_off = 0.0;
		if (la->dosn.stats == STATS_BOLTZMANN && la->dosp.stats == STATS_BOLTZMANN)
			neutral_off = 0.5 * kT_eV * log(la->dosp.Nc / la->dosn.Nc);
		double phi_neutral = -dev.Xi[i] - dev.Eg[i] / 2.0 + neutral_off;
		double frac = dev.x[i] / dev.L;
		double phi_lin = dev.anode.phi_eq + frac * (dev.cathode.phi_eq - dev.anode.phi_eq);
		dev.phi[i] = 0.5 * (phi_neutral + phi_lin);
		dev.xn[i] = dev.phi[i];
		dev.xp[i] = -dev.phi[i];
		for (b = 0; b < dev.nbands; b++)
		{
			dev.xt[b][i] = dev.phi[i];
			dev.xpt[b][i] = -dev.phi[i];
		}
	}

	// neq*peq = 0 during the equilibrium solve (recombination is off anyway)
	for (i = 0; i < dev.N; i++) { dev.n_eq[i] = 0.0; dev.p_eq[i] = 0.0; }

	printf("solving equilibrium...\n");
	dev.Vapplied = 0.0;
	if (newton_solve(&dev, &cfg, 1, getenv("OLED_VERBOSE") == NULL) != 0)
	{
		fprintf(stderr, "equilibrium solve failed\n");
		return 1;
	}
	update_arrays(&dev);
	for (i = 0; i < dev.N; i++)
	{
		dev.n_eq[i] = dev.n[i];
		dev.p_eq[i] = dev.p[i];
	}
	printf("equilibrium ok (n[0]=%.2e p[0]=%.2e n[N-1]=%.2e)\n",
	       dev.n[0], dev.p[0], dev.n[dev.N - 1]);

	output_open(&cfg);

	nsteps = (int)floor((cfg.V_stop - cfg.V_start) / cfg.V_step + 1.5);
	for (l = 0; l < nsteps; l++)
	{
		V = cfg.V_start + l * cfg.V_step;
		dev.Vapplied = V;
		if (newton_solve(&dev, &cfg, 0, getenv("OLED_VERBOSE") == NULL) != 0)
		{
			fprintf(stderr, "stopping sweep at V=%g\n", V);
			break;
		}
		printf("V=%6.3f  J=%12.5e A/m^2 (anode %.4e, cathode %.4e)\n",
		       V, 0.5 * (dev.J_anode + dev.J_cathode), dev.J_anode, dev.J_cathode);
		output_profiles(&dev, &cfg);
		output_figures(&dev, &cfg);
	}

	// exciton stage: stub (emission profile = recombination profile)
	exciton_solve_stub(&dev);

	if (cfg.outcoupling_enabled)
		outcoupling_run(&dev, &cfg);

	output_close();
	printf("outputs written to %s/\n", cfg.output_dir);
	return 0;
}
