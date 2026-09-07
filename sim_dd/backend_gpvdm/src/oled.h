// oled.h
//
// Self-contained steady-state OLED drift-diffusion + transfer-matrix
// outcoupling simulator, derived from gpvdm v8.0 (Roderick C. I. MacKenzie,
// https://www.gpvdm.com, MIT license).  See README.md for the provenance map
// of which gpvdm source files each module mirrors.
//
// Conventions (identical to gpvdm's newton plugin):
//   * 1D mesh along x (gpvdm's y), anode at x=0, cathode at x=L.
//   * Energies in eV inside the input deck, converted to SI internally.
//   * Reference energy: the equilibrium Fermi level EF = 0.
//   * Solver unknowns per node: phi   electrostatic potential [V]
//                               xn    n = N(xn + Xi)     (EFn = xn - phi)
//                               xp    p = P(xp - Xi - Eg) (EFp = -xp - phi)
//                               xt[]  trap quasi-Fermi per electron-trap band
//                               xpt[] trap quasi-Fermi per hole-trap band
//   * Ec = -phi - Xi,  Ev = Ec - Eg.
//   * At equilibrium xn = phi, xp = -phi (EFn = EFp = 0).

#ifndef OLED_H
#define OLED_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define Qe 1.602176634e-19
#define KB 1.380649e-23
#define EPS0 8.8541878128e-12
#define HBAR 1.054571817e-34
#define M0 9.1093837015e-31
#define H_PLANCK 6.62607015e-34
#define C_LIGHT 2.99792458e8
#define OLED_PI 3.14159265358979323846

#define MAX_LAYERS 16
#define MAX_BANDS 20
#define STR 256

// ---------------------------------------------------------------- DOS ------

enum carrier_stats { STATS_BOLTZMANN, STATS_FERMIDIRAC, STATS_GAUSSIAN };
enum trap_type { TRAPS_NONE, TRAPS_EXPONENTIAL, TRAPS_GAUSSIAN };

// Lookup table for Fermi-Dirac statistics (mirrors gpvdm's gendosfdgaus
// tables, computed in memory at start-up instead of cached to disk).
struct fd_table
{
	int len;
	double u0, du;       // u = EF - Ec (electrons) or Ev - EF (holes) [eV]
	double *n;           // carrier density [m^-3]
	double *w;           // average carrier energy [J] (Einstein: D=mu*2w/3q)
};

// One trap band: gpvdm discretises the trap DOS into srh_bands bands, each
// treated as a discrete level at the band centre E_b with the band-integrated
// density N_b (see gendosfdgaus.c).
struct trap_band
{
	double Eb;           // band centre, eV, negative into the gap from the band edge
	double Nb;           // integrated trap density in this band [m^-3]
};

// Per-carrier DOS description for one material.
struct dos_side
{
	enum carrier_stats stats;
	double Nc;           // effective DOS [m^-3] (Nv for the hole side)
	double meff;         // effective mass ratio, derived from Nc for FD stats
	double dos_sigma;    // Gaussian DOS width [eV] (STATS_GAUSSIAN); defaults
	                     // to the EGDM sigma of the same carrier so transport
	                     // and occupation describe the same material
	struct fd_table fd;  // built for STATS_FERMIDIRAC and STATS_GAUSSIAN

	enum trap_type trap_type;
	int nbands;
	double Nt;           // trap DOS prefactor [m^-3 eV^-1]
	double Et;           // exponential tail slope [eV] (TRAPS_EXPONENTIAL)
	double sigma;        // gaussian width [eV]        (TRAPS_GAUSSIAN)
	double Ecen;         // gaussian centre [eV, negative into gap]
	double Eonset;       // exponential-tail onset [eV, <=0]; 0 = gpvdm's
	                     // band-edge reference.  See trap_rho() in dos.c.
	double srh_start;    // deepest band edge [eV, negative], e.g. -1.0
	double srh_sigman;   // capture cross sections [m^2]
	double srh_sigmap;
	double srh_vth;      // thermal velocity [m/s]
	struct trap_band band[MAX_BANDS];
};

// ------------------------------------------------------------ mobility -----

enum mobility_model { MOB_CONSTANT, MOB_POOLE_FRENKEL, MOB_EGDM };

struct mobility_side
{
	enum mobility_model model;
	double mu0;          // zero-field mobility [m^2/Vs]
	double pf_gamma;     // Poole-Frenkel factor [ (m/V)^1/2 ]
	double egdm_sigma;   // EGDM disorder [eV]
	double egdm_a;       // EGDM lattice constant [m]
	double egdm_Nt;      // EGDM site density [m^-3] (default 1/a^3)
	double egdm_c2;      // exponent of sigma_hat^2 in the T prefactor;
	                     // Pasveer's published 0.42, default here 0.39
	double egdm_x_clamp; // max of x = n/(2 Nt) in g1 (default 0.5, i.e. n<=Nt)
	double egdm_f_clamp; // ceiling on the dimensionless REDUCED FIELD
	                     // Fhat = q a F / sigma in g2 (q = elementary charge,
	                     // so only F varies).  Pasveer's fit covers Fhat <~ 2;
	                     // past that g2 is held fixed at its clamp value.
	                     // 0 = no clamp (default), i.e. raw Pasveer.
};

// -------------------------------------------------------- recombination ----

enum recomb_model { RECOMB_CONSTANT_B, RECOMB_LANGEVIN };

// How trapped carriers exchange with the free bands.
//
//  * TRAP_KINETIC_SRH   each trap band carries its own quasi-Fermi level,
//    set by the steady-state SRH balance p*r1 - r2 - n*r3 + r4 = 0, and the
//    net capture appears as a recombination term in the continuity equations.
//    This is gpvdm's model.
//
//  * TRAP_QUASI_EQUILIBRIUM  traps are in equilibrium with the free carriers
//    of the same sign and share their quasi-Fermi level (xt = xn, xpt = xp).
//    Trapped charge then enters ONLY Poisson; there is no trap-assisted
//    recombination and the drift-diffusion currents carry free carriers
//    alone.  This is the model used by Knapp et al., J. Appl. Phys. 108,
//    054504 (2010), where the occupied-DOS integral N(E) f(E,EF) dE is the
//    sum of free and trapped carriers against a single EF.
enum trap_kinetics { TRAP_KINETIC_SRH, TRAP_QUASI_EQUILIBRIUM };

struct recomb_config
{
	enum recomb_model model;
	enum trap_kinetics trap_kinetics;
	double B;            // bimolecular coefficient [m^3/s] (RECOMB_CONSTANT_B)
	double langevin_pre; // Langevin prefactor (dimensionless, default 1)
	int auger_enabled;
	double Cn, Cp;       // Auger coefficients [m^6/s]
	int ss_srh_enabled;  // steady-state (analytic) SRH
	double tau_n, tau_p; // [s]
	double n1, p1;       // [m^-3]
};

// ------------------------------------------------------------- layers ------

struct layer
{
	char name[STR];
	double thickness;    // [m]
	int points;          // mesh points in this layer

	// band structure
	double Eg;           // [eV]
	double Xi;           // electron affinity [eV]
	double epsr;
	double Nad;          // net doping Nd-Na [m^-3], signed

	struct dos_side dosn;
	struct dos_side dosp;
	struct mobility_side mun;
	struct mobility_side mup;
	struct recomb_config recomb;
};

// ------------------------------------------------------------ contacts -----

enum contact_model { CONTACT_OHMIC_NP, CONTACT_OHMIC_BARRIER, CONTACT_THERMIONIC };
enum contact_majority { MAJORITY_ELECTRON, MAJORITY_HOLE };

struct contact
{
	enum contact_model model;
	enum contact_majority majority;
	double np;           // majority carrier density for ohmic-np contacts [m^-3]
	double barrier;      // injection barrier [eV] for ohmic-barrier / thermionic

	// derived at set-up / updated during the solve
	double phi_eq;       // equilibrium potential of this contact [V]
	double n_ghost;      // ghost-node electron density [m^-3]
	double p_ghost;      // ghost-node hole density [m^-3]
	double dnmaj_dphi;   // d(majority ghost density)/d(phi at adjacent node),
	                     // for the Newton Jacobian; 0 for field-independent
	                     // contact models [m^-3 V^-1]
};

// -------------------------------------------------------------- device -----

struct device
{
	int nlayers;
	struct layer layers[MAX_LAYERS];

	int N;               // total mesh points
	double *x;           // node positions [m]
	int *lay;            // layer index per node
	double L;            // device thickness [m]

	double T;            // [K]

	struct contact anode;    // at x=0
	struct contact cathode;  // at x=L
	int apply_voltage_to_anode;

	// per-node material arrays (filled from layers)
	double *Eg, *Xi, *epsr, *Nad;

	// solver state
	double *phi, *xn, *xp;
	double **xt, **xpt;      // [band][node]
	int nbands;              // max bands over layers (bands only active where layer has traps)

	// derived per node, refreshed by update_arrays()
	double *n, *p, *dn, *dp, *wn, *wp;
	double *mun, *mup;       // node mobilities (lagged field/carrier dependence)
	double *Rfree, *Rsrh, *Rauger, *Rtot;
	double *nt_all, *pt_all;
	double *Jn, *Jp;         // node current densities [A/m^2]
	double *n_eq, *p_eq;     // equilibrium densities (for R = B(np - neq peq))

	// trap working arrays per band per node
	double **nt, **dnt, **srh_n_r1, **srh_n_r2, **srh_n_r3, **srh_n_r4;
	double **pt, **dpt, **srh_p_r1, **srh_p_r2, **srh_p_r3, **srh_p_r4;
	double **dsrh_n_r1, **dsrh_n_r2, **dsrh_n_r3, **dsrh_n_r4;
	double **dsrh_p_r1, **dsrh_p_r2, **dsrh_p_r3, **dsrh_p_r4;

	double Vapplied;
	double J_anode, J_cathode;   // contact current densities

	// The equilibrium solution is mobility-independent (J=0), so field/
	// density-dependent mobility models are frozen at mu0 there; during the
	// bias sweep the lagged mobility is under-relaxed for stability.
	int freeze_mobility;
};

// ------------------------------------------------------------ sim config ---

struct sim_config
{
	double T;
	double V_start, V_stop, V_step;
	int max_newton_steps;
	double newton_tol;       // target max |du| [V]
	double newton_tol_relaxed; // accepted after many iterations (stiff cases
	                         // hit a floating-point noise floor above
	                         // newton_tol; gpvdm's "clever exit" equivalent)
	double newton_clamp;     // gpvdm electrical_clamp (units of kT/q)
	int points_per_layer;    // default mesh density
	char output_dir[STR];

	// outcoupling
	int outcoupling_enabled;
	char optical_stack[STR];
	char emission_layer[STR];    // electrical layer the '*' optical layer maps to
	double lam_start, lam_stop;
	int lam_points;
	int mesh_points_optical;
	char spectrum_file[STR];
};

// ------------------------------------------------------------ functions ----

// config.c
int config_load(const char *path, struct sim_config *cfg, struct device *dev);

// mesh.c
void mesh_build(struct device *dev, struct sim_config *cfg);

// dos.c
void dos_init_layer(struct layer *l, double T);
void dos_get_n(const struct dos_side *d, double u_eV, double T, double *n, double *dn, double *w);
void dos_get_p(const struct dos_side *d, double u_eV, double T, double *p, double *dp, double *w);
double dos_u_from_n(const struct dos_side *d, double n, double T);
void dos_trap_bands_setup(struct dos_side *d);
void trap_get_n(const struct dos_side *d, const struct layer *l, int band, double ut_eV, double T,
                double *nt, double *r1, double *r2, double *r3, double *r4,
                double *dnt, double *dr1, double *dr2, double *dr3, double *dr4);
void trap_get_p(const struct dos_side *d, const struct layer *l, int band, double ut_eV, double T,
                double *pt, double *r1, double *r2, double *r3, double *r4,
                double *dpt, double *dr1, double *dr2, double *dr3, double *dr4);

// mobility.c
double mobility_get(const struct mobility_side *m, double T, double F, double carrier_density);

// recomb.c
double recomb_get_B(const struct recomb_config *r, double mun, double mup, double epsr);

// contacts.c
void contacts_setup(struct device *dev);
void contacts_update(struct device *dev);   // refresh field-dependent injection (Scott-Malliaras)

// newton.c
int newton_solve(struct device *dev, struct sim_config *cfg, int is_equilibrium, int quiet);
void update_arrays(struct device *dev);

// exciton.c (stub)
void exciton_solve_stub(struct device *dev);

// outcoupling.c
int outcoupling_run(struct device *dev, struct sim_config *cfg);

// output.c
void output_open(struct sim_config *cfg);
void output_profiles(struct device *dev, struct sim_config *cfg);
void output_figures(struct device *dev, struct sim_config *cfg);
void output_close(void);

// util.c
double bernoulli_B(double x);
double bernoulli_dB(double x);
int band_lu_solve(double *AB, double *b, int n, int kl, int ku);
void *xmalloc(size_t s);
double **alloc2d(int a, int b);

#endif
