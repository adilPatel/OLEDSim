// exciton.c - STUB (matching the missing gpvdm exciton module).
//
// gpvdm's open-source release ships without its exciton diffusion solver
// (libexciton), and the simulator specification asks for a stub in its
// place.  In the full model the exciton solver would take the recombination
// profile R(x) produced by the electrical solve as the exciton generation
// term, solve the steady-state exciton diffusion equation
//
//    0 = D d2S/dx2 - S/tau + R(x) - quenching terms
//
// and hand the resulting radiative-decay profile to the outcoupling stage.
// With the stub in place, the outcoupling stage uses the electron-hole
// recombination profile R(x) directly as the emission profile - i.e. the
// excitons are assumed to decay radiatively where they are formed.
//
// TO IMPLEMENT: discretise the diffusion equation on dev->x (the mesh is
// already built), with exciton diffusion length/lifetime/quenching as new
// material parameters in config.c, and replace the identity mapping in
// outcoupling.c (emission_weight <- Rtot) with the solved profile.

#include "oled.h"

void exciton_solve_stub(struct device *dev)
{
	(void)dev;
	// intentionally empty: emission profile = recombination profile
}
