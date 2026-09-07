# GPVDM-based OLED Simulator

This is a self-contained essential OLED simulator derived from the GPVDM code. It performs both drift-diffusion and transfer matrix outcoupling simulation. It should only focus on solving OLED devices in the steady state, and ignore everything else. Because the exciton module is missing, put a stub. The following features from GPVDM are included:

## Features to Implement
- Device discretisation and meshing.
- Setting up simulation parameters, voltage sweeps, etc.
- Setting up device parameters such as device width, materials, contact models, etc.
- Setting material parameters.
- PDE solution using the Newton method.
- DOS models used in GPVDM (specify in the code), modular so different DOS models can be implemented in the future such as EGDM.
- All recombination mechanisms (such as SRH and others) implemented in GPVDM, implemented in a modular way so different bimolecular mechanisms can be added in the future.
- Trap models.
- Device contacts with ohmic-np (fixed density), ohmic-barrier (plain thermionic), or thermionic (field-dependent, using the reduced electric field per Scott, Malliaras) injection.
- The option to use Boltzmann or Fermi-dirac statistics for computing charge densities.
- The option to use field, temperature, and concentration-dependent mobilities (tying back to the desire to include models like EGDM). See "Numerical simulation of charge transport in disordered organic semiconductor devices" by Knapp et al, 2010.

## Calculated device outputs
The simulation outputs should be in the form of multiple text files shown below. They should be simple and easy to parse.

### File 1: Device Profiles
This file contains the position-dependent outputs. It should contain the electron density, hole density, electric potential, electric field, recombination rate, electron mobility, and hole mobility. All as a function of x, across all swept voltage values.

### File 2: Device Figures
This file contains values that depend on voltage. This includes current density vs applied voltage.

Ultimately it should build as a self-contained software where the user can put in the device and material parameters, and the output gets saved. Structure the code to make it modular for each stage.

