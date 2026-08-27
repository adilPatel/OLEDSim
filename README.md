# OLED PINN Simulator

## Overview
This project is aimed at simulating OLED devices using physics-informed neural networks (PINNs). It contains both a conventional PDE solver based on DEVSIM's numerical semiconductor tools, and a PINN solver. The DEVSIM solver is used to supplement the PINN solver to validate the results and provide ground-truths. A brief overview of each solver is presented below.

## Solver 1: DEVSIM-based Solver
This solves the drift-diffusion semiconductor equations using a conventional Newton-based PDE solver. It also uses Gummel preconditioning to aid convergence. Below are the current capabilites and the planned future work.

### Current Capabilities
- Ohmic and thermionic carrier injection.
- Langevin recombination.

### Work in Progress
- Implementing disorder-based density of states models, including field, temperature, and concentration-dependant mobilities.
- Implementing carrier trapping.
- Implementing excitons and internal quantum efficiency calculation.

## Planned Goals
- Implementing an optical outcoupling solver to calculate external quantum efficiency.
- A custom C++-based solver to add more speed and flexibility.

## Solver 2: PINN Solver

In its current state, this encodes the drift-diffusion equations as loss functions that are minimised in a fully-connected neural network model. It is composed of three sub-networks that independently solve the potential, electron densities, and hole densities.

### Current Capabilities
- Ohmic injection.
- Loss weighting through the inverse Dirichlet method (Maddu et al, 2021).
- Loss weighting through SoftAdapt (Heydari et al, 2019).

### Work in Progress
- Thermionic injection (not currently validated).
- Increased current accuracy.
- Testing against a wide variety of devices for a robust implementation.

### Planned Goals
- Gaussian disorder models to more accurately simulate organic semiconductors.
- Domain decomposition to more accurately simulate boundary and space charge layers.
- An inverse solver to determine material and device parameters from desired characteristics.
