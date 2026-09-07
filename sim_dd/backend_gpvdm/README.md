# backend_gpvdm — self-contained steady-state OLED simulator

A small, dependency-free C program (only libm) implementing the OLED-relevant
core of GPVDM per [Simulator.md](Simulator.md): 1D steady-state
drift–diffusion with GPVDM's DOS/trap/recombination models, plus
transfer-matrix outcoupling of the computed recombination zone.

```bash
make                      # builds ./oled_sim
./oled_sim examples/oled.ini
make test                 # runs the validation suite in tests/
```

## Architecture (one module per stage)

| Stage | File | Ported from / notes |
|---|---|---|
| Input deck | `src/config.c` | INI-style key=value replaces sim.json |
| Meshing | `src/mesh.c` | uniform per-layer 1D mesh, per-layer point counts |
| DOS + traps | `src/dos.c` | gpvdm `libdos/gendosfdgaus.c` + `dos_trap.c`: MB analytic or FD numeric-integral statistics; trap DOS (exponential tail / Gaussian) discretised into SRH bands with the r1–r4 rate structure. Tables built in memory, band rates analytic |
| Mobility | `src/mobility.c` | constant (gpvdm's model) + Poole–Frenkel + EGDM (Pasveer PRL 94, 206601, per Knapp JAP 108, 054504). Modular switch for future models |
| Recombination | `src/recomb.c` + `src/newton.c` | Rfree=B(np−n₀p₀), Langevin B, Auger, steady-state SRH, dynamic multi-band SRH traps — the exact expressions from `plugins/newton/newton.c` |
| Contacts | `src/contacts.c` | Dirichlet ghost nodes (gpvdm `lib/initial.c` scheme): ohmic-np (fixed density), ohmic-barrier (plain thermionic), and thermionic (Scott–Malliaras injection with image-force lowering via the reduced field, Chem. Phys. Lett. 299, 115 (1999)) — the latter added per the spec, it is not in the gpvdm release |
| Newton solver | `src/newton.c` | port of `plugins/newton/newton.c` `fill_matrix`: same unknowns (φ, xn, xp, trap quasi-Fermi per band), same Scharfetter–Gummel/Bernoulli fluxes with the generalised Einstein relation D=μ(2/3)w/q, same clamped update. UMFPACK replaced by a banded LU over interleaved per-node variables; Jacobian fully analytic |
| Exciton | `src/exciton.c` | **stub** (module missing from the gpvdm release; spec asks for a stub). Emission profile = recombination profile |
| Outcoupling | `src/outcoupling.c` | the same port of gpvdm's transfer-matrix light model as `../oled_outcoupling.c`, driven by the Rfree(x) profile of the emissive layer as the dipole weighting |
| Outputs | `src/output.c` | two plain-text files, see below |

## Outputs

- `device_profiles.dat` — per voltage step (blank-line-separated blocks with
  `# V =` headers): `x n p phi E R mun mup nt pt Jn Jp`
- `device_figures.dat` — one line per voltage: `V J J_anode J_cathode`
- `outcoupling.dat` — η(λ) outcoupled / bottom-lost / absorbed and the
  spectrum-averaged summary

All whitespace-separated, `numpy.loadtxt`-friendly.

## Validation (tests/run_tests.sh — all pass on this machine)

1. **Mott–Gurney law**: hole-only trap-free device. J/J_MG → 1 from above
   (1.09 at 24 V on 300 nm), log-log slope → 2 (1.94). The approach from
   above is the textbook diffusion correction.
2. **Against GPVDM itself**: a single-layer device using gpvdm's default
   material (exponential traps, 5 SRH bands, gpvdm capture cross sections)
   run in both simulators. Against `jv_internal.csv` from the gpvdm_core
   binary built in this repo: **mean |ΔJ/J| = 0.006%, max 0.118%** over the
   32 sweep points above turn-on (below turn-on both solvers sit at their
   numerical noise floors — gpvdm's own J there is non-monotonic and
   partly negative). Reference stored in `tests/reference_gpvdm_jv.csv`;
   regenerate with `tests/make_gpvdm_ref.py`.
3. **OLED example**: 3-layer HTL/EML/ETL with heterojunction confinement,
   Scott–Malliaras anode, Langevin recombination + traps in the EML:
   converges over the full sweep, 100% of recombination inside the EML,
   outcoupling runs off the recombination profile.
4. **Fermi–Dirac option**: matches Boltzmann to 0.3% on a non-degenerate
   device (as it must).
5. **EGDM mobility**: full OLED sweep converges; JV monotonic.

## Numerical notes / judgement calls

- Field/density-dependent mobility and Scott–Malliaras injection are applied
  *lagged* (Gummel-style): updated while the solution settles, then held so
  Newton converges to full tolerance. This is the standard approach and is
  also why gpvdm keeps constant mobilities inside its Jacobian.
- The equilibrium solve runs with recombination off and mobility frozen at
  μ0 (the equilibrium solution depends on neither); n₀p₀ for
  R = B(np − n₀p₀) is captured from it, exactly as gpvdm's `nfequlib`.
- The convergence norm down-weights quasi-Fermi updates at nodes with
  < 1 carrier/m³, where the continuity equations are round-off noise.
  `newton_tol_relaxed` (default 1e-6 V ≈ kT/26000) accepts stiff cases that
  plateau above `newton_tol` — the analogue of gpvdm's clever-exit.
- EGDM's `mun`/`mup` are the Pasveer μ0 *prefactors*, not device
  mobilities: μ(300 K, low n) ≈ μ0·1.8e-9·exp(−0.42 σ̂²). See
  `tests/oled_egdm.ini` for a worked example.
- The outcoupling model is normal-incidence transfer matrix (inherited from
  gpvdm's optical solver): no in-plane wavevector integration, so
  waveguide/plasmon losses are absent and η is an upper bound. See the
  header of `../oled_outcoupling.c`.
- Missing-by-design (out of the OLED steady-state scope per Simulator.md):
  time domain, illumination/photogeneration, 2D/3D, thermal solve, mobile
  ions, circuit elements.

## Input deck reference

See the annotated [examples/oled.ini](examples/oled.ini) (every key) and
[tests/gpvdm_match.ini](tests/gpvdm_match.ini) (gpvdm-equivalent settings).
Sections: `[simulation]`, `[layer <name>]` (repeated, anode side first),
`[contact_anode]`, `[contact_cathode]`, `[outcoupling]`.

Key conventions: energies in eV, lengths in m, densities in m⁻³,
mobilities in m²/Vs. `Xi` is the electron affinity (LUMO depth), `Eg` the
gap; trap energies are measured into the gap from the respective band edge
(negative `srh_start`). A trap's `srh_sigman/srh_sigmap` follow gpvdm's
convention (capture cross section for electrons/holes respectively, per
trap type — see `gpvdm_match.ini` for the `_e`/`_h` suffixed forms).
