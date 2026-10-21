# Production correlated QED-GF2 PES workflow

This workflow is for the JCTC revision production calculations. It uses the
corrected CS-GF2 implementation and the zero-boson-projected LF-GF2 Hamiltonian
with a fully converged fixed-lambda coupled GF2 inner solve.

## Frozen LF production criterion

At each geometry, the LF vector is optimized against the synchronized
self-consistent LF-GF2 total energy. The production outer-energy resolution is

`1e-6 Eh`.

The local confirmation stage uses symmetric Rademacher directions at radius
`1e-3`. A point is accepted as operationally optimized after ten consecutive
independent directions fail to lower the energy by more than `1e-6 Eh`.
The inner GF2 convergence thresholds remain those in the molecular input and
are not loosened by this outer stopping criterion.

The confirmed LiH reference at R = 1.60 Angstrom obtained before launching the
PES is preserved separately and is used only to seed the neighboring geometry
with its optimized LF vector.

## Geometry continuation rule

Across different nuclear geometries, transfer **only** the optimized LF vector:

`lambda*(R_i) -> initial lambda(R_{i+1})`.

Do not transfer the electronic/bosonic Green's functions, self-energies,
polarization, density, Fock matrix, or DIIS histories between geometries. The
state fingerprint in `gf2/lf_state.py` intentionally prevents such cross-
geometry reuse. `tools/run_lf_pes_chain.py` therefore starts a fresh GF2 fixed
point for each new geometry at the continued lambda and uses warm states only
among lambda trials belonging to that same geometry.

If a geometry does not achieve the ten-miss confirmation within the requested
maximum number of directions, the chain stops before its lambda is propagated
to the next geometry. Re-running the same command resumes from already
confirmed point checkpoints.

## Example LiH scan

An example short-range scan uses:

`1.40, 1.45, ..., 2.20 Angstrom`.

LF-GF2 is split into two independent continuation chains from the confirmed
R=1.60 Angstrom reference:

- compression: `1.55 -> 1.50 -> 1.45 -> 1.40`;
- dissociation: `1.65 -> 1.70 -> ... -> 2.20`.

CS-GF2 has no LF outer optimization and is evaluated directly over the full
17-point grid.

Run the portable scripts in `tools/` directly or wrap them in a scheduler
script adapted to the local cluster. Supply an equilibrium LF checkpoint to
initialize each continuation chain.

## Per-point outputs

Each LF geometry receives its own directory containing the rendered molecular
input, initial fixed-lambda state, optimizer CSV, coarse state, confirmation
CSV, confirmed final state, and `point.json`. The chain also maintains a
sorted `lf_pes.csv` summary. A confirmed point is restartable and is skipped on
a later invocation unless `--no-resume` is requested.

The CS workflow writes one directory per geometry with the rendered input and
structured result JSON, plus a sorted `cs_pes.csv` summary.

## Suggested controls for the example scan

- outer LF energy tolerance: `1e-6 Eh`;
- correlated optimizer iterations per geometry: `10`;
- initial simultaneous probe radius: `4e-3`;
- initial trust radius: `4e-3`;
- minimum local radius: `1e-3`;
- radial bootstrap across continued geometries: disabled;
- confirmation radius: `1e-3`;
- required consecutive confirmation misses: `10`;
- maximum confirmation directions: `30`;
- optimizer seed: `271828`;
- confirmation seed: `161803`;
- no LF component bound.

The fixed-lambda Matsubara grids, inverse temperature, electron-number
criterion, and Dyson convergence criteria come directly from the input file.
They must remain identical between CS-GF2 and LF-GF2 for the same PES unless a
separate convergence study explicitly changes them.

## H2

`tools/run_lf_pes_chain.py` and `tools/run_cs_pes.py` are molecule-independent
for two-atom Cartesian inputs. After the H2 equilibrium LF-GF2 reference is
confirmed, the same lambda-only continuation protocol is used for H2. A common
CS/LF geometry grid should be supplied to both tools so the revised comparison
is point-for-point consistent.

