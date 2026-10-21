# Correlated-energy LF-GF2 outer optimizer

This stage optimizes the LF displacement vector against the **fully converged projected LF-GF2 total energy** rather than the static LF-HF functional.

For a trial vector `lambda`, the objective is

\[
E_{\mathrm{LF-GF2}}^{\mathrm{SC}}(\boldsymbol\lambda),
\]

where the fixed-lambda inner solver reconverges the electronic and bosonic Dyson equations and imposes the stationary coherent displacement at the correlated density. The outer optimizer never changes `lambda` inside one GF2 fixed-point solve.

## Why simultaneous perturbations

A 16-parameter central finite-difference gradient would require 32 fully converged GF2 calculations for one gradient. The current outer optimizer instead probes one simultaneous direction on both sides of the current point,

\[
E_+=E(\boldsymbol\lambda+c\mathbf d),\qquad
E_-=E(\boldsymbol\lambda-c\mathbf d),
\]

and estimates the directional derivative

\[
\frac{\partial E}{\partial \mathbf d}
\approx
\frac{E_+-E_-}{2c}.
\]

A normalized trust-radius step is then taken in the downhill direction. The usual iteration therefore needs only three correlated objective evaluations: the `+` probe, the `-` probe, and one trial point. The cost is independent of the number of LF parameters at fixed iteration count.

The first few directions are radial with respect to the static LF-HF starting vector. This relaxes the overall LF displacement scale without imposing a system-specific factor such as `0.65`. Later directions are fixed-seed Rademacher directions, so individual LF components can move away from the static-LF ray.

## Warm starts and persistent restart states

Every nearby objective call is initialized from a converged correlated state. The trial point is warm-started from the nearest available converged state among the current point and the two probes. The fixed-lambda inner solver still rebuilds the transformed Hamiltonian at the requested `lambda` and resets DIIS histories before reconvergence.

The molecular optimizer now writes the **current accepted correlated state** to an NPZ checkpoint after initialization and after every outer iteration. A JSON sidecar stores the global iteration count, current trust radius, radial-bootstrap count, and random seed. The state file contains the converged density, chemical potential, Green's function, self-energy, dressed photon propagator, polarization, grids, and fixed LF vector.

A restarted run loads this state and supplies it directly as the initial correlated objective evaluation. Therefore the expensive initial fixed-lambda GF2 solve is **not repeated**. The iteration offset also advances the deterministic Rademacher sequence so a restart continues rather than recycling the first random direction.

The state loader uses `allow_pickle=False`, and the fixed-lambda inner solver still validates the physical problem signature and numerical grids when the state is used to warm-start the first new probe.

## Accepted iterate versus lowest sampled point

The optimizer distinguishes two quantities:

- `energy_opt`, `lambda_opt`: the final **accepted iterate**, subject to `energy_tolerance`;
- `energy_sampled_best`, `lambda_sampled_best`: the lowest fully converged objective value encountered anywhere among accepted points, probes, and candidates.

This distinction matters when a sampled point is lower by an amount smaller than the acceptance threshold. Such a point is reported as the lowest sampled energy but does not silently change the accepted optimization trajectory.

## Convergence safeguards

An unconverged `+` or `-` probe aborts the outer optimization because a directional derivative built from an unconverged GF2 energy is not meaningful. An unconverged trial point is rejected rather than accepted as an optimization step.

A trial is accepted only if its fully converged total energy is lower than the current energy by more than `energy_tolerance`. If the trust-radius trial is not acceptable but one of the already-computed probes is lower by more than the threshold, that probe is accepted instead. Otherwise the step is rejected and the trust radius is reduced.

This is an optimization of an approximate self-consistent GF2 energy. It is **not** a Rayleigh--Ritz variational bound to the exact QED ground-state energy, and the simultaneous-direction optimizer does not by itself prove a global minimum.

## Fresh molecular run

A fresh LiH run can be started with

```bash
python tools/optimize_lf_lambda.py \
  examples/lih_lf_gf2_option_a.inp \
  --initial-scale 0.75 \
  --maxiter 3 \
  --csv lih_stage2_bootstrap.csv \
  --state-out lih_stage2_state.npz
```

After the initial cold correlated solve, all nearby fixed-lambda evaluations use correlated warm starts. The accepted state and outer-optimizer continuation metadata are updated automatically.

## Restart without another cold solve

Continue from the saved accepted state with

```bash
python tools/optimize_lf_lambda.py \
  examples/lih_lf_gf2_option_a.inp \
  --restart-state lih_stage2_state.npz \
  --maxiter 6 \
  --csv lih_stage2_multidim.csv \
  --state-out lih_stage2_state.npz
```

If the previous run completed the default three radial bootstrap iterations, the restored iteration offset is three, so the next iteration is automatically a Rademacher multidimensional direction. The previous trust radius and random-seed sequence are also restored unless explicitly overridden on the command line.

Useful controls are

```text
--probe-radius       8e-3
--min-probe-radius   1e-3
--trust-radius       restored on restart, otherwise 2e-2
--min-trust-radius   1e-3
--max-trust-radius   5e-2
--energy-tol         5e-8
--radial-bootstrap   restored on restart, otherwise 3
--seed               restored on restart, otherwise 10086
```

The deterministic unit tests are

```bash
python -m pytest -q \
  tests/test_lf_correlated_optimizer.py \
  tests/test_lf_warm_start_state.py
```

Before production calculations, also rerun the fixed-lambda and reduced-LF regression suites on the same branch.

