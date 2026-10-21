# Fixed-lambda correlated LF-GF2 inner solver

This branch provides the production inner solver required for correlated-energy optimization of the projected LF-GF2 parameters, together with reusable correlated-state warm starts for nearby LF vectors.

## Definition

For a supplied orbital LF vector `lambda_fixed`, the solver keeps `lambda = lambda_fixed` for the entire coupled GF2 calculation. The coherent displacement (denoted \(\xi\) in the manuscript/SI and `z` in the current code) is not a numerical optimization coordinate. At every outer GF2 iteration it is rebuilt from the current correlated density using

```text
xi(lambda; P) = -Tr[g_LF(lambda) P] / omega
```

so the final fixed point is simultaneously stationary in the coherent coordinate and self-consistent in the electronic and bosonic Dyson equations.

The compatibility entry point remains

```python
from gf2.driver_theory import run_lf_gf2_fixed_lambda

core, corr, summary = run_lf_gf2_fixed_lambda(
    mol,
    psi4_options,
    gf2_options,
    lambda_fixed=lambda_trial,
    cavity=cavity,
)
```

For correlated optimization, the state-returning entry point is

```python
from gf2.driver_theory import solve_lf_gf2_fixed_lambda

result = solve_lf_gf2_fixed_lambda(
    mol,
    psi4_options,
    gf2_options,
    lambda_fixed=lambda_trial,
    cavity=cavity,
)

energy = result.total_energy
state = result.state
```

The objective is the synchronized coupled LF-GF2 total energy for that fixed LF vector.

## Reference initialization

A cold fixed-lambda inner solve deliberately does **not** run the static LF-HF optimizer. The ordinary cavity reference density is used only as an initial seed. The requested lambda vector is installed immediately and the initial coherent displacement is obtained analytically from that seed density.

This separation is essential for an outer correlated-energy optimizer: arbitrary lambda vectors can be evaluated without silently reoptimizing them against the static LF functional.

## Warm-start state transfer

A converged `LFGF2FixedState` stores the expensive physical fixed-point quantities

```text
P, F, mu, G(iw), Sigma(iw), D(iv), Pi(iv), xi
```

plus the fixed lambda vector, energies, grids, convergence flag, and outer-iteration count. A nearby trial can reuse that state as

```python
next_result = solve_lf_gf2_fixed_lambda(
    mol,
    psi4_options,
    gf2_options,
    lambda_fixed=lambda_next,
    cavity=cavity,
    initial_state=result.state,
)
```

The new transformed Hamiltonian is always rebuilt from `lambda_next`. In particular, the old Fock matrix is **not** used as the new Hamiltonian Fock; it is stored only as part of the state snapshot. The previous correlated density is used to construct the new analytic coherent displacement and new LF Hamiltonian, while `mu`, `G(iw)`, `Sigma(iw)`, `D(iv)`, and `Pi(iv)` are used as nearby fixed-point guesses.

All Fock and Green-function DIIS histories are reset when lambda changes. Reusing DIIS extrapolation vectors from the old nonlinear map would be unsafe even when the physical states are close.

## Compatibility gate

Warm-state reuse is currently restricted to the same molecule/geometry, basis, Pauli--Fierz tensors, electron count, cavity frequency, inverse temperature, Matsubara grids, and imaginary-time grid. A deterministic SHA-256 problem signature plus explicit grid checks reject accidental reuse across different physical problems or discretizations. An unconverged state is never accepted as a warm start.

Geometry continuation will be treated separately; Stage 1.5 does not assume that AO matrices from different nuclear geometries can be transferred directly.

## Convergence requirements

The existing coupled GF2 fixed-point criteria remain active. In addition, the fixed-lambda path requires zero lambda motion to numerical precision, convergence of the analytic coherent displacement between iterations, and the explicit coherent-displacement stationarity residual.

Metadata reports either

```text
lf_parameter_mode = fixed-correlated-inner
lf_reference_mode = fixed-lambda-cs-hf-seed
lf_warm_start_used = False
```

for a cold solve, or

```text
lf_parameter_mode = fixed-correlated-inner
lf_reference_mode = fixed-lambda-warm-state
lf_warm_start_used = True
```

for a warm solve. The requested lambda vector, source lambda norm, and number of outer GF2 iterations are also retained in metadata.

## Diagnostic scan

`tools/scan_lf_lambda.py` calls the native fixed-lambda solver directly. It no longer monkeypatches the LF static optimizer. The one-dimensional scan therefore exercises the same inner-solver interface used by the multidimensional correlated lambda optimizer.

## Stage 1.5 validation

Run the focused tests:

```bash
pytest -q \
  tests/test_lf_fixed_inner_solver.py \
  tests/test_lf_warm_start_state.py \
  tests/test_lf_reduced_optimizer.py
```

Then validate cold/warm equivalence on LiH:

```bash
python tools/validate_lf_warm_start.py \
  examples/lih_lf_gf2_option_a.inp \
  --source 0.65 \
  --target 0.60
```

The target cold and warm energies must agree within the configured energy tolerance. A reduction in wall time or outer-iteration count is desirable but is not part of the correctness criterion.

Once this regression passes, Stage 2 can place the multidimensional correlated-energy optimizer around `solve_lf_gf2_fixed_lambda`, using each accepted correlated state to warm-start nearby objective evaluations.

