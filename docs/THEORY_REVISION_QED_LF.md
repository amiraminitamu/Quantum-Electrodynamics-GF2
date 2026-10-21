# Theory revision: coupled QED energy and projected LF-GF2

The canonical deterministic solver is `gf2.driver_theory.run_gf2`.

## Coupled QED-GF2 total energy

For the restricted one-spin electronic Green's function, define

- `C_ee = - integral Tr[Sigma_ee(tau) G(beta-tau)] d tau`
- `C_eb = - integral Tr[Sigma_eb(tau) G(beta-tau)] d tau`

With `D(tau) = -<T X(tau) X(0)>`, the Fan kernel is `Sigma_eb = -g G g D` and the closed-shell polarization is `Pi = -2 Tr[g G g G(beta-tau)]`.

The residual electron-boson interaction energy is `E_eb = 2*C_eb`, independently checked as `E_eb = - integral Tr_b[Pi^T D] d tau`.

The photon fluctuation energy, excluding zero-point energy, is

`E_b = -1/2 sum_x omega_x [D_xx(0-) + 1] + 1/2 E_eb`.

`D(0-)` is evaluated by restoring the analytic bare tail: the finite Matsubara sum is applied only to `D-D0` and added to the exact bare equal-time value `-coth(beta*omega/2)`.

The coupled total returned by CS-GF2 is therefore

`E = E_static + C_ee + E_eb + E_b`.

The CS static energy is evaluated directly from the current correlated density with a freshly rebuilt static Pauli-Fierz Fock operator; the DIIS-extrapolated Fock is not treated as the energy functional.

## Projected LF-GF2: Option A

The LF transformation is defined in one fixed symmetric-orthogonal basis. The zero-boson projection produces ordinary effective electronic one- and two-electron quantities `h_LF` and `v_LF`, including the Franck-Condon factors and the induced lambda-dependent interactions used by the LF-HF construction.

Electronic correlation uses the ordinary restricted GF2 second-Born kernel with `v_LF`. No new electronic diagram topology is introduced.

The residual projected linear photon vertex is

`g_LF[p,q] = FC1[p,q]*g[p,q] - omega*lambda[p]*delta[p,q]`,

where `FC1[p,q] = exp[-(lambda[q]-lambda[p])^2/2]`.

The existing corrected Fan and bubble kernels are then reused with `g_LF`:

- `Sigma_eb_LF = -g_LF G g_LF D`
- `Pi_LF = -2 Tr[g_LF G g_LF G(beta-tau)]`

The same coupled total-energy functional is evaluated with the LF static core and these LF-renormalized dynamical quantities.

## LF parameter optimization

For correlated optimization, the LF displacement vector is an outer parameter.
Each trial vector is held fixed while the coupled GF2 equations are converged;
the outer objective is the synchronized coupled total energy. The coherent
displacement is rebuilt analytically from the correlated density. Same-geometry
trials may reuse compatible states; across geometries only the LF vector is
transferred. See `LF_CORRELATED_INNER_SOLVER.md` and `LF_CORRELATED_OPTIMIZER.md`.

## Current scope

Projected LF-GF2 on this branch is intentionally restricted to closed-shell, single-mode calculations with `eri_type='std'`. Density-fitted LF-GF2 and a multimode LF parameterization are not silently approximated.

The first LF-GF2 iteration is second order for the projected LF Hamiltonian defined here. Equality to the complete cloud-fluctuation LF-MP2 construction is a validation target, not an assumption; it will be tested numerically before any stronger manuscript claim is made.

