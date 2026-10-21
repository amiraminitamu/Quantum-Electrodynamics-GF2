# Scientific conventions

## Basis, indices, and spin

- All electronic matrices are in the real nonorthogonal AO basis unless a
  function explicitly says `orth`.
- AO indices are ordered `(p,q)`; exact ERIs use chemist notation `(pq|rs)`.
- `G[t,p,q]`, `Sigma[t,p,q]`, and `P[p,q]` have time/frequency first.
- `P` is spin summed and satisfies `Tr[P S] = Ne`; `G` is a one-spin
  propagator. Closed-shell spin sums appear as `J-K/2`, `2 direct-exchange`,
  and the `-2` polarization factor.
- Mode layouts are `g[x,p,q]`, `z[x]`, and `D[t,x,y]`/`Pi[t,x,y]`.

## Units

Atomic units are used: energies and mode frequencies are Hartree, AO dipoles
are `e a0`, coordinates used in the dipole construction are bohr, `beta` is
Hartree inverse, and scalar couplings/lambda components are atomic units.
Psi4 converts input geometries before the driver accesses `mol.geometry()`.

## Polarization and coupling

`PhotonMode(coupling=eta, polarization=e)` requires a finite scalar `eta` and
a normalized three-component vector `e`. Internally,
`lambda = eta * e`; the legacy adapter is an explicit conversion from a
three-component `lambda_vec`. These representations are never mixed after
configuration parsing.

## Dipole and origin

Psi4 `MintsHelper.ao_dipole()` returns electronic `-r` AO integrals and does
not include nuclei. Production constructs the total clamped-nuclei dipole in
the fixed-`Ne` sector as

`mu_total = mu_electronic + (sum_A Z_A R_A / Ne) S`.

Contracting with the spin-summed density gives the physical total molecular
dipole. The division by `Ne` is the fixed-particle-number representation of a
constant nuclear operator as a one-body number-operator term. It makes neutral
molecules invariant under a common translation. For an ion with total charge
`Q`, translation changes the total dipole by `Q a`; charged-system cavity
energies therefore retain the expected origin dependence. Nuclear motion,
vibrational dipoles, and beyond-Born--Oppenheimer effects are not included.

## Imaginary-time transforms

Fermionic forward transforms use `exp(+i omega tau)` and inverse transforms
use `exp(-i omega tau)`. Fermionic frequencies are positive odd frequencies
on the stored sparse grid; negative frequencies are reconstructed by real-
Hamiltonian conjugation. `G(tau+beta)=-G(tau)`.

Bosonic transforms use all FFT-ordered positive and negative even frequencies,
`nu=2*pi*fftfreq(...)`. `D(tau+beta)=D(tau)`. On a uniform half-open grid the
FFT path is exact for represented modes. In production, polarization is sampled
on the nonuniform Chebyshev grid, so the forward transform uses its explicit
quadrature weights and the inverse evaluates the truncated Fourier series at
those same nodes. No fermionic antiperiodic transform is reused for bosons.

## Energy and constants

`core_gf2` includes the electronic/QED mean-field half-trace and nuclear
repulsion exactly once. For CS calculations it additionally includes the
coherent linear and `omega*z**2` constants exactly once. `corr_gf2` is the
Galitskii--Migdal contribution with factor `1/2` and total fermionic self-energy.
The boson Hamiltonian is normal ordered (`omega b^dagger b`), so no bare
zero-point `omega/2` is reported.

## Convergence and mixing

The manuscript defines convergence through total-energy change. Production
also enforces particle number during every chemical-potential update and an
inner Fock-RMS threshold. The result now reports density, Fock, `Sigma_ee`,
`Sigma_eb`, `z`, `Pi`, `D`, and both Dyson residuals. These extra residuals are
diagnostics, not additional default stopping thresholds, preserving the
documented solver trajectory. Fermionic quantities use existing DIIS and
optional linear mixing; new bosonic `D` uses conservative linear mixing
(`dx_mix`, defaulting to `sigma_mix` or one). A non-unit bosonic mix naturally
leaves a nonzero instantaneous bosonic Dyson residual until convergence.

