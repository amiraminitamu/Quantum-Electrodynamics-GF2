# Density-fitted CS-GF2 interaction design

## Existing electronic convention

The electronic DF path obtains raw Psi4 three-index AO integrals
`Ppq[P,p,q] = (P|pq)` and the Coulomb metric `V[P,Q] = (P|Q)`. Psi4's
matrix power with cutoff `1e-20` forms `V^-1/2`, after which

```text
B[Q,p,q] = sum_P (V^-1/2)[Q,P] (P|pq)
(pq|rs)_DF = sum_Q B[Q,p,q] B[Q,r,s].
```

The factor layout is `(naux, nao, nao)`, and the last two indices are real and
symmetric. Independent factors are constructed for the self-energy and Fock
auxiliary bases. The restricted Fock path evaluates `J - K/2`; the DF GF2
self-energy uses the same factor twice and therefore includes all pairs of
auxiliary channels. Ordinary electronic DF calculations pass the original
electronic factors directly, preserving their arithmetic path and storage.

## QED extension

For independent dipole-gauge modes, the two-electron DSE is already exactly
factorized:

```text
v_DSE[p,q,r,s] = sum_x h_c[x,p,q] h_c[x,r,s].
```

Each nonzero `h_c[x]` is therefore an exact auxiliary channel with no metric
or normalization factor. The effective factors are

```text
B_effective = block(B_electronic, h_c)
B_electronic : (naux, nao, nao)
h_c          : (nmode, nao, nao)
B_effective  : (naux + nactive_mode, nao, nao).
```

Zero channels are omitted from the numerical block so adding a zero-coupling
mode leaves the electronic contraction path unchanged. Configured and active
mode counts are recorded separately.

The factors are represented by `FactorizedInteraction`, which retains the
electronic and exact-channel blocks and materializes a combined contiguous
factor array once, only when exact channels exist. This one-time copy costs
`O((naux+nmode) nao^2)` storage and lets the existing optimized DF Fock and GF2
self-energy contractions include electronic-electronic, electronic-DSE, and
DSE-DSE terms without constructing `O(nao^4)` ERIs. The ordinary electronic
path returns the original electronic array without a copy.

The one-body Hamiltonian remains

```text
h_eff = h_bare + 1/2 sum_x h_c[x] S^-1 h_c[x],
```

and this one-electron DSE is included exactly once. Coherent displacement,
bosonic polarization, matrix Dyson propagation, and Fan self-energy are
independent of the electronic ERI representation and are unchanged.

## Exact and factorized representations

The driver constructs one `ElectronInteraction` model:

- exact electronic: bare four-index AO ERIs;
- exact CS-GF2: explicit electronic ERIs plus explicit DSE, retained for small
  validation calculations;
- electronic DF: electronic self-energy/Fock factor bundles only;
- DF CS-GF2: electronic factor bundles plus exact `h_c` channels.

The model owns Fock and electronic-self-energy dispatch, so the coupled Dyson
loop does not contain QED/DF method branches. DF CS-GF2 never calls
`ao_eri()` for four orbital bases and keeps `eri`/`eri_bare` unset.

## Scaling and invariants

Factor storage changes from `naux * nao^2` to
`(naux + nactive_mode) * nao^2`; an explicit effective tensor would require
`nao^4`. The optimized DF GF2 contraction retains its existing
`O(ntime * naux * nao^4)` leading structure with `naux` replaced by the small
effective rank increase. Bosonic storage remains
`O(ntime * nmode^2)`.

Public-boundary validation enforces real rank-3 factors, square symmetric AO
matrices, a common AO dimension, and finite values. Reconstruction and Fock
tests prove the DSE channel normalization and restricted-spin prefactors near
machine precision. Self-energy tests use an independent explicit four-index
reference to verify all channel cross terms.

