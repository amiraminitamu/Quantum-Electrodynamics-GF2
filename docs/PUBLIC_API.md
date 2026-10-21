# Public API for version 0.1

The supported import surface is intentionally small:

```python
from gf2 import (
    CavityConfiguration,
    GF2Options,
    PhotonMode,
    RunSummary,
    run_gf2,
)
```

`gf2.__version__` reports the installed version. Modules and functions not
listed above are implementation details and may change during the 0.1 series.

## Calculation entry point

`run_gf2(molecule, psi4_options, gf2_options, *, method="gf2", cavity=None)`
supports these combinations:

| Capability | `method` | `eri_type` | Cavity |
|---|---|---|---|
| electronic exact GF2 | `gf2` | `std` | omitted |
| electronic DF GF2 | `gf2` | `df` | omitted |
| one-/multimode exact CS-GF2 | `cs-gf2` | `std` | required |
| one-/multimode DF CS-GF2 | `cs-gf2` | `df` | required |
| single-mode projected LF-GF2 | `lf-gf2` | `std` | required |

The return value is `(core_energy, correlation_energy, RunSummary)`. The two
energies and `RunSummary.total_gf2` are Hartree. A nonconverged Python API call
returns a summary with `converged=False` and final residuals so applications can
decide how to recover; the CLI treats this as an error by default.

## Cavity configuration

`PhotonMode(frequency, coupling, polarization)` validates positive finite
frequency, finite coupling, and a normalized finite three-component
polarization. Frequency is Hartree and coupling is atomic units.
`CavityConfiguration(modes)` requires at least one `PhotonMode`.

## Input validation

Version 0.1 rejects unsupported ERI names, nonpositive beta, malformed grids,
non-singlet molecules, missing/unexpected cavity configurations, nonpositive
mode frequencies, and malformed polarizations with explicit `ValueError` or
`TypeError` messages. Tensor-kernel public boundaries validate leading shapes.

## Internal modules

Grid builders, Fourier implementations, Fock/self-energy contractions,
chemical-potential search routines, QED integral builders, and diagnostic
callbacks are internal implementation interfaces. They remain tested and
documented scientifically but are not covered by the 0.1 public API promise.

Version 0.1.x is pre-1.0 software and may evolve; incompatible changes will be
documented with the corresponding release.

