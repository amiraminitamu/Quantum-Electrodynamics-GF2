<div align="center">

# Quantum Electrodynamics GF2

<img width="250" alt="image" src="https://github.com/user-attachments/assets/62196e6b-d0e3-4047-8f26-dbb14fa4cd58" />

[![Tests](https://github.com/amiraminitamu/Quantum-Electrodynamics-GF2/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/amiraminitamu/Quantum-Electrodynamics-GF2/actions/workflows/python-package.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![JCTC](https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.6c01285-blue.svg)](https://doi.org/10.1021/acs.jctc.6c01285)

</div>

Deterministic electronic GF2, coherent-state QED-GF2 (CS-GF2), and
single-mode projected Lang-Firsov GF2 (LF-GF2). Includes exact and
density-fitted kernels, parallel self-energy construction, PES tools,
and a two-site Hubbard-Holstein benchmark. Molecular calculations use Psi4.

## Install and run

```bash
conda env create -f environment.yml
conda activate qedgf2
python -m pip install .
qed-gf2 examples/h2.inp --json result.json
```

Use `from gf2 import run_gf2` for Python calculations. Runnable exact/DF and
one-/two-mode CS examples are in `examples/`; LF/PES utilities are in `tools/`.
Set `QED_GF2_SIGMA_WORKERS` to control self-energy workers separately from
Psi4 threads.

The input deck starts with optional ``%`` lines, then a route card, a title,
charge and multiplicity, Cartesian coordinates, and keywords. A
``$molecule`` / ``$rem`` namelist is also accepted, as is the older
``[section]`` form.

```
%nproc=2
#P GF2/cc-pVDZ ERI=DF SCF=PK SYMMETRY=C1

H2 cc-pVDZ

0 1
H  0.0  0.0  0.0
H  0.0  0.0  0.74

beta 50
mp2 true
```

## Unrestricted electronic and cavity GF2

Use `UGF2` or `UCS-GF2` on the route card for a spin-resolved UGF2 or
coherent-state QED-UGF2 calculation (`spin_mode unrestricted` in a `$rem`
or `[gf2]` block does the same). The UHF reference and GF2 iterations
retain separate alpha and beta populations. Run the included small examples:

```bash
qed-gf2 examples/open_shell_ugf2.inp --json ugf2.json
qed-gf2 examples/open_shell_ucs_gf2.inp --json ucs_gf2.json
```

The first is an electronic Li doublet with exact ERIs. The second is a
one-mode LiH+ doublet using density fitting. Check `converged`, the spin
populations, and grid convergence before using an energy in research. The
unrestricted implementation supports electronic GF2 and coherent-state
QED-GF2; projected LF-GF2 remains restricted.

## Restricted chemical potential

Electronic GF2 and CS-GF2 now jointly update the restricted Fock matrix and
chemical potential by default. Each outer iteration finishes with a strict
electron-count search and Fock check. Set `restricted_mu_strategy nested`
in the keyword block to use the earlier nested chemical-potential loop. The `auto`
default retains that earlier loop for projected LF-GF2 and when explicitly
selecting bisection, disabling Newton, or disabling Fock DIIS.

Run the small exact molecular input with
`qed-gf2 examples/h2_coupled_mu.inp --json h2_coupled_mu.json`.
For an exact two-site CS-GF2 case where the routes can differ in convergence,
run `python examples/coupled_mu_model.py`. This compares both routes on the
same Hamiltonian and prints the outer-iteration count and electron error.
Convergence and speed depend on the system and numerical settings.

## Hubbard-Holstein benchmark

```bash
python scripts/run_two_site_figure.py --U-values 0.1 6.0 --delta-values 0 --output model_final
```

## Tests

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

Molecular tests require Psi4 and skip when it is unavailable. Numerical
convergence with temperature and grid sizes must be checked for each system.
The package supports real restricted and unrestricted electronic and CS-GF2.
Projected LF-GF2 supports restricted closed-shell singlets, is single-mode,
and requires exact ERIs (`eri_type="std"`).

Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, and Arkajit Mandal.
Please cite [the paper](https://arxiv.org/abs/2606.26076).
[Figure data and plotting scripts](https://github.com/amiraminitamu/qed-gf2-results)
are maintained separately. See `CITATION.cff` for citation details.

MIT license.
