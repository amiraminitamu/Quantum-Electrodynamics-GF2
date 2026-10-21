# Quantum Electrodynamics GF2

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
The package supports real, restricted closed-shell singlets. Projected LF-GF2
is single-mode and requires exact ERIs (`eri_type="std"`).

Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, and Arkajit Mandal.
Please cite [the paper](https://arxiv.org/abs/2606.26076).
[Figure data and plotting scripts](https://github.com/amiraminitamu/qed-gf2-results)
are maintained separately. See `CITATION.cff` for citation details.

MIT license.
