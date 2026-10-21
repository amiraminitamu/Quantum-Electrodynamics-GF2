# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Small deterministic inputs shared by the installed-package examples."""

from __future__ import annotations

from typing import Iterable

import psi4

from gf2 import CavityConfiguration, PhotonMode, run_gf2
from gf2.reporting import Reporter, RunSummary


BASE_OPTIONS = {
    "beta": 20.0,  # Hartree^-1
    "n_ifreq": 30,
    "mid_pt": 10,
    "n_ifreq_full": 200,
    "n_itime": 30,
    "n_poly": 20,
    "mp2": False,
    "maxiter_gf_diis": 30,
    "maxiter_fock_diis": 8,
    "n_fit_pts": 1000,
    "n_threads": 1,
    "cs_hf_mix": 0.2,
}

ONE_MODE = (PhotonMode(0.10, 0.01, (0.0, 0.0, 1.0)),)
TWO_MODES = (
    PhotonMode(0.10, 0.01, (0.0, 0.0, 1.0)),
    PhotonMode(0.15, 0.007, (2.0**-0.5, 0.0, 2.0**-0.5)),
)


def h2() -> psi4.core.Molecule:
    """Centered H2 at 1.3983973322 bohr; coordinates are explicitly bohr."""
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    return psi4.geometry(
        """0 1
H 0 0 -0.6991986661
H 0 0  0.6991986661
units bohr
no_reorient
no_com
symmetry c1"""
    )


def run_example(eri_type: str, modes: Iterable[PhotonMode] | None) -> RunSummary:
    options = {
        **BASE_OPTIONS,
        "eri_type": eri_type,
        "se_fit_basis": "cc-pvdz-ri",
        "fock_fit_basis": "cc-pvdz-ri",
    }
    kwargs = {}
    if modes is not None:
        kwargs = {
            "method": "cs-gf2",
            "cavity": CavityConfiguration(modes),
        }
    _, _, summary = run_gf2(
        h2(),
        {"basis": "sto-3g", "scf_type": "pk"},
        options,
        reporter=Reporter(echo=False),
        **kwargs,
    )
    if not summary.converged:
        raise RuntimeError(f"GF2 failed to converge: {summary.convergence_diagnostics}")
    return summary


def print_summary(label: str, summary: RunSummary) -> None:
    print(label)
    print(f"  core energy        = {summary.core_gf2:.12f} Ha")
    print(f"  correlation energy = {summary.corr_gf2:.12f} Ha")
    print(f"  total energy       = {summary.total_gf2:.12f} Ha")
    print(f"  converged          = {summary.converged}")
    print(f"  particle error     = {summary.convergence_diagnostics['particle_number']:.3e}")
