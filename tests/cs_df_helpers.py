# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Cached molecular fixtures for exact/DF CS-GF2 integration tests."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pytest

psi4 = pytest.importorskip("psi4")

from gf2 import run_gf2
from gf2.qed import CavityConfiguration, PhotonMode
from gf2.reporting import Reporter

COMMON_OPTIONS: dict[str, Any] = {
    "beta": 20.0,
    "n_ifreq": 30,
    "mid_pt": 10,
    "n_ifreq_full": 200,
    "n_itime": 30,
    "n_poly": 20,
    "mp2": False,
    "maxiter_gf_diis": 200,
    "maxiter_fock_diis": 8,
    "n_threads": 1,
    "thresh_energy": 1.0e-5,
    "thresh_ne": 1.0e-6,
    "thresh_fock_rms": 1.0e-6,
    "newton_damping": 0.5,
    "maxiter_newton": 20,
    "n_fit_pts": 1000,
    "cs_hf_mix": 0.2,
    "se_fit_basis": "cc-pvdz-ri",
    "fock_fit_basis": "cc-pvdz-ri",
}

ONE_Z = (PhotonMode(0.10, 0.01, (0.0, 0.0, 1.0)),)
TWO_TILTED = (
    PhotonMode(0.10, 0.01, (0.0, 0.0, 1.0)),
    PhotonMode(0.15, 0.007, (2.0**-0.5, 0.0, 2.0**-0.5)),
)
WATER_MODES = (
    PhotonMode(0.10, 0.006, (2.0**-0.5, 0.0, 2.0**-0.5)),
    PhotonMode(0.16, 0.008, (0.0, 0.0, 1.0)),
)


def molecule(system: str) -> psi4.core.Molecule:
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    geometries = {
        "h2": """0 1
H 0 0 -0.6991986661
H 0 0  0.6991986661
units bohr
no_reorient
no_com
symmetry c1""",
        "lih": """0 1
Li 0 0 0
H  0 0 3.015
units bohr
no_reorient
no_com
symmetry c1""",
        "h2o": """0 1
O  0.000000  0.000000  0.000000
H  1.430428  0.000000  1.107152
H -1.430428  0.000000  1.107152
units bohr
no_reorient
no_com
symmetry c1""",
    }
    return psi4.geometry(geometries[system])


@lru_cache(maxsize=None)
def run_case(
    system: str,
    basis: str,
    eri_type: str,
    modes: tuple[PhotonMode, ...] | None,
) -> tuple[float, float, Any, dict[str, Any], dict[str, Any]]:
    options = {**COMMON_OPTIONS, "eri_type": eri_type}
    p4 = {
        "basis": basis,
        "scf_type": "pk",
        "maxiter": 1000,
        "e_convergence": 1.0e-10,
        "d_convergence": 1.0e-10,
    }
    captures: dict[str, dict[str, Any]] = {}
    kwargs: dict[str, Any] = {}
    if modes is not None:
        kwargs = {
            "method": "cs-gf2",
            "cavity": CavityConfiguration(modes),
        }
    core, corr, summary = run_gf2(
        molecule(system),
        p4,
        options,
        reporter=Reporter(echo=False),
        diagnostic_callback=lambda stage, data: captures.update({stage: data}),
        **kwargs,
    )
    final_name = max(
        (name for name in captures if name.startswith("iteration_")),
        key=lambda name: int(name.split("_")[1]),
    )
    return core, corr, summary, captures[final_name], captures["reference"]
