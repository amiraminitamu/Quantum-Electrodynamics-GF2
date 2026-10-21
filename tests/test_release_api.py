# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Release-facing API, validation, and metadata checks."""

from __future__ import annotations

import numpy as np
import pytest

psi4 = pytest.importorskip("psi4")

import gf2
from gf2 import CavityConfiguration, GF2Options, PhotonMode, RunSummary, run_gf2
from gf2.reporting import Reporter


def test_public_api_and_authoritative_version() -> None:
    assert gf2.__version__ == "0.1.1"
    assert gf2.__all__ == [
        "CavityConfiguration",
        "GF2Options",
        "PhotonMode",
        "RunSummary",
        "run_gf2",
        "__version__",
    ]
    assert GF2Options and RunSummary and run_gf2
    assert run_gf2.__module__ == "gf2.driver_theory"


def test_public_configuration_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        PhotonMode(0.0, 0.01, (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="three finite"):
        PhotonMode(0.1, 0.01, (0.0, 1.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="normalized"):
        PhotonMode(0.1, 0.01, (0.0, 0.0, 2.0))
    with pytest.raises(TypeError, match="PhotonMode"):
        CavityConfiguration([object()])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="eri_type"):
        GF2Options.from_mapping({"eri_type": "cholesky"})


def test_driver_rejects_unsupported_spin_and_missing_cavity() -> None:
    triplet = psi4.geometry("0 3\nH 0 0 0\nH 0 0 1.4\nunits bohr\nsymmetry c1")
    with pytest.raises(ValueError, match="singlet"):
        run_gf2(triplet, {"basis": "sto-3g"}, {}, reporter=Reporter(echo=False))

    singlet = psi4.geometry("0 1\nH 0 0 0\nH 0 0 1.4\nunits bohr\nsymmetry c1")
    with pytest.raises(ValueError, match="cavity is required"):
        run_gf2(
            singlet,
            {"basis": "sto-3g"},
            {},
            reporter=Reporter(echo=False),
            method="cs-gf2",
        )


def test_run_summary_is_json_serializable_data() -> None:
    summary = RunSummary(
        job_name="test",
        core_gf2=-1.0,
        corr_gf2=-0.1,
        converged=True,
        wall_time_s=0.2,
        convergence_diagnostics={"particle_number": 1.0e-9},
    )
    payload = summary.as_dict()
    assert payload["total_gf2"] == pytest.approx(-1.1)
    assert payload["converged"] is True
    assert np.isfinite(payload["convergence_diagnostics"]["particle_number"])
