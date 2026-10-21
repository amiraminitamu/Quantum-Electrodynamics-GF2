# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""One-mode molecular DF CS-GF2 production calculations."""

from __future__ import annotations


import numpy as np
import pytest

from gf2 import run_gf2
from gf2.qed import CavityConfiguration
from gf2.reporting import Reporter
from tests.cs_df_helpers import (
    COMMON_OPTIONS,
    ONE_Z,
    molecule,
    run_case,
)


def _assert_run(system: str, basis: str) -> None:
    result = run_case(system, basis, "df", ONE_Z)
    core, corr, summary, final, reference = result
    assert summary.converged
    assert np.isfinite(core + corr)
    assert abs(final["particle_number"] - reference["Ne"]) < 1.0e-6
    assert np.allclose(final["density"], final["density"].T, atol=1.0e-12)
    assert np.allclose(final["fock"], final["fock"].T, atol=1.0e-12)
    metadata = summary.interaction_metadata
    assert metadata["interaction_mode"] == "df"
    assert metadata["qed_low_rank_channels"] == 1
    assert metadata["effective_factor_rank"] == metadata["electronic_aux_rank"] + 1
    assert metadata["full_four_index_allocated"] is False


def test_h2_sto3g_one_mode_df_cs_gf2() -> None:
    _assert_run("h2", "sto-3g")


def test_h2_ccpvdz_one_mode_df_cs_gf2() -> None:
    _assert_run("h2", "cc-pvdz")


def test_lih_sto3g_has_nonzero_coherent_displacement() -> None:
    result = run_case("lih", "sto-3g", "df", ONE_Z)
    _assert_run("lih", "sto-3g")
    assert np.max(np.abs(result[3]["z"])) > 1.0e-6


@pytest.mark.parametrize(
    ("basis", "energy_tolerance", "density_tolerance"),
    [("sto-3g", 1.0e-4, 2.0e-5), ("cc-pvdz", 2.0e-4, 1.0e-4)],
)
def test_one_mode_df_is_close_to_exact_cs_gf2(
    basis: str, energy_tolerance: float, density_tolerance: float
) -> None:
    df = run_case("h2", basis, "df", ONE_Z)
    exact = run_case("h2", basis, "std", ONE_Z)
    assert df[2].converged and exact[2].converged
    assert abs((df[0] + df[1]) - (exact[0] + exact[1])) < energy_tolerance
    assert np.max(np.abs(df[3]["density"] - exact[3]["density"])) < density_tolerance
    assert np.max(np.abs(df[3]["z"] - exact[3]["z"])) < 1.0e-10


def test_low_rank_qed_increment_is_independent_of_electronic_df_error() -> None:
    df_electronic = run_case("h2", "sto-3g", "df", None)
    df_cs = run_case("h2", "sto-3g", "df", ONE_Z)
    exact_electronic = run_case("h2", "sto-3g", "std", None)
    exact_cs = run_case("h2", "sto-3g", "std", ONE_Z)
    df_increment = (df_cs[0] + df_cs[1]) - (df_electronic[0] + df_electronic[1])
    exact_increment = (exact_cs[0] + exact_cs[1]) - (exact_electronic[0] + exact_electronic[1])
    assert abs(df_increment - exact_increment) < 5.0e-9


def test_larger_auxiliary_bases_improve_h2_ccpvdz_total_energy() -> None:
    exact = run_case("h2", "cc-pvdz", "std", ONE_Z)
    exact_total = exact[0] + exact[1]
    errors = []
    for auxiliary in ("cc-pvdz-ri", "cc-pvtz-ri", "aug-cc-pvdz-ri"):
        options = {
            **COMMON_OPTIONS,
            "eri_type": "df",
            "se_fit_basis": auxiliary,
            "fock_fit_basis": auxiliary,
        }
        core, corr, summary = run_gf2(
            molecule("h2"),
            {"basis": "cc-pvdz", "scf_type": "pk"},
            options,
            reporter=Reporter(echo=False),
            method="cs-gf2",
            cavity=CavityConfiguration(ONE_Z),
        )
        assert summary.converged
        errors.append(abs(core + corr - exact_total))
    assert errors[1] < errors[0]
    assert errors[2] < errors[0]
