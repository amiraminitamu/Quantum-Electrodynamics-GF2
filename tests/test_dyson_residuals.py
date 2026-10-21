# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Residual diagnostics from converged electronic and coupled Dyson loops."""

from __future__ import annotations

from tests.cs_df_helpers import ONE_Z, TWO_TILTED, run_case


def test_converged_exact_cs_reports_small_coupled_residuals() -> None:
    result = run_case("h2", "sto-3g", "std", ONE_Z)
    residuals = result[2].convergence_diagnostics
    assert result[2].converged
    assert residuals["particle_number"] < 1.0e-6
    assert residuals["electronic_dyson"] < 2.0e-4
    assert residuals["bosonic_dyson"] < 1.0e-11
    assert residuals["pi"] < 2.0e-8
    assert residuals["d"] < 3.0e-6


def test_converged_df_multimode_reports_small_coupled_residuals() -> None:
    result = run_case("h2", "sto-3g", "df", TWO_TILTED)
    residuals = result[2].convergence_diagnostics
    assert result[2].converged
    assert residuals["particle_number"] < 1.0e-6
    assert residuals["electronic_dyson"] < 2.0e-4
    assert residuals["bosonic_dyson"] < 1.0e-11
