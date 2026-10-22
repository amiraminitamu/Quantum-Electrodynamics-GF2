# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Multimode density-fitted CS-GF2 molecular validation."""

from __future__ import annotations

import numpy as np
import pytest

from gf2.integrals import FactorizedInteraction
from tests.cs_df_helpers import TWO_TILTED, WATER_MODES, run_case


def test_h2_two_mode_df_cs_converges_and_has_full_mode_propagator() -> None:
    result = run_case("h2", "sto-3g", "df", TWO_TILTED)
    _, _, summary, final, reference = result
    assert summary.converged
    assert final["z"].shape == (2,)
    assert final["Pi_iv"].shape[1:] == (2, 2)
    assert final["D_iv"].shape[1:] == (2, 2)
    assert np.max(np.abs(final["Pi_iv"][:, 0, 1])) > 1.0e-12
    assert np.max(np.abs(final["D_iv"][:, 0, 1])) > 1.0e-12
    assert reference["df"].self_energy.active_exact_channels == 2


def test_df_mode_permutation_covariance() -> None:
    direct = run_case("h2", "sto-3g", "df", TWO_TILTED)
    permuted = run_case("h2", "sto-3g", "df", TWO_TILTED[::-1])
    permutation = np.array([1, 0])
    assert direct[2].converged and permuted[2].converged
    assert permuted[0] == pytest.approx(direct[0], abs=2.0e-11)
    assert permuted[1] == pytest.approx(direct[1], abs=2.0e-11)
    assert np.allclose(permuted[3]["density"], direct[3]["density"], atol=2.0e-11)
    assert np.allclose(permuted[3]["z"], direct[3]["z"][permutation], atol=2.0e-11)
    assert np.allclose(
        permuted[3]["Pi_iv"],
        direct[3]["Pi_iv"][:, permutation][:, :, permutation],
        atol=2.0e-10,
    )
    assert np.allclose(
        permuted[3]["D_iv"],
        direct[3]["D_iv"][:, permutation][:, :, permutation],
        atol=2.0e-10,
    )


def test_multimode_df_dse_is_sum_of_separate_channels() -> None:
    result = run_case("h2", "sto-3g", "df", TWO_TILTED)
    reference = result[4]
    electronic = reference["df"].self_energy.electronic
    h_c = reference["qed_data"].h_c
    combined = FactorizedInteraction(electronic, h_c).reconstruct()
    expected = FactorizedInteraction(electronic).reconstruct()
    expected += sum(np.einsum("pq,rs->pqrs", h, h) for h in h_c)
    assert np.allclose(combined, expected, atol=2.0e-16)


def test_bent_water_two_modes_have_nonzero_offdiagonal_response() -> None:
    result = run_case("h2o", "sto-3g", "df", WATER_MODES)
    _, _, summary, final, reference = result
    assert summary.converged, {
        "iterations": final["iteration"],
        "residuals": summary.convergence_diagnostics,
    }
    assert np.all(np.linalg.norm(reference["qed_data"].h_c, axis=(1, 2)) > 1.0e-8)
    assert np.max(np.abs(final["z"])) > 1.0e-8
    assert np.max(np.abs(final["Pi_iv"][:, 0, 1])) > 1.0e-12
    assert np.max(np.abs(final["D_iv"][:, 0, 1])) > 1.0e-12


def test_two_mode_df_is_close_to_exact_multimode_cs_gf2() -> None:
    df = run_case("h2", "sto-3g", "df", TWO_TILTED)
    exact = run_case("h2", "sto-3g", "std", TWO_TILTED)
    assert df[2].converged and exact[2].converged
    assert abs((df[0] + df[1]) - (exact[0] + exact[1])) < 1.0e-4
    assert np.max(np.abs(df[3]["density"] - exact[3]["density"])) < 2.0e-5
    assert np.max(np.abs(df[3]["Sigma_ee_tau"] - exact[3]["Sigma_ee_tau"])) < 1.0e-5
    assert np.max(np.abs(df[3]["Sigma_eb_tau"] - exact[3]["Sigma_eb_tau"])) < 1.0e-8
