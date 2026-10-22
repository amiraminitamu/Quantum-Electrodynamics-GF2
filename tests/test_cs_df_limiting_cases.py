# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""End-to-end limiting cases for production density-fitted CS-GF2."""

from __future__ import annotations

import numpy as np
import pytest

from gf2.qed import PhotonMode
from tests.cs_df_helpers import COMMON_OPTIONS, ONE_Z, run_case


ZERO_Z = (PhotonMode(0.1, 0.0, (0.0, 0.0, 1.0)),)
ONE_PLUS_ZERO = ONE_Z + (PhotonMode(0.2, 0.0, (1.0, 0.0, 0.0)),)


def test_zero_coupling_df_cs_equals_electronic_df_plus_free_photon() -> None:
    electronic = run_case("h2", "sto-3g", "df", None)
    cavity = run_case("h2", "sto-3g", "df", ZERO_Z)
    assert electronic[2].converged and cavity[2].converged
    free_photon = 0.1 / np.expm1(COMMON_OPTIONS["beta"] * 0.1)
    # The two reference paths seed mu differently; compare within solver accuracy.
    assert cavity[0] == pytest.approx(electronic[0], abs=1.0e-8, rel=0.0)
    assert cavity[1] == pytest.approx(electronic[1] + free_photon, abs=1.0e-8, rel=0.0)
    np.testing.assert_allclose(cavity[3]["density"], electronic[3]["density"], atol=1.0e-7, rtol=0.0)
    np.testing.assert_allclose(
        cavity[3]["Sigma_ee_tau"], electronic[3]["Sigma_ee_tau"], atol=1.0e-6, rtol=0.0
    )
    np.testing.assert_array_equal(cavity[3]["Sigma_eb_tau"], 0.0)
    assert cavity[2].interaction_metadata["active_qed_low_rank_channels"] == 0


def test_adding_zero_mode_leaves_df_cs_observables_unchanged() -> None:
    one = run_case("h2", "sto-3g", "df", ONE_Z)
    extra = run_case("h2", "sto-3g", "df", ONE_PLUS_ZERO)
    assert one[2].converged and extra[2].converged
    assert extra[0] == pytest.approx(one[0], abs=2.0e-12)
    free_photon = 0.2 / np.expm1(COMMON_OPTIONS["beta"] * 0.2)
    assert extra[1] == pytest.approx(one[1] + free_photon, abs=2.0e-12)
    assert np.allclose(extra[3]["density"], one[3]["density"], atol=2.0e-12)
    assert extra[3]["z"][1] == 0.0
    assert extra[2].interaction_metadata["active_qed_low_rank_channels"] == 1


def test_ordinary_electronic_df_uses_no_qed_channels() -> None:
    result = run_case("h2", "sto-3g", "df", None)
    metadata = result[2].interaction_metadata
    assert metadata["interaction_mode"] == "df"
    assert metadata["qed_low_rank_channels"] == 0
    assert metadata["effective_factor_rank"] == metadata["electronic_aux_rank"]
    assert metadata["full_four_index_allocated"] is False
