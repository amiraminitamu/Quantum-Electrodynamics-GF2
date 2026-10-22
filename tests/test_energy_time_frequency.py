# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Independent CS energy decomposition and time/frequency cross-checks."""

from __future__ import annotations

import numpy as np

from gf2 import greens_function as gf
from gf2 import spline_ft
from gf2.energy import (
    coherent_state_mean_field_energy,
    gf2_correlation,
    gf2_correlation_frequency_positive,
)
from gf2.grids import build_grids
from gf2.options import GF2Options
from gf2.fock import build_fock
from gf2.qed import coherent_fock_correction
from tests.cs_df_helpers import COMMON_OPTIONS, ONE_Z, run_case


def test_reported_cs_energy_decomposition_reconstructs_exactly() -> None:
    core, corr, _, final, reference = run_case("h2", "sto-3g", "std", ONE_Z)
    q = reference["qed_data"]
    physical_fock = build_fock(reference["H_core"], final["density"], reference["eri"])
    physical_fock += coherent_fock_correction(q.g, final["z"])
    rebuilt_core = coherent_state_mean_field_energy(
        physical_fock,
        reference["H_core"],
        final["density"],
        reference["E_nuc"],
        q.g,
        final["z"],
        q.omega,
    )
    np.testing.assert_allclose(rebuilt_core, core, atol=2.0e-15)
    np.testing.assert_allclose(rebuilt_core + corr, final["total_energy"], atol=2.0e-15)


def test_converged_molecular_time_and_frequency_energies_agree_with_grid_error() -> None:
    _, _, _, final, reference = run_case("h2", "sto-3g", "std", ONE_Z)
    opts = GF2Options.from_mapping({**COMMON_OPTIONS, "eri_type": "std"})
    grids = build_grids(opts)
    time_energy = gf2_correlation(final["G_tau"], final["Sigma_total_tau"], grids.d_itime)
    interpolation = spline_ft.spline_v_matrix(grids.full_ifreq_grid, grids.ifreq_grid)
    nbf = reference["nbf"]
    zeros_sparse = np.zeros_like(final["Sigma_total_iw_current"])
    zeros_full = np.zeros((len(grids.full_ifreq_grid), nbf, nbf), dtype=complex)
    g0_sparse = gf.build_g_g0_ifreq(
        nbf, final["mu"], grids.ifreq_grid, reference["S"], final["fock"], zeros_sparse
    )[1]
    g0_full = gf.build_g_g0_ifreq(
        nbf, final["mu"], grids.full_ifreq_grid, reference["S"], final["fock"], zeros_full
    )[1]
    g_full = g0_full + np.einsum(
        "fi,iab->fab", interpolation, final["G_iw"] - g0_sparse, optimize=True
    )
    sigma_full = np.einsum(
        "fi,iab->fab", interpolation, final["Sigma_total_iw_current"], optimize=True
    )
    frequency_energy = gf2_correlation_frequency_positive(g_full, sigma_full, opts.beta)
    # Compare the same fermionic trace in both representations, excluding photon energy.
    assert abs(frequency_energy - time_energy) < 4.5e-4
