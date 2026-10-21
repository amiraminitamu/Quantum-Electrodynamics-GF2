# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""GF2 self-energy validation for electronic and exact QED factor channels."""

from __future__ import annotations

import numpy as np

from gf2.reference import _build_transform_objects
from gf2.energy import gf2_correlation
from gf2.grids import build_grids
from gf2.integrals import FactorizedInteraction
from gf2.legendre_ft import legendre_coefficients, time_to_frequency
from gf2.options import GF2Options
from gf2.self_energy import gf2_self_energy_df
from tests.cs_df_helpers import ONE_Z, run_case
from tests.reference_helpers.explicit_gf2 import explicit_gf2_self_energy


def _symmetric(rng, shape):
    value = rng.normal(size=shape)
    return 0.5 * (value + value.swapaxes(-1, -2))


def _transform(values: np.ndarray) -> np.ndarray:
    opts = GF2Options.from_mapping(
        {
            "beta": 12.0,
            "n_ifreq": 12,
            "mid_pt": 4,
            "n_ifreq_full": 100,
            "n_itime": values.shape[0],
            "n_poly": 4,
            "mp2": False,
        }
    )
    grids = build_grids(opts)
    transform = _build_transform_objects(grids, opts)
    coefficients = legendre_coefficients(values, transform["legendre_weights"])
    return time_to_frequency(transform["legendre_trans"], coefficients)


def test_synthetic_effective_self_energy_matches_slow_four_index_reference() -> None:
    rng = np.random.default_rng(521)
    electronic = _symmetric(rng, (3, 2, 2))
    channels = 0.1 * _symmetric(rng, (2, 2, 2))
    green = 0.2 * _symmetric(rng, (6, 2, 2))
    interaction = FactorizedInteraction(electronic, channels)
    exact = explicit_gf2_self_energy(green, interaction.reconstruct())
    factorized = gf2_self_energy_df(2, green, interaction.factors, len(green))
    assert np.allclose(factorized, exact, atol=3.0e-14, rtol=3.0e-13)
    assert np.allclose(_transform(factorized), _transform(exact), atol=2.0e-14)
    weights = np.linspace(0.1, 0.3, len(green))
    np.testing.assert_allclose(
        gf2_correlation(green, factorized, weights),
        gf2_correlation(green, exact, weights),
        atol=2.0e-14,
    )


def test_dyson_update_from_factorized_and_explicit_sigma_agrees() -> None:
    rng = np.random.default_rng(522)
    electronic = _symmetric(rng, (4, 2, 2))
    channels = 0.03 * _symmetric(rng, (1, 2, 2))
    green = 0.1 * _symmetric(rng, (6, 2, 2))
    factors = FactorizedInteraction(electronic, channels)
    sigma_df = _transform(gf2_self_energy_df(2, green, factors.factors, 6))
    sigma_exact = _transform(explicit_gf2_self_energy(green, factors.reconstruct()))
    fock = _symmetric(rng, (1, 2, 2))[0]
    frequencies = (2 * np.arange(len(sigma_df)) + 1) * np.pi / 12.0
    g_df = np.array(
        [
            np.linalg.inv((0.2 + 1j * w) * np.eye(2) - fock - s)
            for w, s in zip(frequencies, sigma_df)
        ]
    )
    g_exact = np.array(
        [
            np.linalg.inv((0.2 + 1j * w) * np.eye(2) - fock - s)
            for w, s in zip(frequencies, sigma_exact)
        ]
    )
    density_df = np.sum(g_df.real, axis=0)
    density_exact = np.sum(g_exact.real, axis=0)
    assert np.allclose(g_df, g_exact, atol=2.0e-13)
    assert np.allclose(density_df, density_exact, atol=2.0e-13)


def test_molecular_df_effective_factors_match_their_explicit_reconstruction() -> None:
    _, _, summary, final, reference = run_case("h2", "sto-3g", "df", ONE_Z)
    assert summary.converged
    factors = reference["df"].self_energy
    exact = explicit_gf2_self_energy(final["G_tau"], factors.reconstruct())
    actual = gf2_self_energy_df(
        reference["nbf"], final["G_tau"], factors.factors, final["G_tau"].shape[0]
    )
    assert np.allclose(actual, exact, atol=2.0e-13, rtol=2.0e-12)
