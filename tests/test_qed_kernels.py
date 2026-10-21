# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
from __future__ import annotations

import numpy as np
import pytest

from gf2.qed import (
    CavityConfiguration,
    PhotonMode,
    add_multimode_dse,
    bare_boson_iv,
    bosonic_dyson,
    bosonic_frequencies,
    build_qed_integrals,
    coherent_displacement,
    coherent_fock_correction,
    polarization_tau,
    sigma_eb_tau,
    sigma_eb_tau_reference,
)


def _cavity() -> CavityConfiguration:
    return CavityConfiguration(
        (PhotonMode(0.1, 0.05, (0.0, 0.0, 1.0)), PhotonMode(0.15, 0.03, (1.0, 0.0, 0.0)))
    )


def test_multimode_dse_is_sum_of_independent_modes() -> None:
    rng = np.random.default_rng(2)
    mu = rng.normal(size=(3, 3, 3))
    mu = 0.5 * (mu + mu.swapaxes(1, 2))
    data = build_qed_integrals(mu, np.eye(3), _cavity())
    eri = rng.normal(size=(3, 3, 3, 3))
    expected = eri + sum(np.einsum("pq,rs->pqrs", h, h) for h in data.h_c)
    assert np.allclose(add_multimode_dse(eri, data.h_c), expected)


def test_fan_optimized_matches_explicit_and_has_mode_cross_terms() -> None:
    rng = np.random.default_rng(3)
    g = rng.normal(size=(2, 2, 2))
    g = 0.5 * (g + g.swapaxes(1, 2))
    G = rng.normal(size=(5, 2, 2))
    D = rng.normal(size=(5, 2, 2))
    D = 0.5 * (D + D.swapaxes(1, 2))
    assert np.allclose(sigma_eb_tau(G, D, g), sigma_eb_tau_reference(G, D, g))
    assert np.any(np.abs(polarization_tau(G, g)[:, 0, 1]) > 1.0e-12)


def test_boson_dyson_is_full_mode_matrix() -> None:
    nu = bosonic_frequencies(20.0, 8)
    d0 = bare_boson_iv(np.array([0.1, 0.15]), nu)
    pi = np.zeros_like(d0, dtype=complex)
    pi[:, 0, 1] = pi[:, 1, 0] = 0.01
    d = bosonic_dyson(d0, pi)
    assert np.allclose(d0[:, 0, 1], 0.0)
    assert np.any(np.abs(d[:, 0, 1]) > 1.0e-12)


def test_displacement_and_fock_are_mode_resolved() -> None:
    g = np.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 2.0]]])
    p = np.diag([2.0, 1.0])
    z = coherent_displacement(g, p, np.array([0.5, 1.0]))
    assert np.allclose(z, [-4.0, -2.0])
    assert np.allclose(coherent_fock_correction(g, z), 2 * (z[0] * g[0] + z[1] * g[1]))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(frequency=0.0, coupling=0.1, polarization=(0.0, 0.0, 1.0)),
        dict(frequency=0.1, coupling=0.1, polarization=(0.0, 0.0, 0.0)),
    ],
)
def test_invalid_mode_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        PhotonMode(**kwargs)
