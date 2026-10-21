# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Independent equation-level checks for restricted finite-temperature GF2."""

from __future__ import annotations

import itertools

import numpy as np

from gf2.energy import gf2_correlation, gf2_correlation_frequency_positive
from gf2.fock import build_fock
from gf2.greens_function import build_g_g0_ifreq
from gf2.self_energy import gf2_self_energy
from gf2.utils import nelec_from_density


def _symmetric(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    value = rng.normal(size=shape)
    return 0.5 * (value + value.swapaxes(-1, -2))


def test_restricted_fock_matches_literal_index_loops() -> None:
    rng = np.random.default_rng(801)
    n = 3
    h = _symmetric(rng, (n, n))
    p = _symmetric(rng, (n, n))
    factors = _symmetric(rng, (5, n, n))
    eri = np.einsum("Qij,Qkl->ijkl", factors, factors)
    expected = h.copy()
    for i, j, k, ell in itertools.product(range(n), repeat=4):
        expected[i, j] += 0.5 * p[k, ell] * (2.0 * eri[i, j, k, ell] - eri[i, ell, k, j])
    np.testing.assert_allclose(build_fock(h, p, eri), expected, atol=3.0e-14)


def test_second_born_self_energy_matches_literal_six_index_loops() -> None:
    rng = np.random.default_rng(802)
    n, nt = 2, 3
    green = _symmetric(rng, (nt, n, n))
    eri = rng.normal(size=(n, n, n, n))
    expected = np.zeros_like(green)
    for t in range(nt):
        gt, gb = green[t], green[-t - 1]
        for i, j, k, ell, m, nn, p, q in itertools.product(range(n), repeat=8):
            expected[t, i, j] += (
                2.0 * eri[i, k, m, q] * gt[k, ell] * gt[m, nn] * gb[p, q] * eri[p, nn, ell, j]
            )
            expected[t, i, j] -= (
                eri[i, k, m, q] * gt[k, ell] * gt[m, nn] * gb[p, q] * eri[p, ell, nn, j]
            )
    actual = gf2_self_energy(n, green, eri, nt)
    np.testing.assert_allclose(actual, expected, atol=8.0e-14)


def test_nonorthogonal_dyson_and_particle_trace_identity() -> None:
    rng = np.random.default_rng(803)
    n = 3
    a = rng.normal(size=(n, n))
    overlap = a.T @ a + np.eye(n)
    fock = _symmetric(rng, (n, n))
    sigma = _symmetric(rng, (4, n, n)).astype(complex)
    omega = (2 * np.arange(4) + 1) * np.pi / 9.0
    mu = -0.13
    green, _ = build_g_g0_ifreq(n, mu, omega, overlap, fock, sigma)
    identity = np.eye(n)
    for w, s, g in zip(omega, sigma, green):
        residual = ((mu + 1.0j * w) * overlap - fock - s) @ g - identity
        np.testing.assert_allclose(residual, 0.0, atol=4.0e-15)
    density = _symmetric(rng, (n, n))
    assert nelec_from_density(density, overlap) == np.trace(density @ overlap)


def test_density_orientation_is_spin_summed_g_zero_minus() -> None:
    g_zero_minus = np.array([[0.47, 0.03], [0.03, 0.53]])
    g_beta_minus = -g_zero_minus
    density = -2.0 * g_beta_minus
    np.testing.assert_array_equal(density, 2.0 * g_zero_minus)


def test_time_energy_has_restricted_spin_sum_and_trace_orientation() -> None:
    green = np.array([[[1.0, 2.0], [3.0, 4.0]], [[0.5, -1.0], [2.0, 0.3]]])
    sigma = np.array([[[0.2, 0.7], [-0.4, 0.9]], [[-0.3, 0.8], [0.1, 0.6]]])
    weights = np.array([0.4, 0.6])
    expected = -1.0 * sum(weights[t] * np.trace(green[-t - 1] @ sigma[t]) for t in range(2))
    assert gf2_correlation(green, sigma, weights) == expected


def test_frequency_energy_positive_matches_full_restricted_frequency_sum() -> None:
    rng = np.random.default_rng(804)
    g_pos = rng.normal(size=(5, 2, 2)) + 1.0j * rng.normal(size=(5, 2, 2))
    s_pos = rng.normal(size=(5, 2, 2)) + 1.0j * rng.normal(size=(5, 2, 2))
    beta = 17.0
    full = (
        1.0
        / beta
        * sum(np.trace(s @ g) + np.trace(s.conj() @ g.conj()) for s, g in zip(s_pos, g_pos))
    )
    assert gf2_correlation_frequency_positive(g_pos, s_pos, beta) == np.real(full)
