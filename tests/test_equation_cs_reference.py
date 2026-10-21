# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Equation audit for multimode Pauli--Fierz and coherent-state mean field."""

from __future__ import annotations

import numpy as np

from gf2.energy import coherent_state_mean_field_energy
from gf2.fock import build_fock, core_energy
from gf2.qed import (
    CavityConfiguration,
    PhotonMode,
    add_multimode_dse,
    build_qed_integrals,
    coherent_displacement,
    coherent_fock_correction,
    molecular_dipole_ao,
)


def test_nonorthogonal_multimode_dse_matches_direct_mode_sum() -> None:
    rng = np.random.default_rng(811)
    n = 3
    a = rng.normal(size=(n, n))
    overlap = a.T @ a + np.eye(n)
    dipole = rng.normal(size=(3, n, n))
    dipole = 0.5 * (dipole + dipole.swapaxes(-1, -2))
    cavity = CavityConfiguration(
        (PhotonMode(0.2, 0.04, (1.0, 0.0, 0.0)), PhotonMode(0.3, 0.02, (0.0, 0.0, 1.0)))
    )
    data = build_qed_integrals(dipole, overlap, cavity)
    expected = sum(0.5 * h @ np.linalg.inv(overlap) @ h for h in data.h_c)
    np.testing.assert_allclose(data.h_dse, expected, atol=2.0e-16)
    eri = rng.normal(size=(n, n, n, n))
    dse = add_multimode_dse(eri, data.h_c) - eri
    direct = sum(np.einsum("pq,rs->pqrs", h, h) for h in data.h_c)
    np.testing.assert_allclose(dse, direct, atol=3.0e-16)


def test_displacement_is_stationary_for_spin_summed_density() -> None:
    rng = np.random.default_rng(812)
    g = rng.normal(size=(2, 3, 3))
    g = 0.5 * (g + g.swapaxes(-1, -2))
    density = rng.normal(size=(3, 3))
    density = 0.5 * (density + density.T)
    omega = np.array([0.17, 0.31])
    z = coherent_displacement(g, density, omega)
    expectation = np.einsum("xij,ji->x", g, density)
    gradient = 2.0 * expectation + 2.0 * omega * z
    np.testing.assert_allclose(gradient, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        coherent_fock_correction(g, z),
        2.0 * sum(z[x] * g[x] for x in range(len(z))),
    )


def test_cs_energy_counts_coherent_terms_once() -> None:
    rng = np.random.default_rng(813)
    n = 3
    h = rng.normal(size=(n, n))
    h = 0.5 * (h + h.T)
    p = rng.normal(size=(n, n))
    p = 0.5 * (p + p.T)
    eri = rng.normal(size=(n, n, n, n))
    g = rng.normal(size=(2, n, n))
    g = 0.5 * (g + g.swapaxes(-1, -2))
    omega = np.array([0.2, 0.4])
    z = coherent_displacement(g, p, omega)
    f_static = build_fock(h, p, eri)
    f_cs = f_static + coherent_fock_correction(g, z)
    a = np.einsum("xij,ji->x", g, p)
    direct = core_energy(f_static, h, p, 0.73) + np.sum(2 * z * a + omega * z**2)
    actual = coherent_state_mean_field_energy(f_cs, h, p, 0.73, g, z, omega)
    np.testing.assert_allclose(actual, direct, atol=2.0e-14)
    wrong_double_count = core_energy(f_cs, h, p, 0.73) + np.sum(2 * z * a + omega * z**2)
    assert abs(actual - wrong_double_count) > 1.0e-8


def test_fixed_number_total_dipole_is_neutral_translation_invariant() -> None:
    rng = np.random.default_rng(814)
    n, ne = 3, 4
    a = rng.normal(size=(n, n))
    overlap = a.T @ a + np.eye(n)
    electronic = rng.normal(size=(3, n, n))
    electronic = 0.5 * (electronic + electronic.swapaxes(-1, -2))
    nuclear = np.array([0.4, -0.2, 0.7])
    shift = np.array([1.3, -0.5, 0.2])
    total = molecular_dipole_ao(electronic, overlap, nuclear, ne)
    translated_electronic = electronic - shift[:, None, None] * overlap
    translated_nuclear = nuclear + ne * shift
    translated = molecular_dipole_ao(translated_electronic, overlap, translated_nuclear, ne)
    np.testing.assert_allclose(translated, total, atol=2.0e-15)
