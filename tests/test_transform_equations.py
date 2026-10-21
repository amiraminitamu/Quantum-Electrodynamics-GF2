# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Analytic and round-trip tests for fermionic and bosonic conventions."""

from __future__ import annotations

import numpy as np

from gf2.greens_function import build_g0_itime_all, prepare_fock_orth
from gf2.qed import (
    bare_boson_iv,
    bosonic_frequencies,
    periodic_iv_to_time,
    periodic_time_to_iv,
)
from gf2.grids import cheb_dt, cheb_grid, theta_grid


def test_uniform_boson_transform_sign_and_round_trip() -> None:
    beta, n = 13.0, 16
    tau = beta * np.arange(n) / n
    nu = bosonic_frequencies(beta, n)
    mode = 3
    values = np.exp(-1.0j * nu[mode] * tau)[:, None, None]
    transformed = periodic_time_to_iv(values, beta)
    assert abs(transformed[mode, 0, 0] - beta) < 2.0e-14
    assert np.max(np.abs(np.delete(transformed[:, 0, 0], mode))) < 2.0e-14
    recovered = periodic_iv_to_time(transformed, beta)
    np.testing.assert_allclose(recovered, values, atol=2.0e-15)


def test_bare_boson_on_chebyshev_grid_matches_analytic_time_function() -> None:
    beta, omega = 40.0, np.array([0.3])
    errors = []
    for n in (400, 800):
        theta, shifted = theta_grid(n)
        tau = cheb_grid(np.cos(theta), beta)
        weights = cheb_dt(shifted, beta, n)
        nu = bosonic_frequencies(beta, n)
        d0_iv = bare_boson_iv(omega, nu)
        numerical = periodic_iv_to_time(d0_iv, beta, tau)[:, 0, 0]
        exact = -(np.exp(-omega[0] * tau) + np.exp(-omega[0] * (beta - tau))) / (
            1.0 - np.exp(-beta * omega[0])
        )
        errors.append(np.max(np.abs(numerical - exact)))
        reconstructed_iv = periodic_time_to_iv(exact[:, None, None], beta, tau, weights)
        assert np.max(np.abs(reconstructed_iv[:20, 0, 0] - d0_iv[:20, 0, 0])) < 1.1e-4
    # The endpoint error is the ordinary finite Matsubara-cutoff error and
    # decreases linearly when the symmetric frequency set is doubled.
    assert errors[0] < 6.1e-3
    assert errors[1] < 0.51 * errors[0]


def test_noninteracting_fermion_antiperiodicity() -> None:
    beta = 17.0
    fock = np.diag([-0.5, 0.4])
    state = prepare_fock_orth(fock, np.eye(2))
    tau = np.array([0.3, beta - 0.3])
    positive = build_g0_itime_all(state, np.eye(2), 0.0, tau, beta)
    # G(-tau) = -G(beta-tau), the KMS anti-periodicity used by beta-tau indexing.
    negative_tau = -positive[1]
    assert np.all(np.isfinite(negative_tau))
    np.testing.assert_allclose(negative_tau, -positive[1], atol=0.0)
