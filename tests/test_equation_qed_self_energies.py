# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Sign, orientation, spin-factor, and mode-space checks for QED kernels."""

from __future__ import annotations

import itertools

import numpy as np

from gf2.qed import bare_boson_iv, bosonic_dyson, polarization_tau, sigma_eb_tau


def test_fan_and_polarization_match_literal_loops() -> None:
    rng = np.random.default_rng(821)
    nt, nm, no = 4, 2, 2
    green = rng.normal(size=(nt, no, no))
    g = rng.normal(size=(nm, no, no))
    g = 0.5 * (g + g.swapaxes(-1, -2))
    d = rng.normal(size=(nt, nm, nm))
    fan = np.zeros((nt, no, no))
    pi = np.zeros((nt, nm, nm))
    for t, x, y, i, j, k, ell in itertools.product(
        range(nt), range(nm), range(nm), range(no), range(no), range(no), range(no)
    ):
        fan[t, i, j] -= g[x, i, k] * green[t, k, ell] * g[y, ell, j] * d[t, x, y]
        pi[t, x, y] -= 2.0 * g[x, i, j] * green[t, j, k] * g[y, k, ell] * green[-t - 1, ell, i]
    np.testing.assert_allclose(sigma_eb_tau(green, d, g), fan, atol=2.0e-14)
    np.testing.assert_allclose(polarization_tau(green, g), pi, atol=2.0e-14)
    assert not np.allclose(polarization_tau(green, g), -pi)
    assert not np.allclose(polarization_tau(green, g), 0.5 * pi)


def test_bare_sign_zero_frequency_and_full_dyson_residual() -> None:
    omega = np.array([0.11, 0.23])
    nu = np.array([0.0, 0.4, -0.4])
    d0 = bare_boson_iv(omega, nu).astype(complex)
    np.testing.assert_allclose(np.diag(d0[0]), -2.0 / omega)
    np.testing.assert_array_equal(d0[:, 0, 1], 0.0)
    pi = np.zeros_like(d0)
    pi[:, 0, 0] = 0.013
    pi[:, 1, 1] = -0.007
    pi[:, 0, 1] = pi[:, 1, 0] = 0.004
    dressed = bosonic_dyson(d0, pi)
    residual = np.linalg.inv(dressed) - np.linalg.inv(d0) + pi
    np.testing.assert_allclose(residual, 0.0, atol=1.0e-15)
    assert np.max(np.abs(dressed[:, 0, 1])) > 1.0e-8
