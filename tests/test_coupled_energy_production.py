# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Equation-level tests for the production coupled fermion--boson energy."""

from __future__ import annotations

import numpy as np

from gf2.coupled_energy import (
    coupled_energy_from_state,
    tail_corrected_boson_equal_time,
)
from gf2.qed import (
    bare_boson_iv,
    bosonic_frequencies,
    periodic_iv_to_time,
    polarization_tau,
    sigma_eb_tau,
)


def test_tail_corrected_free_boson_equal_time_is_minus_coth():
    beta = 7.0
    omega = np.array([0.4, 0.9])
    nu = bosonic_frequencies(beta, 64)
    d0 = bare_boson_iv(omega, nu).astype(complex)
    equal = tail_corrected_boson_equal_time(d0, d0, omega, beta)
    expected = np.diag(-1.0 / np.tanh(0.5 * beta * omega))
    np.testing.assert_allclose(equal, expected, atol=1.0e-15)


def test_free_boson_limit_returns_thermal_photon_energy():
    beta = 8.0
    omega = np.array([0.35])
    ntime = 48
    nu = bosonic_frequencies(beta, ntime)
    d0_iv = bare_boson_iv(omega, nu).astype(complex)

    ztau = np.zeros((ntime, 1, 1))
    weights = np.full(ntime, beta / ntime)

    value = coupled_energy_from_state(
        static_energy=-1.25,
        G_tau=ztau,
        Sigma_ee_tau=ztau,
        Sigma_eb_tau=ztau,
        Pi_tau=ztau,
        D_tau=periodic_iv_to_time(d0_iv, beta),
        D_iv=d0_iv,
        D0_iv=d0_iv,
        d_itime=weights,
        omega=omega,
        beta=beta,
    )
    n_bose = 1.0 / (np.exp(beta * omega[0]) - 1.0)
    expected_photon = omega[0] * n_bose
    np.testing.assert_allclose(value.photon_fluctuation_energy, expected_photon, atol=2.0e-15)
    np.testing.assert_allclose(value.coupled_total, -1.25 + expected_photon, atol=2.0e-15)


def test_sigma_g_and_pi_d_routes_give_identical_interaction_energy():

    # algebraic and does not depend on a physical molecular calculation.
    beta = 5.0
    ntime = 6
    G = np.array([-0.63, -0.48, -0.31, 0.19, 0.33, 0.52])[:, None, None]
    g = np.array([[[0.27]]])
    D_tau = np.array([-1.1, -0.9, -0.8, -0.8, -0.9, -1.1])[:, None, None]
    Sigma_eb = sigma_eb_tau(G, D_tau, g)
    Pi = polarization_tau(G, g)
    weights = np.array([0.25, 0.70, 1.05, 1.05, 0.70, 0.25])

    omega = np.array([0.6])
    nu = bosonic_frequencies(beta, ntime)
    d0_iv = bare_boson_iv(omega, nu).astype(complex)
    value = coupled_energy_from_state(
        static_energy=-2.0,
        G_tau=G,
        Sigma_ee_tau=np.zeros_like(G),
        Sigma_eb_tau=Sigma_eb,
        Pi_tau=Pi,
        D_tau=D_tau,
        D_iv=d0_iv,
        D0_iv=d0_iv,
        d_itime=weights,
        omega=omega,
        beta=beta,
    )

    np.testing.assert_allclose(
        value.eb_interaction_sigma_g,
        value.eb_interaction_pi_d,
        atol=2.0e-15,
    )
    assert abs(value.sigma_g_pi_d_residual) < 2.0e-15


def test_compact_coupled_energy_decomposition():
    beta = 9.0
    ntime = 8
    G = np.linspace(-0.6, 0.5, ntime)[:, None, None]
    Sigma_ee = np.linspace(0.08, -0.03, ntime)[:, None, None]
    g = np.array([[[0.18]]])
    omega = np.array([0.5])
    nu = bosonic_frequencies(beta, ntime)
    d0_iv = bare_boson_iv(omega, nu).astype(complex)
    D_tau = periodic_iv_to_time(d0_iv, beta)
    Sigma_eb = sigma_eb_tau(G, D_tau, g)
    Pi = polarization_tau(G, g)
    weights = np.full(ntime, beta / ntime)

    value = coupled_energy_from_state(
        static_energy=-1.7,
        G_tau=G,
        Sigma_ee_tau=Sigma_ee,
        Sigma_eb_tau=Sigma_eb,
        Pi_tau=Pi,
        D_tau=D_tau,
        D_iv=d0_iv,
        D0_iv=d0_iv,
        d_itime=weights,
        omega=omega,
        beta=beta,
    )
    compact = (
        value.static_energy
        + value.ee_correlation
        + 3.0 * value.eb_fermionic_trace
        + value.photon_coordinate_term
    )
    np.testing.assert_allclose(value.coupled_total, compact, atol=2.0e-15)
    np.testing.assert_allclose(
        value.coupled_correlation,
        value.coupled_total - value.static_energy,
        atol=2.0e-15,
    )
