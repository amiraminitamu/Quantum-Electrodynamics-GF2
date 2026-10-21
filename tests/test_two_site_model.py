# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Regression tests for the exactly solvable two-site model benchmark."""

import numpy as np

from gf2.model_benchmark import (
    DimerParameters,
    csmp2_energy,
    default_gf2_options,
    exact_ground_state,
    exact_thermal,
    run_csgf2,
    solve_cshf,
)


def test_zero_coupling_hubbard_dimer_matches_analytic_energy():
    for U in (0.0, 2.0, 4.0):
        parameters = DimerParameters(U=U, g=0.0, nph=6)
        exact = exact_ground_state(parameters)
        analytic = 0.5 * (U - np.sqrt(U * U + 16.0 * parameters.t**2))
        assert np.isclose(exact.energy, analytic, atol=1.0e-12)


def test_csmp2_has_correct_weak_electron_boson_limit():
    parameters = DimerParameters(U=0.0, g=1.0e-3, nph=8)
    reference = solve_cshf(parameters)
    result = csmp2_energy(parameters, reference)
    expected = -2.0 * parameters.g**2 / (2.0 * parameters.t + parameters.omega)
    assert np.isclose(result.electron_boson_correlation, expected, atol=1.0e-14)
    assert np.isclose(result.energy - reference.energy, expected, atol=1.0e-14)


def test_asymmetric_cshf_satisfies_coherent_stationarity():
    parameters = DimerParameters(U=2.0, g=0.3, delta=0.25, nph=8)
    result = solve_cshf(parameters)
    assert result.converged
    assert result.stationary_residual < 1.0e-12
    assert abs(result.xi) > 1.0e-4


def test_thermal_ed_reaches_ground_state_at_large_beta():
    parameters = DimerParameters(U=2.0, g=0.3, delta=0.2, nph=12)
    ground = exact_ground_state(parameters)
    thermal = exact_thermal(parameters, beta=100.0)
    assert abs(thermal.electron_number - 2.0) < 1.0e-10
    assert np.isclose(thermal.energy, ground.energy, atol=1.0e-10)


def test_production_csgf2_model_entrypoint_in_free_limit():
    parameters = DimerParameters(U=0.0, g=0.0, nph=6)
    reference = solve_cshf(parameters)
    options = default_gf2_options(
        beta=20.0,
        ntime=40,
        nfreq=41,
        nfull=120,
        npoly=12,
        max_iterations=20,
    )
    options.update(
        thresh_energy=1.0e-7,
        thresh_ne=1.0e-7,
        thresh_fock_rms=1.0e-7,
        thresh_sigma_iw=1.0e-6,
        thresh_bosonic=1.0e-6,
    )
    result = run_csgf2(parameters, reference, options)
    assert result.converged
    assert np.isclose(result.energy, -2.0, atol=2.0e-8)
    assert result.residuals["electronic_dyson"] < 1.0e-10
    assert result.residuals["bosonic_dyson"] < 1.0e-10
