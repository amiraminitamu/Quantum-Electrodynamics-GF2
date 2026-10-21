# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Regression tests for the reduced variational LF optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from gf2.lf import (
    LFOrthogonalContext,
    LFParameters,
    damp_lf_parameter_update,
    lf_static_energy,
    lf_stationary_z,
    optimize_lf_parameters_fixed_density,
)


def _context() -> LFOrthogonalContext:
    h = np.array([[-1.10, 0.08], [0.08, -0.25]])
    g = np.array([[0.07, -0.025], [-0.025, -0.045]])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 0.16
    eri[1, 1, 1, 1] = 0.11
    eri[0, 0, 1, 1] = 0.05
    eri[1, 1, 0, 0] = 0.05
    return LFOrthogonalContext.from_ao(
        overlap=np.eye(2),
        s_inv_half=np.eye(2),
        h_pf_ao=h,
        eri_pf_ao=eri,
        g_pf_ao=g,
        omega=0.22,
        e_nuc=0.65,
    )


def _density() -> np.ndarray:
    return np.array([[1.72, 0.10], [0.10, 0.28]])


def test_reduced_optimizer_imposes_exact_stationary_z() -> None:
    ctx = _context()
    density = _density()
    initial = LFParameters(np.array([0.12, -0.08]), -0.3)
    initial_stationary = LFParameters(
        initial.lam,
        lf_stationary_z(ctx, density, initial.lam),
    )

    result = optimize_lf_parameters_fixed_density(
        ctx,
        density,
        initial,
        {
            "lf_opt_method": "BFGS",
            "lf_opt_maxiter": 300,
            "lf_opt_tol": 1.0e-10,
            "lf_opt_gtol": 1.0e-7,
            "lf_param_mix": 1.0,
            "lf_opt_ntrial": 1,
        },
    )

    expected_z = lf_stationary_z(ctx, density, result.params.lam)
    assert result.params.z == pytest.approx(expected_z, abs=1.0e-13)
    assert result.raw_params.z == pytest.approx(
        lf_stationary_z(ctx, density, result.raw_params.lam), abs=1.0e-13
    )
    assert result.objective <= lf_static_energy(ctx, density, initial_stationary) + 1.0e-10
    assert np.isfinite(result.gradient_norm)


def test_lambda_mixing_occurs_before_exact_z_reconstruction() -> None:
    ctx = _context()
    density = _density()
    initial = LFParameters(np.array([0.18, -0.13]), 0.0)
    options = {
        "lf_opt_method": "BFGS",
        "lf_opt_maxiter": 300,
        "lf_opt_tol": 1.0e-10,
        "lf_opt_gtol": 1.0e-7,
        "lf_param_mix": 0.5,
        "lf_max_dlambda": 10.0,
        "lf_opt_ntrial": 1,
    }

    result = optimize_lf_parameters_fixed_density(ctx, density, initial, options)
    expected_lam = initial.lam + 0.5 * (result.raw_params.lam - initial.lam)
    np.testing.assert_allclose(result.params.lam, expected_lam, atol=2.0e-12)
    assert result.params.z == pytest.approx(
        lf_stationary_z(ctx, density, expected_lam), abs=1.0e-13
    )

    # prepared reduced-optimizer result must not be damped a second time.
    accepted = damp_lf_parameter_update(initial, result.params, options)
    np.testing.assert_allclose(accepted.lam, result.params.lam, atol=0.0)
    assert accepted.z == pytest.approx(result.params.z, abs=0.0)


def test_multistart_is_deterministic_for_fixed_seed() -> None:
    ctx = _context()
    density = _density()
    initial = LFParameters(np.zeros(2), 0.0)
    options = {
        "lf_opt_method": "BFGS",
        "lf_opt_maxiter": 300,
        "lf_opt_tol": 1.0e-9,
        "lf_opt_gtol": 1.0e-6,
        "lf_param_mix": 1.0,
        "lf_opt_ntrial": 4,
        "lf_random_seed": 10086,
    }

    first = optimize_lf_parameters_fixed_density(ctx, density, initial, options)
    second = optimize_lf_parameters_fixed_density(ctx, density, initial, options)

    assert first.ntrial == 4
    assert second.ntrial == 4
    np.testing.assert_allclose(first.raw_params.lam, second.raw_params.lam, atol=1.0e-12)
    np.testing.assert_allclose(first.params.lam, second.params.lam, atol=1.0e-12)
    assert first.raw_objective == pytest.approx(second.raw_objective, abs=1.0e-12)


def test_bfgs_does_not_silently_ignore_lambda_bounds() -> None:
    ctx = _context()
    density = _density()
    initial = LFParameters(np.zeros(2), 0.0)
    with pytest.raises(ValueError, match="incompatible"):
        optimize_lf_parameters_fixed_density(
            ctx,
            density,
            initial,
            {
                "lf_opt_method": "BFGS",
                "lf_lambda_bound": 1.0,
            },
        )
