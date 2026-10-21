# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Equation-level tests for the zero-boson-projected LF Option-A kernels."""

from __future__ import annotations

import numpy as np

from gf2.lf import (
    LFOrthogonalContext,
    LFParameters,
    build_lf_effective_orth,
    franck_condon_one,
    franck_condon_two,
    lf_static_energy,
    lf_stationary_z,
)


def _context(n: int = 3) -> LFOrthogonalContext:

    h = np.array(
        [
            [-1.10, 0.12, -0.04],
            [0.12, -0.30, 0.07],
            [-0.04, 0.07, 0.40],
        ]
    )[:n, :n]
    g = np.array(
        [
            [0.08, -0.03, 0.02],
            [-0.03, -0.05, 0.01],
            [0.02, 0.01, 0.04],
        ]
    )[:n, :n]
    eri = np.zeros((n, n, n, n))
    for p in range(n):
        for q in range(n):
            eri[p, p, q, q] = 0.10 / (1 + p + q)
    return LFOrthogonalContext.from_ao(
        overlap=np.eye(n),
        s_inv_half=np.eye(n),
        h_pf_ao=h,
        eri_pf_ao=eri,
        g_pf_ao=g,
        omega=0.2,
        e_nuc=0.7,
    )


def test_franck_condon_factors_match_literal_definitions():
    lam = np.array([0.11, -0.07, 0.04])
    fc1 = franck_condon_one(lam)
    fc2 = franck_condon_two(lam)

    for p in range(3):
        for q in range(3):
            expected = np.exp(-0.5 * (lam[q] - lam[p]) ** 2)
            np.testing.assert_allclose(fc1[p, q], expected, atol=1.0e-15)
            for r in range(3):
                for s in range(3):
                    delta = lam[q] - lam[p] + lam[s] - lam[r]
                    np.testing.assert_allclose(
                        fc2[p, q, r, s],
                        np.exp(-0.5 * delta * delta),
                        atol=1.0e-15,
                    )


def test_trivial_lf_limit_recovers_pf_integrals_and_bare_vertex():
    ctx = _context()
    params = LFParameters(np.zeros(ctx.nbf), 0.0)
    h_lf, eri_lf, g_lf, fc1, fc2 = build_lf_effective_orth(ctx, params)

    np.testing.assert_allclose(fc1, 1.0, atol=0.0)
    np.testing.assert_allclose(fc2, 1.0, atol=0.0)
    np.testing.assert_allclose(h_lf, ctx.h_pf_orth, atol=1.0e-15)
    np.testing.assert_allclose(eri_lf, ctx.eri_pf_orth, atol=1.0e-15)
    np.testing.assert_allclose(g_lf, ctx.g_pf_orth, atol=1.0e-15)


def test_projected_residual_vertex_is_fc_g_minus_omega_lambda_diagonal():
    ctx = _context()
    lam = np.array([0.12, -0.05, 0.03])
    params = LFParameters(lam, -0.2)
    _, _, g_lf, fc1, _ = build_lf_effective_orth(ctx, params)

    expected = fc1 * ctx.g_pf_orth
    expected[np.diag_indices(ctx.nbf)] -= ctx.omega * lam
    np.testing.assert_allclose(g_lf, expected, atol=1.0e-15)

    np.testing.assert_allclose(
        np.diag(g_lf),
        np.diag(ctx.g_pf_orth) - ctx.omega * lam,
        atol=1.0e-15,
    )


def test_fixed_density_z_derivative_matches_projected_residual_vertex():
    ctx = _context(2)
    density = np.array([[1.72, 0.13], [0.13, 0.28]])
    lam = np.array([0.09, -0.04])
    z = -0.31
    params = LFParameters(lam, z)
    _, _, g_lf, _, _ = build_lf_effective_orth(ctx, params)

    expected_derivative = 2.0 * (
        np.einsum("ij,ji->", g_lf, density, optimize=True).real + ctx.omega * z
    )

    step = 1.0e-6
    e_plus = lf_static_energy(ctx, density, LFParameters(lam, z + step))
    e_minus = lf_static_energy(ctx, density, LFParameters(lam, z - step))
    finite_difference = (e_plus - e_minus) / (2.0 * step)
    np.testing.assert_allclose(finite_difference, expected_derivative, rtol=5.0e-9, atol=5.0e-10)


def test_lf_stationary_z_annuls_fixed_density_derivative():
    ctx = _context(2)
    density = np.array([[1.60, 0.08], [0.08, 0.40]])
    lam = np.array([0.07, -0.02])
    z = lf_stationary_z(ctx, density, lam)
    _, _, g_lf, _, _ = build_lf_effective_orth(ctx, LFParameters(lam, z))
    residual = np.einsum("ij,ji->", g_lf, density, optimize=True).real + ctx.omega * z
    assert abs(residual) < 1.0e-14


def test_density_transform_preserves_particle_number():

    S = np.array([[1.0, 0.18], [0.18, 1.0]])
    eig, vec = np.linalg.eigh(S)
    X = vec @ np.diag(eig**-0.5) @ vec.T
    ctx = LFOrthogonalContext.from_ao(
        overlap=S,
        s_inv_half=X,
        h_pf_ao=np.diag([-1.0, 0.2]),
        eri_pf_ao=np.zeros((2, 2, 2, 2)),
        g_pf_ao=np.array([[0.04, 0.01], [0.01, -0.02]]),
        omega=0.3,
        e_nuc=0.0,
    )
    p_orth = np.array([[1.7, 0.1], [0.1, 0.3]])
    p_ao = X @ p_orth @ X.T
    transformed = ctx.density_to_orth(p_ao)

    np.testing.assert_allclose(transformed, p_orth, atol=2.0e-14)
    np.testing.assert_allclose(np.einsum("ij,ji->", p_ao, S), np.trace(p_orth), atol=2.0e-14)
