# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Literal-loop cross-checks for projected LF effective integrals."""

from __future__ import annotations

import numpy as np

from gf2.lf import LFOrthogonalContext, LFParameters, build_lf_effective_orth


def test_lf_effective_integrals_match_literal_option_a_formulas():
    rng = np.random.default_rng(44)
    n = 3
    h = rng.normal(scale=0.2, size=(n, n))
    h = 0.5 * (h + h.T)
    g = rng.normal(scale=0.08, size=(n, n))
    g = 0.5 * (g + g.T)
    eri = rng.normal(scale=0.05, size=(n, n, n, n))
    omega = 0.37
    lam = np.array([0.12, -0.04, 0.07])
    z = -0.23

    ctx = LFOrthogonalContext.from_ao(
        overlap=np.eye(n),
        s_inv_half=np.eye(n),
        h_pf_ao=h,
        eri_pf_ao=eri,
        g_pf_ao=g,
        omega=omega,
        e_nuc=0.0,
    )
    h_lf, eri_lf, g_lf, _, _ = build_lf_effective_orth(ctx, LFParameters(lam, z))

    h_expected = np.zeros_like(h)
    eri_expected = np.zeros_like(eri)
    g_expected = np.zeros_like(g)

    for p in range(n):
        for q in range(n):
            delta_pq = lam[q] - lam[p]
            fc1 = np.exp(-0.5 * delta_pq**2)
            h_expected[p, q] = fc1 * (
                h[p, q] + 2.0 * z * g[p, q] - lam[p] * g[p, q] - g[p, q] * lam[q]
            )
            if p == q:
                h_expected[p, q] += omega * lam[p] ** 2 - 2.0 * omega * z * lam[p]
            g_expected[p, q] = fc1 * g[p, q]
            if p == q:
                g_expected[p, q] -= omega * lam[p]

            for r in range(n):
                for s in range(n):
                    delta_2 = lam[q] - lam[p] + lam[s] - lam[r]
                    eri_expected[p, q, r, s] = eri[p, q, r, s] * np.exp(-0.5 * delta_2**2)

    for p in range(n):
        for q in range(n):
            eri_expected[p, p, q, q] += 2.0 * omega * lam[p] * lam[q]

    for p in range(n):
        for q in range(n):
            fc1 = np.exp(-0.5 * (lam[q] - lam[p]) ** 2)
            for r in range(n):
                pref = -2.0 * g[p, q] * fc1 * lam[r]
                eri_expected[p, q, r, r] += pref
                eri_expected[r, r, p, q] += pref

    h_expected = 0.5 * (h_expected + h_expected.T)
    g_expected = 0.5 * (g_expected + g_expected.T)

    np.testing.assert_allclose(h_lf, h_expected, atol=2.0e-15)
    np.testing.assert_allclose(eri_lf, eri_expected, atol=2.0e-15)
    np.testing.assert_allclose(g_lf, g_expected, atol=2.0e-15)
