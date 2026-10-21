# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
import numpy as np

from gf2.lf import build_lf_fock


def test_lf_fock_uses_literal_reduced_symmetry_exchange():
    rng = np.random.default_rng(9182)

    n = 4

    h = rng.normal(size=(n, n))
    h = 0.5 * (h + h.T)

    p = rng.normal(size=(n, n))
    p = 0.5 * (p + p.T)

    # Deliberately only impose pair-exchange symmetry:

    # Do NOT impose p<->q or r<->s symmetry.
    v = rng.normal(size=(n, n, n, n))
    v = 0.5 * (v + v.transpose(2, 3, 0, 1))

    j = np.einsum(
        "pqrs,rs->pq",
        v,
        p,
        optimize=True,
    )

    k_correct = np.einsum(
        "psrq,rs->pq",
        v,
        p,
        optimize=True,
    )

    expected = h + j - 0.5 * k_correct
    expected = 0.5 * (expected + expected.T)

    actual = build_lf_fock(
        h,
        p,
        v,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1.0e-13,
        rtol=1.0e-13,
    )

    # Confirm that the formerly used symmetry-dependent contraction is
    # genuinely different for this reduced-symmetry tensor.
    k_old = np.einsum(
        "prqs,rs->pq",
        v,
        p,
        optimize=True,
    )

    assert np.max(np.abs(k_correct - k_old)) > 1.0e-6
