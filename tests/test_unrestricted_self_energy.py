"""Explicit spin-orbital second Born checks on very small AO tensors."""

import numpy as np
import pytest

from gf2.self_energy import (
    gf2_self_energy,
    gf2_self_energy_df,
    gf2_self_energy_unrestricted,
    gf2_self_energy_unrestricted_df,
)


def reference(a, b, v):
    out = [np.zeros_like(a), np.zeros_like(b)]
    for si, ext in enumerate((a, b)):
        for t in range(a.shape[0]):
            for i in range(a.shape[1]):
                for j in range(a.shape[1]):
                    value = 0.0
                    for internal in (a, b):
                        for k in range(a.shape[1]):
                            for m in range(a.shape[1]):
                                for q in range(a.shape[1]):
                                    for ell in range(a.shape[1]):
                                        for n in range(a.shape[1]):
                                            for p in range(a.shape[1]):
                                                value += (
                                                    v[i, k, m, q]
                                                    * ext[t, k, ell]
                                                    * internal[t, m, n]
                                                    * internal[-t - 1, p, q]
                                                    * v[p, n, ell, j]
                                                )
                    for k in range(a.shape[1]):
                        for m in range(a.shape[1]):
                            for q in range(a.shape[1]):
                                for ell in range(a.shape[1]):
                                    for n in range(a.shape[1]):
                                        for p in range(a.shape[1]):
                                            value -= (
                                                v[i, k, m, q]
                                                * ext[t, k, ell]
                                                * ext[t, m, n]
                                                * ext[-t - 1, p, q]
                                                * v[p, ell, n, j]
                                            )
                    out[si][t, i, j] = value
    return tuple(out)


@pytest.mark.parametrize("mode", ["std", "df"])
def test_unrestricted_sigma_reference_and_collapse(mode):
    rng = np.random.default_rng(12)
    q = rng.normal(size=(3, 2, 2))
    q = 0.5 * (q + q.swapaxes(1, 2))
    v = np.einsum("Qpq,Qrs->pqrs", q, q)
    a = rng.normal(size=(3, 2, 2))
    b = rng.normal(size=(3, 2, 2))
    if mode == "std":
        result = gf2_self_energy_unrestricted(a, b, v, n_workers=2)
        collapse = gf2_self_energy_unrestricted(a, a, v)
        restricted = gf2_self_energy(2, a, v, 3)
    else:
        result = gf2_self_energy_unrestricted_df(a, b, q, n_workers=2)
        collapse = gf2_self_energy_unrestricted_df(a, a, q)
        restricted = gf2_self_energy_df(2, a, q, 3)
    expected = reference(a, b, v)
    for got, want in zip(result, expected):
        np.testing.assert_allclose(got, want, atol=1e-10)
    for got in collapse:
        np.testing.assert_allclose(got, restricted, atol=1e-10)
