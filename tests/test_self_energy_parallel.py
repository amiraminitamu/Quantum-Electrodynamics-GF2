# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Regression tests for imaginary-time self-energy parallelism."""

from __future__ import annotations

import numpy as np

from gf2.self_energy import gf2_self_energy, gf2_self_energy_df


def test_exact_parallel_tau_loop_matches_serial() -> None:
    rng = np.random.default_rng(20260807)
    nbf = 3
    n_itime = 7
    green_tau = rng.standard_normal((n_itime, nbf, nbf))
    eri = rng.standard_normal((nbf, nbf, nbf, nbf))

    serial = gf2_self_energy(
        nbf,
        green_tau,
        eri,
        n_itime,
        n_workers=1,
    )
    parallel = gf2_self_energy(
        nbf,
        green_tau,
        eri,
        n_itime,
        n_workers=3,
    )

    np.testing.assert_allclose(parallel, serial, rtol=0.0, atol=1.0e-13)


def test_df_parallel_tau_loop_matches_serial() -> None:
    rng = np.random.default_rng(20260808)
    nbf = 3
    naux = 5
    n_itime = 7
    green_tau = rng.standard_normal((n_itime, nbf, nbf))
    qpq = rng.standard_normal((naux, nbf, nbf))
    qpq = 0.5 * (qpq + np.swapaxes(qpq, 1, 2))

    serial = gf2_self_energy_df(
        nbf,
        green_tau,
        qpq,
        n_itime,
        n_workers=1,
    )
    parallel = gf2_self_energy_df(
        nbf,
        green_tau,
        qpq,
        n_itime,
        n_workers=3,
    )

    np.testing.assert_allclose(parallel, serial, rtol=0.0, atol=1.0e-13)
