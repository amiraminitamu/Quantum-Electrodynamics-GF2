# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Restricted Fock contractions for exact QED auxiliary channels."""

from __future__ import annotations

import numpy as np
import pytest

psi4 = pytest.importorskip("psi4")

from gf2.fock import build_jk, build_jk_df
from gf2.integrals import FactorizedInteraction
from gf2.qed import CavityConfiguration, PhotonMode, build_qed_integrals
from tests.cs_df_helpers import molecule


def _symmetric(rng, count, n):
    value = rng.normal(size=(count, n, n))
    return 0.5 * (value + value.swapaxes(1, 2))


def _compare_jk(channels: np.ndarray, density: np.ndarray) -> None:
    explicit = np.einsum("xpq,xrs->pqrs", channels, channels)
    j_exact, k_exact = build_jk(explicit, density)
    j_fact, k_fact = build_jk_df(channels, density)
    assert np.allclose(j_fact, j_exact, atol=2.0e-13, rtol=2.0e-14)
    assert np.allclose(k_fact, k_exact, atol=2.0e-13, rtol=2.0e-14)
    assert np.allclose(j_fact - 0.5 * k_fact, j_exact - 0.5 * k_exact, atol=2.0e-13)


def test_random_multimode_coulomb_and_exchange_separately() -> None:
    rng = np.random.default_rng(511)
    channels = _symmetric(rng, 4, 3)
    density = _symmetric(rng, 1, 3)[0]
    _compare_jk(channels, density)


def test_effective_fock_matches_explicit_full_interaction() -> None:
    rng = np.random.default_rng(512)
    electronic = _symmetric(rng, 5, 3)
    channels = _symmetric(rng, 2, 3)
    density = _symmetric(rng, 1, 3)[0]
    composite = FactorizedInteraction(electronic, channels)
    j_exact, k_exact = build_jk(composite.reconstruct(), density)
    j_df, k_df = build_jk_df(composite.factors, density)
    assert np.allclose(j_df, j_exact, atol=5.0e-13)
    assert np.allclose(k_df, k_exact, atol=5.0e-13)


def test_molecular_dse_fock_matches_explicit_tensor() -> None:
    mol = molecule("lih")
    psi4.set_options({"basis": "sto-3g"})
    basis = psi4.core.Wavefunction.build(mol, "sto-3g").basisset()
    mints = psi4.core.MintsHelper(basis)
    qed = build_qed_integrals(
        np.asarray(mints.ao_dipole()),
        np.asarray(mints.ao_overlap()),
        CavityConfiguration(
            (
                PhotonMode(0.1, 0.012, (0.0, 0.0, 1.0)),
                PhotonMode(0.2, 0.007, (1.0, 0.0, 0.0)),
            )
        ),
    )
    rng = np.random.default_rng(513)
    density = _symmetric(rng, 1, basis.nbf())[0]
    _compare_jk(qed.h_c, density)
