# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Exact low-rank representation of multimode dipole self-energy tensors."""

from __future__ import annotations

import numpy as np
import pytest

psi4 = pytest.importorskip("psi4")

from gf2.integrals import FactorizedInteraction
from gf2.qed import CavityConfiguration, PhotonMode, build_qed_integrals
from tests.cs_df_helpers import molecule


def _symmetric(rng: np.random.Generator, shape: tuple[int, int, int]) -> np.ndarray:
    value = rng.normal(size=shape)
    return 0.5 * (value + value.swapaxes(1, 2))


@pytest.mark.parametrize("nmode", [1, 4])
def test_random_dse_factorization(nmode: int) -> None:
    rng = np.random.default_rng(501 + nmode)
    electronic = _symmetric(rng, (3, 4, 4))
    h_c = _symmetric(rng, (nmode, 4, 4))
    factors = FactorizedInteraction(electronic, h_c)
    expected = np.einsum("Qpq,Qrs->pqrs", electronic, electronic)
    expected += np.einsum("xpq,xrs->pqrs", h_c, h_c)
    assert np.allclose(factors.reconstruct(), expected, atol=3.0e-14, rtol=1.0e-14)
    assert factors.effective_rank == electronic.shape[0] + nmode


def test_unequal_modes_permutation_and_zero_channel() -> None:
    rng = np.random.default_rng(506)
    electronic = _symmetric(rng, (5, 3, 3))
    h_c = _symmetric(rng, (3, 3, 3)) * np.array([0.01, 0.07, 0.0])[:, None, None]
    direct = FactorizedInteraction(electronic, h_c)
    permuted = FactorizedInteraction(electronic, h_c[[2, 0, 1]])
    assert direct.configured_exact_channels == 3
    assert direct.active_exact_channels == 2
    assert direct.effective_rank == 7
    assert np.allclose(direct.reconstruct(), permuted.reconstruct(), atol=2.0e-15)


@pytest.mark.parametrize(
    "modes",
    [
        (PhotonMode(0.1, 0.02, (0.0, 0.0, 1.0)),),
        (
            PhotonMode(0.1, 0.02, (0.0, 0.0, 1.0)),
            PhotonMode(0.2, 0.013, (2**-0.5, 0.0, 2**-0.5)),
        ),
    ],
)
def test_molecular_dipole_channels_reconstruct_in_nonorthogonal_ao_basis(modes) -> None:
    mol = molecule("h2")
    psi4.set_options({"basis": "sto-3g"})
    basis = psi4.core.Wavefunction.build(mol, "sto-3g").basisset()
    mints = psi4.core.MintsHelper(basis)
    overlap = np.asarray(mints.ao_overlap())
    assert not np.allclose(overlap, np.eye(overlap.shape[0]))
    qed = build_qed_integrals(np.asarray(mints.ao_dipole()), overlap, CavityConfiguration(modes))
    factors = FactorizedInteraction(np.empty((0, overlap.shape[0], overlap.shape[0])), qed.h_c)
    expected = np.einsum("xpq,xrs->pqrs", qed.h_c, qed.h_c)
    assert np.allclose(factors.reconstruct(), expected, atol=1.0e-16)
