# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Molecular origin behavior of the clamped-nuclei dipole representation."""

from __future__ import annotations

import numpy as np
import pytest

psi4 = pytest.importorskip("psi4")

from gf2.qed import molecular_dipole_ao


def _dipole_at_shift(shift: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mol = psi4.geometry(
        f"""0 1
H {shift[0]} {shift[1]} {shift[2] - 0.7}
H {shift[0]} {shift[1]} {shift[2] + 0.7}
units bohr
no_reorient
no_com
symmetry c1"""
    )
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", "sto-3g")
    mints = psi4.core.MintsHelper(basis)
    overlap = np.asarray(mints.ao_overlap())
    electronic = np.asarray(mints.ao_dipole())
    coordinates = np.asarray(mol.geometry())
    charges = np.asarray([mol.Z(a) for a in range(mol.natom())])
    nuclear = np.einsum("a,ax->x", charges, coordinates)
    total = molecular_dipole_ao(electronic, overlap, nuclear, electron_count=2)
    return overlap, electronic, total


def test_neutral_h2_total_dipole_integrals_are_translation_invariant() -> None:
    psi4.core.clean()
    base_s, base_e, base_total = _dipole_at_shift(np.zeros(3))
    shift = np.array([1.25, -0.4, 0.33])
    shifted_s, shifted_e, shifted_total = _dipole_at_shift(shift)
    np.testing.assert_allclose(shifted_s, base_s, atol=2.0e-15)
    np.testing.assert_allclose(shifted_e, base_e - shift[:, None, None] * base_s, atol=3.0e-14)
    np.testing.assert_allclose(shifted_total, base_total, atol=3.0e-14)


def test_psi4_ao_dipole_is_electronic_only() -> None:
    overlap, electronic, _ = _dipole_at_shift(np.array([0.8, 0.0, 0.0]))
    np.testing.assert_allclose(electronic[0], -0.8 * overlap, atol=2.0e-14)
