# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Regression tests for the corrected, single-DSE CS-GF2 Hamiltonian."""

from __future__ import annotations

import numpy as np
import pytest

psi4 = pytest.importorskip("psi4")

from gf2 import run_gf2
from gf2 import fock as fock_mod
from gf2.reference import _build_cs_fock, _rhf_reference
from gf2.options import GF2Options
from gf2.qed import (
    CavityConfiguration,
    PhotonMode,
    QEDIntegralData,
    add_multimode_dse,
    coherent_displacement,
    coherent_fock_correction,
)


PSI4_OPTIONS = {
    "basis": "sto-3g",
    "scf_type": "pk",
    "maxiter": 1000,
    "e_convergence": 1.0e-10,
    "d_convergence": 1.0e-10,
}
GF2_OPTIONS = {
    "eri_type": "std",
    "beta": 20.0,
    "n_ifreq": 30,
    "mid_pt": 10,
    "n_ifreq_full": 200,
    "n_itime": 30,
    "n_poly": 20,
    "mp2": False,
    "maxiter_gf_diis": 5,
    "maxiter_fock_diis": 8,
    "n_threads": 1,
    "thresh_energy": 1.0e-5,
    "thresh_ne": 1.0e-6,
    "thresh_fock_rms": 1.0e-6,
    "newton_damping": 0.5,
    "maxiter_newton": 20,
    "n_fit_pts": 1000,
    "cs_hf_mix": 0.2,
}


def _h2() -> psi4.core.Molecule:
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    return psi4.geometry("""0 1
H 0 0 -0.6991986661
H 0 0  0.6991986661
units bohr
no_reorient
no_com
symmetry c1""")


def test_removed_dse_configuration_is_rejected() -> None:
    mode = PhotonMode(0.1, 0.01, (0.0, 0.0, 1.0))
    assert not hasattr(CavityConfiguration([mode]), "dse_convention")
    with pytest.raises(TypeError, match="dse_convention"):
        CavityConfiguration([mode], dse_convention="legacy-double-one-body")  # type: ignore[call-arg]


def test_corrected_fock_contains_one_body_dse_once() -> None:
    rng = np.random.default_rng(91)
    n = 3
    h_bare = rng.normal(size=(n, n))
    h_bare = 0.5 * (h_bare + h_bare.T)
    h_dse = rng.normal(size=(n, n))
    h_dse = 0.5 * (h_dse + h_dse.T)
    density = rng.normal(size=(n, n))
    density = 0.5 * (density + density.T)
    eri = rng.normal(size=(n, n, n, n))
    h_c = rng.normal(size=(1, n, n))
    h_c = 0.5 * (h_c + h_c.swapaxes(1, 2))
    omega = np.array([0.2])
    g = -np.sqrt(omega[:, None, None] / 2.0) * h_c
    data = QEDIntegralData(omega=omega, h_c=h_c, g=g, h_dse=h_dse)
    h_qed = h_bare + h_dse
    eri_qed = add_multimode_dse(eri, h_c)
    z = coherent_displacement(g, density, omega)
    direct = fock_mod.build_fock(h_qed, density, eri_qed) + coherent_fock_correction(g, z)
    actual = _build_cs_fock(h_qed, density, eri_qed, data)
    assert np.allclose(actual, direct, atol=1.0e-14, rtol=1.0e-14)
    assert not np.allclose(actual, direct + h_dse)


def test_qed_hf_and_gf_use_same_corrected_fock_builder() -> None:
    cavity = CavityConfiguration([PhotonMode(0.1, 0.01, (0.0, 0.0, 1.0))])
    ref = _rhf_reference(_h2(), PSI4_OPTIONS, GF2Options.from_mapping(GF2_OPTIONS), cavity)
    q = ref["qed_data"]
    expected = _build_cs_fock(ref["H_core"], ref["P"], ref["eri"], q)
    assert np.allclose(ref["H_core"], ref["H_bare"] + q.h_dse, atol=1.0e-14)
    assert np.allclose(ref["eri"], add_multimode_dse(ref["eri_bare"], q.h_c), atol=1.0e-14)
    assert np.allclose(ref["F"], expected, atol=1.0e-13)


def test_zero_coupling_reduces_to_electronic_gf2_plus_free_photon_energy() -> None:
    # This test must exercise the public theory-revision driver, not the

    # DSE helper checks.  Give the small no-mixing regression enough iterations

    options = dict(GF2_OPTIONS)
    options["maxiter_gf_diis"] = 10

    electronic = run_gf2(_h2(), PSI4_OPTIONS, options)
    omega = 0.1
    cavity = CavityConfiguration([PhotonMode(omega, 0.0, (0.0, 0.0, 1.0))])
    coupled = run_gf2(_h2(), PSI4_OPTIONS, options, method="cs-gf2", cavity=cavity)

    assert electronic[2].converged and coupled[2].converged

    assert coupled[0] == pytest.approx(electronic[0], abs=1.0e-12)
    n_bose = 1.0 / (np.exp(options["beta"] * omega) - 1.0)
    free_photon_energy = omega * n_bose
    assert coupled[1] == pytest.approx(electronic[1] + free_photon_energy, abs=1.0e-11)
    assert coupled[2].total_gf2 == pytest.approx(
        electronic[2].total_gf2 + free_photon_energy, abs=1.0e-11
    )
