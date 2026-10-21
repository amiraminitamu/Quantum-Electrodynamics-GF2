# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Synthetic and molecular validation of the full mode-space CS-GF2 path."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

psi4 = pytest.importorskip("psi4")

from gf2 import run_gf2
from gf2.qed import (
    CavityConfiguration,
    PhotonMode,
    bare_boson_iv,
    bosonic_dyson,
    bosonic_frequencies,
    periodic_iv_to_time,
    periodic_time_to_iv,
    polarization_tau,
    sigma_eb_tau,
)


OPTS = {
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
P4 = {
    "basis": "sto-3g",
    "scf_type": "pk",
    "maxiter": 1000,
    "e_convergence": 1.0e-10,
    "d_convergence": 1.0e-10,
}


def _mol() -> psi4.core.Molecule:
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


def _run(modes: tuple[PhotonMode, ...]):
    captures = {}
    core, corr, summary = run_gf2(
        _mol(),
        P4,
        OPTS,
        method="cs-gf2",
        cavity=CavityConfiguration(modes),
        diagnostic_callback=lambda stage, data: captures.update({stage: data}),
    )
    final_stage = max(
        (s for s in captures if s.startswith("iteration_")),
        key=lambda s: int(s.split("_")[1]),
    )
    return core, corr, summary, captures[final_stage]


@lru_cache(maxsize=None)
def _molecular_runs():
    z = PhotonMode(0.10, 0.01, (0.0, 0.0, 1.0))
    tilted = PhotonMode(0.15, 0.02, (2**-0.5, 0.0, 2**-0.5))
    zero = PhotonMode(0.20, 0.0, (1.0, 0.0, 0.0))
    return {
        "one": _run((z,)),
        "one_plus_zero": _run((z, zero)),
        "two": _run((z, tilted)),
        "two_permuted": _run((tilted, z)),
    }


def test_periodic_boson_transform_round_trip() -> None:
    rng = np.random.default_rng(43)
    values = rng.normal(size=(12, 3, 3))
    recovered = periodic_iv_to_time(periodic_time_to_iv(values, 37.0), 37.0)
    assert np.allclose(recovered, values, atol=2.0e-15)


def test_mode_permutation_covariance_of_kernels() -> None:
    rng = np.random.default_rng(44)
    g = rng.normal(size=(3, 2, 2))
    g = 0.5 * (g + g.swapaxes(1, 2))
    G = rng.normal(size=(10, 2, 2))
    G = 0.5 * (G + G.swapaxes(1, 2))
    pi = polarization_tau(G, g)
    d0 = bare_boson_iv(np.array([0.1, 0.2, 0.3]), bosonic_frequencies(20.0, 10))
    d = bosonic_dyson(d0, periodic_time_to_iv(pi, 20.0))
    sigma = sigma_eb_tau(G, periodic_iv_to_time(d, 20.0), g)
    perm = np.array([2, 0, 1])
    pi_p = polarization_tau(G, g[perm])
    d0_p = d0[:, perm][:, :, perm]
    d_p = bosonic_dyson(d0_p, periodic_time_to_iv(pi_p, 20.0))
    sigma_p = sigma_eb_tau(G, periodic_iv_to_time(d_p, 20.0), g[perm])
    assert np.allclose(pi_p, pi[:, perm][:, :, perm])
    assert np.allclose(d_p, d[:, perm][:, :, perm])
    assert np.allclose(sigma_p, sigma)


def test_uncoupled_synthetic_modes_remain_block_diagonal() -> None:
    tau = np.linspace(0.2, 1.1, 8)
    G = np.array([np.diag([x, 1.3 - 0.2 * x]) for x in tau])
    g = np.array([np.diag([0.04, 0.0]), np.diag([0.0, 0.07])])
    pi = polarization_tau(G, g)
    pi_iv = periodic_time_to_iv(pi, 20.0)
    d0 = bare_boson_iv(np.array([0.1, 0.2]), bosonic_frequencies(20.0, 8))
    d = bosonic_dyson(d0, pi_iv)
    assert np.allclose(pi[:, 0, 1], 0.0)
    assert np.allclose(pi[:, 1, 0], 0.0)
    assert np.allclose(d[:, 0, 1], 0.0)
    assert np.allclose(d[:, 1, 0], 0.0)


def test_identical_modes_have_exchange_symmetry() -> None:
    rng = np.random.default_rng(45)
    G = rng.normal(size=(10, 2, 2))
    G = 0.5 * (G + G.swapaxes(1, 2))
    one = rng.normal(size=(2, 2))
    one = 0.5 * (one + one.T)
    g = np.stack([one, one])
    pi = polarization_tau(G, g)
    d0 = bare_boson_iv(np.array([0.2, 0.2]), bosonic_frequencies(20.0, 10))
    d = bosonic_dyson(d0, periodic_time_to_iv(pi, 20.0))
    assert np.allclose(pi[:, 0, 0], pi[:, 1, 1])
    assert np.allclose(pi[:, 0, 1], pi[:, 1, 0])
    assert np.allclose(d[:, 0, 0], d[:, 1, 1])
    assert np.allclose(d[:, 0, 1], d[:, 1, 0])


def test_one_mode_uses_mode_matrix_layout() -> None:
    data = _molecular_runs()["one"][3]
    assert data["z"].shape == (1,)
    assert data["Pi_tau"].shape[1:] == (1, 1)
    assert data["D_iv"].shape[1:] == (1, 1)


def test_zero_coupling_mode_leaves_electronic_result_unchanged() -> None:
    one = _molecular_runs()["one"]
    extra = _molecular_runs()["one_plus_zero"]
    assert one[2].converged and extra[2].converged
    assert extra[0] == pytest.approx(one[0], abs=1.0e-12)
    assert extra[1] == pytest.approx(one[1], abs=1.0e-12)
    assert np.allclose(extra[3]["density"], one[3]["density"], atol=1.0e-12)
    assert extra[3]["z"][1] == 0.0


def test_molecular_mode_permutation_and_offdiagonal_correlations() -> None:
    direct = _molecular_runs()["two"]
    permuted = _molecular_runs()["two_permuted"]
    a, b = direct[3], permuted[3]
    perm = np.array([1, 0])
    assert direct[2].converged and permuted[2].converged
    assert permuted[0] == pytest.approx(direct[0], abs=1.0e-11)
    assert permuted[1] == pytest.approx(direct[1], abs=1.0e-11)
    assert np.allclose(b["z"], a["z"][perm], atol=1.0e-12)
    assert np.allclose(b["Pi_iv"], a["Pi_iv"][:, perm][:, :, perm], atol=1.0e-12)
    assert np.allclose(b["D_iv"], a["D_iv"][:, perm][:, :, perm], atol=1.0e-11)
    assert np.allclose(a["D0_iv"][:, 0, 1], 0.0)
    assert np.max(np.abs(a["Pi_iv"][:, 0, 1])) > 1.0e-10
    assert np.max(np.abs(a["D_iv"][:, 0, 1])) > 1.0e-10
    assert np.allclose(a["Pi_iv"], a["Pi_iv"].swapaxes(1, 2), atol=1.0e-12)
    assert np.allclose(a["D_iv"], a["D_iv"].swapaxes(1, 2), atol=1.0e-12)
    assert np.allclose(a["density"], a["density"].T, atol=1.0e-12)
    assert np.allclose(a["fock"], a["fock"].T, atol=1.0e-12)
    assert float(a["particle_number"]) == pytest.approx(2.0, abs=1.0e-6)
