# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
from __future__ import annotations
import pytest

psi4 = pytest.importorskip("psi4")
from gf2 import run_gf2

PSI4_OPTIONS = {"basis": "cc-pvdz", "scf_type": "pk"}

GF2_OPTIONS_DF = {
    "module": "ft",
    "type": "deterministic",
    "eri_type": "df",
    "se_fit_basis": "cc-pvdz-ri",
    "fock_fit_basis": "cc-pvdz-ri",
    "df_type": "RIFIT",
    "beta": 50,
    "n_ifreq": 300,
    "mid_pt": 100,
    "n_ifreq_full": 10000,
    "n_itime": 300,
    "n_poly": 50,
    "grid_changed": True,
    "thresh_ne": 1.0e-6,
    "thresh_fock_rms": 1.0e-6,
    "thresh_energy": 1.0e-6,
    "bisection_on": False,
    "maxiter_bisection": 60,
    "newton_on": True,
    "newton_damping": 0.5,
    "maxiter_newton": 60,
    "mu_guess_thresh": 1.0e-6,
    "fit_width": 0.05,
    "n_fit_pts": 50000,
    "maxiter_fock_diis": 30,
    "maxiter_gf_diis": 30,
    "n_threads": 2,
    "fock_diis_on_off": "on",
    "debug": False,
    "hf": True,
    "mp2": False,
}


def _h2_geometry(bond: float) -> psi4.core.Molecule:
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    return psi4.geometry(
        f"""
        0 1
        H 0.0 0.0 0.0
        H 0.0 0.0 {bond}
        symmetry c1
        """
    )


def _run(bond: float, **gf2_overrides: object) -> tuple[float, float, bool]:
    opts = {**GF2_OPTIONS_DF, **gf2_overrides}
    mol = _h2_geometry(bond)
    core, corr, summary = run_gf2(mol, PSI4_OPTIONS, opts)
    return core, corr, summary.converged


def test_h2_equilibrium_repeatable() -> None:
    """Back-to-back runs at 0.74 Angstrom remain deterministic."""
    bond = 0.74
    c1, r1, conv1 = _run(bond)
    c2, r2, conv2 = _run(bond)
    assert conv1 and conv2
    assert c1 == c2
    assert r1 == r2


def test_h2_two_angstrom_exact_vs_df_sanity() -> None:
    """The difficult DF point remains close to the independent exact-ERI path."""
    core_df, corr_df, converged_df = _run(2.0, maxiter_gf_diis=250)
    core_exact, corr_exact, converged_exact = _run(2.0, eri_type="std", maxiter_gf_diis=250)
    assert converged_df and converged_exact
    assert abs(core_df - core_exact) < 5.0e-4
    assert abs(corr_df - corr_exact) < 1.0e-3
    assert abs((core_df + corr_df) - (core_exact + corr_exact)) < 4.0e-4


def test_h2_equilibrium_newton_vs_bisection():
    c1, e1, ok1 = _run(0.74, newton_on=True, bisection_on=False)
    c2, e2, ok2 = _run(0.74, newton_on=False, bisection_on=True)
    assert ok1 and ok2
    assert c1 + e1 == pytest.approx(c2 + e2, abs=2e-6)
