# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Small exact molecular check for the restricted default and nested route."""

import pytest

psi4 = pytest.importorskip("psi4")

from gf2 import CavityConfiguration, PhotonMode, run_gf2
from gf2.reporting import Reporter


@pytest.mark.parametrize("cavity_enabled", [False, True])
def test_default_coupled_matches_nested_exact_h2(cavity_enabled):
    options = {
        "eri_type": "std", "beta": 20., "n_ifreq": 30, "mid_pt": 10,
        "n_ifreq_full": 200, "n_itime": 30, "n_poly": 20,
        "mp2": False, "maxiter_gf_diis": 60, "maxiter_fock_diis": 8,
        "n_fit_pts": 1000, "n_threads": 1, "cs_hf_mix": .2,
        "thresh_energy": 1e-5,
    }
    kwargs = {}
    if cavity_enabled:
        kwargs = {
            "method": "cs-gf2",
            "cavity": CavityConfiguration(
                [PhotonMode(.10, .01, (0., 0., 1.))]),
        }
    energies = {}
    for strategy in ("nested", "auto"):
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.clean_variables()
        molecule = psi4.geometry("""0 1
H 0 0 -0.6991986661
H 0 0  0.6991986661
units bohr
no_reorient
no_com
symmetry c1""")
        selected = {**options}
        if strategy == "nested":
            selected["restricted_mu_strategy"] = "nested"
        _, _, summary = run_gf2(
            molecule, {"basis": "sto-3g", "scf_type": "pk"},
            selected, reporter=Reporter(echo=False), **kwargs,
        )
        assert summary.converged
        assert summary.convergence_diagnostics["particle_number"] < 1e-6
        assert summary.interaction_metadata["restricted_mu_strategy"] == (
            "nested" if strategy == "nested" else "coupled"
        )
        if strategy == "auto":
            assert summary.interaction_metadata["restricted_mu_strict_searches"] > 0
        energies[strategy] = summary.total_gf2
    assert abs(energies["auto"] - energies["nested"]) * 627.5094740631 < .02
