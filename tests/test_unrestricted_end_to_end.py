"""Small molecule checks of fixed spin populations and canonical dispatch."""

import numpy as np
import psi4
import pytest
from pathlib import Path

from gf2 import CavityConfiguration, PhotonMode, run_gf2
from gf2.options import GF2Options
from gf2.input_parser import parse_input_file
from gf2.reporting import Reporter
from gf2.unrestricted_driver import _uhf_reference


def molecule(text):
    psi4.core.clean()
    psi4.core.clean_options()
    return psi4.geometry(text)


def options(**changes):
    base = dict(
        eri_type="std",
        beta=10,
        n_ifreq=48,
        mid_pt=12,
        n_ifreq_full=768,
        n_itime=48,
        n_poly=16,
        maxiter_gf_diis=50,
        maxiter_fock_diis=12,
        mp2=False,
        n_threads=1,
        g_mix=0.5,
        sigma_mix=0.5,
        thresh_energy=1e-10,
        thresh_ne=1e-10,
        thresh_fock_rms=1e-10,
        thresh_sigma_iw=1e-10,
    )
    return base | changes


def run(mol, opts, method="gf2", cavity=None):
    snapshots = []

    def capture(stage, payload):
        if stage.startswith("iteration_"):
            snapshots.append(payload)

    result = run_gf2(
        mol,
        {"basis": "sto-3g", "scf_type": "pk"},
        opts,
        reporter=Reporter(echo=False),
        method=method,
        cavity=cavity,
        diagnostic_callback=capture,
    )
    return result, snapshots[-1]


def test_default_is_restricted_and_nonsinglet_rejected():
    assert GF2Options.from_mapping({}).spin_mode == "restricted"
    with pytest.raises(ValueError, match="spin_mode"):
        GF2Options.from_mapping({"spin_mode": "automatic"})
    mol = molecule("0 2\nLi 0 0 0\nsymmetry c1")
    with pytest.raises(ValueError, match="closed-shell singlet"):
        run(mol, options(maxiter_gf_diis=1))
    with pytest.raises(NotImplementedError, match="projected LF"):
        run_gf2(
            mol,
            {"basis": "sto-3g"},
            options(spin_mode="unrestricted"),
            method="lf-gf2",
            cavity=CavityConfiguration([PhotonMode(0.5, 0.01, (0, 0, 1))]),
            reporter=Reporter(echo=False),
        )


@pytest.mark.parametrize(
    "name,method",
    [
        ("open_shell_ugf2.inp", "gf2"),
        ("open_shell_ucs_gf2.inp", "cs-gf2"),
    ],
)
def test_example_input_spin_mode(name, method):
    path = Path(__file__).resolve().parents[1] / "examples" / name
    job, mol, _, opts = parse_input_file(path)
    assert job.method == method
    assert mol.multiplicity() == 2
    assert opts.spin_mode == "unrestricted"


def test_closed_shell_restricted_collapse():
    mol = molecule("0 1\nH 0 0 0\nH 0 0 .74\nsymmetry c1")
    (r_core, r_corr, r_summary), r = run(mol, options())
    (u_core, u_corr, u_summary), u = run(mol, options(spin_mode="unrestricted"))
    assert r_summary.converged and u_summary.converged
    np.testing.assert_allclose(u["density_alpha"] + u["density_beta"], r["density"], atol=1e-8)
    for s in ("alpha", "beta"):
        np.testing.assert_allclose(u[f"fock_{s}"], r["fock"], atol=1e-8)
        np.testing.assert_allclose(u[f"G_{s}_iw"], r["G_iw"], atol=1e-7)
        np.testing.assert_allclose(u[f"G_{s}_tau"], r["G_tau"], atol=1e-7)
        np.testing.assert_allclose(u[f"Sigma_ee_{s}_tau"], r["Sigma_ee_tau"], atol=1e-7)
    np.testing.assert_allclose(
        [u_core, u_corr, u_summary.total_gf2], [r_core, r_corr, r_summary.total_gf2], atol=1e-8
    )


@pytest.mark.parametrize("eri_type", ["std", "df"])
@pytest.mark.parametrize("coupling", [0.0, 0.02])
def test_closed_shell_cs_restricted_collapse(eri_type, coupling):
    mol = molecule("0 1\nH 0 0 0\nH 0 0 .74\nsymmetry c1")
    cavity = CavityConfiguration([PhotonMode(0.5, coupling, (0, 0, 1))])
    opts = options(
        eri_type=eri_type,
        thresh_energy=1e-9,
        thresh_ne=1e-9,
        thresh_fock_rms=1e-9,
        thresh_sigma_iw=1e-9,
    )
    (rc, re, rs), r = run(mol, opts, "cs-gf2", cavity)
    (uc, ue, us), u = run(mol, opts | {"spin_mode": "unrestricted"}, "cs-gf2", cavity)
    assert rs.converged and us.converged
    np.testing.assert_allclose([uc, ue, us.total_gf2], [rc, re, rs.total_gf2], atol=1e-8)
    np.testing.assert_allclose(u["density_alpha"] + u["density_beta"], r["density"], atol=1e-8)
    np.testing.assert_allclose(u["Pi_tau"], r["Pi_tau"], atol=1e-8)
    np.testing.assert_allclose(u["D_iv"], r["D_iv"], atol=1e-8)
    for field in (
        "ee_correlation",
        "eb_interaction_sigma_g",
        "eb_interaction_pi_d",
        "photon_fluctuation_energy",
        "coupled_total",
    ):
        np.testing.assert_allclose(
            getattr(u["energy_breakdown"], field),
            getattr(r["energy_breakdown"], field),
            atol=1e-8,
        )


@pytest.mark.parametrize("eri_type", ["std", "df"])
def test_open_shell_electronic_and_zero_coupling(eri_type):
    mol = molecule("0 2\nLi 0 0 0\nsymmetry c1")
    opts = options(
        spin_mode="unrestricted",
        eri_type=eri_type,
        n_ifreq=24,
        mid_pt=8,
        n_ifreq_full=384,
        n_itime=24,
        n_poly=10,
        maxiter_gf_diis=3,
        maxiter_fock_diis=3,
        thresh_ne=1e-6,
    )
    (_, _, electronic), e = run(mol, opts)
    zero = CavityConfiguration([PhotonMode(0.5, 0.0, (0, 0, 1))])
    (_, _, cs), q = run(mol, opts, "cs-gf2", zero)
    assert electronic.n_alpha == 2 and electronic.n_beta == 1
    for summary, state in ((electronic, e), (cs, q)):
        assert np.isfinite(summary.total_gf2)
        assert abs(summary.trace_p_alpha_s - 2) < 1e-5
        assert abs(summary.trace_p_beta_s - 1) < 1e-5
        for s in ("alpha", "beta"):
            p = state[f"density_{s}"]
            f = state[f"fock_{s}"]
            np.testing.assert_allclose(p, p.T, atol=1e-10)
            np.testing.assert_allclose(f, f.T, atol=1e-10)
            assert np.isfinite(state[f"Sigma_ee_{s}_tau"]).all()
    # The canonical coupled functional retains a free thermal photon term.
    # Its electronic static and second-Born pieces must agree at zero coupling.
    np.testing.assert_allclose(cs.core_gf2, electronic.core_gf2, atol=1e-8)
    np.testing.assert_allclose(q["energy_breakdown"].ee_correlation, electronic.corr_gf2, atol=1e-8)
    for s in ("alpha", "beta"):
        np.testing.assert_allclose(q[f"G_{s}_iw"], e[f"G_{s}_iw"], atol=1e-8)
    np.testing.assert_allclose(q["coherent_displacement"], 0, atol=1e-14)
    np.testing.assert_allclose(q["Sigma_eb_alpha_tau"], 0, atol=1e-14)
    np.testing.assert_allclose(q["Sigma_eb_beta_tau"], 0, atol=1e-14)
    if eri_type == "df":
        assert not cs.interaction_metadata["full_four_index_allocated"]


@pytest.mark.parametrize("eri_type,nmodes", [("std", 1), ("std", 2), ("df", 1), ("df", 2)])
def test_polar_open_shell_qed(eri_type, nmodes):
    mol = molecule("1 2\nLi 0 0 0\nH 0 0 1.6\nsymmetry c1")
    modes = [
        PhotonMode(0.5, 0.02, (0, 0, 1)),
        PhotonMode(0.7, 0.01, (1, 0, 0)),
    ]
    cavity = CavityConfiguration(modes[:nmodes])
    opts = options(
        spin_mode="unrestricted",
        eri_type=eri_type,
        n_ifreq=24,
        mid_pt=8,
        n_ifreq_full=384,
        n_itime=24,
        n_poly=10,
        maxiter_gf_diis=3,
        maxiter_fock_diis=3,
        thresh_ne=1e-6,
    )
    (_, _, summary), state = run(mol, opts, "cs-gf2", cavity)
    assert summary.n_alpha == 2 and summary.n_beta == 1
    assert np.isfinite(summary.total_gf2)
    assert np.isfinite(state["Pi_tau"]).all()
    assert np.isfinite(state["D_iv"]).all()
    assert np.linalg.norm(state["coherent_displacement"]) > 0
    assert summary.interaction_metadata["active_qed_low_rank_channels"] == nmodes
    assert summary.interaction_metadata["full_four_index_allocated"] == (eri_type == "std")


def test_empty_beta_channel():
    mol = molecule("0 2\nH 0 0 0\nsymmetry c1")
    opts = options(
        spin_mode="unrestricted",
        n_ifreq=24,
        mid_pt=8,
        n_ifreq_full=384,
        n_itime=24,
        n_poly=10,
        maxiter_gf_diis=2,
        maxiter_fock_diis=2,
        thresh_ne=1e-6,
    )
    (_, _, summary), _ = run(mol, opts)
    assert summary.n_beta == 0
    assert abs(summary.trace_p_beta_s) < 1e-5


def test_zero_cavity_uhf_reference_matches_psi4():
    mol = molecule("0 2\nLi 0 0 0\nsymmetry c1")
    opts = GF2Options.from_mapping({"spin_mode": "unrestricted", "eri_type": "std"})
    zero = CavityConfiguration([PhotonMode(0.5, 0.0, (0, 0, 1))])
    ref = _uhf_reference(mol, {"basis": "sto-3g", "scf_type": "pk"}, opts, zero)
    wfn = ref["scf_wfn"]
    np.testing.assert_allclose(ref["P_alpha"], np.asarray(wfn.Da()), atol=1e-10)
    np.testing.assert_allclose(ref["P_beta"], np.asarray(wfn.Db()), atol=1e-10)
    np.testing.assert_allclose(ref["F_alpha"], np.asarray(wfn.Fa()), atol=1e-8)
    np.testing.assert_allclose(ref["F_beta"], np.asarray(wfn.Fb()), atol=1e-8)
    assert ref["Ne"] == ref["Nalpha"] + ref["Nbeta"]


def test_open_shell_exact_df_consistency():
    mol = molecule("0 2\nLi 0 0 0\nsymmetry c1")
    opts = options(
        spin_mode="unrestricted",
        beta=5,
        n_ifreq=32,
        mid_pt=10,
        n_ifreq_full=512,
        n_itime=32,
        n_poly=12,
        maxiter_gf_diis=120,
        maxiter_fock_diis=20,
    )
    (_, _, exact), _ = run(mol, opts | {"eri_type": "std"})
    (_, _, fitted), _ = run(mol, opts | {"eri_type": "df"})
    assert exact.converged and fitted.converged
    assert abs(exact.total_gf2 - fitted.total_gf2) < 1e-3
