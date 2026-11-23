"""Fixed-(N_alpha,N_beta) finite-temperature UGF2 and coherent-state QED-GF2.

The restricted production loop is intentionally separate and unchanged.  This
module shares its grids, transforms, interaction abstraction, and boson energy
convention while keeping every electronic fixed-point object spin resolved.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

import numpy as np
import psi4

from . import chemical_potential, energy, greens_function as gf, integrals, legendre_ft, spline_ft
from .coupled_energy import coupled_energy_unrestricted_from_state
from .diis import ComplexDIIS, RealDIIS
from .reference import _build_transform_objects
from .grids import build_grids
from .interaction import ElectronInteraction
from .options import GF2Options
from .psi4_io import apply_quiet_psi4_options, silence_psi4
from .qed import (
    CavityConfiguration,
    add_multimode_dse,
    bare_boson_iv,
    bosonic_dyson,
    bosonic_frequencies,
    build_qed_integrals,
    coherent_displacement,
    coherent_fock_correction,
    molecular_dipole_ao,
    periodic_iv_to_time,
    periodic_time_to_iv,
    polarization_tau_unrestricted,
    sigma_eb_tau,
)
from .reporting import Reporter, RunSummary, d_real
from .utils import linear_mix, nelec_from_density

SPINS = ("alpha", "beta")


def _uhf_reference(
    mol: psi4.core.Molecule,
    psi4_options: Mapping[str, Any],
    opts: GF2Options,
    cavity: CavityConfiguration | None,
) -> dict[str, Any]:
    """Psi4 UHF densities and synchronized spin Focks, with optional CS-HF."""
    psi4_opts = apply_quiet_psi4_options(psi4_options)
    psi4_opts["reference"] = "uhf"
    psi4.set_options(psi4_opts)
    hf_energy, wfn = psi4.energy("scf", return_wfn=True)
    basis = wfn.basisset()
    mints = psi4.core.MintsHelper(basis)
    S = np.asarray(mints.ao_overlap())
    sh = mints.ao_overlap()
    sh.power(-0.5, 1.0e-14)
    X = np.asarray(sh)
    h = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    n = {"alpha": int(wfn.nalpha()), "beta": int(wfn.nbeta())}
    if n["alpha"] + n["beta"] <= 0:
        raise ValueError("Unrestricted GF2 requires at least one electron.")
    C = {"alpha": np.asarray(wfn.Ca()), "beta": np.asarray(wfn.Cb())}
    eps = {"alpha": np.asarray(wfn.epsilon_a()), "beta": np.asarray(wfn.epsilon_b())}
    P = {s: np.einsum("pi,qi->pq", C[s][:, : n[s]], C[s][:, : n[s]], optimize=True) for s in SPINS}
    qed = None
    eri = None
    df = None
    if cavity is not None:
        elec_dipole = np.asarray(mints.ao_dipole())
        nuclear_dipole = np.einsum(
            "a,ax->x",
            np.asarray([mol.Z(a) for a in range(mol.natom())]),
            np.asarray(mol.geometry()),
            optimize=True,
        )
        dipole = molecular_dipole_ao(elec_dipole, S, nuclear_dipole, sum(n.values()))
        qed = build_qed_integrals(dipole, S, cavity)
        h = h + qed.h_dse
    if opts.eri_type == "std":
        eri = mints.ao_eri().to_array()
        if qed is not None:
            eri = add_multimode_dse(eri, qed.h_c)
        interaction = ElectronInteraction.exact(
            eri,
            str(basis.name()),
            configured_qed_channels=0 if qed is None else len(qed.omega),
            active_qed_channels=0
            if qed is None
            else int(np.count_nonzero(np.any(qed.h_c != 0.0, axis=(1, 2)))),
        )
    else:
        df = integrals.build_df_eri_bundle(mol, basis, opts.se_fit_basis, opts.fock_fit_basis)
        if qed is not None:
            df = df.with_exact_channels(qed.h_c)
        interaction = ElectronInteraction.factorized(
            df, str(basis.name()), opts.se_fit_basis, opts.fock_fit_basis
        )

    def static_focks(density: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        fa, fb = interaction.build_fock_unrestricted(h, density["alpha"], density["beta"])
        return {"alpha": fa, "beta": fb}

    def cs_focks(density: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        f = static_focks(density)
        if qed is not None:
            z = coherent_displacement(qed.g, density["alpha"] + density["beta"], qed.omega)
            correction = coherent_fock_correction(qed.g, z)
            f = {s: f[s] + correction for s in SPINS}
        return f

    cs_hf_converged = qed is None or not np.any(qed.h_c)
    cs_hf_iterations = 0
    if qed is not None and np.any(qed.h_c):
        mix = float(opts.extra.get("cs_hf_mix", 0.1))
        e_last = None
        for iteration in range(int(opts.extra.get("maxiter_qed_scf", 1000))):
            cs_hf_iterations = iteration + 1
            f = cs_focks(P)
            built = {}
            for s in SPINS:
                fo = X @ f[s] @ X
                _, u = np.linalg.eigh(0.5 * (fo + fo.T))
                occ = (X @ u)[:, : n[s]]
                built[s] = occ @ occ.T
            p_new = {
                s: 0.5 * (mix * built[s] + (1 - mix) * P[s] + (mix * built[s] + (1 - mix) * P[s]).T)
                for s in SPINS
            }
            fs = static_focks(p_new)
            e_new = energy.static_energy_unrestricted(
                h,
                fs["alpha"],
                fs["beta"],
                p_new["alpha"],
                p_new["beta"],
                float(mol.nuclear_repulsion_energy()),
            )
            total = p_new["alpha"] + p_new["beta"]
            z = coherent_displacement(qed.g, total, qed.omega)
            a = np.einsum("xij,ji->x", qed.g, total, optimize=True).real
            e_new += float(np.sum(2 * z * a + qed.omega * z * z))
            dp = max(np.linalg.norm(p_new[s] - P[s]) for s in SPINS)
            P = p_new
            if e_last is not None and abs(e_new - e_last) < 1e-8 and dp < 1e-7:
                cs_hf_converged = True
                break
            e_last = e_new
    F = cs_focks(P)
    return {
        "hf_energy": float(hf_energy),
        "scf_wfn": wfn,
        "nbf": S.shape[0],
        "Nalpha": n["alpha"],
        "Nbeta": n["beta"],
        "Ne": sum(n.values()),
        "S": S,
        "S_inv_half": X,
        "H_core": h,
        "P_alpha": P["alpha"],
        "P_beta": P["beta"],
        "P_total": P["alpha"] + P["beta"],
        "F_alpha": F["alpha"],
        "F_beta": F["beta"],
        "epsilon_alpha": eps["alpha"],
        "epsilon_beta": eps["beta"],
        "Ca": C["alpha"],
        "Cb": C["beta"],
        "E_nuc": float(mol.nuclear_repulsion_energy()),
        "eri": eri,
        "df": df,
        "interaction": interaction,
        "qed_data": qed,
        "interaction_metadata": interaction.metadata(),
        "cs_hf_converged": cs_hf_converged,
        "cs_hf_iterations": cs_hf_iterations,
    }


def _initial_mu(evals: np.ndarray, nocc: int, beta: float) -> float:
    if nocc == 0:
        return float(evals[0] - max(1.0, 30.0 / beta))
    if nocc == len(evals):
        return float(evals[-1] + max(1.0, 30.0 / beta))
    return float(0.5 * (evals[nocc - 1] + evals[nocc]))


def run_unrestricted_gf2(
    mol: psi4.core.Molecule,
    psi4_options: Mapping[str, Any],
    opts: GF2Options,
    *,
    reporter: Reporter | None = None,
    psi4_verbose: bool = False,
    method: str = "gf2",
    cavity: CavityConfiguration | None = None,
    diagnostic_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> tuple[float, float, RunSummary]:
    """Spin-resolved electronic or coherent-state GF2 fixed point."""
    method_key = method.lower().replace("_", "-")
    if method_key == "lf-gf2":
        raise NotImplementedError("Unrestricted projected LF-GF2 has not yet been implemented.")
    if method_key not in {"gf2", "electronic", "cs-gf2"}:
        raise ValueError("Unrestricted method must be 'gf2', 'electronic', or 'cs-gf2'.")
    coupled = method_key == "cs-gf2"
    if coupled != (cavity is not None):
        raise ValueError("A cavity is required exactly for method='cs-gf2'.")
    if not psi4_verbose:
        silence_psi4()
    rep = reporter or Reporter(job_name="untitled")
    rep.ensure_banner("Unrestricted Matsubara GF2")
    psi4.core.set_num_threads(opts.n_threads)
    started = time.time()
    rep.section("Unrestricted UHF reference")
    ref = _uhf_reference(mol, psi4_options, opts, cavity)
    if diagnostic_callback:
        diagnostic_callback("reference", ref)
    nbf, S, X, h = ref["nbf"], ref["S"], ref["S_inv_half"], ref["H_core"]
    n = {"alpha": ref["Nalpha"], "beta": ref["Nbeta"]}
    P = {s: ref[f"P_{s}"].copy() for s in SPINS}
    F = {s: ref[f"F_{s}"].copy() for s in SPINS}
    interaction = ref["interaction"]
    qed = ref["qed_data"]
    rep.kv("Spin mode", "unrestricted")
    rep.kv("N alpha", n["alpha"])
    rep.kv("N beta", n["beta"])
    rep.kv("Psi4 UHF energy", f"{ref['hf_energy']:.12f}")
    if coupled:
        rep.kv("CS-HF converged", ref["cs_hf_converged"])
        rep.kv("CS-HF iterations", ref["cs_hf_iterations"])
    for s in SPINS:
        rep.kv(f"Tr[P_{s} S]", f"{nelec_from_density(P[s], S):.8f}")
    rep.kv("Basis functions", nbf)
    rep.kv("ERI type", opts.eri_type)
    grids = build_grids(opts)
    tx = _build_transform_objects(grids, opts)
    nu = bosonic_frequencies(opts.beta, opts.n_itime) if coupled else None
    if diagnostic_callback:
        diagnostic_callback(
            "grids",
            {
                "fermionic_sparse_iomega": grids.ifreq_grid.copy(),
                "imaginary_time": grids.itime_grid.copy(),
                "imaginary_time_weights": grids.d_itime.copy(),
                "bosonic_inu": None if nu is None else nu.copy(),
            },
        )

    mu = {}
    G0_t = {}
    for s in SPINS:
        evals = np.linalg.eigvalsh(X @ F[s] @ X)
        mu[s] = _initial_mu(evals, n[s], opts.beta)
        state = gf.prepare_fock_orth(F[s], X)
        G0_t[s] = gf.build_g0_itime_all(state, X, mu[s], grids.itime_grid, opts.beta)
    sigma_ee = dict(
        zip(SPINS, interaction.build_self_energy_unrestricted(G0_t["alpha"], G0_t["beta"]))
    )
    if coupled:
        assert qed is not None and nu is not None
        D0_iv = bare_boson_iv(qed.omega, nu).astype(complex)
        D_iv = D0_iv.copy()
        D_tau = periodic_iv_to_time(D_iv, opts.beta, grids.itime_grid)
        Pi_tau = polarization_tau_unrestricted(G0_t["alpha"], G0_t["beta"], qed.g)
        Pi_iv = periodic_time_to_iv(Pi_tau, opts.beta, grids.itime_grid, grids.d_itime)
        sigma_eb = {s: sigma_eb_tau(G0_t[s], D_tau, qed.g) for s in SPINS}
    else:
        D0_iv = D_iv = D_tau = Pi_tau = Pi_iv = None
        sigma_eb = {s: np.zeros_like(sigma_ee[s]) for s in SPINS}
    one_shot = (
        energy.mp2_correlation_unrestricted(
            G0_t["alpha"], G0_t["beta"], sigma_ee["alpha"], sigma_ee["beta"], grids.d_itime
        )
        if opts.mp2
        else None
    )
    if one_shot is not None:
        rep.kv("Electronic first-iteration E2", f"{one_shot:.12f}")
        rep.detail("Psi4 MP2 comparison omitted for unrestricted finite-temperature E2.")
    sigma_if = {}
    for s in SPINS:
        coeff = legendre_ft.legendre_coefficients(sigma_ee[s] + sigma_eb[s], tx["legendre_weights"])
        sigma_if[s] = legendre_ft.time_to_frequency(tx["legendre_trans"], coeff)

    f_diis = {s: RealDIIS() for s in SPINS}
    g_diis = {s: [ComplexDIIS() for _ in range(opts.n_ifreq)] for s in SPINS}
    G_iw_prev = {s: None for s in SPINS}
    sigma_t_prev = {s: None for s in SPINS}
    P_prev = {s: P[s].copy() for s in SPINS}
    F_prev = {s: F[s].copy() for s in SPINS}
    sigma_ee_prev = {s: sigma_ee[s].copy() for s in SPINS}
    sigma_eb_prev = {s: sigma_eb[s].copy() for s in SPINS}
    z_prev = (
        coherent_displacement(qed.g, P["alpha"] + P["beta"], qed.omega) if coupled else np.empty(0)
    )
    Pi_prev = None if Pi_iv is None else Pi_iv.copy()
    D_prev = None if D_iv is None else D_iv.copy()
    core = corr = total = 0.0
    converged = False
    residuals: dict[str, float] = {}
    breakdown = None
    for outer in range(opts.maxiter_gf_diis):
        rep.outer_iter(outer + 1, g_mix=opts.g_mix, sigma_mix=opts.sigma_mix)
        fock_rms = {s: np.inf for s in SPINS}
        for inner in range(opts.maxiter_fock_diis):
            G_iw = {}
            G0_iw = {}
            for s in SPINS:
                mu[s], P[s], G_iw[s], G0_iw[s] = chemical_potential.tune_spin_population(
                    F[s],
                    S,
                    X,
                    sigma_if[s],
                    grids.ifreq_grid,
                    tx["B_beta"],
                    n[s],
                    mu[s],
                    beta=opts.beta,
                    thresh=opts.thresh_ne,
                    maxiter=max(opts.maxiter_bisection, opts.maxiter_newton),
                    reporter=rep,
                    use_newton=opts.newton_on,
                    damping=opts.newton_damping,
                )
            fa, fb = interaction.build_fock_unrestricted(h, P["alpha"], P["beta"])
            built = {"alpha": fa, "beta": fb}
            if coupled:
                assert qed is not None
                z = coherent_displacement(qed.g, P["alpha"] + P["beta"], qed.omega)
                correction = coherent_fock_correction(qed.g, z)
                built = {s: built[s] + correction for s in SPINS}
            for s in SPINS:
                f_diis[s].add(built[s], F_prev[s] - built[s])
                F[s] = f_diis[s].extrapolate() if opts.fock_diis_on_off == "on" else built[s]
                fock_rms[s] = float(np.sqrt(np.mean((F[s] - F_prev[s]) ** 2)))
                F_prev[s] = F[s].copy()
            rep.detail(
                f"Iter {inner + 1:4d}   RMS(F alpha) = {d_real(fock_rms['alpha']):>12}"
                f"   RMS(F beta) = {d_real(fock_rms['beta']):>12}"
            )
            if opts.fock_diis_on_off == "off" or max(fock_rms.values()) < opts.thresh_fock_rms:
                break
        g_change = {}
        for s in SPINS:
            old = G0_iw[s] if G_iw_prev[s] is None else G_iw_prev[s]
            mixed = np.empty_like(G_iw[s])
            for k in range(opts.n_ifreq):
                g_diis[s][k].add(G_iw[s][k], old[k] - G_iw[s][k])
                mixed[k] = g_diis[s][k].extrapolate()
            if opts.g_mix is not None and G_iw_prev[s] is not None:
                G_iw[s] = linear_mix(mixed, G_iw_prev[s], opts.g_mix)
            else:
                G_iw[s] = mixed
            g_change[s] = float(np.linalg.norm(G_iw[s] - old))
            G_iw_prev[s] = G_iw[s].copy()
        G_tau = {}
        for s in SPINS:
            state = gf.prepare_fock_orth(F[s], X)
            G0_t[s] = gf.build_g0_itime_all(state, X, mu[s], grids.itime_grid, opts.beta)
            G_tau[s] = spline_ft.giw_to_gtau(tx["B"], G_iw[s], G0_iw[s], G0_t[s], opts.beta)
        sigma_ee_new = dict(
            zip(SPINS, interaction.build_self_energy_unrestricted(G_tau["alpha"], G_tau["beta"]))
        )
        if coupled:
            assert qed is not None and D0_iv is not None and D_iv is not None
            Pi_tau = polarization_tau_unrestricted(G_tau["alpha"], G_tau["beta"], qed.g)
            Pi_iv = periodic_time_to_iv(Pi_tau, opts.beta, grids.itime_grid, grids.d_itime)
            new_d = bosonic_dyson(D0_iv, Pi_iv)
            boson_mix = float(
                opts.extra.get("dx_mix", opts.extra.get("Dx_mix", opts.sigma_mix or 1.0))
            )
            if not 0.0 < boson_mix <= 1.0:
                raise ValueError("dx_mix/Dx_mix must be in (0,1].")
            D_iv = linear_mix(new_d, D_iv, boson_mix)
            D_tau = periodic_iv_to_time(D_iv, opts.beta, grids.itime_grid)
            sigma_eb_new = {s: sigma_eb_tau(G_tau[s], D_tau, qed.g) for s in SPINS}
        else:
            sigma_eb_new = {s: np.zeros_like(sigma_ee_new[s]) for s in SPINS}
        sigma_iw_update = {}
        for s in SPINS:
            sig = sigma_ee_new[s] + sigma_eb_new[s]
            if opts.sigma_mix is not None and sigma_t_prev[s] is not None:
                sig = linear_mix(sig, sigma_t_prev[s], opts.sigma_mix)
            sigma_t_prev[s] = sig.copy()
            coeff = legendre_ft.legendre_coefficients(sig, tx["legendre_weights"])
            new_if = legendre_ft.time_to_frequency(tx["legendre_trans"], coeff)
            sigma_iw_update[s] = float(np.max(np.abs(new_if - sigma_if[s])))
            sigma_if[s] = new_if
        fa, fb = interaction.build_fock_unrestricted(h, P["alpha"], P["beta"])
        core_old, total_old = core, total
        core = energy.static_energy_unrestricted(h, fa, fb, P["alpha"], P["beta"], ref["E_nuc"])
        z = np.empty(0)
        if coupled:
            assert (
                qed is not None
                and Pi_tau is not None
                and D_tau is not None
                and D_iv is not None
                and D0_iv is not None
            )
            p_total = P["alpha"] + P["beta"]
            z = coherent_displacement(qed.g, p_total, qed.omega)
            a = np.einsum("xij,ji->x", qed.g, p_total, optimize=True).real
            core += float(np.sum(2 * z * a + qed.omega * z * z))
            breakdown = coupled_energy_unrestricted_from_state(
                static_energy=core,
                G_alpha=G_tau["alpha"],
                G_beta=G_tau["beta"],
                Sigma_ee_alpha=sigma_ee_new["alpha"],
                Sigma_ee_beta=sigma_ee_new["beta"],
                Sigma_eb_alpha=sigma_eb_new["alpha"],
                Sigma_eb_beta=sigma_eb_new["beta"],
                Pi_tau=Pi_tau,
                D_tau=D_tau,
                D_iv=D_iv,
                D0_iv=D0_iv,
                d_itime=grids.d_itime,
                omega=qed.omega,
                beta=opts.beta,
            )
            corr = breakdown.coupled_correlation
        else:
            corr = energy.gf2_correlation_unrestricted(
                G_tau["alpha"],
                G_tau["beta"],
                sigma_ee_new["alpha"],
                sigma_ee_new["beta"],
                grids.d_itime,
            )
        total = core + corr
        residuals = {"energy": abs(total - total_old)}
        for s in SPINS:
            delta_n = abs(nelec_from_density(P[s], S) - n[s])
            dyson = (
                (mu[s] + 1j * grids.ifreq_grid)[:, None, None] * S[None] - F[s][None] - sigma_if[s]
            ) @ G_iw[s] - np.eye(nbf)[None]
            residuals.update(
                {
                    f"particle_number_{s}": float(delta_n),
                    f"delta_N_{s}": float(nelec_from_density(P[s], S) - n[s]),
                    f"density_{s}": float(np.linalg.norm(P[s] - P_prev[s])),
                    f"fock_{s}": fock_rms[s],
                    f"sigma_{s}": float(np.linalg.norm(sigma_ee_new[s] - sigma_ee_prev[s])),
                    f"sigma_eb_{s}": float(np.linalg.norm(sigma_eb_new[s] - sigma_eb_prev[s])),
                    f"sigma_iw_{s}": sigma_iw_update[s],
                    f"G_{s}": g_change[s],
                    f"electronic_dyson_{s}": float(np.linalg.norm(dyson)),
                }
            )
        residuals["particle_number"] = max(residuals[f"particle_number_{s}"] for s in SPINS)
        residuals["fock_rms"] = max(fock_rms.values())
        residuals["sigma_iw"] = max(sigma_iw_update.values())
        if coupled:
            assert (
                Pi_iv is not None
                and D_iv is not None
                and D0_iv is not None
                and Pi_prev is not None
                and D_prev is not None
            )
            residuals.update(
                {
                    "z": float(np.linalg.norm(z - z_prev)),
                    "pi": float(np.linalg.norm(Pi_iv - Pi_prev)),
                    "d": float(np.linalg.norm(D_iv - D_prev)),
                    "bosonic_dyson": float(
                        np.linalg.norm(np.linalg.inv(D_iv) - np.linalg.inv(D0_iv) + Pi_iv)
                    ),
                }
            )
        if diagnostic_callback:
            payload = {
                "iteration": outer + 1,
                "spin_mode": "unrestricted",
                "density_alpha": P["alpha"].copy(),
                "density_beta": P["beta"].copy(),
                "fock_alpha": F["alpha"].copy(),
                "fock_beta": F["beta"].copy(),
                "mu_alpha": mu["alpha"],
                "mu_beta": mu["beta"],
                "G_alpha_iw": G_iw["alpha"].copy(),
                "G_beta_iw": G_iw["beta"].copy(),
                "G_alpha_tau": G_tau["alpha"].copy(),
                "G_beta_tau": G_tau["beta"].copy(),
                "Sigma_ee_alpha_tau": sigma_ee_new["alpha"].copy(),
                "Sigma_ee_beta_tau": sigma_ee_new["beta"].copy(),
                "Sigma_eb_alpha_tau": sigma_eb_new["alpha"].copy(),
                "Sigma_eb_beta_tau": sigma_eb_new["beta"].copy(),
                "core_energy": core,
                "correlation_energy": corr,
                "total_energy": total,
                "particle_number_alpha": nelec_from_density(P["alpha"], S),
                "particle_number_beta": nelec_from_density(P["beta"], S),
                "residuals": dict(residuals),
            }
            if coupled:
                payload.update(
                    {
                        "Pi_tau": Pi_tau.copy(),
                        "Pi_iv": Pi_iv.copy(),
                        "D_tau": D_tau.copy(),
                        "D_iv": D_iv.copy(),
                        "D0_iv": D0_iv.copy(),
                        "coherent_displacement": z.copy(),
                        "energy_breakdown": breakdown,
                    }
                )
            diagnostic_callback(f"iteration_{outer + 1}", payload)
        rep.energy_row("Static/core", core, abs(core - core_old))
        rep.energy_row("Total GF2", total, residuals["energy"])
        rep.kv("mu alpha", f"{float(mu['alpha']):.10f}")
        rep.kv("mu beta", f"{float(mu['beta']):.10f}")
        rep.kv("N alpha", f"{float(nelec_from_density(P['alpha'], S)):.8f}")
        rep.kv("N beta", f"{float(nelec_from_density(P['beta'], S)):.8f}")
        for s in SPINS:
            P_prev[s] = P[s].copy()
            sigma_ee_prev[s] = sigma_ee_new[s].copy()
            sigma_eb_prev[s] = sigma_eb_new[s].copy()
        z_prev = z.copy()
        if coupled:
            Pi_prev = Pi_iv.copy()
            D_prev = D_iv.copy()
        min_outer = int(opts.extra.get("min_outer_iterations", 1))
        sigma_thresh = float(opts.extra.get("thresh_sigma_iw", 1e-6))
        boson_thresh = float(opts.extra.get("thresh_bosonic", 1e-6))
        converged = (
            outer + 1 >= min_outer
            and residuals["energy"] < opts.thresh_energy
            and residuals["particle_number"] < opts.thresh_ne
            and residuals["fock_rms"] < opts.thresh_fock_rms
            and residuals["sigma_iw"] < sigma_thresh
            and (
                not coupled
                or (
                    residuals["pi"] < boson_thresh
                    and residuals["d"] < boson_thresh
                    and residuals["bosonic_dyson"] < boson_thresh
                )
            )
        )
        if converged:
            rep.ok("Unrestricted GF2 fixed point converged")
            break
    rep.kv("mu alpha", f"{mu['alpha']:.10f}")
    rep.kv("mu beta", f"{mu['beta']:.10f}")
    for s in SPINS:
        rep.kv(f"Tr[P_{s} S]", f"{nelec_from_density(P[s], S):.8f}")
    rep.kv("Total electrons", f"{nelec_from_density(P['alpha'] + P['beta'], S):.8f}")
    rep.kv(
        "||P_alpha - P_beta||",
        d_real(float(np.linalg.norm(P["alpha"] - P["beta"])), 8),
    )
    metadata = interaction.metadata()
    metadata.update(
        {
            "method": method_key,
            "spin_mode": "unrestricted",
            "Nalpha": n["alpha"],
            "Nbeta": n["beta"],
            "mu_alpha": mu["alpha"],
            "mu_beta": mu["beta"],
            "Tr_Palpha_S": nelec_from_density(P["alpha"], S),
            "Tr_Pbeta_S": nelec_from_density(P["beta"], S),
        }
    )
    if coupled:
        metadata["cs_hf_converged"] = bool(ref["cs_hf_converged"])
        metadata["cs_hf_iterations"] = int(ref["cs_hf_iterations"])
    if breakdown is not None:
        metadata["coupled_energy_components"] = {
            "C_ee": breakdown.ee_correlation,
            "C_eb": breakdown.eb_fermionic_trace,
            "E_eb": breakdown.eb_interaction_sigma_g,
            "E_ph": breakdown.photon_fluctuation_energy,
            "photon_coordinate": breakdown.photon_coordinate_term,
            "sigma_g_pi_d_residual": breakdown.sigma_g_pi_d_residual,
        }
    summary = RunSummary(
        job_name=rep.job_name,
        core_gf2=float(core),
        corr_gf2=float(corr),
        converged=converged,
        wall_time_s=time.time() - started,
        hf_energy=ref["hf_energy"],
        mp2_psi4=None,
        mp2_gf2_estimate=one_shot,
        g_mix=opts.g_mix,
        sigma_mix=opts.sigma_mix,
        interaction_metadata=metadata,
        convergence_diagnostics=residuals,
        spin_mode="unrestricted",
        n_alpha=n["alpha"],
        n_beta=n["beta"],
        mu_alpha=mu["alpha"],
        mu_beta=mu["beta"],
        trace_p_alpha_s=nelec_from_density(P["alpha"], S),
        trace_p_beta_s=nelec_from_density(P["beta"], S),
    )
    rep.write_summary_block(summary)
    return float(core), float(corr), summary
