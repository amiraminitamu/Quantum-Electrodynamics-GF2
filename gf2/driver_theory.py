# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Theory-revision deterministic GF2 driver."

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

import numpy as np

try:
    import psi4
except ImportError:
    psi4 = None  # Model-Hamiltonian entry point does not require Psi4.

from . import chemical_potential
from . import energy as gf2_energy
from . import fock as fock_mod
from . import greens_function as gf
from . import legendre_ft
from . import spline_ft
from .coupled_energy import CoupledEnergyBreakdown, coupled_energy_from_state
from .convergence import outer_fixed_point_converged
from .diis import ComplexDIIS, RealDIIS
from .reference import _build_cs_fock_interaction, _build_transform_objects, _rhf_reference
from .grids import build_grids
from .interaction import ElectronInteraction
from .lf import (
    LFOrthogonalContext,
    LFParameters,
    build_lf_effective,
    damp_lf_parameter_update,
    lf_stationary_z,
    optimize_lf_parameters_fixed_density,
    parameters_from_lambda,
    solve_lf_hf_reference,
)
from .lf_state import (
    LFGF2FixedResult,
    LFGF2FixedState,
    build_lf_problem_signature,
    validate_lf_warm_start_grids,
    validate_lf_warm_start_problem,
)
from .options import GF2Options
from .psi4_io import silence_psi4
from .qed import (
    CavityConfiguration,
    bare_boson_iv,
    bosonic_dyson,
    bosonic_frequencies,
    coherent_displacement,
    periodic_iv_to_time,
    periodic_time_to_iv,
    polarization_tau,
    sigma_eb_tau,
)
from .reporting import Reporter, RunSummary, format_mix_value
from .utils import linear_mix, nelec_from_density


def _cs_static_energy(
    *,
    h_qed: np.ndarray,
    density: np.ndarray,
    interaction: ElectronInteraction,
    g: np.ndarray,
    omega: np.ndarray,
    e_nuc: float,
) -> tuple[float, np.ndarray]:
    """Direct coherent-state static functional, independent of DIIS Fock."""
    z = coherent_displacement(g, density, omega)
    f_static = interaction.build_fock(h_qed, density)
    electronic = fock_mod.core_energy(f_static, h_qed, density, e_nuc)
    a = np.einsum("xij,ji->x", g, density, optimize=True).real
    coherent = float(np.sum(2.0 * z * a + omega * z * z))
    return float(electronic + coherent), z


def _lf_static_energy_from_active_state(
    *,
    h_lf: np.ndarray,
    density: np.ndarray,
    interaction: ElectronInteraction,
    omega: float,
    params: LFParameters,
    e_nuc: float,
) -> float:
    """Projected LF static energy for the current synchronized LF parameters."""
    f_static = interaction.build_fock(h_lf, density)
    return float(fock_mod.core_energy(f_static, h_lf, density, e_nuc) + omega * params.z * params.z)


def _lf_problem_signature_from_reference(ref: Mapping[str, Any]) -> str:
    """Fingerprint the Pauli--Fierz problem before any LF transformation."""
    qed_data = ref.get("qed_data")
    if qed_data is None or len(qed_data.omega) != 1:
        raise ValueError("LF problem signatures require exactly one cavity mode.")
    if ref.get("eri") is None:
        raise RuntimeError("LF problem signatures require an exact ERI tensor.")
    return build_lf_problem_signature(
        h_pf=np.asarray(ref["H_core"], dtype=float),
        eri_pf=np.asarray(ref["eri"], dtype=float),
        g_pf=np.asarray(qed_data.g[0], dtype=float),
        overlap=np.asarray(ref["S"], dtype=float),
        omega=float(qed_data.omega[0]),
        e_nuc=float(ref["E_nuc"]),
        nelec=float(ref["Ne"]),
        nocc=int(ref["nocc"]),
    )


def _prepare_lf_reference(
    ref: dict[str, Any],
    opts: GF2Options,
    *,
    fixed_lambda: np.ndarray | None = None,
    seed_density_override: np.ndarray | None = None,
) -> dict[str, Any]:
    "Prepare the one-mode projected LF reference."
    qed_data = ref["qed_data"]
    if qed_data is None:
        raise ValueError("LF-GF2 requires a cavity.")
    if len(qed_data.omega) != 1:
        raise ValueError("The current projected LF-GF2 implementation is single-mode only.")
    if opts.eri_type != "std":
        raise ValueError(
            "Projected LF-GF2 currently requires eri_type='std'. The LF "
            "Franck--Condon and induced interactions are stored as a dense "
            "four-index tensor; a factorized LF representation has not yet "
            "been introduced."
        )
    if ref["eri"] is None:
        raise RuntimeError("LF-GF2 requires the exact Pauli--Fierz ERI tensor.")
    if seed_density_override is not None and fixed_lambda is None:
        raise ValueError("A correlated seed density is valid only in fixed-lambda mode.")

    h_pf = np.asarray(ref["H_core"], dtype=float).copy()
    eri_pf = np.asarray(ref["eri"], dtype=float).copy()
    g_pf = np.asarray(qed_data.g[0], dtype=float).copy()

    context = LFOrthogonalContext.from_ao(
        overlap=ref["S"],
        s_inv_half=ref["S_inv_half"],
        h_pf_ao=h_pf,
        eri_pf_ao=eri_pf,
        g_pf_ao=g_pf,
        omega=float(qed_data.omega[0]),
        e_nuc=float(ref["E_nuc"]),
    )

    original_meta = ref["interaction_metadata"]
    active = int(original_meta.get("active_qed_low_rank_channels", 1) or 0)
    if seed_density_override is None:
        seed_density = np.asarray(ref["P"], dtype=float).copy()
    else:
        seed_density = np.asarray(seed_density_override, dtype=float)
        if seed_density.shape != (context.nbf, context.nbf):
            raise ValueError("Warm-start density has an incompatible LF dimension.")
        if not np.all(np.isfinite(seed_density)):
            raise ValueError("Warm-start density must contain only finite values.")
        seed_density = 0.5 * (seed_density + seed_density.T)

    if fixed_lambda is None:
        lf_hf = solve_lf_hf_reference(
            context,
            seed_density,
            int(ref["nocc"]),
            opts.extra,
        )
        params = lf_hf.params
        effective = lf_hf.effective
        reference_density = lf_hf.density_ao
        reference_fock = lf_hf.fock_ao
        reference_energy = float(lf_hf.energy)
        reference_mode = "static-lf-hf"
        lf_hf_converged: bool | None = bool(lf_hf.converged)
        lf_hf_iterations = int(lf_hf.iterations)
        lf_hf_residuals = dict(lf_hf.residuals)
    else:
        lam = np.asarray(fixed_lambda, dtype=float)
        if lam.shape != (context.nbf,):
            raise ValueError(
                "lf_fixed_lambda must contain one finite value per LF orbital; "
                f"expected shape {(context.nbf,)}, received {lam.shape}."
            )
        if not np.all(np.isfinite(lam)):
            raise ValueError("lf_fixed_lambda must contain only finite values.")
        params = parameters_from_lambda(context, seed_density, lam)
        effective = build_lf_effective(context, params)
        reference_density = seed_density
        reference_mode = (
            "fixed-lambda-warm-state"
            if seed_density_override is not None
            else "fixed-lambda-cs-hf-seed"
        )
        lf_hf_converged = None
        lf_hf_iterations = 0
        lf_hf_residuals = {}

    interaction = ElectronInteraction.exact(
        effective.eri_ao,
        "LF projected / fixed symmetric-orthogonal basis",
        configured_qed_channels=1,
        active_qed_channels=active,
        reduced_symmetry=True,
    )

    if fixed_lambda is not None:
        reference_fock = interaction.build_fock(effective.h_ao, reference_density)
        reference_energy = _lf_static_energy_from_active_state(
            h_lf=effective.h_ao,
            density=reference_density,
            interaction=interaction,
            omega=float(qed_data.omega[0]),
            params=params,
            e_nuc=float(ref["E_nuc"]),
        )

    updated = dict(ref)
    updated.update(
        {
            "H_pf": h_pf,
            "eri_pf": eri_pf,
            "g_pf": g_pf,
            "H_core": effective.h_ao,
            "eri": effective.eri_ao,
            "P": reference_density,
            "F": reference_fock,
            "interaction": interaction,
            "interaction_metadata": interaction.metadata(),
            "boson_g": effective.g_lf_ao[None, :, :],
            "lf_context": context,
            "lf_params": params,
            "lf_reference_mode": reference_mode,
            "lf_reference_energy": float(reference_energy),
            "lf_hf_energy": float(reference_energy),
            "lf_hf_converged": lf_hf_converged,
            "lf_hf_iterations": lf_hf_iterations,
            "lf_hf_residuals": lf_hf_residuals,
        }
    )
    return updated


def run_gf2(
    mol: psi4.core.Molecule,
    psi4_options: Mapping[str, Any],
    gf2_options: Mapping[str, Any] | GF2Options,
    *,
    reporter: Reporter | None = None,
    psi4_verbose: bool = False,
    method: str = "gf2",
    cavity: CavityConfiguration | None = None,
    diagnostic_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    lf_fixed_lambda: np.ndarray | None = None,
    lf_initial_state: LFGF2FixedState | None = None,
    lf_state_sink: dict[str, LFGF2FixedState] | None = None,
    reference_data: Mapping[str, Any] | None = None,
) -> tuple[float, float, RunSummary]:
    "Run electronic GF2, CS-GF2, or projected LF-GF2."
    np.set_printoptions(precision=5, linewidth=200, suppress=True)
    if reference_data is None and psi4 is None:
        raise RuntimeError("Molecular calculations require Psi4; models require reference_data.")
    if reference_data is not None and method.lower().replace("_", "-") != "cs-gf2":
        raise ValueError("Model reference injection currently supports CS-GF2 only.")
    if reference_data is None and int(mol.multiplicity()) != 1:
        raise ValueError(
            "Only closed-shell singlet calculations are supported; "
            f"received multiplicity={int(mol.multiplicity())}."
        )

    method_key = method.lower().replace("_", "-")
    if method_key not in {"gf2", "electronic", "cs-gf2", "lf-gf2"}:
        raise ValueError("method must be 'gf2'/'electronic', 'cs-gf2', or 'lf-gf2'.")
    is_cs = method_key == "cs-gf2"
    is_lf = method_key == "lf-gf2"
    is_coupled = is_cs or is_lf
    if is_coupled != (cavity is not None):
        raise ValueError("A cavity is required exactly for method='cs-gf2' or 'lf-gf2'.")

    fixed_lambda: np.ndarray | None = None
    if lf_fixed_lambda is not None:
        if not is_lf:
            raise ValueError("lf_fixed_lambda is valid only for method='lf-gf2'.")
        fixed_lambda = np.asarray(lf_fixed_lambda, dtype=float)
        if fixed_lambda.ndim != 1 or not np.all(np.isfinite(fixed_lambda)):
            raise ValueError("lf_fixed_lambda must be a finite one-dimensional array.")
        fixed_lambda = fixed_lambda.copy()

    if lf_initial_state is not None and fixed_lambda is None:
        raise ValueError("lf_initial_state requires fixed-lambda LF-GF2 mode.")
    if lf_state_sink is not None and fixed_lambda is None:
        raise ValueError("lf_state_sink requires fixed-lambda LF-GF2 mode.")

    if not psi4_verbose:
        silence_psi4()

    opts = (
        gf2_options if isinstance(gf2_options, GF2Options) else GF2Options.from_mapping(gf2_options)
    )
    rep = reporter or Reporter(job_name="untitled")
    if psi4 is not None:
        psi4.core.set_num_threads(opts.n_threads)

    t_start = time.time()
    mp2_psi4: float | None = None
    mp2_gf2_estimate: float | None = None

    rep.section("Reference")
    raw_ref = (
        dict(reference_data)
        if reference_data is not None
        else _rhf_reference(mol, psi4_options, opts, cavity)
    )
    lf_problem_signature: str | None = None
    if is_lf:
        lf_problem_signature = _lf_problem_signature_from_reference(raw_ref)
        if lf_initial_state is not None:
            validate_lf_warm_start_problem(
                lf_initial_state,
                problem_signature=lf_problem_signature,
                nbf=int(raw_ref["nbf"]),
            )
        ref = _prepare_lf_reference(
            raw_ref,
            opts,
            fixed_lambda=fixed_lambda,
            seed_density_override=(
                lf_initial_state.density if lf_initial_state is not None else None
            ),
        )
    else:
        ref = raw_ref

    nbf = ref["nbf"]
    nocc = ref["nocc"]
    Ne = ref["Ne"]
    S = ref["S"]
    S_inv_half = ref["S_inv_half"]
    H_core = np.asarray(ref["H_core"], dtype=float)
    eps = np.asarray(ref["eps"], dtype=float)
    P = np.asarray(ref["P"], dtype=float)
    F = np.asarray(ref["F"], dtype=float)
    E_nuc = float(ref["E_nuc"])
    hf_energy = float(ref["hf_energy"])
    qed_data = ref["qed_data"]
    interaction: ElectronInteraction = ref["interaction"]
    boson_g = (
        np.asarray(ref["boson_g"], dtype=float)
        if is_lf
        else (np.asarray(qed_data.g, dtype=float) if qed_data is not None else None)
    )
    lf_context: LFOrthogonalContext | None = ref.get("lf_context")
    lf_params: LFParameters | None = ref.get("lf_params")

    if diagnostic_callback is not None:
        diagnostic_callback("reference", ref)

    component_timings = {
        "fock": 0.0,
        "sigma_ee": 0.0,
        "bosonic": 0.0,
        "lf_parameter_optimization": 0.0,
        "lf_fixed_lambda_update": 0.0,
    }

    def _build_fock(p_in: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        if is_cs:
            assert qed_data is not None
            value = _build_cs_fock_interaction(H_core, p_in, interaction, qed_data)
        else:
            value = interaction.build_fock(H_core, p_in)
        component_timings["fock"] += time.perf_counter() - started
        return value

    def _build_sigma(g_tau_in: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        value = interaction.build_self_energy(nbf, g_tau_in, opts.n_itime)
        component_timings["sigma_ee"] += time.perf_counter() - started
        return value

    if is_coupled:
        eps = np.linalg.eigvalsh(S_inv_half @ F @ S_inv_half)
    midpoint_mu = float(0.5 * (eps[nocc - 1] + eps[nocc]))
    mu = float(lf_initial_state.mu) if lf_initial_state is not None else midpoint_mu
    suggested_beta = float(1.0 / (0.1 * (eps[nocc] - eps[nocc - 1])))

    reference_label = "Model RHF energy" if reference_data is not None else "Psi4 RHF energy"
    rep.kv(reference_label, f"{hf_energy: .12f}")
    if is_lf:
        assert lf_params is not None
        if fixed_lambda is None:
            rep.kv("LF-HF static energy", f"{float(ref['lf_hf_energy']): .12f}")
            rep.kv("LF-HF converged", bool(ref["lf_hf_converged"]))
        elif lf_initial_state is not None:
            rep.kv("LF reference", "fixed lambda / correlated warm-state density")
            rep.kv(
                "Warm-start source ||lambda||",
                f"{np.linalg.norm(lf_initial_state.lambda_fixed):.8e}",
            )
            rep.kv("LF seed static energy", f"{float(ref['lf_reference_energy']): .12f}")
        else:
            rep.kv("LF reference", "fixed lambda / CS-HF density seed")
            rep.kv("LF seed static energy", f"{float(ref['lf_reference_energy']): .12f}")
        rep.kv("LF ||lambda||", f"{np.linalg.norm(lf_params.lam):.8e}")
        rep.kv("LF z", f"{lf_params.z:.8e}")
    rep.kv("Electrons Tr[P S]", f"{nelec_from_density(P, S): .8f}")
    rep.kv("Occupied orbitals", nocc)
    rep.kv("Virtual orbitals", nbf - nocc)
    rep.kv("Basis functions", nbf)
    rep.kv("Chemical potential mu", f"{mu:.10f}")
    rep.kv("Suggested beta", f"{suggested_beta:.6f}")
    rep.kv("Inverse temperature beta", opts.beta)
    rep.kv("ERI type", opts.eri_type)
    if is_cs:
        rep.kv("Method", "CS-GF2 + coupled G/D total-energy functional")
    elif is_lf and fixed_lambda is not None:
        rep.kv("Method", "projected LF-GF2 fixed-lambda correlated inner solve")
        rep.kv("Warm start", bool(lf_initial_state is not None))
    elif is_lf:
        rep.kv("Method", "projected LF-GF2 (Option A) + coupled G/D energy")
    else:
        rep.kv("Method", "electronic GF2")
    if qed_data is not None:
        rep.kv("Photon modes", len(qed_data.omega))
    rep.kv("G mix", format_mix_value(opts.g_mix))
    rep.kv("Sigma mix", format_mix_value(opts.sigma_mix))

    rep.section("Grids")
    grids = build_grids(opts)
    rep.kv("Matsubara (full)", opts.n_ifreq_full)
    rep.kv("Matsubara (sparse)", opts.n_ifreq)
    rep.kv("Imaginary time", opts.n_itime)
    rep.kv("Legendre polynomials", opts.n_poly)

    nu = bosonic_frequencies(opts.beta, opts.n_itime) if is_coupled else None
    if lf_initial_state is not None:
        assert nu is not None
        validate_lf_warm_start_grids(
            lf_initial_state,
            beta=opts.beta,
            fermionic_iomega=grids.ifreq_grid,
            imaginary_time_grid=grids.itime_grid,
            bosonic_inu=nu,
            nbf=nbf,
        )

    rep.section("Transforms")
    tx = _build_transform_objects(grids, opts)
    if diagnostic_callback is not None:
        diagnostic_callback(
            "grids",
            {
                "fermionic_sparse_iomega": grids.ifreq_grid.copy(),
                "fermionic_full_iomega": grids.full_ifreq_grid.copy(),
                "imaginary_time": grids.itime_grid.copy(),
                "imaginary_time_weights": grids.d_itime.copy(),
                "bosonic_inu": nu.copy() if nu is not None else None,
            },
        )

    rep.section("Initial Green's function")
    orth_state = gf.prepare_fock_orth(F, S_inv_half)
    Gnot_t = gf.build_g0_itime_all(orth_state, S_inv_half, mu, grids.itime_grid, opts.beta)
    Sigma_ee_t = _build_sigma(Gnot_t)

    D_tau = D_iv = D0_iv = Pi_tau = Pi_iv = None
    if is_coupled:
        assert qed_data is not None and boson_g is not None and nu is not None
        boson_started = time.perf_counter()
        D0_iv = bare_boson_iv(qed_data.omega, nu).astype(complex)
        if lf_initial_state is not None:
            D_iv = np.asarray(lf_initial_state.D_iv, dtype=complex).copy()
            Pi_iv = np.asarray(lf_initial_state.Pi_iv, dtype=complex).copy()
            D_tau = periodic_iv_to_time(D_iv, opts.beta, grids.itime_grid)
            Pi_tau = periodic_iv_to_time(Pi_iv, opts.beta, grids.itime_grid)
        else:
            D_iv = D0_iv.copy()
            D_tau = periodic_iv_to_time(D_iv, opts.beta, grids.itime_grid)
            Pi_tau = polarization_tau(Gnot_t, boson_g)
            Pi_iv = periodic_time_to_iv(Pi_tau, opts.beta, grids.itime_grid, grids.d_itime)
        Sigma_eb_t = sigma_eb_tau(Gnot_t, D_tau, boson_g)
        Sigma_t = Sigma_ee_t + Sigma_eb_t
        component_timings["bosonic"] += time.perf_counter() - boson_started
    else:
        Sigma_eb_t = np.zeros_like(Sigma_ee_t)
        Sigma_t = Sigma_ee_t

    if opts.mp2:
        mp2_gf2_estimate = gf2_energy.mp2_correlation(Gnot_t, Sigma_ee_t, grids.d_itime)
        if not is_coupled:
            mp2_psi4 = gf2_energy.psi4_mp2_correlation(hf_energy, ref["scf_wfn"])
            gf2_energy.print_mp2_comparison(mp2_psi4, mp2_gf2_estimate, rep)
        else:
            rep.kv("Electronic first-iteration E2", f"{mp2_gf2_estimate: .12f}")

    if lf_initial_state is not None:
        Sigma_if = np.asarray(lf_initial_state.Sigma_iw, dtype=complex).copy()
    else:
        coeffs = legendre_ft.legendre_coefficients(Sigma_t, tx["legendre_weights"])
        Sigma_if = legendre_ft.time_to_frequency(tx["legendre_trans"], coeffs)

    rep.section("Self-consistent iteration")
    rep.mixing_kv(opts.g_mix, opts.sigma_mix, indent=2)
    diis_per_freq = [ComplexDIIS() for _ in range(opts.n_ifreq)]
    F_diis = RealDIIS()
    F_prev = F.copy()
    G_iw_prev = np.zeros((opts.n_ifreq, nbf, nbf), dtype=np.complex128)
    if lf_initial_state is not None:
        G_iw = np.asarray(lf_initial_state.G_iw, dtype=complex).copy()
        G_iw_outer_prev: np.ndarray | None = G_iw.copy()
    else:
        G_iw = np.zeros_like(G_iw_prev)
        G_iw_outer_prev = None
    Gnot_iw = np.zeros_like(G_iw_prev)
    rho = P.copy()

    if lf_initial_state is not None:
        core_GF2 = float(lf_initial_state.core_energy)
        gf_corr = float(lf_initial_state.correlation_energy)
        gf_total = float(lf_initial_state.total_energy)
    else:
        core_GF2 = 0.0
        gf_corr = 0.0
        gf_total = 0.0
    last_breakdown: CoupledEnergyBreakdown | None = None
    converged = False
    iter_count = 0
    Sigma_t_prev: np.ndarray | None = None
    rho_outer_prev = rho.copy()
    sigma_ee_outer_prev = Sigma_ee_t.copy()
    sigma_eb_outer_prev = Sigma_eb_t.copy()
    if is_cs:
        assert qed_data is not None
        z_outer_prev = coherent_displacement(qed_data.g, rho, qed_data.omega)
    elif is_lf:
        assert lf_params is not None
        z_outer_prev = np.array([lf_params.z])
    else:
        z_outer_prev = np.empty(0)
    pi_outer_prev = Pi_iv.copy() if Pi_iv is not None else None
    d_outer_prev = D_iv.copy() if D_iv is not None else None
    final_residuals: dict[str, float] = {}

    for outer in range(opts.maxiter_gf_diis):
        iter_count += 1
        rep.outer_iter(iter_count, g_mix=opts.g_mix, sigma_mix=opts.sigma_mix)

        Fock_RMS = np.inf
        for inner in range(opts.maxiter_fock_diis):
            mu_eps = np.linalg.eigvalsh(S_inv_half @ F @ S_inv_half) if is_coupled else eps
            if opts.bisection_on:
                mu, rho, G_iw, Gnot_iw = chemical_potential.tune_bisection(
                    nbf,
                    mu,
                    grids.ifreq_grid,
                    S,
                    S_inv_half,
                    F,
                    Sigma_if,
                    Ne,
                    tx["B_beta"],
                    mu_eps,
                    nocc,
                    beta=opts.beta,
                    thresh=opts.thresh_ne,
                    maxiter=opts.maxiter_bisection,
                    reporter=rep,
                )
            if opts.newton_on:
                mu, rho, G_iw, Gnot_iw = chemical_potential.tune_newton(
                    nbf,
                    mu,
                    grids.ifreq_grid,
                    S,
                    S_inv_half,
                    F,
                    Sigma_if,
                    Ne,
                    tx["B_beta"],
                    mu_eps,
                    nocc,
                    beta=opts.beta,
                    thresh=opts.thresh_ne,
                    maxiter=opts.maxiter_newton,
                    damping=opts.newton_damping,
                    mu_guess_thresh=opts.mu_guess_thresh,
                    fit_width=opts.fit_width,
                    n_fit_pts=opts.n_fit_pts,
                    reporter=rep,
                )

            F_prev = F.copy() if inner > 0 else F_prev
            F = _build_fock(rho)
            err = F_prev - F
            F_diis.add(F, err)
            F = F_diis.extrapolate()
            Fock_RMS = float(np.sqrt(np.mean((F - F_prev) ** 2)))
            rep.inner_iter(inner + 1, Fock_RMS)
            if opts.fock_diis_on_off == "off" or Fock_RMS < opts.thresh_fock_rms:
                break

        if outer == 0:
            if lf_initial_state is not None:
                G_iw_prev = np.asarray(lf_initial_state.G_iw, dtype=complex).copy()
            else:
                G_iw_prev = Gnot_iw.copy()
        else:
            G_iw_prev = G_iw.copy()
        G_iw_diis = np.empty_like(G_iw)
        for n in range(opts.n_ifreq):
            diis_per_freq[n].add(G_iw[n], G_iw_prev[n] - G_iw[n])
            G_iw_diis[n] = diis_per_freq[n].extrapolate()
        if opts.g_mix is not None and G_iw_outer_prev is not None:
            G_iw = linear_mix(G_iw_diis, G_iw_outer_prev, opts.g_mix)
        else:
            G_iw = G_iw_diis
        G_iw_outer_prev = G_iw.copy()

        orth_state = gf.prepare_fock_orth(F, S_inv_half)
        Gnot_t = gf.build_g0_itime_all(orth_state, S_inv_half, mu, grids.itime_grid, opts.beta)
        G_tau = spline_ft.giw_to_gtau(tx["B"], G_iw, Gnot_iw, Gnot_t, opts.beta)

        Sigma_ee_new = _build_sigma(G_tau)
        if is_coupled:
            assert (
                qed_data is not None
                and boson_g is not None
                and D_tau is not None
                and D_iv is not None
                and D0_iv is not None
            )
            boson_started = time.perf_counter()
            Pi_new_tau = polarization_tau(G_tau, boson_g)
            Pi_new_iv = periodic_time_to_iv(Pi_new_tau, opts.beta, grids.itime_grid, grids.d_itime)
            D_new_iv = bosonic_dyson(D0_iv, Pi_new_iv)
            boson_mix = float(
                opts.extra.get(
                    "dx_mix",
                    opts.extra.get("Dx_mix", opts.sigma_mix if opts.sigma_mix is not None else 1.0),
                )
            )
            if not (0.0 < boson_mix <= 1.0):
                raise ValueError("dx_mix/Dx_mix must be in (0,1].")
            Pi_tau = Pi_new_tau
            Pi_iv = Pi_new_iv
            D_iv = linear_mix(D_new_iv, D_iv, boson_mix)
            D_tau = periodic_iv_to_time(D_iv, opts.beta, grids.itime_grid)
            Sigma_eb_new = sigma_eb_tau(G_tau, D_tau, boson_g)
            Sigma_new = Sigma_ee_new + Sigma_eb_new
            component_timings["bosonic"] += time.perf_counter() - boson_started
        else:
            Sigma_eb_new = np.zeros_like(Sigma_ee_new)
            Sigma_new = Sigma_ee_new

        if opts.sigma_mix is not None and Sigma_t_prev is not None:
            Sigma_t = linear_mix(Sigma_new, Sigma_t_prev, opts.sigma_mix)
        else:
            Sigma_t = Sigma_new
        Sigma_t_prev = Sigma_t.copy()
        sigma_total_coeff = legendre_ft.legendre_coefficients(Sigma_t, tx["legendre_weights"])
        sigma_total_iw_updated = legendre_ft.time_to_frequency(
            tx["legendre_trans"], sigma_total_coeff
        )
        sigma_iw_update = float(np.max(np.abs(sigma_total_iw_updated - Sigma_if)))

        core_old, corr_old, total_old = core_GF2, gf_corr, gf_total

        if is_cs:
            assert qed_data is not None
            core_GF2, z_current = _cs_static_energy(
                h_qed=H_core,
                density=rho,
                interaction=interaction,
                g=qed_data.g,
                omega=qed_data.omega,
                e_nuc=E_nuc,
            )
        elif is_lf:
            assert qed_data is not None and lf_params is not None
            core_GF2 = _lf_static_energy_from_active_state(
                h_lf=H_core,
                density=rho,
                interaction=interaction,
                omega=float(qed_data.omega[0]),
                params=lf_params,
                e_nuc=E_nuc,
            )
            z_current = np.array([lf_params.z])
        else:
            physical_fock = interaction.build_fock(H_core, rho)
            core_GF2 = fock_mod.core_energy(physical_fock, H_core, rho, E_nuc)
            z_current = np.empty(0)

        if is_coupled:
            assert (
                qed_data is not None
                and Pi_tau is not None
                and D_tau is not None
                and D_iv is not None
                and D0_iv is not None
            )
            last_breakdown = coupled_energy_from_state(
                static_energy=core_GF2,
                G_tau=G_tau,
                Sigma_ee_tau=Sigma_ee_new,
                Sigma_eb_tau=Sigma_eb_new,
                Pi_tau=Pi_tau,
                D_tau=D_tau,
                D_iv=D_iv,
                D0_iv=D0_iv,
                d_itime=grids.d_itime,
                omega=qed_data.omega,
                beta=opts.beta,
            )
            gf_corr = last_breakdown.coupled_correlation
            gf_total = last_breakdown.coupled_total
        else:
            gf_corr = gf2_energy.gf2_correlation(G_tau, Sigma_ee_new, grids.d_itime)
            gf_total = core_GF2 + gf_corr

        identity = np.broadcast_to(np.eye(nbf), G_iw.shape)
        electronic_inverse = (
            (mu + 1.0j * grids.ifreq_grid)[:, None, None] * S[None, :, :] - F[None, :, :] - Sigma_if
        )
        electronic_dyson = (
            np.einsum("nij,njk->nik", electronic_inverse, G_iw, optimize=True) - identity
        )

        final_residuals = {
            "energy": float(abs(gf_total - total_old)),
            "particle_number": float(abs(nelec_from_density(rho, S) - Ne)),
            "density": float(np.linalg.norm(rho - rho_outer_prev)),
            "fock_rms": float(Fock_RMS),
            "sigma_ee": float(np.linalg.norm(Sigma_ee_new - sigma_ee_outer_prev)),
            "sigma_eb": float(np.linalg.norm(Sigma_eb_new - sigma_eb_outer_prev)),
            "sigma_iw": sigma_iw_update,
            "z": float(np.linalg.norm(z_current - z_outer_prev)),
            "electronic_dyson": float(np.linalg.norm(electronic_dyson)),
        }
        if is_coupled:
            assert (
                Pi_iv is not None
                and D_iv is not None
                and D0_iv is not None
                and pi_outer_prev is not None
                and d_outer_prev is not None
                and last_breakdown is not None
            )
            bosonic_dyson_residual = np.linalg.inv(D_iv) - np.linalg.inv(D0_iv) + Pi_iv
            final_residuals.update(
                {
                    "pi": float(np.linalg.norm(Pi_iv - pi_outer_prev)),
                    "d": float(np.linalg.norm(D_iv - d_outer_prev)),
                    "bosonic_dyson": float(np.linalg.norm(bosonic_dyson_residual)),
                    "sigma_g_pi_d": float(last_breakdown.sigma_g_pi_d_residual),
                }
            )

        lf_dlam = 0.0
        lf_dz = 0.0
        lf_stationarity = 0.0
        lf_next: LFParameters | None = None
        lf_effective_next = None
        if is_lf:
            assert lf_context is not None and lf_params is not None and qed_data is not None
            lf_started = time.perf_counter()
            if fixed_lambda is None:
                opt = optimize_lf_parameters_fixed_density(lf_context, rho, lf_params, opts.extra)
                lf_next = damp_lf_parameter_update(lf_params, opt.params, opts.extra, prefix="lf")
                component_timings["lf_parameter_optimization"] += time.perf_counter() - lf_started
                update_label = "LF update"
            else:
                fixed_z = lf_stationary_z(lf_context, rho, fixed_lambda)
                lf_next = LFParameters(fixed_lambda, fixed_z)
                component_timings["lf_fixed_lambda_update"] += time.perf_counter() - lf_started
                update_label = "LF fixed-lambda update"

            lf_dlam = float(np.linalg.norm(lf_next.lam - lf_params.lam))
            lf_dz = float(abs(lf_next.z - lf_params.z))
            stationary_z = lf_stationary_z(lf_context, rho, lf_next.lam)
            lf_stationarity = float(abs(lf_next.z - stationary_z))
            final_residuals.update(
                {
                    "lambda": lf_dlam,
                    "lf_parameter_z": lf_dz,
                    "lf_z_stationarity": lf_stationarity,
                }
            )
            lf_effective_next = build_lf_effective(lf_context, lf_next)
            rep.detail(
                f"{update_label}: "
                f"dlam={lf_dlam:.3e} dz={lf_dz:.3e} "
                f"z-stationarity={lf_stationarity:.3e}"
            )

        if diagnostic_callback is not None:
            payload: dict[str, Any] = {
                "iteration": iter_count,
                "density": rho.copy(),
                "fock": F.copy(),
                "chemical_potential": float(mu),
                "G_iw": G_iw.copy(),
                "G_tau": G_tau.copy(),
                "Sigma_ee_tau": Sigma_ee_new.copy(),
                "Sigma_eb_tau": Sigma_eb_new.copy(),
                "Sigma_total_tau": Sigma_t.copy(),
                "core_energy": float(core_GF2),
                "correlation_energy": float(gf_corr),
                "total_energy": float(gf_total),
                "particle_number": float(nelec_from_density(rho, S)),
                "residuals": dict(final_residuals),
            }
            if is_coupled and last_breakdown is not None:
                payload.update(
                    {
                        "Pi_tau": Pi_tau.copy(),
                        "Pi_iv": Pi_iv.copy(),
                        "D0_iv": D0_iv.copy(),
                        "D_iv": D_iv.copy(),
                        "D_tau": D_tau.copy(),
                        "energy_breakdown": last_breakdown,
                        "boson_vertex": boson_g.copy(),
                    }
                )
            if is_lf and lf_params is not None:
                payload.update(
                    {
                        "lf_lambda": lf_params.lam.copy(),
                        "lf_z": float(lf_params.z),
                        "lf_parameter_mode": (
                            "fixed" if fixed_lambda is not None else "static-density-optimized"
                        ),
                    }
                )
            diagnostic_callback(f"iteration_{iter_count}", payload)

        rho_outer_prev = rho.copy()
        sigma_ee_outer_prev = Sigma_ee_new.copy()
        sigma_eb_outer_prev = Sigma_eb_new.copy()
        z_outer_prev = z_current.copy()
        if Pi_iv is not None:
            pi_outer_prev = Pi_iv.copy()
        if D_iv is not None:
            d_outer_prev = D_iv.copy()

        rep.energy_row("Static/core", core_GF2, abs(core_GF2 - core_old))
        rep.energy_row(
            "Coupled correlation" if is_coupled else "Correlation",
            gf_corr,
            abs(gf_corr - corr_old),
        )
        rep.energy_row("Total GF2", gf_total, abs(gf_total - total_old))
        if last_breakdown is not None:
            rep.detail(
                "energy components: "
                f"Cee={last_breakdown.ee_correlation:.9f} "
                f"Ceb={last_breakdown.eb_fermionic_trace:.9f} "
                f"Eph={last_breakdown.photon_fluctuation_energy:.9f} "
                f"SigmaG-PiD={last_breakdown.sigma_g_pi_d_residual:.3e}"
            )

        Sigma_if = sigma_total_iw_updated
        min_outer = int(opts.extra.get("min_outer_iterations", 1))
        sigma_iw_threshold = float(opts.extra.get("thresh_sigma_iw", 1.0e-6))
        bosonic_threshold = float(opts.extra.get("thresh_bosonic", 1.0e-6))
        fixed_point_ok = outer_fixed_point_converged(
            final_residuals,
            energy_threshold=opts.thresh_energy,
            particle_threshold=opts.thresh_ne,
            fock_threshold=opts.thresh_fock_rms,
            sigma_iw_threshold=sigma_iw_threshold,
            coupled=is_coupled,
            bosonic_threshold=bosonic_threshold,
        )
        if is_lf:
            lf_thresh = float(opts.extra.get("lf_param_thresh", 1.0e-6))
            fixed_point_ok = bool(
                fixed_point_ok
                and lf_dlam < lf_thresh
                and lf_dz < lf_thresh
                and lf_stationarity < lf_thresh
            )

        if fixed_point_ok and iter_count >= min_outer:
            converged = True
            rep.ok(
                "LF-GF2 fixed-lambda coupled fixed point converged"
                if is_lf and fixed_lambda is not None
                else (
                    "LF-GF2 coupled fixed point converged"
                    if is_lf
                    else (
                        "CS-GF2 coupled fixed point converged"
                        if is_cs
                        else "GF2 fixed point converged"
                    )
                )
            )
            break

        if is_lf:
            assert lf_next is not None and lf_effective_next is not None and qed_data is not None
            lf_params = lf_next
            H_core = lf_effective_next.h_ao
            interaction = ElectronInteraction.exact(
                lf_effective_next.eri_ao,
                "LF projected / fixed symmetric-orthogonal basis",
                configured_qed_channels=1,
                active_qed_channels=1,
                reduced_symmetry=True,
            )
            boson_g = lf_effective_next.g_lf_ao[None, :, :]
            F = interaction.build_fock(H_core, rho)
            F_prev = F.copy()
            # Never reuse Fock-DIIS history across a changed LF Hamiltonian.
            F_diis = RealDIIS()

        if iter_count == opts.maxiter_gf_diis:
            rep.warn("maximum outer DIIS iterations reached")
            break

    total_time = time.time() - t_start
    rep.kv("Wall time", f"{total_time:.2f} s")
    if not converged:
        rep.warn("GF2 SCF did not reach the requested fixed-point convergence criteria")

    metadata = interaction.metadata()
    metadata["method"] = method_key
    if is_lf and lf_params is not None:
        metadata.update(
            {
                "lf_projection": "zero-boson Option A",
                "lf_lambda_norm": float(np.linalg.norm(lf_params.lam)),
                "lf_z": float(lf_params.z),
                "lf_basis": "symmetric orthogonalized",
                "lf_reference_mode": str(ref.get("lf_reference_mode", "unknown")),
                "lf_parameter_mode": (
                    "fixed-correlated-inner"
                    if fixed_lambda is not None
                    else "static-density-optimized"
                ),
                "lf_warm_start_used": bool(lf_initial_state is not None),
                "lf_outer_iterations": int(iter_count),
            }
        )
        if fixed_lambda is not None:
            metadata["lf_fixed_lambda"] = fixed_lambda.tolist()
        if lf_initial_state is not None:
            metadata["lf_warm_start_source_lambda_norm"] = float(
                np.linalg.norm(lf_initial_state.lambda_fixed)
            )
    if last_breakdown is not None:
        metadata["coupled_energy_components"] = {
            "C_ee": last_breakdown.ee_correlation,
            "C_eb": last_breakdown.eb_fermionic_trace,
            "E_eb": last_breakdown.eb_interaction_sigma_g,
            "E_ph": last_breakdown.photon_fluctuation_energy,
            "photon_coordinate": last_breakdown.photon_coordinate_term,
            "sigma_g_pi_d_residual": last_breakdown.sigma_g_pi_d_residual,
        }

    summary = RunSummary(
        job_name=rep.job_name,
        core_gf2=core_GF2,
        corr_gf2=gf_corr,
        converged=converged,
        wall_time_s=total_time,
        hf_energy=hf_energy,
        mp2_psi4=mp2_psi4,
        mp2_gf2_estimate=mp2_gf2_estimate,
        g_mix=opts.g_mix,
        sigma_mix=opts.sigma_mix,
        interaction_metadata=metadata,
        component_timings_s=component_timings,
        convergence_diagnostics=final_residuals,
    )

    if lf_state_sink is not None:
        assert (
            fixed_lambda is not None
            and lf_problem_signature is not None
            and lf_params is not None
            and nu is not None
            and D_iv is not None
            and Pi_iv is not None
        )
        state = LFGF2FixedState(
            problem_signature=lf_problem_signature,
            lambda_fixed=fixed_lambda,
            xi=float(lf_params.z),
            beta=float(opts.beta),
            fermionic_iomega=grids.ifreq_grid,
            imaginary_time_grid=grids.itime_grid,
            bosonic_inu=nu,
            density=rho,
            fock=F,
            mu=float(mu),
            G_iw=G_iw,
            Sigma_iw=Sigma_if,
            D_iv=D_iv,
            Pi_iv=Pi_iv,
            core_energy=float(core_GF2),
            correlation_energy=float(gf_corr),
            converged=bool(converged),
            outer_iterations=int(iter_count),
        )
        lf_state_sink.clear()
        lf_state_sink["state"] = state

    rep.write_summary_block(summary)
    return core_GF2, gf_corr, summary


def run_lf_gf2_fixed_lambda(
    mol: psi4.core.Molecule,
    psi4_options: Mapping[str, Any],
    gf2_options: Mapping[str, Any] | GF2Options,
    *,
    lambda_fixed: np.ndarray,
    cavity: CavityConfiguration,
    reporter: Reporter | None = None,
    psi4_verbose: bool = False,
    diagnostic_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    initial_state: LFGF2FixedState | None = None,
) -> tuple[float, float, RunSummary]:
    """Compatibility wrapper for a fixed-lambda LF-GF2 energy calculation."""
    return run_gf2(
        mol,
        psi4_options,
        gf2_options,
        reporter=reporter,
        psi4_verbose=psi4_verbose,
        method="lf-gf2",
        cavity=cavity,
        diagnostic_callback=diagnostic_callback,
        lf_fixed_lambda=np.asarray(lambda_fixed, dtype=float),
        lf_initial_state=initial_state,
    )


def solve_lf_gf2_fixed_lambda(
    mol: psi4.core.Molecule,
    psi4_options: Mapping[str, Any],
    gf2_options: Mapping[str, Any] | GF2Options,
    *,
    lambda_fixed: np.ndarray,
    cavity: CavityConfiguration,
    reporter: Reporter | None = None,
    psi4_verbose: bool = False,
    diagnostic_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    initial_state: LFGF2FixedState | None = None,
) -> LFGF2FixedResult:
    "Solve fixed-lambda LF-GF2 and return a reusable converged-state snapshot."
    sink: dict[str, LFGF2FixedState] = {}
    core, corr, summary = run_gf2(
        mol,
        psi4_options,
        gf2_options,
        reporter=reporter,
        psi4_verbose=psi4_verbose,
        method="lf-gf2",
        cavity=cavity,
        diagnostic_callback=diagnostic_callback,
        lf_fixed_lambda=np.asarray(lambda_fixed, dtype=float),
        lf_initial_state=initial_state,
        lf_state_sink=sink,
    )
    state = sink.get("state")
    if state is None:
        raise RuntimeError("Fixed-lambda LF-GF2 solve did not return a state snapshot.")
    return LFGF2FixedResult(
        core_energy=float(core),
        correlation_energy=float(corr),
        summary=summary,
        state=state,
    )
