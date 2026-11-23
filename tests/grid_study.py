# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Three-level CS-GF2 grid-convergence study for the release test."""

from __future__ import annotations

import numpy as np
import psi4

from gf2 import CavityConfiguration, PhotonMode, run_gf2
from gf2 import greens_function as green
from gf2 import spline_ft
from gf2.energy import gf2_correlation, gf2_correlation_frequency_positive
from gf2.grids import build_grids
from gf2.options import GF2Options
from gf2.reporting import Reporter


BASE_OPTIONS = {
    "beta": 20.0,
    "n_ifreq": 30,
    "mid_pt": 10,
    "n_ifreq_full": 200,
    "n_itime": 30,
    "n_poly": 20,
    "mp2": False,
    "maxiter_gf_diis": 30,
    "maxiter_fock_diis": 8,
    "n_fit_pts": 1000,
    "n_threads": 1,
    "cs_hf_mix": 0.2,
}

ONE_MODE = (PhotonMode(0.10, 0.01, (0.0, 0.0, 1.0)),)

LEVELS = (
    ("coarse", 30, 200, 30, 20),
    ("medium", 60, 800, 60, 30),
    ("fine", 120, 3000, 120, 60),
)


def h2() -> psi4.core.Molecule:
    """Centered H2 at 1.3983973322 bohr."""
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    return psi4.geometry(
        """0 1
H 0 0 -0.6991986661
H 0 0  0.6991986661
units bohr
no_reorient
no_com
symmetry c1"""
    )


def _frequency_energy(final: dict, reference: dict, options: dict) -> float:
    opts = GF2Options.from_mapping(options)
    grids = build_grids(opts)
    interpolation = spline_ft.spline_v_matrix(grids.full_ifreq_grid, grids.ifreq_grid)
    nbf = reference["nbf"]
    sparse_zero = np.zeros_like(final["Sigma_total_iw_current"])
    full_zero = np.zeros((len(grids.full_ifreq_grid), nbf, nbf), dtype=complex)
    g0_sparse = green.build_g_g0_ifreq(
        nbf,
        final["mu"],
        grids.ifreq_grid,
        reference["S"],
        final["fock"],
        sparse_zero,
    )[1]
    g0_full = green.build_g_g0_ifreq(
        nbf,
        final["mu"],
        grids.full_ifreq_grid,
        reference["S"],
        final["fock"],
        full_zero,
    )[1]
    g_full = g0_full + np.einsum(
        "fi,iab->fab",
        interpolation,
        final["G_iw"] - g0_sparse,
        optimize=True,
    )
    sigma_full = np.einsum(
        "fi,iab->fab",
        interpolation,
        final["Sigma_total_iw_current"],
        optimize=True,
    )
    return gf2_correlation_frequency_positive(g_full, sigma_full, opts.beta)


def run_convergence_study() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for label, n_ifreq, n_full, n_itime, n_poly in LEVELS:
        options = {
            **BASE_OPTIONS,
            "eri_type": "std",
            "n_ifreq": n_ifreq,
            "mid_pt": max(10, n_ifreq // 3),
            "n_ifreq_full": n_full,
            "n_itime": n_itime,
            "n_poly": n_poly,
        }
        captured: dict[str, dict] = {}
        core, correlation, summary = run_gf2(
            h2(),
            {"basis": "sto-3g", "scf_type": "pk"},
            options,
            method="cs-gf2",
            cavity=CavityConfiguration(ONE_MODE),
            reporter=Reporter(echo=False),
            diagnostic_callback=lambda stage, data: captured.update({stage: data}),
        )
        if not summary.converged:
            raise RuntimeError(f"{label} grid did not converge: {summary.convergence_diagnostics}")
        final_key = max(
            (key for key in captured if key.startswith("iteration_")),
            key=lambda key: int(key.split("_")[1]),
        )
        final = captured[final_key]
        frequency = _frequency_energy(final, captured["reference"], options)
        grids = build_grids(GF2Options.from_mapping(options))
        fermionic = gf2_correlation(final["G_tau"], final["Sigma_total_tau"], grids.d_itime)
        rows.append(
            {
                "level": label,
                "total": core + correlation,
                "correlation": correlation,
                "particle_error": summary.convergence_diagnostics["particle_number"],
                "time_frequency_error": abs(frequency - fermionic),
                "wall_time_s": summary.wall_time_s,
            }
        )
    return rows
