# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Three-level CS-GF2 grid-convergence study."

from __future__ import annotations

import numpy as np

from _common import BASE_OPTIONS, ONE_MODE, h2
from gf2 import CavityConfiguration, run_gf2
from gf2 import greens_function as green
from gf2 import spline_ft
from gf2.energy import gf2_correlation_frequency_positive
from gf2.grids import build_grids
from gf2.options import GF2Options
from gf2.reporting import Reporter


LEVELS = (
    ("coarse", 30, 200, 30, 20),
    ("medium", 60, 800, 60, 30),
    ("fine", 120, 3000, 120, 60),
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
        rows.append(
            {
                "level": label,
                "total": core + correlation,
                "correlation": correlation,
                "particle_error": summary.convergence_diagnostics["particle_number"],
                "time_frequency_error": abs(frequency - correlation),
                "wall_time_s": summary.wall_time_s,
            }
        )
    return rows


if __name__ == "__main__":
    print("level       total/Ha          corr/Ha       particle error  |Etime-Efreq|/Ha  wall/s")
    for row in run_convergence_study():
        print(
            f"{row['level']:<8} {row['total']: .12f} {row['correlation']: .12f} "
            f"{row['particle_error']:14.3e} {row['time_frequency_error']:16.3e} "
            f"{row['wall_time_s']:7.3f}"
        )
