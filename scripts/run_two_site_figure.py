#!/usr/bin/env python3
# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Generate ED, CS-HF, CS-MP2, and CS-GF2 data for the model figure."

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np

from gf2.model_benchmark import (
    DimerParameters,
    csmp2_energy,
    default_gf2_options,
    exact_ground_state,
    exact_thermal,
    run_csgf2,
    solve_cshf,
)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _point_label(parameters: DimerParameters) -> str:
    def clean(value: float) -> str:
        return f"{value:.8g}".replace("-", "m").replace(".", "p")

    return f"U{clean(parameters.U)}_d{clean(parameters.delta)}_g{clean(parameters.g)}"


def _flatten_residuals(row: dict[str, Any], residuals: Mapping[str, float]) -> None:
    for name in (
        "energy",
        "particle_number",
        "fock_rms",
        "sigma_iw",
        "electronic_dyson",
        "pi",
        "d",
        "bosonic_dyson",
        "sigma_g_pi_d",
    ):
        row[f"gf2_residual_{name}"] = float(residuals.get(name, float("nan")))


def calculate_point(
    args: argparse.Namespace, *, U: float, delta: float, g: float
) -> dict[str, Any]:
    parameters = DimerParameters(
        t=args.t,
        U=U,
        omega=args.omega,
        g=g,
        delta=delta,
        nph=args.nph,
    )
    cutoff_parameters = DimerParameters(
        t=args.t,
        U=U,
        omega=args.omega,
        g=g,
        delta=delta,
        nph=args.nph - args.nph_check_step,
    )
    label = _point_label(parameters)

    started = time.perf_counter()
    exact = exact_ground_state(parameters)
    thermal = exact_thermal(parameters, args.beta)
    exact_low = exact_ground_state(cutoff_parameters)
    thermal_low = exact_thermal(cutoff_parameters, args.beta)
    exact_time = time.perf_counter() - started

    started = time.perf_counter()
    cshf = solve_cshf(
        parameters,
        starts=args.hf_starts,
        tolerance=args.hf_tolerance,
        max_iterations=args.hf_max_iterations,
        damping=args.hf_damping,
    )
    cshf_time = time.perf_counter() - started

    started = time.perf_counter()
    csmp2 = csmp2_energy(parameters, cshf)
    csmp2_time = time.perf_counter() - started

    row: dict[str, Any] = {
        **asdict(parameters),
        "beta": args.beta,
        "E_exact_ground": exact.energy,
        "E_exact_thermal": thermal.energy,
        "E_cshf": cshf.energy,
        "E_csmp2": csmp2.energy,
        "E_csgf2": float("nan"),
        "error_cshf_ground": cshf.energy - exact.energy,
        "error_csmp2_ground": csmp2.energy - exact.energy,
        "error_csgf2_ground": float("nan"),
        "error_csgf2_thermal": float("nan"),
        "thermal_exact_shift": thermal.energy - exact.energy,
        "ed_cutoff_change_ground": exact.energy - exact_low.energy,
        "ed_cutoff_change_thermal": thermal.energy - thermal_low.energy,
        "ed_photon_top_probability": exact.photon_top_probability,
        "exact_one_body": exact.one_body,
        "exact_interaction": exact.interaction,
        "exact_electron_boson": exact.electron_boson,
        "exact_photon": exact.photon,
        "thermal_exact_electron_number": thermal.electron_number,
        "thermal_exact_chemical_potential": thermal.chemical_potential,
        "cshf_xi": cshf.xi,
        "cshf_start": cshf.start,
        "cshf_converged_starts": cshf.converged_starts,
        "cshf_iterations": cshf.iterations,
        "cshf_density_residual": cshf.residual,
        "cshf_stationary_residual": cshf.stationary_residual,
        "csmp2_electronic_correlation": csmp2.electronic_correlation,
        "csmp2_electron_boson_correlation": csmp2.electron_boson_correlation,
        "gf2_converged": False,
        "gf2_iterations": 0,
        "time_exact_s": exact_time,
        "time_cshf_s": cshf_time,
        "time_csmp2_s": csmp2_time,
        "time_csgf2_s": float("nan"),
        "status": "PRE_GF2",
    }

    if args.skip_gf2:
        row["status"] = "SKIPPED_GF2"
        return row

    gf2_options = default_gf2_options(
        beta=args.beta,
        ntime=args.ntime,
        nfreq=args.nfreq,
        nfull=args.nfull,
        npoly=args.npoly,
        max_iterations=args.gf2_max_iterations,
    )
    gf2_options.update(
        {
            "g_mix": args.g_mix,
            "sigma_mix": args.sigma_mix,
            "dx_mix": args.dx_mix,
            "thresh_energy": args.gf2_energy_tolerance,
            "thresh_ne": args.gf2_number_tolerance,
            "thresh_fock_rms": args.gf2_fock_tolerance,
            "thresh_sigma_iw": args.gf2_self_energy_tolerance,
            "thresh_bosonic": args.gf2_boson_tolerance,
        }
    )
    gf2 = run_csgf2(
        parameters,
        cshf,
        gf2_options,
        job_name=label,
        save_final_state=args.save_propagators,
    )
    row.update(
        {
            "E_csgf2": gf2.energy,
            "error_csgf2_ground": gf2.energy - exact.energy,
            "error_csgf2_thermal": gf2.energy - thermal.energy,
            "gf2_converged": gf2.converged,
            "gf2_iterations": gf2.iterations,
            "time_csgf2_s": gf2.wall_time_s,
            "gf2_static_energy": gf2.static_energy,
            "gf2_correlation_energy": gf2.correlation_energy,
            "gf2_C_ee": gf2.energy_components.get("C_ee", float("nan")),
            "gf2_C_eb": gf2.energy_components.get("C_eb", float("nan")),
            "gf2_E_eb": gf2.energy_components.get("E_eb", float("nan")),
            "gf2_E_ph": gf2.energy_components.get("E_ph", float("nan")),
            "status": "PASS" if gf2.converged else "GF2_NOT_CONVERGED",
        }
    )
    _flatten_residuals(row, gf2.residuals)

    logs = args.output / "logs"
    iterations = args.output / "iterations"
    logs.mkdir(parents=True, exist_ok=True)
    iterations.mkdir(parents=True, exist_ok=True)
    (logs / f"{label}.log").write_text("\n".join(gf2.log_lines) + "\n")
    _write_csv(iterations / f"{label}.csv", gf2.iteration_history)
    if args.save_propagators:
        arrays = {
            name: value for name, value in gf2.final_state.items() if isinstance(value, np.ndarray)
        }
        if arrays:
            propagators = args.output / "propagators"
            propagators.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(propagators / f"{label}.npz", **arrays)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--U-values", type=float, nargs="+", default=[0.0, 2.0])
    parser.add_argument("--delta-values", type=float, nargs="+", default=[0.0])
    parser.add_argument("--g-values", type=float, nargs="+")
    parser.add_argument("--g-min", type=float, default=0.0)
    parser.add_argument("--g-max", type=float, default=0.5)
    parser.add_argument("--g-count", type=int, default=11)
    parser.add_argument("--nph", type=int, default=24)
    parser.add_argument("--nph-check-step", type=int, default=8)
    parser.add_argument("--beta", type=float, default=200.0)
    parser.add_argument("--ntime", type=int, default=300)
    parser.add_argument("--nfreq", type=int, default=300)
    parser.add_argument("--nfull", type=int, default=10000)
    parser.add_argument("--npoly", type=int, default=50)
    parser.add_argument("--gf2-max-iterations", type=int, default=200)
    parser.add_argument("--g-mix", type=float, default=0.5)
    parser.add_argument("--sigma-mix", type=float, default=0.5)
    parser.add_argument("--dx-mix", type=float, default=0.5)
    parser.add_argument("--gf2-energy-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--gf2-number-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--gf2-fock-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--gf2-self-energy-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--gf2-boson-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--hf-starts", nargs="+", default=["bonding", "site1", "site2"])
    parser.add_argument("--hf-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--hf-max-iterations", type=int, default=500)
    parser.add_argument("--hf-damping", type=float, default=0.5)
    parser.add_argument("--ed-cutoff-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--thermal-shift-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--skip-gf2", action="store_true")
    parser.add_argument("--save-propagators", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("model_figure_results"))
    args = parser.parse_args()

    if args.g_values is None:
        if args.g_count < 1:
            parser.error("g-count must be positive.")
        args.g_values = np.linspace(args.g_min, args.g_max, args.g_count).tolist()
    if (
        args.t <= 0.0
        or args.omega <= 0.0
        or args.beta <= 0.0
        or args.nph_check_step < 1
        or args.nph - args.nph_check_step < 2
        or min(args.U_values) < 0.0
        or min(args.g_values) < 0.0
        or args.ntime < 20
        or args.nfreq < 3
        or args.nfull <= args.nfreq
        or args.npoly < 2
        or not 0.0 < args.hf_damping <= 1.0
    ):
        parser.error("Invalid physical parameters, cutoffs, or GF2 grid dimensions.")
    return args


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    started = time.perf_counter()
    for U in args.U_values:
        for delta in args.delta_values:
            for g in args.g_values:
                point = f"U={U:g}, delta={delta:g}, g={g:g}"
                try:
                    row = calculate_point(args, U=U, delta=delta, g=g)
                    rows.append(row)
                    _write_csv(args.output / "benchmark.csv", rows)
                    cutoff_error = max(
                        abs(row["ed_cutoff_change_ground"]),
                        abs(row["ed_cutoff_change_thermal"]),
                    )
                    if cutoff_error > args.ed_cutoff_tolerance:
                        failures.append(
                            f"{point}: ED cutoff change {cutoff_error:.3e} exceeds tolerance"
                        )
                    if abs(row["thermal_exact_shift"]) > args.thermal_shift_tolerance:
                        failures.append(
                            f"{point}: exact thermal shift {row['thermal_exact_shift']:.3e} "
                            "exceeds tolerance"
                        )
                    if not args.skip_gf2 and not row["gf2_converged"]:
                        failures.append(f"{point}: CS-GF2 did not converge")
                    print(
                        f"{point}: ED={row['E_exact_ground']:.10f} "
                        f"CS-HF={row['E_cshf']:.10f} CS-MP2={row['E_csmp2']:.10f} "
                        f"CS-GF2={row['E_csgf2']:.10f} [{row['status']}]",
                        flush=True,
                    )
                except Exception as error:
                    failures.append(f"{point}: {type(error).__name__}: {error}")
                    if not args.continue_on_error:
                        raise
                    print(f"{point}: ERROR {type(error).__name__}: {error}", file=sys.stderr)

    metadata = {
        "status": "PASS" if not failures else "FAIL",
        "model": (
            "-t sum_sigma(c1^dagger c2+h.c.) + delta/2(n1-n2) + "
            "U sum_i n_i_up n_i_down + omega b^dagger b + g(n1-n2)(b+b^dagger)"
        ),
        "energy_units": "t",
        "primary_exact_reference": "canonical N=2 ground-state exact diagonalization",
        "finite_temperature_check": "grand-canonical ED at target <N>=2 and the GF2 beta",
        "csmp2_definition": (
            "CS-HF + zero-photon electronic MP2 + one-photon electron-boson second order"
        ),
        "parameters": vars(args),
        "points_completed": len(rows),
        "failures": failures,
        "wall_time_s": time.perf_counter() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=_json_value) + "\n"
    )
    print(json.dumps({"status": metadata["status"], "failures": failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
