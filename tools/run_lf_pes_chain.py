#!/usr/bin/env python
# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Run one sequential correlated LF-GF2 diatomic PES continuation chain."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gf2 import driver_theory
from gf2.input_parser import parse_input_file
from gf2.lf_correlated_optimizer import (
    LFCorrelatedEvaluationRecord,
    LFCorrelatedObjectiveEvaluation,
    LFCorrelatedOptimizerOptions,
    optimize_correlated_lambda,
)
from gf2.lf_minimum_confirmation import (
    LFMinimumConfirmationOptions,
    confirm_correlated_minimum,
)
from gf2.lf_state import LFGF2FixedState, load_lf_gf2_state, save_lf_gf2_state
from gf2.pes_workflow import distance_tag, write_diatomic_input
from gf2.reporting import Reporter


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _append_master(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "distance",
        "total_energy",
        "core_energy",
        "correlation_energy",
        "xi",
        "lambda_norm",
        "lambda",
        "optimizer_iterations",
        "optimizer_evaluations",
        "confirmation_directions",
        "confirmed",
        "wall_time_s",
        "state",
        "input",
    ]
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    distance = f"{float(row['distance']):.8f}"
    rows = [item for item in rows if item.get("distance") != distance]
    normalized = dict(row)
    normalized["distance"] = distance
    rows.append(normalized)
    rows.sort(key=lambda item: float(item["distance"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _evaluation_from_state(
    state: LFGF2FixedState, wall_time_s: float = 0.0
) -> LFCorrelatedObjectiveEvaluation:
    return LFCorrelatedObjectiveEvaluation(
        lambda_value=state.lambda_fixed,
        total_energy=state.total_energy,
        converged=state.converged,
        state=state,
        core_energy=state.core_energy,
        correlation_energy=state.correlation_energy,
        outer_iterations=state.outer_iterations,
        wall_time_s=wall_time_s,
    )


def _serialize_optimizer_history(result: Any) -> list[dict[str, Any]]:
    rows = []
    for item in result.history:
        payload = asdict(item)
        payload["direction"] = np.asarray(payload["direction"], dtype=float).tolist()
        rows.append(payload)
    return rows


def _serialize_confirmation_history(result: Any) -> list[dict[str, Any]]:
    rows = []
    for item in result.history:
        payload = asdict(item)
        payload["direction"] = np.asarray(payload["direction"], dtype=float).tolist()
        rows.append(payload)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential diatomic LF-GF2 PES continuation. Across geometries only "
            "the optimized LF vector is transferred; Green's functions and self-"
            "energies are always rebuilt for the new molecular problem."
        )
    )
    parser.add_argument("template", help="LF-GF2 diatomic input template.")
    parser.add_argument(
        "--reference-state",
        required=True,
        help="Confirmed LF-GF2 state used only for its lambda vector.",
    )
    parser.add_argument("--distances", nargs="+", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--energy-tol", type=float, default=1.0e-6)
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--probe-radius", type=float, default=4.0e-3)
    parser.add_argument("--min-probe-radius", type=float, default=1.0e-3)
    parser.add_argument("--trust-radius", type=float, default=4.0e-3)
    parser.add_argument("--min-trust-radius", type=float, default=1.0e-3)
    parser.add_argument("--max-trust-radius", type=float, default=2.0e-2)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--confirm-radius", type=float, default=1.0e-3)
    parser.add_argument("--required-misses", type=int, default=10)
    parser.add_argument("--max-directions", type=int, default=30)
    parser.add_argument("--confirm-seed", type=int, default=161803)
    parser.add_argument("--component-bound", type=float, default=None)
    parser.add_argument("--psi4-verbose", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute points even when a confirmed checkpoint exists.",
    )
    args = parser.parse_args()

    template = Path(args.template).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    master_csv = output_dir / "lf_pes.csv"

    reference_state = load_lf_gf2_state(Path(args.reference_state).expanduser().resolve())
    if not reference_state.converged:
        raise ValueError("--reference-state must be converged")
    lambda_seed = reference_state.lambda_fixed.copy()

    for distance in args.distances:
        tag = distance_tag(distance)
        point_dir = output_dir / f"R_{tag}"
        point_dir.mkdir(parents=True, exist_ok=True)
        input_path = point_dir / "lf.inp"
        final_state_path = point_dir / "final_state.npz"
        point_json = point_dir / "point.json"

        if point_json.exists() and final_state_path.exists() and not args.no_resume:
            payload = json.loads(point_json.read_text(encoding="utf-8"))
            if bool(payload.get("confirmed", False)):
                state = load_lf_gf2_state(final_state_path)
                lambda_seed = state.lambda_fixed.copy()
                print(f"R={distance:.5f}: confirmed checkpoint exists; skipping.", flush=True)
                continue

        write_diatomic_input(
            template,
            input_path,
            distance,
            method="lf-gf2",
            job_name=f"LF-GF2 PES R={distance:.5f}",
        )
        job, mol, psi4_options, opts = parse_input_file(input_path)
        if job.method != "lf-gf2" or job.cavity is None:
            raise ValueError(f"Rendered input at R={distance:.5f} is not LF-GF2 with a cavity")

        point_start = time.perf_counter()
        objective_counter = 0
        optimizer_rows: list[dict[str, Any]] = []
        confirmation_rows: list[dict[str, Any]] = []

        def objective(lam: np.ndarray, warm_state: Any | None) -> LFCorrelatedObjectiveEvaluation:
            nonlocal objective_counter
            objective_counter += 1
            reporter = Reporter(job_name=f"LF PES R={distance:.5f} eval {objective_counter:03d}")
            result = driver_theory.solve_lf_gf2_fixed_lambda(
                mol,
                psi4_options,
                opts,
                lambda_fixed=np.asarray(lam, dtype=float),
                cavity=job.cavity,
                reporter=reporter,
                psi4_verbose=args.psi4_verbose,
                initial_state=warm_state,
            )
            return LFCorrelatedObjectiveEvaluation(
                lambda_value=np.asarray(lam, dtype=float),
                total_energy=result.total_energy,
                converged=bool(result.state.converged and result.summary.converged),
                state=result.state,
                core_energy=result.core_energy,
                correlation_energy=result.correlation_energy,
                outer_iterations=result.state.outer_iterations,
                wall_time_s=result.summary.wall_time_s,
            )

        initial = objective(lambda_seed, None)
        if not initial.converged:
            raise RuntimeError(f"Initial fixed-lambda LF-GF2 solve failed at R={distance:.5f}")
        save_lf_gf2_state(point_dir / "initial_state.npz", initial.state)

        def optimizer_callback(record: LFCorrelatedEvaluationRecord) -> None:
            optimizer_rows.append(
                {
                    "evaluation": record.evaluation_index,
                    "iteration": record.iteration,
                    "role": record.role,
                    "energy": record.total_energy,
                    "converged": record.converged,
                    "outer_iterations": record.outer_iterations,
                    "wall_time_s": record.wall_time_s,
                    "lambda_norm": float(np.linalg.norm(record.lambda_value)),
                    "lambda": json.dumps(record.lambda_value.tolist()),
                }
            )
            with (point_dir / "optimizer.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(optimizer_rows[0]))
                writer.writeheader()
                writer.writerows(optimizer_rows)
            print(
                f"R={distance:.5f} OPT {record.evaluation_index:03d} "
                f"iter={record.iteration:02d} {record.role:11s} "
                f"E={record.total_energy:.12f}",
                flush=True,
            )

        def optimizer_accepted(
            iteration: int, evaluation: LFCorrelatedObjectiveEvaluation, trust_radius: float
        ) -> None:
            save_lf_gf2_state(point_dir / "coarse_state.npz", evaluation.state)
            _write_json(
                point_dir / "coarse_state.npz.json",
                {
                    "iteration": int(iteration),
                    "trust_radius": float(trust_radius),
                    "energy": float(evaluation.total_energy),
                    "lambda_norm": float(np.linalg.norm(evaluation.lambda_value)),
                },
            )

        optimizer_options = LFCorrelatedOptimizerOptions(
            max_iterations=args.maxiter,
            probe_radius=args.probe_radius,
            min_probe_radius=args.min_probe_radius,
            probe_decay=0.101,
            trust_radius=args.trust_radius,
            min_trust_radius=args.min_trust_radius,
            max_trust_radius=args.max_trust_radius,
            trust_shrink=0.5,
            trust_grow=1.20,
            energy_tolerance=args.energy_tol,
            stagnation_iterations=3,
            radial_bootstrap_iterations=0,
            random_seed=args.seed,
            component_bound=args.component_bound,
            iteration_offset=0,
        )
        coarse = optimize_correlated_lambda(
            objective,
            lambda_seed,
            options=optimizer_options,
            initial_evaluation=initial,
            evaluation_callback=optimizer_callback,
            accepted_callback=optimizer_accepted,
        )
        coarse_eval = _evaluation_from_state(coarse.state)
        save_lf_gf2_state(point_dir / "coarse_state.npz", coarse.state)

        def confirmation_callback(
            direction_index: int, role: str, evaluation: LFCorrelatedObjectiveEvaluation
        ) -> None:
            confirmation_rows.append(
                {
                    "direction": direction_index,
                    "role": role,
                    "energy": evaluation.total_energy,
                    "converged": evaluation.converged,
                    "outer_iterations": evaluation.outer_iterations,
                    "wall_time_s": evaluation.wall_time_s,
                    "lambda_norm": float(np.linalg.norm(evaluation.lambda_value)),
                    "lambda": json.dumps(evaluation.lambda_value.tolist()),
                }
            )
            with (point_dir / "confirmation.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(confirmation_rows[0]))
                writer.writeheader()
                writer.writerows(confirmation_rows)
            print(
                f"R={distance:.5f} CONFIRM {direction_index:02d} {role:13s} "
                f"E={evaluation.total_energy:.12f}",
                flush=True,
            )

        def confirmation_accepted(
            direction_index: int, evaluation: LFCorrelatedObjectiveEvaluation
        ) -> None:
            save_lf_gf2_state(final_state_path, evaluation.state)

        confirmation_options = LFMinimumConfirmationOptions(
            radius=args.confirm_radius,
            energy_tolerance=args.energy_tol,
            required_misses=args.required_misses,
            max_directions=args.max_directions,
            random_seed=args.confirm_seed,
            component_bound=args.component_bound,
        )
        confirmation = confirm_correlated_minimum(
            objective,
            coarse_eval,
            options=confirmation_options,
            evaluation_callback=confirmation_callback,
            accepted_callback=confirmation_accepted,
        )
        save_lf_gf2_state(final_state_path, confirmation.state)
        state = confirmation.state
        elapsed = time.perf_counter() - point_start

        payload = {
            "distance": float(distance),
            "confirmed": bool(confirmation.confirmed),
            "reason": confirmation.reason,
            "energy_tolerance": float(args.energy_tol),
            "confirmation_radius": float(args.confirm_radius),
            "required_misses": int(args.required_misses),
            "total_energy": float(state.total_energy),
            "core_energy": float(state.core_energy),
            "correlation_energy": float(state.correlation_energy),
            "xi": float(state.xi),
            "lambda_norm": float(np.linalg.norm(state.lambda_fixed)),
            "lambda": state.lambda_fixed.tolist(),
            "optimizer_iterations": int(coarse.iterations),
            "optimizer_evaluations": int(len(coarse.evaluations)),
            "optimizer_converged": bool(coarse.converged),
            "optimizer_reason": coarse.reason,
            "confirmation_directions": int(confirmation.directions),
            "wall_time_s": float(elapsed),
            "input": str(input_path),
            "state": str(final_state_path),
            "optimizer_history": _serialize_optimizer_history(coarse),
            "confirmation_history": _serialize_confirmation_history(confirmation),
        }
        _write_json(point_json, payload)
        _append_master(
            master_csv,
            {
                "distance": distance,
                "total_energy": f"{state.total_energy:.15f}",
                "core_energy": f"{state.core_energy:.15f}",
                "correlation_energy": f"{state.correlation_energy:.15f}",
                "xi": f"{state.xi:.15e}",
                "lambda_norm": f"{np.linalg.norm(state.lambda_fixed):.15e}",
                "lambda": json.dumps(state.lambda_fixed.tolist()),
                "optimizer_iterations": coarse.iterations,
                "optimizer_evaluations": len(coarse.evaluations),
                "confirmation_directions": confirmation.directions,
                "confirmed": confirmation.confirmed,
                "wall_time_s": f"{elapsed:.6f}",
                "state": str(final_state_path),
                "input": str(input_path),
            },
        )

        print(
            f"R={distance:.5f} FINAL E={state.total_energy:.12f} "
            f"||lambda||={np.linalg.norm(state.lambda_fixed):.8e} "
            f"confirmed={confirmation.confirmed}",
            flush=True,
        )
        if not confirmation.confirmed:
            raise RuntimeError(
                f"LF minimum at R={distance:.5f} was not confirmed within "
                f"{args.max_directions} directions; chain stopped before propagating lambda."
            )
        lambda_seed = state.lambda_fixed.copy()


if __name__ == "__main__":
    main()
