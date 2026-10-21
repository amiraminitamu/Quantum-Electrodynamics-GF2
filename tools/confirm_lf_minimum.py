#!/usr/bin/env python
# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Confirm an optimized LF-GF2 state with independent local directions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from gf2 import driver_theory
from gf2.input_parser import parse_input_file
from gf2.lf_correlated_optimizer import LFCorrelatedObjectiveEvaluation
from gf2.lf_minimum_confirmation import (
    LFMinimumConfirmationOptions,
    confirm_correlated_minimum,
)
from gf2.lf_state import load_lf_gf2_state, save_lf_gf2_state
from gf2.reporting import Reporter


def _meta_path(state_path: Path) -> Path:
    return state_path.with_name(state_path.name + ".json")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Confirm a converged correlated LF-GF2 state using independent "
            "symmetric Rademacher directions at a fixed local radius."
        )
    )
    parser.add_argument("input", help="LF-GF2 molecular input file.")
    parser.add_argument("--state", required=True, help="Converged LF-GF2 state checkpoint.")
    parser.add_argument("--radius", type=float, default=1.0e-3)
    parser.add_argument("--energy-tol", type=float, default=1.0e-7)
    parser.add_argument("--required-misses", type=int, default=10)
    parser.add_argument("--max-directions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--component-bound", type=float, default=None)
    parser.add_argument("--csv", default="lf_minimum_confirmation.csv")
    parser.add_argument("--state-out", default="lf_confirmed_state.npz")
    parser.add_argument("--psi4-verbose", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    state_path = Path(args.state).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve()
    state_out = Path(args.state_out).expanduser().resolve()

    job, mol, psi4_options, opts = parse_input_file(input_path)
    if job.method != "lf-gf2" or job.cavity is None:
        raise ValueError("Minimum confirmation requires an LF-GF2 cavity input.")

    start_state = load_lf_gf2_state(state_path)
    if not start_state.converged:
        raise ValueError("--state must contain a converged LF-GF2 fixed point.")

    current_start = LFCorrelatedObjectiveEvaluation(
        lambda_value=start_state.lambda_fixed,
        total_energy=start_state.total_energy,
        converged=True,
        state=start_state,
        core_energy=start_state.core_energy,
        correlation_energy=start_state.correlation_energy,
        outer_iterations=start_state.outer_iterations,
        wall_time_s=0.0,
    )

    options = LFMinimumConfirmationOptions(
        radius=args.radius,
        energy_tolerance=args.energy_tol,
        required_misses=args.required_misses,
        max_directions=args.max_directions,
        random_seed=args.seed,
        component_bound=args.component_bound,
    )

    evaluation_counter = 0
    rows: list[dict[str, object]] = []

    def objective(lam: np.ndarray, warm_state: Any | None) -> LFCorrelatedObjectiveEvaluation:
        nonlocal evaluation_counter
        evaluation_counter += 1
        reporter = Reporter(
            job_name=f"{job.name} | minimum confirmation eval {evaluation_counter:03d}"
        )
        result = driver_theory.solve_lf_gf2_fixed_lambda(
            mol,
            psi4_options,
            opts,
            lambda_fixed=lam,
            cavity=job.cavity,
            reporter=reporter,
            psi4_verbose=args.psi4_verbose,
            initial_state=warm_state,
        )
        return LFCorrelatedObjectiveEvaluation(
            lambda_value=lam,
            total_energy=result.total_energy,
            converged=bool(result.state.converged and result.summary.converged),
            state=result.state,
            core_energy=result.core_energy,
            correlation_energy=result.correlation_energy,
            outer_iterations=result.state.outer_iterations,
            wall_time_s=result.summary.wall_time_s,
        )

    def write_rows() -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "direction",
            "role",
            "energy",
            "converged",
            "outer_iterations",
            "wall_time_s",
            "lambda_norm",
            "lambda",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def evaluation_callback(
        direction_index: int,
        role: str,
        evaluation: LFCorrelatedObjectiveEvaluation,
    ) -> None:
        rows.append(
            {
                "direction": direction_index,
                "role": role,
                "energy": f"{evaluation.total_energy:.15f}",
                "converged": evaluation.converged,
                "outer_iterations": evaluation.outer_iterations,
                "wall_time_s": f"{evaluation.wall_time_s:.6f}",
                "lambda_norm": f"{np.linalg.norm(evaluation.lambda_value):.15e}",
                "lambda": json.dumps(evaluation.lambda_value.tolist()),
            }
        )
        write_rows()
        print(
            f"CONFIRM {direction_index:02d} {role:13s} "
            f"E={evaluation.total_energy:.12f} Eh "
            f"outer={evaluation.outer_iterations:4d} "
            f"wall={evaluation.wall_time_s:8.2f} s",
            flush=True,
        )

    def accepted_callback(
        direction_index: int,
        evaluation: LFCorrelatedObjectiveEvaluation,
    ) -> None:
        save_lf_gf2_state(state_out, evaluation.state)
        _write_json_atomic(
            _meta_path(state_out),
            {
                "format_version": 1,
                "confirmation_direction": int(direction_index),
                "confirmation_radius": float(args.radius),
                "confirmation_energy_tolerance": float(args.energy_tol),
                "confirmation_required_misses": int(args.required_misses),
                "confirmation_seed": int(args.seed),
                "accepted_energy": float(evaluation.total_energy),
                "accepted_lambda_norm": float(np.linalg.norm(evaluation.lambda_value)),
            },
        )

    save_lf_gf2_state(state_out, start_state)

    result = confirm_correlated_minimum(
        objective,
        current_start,
        options=options,
        evaluation_callback=evaluation_callback,
        accepted_callback=accepted_callback,
    )

    # Persist the final state and complete metadata even when the optimum never moved.
    save_lf_gf2_state(state_out, result.state)
    _write_json_atomic(
        _meta_path(state_out),
        {
            "format_version": 1,
            "confirmation_radius": float(args.radius),
            "confirmation_energy_tolerance": float(args.energy_tol),
            "confirmation_required_misses": int(args.required_misses),
            "confirmation_seed": int(args.seed),
            "confirmation_directions_completed": int(result.directions),
            "confirmed": bool(result.confirmed),
            "reason": result.reason,
            "accepted_energy": float(result.energy),
            "accepted_lambda_norm": float(np.linalg.norm(result.lambda_value)),
        },
    )

    print("\n" + "=" * 79)
    print("LF-GF2 MINIMUM CONFIRMATION SUMMARY")
    print("=" * 79)
    print(f"Start E             = {current_start.total_energy:.12f} Eh")
    print(f"Final E             = {result.energy:.12f} Eh")
    print(f"Final ||lambda||    = {np.linalg.norm(result.lambda_value):.12e}")
    print(f"Directions tested   = {result.directions}")
    print(f"Confirmed           = {result.confirmed}")
    print(f"Reason              = {result.reason}")
    print(f"CSV                 = {csv_path}")
    print(f"Confirmed state     = {state_out}")
    print("=" * 79)


if __name__ == "__main__":
    main()
