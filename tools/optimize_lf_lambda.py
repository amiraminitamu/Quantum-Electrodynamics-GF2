#!/usr/bin/env python
# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Optimize the projected LF vector against the fully converged LF-GF2 energy."

from __future__ import annotations

import argparse
import csv
import json
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
from gf2.lf_state import load_lf_gf2_state, save_lf_gf2_state
from gf2.reporting import Reporter
from scan_lf_lambda import _determine_common_lambda_stat


def _write_checkpoint(
    path: Path,
    records: list[LFCorrelatedEvaluationRecord],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "evaluation",
        "iteration",
        "role",
        "total_energy",
        "converged",
        "outer_iterations",
        "wall_time_s",
        "warm_start_used",
        "warm_source_lambda_norm",
        "lambda_norm",
        "lambda",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "evaluation": record.evaluation_index,
                    "iteration": record.iteration,
                    "role": record.role,
                    "total_energy": f"{record.total_energy:.15f}",
                    "converged": record.converged,
                    "outer_iterations": record.outer_iterations,
                    "wall_time_s": f"{record.wall_time_s:.6f}",
                    "warm_start_used": record.warm_start_used,
                    "warm_source_lambda_norm": (
                        ""
                        if record.warm_source_lambda_norm is None
                        else f"{record.warm_source_lambda_norm:.15e}"
                    ),
                    "lambda_norm": f"{np.linalg.norm(record.lambda_value):.15e}",
                    "lambda": json.dumps(record.lambda_value.tolist()),
                }
            )


def _meta_path(state_path: Path) -> Path:
    return state_path.with_name(state_path.name + ".json")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_restart_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Restart metadata in {path} must be a JSON object.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Multidimensional correlated-energy optimization of the projected "
            "LF vector using warm-started simultaneous perturbations."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="examples/lih_lf_gf2_option_a.inp",
        help="LF-GF2 input file.",
    )
    parser.add_argument(
        "--initial-scale",
        type=float,
        default=1.0,
        help=(
            "Scale applied to the static LF-HF vector before a fresh correlated "
            "optimization. Ignored when --restart-state is supplied."
        ),
    )
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--probe-radius", type=float, default=8.0e-3)
    parser.add_argument("--min-probe-radius", type=float, default=1.0e-3)
    parser.add_argument("--probe-decay", type=float, default=0.101)
    parser.add_argument(
        "--trust-radius",
        type=float,
        default=None,
        help="Initial trust radius; restart metadata is used when omitted.",
    )
    parser.add_argument("--min-trust-radius", type=float, default=1.0e-3)
    parser.add_argument("--max-trust-radius", type=float, default=5.0e-2)
    parser.add_argument("--trust-shrink", type=float, default=0.5)
    parser.add_argument("--trust-grow", type=float, default=1.20)
    parser.add_argument("--energy-tol", type=float, default=5.0e-8)
    parser.add_argument("--stagnation", type=int, default=3)
    parser.add_argument(
        "--radial-bootstrap",
        type=int,
        default=None,
        help=(
            "Number of global initial radial iterations. Fresh runs default to 3; "
            "restart metadata is used when omitted."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--iteration-offset",
        type=int,
        default=None,
        help="Global completed-iteration count; normally restored automatically.",
    )
    parser.add_argument(
        "--component-bound",
        type=float,
        default=None,
        help="Optional symmetric |lambda_p| bound; disabled by default.",
    )
    parser.add_argument(
        "--csv",
        default="lf_correlated_optimization.csv",
        help="Checkpoint CSV rewritten after every expensive objective call.",
    )
    parser.add_argument(
        "--state-out",
        default="lf_correlated_state.npz",
        help=(
            "Accepted correlated-state checkpoint. A JSON sidecar stores outer "
            "iteration/trust-radius metadata."
        ),
    )
    parser.add_argument(
        "--restart-state",
        default=None,
        help=(
            "Resume from a previously written accepted-state checkpoint. The "
            "initial correlated GF2 objective call is skipped entirely."
        ),
    )
    parser.add_argument("--psi4-verbose", action="store_true")
    args = parser.parse_args()

    if not np.isfinite(args.initial_scale):
        raise ValueError("--initial-scale must be finite.")

    input_path = Path(args.input).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve()
    state_out = Path(args.state_out).expanduser().resolve()
    job, mol, psi4_options, opts = parse_input_file(input_path)
    if job.method != "lf-gf2" or job.cavity is None:
        raise ValueError("Correlated LF optimization requires an LF-GF2 cavity input.")

    restart_evaluation: LFCorrelatedObjectiveEvaluation | None = None
    restart_meta: dict[str, Any] = {}
    if args.restart_state is not None:
        restart_path = Path(args.restart_state).expanduser().resolve()
        restart_state = load_lf_gf2_state(restart_path)
        if not restart_state.converged:
            raise ValueError("--restart-state must contain a converged LF-GF2 state.")
        restart_meta = _read_restart_meta(_meta_path(restart_path))
        lambda_initial = restart_state.lambda_fixed.copy()
        restart_evaluation = LFCorrelatedObjectiveEvaluation(
            lambda_value=lambda_initial,
            total_energy=restart_state.total_energy,
            converged=True,
            state=restart_state,
            core_energy=restart_state.core_energy,
            correlation_energy=restart_state.correlation_energy,
            outer_iterations=restart_state.outer_iterations,
            wall_time_s=0.0,
        )
    else:
        restart_path = None
        print("Determining converged static LF-HF starting vector ...", flush=True)
        lambda_stat = _determine_common_lambda_stat(input_path)
        lambda_initial = float(args.initial_scale) * lambda_stat

    radial_bootstrap = (
        int(args.radial_bootstrap)
        if args.radial_bootstrap is not None
        else int(restart_meta.get("radial_bootstrap_iterations", 3))
    )
    random_seed = (
        int(args.seed) if args.seed is not None else int(restart_meta.get("random_seed", 10086))
    )
    iteration_offset = (
        int(args.iteration_offset)
        if args.iteration_offset is not None
        else int(restart_meta.get("iteration_offset", 0))
    )
    trust_radius = (
        float(args.trust_radius)
        if args.trust_radius is not None
        else float(restart_meta.get("trust_radius", 2.0e-2))
    )

    print("=" * 79)
    print("CORRELATED-ENERGY LF-GF2 OPTIMIZATION")
    print("=" * 79)
    print(f"Input: {input_path}")
    if restart_path is None:
        print(f"initial scale       = {args.initial_scale:.8f}")
    else:
        print(f"Restart state       = {restart_path}")
        print("Initial GF2 solve   = skipped (persisted converged state reused)")
    print(f"||lambda_initial|| = {np.linalg.norm(lambda_initial):.12e}")
    print(f"iteration offset    = {iteration_offset}")
    print(f"trust radius        = {trust_radius:.6e}")
    print(f"radial bootstrap    = {radial_bootstrap}")
    print(f"random seed         = {random_seed}")
    print("lambda_initial =")
    print(np.array2string(lambda_initial, precision=12, suppress_small=False))
    print("=" * 79, flush=True)

    optimizer_options = LFCorrelatedOptimizerOptions(
        max_iterations=args.maxiter,
        probe_radius=args.probe_radius,
        min_probe_radius=args.min_probe_radius,
        probe_decay=args.probe_decay,
        trust_radius=trust_radius,
        min_trust_radius=args.min_trust_radius,
        max_trust_radius=args.max_trust_radius,
        trust_shrink=args.trust_shrink,
        trust_grow=args.trust_grow,
        energy_tolerance=args.energy_tol,
        stagnation_iterations=args.stagnation,
        radial_bootstrap_iterations=radial_bootstrap,
        random_seed=random_seed,
        component_bound=args.component_bound,
        iteration_offset=iteration_offset,
    )

    evaluation_counter = 0

    def objective(
        lam: np.ndarray,
        warm_state: Any | None,
    ) -> LFCorrelatedObjectiveEvaluation:
        nonlocal evaluation_counter
        evaluation_counter += 1
        label = f"{job.name} | correlated-lambda eval {evaluation_counter:03d}"
        reporter = Reporter(job_name=label)
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

    checkpoint_records: list[LFCorrelatedEvaluationRecord] = []

    def checkpoint(record: LFCorrelatedEvaluationRecord) -> None:
        checkpoint_records.append(record)
        _write_checkpoint(csv_path, checkpoint_records)
        print(
            "OUTER EVAL "
            f"{record.evaluation_index:03d}  "
            f"iter={record.iteration:02d}  role={record.role:11s}  "
            f"E={record.total_energy:.12f} Eh  "
            f"outer={record.outer_iterations:4d}  "
            f"wall={record.wall_time_s:8.2f} s  "
            f"warm={record.warm_start_used}",
            flush=True,
        )

    def accepted_checkpoint(
        iteration: int,
        current: LFCorrelatedObjectiveEvaluation,
        current_trust_radius: float,
    ) -> None:
        if current.state is None or not current.converged:
            raise RuntimeError("Refusing to checkpoint a nonconverged accepted state.")
        save_lf_gf2_state(state_out, current.state)
        _write_json_atomic(
            _meta_path(state_out),
            {
                "format_version": 1,
                "iteration_offset": int(iteration),
                "trust_radius": float(current_trust_radius),
                "radial_bootstrap_iterations": int(radial_bootstrap),
                "random_seed": int(random_seed),
                "accepted_energy": float(current.total_energy),
                "accepted_lambda_norm": float(np.linalg.norm(current.lambda_value)),
            },
        )

    result = optimize_correlated_lambda(
        objective,
        lambda_initial,
        options=optimizer_options,
        initial_evaluation=restart_evaluation,
        evaluation_callback=checkpoint,
        accepted_callback=accepted_checkpoint,
    )

    print("\n" + "=" * 79)
    print("CORRELATED LF OPTIMIZATION SUMMARY")
    print("=" * 79)
    print(f"Accepted E          = {result.energy_opt:.12f} Eh")
    print(f"Accepted ||lambda|| = {np.linalg.norm(result.lambda_opt):.12e}")
    print(f"Lowest sampled E    = {result.energy_sampled_best:.12f} Eh")
    print(f"Sampled ||lambda||  = {np.linalg.norm(result.lambda_sampled_best):.12e}")
    print(f"Sampled-accepted dE = {result.energy_sampled_best - result.energy_opt:+.12e} Eh")
    print(f"Outer iterations    = {result.iterations}")
    print(f"GF2 objective calls = {len(result.evaluations)}")
    print(f"Optimizer converged = {result.converged}")
    print(f"Reason               = {result.reason}")
    print("Accepted lambda =")
    print(np.array2string(result.lambda_opt, precision=12, suppress_small=False))
    if not np.array_equal(result.lambda_sampled_best, result.lambda_opt):
        print("Lowest-sampled lambda =")
        print(
            np.array2string(
                result.lambda_sampled_best,
                precision=12,
                suppress_small=False,
            )
        )
    print(f"Evaluation CSV       = {csv_path}")
    print(f"Restart state        = {state_out}")
    print(f"Restart metadata     = {_meta_path(state_out)}")

    if result.history:
        print("\nIteration history:")
        for row in result.history:
            print(
                f"  {row.iteration:02d} {row.direction_kind:10s}  "
                f"Estart={row.energy_start:.12f}  "
                f"Eacc={row.accepted_energy:.12f}  "
                f"dE={row.improvement:+.3e}  "
                f"accepted={row.accepted_role:11s}  "
                f"trust={row.trust_radius_before:.3e}->{row.trust_radius_after:.3e}"
            )
    print("=" * 79)


if __name__ == "__main__":
    main()
