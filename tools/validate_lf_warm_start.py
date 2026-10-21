#!/usr/bin/env python
# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Validate cold/warm fixed-lambda LF-GF2 equivalence on one molecular case."

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gf2 import driver_theory
from gf2.input_parser import parse_input_file
from gf2.reporting import Reporter
from scan_lf_lambda import _determine_common_lambda_stat


def _solve(
    *,
    label: str,
    mol,
    psi4_options,
    opts,
    cavity,
    lam: np.ndarray,
    initial_state=None,
):
    reporter = Reporter(job_name=label)
    result = driver_theory.solve_lf_gf2_fixed_lambda(
        mol,
        psi4_options,
        opts,
        lambda_fixed=lam,
        cavity=cavity,
        reporter=reporter,
        initial_state=initial_state,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cold and warm fixed-lambda LF-GF2 target solves."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="examples/lih_lf_gf2_option_a.inp",
        help="LF-GF2 input file.",
    )
    parser.add_argument("--source", type=float, default=0.65)
    parser.add_argument("--target", type=float, default=0.60)
    parser.add_argument(
        "--energy-tol",
        type=float,
        default=1.0e-7,
        help="Maximum |E_warm-E_cold| in Eh (default 1e-7).",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    job, mol, psi4_options, opts = parse_input_file(input_path)
    if job.method != "lf-gf2" or job.cavity is None:
        raise ValueError("Warm-start validation requires an LF-GF2 cavity input.")

    print("Determining common static LF-HF direction ...", flush=True)
    lambda_stat = _determine_common_lambda_stat(input_path)
    source_lambda = float(args.source) * lambda_stat
    target_lambda = float(args.target) * lambda_stat

    print(f"source t={args.source:.6f}, ||lambda||={np.linalg.norm(source_lambda):.12e}")
    print(f"target t={args.target:.6f}, ||lambda||={np.linalg.norm(target_lambda):.12e}")

    source = _solve(
        label=f"warm-source t={args.source:.3f}",
        mol=mol,
        psi4_options=psi4_options,
        opts=opts,
        cavity=job.cavity,
        lam=source_lambda,
    )
    if not source.state.converged:
        raise RuntimeError("Source fixed-lambda LF-GF2 calculation did not converge.")

    cold = _solve(
        label=f"cold-target t={args.target:.3f}",
        mol=mol,
        psi4_options=psi4_options,
        opts=opts,
        cavity=job.cavity,
        lam=target_lambda,
    )
    if not cold.state.converged:
        raise RuntimeError("Cold target fixed-lambda LF-GF2 calculation did not converge.")

    warm = _solve(
        label=f"warm-target t={args.target:.3f}",
        mol=mol,
        psi4_options=psi4_options,
        opts=opts,
        cavity=job.cavity,
        lam=target_lambda,
        initial_state=source.state,
    )
    if not warm.state.converged:
        raise RuntimeError("Warm target fixed-lambda LF-GF2 calculation did not converge.")

    delta = float(warm.total_energy - cold.total_energy)
    print("\n" + "=" * 79)
    print("LF-GF2 WARM-START VALIDATION")
    print("=" * 79)
    print(
        f"source  E = {source.total_energy: .12f} Eh   "
        f"outer={source.state.outer_iterations:4d}   "
        f"wall={source.summary.wall_time_s:9.2f} s"
    )
    print(
        f"cold    E = {cold.total_energy: .12f} Eh   "
        f"outer={cold.state.outer_iterations:4d}   "
        f"wall={cold.summary.wall_time_s:9.2f} s"
    )
    print(
        f"warm    E = {warm.total_energy: .12f} Eh   "
        f"outer={warm.state.outer_iterations:4d}   "
        f"wall={warm.summary.wall_time_s:9.2f} s"
    )
    print(f"E_warm - E_cold = {delta:+.12e} Eh")
    if cold.summary.wall_time_s > 0.0:
        speedup = float(cold.summary.wall_time_s / warm.summary.wall_time_s)
        print(f"cold/warm wall-time ratio = {speedup:.3f}x")
    print("=" * 79)

    if abs(delta) > float(args.energy_tol):
        raise SystemExit(
            f"FAILED: |E_warm-E_cold|={abs(delta):.3e} Eh exceeds {args.energy_tol:.3e} Eh"
        )
    print("PASS: cold and warm target energies agree within tolerance.")


if __name__ == "__main__":
    main()
