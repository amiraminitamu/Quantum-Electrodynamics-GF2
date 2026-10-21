#!/usr/bin/env python
# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Scan fully self-consistent projected LF-GF2 energy along one fixed LF direction."

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from gf2 import driver_theory
from gf2.input_parser import parse_input_file
from gf2.reporting import Reporter


def _parse_scales(text: str) -> list[float]:
    values: list[float] = []
    for token in text.replace(",", " ").split():
        value = float(token)
        if not np.isfinite(value):
            raise ValueError("All lambda scan scales must be finite.")
        values.append(value)
    if not values:
        raise ValueError("At least one lambda scan scale is required.")
    return values


def _determine_common_lambda_stat(input_path: Path) -> np.ndarray:
    """Solve one LF-HF reference and return the common scan direction."""
    job, mol, psi4_options, opts = parse_input_file(input_path)
    if job.method != "lf-gf2":
        raise ValueError(
            "The scan input must use method=lf-gf2 so lambda_stat can be "
            "obtained from the projected LF-HF reference."
        )

    ref = driver_theory._rhf_reference(mol, psi4_options, opts, job.cavity)
    ref = driver_theory._prepare_lf_reference(ref, opts)
    params = ref.get("lf_params")
    if params is None:
        raise RuntimeError("LF-HF reference did not return LF parameters.")
    if not bool(ref.get("lf_hf_converged", False)):
        raise RuntimeError(
            "The common LF-HF reference did not converge; refusing to define "
            "a correlated-energy scan direction from an unconverged vector."
        )
    return np.asarray(params.lam, dtype=float).copy()


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_one(
    input_path: Path,
    scale: float,
    lambda_stat: np.ndarray,
) -> dict[str, object]:
    job, mol, psi4_options, opts = parse_input_file(input_path)
    if job.method != "lf-gf2":
        raise ValueError(
            "The scan input must use method=lf-gf2 so the projected LF Hamiltonian is available."
        )

    common = np.asarray(lambda_stat, dtype=float)
    fixed = float(scale) * common
    reporter = Reporter(job_name=f"{job.name} | fixed lambda t={scale:.3f}")
    core, corr, summary = driver_theory.run_lf_gf2_fixed_lambda(
        mol,
        psi4_options,
        opts,
        lambda_fixed=fixed,
        reporter=reporter,
        cavity=job.cavity,
    )

    metadata = dict(summary.interaction_metadata)
    coupled = dict(metadata.get("coupled_energy_components", {}))
    residuals = dict(summary.convergence_diagnostics)
    total = float(core + corr)

    return {
        "t": float(scale),
        "core_energy": float(core),
        "correlation_energy": float(corr),
        "total_energy": total,
        "converged": bool(summary.converged),
        "wall_time_s": float(summary.wall_time_s),
        "lambda_stat_norm": float(np.linalg.norm(common)),
        "lambda_fixed_norm": float(np.linalg.norm(fixed)),
        "lf_z": float(metadata.get("lf_z", np.nan)),
        "lf_parameter_mode": str(metadata.get("lf_parameter_mode", "unknown")),
        "lf_reference_mode": str(metadata.get("lf_reference_mode", "unknown")),
        "C_ee": float(coupled.get("C_ee", np.nan)),
        "C_eb": float(coupled.get("C_eb", np.nan)),
        "E_eb": float(coupled.get("E_eb", np.nan)),
        "E_ph": float(coupled.get("E_ph", np.nan)),
        "sigma_g_pi_d": float(coupled.get("sigma_g_pi_d_residual", np.nan)),
        "res_energy": float(residuals.get("energy", np.nan)),
        "res_particle_number": float(residuals.get("particle_number", np.nan)),
        "res_sigma_iw": float(residuals.get("sigma_iw", np.nan)),
        "res_bosonic_dyson": float(residuals.get("bosonic_dyson", np.nan)),
        "res_lambda": float(residuals.get("lambda", np.nan)),
        "res_z": float(residuals.get("lf_parameter_z", np.nan)),
        "res_z_stationarity": float(residuals.get("lf_z_stationarity", np.nan)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fully self-consistent projected LF-GF2 scan along one common lambda=t*lambda_stat ray."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="examples/lih_lf_gf2_option_a.inp",
        help="LF-GF2 input file (default: examples/lih_lf_gf2_option_a.inp)",
    )
    parser.add_argument(
        "--scales",
        default="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0",
        help="Comma- or space-separated t values.",
    )
    parser.add_argument(
        "--csv",
        default="lf_lambda_scan.csv",
        help="CSV destination; rewritten after every completed point.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve()
    scales = _parse_scales(args.scales)

    print("=" * 79)
    print("Fixed-lambda correlated LF-GF2 scan")
    print(f"Input: {input_path}")
    print(f"Scales: {scales}")
    print(f"CSV: {csv_path}")
    print("Determining one common static LF-HF lambda direction ...", flush=True)
    lambda_stat = _determine_common_lambda_stat(input_path)
    print(f"Common ||lambda_stat|| = {np.linalg.norm(lambda_stat):.12e}")
    print("Common lambda_stat =")
    print(np.array2string(lambda_stat, precision=12, suppress_small=False))
    print("=" * 79, flush=True)

    rows: list[dict[str, object]] = []
    e0: float | None = None

    for index, scale in enumerate(scales, start=1):
        print("\n" + "#" * 79)
        print(f"# scan point {index}/{len(scales)}: t = {scale:.8f}")
        print("#" * 79, flush=True)

        row = _run_one(input_path, scale, lambda_stat)
        energy = float(row["total_energy"])
        if e0 is None and abs(scale) < 1.0e-14:
            e0 = energy
        delta_eh = energy - e0 if e0 is not None else np.nan
        row["delta_vs_t0_Eh"] = delta_eh
        row["delta_vs_t0_kcal_mol"] = delta_eh * 627.509474 if np.isfinite(delta_eh) else np.nan
        rows.append(row)
        _write_rows(csv_path, rows)

        print(
            f"SCAN RESULT t={scale:.3f}: "
            f"E={energy:.12f} Eh  "
            f"converged={row['converged']}  "
            f"delta_vs_t0={delta_eh:+.9e} Eh",
            flush=True,
        )

    converged_rows = [row for row in rows if bool(row["converged"])]
    if converged_rows:
        best = min(converged_rows, key=lambda row: float(row["total_energy"]))
        print("\n" + "=" * 79)
        print("SCAN SUMMARY")
        print(
            f"Lowest converged point: t={float(best['t']):.6f}, "
            f"E={float(best['total_energy']):.12f} Eh"
        )
        if e0 is not None:
            delta = float(best["total_energy"]) - e0
            print(f"Relative to t=0: {delta:+.9e} Eh ({delta * 627.509474:+.6f} kcal/mol)")
        print(f"CSV written to: {csv_path}")
        print("=" * 79)


if __name__ == "__main__":
    main()
