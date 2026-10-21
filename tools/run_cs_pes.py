#!/usr/bin/env python
# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Run a corrected CS-GF2 diatomic PES on a specified geometry grid."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from gf2 import driver_theory
from gf2.input_parser import parse_input_file
from gf2.pes_workflow import distance_tag, write_diatomic_input
from gf2.reporting import Reporter


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _rewrite_master(path: Path, new_row: dict[str, str]) -> None:
    fieldnames = [
        "distance",
        "total_energy",
        "core_energy",
        "correlation_energy",
        "converged",
        "wall_time_s",
        "input",
        "json",
    ]
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    rows = [row for row in rows if row.get("distance") != new_row["distance"]]
    rows.append(new_row)
    rows.sort(key=lambda row: float(row["distance"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CS-GF2 diatomic PES.")
    parser.add_argument(
        "template", help="Diatomic input template; numerical settings and cavity are preserved."
    )
    parser.add_argument("--distances", nargs="+", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--psi4-verbose", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    template = Path(args.template).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    master = output_dir / "cs_pes.csv"

    for distance in args.distances:
        point_dir = output_dir / f"R_{distance_tag(distance)}"
        point_dir.mkdir(parents=True, exist_ok=True)
        input_path = point_dir / "cs.inp"
        result_json = point_dir / "result.json"
        if result_json.exists() and not args.no_resume:
            payload = json.loads(result_json.read_text(encoding="utf-8"))
            if bool(payload.get("converged", False)):
                print(f"R={distance:.5f}: converged CS checkpoint exists; skipping.", flush=True)
                continue

        write_diatomic_input(
            template,
            input_path,
            distance,
            method="cs-gf2",
            job_name=f"CS-GF2 PES R={distance:.5f}",
        )
        job, mol, psi4_options, opts = parse_input_file(input_path)
        reporter = Reporter(job_name=job.name)
        start = time.perf_counter()
        core, corr, summary = driver_theory.run_gf2(
            mol,
            psi4_options,
            opts,
            reporter=reporter,
            psi4_verbose=args.psi4_verbose,
            method="cs-gf2",
            cavity=job.cavity,
        )
        elapsed = time.perf_counter() - start
        payload = {
            "distance": float(distance),
            "total_energy": float(core + corr),
            "core_energy": float(core),
            "correlation_energy": float(corr),
            "converged": bool(summary.converged),
            "wall_time_s": float(elapsed),
            "summary": summary.as_dict(),
            "input": str(input_path),
        }
        _write_json(result_json, payload)
        _rewrite_master(
            master,
            {
                "distance": f"{distance:.8f}",
                "total_energy": f"{core + corr:.15f}",
                "core_energy": f"{core:.15f}",
                "correlation_energy": f"{corr:.15f}",
                "converged": str(bool(summary.converged)),
                "wall_time_s": f"{elapsed:.6f}",
                "input": str(input_path),
                "json": str(result_json),
            },
        )
        print(
            f"R={distance:.5f} CS-GF2 E={core + corr:.12f} converged={summary.converged}",
            flush=True,
        )
        if not summary.converged:
            raise RuntimeError(f"CS-GF2 did not converge at R={distance:.5f}")


if __name__ == "__main__":
    main()
