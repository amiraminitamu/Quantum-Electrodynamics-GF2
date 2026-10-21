#!/usr/bin/env python
# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Generate versioned H2/cc-pVDZ DF-GF2 regression references."

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psi4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gf2 import run_gf2  # noqa: E402
from gf2.reporting import Reporter  # noqa: E402

BONDS = (0.4, 0.5, 0.6, 0.74, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5)
PSI4_OPTIONS: dict[str, Any] = {"basis": "cc-pvdz", "scf_type": "pk"}
GF2_OPTIONS: dict[str, Any] = {
    "module": "ft",
    "type": "deterministic",
    "eri_type": "df",
    "se_fit_basis": "cc-pvdz-ri",
    "fock_fit_basis": "cc-pvdz-ri",
    "df_type": "RIFIT",
    "beta": 50,
    "n_ifreq": 300,
    "mid_pt": 100,
    "n_ifreq_full": 10000,
    "n_itime": 300,
    "n_poly": 50,
    "grid_changed": True,
    "thresh_ne": 1.0e-6,
    "thresh_fock_rms": 1.0e-6,
    "thresh_energy": 1.0e-6,
    "bisection_on": False,
    "maxiter_bisection": 60,
    "newton_on": True,
    "newton_damping": 0.5,
    "maxiter_newton": 60,
    "mu_guess_thresh": 1.0e-6,
    "fit_width": 0.05,
    "n_fit_pts": 50000,
    "maxiter_fock_diis": 30,
    "maxiter_gf_diis": 30,
    "n_threads": 2,
    "fock_diis_on_off": "on",
    "debug": False,
    "hf": True,
    "mp2": False,
}


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
    return result.stdout.strip()


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _numpy_configuration() -> str:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        np.show_config()
    text = output.getvalue().strip()

    text = text.replace(sys.prefix, "<PYTHON_PREFIX>")
    text = text.replace(str(Path.home()), "<HOME>")
    return re.sub(r"/(?:Users|home)/[^/\s,]+", "<HOME>", text)


def _metadata(command: list[str], options: dict[str, Any]) -> dict[str, Any]:
    thread_keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/generate_electronic_df_references.py",
        "generation_command": shlex.join(command),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_worktree_dirty": bool(_git("status", "--porcelain")),
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
        },
        "packages": {
            "psi4": str(psi4.__version__),
            "numpy": np.__version__,
            "scipy": _version("scipy"),
            "opt_einsum": _version("opt_einsum"),
            "qcelemental": _version("qcelemental"),
        },
        "blas": _numpy_configuration(),
        "thread_environment": {key: os.environ.get(key) for key in thread_keys},
        "psi4_threads": options["n_threads"],
    }


def _molecule(bond: float) -> psi4.core.Molecule:
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    return psi4.geometry(
        f"""
        0 1
        H 0.0 0.0 0.0
        H 0.0 0.0 {bond}
        units angstrom
        symmetry c1
        """
    )


def _run_case(bond: float, options: dict[str, Any]) -> dict[str, Any]:
    iterations: list[dict[str, Any]] = []
    reference: dict[str, Any] = {}

    def collect(stage: str, payload: dict[str, Any]) -> None:
        nonlocal reference
        if stage == "reference":
            reference = payload
        elif stage.startswith("iteration_"):
            iterations.append(payload)

    core, corr, summary = run_gf2(
        _molecule(bond),
        PSI4_OPTIONS,
        options,
        reporter=Reporter(echo=False),
        diagnostic_callback=collect,
    )
    last = iterations[-1]
    previous = iterations[-2] if len(iterations) > 1 else iterations[-1]
    return {
        "bond": bond,
        "hf_energy": summary.hf_energy,
        "core": core,
        "corr": corr,
        "total": core + corr,
        "converged": summary.converged,
        "iterations": len(iterations),
        "chemical_potential": float(last["mu"]),
        "particle_number": float(last["particle_number"]),
        "density_residual_fro": float(np.linalg.norm(last["density"] - previous["density"])),
        "self_energy_residual_fro": float(
            np.linalg.norm(last["Sigma_ee_tau"] - previous["Sigma_ee_tau"])
        ),
        "final_energy_change": abs(float(last["total_energy"]) - float(previous["total_energy"])),
        "nuclear_repulsion": float(reference["E_nuc"]),
    }


def _print_table(cases: list[dict[str, Any]], old: dict[float, dict[str, Any]]) -> None:
    print("R/Angstrom          core/Ha          corr/Ha         total/Ha      |dE old|")
    for case in cases:
        prior = old.get(float(case["bond"]))
        delta = (
            abs(case["total"] - prior["core"] - prior["corr"])
            if prior is not None
            else float("nan")
        )
        print(
            f"{case['bond']:10.2f} {case['core']:17.12f} {case['corr']:17.12f} "
            f"{case['total']:17.12f} {delta:12.3e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tests/data/h2_df_reference_energies.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--bonds", type=float, nargs="+", default=list(BONDS))
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error(f"refusing to overwrite {args.output}; pass --overwrite")
    if args.threads < 1:
        parser.error("--threads must be positive")

    options = {**GF2_OPTIONS, "n_threads": args.threads}
    cases = [_run_case(bond, options) for bond in args.bonds]
    payload = {
        "schema_version": 2,
        "description": "Versioned H2/cc-pVDZ density-fitted electronic GF2 references",
        "metadata": _metadata(sys.argv, options),
        "input": {
            "geometry": "H 0 0 0; H 0 0 R",
            "units": "angstrom",
            "charge": 0,
            "multiplicity": 1,
            "psi4_options": PSI4_OPTIONS,
            "gf2_options": options,
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    old: dict[float, dict[str, Any]] = {}
    if args.compare:
        old_payload = json.loads(args.compare.read_text(encoding="utf-8"))
        old = {float(case["bond"]): case for case in old_payload["cases"]}
    _print_table(cases, old)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
