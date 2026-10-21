#!/usr/bin/env python
# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Generate molecular exact-versus-DF CS-GF2 references and timing data."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gf2 import run_gf2  # noqa: E402
from gf2.qed import CavityConfiguration, PhotonMode  # noqa: E402
from gf2.reporting import Reporter  # noqa: E402
from tests.cs_df_helpers import (  # noqa: E402
    COMMON_OPTIONS,
    ONE_Z,
    TWO_TILTED,
    WATER_MODES,
    molecule,
)

CASES = (
    ("h2_sto3g_one", "h2", "sto-3g", ONE_Z),
    ("h2_sto3g_two", "h2", "sto-3g", TWO_TILTED),
    ("h2_ccpvdz_one", "h2", "cc-pvdz", ONE_Z),
    ("h2_ccpvdz_two", "h2", "cc-pvdz", TWO_TILTED),
    ("lih_sto3g_one", "lih", "sto-3g", ONE_Z),
    ("h2o_sto3g_two", "h2o", "sto-3g", WATER_MODES),
)
ARRAY_KEYS = (
    "density",
    "fock",
    "z",
    "Sigma_ee_tau",
    "Sigma_eb_tau",
    "Pi_iv",
    "D_iv",
)


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _mode_dict(mode: PhotonMode) -> dict[str, Any]:
    return {
        "frequency": mode.frequency,
        "coupling": mode.coupling,
        "polarization": list(mode.polarization),
    }


def _run(
    system: str, basis: str, eri_type: str, modes: tuple[PhotonMode, ...]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    captures: dict[str, dict[str, Any]] = {}
    p4 = {
        "basis": basis,
        "scf_type": "pk",
        "maxiter": 1000,
        "e_convergence": 1.0e-10,
        "d_convergence": 1.0e-10,
    }
    options = {**COMMON_OPTIONS, "eri_type": eri_type}
    tracemalloc.start()
    tracemalloc.reset_peak()
    core, corr, summary = run_gf2(
        molecule(system),
        p4,
        options,
        reporter=Reporter(echo=False),
        method="cs-gf2",
        cavity=CavityConfiguration(modes),
        diagnostic_callback=lambda stage, data: captures.update({stage: data}),
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    iteration_names = sorted(
        (name for name in captures if name.startswith("iteration_")),
        key=lambda name: int(name.split("_")[1]),
    )
    final = captures[iteration_names[-1]]
    previous = captures[iteration_names[-2]] if len(iteration_names) > 1 else final
    reference = captures["reference"]
    interaction = summary.interaction_metadata
    nao = int(reference["nbf"])
    arrays = {key: np.asarray(final[key]) for key in ARRAY_KEYS}
    arrays.update(
        {
            "chemical_potential": np.asarray(final["mu"]),
            "particle_number": np.asarray(final["particle_number"]),
            "core_energy": np.asarray(core),
            "correlation_energy": np.asarray(corr),
        }
    )
    result = {
        "core": core,
        "correlation": corr,
        "total": core + corr,
        "hf_energy": summary.hf_energy,
        "converged": summary.converged,
        "iterations": len(iteration_names),
        "chemical_potential": float(final["mu"]),
        "particle_number": float(final["particle_number"]),
        "coherent_shifts": np.asarray(final["z"]).tolist(),
        "density_residual_fro": float(np.linalg.norm(final["density"] - previous["density"])),
        "self_energy_residual_fro": float(
            np.linalg.norm(final["Sigma_total_tau"] - previous["Sigma_total_tau"])
        ),
        "wall_time_s": summary.wall_time_s,
        "component_timings_s": summary.component_timings_s,
        "python_traced_peak_bytes": peak,
        "nao": nao,
        "interaction": interaction,
        "explicit_four_index_bytes": nao**4 * np.dtype(float).itemsize,
    }
    if eri_type == "df":
        bundle = reference["df"]
        result["stored_factor_bytes"] = int(bundle.Qpq.nbytes + bundle.f_Qpq.nbytes)
    else:
        result["stored_factor_bytes"] = 0
        result["allocated_eri_bytes"] = int(reference["eri"].nbytes)
    return result, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tests/reference_data/cs_df")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        parser.error(f"refusing to overwrite nonempty {args.output}; pass --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for case_id, system, basis, modes in CASES:
        methods = {}
        final_arrays = {}
        for eri_type in ("std", "df"):
            methods[eri_type], final_arrays[eri_type] = _run(system, basis, eri_type, modes)
            np.savez_compressed(args.output / f"{case_id}_{eri_type}.npz", **final_arrays[eri_type])
        differences = {
            "core": methods["df"]["core"] - methods["std"]["core"],
            "correlation": methods["df"]["correlation"] - methods["std"]["correlation"],
            "total": methods["df"]["total"] - methods["std"]["total"],
        }
        for key in ARRAY_KEYS:
            differences[f"max_abs_{key}"] = float(
                np.max(np.abs(final_arrays["df"][key] - final_arrays["std"][key]))
            )
        rows.append(
            {
                "id": case_id,
                "system": system,
                "basis": basis,
                "modes": [_mode_dict(mode) for mode in modes],
                "exact": methods["std"],
                "df": methods["df"],
                "df_minus_exact": differences,
            }
        )

    payload = {
        "schema_version": 1,
        "description": "Executed exact-versus-density-fitted multimode CS-GF2 references",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "git_worktree_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "platform": platform.platform(),
        "packages": {
            name: _version(name) for name in ("psi4", "numpy", "scipy", "opt_einsum", "qcelemental")
        },
        "gf2_options": COMMON_OPTIONS,
        "cases": rows,
        "memory_note": "python_traced_peak_bytes excludes untracked native Psi4 allocations",
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("case | exact total | DF total | DF-exact | exact s | DF s | rank")
    for row in rows:
        print(
            f"{row['id']} | {row['exact']['total']:.12f} | "
            f"{row['df']['total']:.12f} | {row['df_minus_exact']['total']:.3e} | "
            f"{row['exact']['wall_time_s']:.3f} | {row['df']['wall_time_s']:.3f} | "
            f"{row['df']['interaction']['effective_factor_rank']}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
