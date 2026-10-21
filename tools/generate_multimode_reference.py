#!/usr/bin/env python
# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Generate deterministic corrected H2 multimode CS-GF2 reference files."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import psi4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gf2 import run_gf2
from gf2.qed import CavityConfiguration, PhotonMode


OPTIONS = {
    "eri_type": "std",
    "beta": 20.0,
    "n_ifreq": 30,
    "mid_pt": 10,
    "n_ifreq_full": 200,
    "n_itime": 30,
    "n_poly": 20,
    "mp2": False,
    "maxiter_gf_diis": 5,
    "maxiter_fock_diis": 8,
    "n_threads": 1,
    "thresh_energy": 1.0e-5,
    "thresh_ne": 1.0e-6,
    "thresh_fock_rms": 1.0e-6,
    "newton_damping": 0.5,
    "maxiter_newton": 20,
    "n_fit_pts": 1000,
    "cs_hf_mix": 0.2,
}
PSI4_OPTIONS = {
    "basis": "sto-3g",
    "scf_type": "pk",
    "maxiter": 1000,
    "e_convergence": 1.0e-10,
    "d_convergence": 1.0e-10,
}


def molecule() -> psi4.core.Molecule:
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    return psi4.geometry("""0 1
H 0 0 -0.6991986661
H 0 0  0.6991986661
units bohr
no_reorient
no_com
symmetry c1""")


def run(modes: tuple[PhotonMode, ...]) -> dict:
    captures = {}
    core, corr, summary = run_gf2(
        molecule(),
        PSI4_OPTIONS,
        OPTIONS,
        method="cs-gf2",
        cavity=CavityConfiguration(modes),
        diagnostic_callback=lambda stage, data: captures.update({stage: data}),
    )
    stage = max(
        (s for s in captures if s.startswith("iteration_")), key=lambda s: int(s.split("_")[1])
    )
    data = {
        k: v
        for k, v in captures[stage].items()
        if isinstance(v, (np.ndarray, float, int, np.number))
    }
    data.update(
        core_energy=np.array(core),
        correlation_energy=np.array(corr),
        total_energy=np.array(core + corr),
        converged=np.array(summary.converged),
    )
    return data


def main() -> None:
    output = ROOT / "tests/reference_data/cs_multimode"
    output.mkdir(parents=True, exist_ok=True)
    z = PhotonMode(0.10, 0.01, (0.0, 0.0, 1.0))
    tilted = PhotonMode(0.15, 0.02, (2**-0.5, 0.0, 2**-0.5))
    zero = PhotonMode(0.20, 0.0, (1.0, 0.0, 0.0))
    cases = {
        "one_mode": (z,),
        "one_plus_zero": (z, zero),
        "two_mode": (z, tilted),
        "two_mode_permuted": (tilted, z),
    }
    for name, modes in cases.items():
        np.savez_compressed(output / f"{name}.npz", **run(modes))
    metadata = {
        "geometry_bohr": [[0.0, 0.0, -0.6991986661], [0.0, 0.0, 0.6991986661]],
        "basis": "sto-3g",
        "options": OPTIONS,
        "modes": {
            name: [
                {
                    "frequency": m.frequency,
                    "coupling": m.coupling,
                    "polarization": list(m.polarization),
                }
                for m in modes
            ]
            for name, modes in cases.items()
        },
        "versions": {"psi4": psi4.__version__, "numpy": np.__version__},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
