# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Installed-command behavior exercised through the module entry point."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    return subprocess.run(
        [sys.executable, "-m", "gf2.cli", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _input(maxiter: int = 30, *, cs: bool = True) -> str:
    method = "cs-gf2" if cs else "gf2"
    cavity = "\n[cavity]\nunits = atomic\nmode1 = 0.10, 0.01, 0, 0, 1\n" if cs else ""
    return f"""[job]
name = CLI smoke
[molecule]
charge = 0
multiplicity = 1
units = bohr
symmetry = c1
H 0 0 -0.6991986661
H 0 0  0.6991986661
[psi4]
basis = sto-3g
scf_type = pk
[gf2]
method = {method}
eri_type = std
beta = 20
n_ifreq = 30
mid_pt = 10
n_ifreq_full = 200
n_itime = 30
n_poly = 20
mp2 = false
n_threads = 1
n_fit_pts = 1000
maxiter_gf_diis = {maxiter}
maxiter_fock_diis = 8
{cavity}"""


def test_cli_help_version_and_invalid_input() -> None:
    help_result = _run("--help")
    assert help_result.returncode == 0
    assert "Lang-Firsov GF2" in help_result.stdout
    version = _run("--version")
    assert version.returncode == 0
    assert version.stdout.strip() == "qed-gf2 0.1.1"
    missing = _run("does-not-exist.inp")
    assert missing.returncode == 2
    assert "input file not found" in missing.stderr
    assert "Traceback" not in missing.stderr


def test_cli_cs_json_and_unconverged_exit_status(tmp_path: Path) -> None:
    pytest.importorskip("psi4")
    input_path = tmp_path / "cs.inp"
    input_path.write_text(_input(), encoding="utf-8")
    result = _run(str(input_path), "--json", "-")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["method"] == "cs-gf2"
    assert payload["eri_type"] == "std"
    assert payload["modes"][0]["frequency_hartree"] == pytest.approx(0.1)
    assert payload["summary"]["converged"] is True
    components = payload["summary"]["interaction_metadata"]["coupled_energy_components"]
    assert "C_ee" in components
    assert "C_eb" in components
    assert "E_ph" in components
    assert abs(components["sigma_g_pi_d_residual"]) < 1.0e-8

    unconverged_path = tmp_path / "unconverged.inp"
    unconverged_path.write_text(_input(1, cs=False), encoding="utf-8")
    unconverged = _run(str(unconverged_path), "--json", "-")
    assert unconverged.returncode == 3
    assert json.loads(unconverged.stdout)["summary"]["converged"] is False
    allowed = _run(str(unconverged_path), "--json", "-", "--allow-unconverged")
    assert allowed.returncode == 0


def test_cli_rejects_invalid_cavity_without_traceback(tmp_path: Path) -> None:
    pytest.importorskip("psi4")
    path = tmp_path / "bad-cavity.inp"
    path.write_text(_input().replace("0.10, 0.01", "0.0, 0.01"), encoding="utf-8")
    result = _run(str(path))
    assert result.returncode == 2
    assert "positive" in result.stderr
    assert "Traceback" not in result.stderr
