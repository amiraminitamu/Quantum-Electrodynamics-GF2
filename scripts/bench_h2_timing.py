#!/usr/bin/env python
# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Wall-time benchmark for gf2_run on h2.inp (median of several runs)."""

from __future__ import annotations

import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "h2.inp"
N_RUNS = 3


def _one_run() -> float:
    code = f"""
import time
import psi4
from gf2 import run_gf2
from gf2.input_parser import parse_input_file
from gf2.reporting import Reporter
from gf2.psi4_io import silence_psi4

silence_psi4()
job, mol, psi4_opts, gf2_opts = parse_input_file({INP!r})
rep = Reporter(job_name=job.name, echo=False)
t0 = time.perf_counter()
run_gf2(mol, psi4_opts, gf2_opts, reporter=rep, psi4_verbose=False)
print(f"WALL_SEC={{time.perf_counter() - t0:.4f}}")
"""
    env = {**subprocess.os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        proc.check_returncode()
    for line in proc.stdout.splitlines():
        if line.startswith("WALL_SEC="):
            return float(line.split("=", 1)[1])
    raise RuntimeError(f"no WALL_SEC in output:\n{proc.stdout}")


def main() -> None:
    print(f"Input: {INP}")
    print(f"Runs: {N_RUNS} (median reported)\n")
    times = []
    for i in range(N_RUNS):
        t = _one_run()
        times.append(t)
        print(f"  run {i + 1}/{N_RUNS}: {t:.2f} s", flush=True)
    med = statistics.median(times)
    print(f"\nMedian wall time: {med:.2f} s")


if __name__ == "__main__":
    main()
