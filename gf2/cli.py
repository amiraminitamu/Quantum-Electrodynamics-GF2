# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Installed command-line interface for QED-GF2."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import os
import platform
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from ._version import __version__


def _banner(*, color: bool = False) -> str:
    width = 74
    lines = (
        f"Q E D - G F 2   |   Version {__version__}",
        "Many-Body Green's Functions for Molecular Quantum Electrodynamics",
        "",
        "Electronic GF2  /  Coherent State  /  Projected Lang-Firsov",
        "",
        "Amirhosein Amini  |  Jaime Cerda",
        "Leopoldo Mejía  |  Arkajit Mandal",
        "2026  |  MIT License",
    )
    border = "+" + "-" * width + "+"
    frame = "\n".join((border, *(f"|{line:^{width}}|" for line in lines), border))
    art = (
        "    ▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄",
        "  ▄▀░▒▒▓▓▓█▐  ▀▄░▒▒▒▓▓▓█▐▄░▒▓█▓▓▒▒░▀▄",
        " ▐░░▄▀▀▀▀▀▀▀    ▐░▒▄▀▀▀▄▀ ▀▀▀▀▀▀▀▄░░▐",
        " ▌▓ ▌▄▄▄▄▄▄▄▄   ▐▓ ▐▄▄▄     ▄▀▀▀▀   ▌",
        "▐▓▒▐ ▐▓▒░▓▓██   ▌▒▓▓▒░▌   ▄▀▒▓▓█▄▄▄▀",
        " ▐▒░▌▐▀▀▓░░▒▓   ▌░ ▐▀▀   ▐▒░░▒▐▀",
        " ▐░░▀▄▄▄▒░ ░▐   ▌  ▀▄    ▓  ░ ▀▄▄▄▄▄",
        "  ▀▄░░ ▄▄   ▐▌▄▀    ▀▄   ▀▓▄▄  ░░▒██▌",
        "    ▀▀▀▀ ▀▀▀▀  ▀▀▀▀▀▀       ▀▀▀▀▀▀▀▀▀",
    )
    indent = " " * ((width + 2 - max(map(len, art))) // 2)
    artwork = "\n".join(indent + line for line in art)
    if color:
        artwork = f"\033[92m{artwork}\033[0m"
    return f"{frame}\n\n{'THIS IS':^{width + 2}}\n{artwork}"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qed-gf2",
        description=(
            "Restricted finite-temperature electronic, coherent-state, and "
            "zero-boson-projected Lang-Firsov GF2."
        ),
        epilog=(
            "Input uses [job], [molecule], [psi4], [gf2], and optional [cavity] "
            "sections. Frequencies and couplings "
            "use atomic units."
        ),
    )
    parser.add_argument("input", nargs="?", help="Path to a QED-GF2 .inp file.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--environment",
        action="store_true",
        help="Print package, runtime, and BLAS information for bug reports.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show Psi4 library output.")
    parser.add_argument(
        "--allow-unconverged",
        action="store_true",
        help="Return status zero even when the GF2 iteration does not converge.",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Write a structured result to PATH, or use '-' for standard output.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show Python tracebacks for unexpected failures.",
    )
    return parser


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _environment() -> dict[str, Any]:
    import numpy as np

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        np.show_config()
    try:
        import psi4

        psi4_version = str(psi4.__version__)
    except ImportError:
        psi4_version = "not-installed"
    return {
        "qed_gf2": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            "psi4": psi4_version,
            "numpy": np.__version__,
            "scipy": _version("scipy"),
            "qcelemental": _version("qcelemental"),
            "opt_einsum": _version("opt_einsum"),
        },
        "numpy_configuration": output.getvalue().strip(),
    }


def _gf2_options_dict(opts: Any) -> dict[str, Any]:
    data = {
        field.name: getattr(opts, field.name) for field in fields(opts) if field.name != "extra"
    }
    data.update(opts.extra)
    return data


def _write_json(path: str, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path == "-":
        sys.stdout.write(text)
    else:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")


def _run(args: argparse.Namespace) -> int:
    if args.environment:
        print(json.dumps(_environment(), indent=2, sort_keys=True))
        return 0
    if not args.input:
        raise ValueError("an input file is required (use --help for syntax)")

    path = Path(args.input).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"input file not found: {path}")

    banner_stream = sys.stderr if args.json == "-" else sys.stdout
    color = (
        banner_stream.isatty()
        and "NO_COLOR" not in os.environ
        and os.environ.get("TERM") != "dumb"
    )
    print(_banner(color=color), file=banner_stream, flush=True)

    from .input_parser import parse_input_file
    from .psi4_io import reopen_psi4_output, silence_psi4
    from .reporting import Reporter

    if args.verbose:
        reopen_psi4_output()
    else:
        silence_psi4()

    job, molecule, psi4_options, gf2_options = parse_input_file(path)
    reporter = Reporter(job_name=job.name, echo=args.json != "-")
    reporter.banner("Electronic, CS-QED, and projected LF-QED GF2")
    reporter.kv("Input file", str(path))
    reporter.kv("Method", job.method)
    reporter.kv("ERI representation", gf2_options.eri_type)

    from .driver_theory import run_gf2

    _, _, summary = run_gf2(
        molecule,
        psi4_options,
        gf2_options,
        reporter=reporter,
        psi4_verbose=args.verbose,
        method=job.method,
        cavity=job.cavity,
    )

    summary.input_path = str(path)
    summary.output_path = job.output_file

    if job.output_file:
        reporter.write_output_file(
            job.output_file,
            summary=summary,
            psi4_options=psi4_options,
            gf2_options=_gf2_options_dict(gf2_options),
        )
    if args.json:
        modes = (
            []
            if job.cavity is None
            else [
                {
                    "frequency_hartree": mode.frequency,
                    "coupling_au": mode.coupling,
                    "polarization": list(mode.polarization),
                }
                for mode in job.cavity.modes
            ]
        )
        _write_json(
            args.json,
            {
                "schema_version": 1,
                "qed_gf2_version": __version__,
                "input": str(path),
                "method": job.method,
                "eri_type": gf2_options.eri_type,
                "modes": modes,
                "summary": summary.as_dict(),
            },
        )

    reporter.line()
    if summary.converged:
        reporter.ok("Calculation converged")
        return 0
    reporter.warn("Calculation did not converge")
    return 0 if args.allow_unconverged else 3


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except Exception as exc:
        if args.debug:
            raise
        print(f"qed-gf2: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
