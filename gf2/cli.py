# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Installed command-line interface for QED-GF2."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import platform
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from ._version import __version__


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qed-gf2",
        description=(
            "Finite-temperature restricted/unrestricted electronic and coherent-state GF2; "
            "restricted zero-boson-projected Lang-Firsov GF2."
        ),
        epilog=(
            "The deck is a route card (#P METHOD/BASIS), a title, "
            "charge and multiplicity, Cartesian geometry, then a $GF2 "
            "namelist. $molecule/$rem namelists and the older [section] "
            "form are also accepted. Frequencies and couplings use atomic "
            "units. A fixed-width .out transcript is written beside the "
            "input (override with %output= or output=)."
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


def _fallback_route(job: Any, psi4_options: dict[str, Any], gf2_options: Any) -> str:
    method = str(job.method).upper()
    if getattr(gf2_options, "spin_mode", "restricted") == "unrestricted" and not method.startswith("U"):
        method = "U" + method
    basis = str(psi4_options.get("basis", "NONE"))
    parts = [f"#P {method}/{basis}", f"ERI={gf2_options.eri_type}"]
    if "scf_type" in psi4_options:
        parts.append(f"SCF={psi4_options['scf_type']}")
    return " ".join(parts).upper()


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

    from .input_parser import parse_input_file
    from .psi4_io import reopen_psi4_output, silence_psi4
    from .reporting import Reporter, resolve_output_path

    if args.verbose:
        reopen_psi4_output()
    else:
        silence_psi4()

    log_stream = sys.stderr if args.json == "-" else sys.stdout
    output_path = resolve_output_path(path, None)
    reporter = Reporter(job_name=path.stem, echo=True, stream=log_stream)
    reporter.banner("Electronic / coherent-state / Lang-Firsov")
    summary = None
    code = 2
    try:
        job, molecule, psi4_options, gf2_options = parse_input_file(path)
        reporter.job_name = job.name
        output_path = resolve_output_path(path, job.output_file)
        reporter.section("Job control")
        reporter.kv("Job name", job.name)
        reporter.kv("Input file", str(path))
        reporter.kv("Output file", str(output_path))
        reporter.kv("Method", str(job.method).upper())
        reporter.kv("ERI representation", str(gf2_options.eri_type).upper())
        reporter.kv("Number of threads", gf2_options.n_threads)
        reporter.route_card(job.route or _fallback_route(job, psi4_options, gf2_options))
        reporter.echo_input(path.read_text(encoding="utf-8"))
        if job.atoms:
            reporter.standard_orientation(
                job.atoms,
                charge=job.charge,
                multiplicity=job.multiplicity,
                units=job.units,
            )
        reporter.options_block("Psi4 options", psi4_options)
        reporter.options_block("GF2 options", _gf2_options_dict(gf2_options))

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
        summary.output_path = str(output_path)
        code = 0 if summary.converged or args.allow_unconverged else 3
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
    except Exception:
        reporter.warn("ABNORMAL TERMINATION -- SEE ERROR MESSAGE")
        raise
    finally:
        reporter.write_output_file(output_path, summary=summary)
    return code


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
