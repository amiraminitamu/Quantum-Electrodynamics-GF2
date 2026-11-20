# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Fixed-width terminal log and ``.out`` archive for GF2 runs.

Presentation only. Numeric results are formatted, never recomputed.
The screen transcript and the output file are the same text: an
80-column log in the style of a Fortran quantum-chemistry program,
overwritten on each run and closed with a ``$PUNCH`` record.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, TextIO

from ._version import __version__

_MONTHS = (
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
)
_DAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")


@dataclass
class RunSummary:
    """Final numbers collected after a :func:`~gf2.driver.run_gf2` call."""

    job_name: str
    core_gf2: float
    corr_gf2: float
    converged: bool
    wall_time_s: float
    input_path: str | None = None
    output_path: str | None = None
    hf_energy: float | None = None
    mp2_psi4: float | None = None
    mp2_gf2_estimate: float | None = None
    g_mix: float | None = None
    sigma_mix: float | None = None
    interaction_metadata: dict[str, Any] = field(default_factory=dict)
    component_timings_s: dict[str, float] = field(default_factory=dict)
    convergence_diagnostics: dict[str, float] = field(default_factory=dict)
    spin_mode: str = "restricted"
    n_alpha: int | None = None
    n_beta: int | None = None
    mu_alpha: float | None = None
    mu_beta: float | None = None
    trace_p_alpha_s: float | None = None
    trace_p_beta_s: float | None = None

    @property
    def total_gf2(self) -> float:
        return self.core_gf2 + self.corr_gf2

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for reproducibility records."""
        return {
            "job_name": self.job_name,
            "core_gf2": self.core_gf2,
            "corr_gf2": self.corr_gf2,
            "total_gf2": self.total_gf2,
            "converged": self.converged,
            "wall_time_s": self.wall_time_s,
            "hf_energy": self.hf_energy,
            "mp2_psi4": self.mp2_psi4,
            "mp2_gf2_estimate": self.mp2_gf2_estimate,
            "g_mix": self.g_mix,
            "sigma_mix": self.sigma_mix,
            "interaction_metadata": dict(self.interaction_metadata),
            "component_timings_s": dict(self.component_timings_s),
            "convergence_diagnostics": dict(self.convergence_diagnostics),
            "spin_mode": self.spin_mode,
            "n_alpha": self.n_alpha,
            "n_beta": self.n_beta,
            "mu_alpha": self.mu_alpha,
            "mu_beta": self.mu_beta,
            "trace_p_alpha_s": self.trace_p_alpha_s,
            "trace_p_beta_s": self.trace_p_beta_s,
        }


def format_mix_value(alpha: float | None) -> str:
    """Linear-mixing factor as printed in the log (``None`` → ``OFF``)."""
    if alpha is None:
        return "OFF"
    return f"{alpha:g}"


def resolve_output_path(input_file: str | Path, requested: str | None = None) -> Path:
    """Return the ``.out`` archive path for a calculation.

    An explicit ``output`` / ``output_file`` entry is used as given.
    Otherwise the archive sits beside the input, with the suffix replaced
    by ``.out`` (``h2.inp`` → ``h2.out``). A path that would overwrite the
    input itself gains an extra ``.out`` suffix.
    """
    source = Path(input_file).expanduser()
    if requested is not None and str(requested).strip():
        return Path(str(requested).strip()).expanduser()
    dest = source.with_suffix(".out")
    try:
        same_file = dest.resolve() == source.resolve()
    except OSError:
        same_file = dest == source
    if same_file:
        return source.with_name(source.name + ".out")
    return dest


def _stamp(when: datetime) -> str:
    local = when.astimezone()
    zone = local.tzname() or "UTC"
    return (
        f"{_DAYS[local.weekday()]} {local.day:02d}-{_MONTHS[local.month - 1]}-{local.year}"
        f"  AT  {local.hour:02d}:{local.minute:02d}:{local.second:02d} {zone}"
    )


def _spaced_heading(title: str) -> str:
    words = str(title).upper().split()

    def space_token(token: str) -> str:
        if any(character.isdigit() for character in token):
            return token
        return " ".join(token)

    return "  ".join(space_token(word) for word in words)


_GF2_MARK = (
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

_ATOMIC_NUMBERS = {
    symbol: number
    for number, symbol in enumerate(
        (
            "X H HE LI BE B C N O F NE NA MG AL SI P S CL AR K CA SC TI V CR MN "
            "FE CO NI CU ZN GA GE AS SE BR KR"
        ).split()
    )
}


def _tty_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def _fortran_float(value: float, width: int = 12, precision: int = 3) -> str:
    """Fortran double-precision field, with a D exponent."""
    return f"{value:{width}.{precision}E}".replace("E", "D")


def _format_printed_value(value: Any) -> str:
    if isinstance(value, bool):
        return "T" if value else "F"
    if value is None:
        return "NONE"
    return str(value)


class Reporter:
    """80-column log buffer mirrored to a stream and, later, a ``.out`` file."""

    def __init__(
        self,
        *,
        job_name: str = "untitled",
        echo: bool = True,
        width: int = 80,
        stream: TextIO | None = None,
        color: bool | None = None,
    ) -> None:
        self.job_name = job_name
        self.echo = echo
        self.width = max(int(width), 72)
        self.stream = sys.stdout if stream is None else stream
        self.color = _tty_color(self.stream) if color is None else color
        self.lines: list[str] = []
        self._sections: set[str] = set()
        self._punched = False
        self._terminated = False
        self._started = datetime.now().astimezone()

    def _emit(self, text: str = "", *, display: str | None = None) -> None:
        self.lines.append(text)
        if self.echo:
            print(text if display is None else display, file=self.stream, flush=True)

    def _rule(self, char: str) -> str:
        return char * self.width

    def _star_box(self, rows: list[str]) -> None:
        inner = self.width - 4
        self._emit(self._rule("*"))
        for row in rows:
            text = row[:inner]
            self._emit(f"* {text.center(inner)} *")
        self._emit(self._rule("*"))

    def ensure_banner(self, subtitle: str = "Matsubara GF2") -> None:
        """Print the program header once, when the log is still empty."""
        if not self.lines:
            self.banner(subtitle)

    def banner(self, subtitle: str = "Matsubara GF2") -> None:
        """Print the QED-GF2 program header."""
        self._star_box(
            [
                "",
                "Q E D - G F 2",
                f"VERSION {__version__}",
                "",
                "MANY-BODY GREEN'S FUNCTION THEORY",
                "FOR AB INITIO MOLECULAR QUANTUM ELECTRODYNAMICS",
                "",
                " ".join(str(subtitle).upper().split()),
                "",
                "A. AMINI    J. CERDA    L. MEJIA    A. MANDAL",
                "PLEASE CITE ARXIV:2606.26076",
                "",
            ]
        )
        self._emit("")
        self._emit("THIS IS".center(self.width))
        self._emit("")
        for row in _GF2_MARK:
            plain = row.center(self.width) if len(row) < self.width else row
            if self.color:
                self._emit(plain, display=f"\033[92m{plain}\033[0m")
            else:
                self._emit(plain)
        self._emit("")
        self._emit(f"  EXECUTION BEGUN  {_stamp(self._started)}")
        self.kv("Host name", platform.node() or "UNKNOWN")
        self.kv("Job name", self.job_name)
        self._emit("")

    def section(self, title: str) -> None:
        key = " ".join(str(title).upper().split())
        self._sections.add(key)
        heading = _spaced_heading(title)
        if len(heading) > self.width - 2:
            heading = key
        self._emit("")
        self._emit(self._rule("-"))
        self._emit(heading.center(self.width))
        self._emit(self._rule("-"))

    def line(self, text: str = "") -> None:
        self._emit(text)

    def kv(self, label: str, value: Any, *, indent: int = 2) -> None:
        name = " ".join(str(label).split())
        text = _format_printed_value(value)
        pad = " " * indent
        field_width = 42
        if len(name) > field_width:
            self._emit(f"{pad}{name}")
            self._emit(f"{pad}{'':<{field_width}} = {text}")
            return
        self._emit(f"{pad}{name:<{field_width}} = {text}")

    def detail(self, text: str) -> None:
        self._emit(f"       {text}")

    def step(self, text: str) -> None:
        self._emit(f"  ... {text}")

    def ok(self, text: str) -> None:
        self._emit(f" **** {text}")

    def warn(self, text: str) -> None:
        self._emit(f" !!!  {text}")

    def route_card(self, route: str) -> None:
        """Echo the route card."""
        text = " ".join(str(route).split())
        if not text.startswith("#"):
            text = "#P " + text
        self.section("Route card")
        words = text.split()
        current = ""
        for word in words:
            piece = word if not current else f"{current} {word}"
            if current and len(piece) > self.width - 2:
                self._emit(f"  {current}")
                current = word
            else:
                current = piece
        if current:
            self._emit(f"  {current}")

    def standard_orientation(
        self,
        atoms: list[tuple[str, float, float, float]] | tuple[tuple[str, float, float, float], ...],
        *,
        charge: int,
        multiplicity: int,
        units: str = "angstrom",
    ) -> None:
        """Print the input geometry in a fixed Cartesian table."""
        label = "BOHR" if str(units).strip().lower() in {"bohr", "au", "a.u."} else "ANGSTROMS"
        self.section("Input orientation")
        self.kv("Coordinate units", label)
        self._emit("  CENTER ATOMIC ATOMIC               COORDINATES")
        self._emit("  NUMBER NUMBER SYMBOL       X            Y            Z")
        self._emit("  " + "-" * 64)
        for number, (symbol, x_coord, y_coord, z_coord) in enumerate(atoms, start=1):
            atomic_number = _ATOMIC_NUMBERS.get(symbol.upper(), 0)
            self._emit(
                f"  {number:6d} {atomic_number:6d}   {symbol:>2}"
                f"   {x_coord:12.6f} {y_coord:12.6f} {z_coord:12.6f}"
            )
        self._emit("  " + "-" * 64)
        self.kv("Charge", int(charge))
        self.kv("Multiplicity", int(multiplicity))

    def echo_input(self, text: str) -> None:
        """Echo the input deck verbatim, as a Fortran program echoes its cards."""
        self.section("Input deck")
        for raw in text.splitlines():
            self._emit(raw.rstrip())

    def options_block(self, title: str, opts: Mapping[str, Any]) -> None:
        self.section(title)
        for key in sorted(opts):
            self.kv(str(key), opts[key])

    def mixing_kv(self, g_mix: float | None, sigma_mix: float | None, *, indent: int = 2) -> None:
        """Print configured ``G`` / ``Sigma`` linear-mixing factors."""
        self.kv("G mix", format_mix_value(g_mix), indent=indent)
        self.kv("Sigma mix", format_mix_value(sigma_mix), indent=indent)

    def outer_iter(
        self,
        n: int,
        *,
        g_mix: float | None = None,
        sigma_mix: float | None = None,
    ) -> None:
        self._emit("")
        self._emit(self._rule("-"))
        self._emit(
            f"  CYCLE {n:4d}            G MIX = {format_mix_value(g_mix):<8}"
            f"SIGMA MIX = {format_mix_value(sigma_mix)}"
        )
        self._emit(self._rule("-"))

    def inner_iter(self, n: int, fock_rms: float) -> None:
        self._emit(f"  Iter {n:4d}   RMS(Fock) = {_fortran_float(fock_rms)}")

    def energy_row(self, label: str, value: float, delta: float | None = None) -> None:
        name = " ".join(str(label).upper().split())
        text = f"  {name:<22}{value:16.12f}"
        if delta is not None:
            text += f"    DELTA = {_fortran_float(delta)}"
        self._emit(text)

    def _punch_lines(self, summary: RunSummary) -> list[str]:
        def rec(key: str, value: str) -> str:
            return f" {key:<8}{value}"

        rows = [
            "",
            " $PUNCH",
            rec("JOB", summary.job_name),
            rec("ECORE", f"{summary.core_gf2:16.12f}"),
            rec("ECORR", f"{summary.corr_gf2:16.12f}"),
            rec("ETOT", f"{summary.total_gf2:16.12f}"),
        ]
        if summary.hf_energy is not None:
            rows.append(rec("EHF", f"{summary.hf_energy:16.12f}"))
        rows.append(rec("CONV", "T" if summary.converged else "F"))
        rows.append(rec("SPIN", summary.spin_mode.upper()))
        if summary.spin_mode == "unrestricted":
            if summary.n_alpha is not None:
                rows.append(rec("NALPHA", str(summary.n_alpha)))
            if summary.n_beta is not None:
                rows.append(rec("NBETA", str(summary.n_beta)))
            if summary.mu_alpha is not None:
                rows.append(rec("MUA", f"{summary.mu_alpha:16.10f}"))
            if summary.mu_beta is not None:
                rows.append(rec("MUB", f"{summary.mu_beta:16.10f}"))
        rows.append(rec("WALL", f"{summary.wall_time_s:16.6f}"))
        rows.append(" $END")
        return rows

    def _emit_punch(self, summary: RunSummary) -> None:
        if self._punched:
            return
        for row in self._punch_lines(summary):
            self._emit(row)
        self._punched = True

    def _emit_termination(self, summary: RunSummary) -> None:
        if self._terminated:
            return
        for row in self._termination_lines(summary):
            self._emit(row)
        self._terminated = True

    def _termination_lines(self, summary: RunSummary) -> list[str]:
        if summary.converged:
            message = "NORMAL TERMINATION OF QED-GF2"
        else:
            message = "QED-GF2 TERMINATED -- CONVERGENCE NOT ACHIEVED"
        finished = f"FINISHED  {_stamp(datetime.now().astimezone())}"
        inner = self.width - 4
        return [
            "",
            self._rule("*"),
            f"* {message.center(inner)} *",
            f"* {finished.center(inner)} *",
            self._rule("*"),
        ]

    def _has_heading(self, title: str) -> bool:
        key = " ".join(str(title).upper().split())
        return key in self._sections

    def write_summary_block(self, summary: RunSummary) -> None:
        self.section("Final results")
        self.kv("Core GF2 energy", f"{summary.core_gf2:20.12f} HARTREE")
        self.kv("GF2 correlation", f"{summary.corr_gf2:20.12f} HARTREE")
        self.kv("Total GF2 energy", f"{summary.total_gf2:20.12f} HARTREE")
        if summary.spin_mode == "unrestricted":
            self.kv("Spin mode", "UNRESTRICTED")
            if summary.n_alpha is not None:
                self.kv("N alpha", summary.n_alpha)
            if summary.n_beta is not None:
                self.kv("N beta", summary.n_beta)
            if summary.mu_alpha is not None:
                self.kv("mu alpha", f"{summary.mu_alpha:.10f}")
            if summary.mu_beta is not None:
                self.kv("mu beta", f"{summary.mu_beta:.10f}")
            if summary.trace_p_alpha_s is not None:
                self.kv("Tr[P_alpha S]", f"{summary.trace_p_alpha_s:.8f}")
            if summary.trace_p_beta_s is not None:
                self.kv("Tr[P_beta S]", f"{summary.trace_p_beta_s:.8f}")
        if summary.hf_energy is not None:
            self.kv("Reference HF energy", f"{summary.hf_energy:20.12f} HARTREE")
        if summary.mp2_psi4 is not None and summary.mp2_gf2_estimate is not None:
            diff = summary.mp2_gf2_estimate - summary.mp2_psi4
            self.kv("Psi4 MP2 correlation", f"{summary.mp2_psi4:20.12f} HARTREE")
            self.kv("GF2 MP2 estimate", f"{summary.mp2_gf2_estimate:20.12f} HARTREE")
            self.kv("MP2 difference", f"{diff:20.12f} HARTREE")
        status = "CONVERGED" if summary.converged else "NOT CONVERGED"
        self.kv("SCF status", status)
        self.kv("G mix", format_mix_value(summary.g_mix))
        self.kv("Sigma mix", format_mix_value(summary.sigma_mix))
        if summary.interaction_metadata:
            self.kv("Interaction mode", summary.interaction_metadata["interaction_mode"])
            if summary.interaction_metadata["interaction_mode"] == "df":
                self.kv(
                    "Effective factor rank",
                    summary.interaction_metadata["effective_factor_rank"],
                )
                self.kv(
                    "QED factor channels",
                    summary.interaction_metadata["qed_low_rank_channels"],
                )
        for name, seconds in summary.component_timings_s.items():
            self.kv(f"Time {name}", f"{seconds:12.3f} SECONDS")
        for name, value in summary.convergence_diagnostics.items():
            self.kv(f"Residual {name}", _fortran_float(value))
        self.kv("Wall time", f"{summary.wall_time_s:12.2f} SECONDS")
        done = f"SCF Done:  E(GF2) = {summary.total_gf2:20.12f} a.u."
        self._emit("")
        self._emit(self._rule("*"))
        self._emit(done.center(self.width))
        self._emit(self._rule("*"))
        self._emit_punch(summary)
        self._emit_termination(summary)

    def write_output_file(
        self,
        path: str | Path,
        *,
        summary: RunSummary | None = None,
        psi4_options: Mapping[str, Any] | None = None,
        gf2_options: Mapping[str, Any] | None = None,
    ) -> Path:
        """Overwrite ``path`` with the screen transcript.

        The archive is the log itself: plain text, one record per line,
        replaced on every run. Optional option maps and a punch record are
        appended only when the transcript does not already contain them.
        """
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        if psi4_options and not self._has_heading("Psi4 options"):
            self.options_block("Psi4 options", psi4_options)
        if gf2_options and not self._has_heading("GF2 options"):
            self.options_block("GF2 options", gf2_options)
        if summary is not None:
            self._emit_punch(summary)
            self._emit_termination(summary)
        stored = f" OUTPUT STORED IN  {out.resolve()}"
        if stored not in self.lines:
            self._emit("")
            self._emit(stored)
        out.write_text("\n".join(self.lines).rstrip() + "\n", encoding="utf-8")
        return out
