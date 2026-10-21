# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Terminal and file reporting for GF2 runs (presentation only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


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
        }


def format_mix_value(alpha: float | None) -> str:
    """Human-readable linear-mixing factor (``None`` → ``off``)."""
    if alpha is None:
        return "off"
    return f"{alpha:g}"


class Reporter:
    """Buffered, consistently styled log lines for stdout and optional files."""

    def __init__(
        self,
        *,
        job_name: str = "untitled",
        echo: bool = True,
        width: int = 72,
    ) -> None:
        self.job_name = job_name
        self.echo = echo
        self.width = width
        self.lines: list[str] = []
        self._started = datetime.now(timezone.utc)

    def _emit(self, text: str = "", *, to_console: bool = True) -> None:
        self.lines.append(text)
        if self.echo and to_console:
            print(text, flush=True)

    def banner(self, subtitle: str = "Matsubara GF2") -> None:
        """Print a polished GF2 startup banner."""
        width = max(self.width, 40)
        inner_width = width - 2
        content_width = inner_width - 2

        def truncate(text: str) -> str:
            text = str(text)

            if len(text) <= content_width:
                return text

            if content_width <= 1:
                return text[:content_width]

            return text[: content_width - 1] + "…"

        def row(text: str = "", alignment: str = "left") -> str:
            text = truncate(text)

            if alignment == "center":
                content = text.center(content_width)
            elif alignment == "right":
                content = text.rjust(content_width)
            else:
                content = text.ljust(content_width)

            return f"│ {content} │"

        title = "G F 2"

        self._emit("╭" + "─" * inner_width + "╮")
        self._emit(row(title, alignment="center"))
        self._emit(row(subtitle, alignment="center"))
        self._emit("├" + "─" * inner_width + "┤")
        self._emit(row(f"Job  ·  {self.job_name}"))
        self._emit("╰" + "─" * inner_width + "╯")
        self._emit()

    def section(self, title: str) -> None:
        self._emit()
        bar = "-" * min(len(title) + 4, self.width)
        self._emit(bar)
        self._emit(f"  {title}")
        self._emit(bar)

    def line(self, text: str = "") -> None:
        self._emit(text)

    def kv(self, label: str, value: Any, *, indent: int = 2) -> None:
        pad = " " * indent
        self._emit(f"{pad}{label:<26} {value}")

    def detail(self, text: str) -> None:
        self._emit(f"    {text}")

    def step(self, text: str) -> None:
        self._emit(f"  > {text}")

    def ok(self, text: str) -> None:
        self._emit(f"  * {text}")

    def warn(self, text: str) -> None:
        self._emit(f"  ! {text}")

    def mixing_kv(self, g_mix: float | None, sigma_mix: float | None, *, indent: int = 4) -> None:
        """Print configured ``G`` / ``Sigma`` linear-mixing factors."""
        pad = " " * indent
        self._emit(f"{pad}G mix     {format_mix_value(g_mix)}")
        self._emit(f"{pad}Sigma mix {format_mix_value(sigma_mix)}")

    def outer_iter(
        self,
        n: int,
        *,
        g_mix: float | None = None,
        sigma_mix: float | None = None,
    ) -> None:
        self._emit()
        self._emit(f"  >> outer iteration {n}")
        self.mixing_kv(g_mix, sigma_mix)

    def inner_iter(self, n: int, fock_rms: float) -> None:
        self.detail(f"inner {n:3d}  Fock RMS = {fock_rms:.3e}")

    def energy_row(self, label: str, value: float, delta: float | None = None) -> None:
        if delta is None:
            self.kv(label, f"{value: .12f}")
        else:
            self.kv(label, f"{value: .12f}  (Δ {delta:.3e})")

    def write_summary_block(self, summary: RunSummary) -> None:
        self.section("Results")
        self.kv("Core GF2 energy", f"{summary.core_gf2:.12f}")
        self.kv("GF2 correlation", f"{summary.corr_gf2:.12f}")
        self.kv("Total GF2 energy", f"{summary.total_gf2:.12f}")
        if summary.hf_energy is not None:
            self.kv("Reference HF energy", f"{summary.hf_energy:.12f}")
        if summary.mp2_psi4 is not None and summary.mp2_gf2_estimate is not None:
            diff = summary.mp2_gf2_estimate - summary.mp2_psi4
            self.kv("Psi4 MP2 correlation", f"{summary.mp2_psi4:.12f}")
            self.kv("GF2 MP2 estimate", f"{summary.mp2_gf2_estimate:.12f}")
            self.kv("MP2 difference", f"{diff:.12f}")
        status = "converged" if summary.converged else "not converged"
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
            self.kv(f"Time {name}", f"{seconds:.3f} s")
        for name, value in summary.convergence_diagnostics.items():
            self.kv(f"Residual {name}", f"{value:.3e}")
        self.kv("Wall time", f"{summary.wall_time_s:.2f} s")

    def write_output_file(
        self,
        path: str | Path,
        *,
        summary: RunSummary,
        psi4_options: Mapping[str, Any] | None = None,
        gf2_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Write a formatted transcript (log + summary) to disk."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        finished = datetime.now(timezone.utc)
        with out.open("w", encoding="utf-8") as fh:
            fh.write(self._file_header(summary, finished))
            if psi4_options:
                fh.write(self._options_block("Psi4 options", psi4_options))
            if gf2_options:
                fh.write(self._options_block("GF2 options", gf2_options))
            fh.write("\n--- run log ---\n\n")
            fh.write("\n".join(self.lines))
            fh.write("\n\n--- summary ---\n\n")
            fh.write(self._plain_summary(summary))
        self.ok(f"Wrote results to {out.resolve()}")

    def _file_header(self, summary: RunSummary, finished: datetime) -> str:
        lines = [
            "GF2 calculation report",
            "=" * 72,
            f"Job name     : {summary.job_name}",
            f"Started (UTC): {self._started.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Finished(UTC): {finished.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if summary.input_path:
            lines.append(f"Input file   : {summary.input_path}")
        if summary.output_path:
            lines.append(f"Output file  : {summary.output_path}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _options_block(title: str, opts: Mapping[str, Any]) -> str:
        lines = [f"{title}", "-" * len(title)]
        for key in sorted(opts.keys()):
            lines.append(f"  {key:<22} {opts[key]}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _plain_summary(summary: RunSummary) -> str:
        lines = [
            f"Core GF2 energy     {summary.core_gf2:20.12f}",
            f"GF2 correlation     {summary.corr_gf2:20.12f}",
            f"Total GF2 energy    {summary.total_gf2:20.12f}",
            f"SCF converged       {summary.converged}",
            f"G mix               {format_mix_value(summary.g_mix):>20}",
            f"Sigma mix           {format_mix_value(summary.sigma_mix):>20}",
            f"Wall time (s)       {summary.wall_time_s:20.6f}",
        ]
        if summary.hf_energy is not None:
            lines.append(f"HF reference        {summary.hf_energy:20.12f}")
        return "\n".join(lines) + "\n"
