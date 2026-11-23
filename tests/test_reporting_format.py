# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Fixed-width Fortran-style log and .out archive."""

from __future__ import annotations

import io
from pathlib import Path

from gf2._version import __version__
from gf2.reporting import Reporter, RunSummary, format_mix_value, resolve_output_path


def _summary(**overrides: object) -> RunSummary:
    data = dict(
        job_name="H2 regression",
        core_gf2=-1.116759307397,
        corr_gf2=-0.034251998877,
        converged=True,
        wall_time_s=1.25,
        hf_energy=-1.116759307396,
        g_mix=None,
        sigma_mix=0.5,
    )
    data.update(overrides)
    return RunSummary(**data)  # type: ignore[arg-type]


def test_banner_keeps_the_star_box_and_the_green_mark() -> None:
    reporter = Reporter(job_name="H2 regression", echo=False, color=False)
    reporter.banner("electronic / coherent-state / lang-firsov")
    text = "\n".join(reporter.lines)
    assert "╭" not in text
    assert "▄" in text
    assert "THIS IS" in text
    assert "\033" not in text
    assert f"VERSION {__version__}" in text
    assert "PLEASE CITE ARXIV:2606.26076" in text
    assert "EXECUTION BEGUN" in text
    assert "ELECTRONIC / COHERENT-STATE / LANG-FIRSOV" in text
    stars = [line for line in reporter.lines if set(line) == {"*"}]
    assert stars
    assert all(len(line) == reporter.width for line in stars)
    boxed = [line for line in reporter.lines if line.startswith("* ") and line.endswith(" *")]
    assert boxed
    assert all(len(line) == reporter.width for line in boxed)


def test_green_mark_is_colored_on_screen_and_plain_in_the_archive() -> None:
    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = Tty()
    reporter = Reporter(echo=True, stream=stream, color=True)
    reporter.banner("Matsubara GF2")
    shown = stream.getvalue()
    archived = "\n".join(reporter.lines)
    assert "\033[92m" in shown
    assert "▄" in shown
    assert "\033" not in archived
    assert "▄" in archived


def test_route_card_and_orientation_table() -> None:
    reporter = Reporter(echo=False)
    reporter.route_card("#P GF2/cc-pVDZ ERI=DF")
    reporter.standard_orientation(
        (("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)),
        charge=0,
        multiplicity=1,
        units="angstrom",
    )
    text = "\n".join(reporter.lines)
    assert "#P GF2/cc-pVDZ ERI=DF" in text
    assert "ANGSTROMS" in text
    assert "0.740000" in text
    assert "Charge" in text and " = 0" in text
    assert all(len(line) <= reporter.width for line in reporter.lines if line.startswith("  "))


def test_sections_cycles_and_energies_use_fixed_fields() -> None:
    reporter = Reporter(echo=False)
    reporter.section("Reference")
    reporter.kv("Basis functions", 10)
    reporter.kv("LF-HF converged", True)
    reporter.outer_iter(2, g_mix=None, sigma_mix=0.25)
    reporter.inner_iter(4, 1.2e-5)
    reporter.energy_row("Total GF2", -1.151011306274, 3.2e-7)
    text = "\n".join(reporter.lines)
    energy = [line for line in reporter.lines if "DELTA" in line][-1]
    assert len(energy) <= reporter.width
    assert energy.startswith("  TOTAL GF2")
    assert "R E F E R E N C E" in text
    assert "Basis functions" in text and " = 10" in text
    assert " = T" in text
    assert "CYCLE    2" in text
    assert "G MIX = OFF" in text
    assert "SIGMA MIX = 0.25" in text
    assert "RMS(Fock) =    1.200D-05" in text
    assert "DELTA =    3.200D-07" in text
    assert "Δ" not in text
    assert format_mix_value(None) == "OFF"


def test_output_archive_matches_the_screen_transcript(tmp_path: Path) -> None:
    reporter = Reporter(job_name="H2 regression", echo=False)
    reporter.banner("Matsubara GF2")
    reporter.echo_input("[job]\nname = H2 regression\n$PUNCH\n")
    reporter.write_summary_block(_summary())
    destination = tmp_path / "h2.out"
    destination.write_text("previous run\n", encoding="utf-8")
    written = reporter.write_output_file(destination, summary=_summary())
    text = written.read_text(encoding="utf-8")
    assert text == "\n".join(reporter.lines).rstrip() + "\n"
    assert "previous run" not in text
    assert text.count("\n $PUNCH\n") == 1
    assert " ECORE    -1.116759307397\n" in text
    assert " ECORR    -0.034251998877\n" in text
    assert " ETOT     -1.151011306274\n" in text
    assert " CONV    T\n" in text
    assert "SIGMA MIX" in text or "Sigma mix" in text
    assert "SCF Done:  E(GF2) =" in text
    assert "NORMAL TERMINATION OF QED-GF2" in text
    assert "HARTREE" in text
    assert text.rstrip().endswith(f"OUTPUT STORED IN  {destination.resolve()}")
    assert text.endswith("\n")
    reporter.write_output_file(destination, summary=_summary())
    assert written.read_text(encoding="utf-8").count("NORMAL TERMINATION OF QED-GF2") == 1


def test_control_namelist_uses_fixed_fortran_fields() -> None:
    reporter = Reporter(echo=False)
    reporter.options_block(
        "GF2 options",
        {"beta": 50.0, "thresh_energy": 1.0e-6, "newton_on": True, "eri_type": "std"},
    )
    text = "\n".join(reporter.lines)
    assert "BETA" in text and "50.00000000" in text
    assert "THRESH_ENERGY" in text and "1.000000D-06" in text
    assert "NEWTON_ON" in text and " = ON" in text
    assert "ERI_TYPE" in text and " = STD" in text


def test_unconverged_punch_and_custom_output_path(tmp_path: Path) -> None:
    source = tmp_path / "water.inp"
    assert resolve_output_path(source) == tmp_path / "water.out"
    assert resolve_output_path(source, "results/water.log") == Path("results/water.log")
    assert resolve_output_path(tmp_path / "already.out").name == "already.out.out"

    reporter = Reporter(echo=False)
    summary = _summary(converged=False, spin_mode="unrestricted", n_alpha=2, n_beta=1)
    summary.mu_alpha = -0.1
    summary.mu_beta = -0.2
    reporter.write_summary_block(summary)
    text = "\n".join(reporter.lines)
    assert "QED-GF2 TERMINATED -- CONVERGENCE NOT ACHIEVED" in text
    assert " CONV    F\n" in text
    assert " NALPHA  2\n" in text
    assert " NBETA   1\n" in text
    assert "NOT CONVERGED" in text
