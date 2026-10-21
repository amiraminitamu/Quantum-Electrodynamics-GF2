# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Input-parser coverage for projected LF-GF2."""

from __future__ import annotations

import pytest

pytest.importorskip("psi4")

import pytest

from gf2.input_parser import parse_input_text


_BASE = """
[job]
name = lf parser test

[molecule]
charge = 0
multiplicity = 1
symmetry = c1
H 0.0 0.0 0.0
H 0.0 0.0 0.74

[psi4]
basis = sto-3g

[gf2]
method = lf-gf2
eri_type = std
beta = 50

{cavity}
"""


def test_lf_gf2_requires_and_accepts_cavity_section():
    text = _BASE.format(
        cavity="""
[cavity]
units = atomic
mode1 = 0.2, 0.05, 0.0, 0.0, 1.0
"""
    )
    job, _, _, opts = parse_input_text(text)
    assert job.method == "lf-gf2"
    assert job.cavity is not None
    assert len(job.cavity.modes) == 1
    assert opts.eri_type == "std"


def test_lf_gf2_without_cavity_is_rejected():
    with pytest.raises(ValueError, match="cavity"):
        parse_input_text(_BASE.format(cavity=""))
