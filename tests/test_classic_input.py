# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Route-card and namelist decks."""

from __future__ import annotations

import pytest

from gf2.classic_input import classic_blocks, deck_kind


def test_route_card_becomes_the_same_sections() -> None:
    text = """%nproc=2
%output=examples/h2.out
#P UGF2/cc-pVDZ ERI=DF SCF=PK SYMMETRY=C1 UNITS=BOHR

Li unrestricted

0 2
Li 0.0 0.0 0.0

beta 5
mp2 false
mode1 = 0.10, 0.01, 0, 0, 1
"""
    blocks, route = classic_blocks(text)
    assert route.startswith("#P ")
    assert "UGF2" in route or "GF2" in route
    assert "name = Li unrestricted" in blocks["job"]
    assert "output = examples/h2.out" in blocks["job"]
    assert "charge = 0" in blocks["molecule"]
    assert "multiplicity = 2" in blocks["molecule"]
    assert "units = bohr" in blocks["molecule"]
    assert "symmetry = c1" in blocks["molecule"]
    assert "Li 0.0 0.0 0.0" in blocks["molecule"]
    assert "method = gf2" in blocks["gf2"]
    assert "spin_mode = unrestricted" in blocks["gf2"]
    assert "eri_type = df" in blocks["gf2"]
    assert "n_threads = 2" in blocks["gf2"]
    assert "beta = 5" in blocks["gf2"]
    assert "basis = cc-pvdz" in blocks["psi4"]
    assert "scf_type = pk" in blocks["psi4"]
    assert "mode1 = 0.10, 0.01, 0, 0, 1" in blocks["cavity"]
    assert deck_kind("[job]\nname = x\n") == "sections"


def test_namelist_deck() -> None:
    text = """$comment
H2 exact
$end

$molecule
0 1
H 0.0 0.0 0.0
H 0.0 0.0 0.74
$end

$rem
method lf-gf2
basis sto-3g
eri_type std
symmetry c1
$end

$cavity
mode1 0.2 0.05 0 0 1
$end
"""
    blocks, route = classic_blocks(text)
    assert blocks["job"].splitlines()[0] == "name = H2 exact"
    assert "method = lf-gf2" in blocks["gf2"]
    assert "basis = sto-3g" in blocks["psi4"]
    assert "H 0.0 0.0 0.74" in blocks["molecule"]
    assert "mode1 = 0.2 0.05 0 0 1" in blocks["cavity"]
    assert route.startswith("#P ")


def test_missing_route_is_rejected() -> None:
    with pytest.raises(ValueError, match="route card"):
        classic_blocks("%nproc=1\n\n0 1\nH 0 0 0\n")


def test_gf2_namelist_accepts_fortran_reals_and_on_off() -> None:
    text = """#P GF2/cc-pVDZ ERI=STD

H2 namelist

0 1
H 0.0 0.0 0.0
H 0.0 0.0 0.74

$GF2
 BETA = 50
 THRESH_ENERGY = 1.0D-6
 NEWTON = ON
 MP2 = OFF
$END

$CAVITY
 MODE1 = 1.0D-1, 2.0D-2, 0, 0, 1
$END
"""
    blocks, route = classic_blocks(text)
    assert route.startswith("#P GF2/cc-pVDZ")
    assert "beta = 50" in blocks["gf2"]
    assert "thresh_energy = 1.0e-6" in blocks["gf2"]
    assert "newton_on = ON" in blocks["gf2"]
    assert "mp2 = OFF" in blocks["gf2"]
    assert "mode1 = 1.0e-1, 2.0e-2, 0, 0, 1" in blocks["cavity"]


def test_section_deck_is_left_alone() -> None:
    assert classic_blocks("[job]\nname = kept\n") is None
