# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
from __future__ import annotations

import numpy as np
import pytest

from gf2.pes_workflow import distance_tag, render_diatomic_input


TEMPLATE = """[job]
name = LiH test

[molecule]
charge = 0
multiplicity = 1
symmetry = c1
units = angstrom
Li 0.0 0.0 0.0
H  0.0 0.0 1.600000

[gf2]
method = lf-gf2
beta = 200

[cavity]
units = atomic
mode1 = 0.12086, 0.05, 0.0, 0.0, 1.0
"""


def test_render_diatomic_input_preserves_axis_and_updates_method() -> None:
    text = render_diatomic_input(
        TEMPLATE,
        1.55,
        method="cs-gf2",
        job_name="CS-GF2 PES R=1.55000",
    )
    assert "method = cs-gf2" in text
    assert "name = CS-GF2 PES R=1.55000" in text
    assert "H 0.000000000000 0.000000000000 1.550000000000" in text
    assert "beta = 200" in text
    assert "mode1 = 0.12086, 0.05, 0.0, 0.0, 1.0" in text


def test_render_diatomic_input_preserves_non_z_axis() -> None:
    template = TEMPLATE.replace("H  0.0 0.0 1.600000", "H  1.0 1.0 0.0")
    text = render_diatomic_input(template, np.sqrt(2.0) * 2.0)
    assert "H 2.000000000000 2.000000000000 0.000000000000" in text


def test_render_diatomic_input_rejects_non_diatomic() -> None:
    template = TEMPLATE.replace("H  0.0 0.0 1.600000", "H  0.0 0.0 1.600000\nH 0.0 1.0 0.0")
    with pytest.raises(ValueError, match="exactly two"):
        render_diatomic_input(template, 1.0)


def test_distance_tag() -> None:
    assert distance_tag(1.6) == "1p60000"
    with pytest.raises(ValueError):
        distance_tag(0.0)
