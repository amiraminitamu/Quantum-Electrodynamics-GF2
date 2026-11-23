# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Lightweight molecular regression for documented grid convergence."""

from __future__ import annotations

import pytest

pytest.importorskip("psi4")

from tests.grid_study import run_convergence_study


def test_documented_time_frequency_error_decreases_with_grid_size() -> None:
    rows = run_convergence_study()
    errors = [float(row["time_frequency_error"]) for row in rows]
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert all(float(row["particle_error"]) < 1.0e-6 for row in rows)
