# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gf2.lf_correlated_optimizer import LFCorrelatedObjectiveEvaluation
from gf2.lf_minimum_confirmation import (
    LFMinimumConfirmationOptions,
    confirm_correlated_minimum,
)


@dataclass(frozen=True)
class _DummyState:
    lambda_fixed: np.ndarray


def _eval(lam: np.ndarray, energy: float) -> LFCorrelatedObjectiveEvaluation:
    arr = np.asarray(lam, dtype=float)
    return LFCorrelatedObjectiveEvaluation(
        lambda_value=arr,
        total_energy=float(energy),
        converged=True,
        state=_DummyState(arr.copy()),
    )


def test_flat_objective_requires_full_confirmation_set() -> None:
    calls = 0

    def objective(lam: np.ndarray, warm_state):
        nonlocal calls
        calls += 1
        return _eval(lam, 0.0)

    initial = _eval(np.array([0.1, -0.2, 0.05, 0.3]), 0.0)
    options = LFMinimumConfirmationOptions(
        radius=1.0e-3,
        energy_tolerance=1.0e-7,
        required_misses=10,
        max_directions=20,
        random_seed=7,
    )
    result = confirm_correlated_minimum(objective, initial, options=options)

    assert result.confirmed
    assert result.directions == 10
    assert calls == 20
    assert result.energy == 0.0
    assert result.history[-1].consecutive_misses == 10


def test_resolved_improvement_resets_confirmation_counter() -> None:
    initial_lambda = np.array([0.0, 0.0])
    target = np.array([0.001, -0.001]) / np.sqrt(2.0)

    def objective(lam: np.ndarray, warm_state):
        lam = np.asarray(lam, dtype=float)
        energy = float(np.dot(lam - target, lam - target))
        return _eval(lam, energy)

    initial_energy = float(np.dot(initial_lambda - target, initial_lambda - target))
    initial = _eval(initial_lambda, initial_energy)
    options = LFMinimumConfirmationOptions(
        radius=1.0e-3,
        energy_tolerance=1.0e-10,
        required_misses=3,
        max_directions=12,
        random_seed=1,
    )
    result = confirm_correlated_minimum(objective, initial, options=options)

    assert result.energy < initial_energy
    accepted = [row for row in result.history if row.accepted_role != "rejected"]
    assert accepted
    assert all(row.consecutive_misses == 0 for row in accepted)


def test_improvement_below_energy_tolerance_is_not_accepted() -> None:
    initial = _eval(np.array([0.0]), 0.0)

    def objective(lam: np.ndarray, warm_state):
        energy = -5.0e-8 if float(lam[0]) > 0.0 else 1.0e-6
        return _eval(lam, energy)

    options = LFMinimumConfirmationOptions(
        radius=1.0e-3,
        energy_tolerance=1.0e-7,
        required_misses=2,
        max_directions=2,
        random_seed=2,
    )
    result = confirm_correlated_minimum(objective, initial, options=options)

    assert result.confirmed
    assert result.energy == 0.0
    assert all(row.accepted_role == "rejected" for row in result.history)
