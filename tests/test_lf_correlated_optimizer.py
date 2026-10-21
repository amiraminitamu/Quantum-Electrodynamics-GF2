# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from gf2.lf_correlated_optimizer import (
    LFCorrelatedObjectiveEvaluation,
    LFCorrelatedOptimizerOptions,
    optimize_correlated_lambda,
)


@dataclass(frozen=True)
class _DummyState:
    lambda_fixed: np.ndarray


def _quadratic_objective(target: np.ndarray, calls: list[tuple[np.ndarray, object]]):
    target = np.asarray(target, dtype=float)

    def objective(lam: np.ndarray, warm_state):
        lam = np.asarray(lam, dtype=float)
        calls.append((lam.copy(), warm_state))
        energy = float(np.dot(lam - target, lam - target))
        return LFCorrelatedObjectiveEvaluation(
            lambda_value=lam,
            total_energy=energy,
            converged=True,
            state=_DummyState(lam.copy()),
            core_energy=energy,
            correlation_energy=0.0,
            outer_iterations=7,
            wall_time_s=1.5,
        )

    return objective


def test_radial_bootstrap_reduces_correlated_objective_without_fixed_scale():
    initial = np.array([1.0, -0.5])
    target = 0.40 * initial
    calls: list[tuple[np.ndarray, object]] = []
    objective = _quadratic_objective(target, calls)
    options = LFCorrelatedOptimizerOptions(
        max_iterations=3,
        probe_radius=5.0e-2,
        min_probe_radius=5.0e-2,
        probe_decay=0.0,
        trust_radius=2.0e-1,
        min_trust_radius=1.0e-2,
        max_trust_radius=2.0e-1,
        trust_grow=1.0,
        energy_tolerance=1.0e-14,
        radial_bootstrap_iterations=3,
        random_seed=3,
    )

    result = optimize_correlated_lambda(objective, initial, options=options)

    initial_energy = float(np.dot(initial - target, initial - target))
    assert result.energy_opt < initial_energy
    assert np.linalg.norm(result.lambda_opt - target) < np.linalg.norm(initial - target)
    assert result.energy_sampled_best <= result.energy_opt
    assert [row.direction_kind for row in result.history] == [
        "radial",
        "radial",
        "radial",
    ]
    assert len(result.evaluations) == 1 + 3 * 3
    assert calls[0][1] is None
    assert all(warm is not None for _, warm in calls[1:])


def test_fixed_seed_rademacher_sequence_is_reproducible():
    initial = np.array([0.2, -0.1, 0.05, 0.3])
    target = np.array([-0.05, 0.08, 0.01, 0.12])
    options = LFCorrelatedOptimizerOptions(
        max_iterations=2,
        probe_radius=1.0e-2,
        min_probe_radius=1.0e-2,
        probe_decay=0.0,
        trust_radius=2.0e-2,
        min_trust_radius=1.0e-3,
        max_trust_radius=2.0e-2,
        trust_grow=1.0,
        radial_bootstrap_iterations=0,
        random_seed=91,
    )

    result_a = optimize_correlated_lambda(
        _quadratic_objective(target, []), initial, options=options
    )
    result_b = optimize_correlated_lambda(
        _quadratic_objective(target, []), initial, options=options
    )

    assert np.allclose(result_a.lambda_opt, result_b.lambda_opt)
    assert result_a.energy_opt == pytest.approx(result_b.energy_opt)
    for row_a, row_b in zip(result_a.history, result_b.history, strict=True):
        assert row_a.direction_kind == "rademacher"
        assert row_b.direction_kind == "rademacher"
        assert np.array_equal(row_a.direction, row_b.direction)


def test_unconverged_probe_is_rejected_as_invalid_gradient_information():
    calls = 0

    def objective(lam: np.ndarray, warm_state):
        nonlocal calls
        calls += 1
        converged = calls != 2
        return LFCorrelatedObjectiveEvaluation(
            lambda_value=lam,
            total_energy=float(np.dot(lam, lam)),
            converged=converged,
            state=_DummyState(np.asarray(lam, dtype=float).copy()),
        )

    options = LFCorrelatedOptimizerOptions(
        max_iterations=1,
        radial_bootstrap_iterations=0,
    )
    with pytest.raises(RuntimeError, match="probe did not converge"):
        optimize_correlated_lambda(objective, np.array([0.1, 0.2]), options=options)


def test_initial_objective_must_be_converged():
    def objective(lam: np.ndarray, warm_state):
        return LFCorrelatedObjectiveEvaluation(
            lambda_value=lam,
            total_energy=0.0,
            converged=False,
            state=None,
        )

    with pytest.raises(RuntimeError, match="initial correlated LF-GF2 objective"):
        optimize_correlated_lambda(objective, np.array([0.1]))


def test_restart_evaluation_skips_initial_objective_call():
    initial = np.array([0.2, -0.1])
    target = np.zeros(2)
    calls: list[tuple[np.ndarray, object]] = []
    objective = _quadratic_objective(target, calls)
    restart = LFCorrelatedObjectiveEvaluation(
        lambda_value=initial,
        total_energy=float(np.dot(initial, initial)),
        converged=True,
        state=_DummyState(initial.copy()),
        core_energy=float(np.dot(initial, initial)),
        correlation_energy=0.0,
    )
    checkpoints: list[tuple[int, float]] = []
    options = LFCorrelatedOptimizerOptions(
        max_iterations=1,
        radial_bootstrap_iterations=0,
        iteration_offset=4,
        probe_radius=1.0e-2,
        min_probe_radius=1.0e-2,
        probe_decay=0.0,
        trust_radius=1.0e-2,
        max_trust_radius=2.0e-2,
    )

    result = optimize_correlated_lambda(
        objective,
        initial,
        options=options,
        initial_evaluation=restart,
        accepted_callback=lambda iteration, current, trust: checkpoints.append((iteration, trust)),
    )

    assert len(calls) == 3
    assert len(result.evaluations) == 3
    assert result.history[0].iteration == 5
    assert checkpoints[0][0] == 4
    assert checkpoints[-1][0] == 5


def test_lowest_sampled_point_is_tracked_even_below_acceptance_threshold():
    initial = np.array([0.1])
    energies = {
        0.1: 0.0,
        0.11: -2.0e-8,
        0.09: 3.0e-8,
        0.12: 1.0e-6,
    }

    def objective(lam: np.ndarray, warm_state):
        key = round(float(lam[0]), 2)
        return LFCorrelatedObjectiveEvaluation(
            lambda_value=lam,
            total_energy=energies[key],
            converged=True,
            state=_DummyState(np.asarray(lam, dtype=float).copy()),
        )

    options = LFCorrelatedOptimizerOptions(
        max_iterations=1,
        probe_radius=1.0e-2,
        min_probe_radius=1.0e-2,
        probe_decay=0.0,
        trust_radius=2.0e-2,
        min_trust_radius=1.0e-3,
        max_trust_radius=2.0e-2,
        energy_tolerance=5.0e-8,
        radial_bootstrap_iterations=1,
    )
    result = optimize_correlated_lambda(objective, initial, options=options)

    assert result.energy_opt == pytest.approx(0.0)
    assert result.energy_sampled_best == pytest.approx(-2.0e-8)
    np.testing.assert_allclose(result.lambda_sampled_best, [0.11])


def test_lower_probe_is_accepted_even_when_candidate_also_improves():
    initial = np.array([0.1])

    def objective(lam: np.ndarray, warm_state):
        x = float(lam[0])
        if np.isclose(x, 0.10):
            energy = 0.0
        elif np.isclose(x, 0.11):
            energy = -3.0e-6
        elif np.isclose(x, 0.09):
            energy = 1.0e-6
        elif np.isclose(x, 0.105):
            energy = -1.0e-6
        else:
            raise AssertionError(f"Unexpected trial lambda {x}")
        return LFCorrelatedObjectiveEvaluation(
            lambda_value=lam,
            total_energy=energy,
            converged=True,
            state=_DummyState(np.asarray(lam, dtype=float).copy()),
        )

    options = LFCorrelatedOptimizerOptions(
        max_iterations=1,
        probe_radius=1.0e-2,
        min_probe_radius=1.0e-2,
        probe_decay=0.0,
        trust_radius=5.0e-3,
        min_trust_radius=1.0e-3,
        max_trust_radius=2.0e-2,
        energy_tolerance=5.0e-8,
        radial_bootstrap_iterations=1,
    )
    result = optimize_correlated_lambda(objective, initial, options=options)

    assert result.history[0].accepted_role == "probe_plus"
    assert result.energy_opt == pytest.approx(-3.0e-6)
    assert result.energy_sampled_best == pytest.approx(-3.0e-6)
    np.testing.assert_allclose(result.lambda_opt, [0.11])


def test_optimizer_option_validation():
    with pytest.raises(ValueError, match="min_probe_radius"):
        LFCorrelatedOptimizerOptions(probe_radius=1.0e-3, min_probe_radius=2.0e-3)
    with pytest.raises(ValueError, match="trust_radius"):
        LFCorrelatedOptimizerOptions(trust_radius=1.0e-5)
    with pytest.raises(ValueError, match="component_bound"):
        LFCorrelatedOptimizerOptions(component_bound=0.0)
    with pytest.raises(ValueError, match="iteration_offset"):
        LFCorrelatedOptimizerOptions(iteration_offset=-1)
