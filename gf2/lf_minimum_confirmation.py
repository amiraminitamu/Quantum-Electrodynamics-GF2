# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Confirm a correlated LF-GF2 minimum with independent local directions."

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from gf2.lf_correlated_optimizer import LFCorrelatedObjectiveEvaluation


ObjectiveFunction = Callable[[np.ndarray, Any | None], LFCorrelatedObjectiveEvaluation]
EvaluationCallback = Callable[[int, str, LFCorrelatedObjectiveEvaluation], None]
AcceptedCallback = Callable[[int, LFCorrelatedObjectiveEvaluation], None]


@dataclass(frozen=True)
class LFMinimumConfirmationOptions:
    """Controls for the local stochastic minimum confirmation stage."""

    radius: float = 1.0e-3
    energy_tolerance: float = 1.0e-7
    required_misses: int = 10
    max_directions: int = 50
    random_seed: int = 314159
    component_bound: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("radius must be finite and positive.")
        if not np.isfinite(self.energy_tolerance) or self.energy_tolerance < 0.0:
            raise ValueError("energy_tolerance must be finite and non-negative.")
        if self.required_misses < 1:
            raise ValueError("required_misses must be at least one.")
        if self.max_directions < self.required_misses:
            raise ValueError("max_directions must be at least required_misses.")
        if self.component_bound is not None:
            if not np.isfinite(self.component_bound) or self.component_bound <= 0.0:
                raise ValueError("component_bound must be finite and positive.")


@dataclass(frozen=True)
class LFMinimumConfirmationRecord:
    """One symmetric confirmation direction."""

    direction_index: int
    energy_start: float
    energy_plus: float
    energy_minus: float
    accepted_role: str
    accepted_energy: float
    improvement: float
    consecutive_misses: int
    direction: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", np.asarray(self.direction, dtype=float).copy())


@dataclass(frozen=True)
class LFMinimumConfirmationResult:
    """Result of local correlated-energy minimum confirmation."""

    lambda_value: np.ndarray
    energy: float
    state: Any
    confirmed: bool
    reason: str
    directions: int
    history: tuple[LFMinimumConfirmationRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "lambda_value", np.asarray(self.lambda_value, dtype=float).copy())
        object.__setattr__(self, "energy", float(self.energy))


def _bounded_lambda(value: np.ndarray, bound: float | None) -> np.ndarray:
    lam = np.asarray(value, dtype=float)
    if bound is None:
        return lam.copy()
    return np.clip(lam, -float(bound), float(bound))


def _rademacher_direction(rng: np.random.Generator, nparam: int) -> np.ndarray:
    signs = rng.choice(np.array([-1.0, 1.0]), size=nparam)
    return signs / np.sqrt(float(nparam))


def confirm_correlated_minimum(
    objective: ObjectiveFunction,
    initial_evaluation: LFCorrelatedObjectiveEvaluation,
    *,
    options: LFMinimumConfirmationOptions | None = None,
    evaluation_callback: EvaluationCallback | None = None,
    accepted_callback: AcceptedCallback | None = None,
) -> LFMinimumConfirmationResult:
    "Confirm a local correlated LF-GF2 minimum."
    opts = options or LFMinimumConfirmationOptions()
    current = initial_evaluation
    if not current.converged or current.state is None:
        raise ValueError("Minimum confirmation requires a converged reusable state.")

    lam = np.asarray(current.lambda_value, dtype=float)
    if lam.ndim != 1 or lam.size == 0 or not np.all(np.isfinite(lam)):
        raise ValueError("initial lambda must be a non-empty finite vector.")

    rng = np.random.default_rng(opts.random_seed)
    misses = 0
    history: list[LFMinimumConfirmationRecord] = []
    confirmed = False
    reason = "maximum confirmation directions reached"

    for direction_index in range(1, opts.max_directions + 1):
        direction = _rademacher_direction(rng, lam.size)
        start_energy = float(current.total_energy)
        plus_lambda = _bounded_lambda(
            current.lambda_value + opts.radius * direction,
            opts.component_bound,
        )
        minus_lambda = _bounded_lambda(
            current.lambda_value - opts.radius * direction,
            opts.component_bound,
        )

        plus = objective(plus_lambda.copy(), current.state)
        if evaluation_callback is not None:
            evaluation_callback(direction_index, "confirm_plus", plus)
        minus = objective(minus_lambda.copy(), current.state)
        if evaluation_callback is not None:
            evaluation_callback(direction_index, "confirm_minus", minus)

        if not plus.converged or not minus.converged:
            raise RuntimeError(
                "A minimum-confirmation probe did not converge; the local "
                "direction cannot be used as evidence of convergence."
            )

        improving = [
            ("confirm_plus", plus),
            ("confirm_minus", minus),
        ]
        improving = [
            (role, item)
            for role, item in improving
            if item.total_energy < current.total_energy - opts.energy_tolerance
        ]

        accepted_role = "rejected"
        accepted = current
        if improving:
            accepted_role, accepted = min(
                improving,
                key=lambda pair: pair[1].total_energy,
            )
            current = accepted
            misses = 0
            if accepted_callback is not None:
                accepted_callback(direction_index, current)
        else:
            misses += 1

        history.append(
            LFMinimumConfirmationRecord(
                direction_index=direction_index,
                energy_start=start_energy,
                energy_plus=float(plus.total_energy),
                energy_minus=float(minus.total_energy),
                accepted_role=accepted_role,
                accepted_energy=float(current.total_energy),
                improvement=float(start_energy - current.total_energy),
                consecutive_misses=misses,
                direction=direction,
            )
        )

        if misses >= opts.required_misses:
            confirmed = True
            reason = (
                f"{misses} consecutive independent local directions showed no "
                "resolvable energy improvement"
            )
            break

    return LFMinimumConfirmationResult(
        lambda_value=current.lambda_value,
        energy=float(current.total_energy),
        state=current.state,
        confirmed=confirmed,
        reason=reason,
        directions=len(history),
        history=tuple(history),
    )
