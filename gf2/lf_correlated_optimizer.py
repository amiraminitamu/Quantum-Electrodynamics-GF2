# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Outer correlated-energy optimization of the projected LF vector."

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class LFCorrelatedOptimizerOptions:
    """Controls for the outer correlated-energy LF optimization."""

    max_iterations: int = 12
    probe_radius: float = 8.0e-3
    min_probe_radius: float = 1.0e-3
    probe_decay: float = 0.101
    trust_radius: float = 2.0e-2
    min_trust_radius: float = 1.0e-3
    max_trust_radius: float = 5.0e-2
    trust_shrink: float = 0.5
    trust_grow: float = 1.20
    energy_tolerance: float = 5.0e-8
    stagnation_iterations: int = 3
    radial_bootstrap_iterations: int = 3
    random_seed: int = 10086
    component_bound: float | None = None
    iteration_offset: int = 0

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least one.")
        if not np.isfinite(self.probe_radius) or self.probe_radius <= 0.0:
            raise ValueError("probe_radius must be finite and positive.")
        if not np.isfinite(self.min_probe_radius) or self.min_probe_radius <= 0.0:
            raise ValueError("min_probe_radius must be finite and positive.")
        if self.min_probe_radius > self.probe_radius:
            raise ValueError("min_probe_radius cannot exceed probe_radius.")
        if not np.isfinite(self.probe_decay) or self.probe_decay < 0.0:
            raise ValueError("probe_decay must be finite and non-negative.")
        if not np.isfinite(self.trust_radius) or self.trust_radius <= 0.0:
            raise ValueError("trust_radius must be finite and positive.")
        if not np.isfinite(self.min_trust_radius) or self.min_trust_radius <= 0.0:
            raise ValueError("min_trust_radius must be finite and positive.")
        if not np.isfinite(self.max_trust_radius) or self.max_trust_radius <= 0.0:
            raise ValueError("max_trust_radius must be finite and positive.")
        if not (self.min_trust_radius <= self.trust_radius <= self.max_trust_radius):
            raise ValueError("Require min_trust_radius <= trust_radius <= max_trust_radius.")
        if not (0.0 < self.trust_shrink < 1.0):
            raise ValueError("trust_shrink must lie in (0,1).")
        if self.trust_grow < 1.0:
            raise ValueError("trust_grow must be at least one.")
        if not np.isfinite(self.energy_tolerance) or self.energy_tolerance < 0.0:
            raise ValueError("energy_tolerance must be finite and non-negative.")
        if self.stagnation_iterations < 1:
            raise ValueError("stagnation_iterations must be at least one.")
        if self.radial_bootstrap_iterations < 0:
            raise ValueError("radial_bootstrap_iterations must be non-negative.")
        if self.iteration_offset < 0:
            raise ValueError("iteration_offset must be non-negative.")
        if self.component_bound is not None:
            if not np.isfinite(self.component_bound) or self.component_bound <= 0.0:
                raise ValueError("component_bound must be finite and positive.")


@dataclass(frozen=True)
class LFCorrelatedObjectiveEvaluation:
    """One fully self-consistent fixed-lambda objective evaluation."""

    lambda_value: np.ndarray
    total_energy: float
    converged: bool
    state: Any
    core_energy: float = np.nan
    correlation_energy: float = np.nan
    outer_iterations: int = 0
    wall_time_s: float = np.nan

    def __post_init__(self) -> None:
        lam = np.asarray(self.lambda_value, dtype=float)
        if lam.ndim != 1 or not np.all(np.isfinite(lam)):
            raise ValueError("lambda_value must be a finite one-dimensional array.")
        if not np.isfinite(self.total_energy):
            raise ValueError("total_energy must be finite.")
        if self.outer_iterations < 0:
            raise ValueError("outer_iterations must be non-negative.")
        object.__setattr__(self, "lambda_value", lam.copy())
        object.__setattr__(self, "total_energy", float(self.total_energy))
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "core_energy", float(self.core_energy))
        object.__setattr__(self, "correlation_energy", float(self.correlation_energy))
        object.__setattr__(self, "outer_iterations", int(self.outer_iterations))
        object.__setattr__(self, "wall_time_s", float(self.wall_time_s))


@dataclass(frozen=True)
class LFCorrelatedEvaluationRecord:
    """Bookkeeping entry for a correlated objective call."""

    evaluation_index: int
    iteration: int
    role: str
    lambda_value: np.ndarray
    total_energy: float
    converged: bool
    outer_iterations: int
    wall_time_s: float
    warm_start_used: bool
    warm_source_lambda_norm: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "lambda_value", np.asarray(self.lambda_value, dtype=float).copy())


@dataclass(frozen=True)
class LFCorrelatedIterationRecord:
    """Summary of one simultaneous-direction outer iteration."""

    iteration: int
    direction_kind: str
    direction: np.ndarray
    probe_radius: float
    trust_radius_before: float
    trust_radius_after: float
    energy_start: float
    energy_plus: float
    energy_minus: float
    directional_derivative: float
    energy_candidate: float
    accepted_role: str
    accepted_energy: float
    improvement: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", np.asarray(self.direction, dtype=float).copy())


@dataclass(frozen=True)
class LFCorrelatedOptimizationResult:
    "Result of the outer correlated-energy LF optimization."

    lambda_opt: np.ndarray
    energy_opt: float
    state: Any
    lambda_sampled_best: np.ndarray
    energy_sampled_best: float
    sampled_best_state: Any
    converged: bool
    reason: str
    iterations: int
    evaluations: tuple[LFCorrelatedEvaluationRecord, ...]
    history: tuple[LFCorrelatedIterationRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "lambda_opt", np.asarray(self.lambda_opt, dtype=float).copy())
        object.__setattr__(
            self,
            "lambda_sampled_best",
            np.asarray(self.lambda_sampled_best, dtype=float).copy(),
        )
        object.__setattr__(self, "energy_opt", float(self.energy_opt))
        object.__setattr__(self, "energy_sampled_best", float(self.energy_sampled_best))


ObjectiveFunction = Callable[[np.ndarray, Any | None], LFCorrelatedObjectiveEvaluation]
EvaluationCallback = Callable[[LFCorrelatedEvaluationRecord], None]
AcceptedCallback = Callable[[int, LFCorrelatedObjectiveEvaluation, float], None]


def _bounded_lambda(value: np.ndarray, bound: float | None) -> np.ndarray:
    lam = np.asarray(value, dtype=float)
    if bound is None:
        return lam.copy()
    return np.clip(lam, -float(bound), float(bound))


def _unit_direction(
    *,
    iteration: int,
    current_lambda: np.ndarray,
    radial_bootstrap_iterations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    """Return a unit radial bootstrap or Rademacher simultaneous direction."""
    lam = np.asarray(current_lambda, dtype=float)
    norm = float(np.linalg.norm(lam))
    if iteration <= radial_bootstrap_iterations and norm > 1.0e-14:
        return lam / norm, "radial"

    signs = rng.choice(np.array([-1.0, 1.0]), size=lam.size)
    return signs / np.sqrt(float(lam.size)), "rademacher"


def _advance_rng_for_restart(
    rng: np.random.Generator,
    *,
    nparam: int,
    iteration_offset: int,
    radial_bootstrap_iterations: int,
) -> None:
    """Advance deterministic Rademacher draws to the restart iteration."""
    previous_random = max(0, iteration_offset - radial_bootstrap_iterations)
    for _ in range(previous_random):
        rng.choice(np.array([-1.0, 1.0]), size=nparam)


def _nearest_state(
    target: np.ndarray,
    candidates: list[LFCorrelatedObjectiveEvaluation],
) -> Any | None:
    usable = [item for item in candidates if item.converged and item.state is not None]
    if not usable:
        return None
    target_arr = np.asarray(target, dtype=float)
    best = min(
        usable,
        key=lambda item: float(np.linalg.norm(target_arr - item.lambda_value)),
    )
    return best.state


def optimize_correlated_lambda(
    objective: ObjectiveFunction,
    initial_lambda: np.ndarray,
    *,
    options: LFCorrelatedOptimizerOptions | None = None,
    initial_state: Any | None = None,
    initial_evaluation: LFCorrelatedObjectiveEvaluation | None = None,
    evaluation_callback: EvaluationCallback | None = None,
    accepted_callback: AcceptedCallback | None = None,
) -> LFCorrelatedOptimizationResult:
    "Minimize the fully converged LF-GF2 energy over the LF vector."
    opts = options or LFCorrelatedOptimizerOptions()
    lam0 = np.asarray(initial_lambda, dtype=float)
    if lam0.ndim != 1 or lam0.size == 0 or not np.all(np.isfinite(lam0)):
        raise ValueError("initial_lambda must be a non-empty finite one-dimensional array.")
    lam0 = _bounded_lambda(lam0, opts.component_bound)
    if initial_state is not None and initial_evaluation is not None:
        raise ValueError("Supply either initial_state or initial_evaluation, not both.")

    rng = np.random.default_rng(opts.random_seed)
    _advance_rng_for_restart(
        rng,
        nparam=lam0.size,
        iteration_offset=opts.iteration_offset,
        radial_bootstrap_iterations=opts.radial_bootstrap_iterations,
    )
    eval_records: list[LFCorrelatedEvaluationRecord] = []
    history: list[LFCorrelatedIterationRecord] = []
    evaluation_index = 0
    sampled_best: LFCorrelatedObjectiveEvaluation | None = None

    def consider_sample(result: LFCorrelatedObjectiveEvaluation) -> None:
        nonlocal sampled_best
        if result.converged and (
            sampled_best is None or result.total_energy < sampled_best.total_energy
        ):
            sampled_best = result

    def evaluate(
        lam: np.ndarray,
        warm_state: Any | None,
        *,
        iteration: int,
        role: str,
    ) -> LFCorrelatedObjectiveEvaluation:
        nonlocal evaluation_index
        trial = _bounded_lambda(lam, opts.component_bound)
        result = objective(trial.copy(), warm_state)
        if result.lambda_value.shape != trial.shape or not np.allclose(
            result.lambda_value, trial, rtol=0.0, atol=1.0e-13
        ):
            raise ValueError("Objective returned an evaluation for the wrong lambda vector.")
        evaluation_index += 1
        source_norm: float | None = None
        if warm_state is not None and hasattr(warm_state, "lambda_fixed"):
            source_norm = float(np.linalg.norm(np.asarray(warm_state.lambda_fixed, dtype=float)))
        record = LFCorrelatedEvaluationRecord(
            evaluation_index=evaluation_index,
            iteration=iteration,
            role=role,
            lambda_value=result.lambda_value,
            total_energy=result.total_energy,
            converged=result.converged,
            outer_iterations=result.outer_iterations,
            wall_time_s=result.wall_time_s,
            warm_start_used=warm_state is not None,
            warm_source_lambda_norm=source_norm,
        )
        eval_records.append(record)
        consider_sample(result)
        if evaluation_callback is not None:
            evaluation_callback(record)
        return result

    if initial_evaluation is not None:
        current = initial_evaluation
        if current.lambda_value.shape != lam0.shape or not np.allclose(
            current.lambda_value, lam0, rtol=0.0, atol=1.0e-13
        ):
            raise ValueError("initial_evaluation does not match initial_lambda.")
        consider_sample(current)
    else:
        current = evaluate(
            lam0,
            initial_state,
            iteration=opts.iteration_offset,
            role="initial",
        )

    if not current.converged:
        raise RuntimeError(
            "The initial correlated LF-GF2 objective did not converge; outer "
            "optimization cannot start from an unconverged fixed point."
        )
    if current.state is None:
        raise RuntimeError("The initial correlated objective has no reusable state.")

    trust_radius = float(opts.trust_radius)
    stagnation = 0
    converged = False
    reason = "maximum iterations reached"
    if accepted_callback is not None:
        accepted_callback(opts.iteration_offset, current, trust_radius)

    for local_iteration in range(1, opts.max_iterations + 1):
        iteration = opts.iteration_offset + local_iteration
        energy_start = float(current.total_energy)
        direction, direction_kind = _unit_direction(
            iteration=iteration,
            current_lambda=current.lambda_value,
            radial_bootstrap_iterations=opts.radial_bootstrap_iterations,
            rng=rng,
        )
        probe_radius = max(
            opts.min_probe_radius,
            opts.probe_radius / (float(iteration) ** opts.probe_decay),
        )

        plus_lambda = _bounded_lambda(
            current.lambda_value + probe_radius * direction,
            opts.component_bound,
        )
        minus_lambda = _bounded_lambda(
            current.lambda_value - probe_radius * direction,
            opts.component_bound,
        )
        plus = evaluate(
            plus_lambda,
            current.state,
            iteration=iteration,
            role="probe_plus",
        )
        minus = evaluate(
            minus_lambda,
            current.state,
            iteration=iteration,
            role="probe_minus",
        )
        if not plus.converged or not minus.converged:
            raise RuntimeError(
                "A simultaneous-perturbation probe did not converge. The "
                "directional derivative is not trustworthy; inspect the inner "
                "GF2 convergence rather than optimizing an unconverged energy."
            )

        actual_probe_separation = float(np.linalg.norm(plus.lambda_value - minus.lambda_value))
        if actual_probe_separation <= 1.0e-14:
            raise RuntimeError("The component bound collapsed the two optimizer probes.")
        directional_derivative = float(
            (plus.total_energy - minus.total_energy) / actual_probe_separation
        )

        if directional_derivative > 0.0:
            downhill = -direction
        elif directional_derivative < 0.0:
            downhill = direction
        else:
            downhill = np.zeros_like(direction)

        trust_before = trust_radius
        candidate_lambda = _bounded_lambda(
            current.lambda_value + trust_radius * downhill,
            opts.component_bound,
        )

        if np.linalg.norm(candidate_lambda - current.lambda_value) <= 1.0e-14:
            candidate = current
            candidate_energy = float(current.total_energy)
        else:
            warm_state = _nearest_state(candidate_lambda, [current, plus, minus])
            candidate = evaluate(
                candidate_lambda,
                warm_state,
                iteration=iteration,
                role="candidate",
            )
            candidate_energy = float(candidate.total_energy)

        # Accept the lowest converged point sampled in this iteration, not the
        # candidate merely because it improves. A symmetric probe can be both

        choices: list[tuple[str, LFCorrelatedObjectiveEvaluation]] = [
            ("probe_plus", plus),
            ("probe_minus", minus),
        ]
        if candidate is not current:
            choices.append(("candidate", candidate))
        improving = [
            (role, item)
            for role, item in choices
            if item.converged and item.total_energy < current.total_energy - opts.energy_tolerance
        ]

        accepted = current
        accepted_role = "rejected"
        if improving:
            accepted_role, accepted = min(
                improving,
                key=lambda pair: pair[1].total_energy,
            )
            if accepted_role == "candidate":
                trust_radius = min(
                    opts.max_trust_radius,
                    trust_radius * opts.trust_grow,
                )
            else:
                trust_radius = max(
                    opts.min_trust_radius,
                    trust_radius * np.sqrt(opts.trust_shrink),
                )
        else:
            trust_radius = max(
                opts.min_trust_radius,
                trust_radius * opts.trust_shrink,
            )

        improvement = float(current.total_energy - accepted.total_energy)
        if accepted is current:
            stagnation += 1
        else:
            current = accepted
            stagnation = 0

        row = LFCorrelatedIterationRecord(
            iteration=iteration,
            direction_kind=direction_kind,
            direction=direction,
            probe_radius=float(probe_radius),
            trust_radius_before=float(trust_before),
            trust_radius_after=float(trust_radius),
            energy_start=energy_start,
            energy_plus=float(plus.total_energy),
            energy_minus=float(minus.total_energy),
            directional_derivative=directional_derivative,
            energy_candidate=candidate_energy,
            accepted_role=accepted_role,
            accepted_energy=float(current.total_energy),
            improvement=improvement,
        )
        history.append(row)
        if accepted_callback is not None:
            accepted_callback(iteration, current, trust_radius)

        if (
            trust_radius <= opts.min_trust_radius * (1.0 + 1.0e-12)
            and stagnation >= opts.stagnation_iterations
        ):
            converged = True
            reason = "minimum trust radius reached without a resolvable energy improvement"
            break

    assert sampled_best is not None
    return LFCorrelatedOptimizationResult(
        lambda_opt=current.lambda_value,
        energy_opt=float(current.total_energy),
        state=current.state,
        lambda_sampled_best=sampled_best.lambda_value,
        energy_sampled_best=float(sampled_best.total_energy),
        sampled_best_state=sampled_best.state,
        converged=converged,
        reason=reason,
        iterations=len(history),
        evaluations=tuple(eval_records),
        history=tuple(history),
    )
