# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Joint restricted Fock and chemical-potential updates."""

from __future__ import annotations

import math

import numpy as np

from .diis import RealDIIS


def safeguarded_mu_step(
    mu: float, population_error: float, slope: float, *,
    damping: float, max_step: float,
) -> float:
    """Bound one derivative-based correction to the chemical potential."""
    if not (math.isfinite(mu) and math.isfinite(population_error)):
        raise ValueError("Chemical potential and electron count must be finite")
    if not 0 < damping <= 1 or max_step <= 0:
        raise ValueError("Invalid coupled mu damping or bound")
    if not math.isfinite(slope) or slope <= 1e-12:
        return mu
    return mu + float(np.clip(-damping * population_error / slope, -max_step, max_step))


class RestrictedFockMuMixer:
    """Pulay mixing of a Fock matrix and its spin-summed population constraint."""

    def __init__(self, nbf: int, *, charge_scale: float, max_mu_step: float, history: int = 6):
        if nbf < 1 or charge_scale <= 0 or max_mu_step <= 0 or history < 2:
            raise ValueError("Invalid coupled Fock/mu mixer settings")
        self.nbf = nbf
        self.charge_scale = charge_scale
        self.max_mu_step = max_mu_step
        self.diis = RealDIIS(max_vec=history)

    def update(
        self, fock: np.ndarray, built: np.ndarray, mu: float,
        mu_proposal: float, charge_error: float,
    ) -> tuple[np.ndarray, float, bool]:
        state = np.concatenate((np.asarray(built, dtype=float).ravel(), [mu_proposal]))
        residual = np.concatenate(
            (np.asarray(fock - built, dtype=float).ravel(),
             [self.charge_scale * charge_error])
        )
        self.diis.add(state, residual)
        trial = self.diis.extrapolate()
        candidate = trial[:-1].reshape(self.nbf, self.nbf)
        reference_rms = float(np.sqrt(np.mean((built - fock) ** 2)))
        trial_rms = float(np.sqrt(np.mean((candidate - fock) ** 2)))
        fallback = bool(
            not np.all(np.isfinite(trial))
            or trial_rms > max(0.02, 3.0 * reference_rms)
        )
        if fallback:
            candidate = 0.5 * (fock + built)
            next_mu = mu_proposal
        else:
            next_mu = float(trial[-1])
        candidate = 0.5 * (candidate + candidate.T)
        next_mu = float(np.clip(next_mu, mu - self.max_mu_step, mu + self.max_mu_step))
        return candidate, next_mu, fallback
