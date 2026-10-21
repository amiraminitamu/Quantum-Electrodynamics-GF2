# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Convergence gates for electronic and coupled QED-GF2 iterations."""

from __future__ import annotations

from collections.abc import Mapping


def outer_fixed_point_converged(
    residuals: Mapping[str, float],
    *,
    energy_threshold: float,
    particle_threshold: float,
    fock_threshold: float,
    sigma_iw_threshold: float,
    coupled: bool,
    bosonic_threshold: float,
) -> bool:
    "Return whether the current outer iterate satisfies fixed-point criteria."
    thresholds = (
        energy_threshold,
        particle_threshold,
        fock_threshold,
        sigma_iw_threshold,
        bosonic_threshold,
    )
    if any(value <= 0.0 for value in thresholds):
        raise ValueError("Convergence thresholds must be positive.")

    electronic_ok = (
        residuals["energy"] < energy_threshold
        and residuals["particle_number"] < particle_threshold
        and residuals["fock_rms"] < fock_threshold
        and residuals["sigma_iw"] < sigma_iw_threshold
    )
    if not electronic_ok:
        return False
    if not coupled:
        return True

    return (
        residuals["pi"] < bosonic_threshold
        and residuals["d"] < bosonic_threshold
        and residuals["bosonic_dyson"] < bosonic_threshold
    )
