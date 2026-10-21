# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Unit tests for the outer GF2 fixed-point convergence gate."""

from __future__ import annotations

from gf2.convergence import outer_fixed_point_converged


_BASE = {
    "energy": 1.0e-7,
    "particle_number": 1.0e-8,
    "fock_rms": 1.0e-8,
    "sigma_iw": 1.0e-7,
    "pi": 1.0e-7,
    "d": 1.0e-7,
    "bosonic_dyson": 1.0e-7,
}


def _gate(residuals: dict[str, float], *, coupled: bool) -> bool:
    return outer_fixed_point_converged(
        residuals,
        energy_threshold=1.0e-6,
        particle_threshold=1.0e-6,
        fock_threshold=1.0e-6,
        sigma_iw_threshold=1.0e-6,
        coupled=coupled,
        bosonic_threshold=1.0e-6,
    )


def test_electronic_gate_accepts_small_fixed_point_residuals() -> None:
    assert _gate(dict(_BASE), coupled=False)


def test_coupled_gate_accepts_small_coupled_residuals() -> None:
    assert _gate(dict(_BASE), coupled=True)


def test_small_energy_alone_does_not_converge_coupled_solver() -> None:
    residuals = dict(_BASE)
    residuals["d"] = 4.8e-2
    residuals["bosonic_dyson"] = 3.5e-3
    assert not _gate(residuals, coupled=True)


def test_unsynchronized_self_energy_blocks_convergence() -> None:
    residuals = dict(_BASE)
    residuals["sigma_iw"] = 2.0e-4
    assert not _gate(residuals, coupled=False)
    assert not _gate(residuals, coupled=True)


def test_electronic_solver_does_not_require_bosonic_keys() -> None:
    residuals = {
        key: value for key, value in _BASE.items() if key not in {"pi", "d", "bosonic_dyson"}
    }
    assert _gate(residuals, coupled=False)
