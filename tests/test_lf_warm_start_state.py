# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Unit tests for reusable fixed-lambda LF-GF2 warm-start states."""

from __future__ import annotations

import numpy as np
import pytest

from gf2.lf_state import (
    LFGF2FixedState,
    build_lf_problem_signature,
    load_lf_gf2_state,
    save_lf_gf2_state,
    validate_lf_warm_start_grids,
    validate_lf_warm_start_problem,
)


def _signature(shift: float = 0.0) -> str:
    h = np.array([[-1.0 + shift, 0.03], [0.03, -0.2]])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 0.2
    g = np.array([[0.06, -0.01], [-0.01, -0.04]])
    return build_lf_problem_signature(
        h_pf=h,
        eri_pf=eri,
        g_pf=g,
        overlap=np.eye(2),
        omega=0.2,
        e_nuc=0.7,
        nelec=2.0,
        nocc=1,
    )


def _state(
    *,
    signature: str | None = None,
    converged: bool = True,
    lambda_fixed: np.ndarray | None = None,
) -> LFGF2FixedState:
    iw = np.array([0.1, 0.3, 0.5])
    tau = np.array([0.0, 0.5, 1.0, 1.5])
    inu = np.array([0.0, 0.2, 0.4, 0.6])
    return LFGF2FixedState(
        problem_signature=signature or _signature(),
        lambda_fixed=(np.array([0.05, -0.02]) if lambda_fixed is None else lambda_fixed),
        xi=-0.17,
        beta=20.0,
        fermionic_iomega=iw,
        imaginary_time_grid=tau,
        bosonic_inu=inu,
        density=np.array([[1.8, 0.05], [0.05, 0.2]]),
        fock=np.array([[-0.8, 0.02], [0.02, 0.1]]),
        mu=-0.3,
        G_iw=np.zeros((3, 2, 2), dtype=complex),
        Sigma_iw=np.zeros((3, 2, 2), dtype=complex),
        D_iv=np.zeros((4, 1, 1), dtype=complex),
        Pi_iv=np.zeros((4, 1, 1), dtype=complex),
        core_energy=-1.2,
        correlation_energy=-0.05,
        converged=converged,
        outer_iterations=12,
    )


def test_problem_signature_changes_when_physical_problem_changes() -> None:
    assert _signature(0.0) == _signature(0.0)
    assert _signature(0.0) != _signature(1.0e-6)


def test_compatible_warm_state_is_accepted() -> None:
    state = _state()
    validate_lf_warm_start_problem(
        state,
        problem_signature=_signature(),
        nbf=2,
    )
    validate_lf_warm_start_grids(
        state,
        beta=20.0,
        fermionic_iomega=np.array([0.1, 0.3, 0.5]),
        imaginary_time_grid=np.array([0.0, 0.5, 1.0, 1.5]),
        bosonic_inu=np.array([0.0, 0.2, 0.4, 0.6]),
        nbf=2,
    )


def test_unconverged_state_is_rejected() -> None:
    with pytest.raises(ValueError, match="converged"):
        validate_lf_warm_start_problem(
            _state(converged=False),
            problem_signature=_signature(),
            nbf=2,
        )


def test_different_problem_is_rejected() -> None:
    with pytest.raises(ValueError, match="different molecular/cavity"):
        validate_lf_warm_start_problem(
            _state(signature=_signature(1.0e-6)),
            problem_signature=_signature(),
            nbf=2,
        )


def test_different_grid_is_rejected() -> None:
    state = _state()
    with pytest.raises(ValueError, match="fermionic Matsubara"):
        validate_lf_warm_start_grids(
            state,
            beta=20.0,
            fermionic_iomega=np.array([0.1, 0.31, 0.5]),
            imaginary_time_grid=np.array([0.0, 0.5, 1.0, 1.5]),
            bosonic_inu=np.array([0.0, 0.2, 0.4, 0.6]),
            nbf=2,
        )


def test_state_copies_mutable_arrays() -> None:
    lam = np.array([0.05, -0.02])
    state = _state(lambda_fixed=lam)
    lam[:] = 99.0
    np.testing.assert_allclose(state.lambda_fixed, [0.05, -0.02])
    assert state.total_energy == pytest.approx(-1.25)


def test_state_checkpoint_roundtrip(tmp_path) -> None:
    original = _state()
    path = save_lf_gf2_state(tmp_path / "state.npz", original)
    restored = load_lf_gf2_state(path)

    assert restored.problem_signature == original.problem_signature
    assert restored.xi == pytest.approx(original.xi)
    assert restored.beta == pytest.approx(original.beta)
    assert restored.mu == pytest.approx(original.mu)
    assert restored.core_energy == pytest.approx(original.core_energy)
    assert restored.correlation_energy == pytest.approx(original.correlation_energy)
    assert restored.converged is original.converged
    assert restored.outer_iterations == original.outer_iterations
    for name in (
        "lambda_fixed",
        "fermionic_iomega",
        "imaginary_time_grid",
        "bosonic_inu",
        "density",
        "fock",
        "G_iw",
        "Sigma_iw",
        "D_iv",
        "Pi_iv",
    ):
        np.testing.assert_allclose(getattr(restored, name), getattr(original, name))
