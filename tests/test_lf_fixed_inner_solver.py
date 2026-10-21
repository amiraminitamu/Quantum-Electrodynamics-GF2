# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Unit tests for the native fixed-lambda LF-GF2 inner-solver path."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gf2 import driver_theory
from gf2.lf import lf_stationary_z


def _synthetic_reference() -> dict[str, object]:
    h = np.array([[-1.05, 0.06], [0.06, -0.28]])
    g = np.array([[0.075, -0.018], [-0.018, -0.041]])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 0.17
    eri[1, 1, 1, 1] = 0.12
    eri[0, 0, 1, 1] = 0.045
    eri[1, 1, 0, 0] = 0.045
    density = np.array([[1.80, 0.08], [0.08, 0.20]])
    return {
        "qed_data": SimpleNamespace(
            omega=np.array([0.21]),
            g=np.asarray([g]),
        ),
        "H_core": h,
        "eri": eri,
        "S": np.eye(2),
        "S_inv_half": np.eye(2),
        "P": density,
        "nocc": 1,
        "E_nuc": 0.70,
        "interaction_metadata": {
            "active_qed_low_rank_channels": 1,
        },
    }


def test_fixed_lambda_reference_skips_static_lf_hf(monkeypatch) -> None:
    ref = _synthetic_reference()
    options = SimpleNamespace(eri_type="std", extra={})
    fixed = np.array([0.11, -0.04])

    def forbidden_static_optimizer(*args, **kwargs):
        raise AssertionError("fixed-lambda inner solver must not run LF-HF optimization")

    monkeypatch.setattr(
        driver_theory,
        "solve_lf_hf_reference",
        forbidden_static_optimizer,
    )

    prepared = driver_theory._prepare_lf_reference(
        ref,
        options,
        fixed_lambda=fixed,
    )

    params = prepared["lf_params"]
    context = prepared["lf_context"]
    np.testing.assert_allclose(params.lam, fixed, atol=0.0)
    assert params.z == pytest.approx(
        lf_stationary_z(context, ref["P"], fixed),
        abs=1.0e-13,
    )
    assert prepared["lf_reference_mode"] == "fixed-lambda-cs-hf-seed"
    assert prepared["lf_hf_converged"] is None
    assert prepared["lf_hf_iterations"] == 0
    np.testing.assert_allclose(prepared["P"], ref["P"], atol=0.0)

    expected_fock = prepared["interaction"].build_fock(prepared["H_core"], prepared["P"])
    np.testing.assert_allclose(prepared["F"], expected_fock, atol=1.0e-13)


def test_fixed_lambda_reference_accepts_correlated_warm_density(monkeypatch) -> None:
    ref = _synthetic_reference()
    options = SimpleNamespace(eri_type="std", extra={})
    fixed = np.array([0.08, -0.03])
    warm_density = np.array([[1.76, 0.11], [0.11, 0.24]])

    def forbidden_static_optimizer(*args, **kwargs):
        raise AssertionError("warm fixed-lambda setup must not run LF-HF optimization")

    monkeypatch.setattr(
        driver_theory,
        "solve_lf_hf_reference",
        forbidden_static_optimizer,
    )

    prepared = driver_theory._prepare_lf_reference(
        ref,
        options,
        fixed_lambda=fixed,
        seed_density_override=warm_density,
    )
    context = prepared["lf_context"]
    params = prepared["lf_params"]

    assert prepared["lf_reference_mode"] == "fixed-lambda-warm-state"
    np.testing.assert_allclose(prepared["P"], warm_density, atol=0.0)
    np.testing.assert_allclose(params.lam, fixed, atol=0.0)
    assert params.z == pytest.approx(
        lf_stationary_z(context, warm_density, fixed),
        abs=1.0e-13,
    )
    expected_fock = prepared["interaction"].build_fock(prepared["H_core"], warm_density)
    np.testing.assert_allclose(prepared["F"], expected_fock, atol=1.0e-13)


def test_fixed_lambda_reference_rejects_wrong_dimension() -> None:
    ref = _synthetic_reference()
    options = SimpleNamespace(eri_type="std", extra={})
    with pytest.raises(ValueError, match="expected shape"):
        driver_theory._prepare_lf_reference(
            ref,
            options,
            fixed_lambda=np.array([0.1, 0.2, 0.3]),
        )


def test_public_fixed_lambda_wrapper_forwards_native_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = (1.0, -0.1, object())
    warm_state = object()

    def fake_run_gf2(mol, psi4_options, gf2_options, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(driver_theory, "run_gf2", fake_run_gf2)

    fixed = np.array([0.03, -0.02])
    result = driver_theory.run_lf_gf2_fixed_lambda(
        object(),
        {"basis": "synthetic"},
        {},
        lambda_fixed=fixed,
        cavity=object(),
        initial_state=warm_state,
    )

    assert result is sentinel
    assert captured["method"] == "lf-gf2"
    assert captured["lf_initial_state"] is warm_state
    np.testing.assert_allclose(captured["lf_fixed_lambda"], fixed, atol=0.0)


def test_state_returning_solver_collects_snapshot(monkeypatch) -> None:
    summary = object()
    state = object()

    def fake_run_gf2(mol, psi4_options, gf2_options, **kwargs):
        kwargs["lf_state_sink"]["state"] = state
        return 1.25, -0.15, summary

    monkeypatch.setattr(driver_theory, "run_gf2", fake_run_gf2)

    result = driver_theory.solve_lf_gf2_fixed_lambda(
        object(),
        {},
        {},
        lambda_fixed=np.array([0.01, -0.02]),
        cavity=object(),
    )

    assert result.core_energy == pytest.approx(1.25)
    assert result.correlation_energy == pytest.approx(-0.15)
    assert result.total_energy == pytest.approx(1.10)
    assert result.summary is summary
    assert result.state is state
