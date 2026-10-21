# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Reusable state objects for fixed-lambda projected LF-GF2 solves."

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def _finite_array(value: np.ndarray, *, name: str, complex_ok: bool) -> np.ndarray:
    arr = np.asarray(value)
    if not complex_ok and np.iscomplexobj(arr):
        if np.max(np.abs(arr.imag)) > 1.0e-14:
            raise ValueError(f"{name} must be real.")
        arr = arr.real
    dtype = np.complex128 if complex_ok and np.iscomplexobj(arr) else np.float64
    arr = np.asarray(arr, dtype=dtype)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr.copy()


def _digest_array(digest: Any, label: str, value: np.ndarray) -> None:
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        canonical = np.ascontiguousarray(arr, dtype=np.complex128)
    else:
        canonical = np.ascontiguousarray(arr, dtype=np.float64)
    digest.update(label.encode("utf-8"))
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(canonical.tobytes(order="C"))


def build_lf_problem_signature(
    *,
    h_pf: np.ndarray,
    eri_pf: np.ndarray,
    g_pf: np.ndarray,
    overlap: np.ndarray,
    omega: float,
    e_nuc: float,
    nelec: float,
    nocc: int,
) -> str:
    """Return a deterministic fingerprint for one fixed molecular QED problem."""
    digest = hashlib.sha256()
    _digest_array(digest, "h_pf", h_pf)
    _digest_array(digest, "eri_pf", eri_pf)
    _digest_array(digest, "g_pf", g_pf)
    _digest_array(digest, "overlap", overlap)
    _digest_array(
        digest,
        "scalars",
        np.array([float(omega), float(e_nuc), float(nelec), float(nocc)]),
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class LFGF2FixedState:
    "Converged fixed-lambda LF-GF2 state suitable for a nearby warm start."

    problem_signature: str
    lambda_fixed: np.ndarray
    xi: float
    beta: float
    fermionic_iomega: np.ndarray
    imaginary_time_grid: np.ndarray
    bosonic_inu: np.ndarray
    density: np.ndarray
    fock: np.ndarray
    mu: float
    G_iw: np.ndarray
    Sigma_iw: np.ndarray
    D_iv: np.ndarray
    Pi_iv: np.ndarray
    core_energy: float
    correlation_energy: float
    converged: bool
    outer_iterations: int

    def __post_init__(self) -> None:
        if not self.problem_signature:
            raise ValueError("problem_signature must be non-empty.")
        if not np.isfinite(self.xi):
            raise ValueError("xi must be finite.")
        if not np.isfinite(self.beta) or self.beta <= 0.0:
            raise ValueError("beta must be finite and positive.")
        if not np.isfinite(self.mu):
            raise ValueError("mu must be finite.")
        if not np.isfinite(self.core_energy) or not np.isfinite(self.correlation_energy):
            raise ValueError("state energies must be finite.")
        if int(self.outer_iterations) < 0:
            raise ValueError("outer_iterations must be non-negative.")

        object.__setattr__(
            self,
            "lambda_fixed",
            _finite_array(self.lambda_fixed, name="lambda_fixed", complex_ok=False),
        )
        object.__setattr__(
            self,
            "fermionic_iomega",
            _finite_array(self.fermionic_iomega, name="fermionic_iomega", complex_ok=False),
        )
        object.__setattr__(
            self,
            "imaginary_time_grid",
            _finite_array(
                self.imaginary_time_grid,
                name="imaginary_time_grid",
                complex_ok=False,
            ),
        )
        object.__setattr__(
            self,
            "bosonic_inu",
            _finite_array(self.bosonic_inu, name="bosonic_inu", complex_ok=False),
        )
        object.__setattr__(
            self,
            "density",
            _finite_array(self.density, name="density", complex_ok=False),
        )
        object.__setattr__(
            self,
            "fock",
            _finite_array(self.fock, name="fock", complex_ok=False),
        )
        object.__setattr__(
            self,
            "G_iw",
            _finite_array(self.G_iw, name="G_iw", complex_ok=True),
        )
        object.__setattr__(
            self,
            "Sigma_iw",
            _finite_array(self.Sigma_iw, name="Sigma_iw", complex_ok=True),
        )
        object.__setattr__(
            self,
            "D_iv",
            _finite_array(self.D_iv, name="D_iv", complex_ok=True),
        )
        object.__setattr__(
            self,
            "Pi_iv",
            _finite_array(self.Pi_iv, name="Pi_iv", complex_ok=True),
        )
        object.__setattr__(self, "xi", float(self.xi))
        object.__setattr__(self, "beta", float(self.beta))
        object.__setattr__(self, "mu", float(self.mu))
        object.__setattr__(self, "core_energy", float(self.core_energy))
        object.__setattr__(self, "correlation_energy", float(self.correlation_energy))
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "outer_iterations", int(self.outer_iterations))

    @property
    def total_energy(self) -> float:
        return float(self.core_energy + self.correlation_energy)


@dataclass(frozen=True)
class LFGF2FixedResult:
    """Fixed-lambda correlated objective value together with its reusable state."""

    core_energy: float
    correlation_energy: float
    summary: Any
    state: LFGF2FixedState

    @property
    def total_energy(self) -> float:
        return float(self.core_energy + self.correlation_energy)


def save_lf_gf2_state(path: str | Path, state: LFGF2FixedState) -> Path:
    "Persist a reusable correlated LF-GF2 state without pickle."
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            format_version=np.asarray(1, dtype=np.int64),
            problem_signature=np.asarray(state.problem_signature),
            lambda_fixed=state.lambda_fixed,
            xi=np.asarray(state.xi, dtype=np.float64),
            beta=np.asarray(state.beta, dtype=np.float64),
            fermionic_iomega=state.fermionic_iomega,
            imaginary_time_grid=state.imaginary_time_grid,
            bosonic_inu=state.bosonic_inu,
            density=state.density,
            fock=state.fock,
            mu=np.asarray(state.mu, dtype=np.float64),
            G_iw=state.G_iw,
            Sigma_iw=state.Sigma_iw,
            D_iv=state.D_iv,
            Pi_iv=state.Pi_iv,
            core_energy=np.asarray(state.core_energy, dtype=np.float64),
            correlation_energy=np.asarray(state.correlation_energy, dtype=np.float64),
            converged=np.asarray(int(state.converged), dtype=np.int8),
            outer_iterations=np.asarray(state.outer_iterations, dtype=np.int64),
        )
    temporary.replace(target)
    return target


def load_lf_gf2_state(path: str | Path) -> LFGF2FixedState:
    """Load a state written by :func:`save_lf_gf2_state`."""
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        required = {
            "format_version",
            "problem_signature",
            "lambda_fixed",
            "xi",
            "beta",
            "fermionic_iomega",
            "imaginary_time_grid",
            "bosonic_inu",
            "density",
            "fock",
            "mu",
            "G_iw",
            "Sigma_iw",
            "D_iv",
            "Pi_iv",
            "core_energy",
            "correlation_energy",
            "converged",
            "outer_iterations",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(
                "LF-GF2 state checkpoint is missing required fields: " + ", ".join(missing)
            )
        version = int(np.asarray(data["format_version"]).item())
        if version != 1:
            raise ValueError(f"Unsupported LF-GF2 state checkpoint version {version}.")
        return LFGF2FixedState(
            problem_signature=str(np.asarray(data["problem_signature"]).item()),
            lambda_fixed=np.asarray(data["lambda_fixed"]),
            xi=float(np.asarray(data["xi"]).item()),
            beta=float(np.asarray(data["beta"]).item()),
            fermionic_iomega=np.asarray(data["fermionic_iomega"]),
            imaginary_time_grid=np.asarray(data["imaginary_time_grid"]),
            bosonic_inu=np.asarray(data["bosonic_inu"]),
            density=np.asarray(data["density"]),
            fock=np.asarray(data["fock"]),
            mu=float(np.asarray(data["mu"]).item()),
            G_iw=np.asarray(data["G_iw"]),
            Sigma_iw=np.asarray(data["Sigma_iw"]),
            D_iv=np.asarray(data["D_iv"]),
            Pi_iv=np.asarray(data["Pi_iv"]),
            core_energy=float(np.asarray(data["core_energy"]).item()),
            correlation_energy=float(np.asarray(data["correlation_energy"]).item()),
            converged=bool(int(np.asarray(data["converged"]).item())),
            outer_iterations=int(np.asarray(data["outer_iterations"]).item()),
        )


def validate_lf_warm_start_problem(
    state: LFGF2FixedState,
    *,
    problem_signature: str,
    nbf: int,
) -> None:
    """Reject a state that does not belong to the same converged QED problem."""
    if not state.converged:
        raise ValueError("LF-GF2 warm starts require a previously converged state.")
    if state.problem_signature != problem_signature:
        raise ValueError(
            "LF-GF2 warm-start state belongs to a different molecular/cavity "
            "problem. Reuse is currently allowed only when geometry, basis, "
            "Pauli--Fierz tensors, electron count, and cavity frequency are unchanged."
        )
    if state.lambda_fixed.shape != (nbf,):
        raise ValueError("Warm-start lambda dimension is incompatible with this problem.")
    if state.density.shape != (nbf, nbf) or state.fock.shape != (nbf, nbf):
        raise ValueError("Warm-start density/Fock dimensions are incompatible.")


def validate_lf_warm_start_grids(
    state: LFGF2FixedState,
    *,
    beta: float,
    fermionic_iomega: np.ndarray,
    imaginary_time_grid: np.ndarray,
    bosonic_inu: np.ndarray,
    nbf: int,
) -> None:
    """Require identical Matsubara/time grids before reusing propagators."""
    if not np.isclose(state.beta, float(beta), rtol=0.0, atol=1.0e-14):
        raise ValueError("Warm-start beta differs from the requested calculation.")

    def same_grid(old: np.ndarray, new: np.ndarray, name: str) -> None:
        new_arr = np.asarray(new, dtype=float)
        if old.shape != new_arr.shape or not np.allclose(old, new_arr, rtol=0.0, atol=1.0e-14):
            raise ValueError(f"Warm-start {name} grid is incompatible.")

    same_grid(state.fermionic_iomega, fermionic_iomega, "fermionic Matsubara")
    same_grid(state.imaginary_time_grid, imaginary_time_grid, "imaginary-time")
    same_grid(state.bosonic_inu, bosonic_inu, "bosonic Matsubara")

    nif = int(np.asarray(fermionic_iomega).size)
    nnu = int(np.asarray(bosonic_inu).size)
    if state.G_iw.shape != (nif, nbf, nbf):
        raise ValueError("Warm-start G(iw) dimensions are incompatible.")
    if state.Sigma_iw.shape != (nif, nbf, nbf):
        raise ValueError("Warm-start Sigma(iw) dimensions are incompatible.")
    if state.D_iv.shape != (nnu, 1, 1) or state.Pi_iv.shape != (nnu, 1, 1):
        raise ValueError("Warm-start bosonic propagator/polarization dimensions are incompatible.")
