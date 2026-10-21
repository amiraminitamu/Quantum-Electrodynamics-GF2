# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Zero-boson-projected Lang--Firsov (LF) building blocks."

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize

from . import fock as fock_mod


def _sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(a) + np.asarray(a).conj().T)


@dataclass(frozen=True)
class LFParameters:
    """Diagonal LF displacement parameters for one photon mode."""

    lam: np.ndarray
    z: float

    def __post_init__(self) -> None:
        lam = np.asarray(self.lam, dtype=float)
        if lam.ndim != 1 or not np.all(np.isfinite(lam)):
            raise ValueError("LF lambda parameters must be a finite 1D array.")
        if not np.isfinite(self.z):
            raise ValueError("LF coherent displacement z must be finite.")
        object.__setattr__(self, "lam", lam.copy())
        object.__setattr__(self, "z", float(self.z))


class _PreparedLFParameters(LFParameters):
    "Marker for an optimizer result whose lambda stabilization is complete."


@dataclass(frozen=True)
class LFOrthogonalContext:
    """Bare Pauli--Fierz quantities in the fixed orthonormal LF basis."""

    X: np.ndarray
    X_inv: np.ndarray
    h_pf_orth: np.ndarray
    eri_pf_orth: np.ndarray
    g_pf_orth: np.ndarray
    omega: float
    e_nuc: float

    @classmethod
    def from_ao(
        cls,
        *,
        overlap: np.ndarray,
        s_inv_half: np.ndarray,
        h_pf_ao: np.ndarray,
        eri_pf_ao: np.ndarray,
        g_pf_ao: np.ndarray,
        omega: float,
        e_nuc: float,
    ) -> "LFOrthogonalContext":
        s = np.asarray(overlap, dtype=float)
        x = np.asarray(s_inv_half, dtype=float)
        h = np.asarray(h_pf_ao, dtype=float)
        eri = np.asarray(eri_pf_ao, dtype=float)
        g = np.asarray(g_pf_ao, dtype=float)
        n = s.shape[0]
        if s.shape != (n, n) or x.shape != (n, n):
            raise ValueError("overlap and s_inv_half must be square and compatible.")
        if h.shape != (n, n) or g.shape != (n, n):
            raise ValueError("h_pf_ao and g_pf_ao must match the AO dimension.")
        if eri.shape != (n, n, n, n):
            raise ValueError("eri_pf_ao must have shape (nao,nao,nao,nao).")
        if not np.isfinite(omega) or omega <= 0.0:
            raise ValueError("LF currently requires one finite positive photon frequency.")

        x_inv = np.linalg.inv(x)
        h_orth = _sym(x.T @ h @ x)
        g_orth = _sym(x.T @ g @ x)
        eri_orth = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, x, x, x, x, optimize=True)
        return cls(
            X=x.copy(),
            X_inv=x_inv,
            h_pf_orth=h_orth,
            eri_pf_orth=np.asarray(eri_orth, dtype=float),
            g_pf_orth=g_orth,
            omega=float(omega),
            e_nuc=float(e_nuc),
        )

    @property
    def nbf(self) -> int:
        return int(self.h_pf_orth.shape[0])

    def density_to_orth(self, density_ao: np.ndarray) -> np.ndarray:
        """Transform a contravariant AO density so ``Tr(P_orth)=Tr(P_AO S)``."""
        p = np.asarray(density_ao, dtype=float)
        if p.shape != (self.nbf, self.nbf):
            raise ValueError("density_ao has an incompatible shape.")
        return _sym(self.X_inv @ p @ self.X_inv.T)

    def one_body_to_ao(self, value_orth: np.ndarray) -> np.ndarray:
        return _sym(self.X_inv.T @ value_orth @ self.X_inv)

    def eri_to_ao(self, eri_orth: np.ndarray) -> np.ndarray:
        return np.einsum(
            "ijkl,pi,qj,rk,sl->pqrs",
            eri_orth,
            self.X_inv,
            self.X_inv,
            self.X_inv,
            self.X_inv,
            optimize=True,
        )


@dataclass(frozen=True)
class LFEffectiveData:
    """Projected LF Hamiltonian and residual linear vertex."""

    params: LFParameters
    fc1: np.ndarray
    fc2: np.ndarray
    h_orth: np.ndarray
    eri_orth: np.ndarray
    g_lf_orth: np.ndarray
    h_ao: np.ndarray
    eri_ao: np.ndarray
    g_lf_ao: np.ndarray


@dataclass(frozen=True)
class LFOptimizationResult:
    """Result of the fixed-density reduced LF parameter minimization."""

    params: LFParameters
    raw_params: LFParameters
    objective: float
    raw_objective: float
    success: bool
    message: str
    niter: int
    ntrial: int
    gradient_norm: float


@dataclass(frozen=True)
class LFHFResult:
    context: LFOrthogonalContext
    params: LFParameters
    effective: LFEffectiveData
    density_ao: np.ndarray
    fock_ao: np.ndarray
    energy: float
    converged: bool
    iterations: int
    residuals: dict[str, float]


def franck_condon_one(lam: np.ndarray) -> np.ndarray:
    """``FC1[p,q] = exp[-(lambda_q-lambda_p)^2/2]``."""
    lam = np.asarray(lam, dtype=float)
    delta = lam[None, :] - lam[:, None]
    return np.exp(-0.5 * delta * delta)


def franck_condon_two(lam: np.ndarray) -> np.ndarray:
    """Vacuum cloud factor for a two-electron operator in chemist notation."""
    lam = np.asarray(lam, dtype=float)
    delta = lam[None, :] - lam[:, None]
    total = delta[:, :, None, None] + delta[None, None, :, :]
    return np.exp(-0.5 * total * total)


def build_lf_effective_orth(
    context: LFOrthogonalContext,
    params: LFParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    "Build ``h_LF``, ``v_LF`` and ``g_LF`` in the fixed orthonormal basis."
    lam = np.asarray(params.lam, dtype=float)
    if lam.shape != (context.nbf,):
        raise ValueError("LF lambda vector must contain one value per LF orbital.")
    z = float(params.z)
    omega = context.omega
    g = context.g_pf_orth

    fc1 = franck_condon_one(lam)
    fc2 = franck_condon_two(lam)

    hbar = context.h_pf_orth + 2.0 * z * g - lam[:, None] * g - g * lam[None, :]
    h_lf = np.asarray(hbar * fc1, dtype=float)
    diag = np.diag_indices(context.nbf)
    h_lf[diag] += omega * lam * lam - 2.0 * omega * z * lam
    h_lf = _sym(h_lf)

    eri_lf = np.asarray(context.eri_pf_orth * fc2, dtype=float).copy()

    for p in range(context.nbf):
        for q in range(context.nbf):
            eri_lf[p, p, q, q] += 2.0 * omega * lam[p] * lam[q]

    for p in range(context.nbf):
        for q in range(context.nbf):
            pref = -2.0 * g[p, q] * fc1[p, q]
            for r in range(context.nbf):
                eri_lf[p, q, r, r] += pref * lam[r]
                eri_lf[r, r, p, q] += pref * lam[r]

    g_lf = np.asarray(fc1 * g, dtype=float)
    g_lf[diag] -= omega * lam
    g_lf = _sym(g_lf)

    return h_lf, eri_lf, g_lf, fc1, fc2


def build_lf_effective(
    context: LFOrthogonalContext,
    params: LFParameters,
) -> LFEffectiveData:
    h_orth, eri_orth, g_lf_orth, fc1, fc2 = build_lf_effective_orth(context, params)
    return LFEffectiveData(
        params=params,
        fc1=fc1,
        fc2=fc2,
        h_orth=h_orth,
        eri_orth=eri_orth,
        g_lf_orth=g_lf_orth,
        h_ao=context.one_body_to_ao(h_orth),
        eri_ao=context.eri_to_ao(eri_orth),
        g_lf_ao=context.one_body_to_ao(g_lf_orth),
    )


def build_lf_fock(
    h: np.ndarray,
    density: np.ndarray,
    eri: np.ndarray,
) -> np.ndarray:
    "RHF Fock matrix for the reduced-symmetry LF two-electron tensor."
    h = np.asarray(h, dtype=float)
    p = np.asarray(density, dtype=float)
    v = np.asarray(eri, dtype=float)

    j = np.einsum(
        "pqrs,rs->pq",
        v,
        p,
        optimize=True,
    )

    k = np.einsum(
        "psrq,rs->pq",
        v,
        p,
        optimize=True,
    )

    return _sym(h + j - 0.5 * k)


def lf_static_energy(
    context: LFOrthogonalContext,
    density_ao: np.ndarray,
    params: LFParameters,
) -> float:
    """Evaluate the zero-boson LF static/core energy at fixed density."""
    p_orth = context.density_to_orth(density_ao)
    h, eri, _, _, _ = build_lf_effective_orth(context, params)
    f = build_lf_fock(h, p_orth, eri)
    return float(
        fock_mod.core_energy(f, h, p_orth, context.e_nuc) + context.omega * params.z * params.z
    )


def lf_stationary_z(
    context: LFOrthogonalContext,
    density_ao: np.ndarray,
    lam: np.ndarray,
) -> float:
    """Return the exact fixed-density stationary coherent displacement."""
    p_orth = context.density_to_orth(density_ao)
    trial = LFParameters(np.asarray(lam, dtype=float), 0.0)
    _, _, g_lf, _, _ = build_lf_effective_orth(context, trial)
    a_lf = float(np.einsum("ij,ji->", g_lf, p_orth, optimize=True).real)
    return -a_lf / context.omega


def parameters_from_lambda(
    context: LFOrthogonalContext,
    density_ao: np.ndarray,
    lam: np.ndarray,
) -> LFParameters:
    """Construct LF parameters with ``z`` eliminated analytically."""
    lam = np.asarray(lam, dtype=float)
    return LFParameters(lam, lf_stationary_z(context, density_ao, lam))


def initial_lf_parameters(
    context: LFOrthogonalContext,
    density_ao: np.ndarray,
) -> LFParameters:
    return parameters_from_lambda(context, density_ao, np.zeros(context.nbf))


def _optional_positive_float(
    options: Mapping[str, Any],
    key: str,
) -> float | None:
    value = options.get(key)
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "off", "inf", "infinity"}:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    if number <= 0.0:
        raise ValueError(f"{key} must be positive when specified.")
    return number


def _optimizer_bounds(
    method: str,
    options: Mapping[str, Any],
    nbf: int,
) -> list[tuple[float, float]] | None:
    bound = _optional_positive_float(options, "lf_lambda_bound")
    if bound is None:
        return None
    bounded_methods = {
        "l-bfgs-b",
        "tnc",
        "slsqp",
        "powell",
        "nelder-mead",
        "cobyla",
        "cobyqa",
        "trust-constr",
    }
    if method.lower() not in bounded_methods:
        raise ValueError(
            f"lf_lambda_bound is incompatible with lf_opt_method={method!r}; "
            "remove the bound for BFGS or use a bounded SciPy optimizer such "
            "as L-BFGS-B."
        )
    return [(-bound, bound)] * nbf


def _trial_starts(
    context: LFOrthogonalContext,
    initial_lam: np.ndarray,
    ntrial: int,
    seed: int,
) -> list[np.ndarray]:
    """Deterministic multi-start lambda guesses inspired by variational LF work."""
    if ntrial < 1:
        raise ValueError("lf_opt_ntrial must be at least 1.")
    starts = [np.asarray(initial_lam, dtype=float).copy()]
    if ntrial == 1:
        return starts

    starts.append(np.zeros_like(initial_lam, dtype=float))
    if ntrial <= 2:
        return starts[:ntrial]

    rng = np.random.default_rng(seed)
    scale = float(np.max(np.abs(context.g_pf_orth)) / context.omega)
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    for trial in range(2, ntrial):
        guess = (rng.random(context.nbf) - 0.5) * scale
        if trial % 2 == 0:
            guess *= 0.1
        starts.append(guess)
    return starts


def _stabilize_lambda_update(
    old_lam: np.ndarray,
    target_lam: np.ndarray,
    options: Mapping[str, Any],
) -> np.ndarray:
    "Apply only the outer fixed-point stabilization to ``lambda``."
    alpha = float(options.get("lf_param_mix", 1.0))
    if not (0.0 < alpha <= 1.0):
        raise ValueError("lf_param_mix must be in (0,1].")

    old = np.asarray(old_lam, dtype=float)
    target = np.asarray(target_lam, dtype=float)
    accepted = old + alpha * (target - old)

    max_dlam = _optional_positive_float(options, "lf_max_dlambda")
    if max_dlam is not None:
        delta = accepted - old
        norm = float(np.linalg.norm(delta))
        if norm > max_dlam:
            accepted = old + delta * (max_dlam / norm)
    return accepted


def optimize_lf_parameters_fixed_density(
    context: LFOrthogonalContext,
    density_ao: np.ndarray,
    initial: LFParameters,
    options: Mapping[str, Any],
) -> LFOptimizationResult:
    "Minimize the reduced LF static functional at fixed correlated density."
    p = _sym(np.asarray(density_ao, dtype=float))
    if initial.lam.shape != (context.nbf,):
        raise ValueError("Initial LF lambda vector has an incompatible shape.")

    method = str(options.get("lf_opt_method", "BFGS")).strip()
    if not method:
        raise ValueError("lf_opt_method must be a non-empty SciPy method name.")
    bounds = _optimizer_bounds(method, options, context.nbf)

    penalty = float(options.get("lf_penalty", 0.0))
    if penalty < 0.0:
        raise ValueError("lf_penalty must be non-negative.")

    ntrial = int(options.get("lf_opt_ntrial", 1))
    seed = int(options.get("lf_random_seed", 10086))
    maxiter = int(options.get("lf_opt_maxiter", 1000))
    if maxiter < 1:
        raise ValueError("lf_opt_maxiter must be at least 1.")
    tol = float(options.get("lf_opt_tol", 1.0e-8))
    gtol = float(options.get("lf_opt_gtol", 1.0e-6))
    if tol <= 0.0 or gtol <= 0.0:
        raise ValueError("lf_opt_tol and lf_opt_gtol must be positive.")

    initial_lam = np.asarray(initial.lam, dtype=float).copy()

    def reduced_objective(lam: np.ndarray) -> float:
        lam = np.asarray(lam, dtype=float)
        params = parameters_from_lambda(context, p, lam)
        value = lf_static_energy(context, p, params)
        if penalty:
            delta = lam - initial_lam
            value += penalty * float(np.dot(delta, delta))
        return float(value) if np.isfinite(value) else 1.0e100

    starts = _trial_starts(context, initial_lam, ntrial, seed)
    best_lam: np.ndarray | None = None
    best_value = np.inf
    best_success = False
    best_message = "no optimizer trial completed"
    best_gradient = np.inf
    total_niter = 0
    chosen_trial = -1

    for trial_index, start in enumerate(starts):
        x0 = np.asarray(start, dtype=float).copy()
        if bounds is not None:
            lo = np.asarray([b[0] for b in bounds])
            hi = np.asarray([b[1] for b in bounds])
            x0 = np.clip(x0, lo, hi)

        local_best_x = x0.copy()
        local_best_value = reduced_objective(local_best_x)

        def objective(lam: np.ndarray) -> float:
            nonlocal local_best_x, local_best_value
            value = reduced_objective(lam)
            if np.isfinite(value) and value < local_best_value:
                local_best_value = float(value)
                local_best_x = np.asarray(lam, dtype=float).copy()
            return value

        opt_options: dict[str, Any] = {
            "maxiter": maxiter,
            "gtol": gtol,
        }
        result = minimize(
            objective,
            x0,
            method=method,
            jac=None,
            bounds=bounds,
            tol=tol,
            options=opt_options,
        )
        total_niter += int(getattr(result, "nit", 0))

        candidate = (
            np.asarray(result.x, dtype=float)
            if np.all(np.isfinite(np.asarray(result.x, dtype=float)))
            else local_best_x
        )
        candidate_value = reduced_objective(candidate)
        if local_best_value < candidate_value:
            candidate = local_best_x
            candidate_value = local_best_value

        jac = getattr(result, "jac", None)
        if jac is None:
            gradient_norm = np.inf
        else:
            jac_array = np.asarray(jac, dtype=float)
            gradient_norm = (
                float(np.linalg.norm(jac_array)) if np.all(np.isfinite(jac_array)) else np.inf
            )

        if candidate_value < best_value:
            best_lam = candidate.copy()
            best_value = float(candidate_value)
            best_success = bool(result.success)
            best_message = str(result.message)
            best_gradient = gradient_norm
            chosen_trial = trial_index

    if best_lam is None:
        raise RuntimeError("All LF optimizer trials failed to produce a finite point.")

    raw_params = parameters_from_lambda(context, p, best_lam)
    accepted_lam = _stabilize_lambda_update(initial_lam, best_lam, options)
    accepted_z = lf_stationary_z(context, p, accepted_lam)
    accepted_params = _PreparedLFParameters(accepted_lam, accepted_z)
    accepted_objective = reduced_objective(accepted_lam)

    return LFOptimizationResult(
        params=accepted_params,
        raw_params=raw_params,
        objective=float(accepted_objective),
        raw_objective=float(best_value),
        success=best_success,
        message=f"trial {chosen_trial + 1}/{ntrial}: {best_message}",
        niter=total_niter,
        ntrial=ntrial,
        gradient_norm=float(best_gradient),
    )


def damp_lf_parameter_update(
    old: LFParameters,
    target: LFParameters,
    options: Mapping[str, Any],
    *,
    prefix: str = "lf",
) -> LFParameters:
    "Apply legacy damping, unless the revised optimizer already prepared it."
    if isinstance(target, _PreparedLFParameters):
        return LFParameters(target.lam, target.z)

    alpha = float(options.get(f"{prefix}_param_mix", options.get("lf_param_mix", 1.0)))
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"{prefix}_param_mix must be in (0,1].")
    lam = old.lam + alpha * (target.lam - old.lam)
    z = old.z + alpha * (target.z - old.z)

    max_dlam = _optional_positive_float(options, "lf_max_dlambda")
    max_dz = _optional_positive_float(options, "lf_max_dz")
    delta = lam - old.lam
    norm = float(np.linalg.norm(delta))
    if max_dlam is not None and norm > max_dlam:
        lam = old.lam + delta * (max_dlam / norm)
    if max_dz is not None:
        dz = float(z - old.z)
        z = old.z + float(np.clip(dz, -max_dz, max_dz))
    return LFParameters(lam, z)


def solve_lf_hf_reference(
    context: LFOrthogonalContext,
    density_ao: np.ndarray,
    nocc: int,
    options: Mapping[str, Any],
) -> LFHFResult:
    "Self-consistently optimize the projected LF-HF reference."
    p = _sym(np.asarray(density_ao, dtype=float))
    params = initial_lf_parameters(context, p)
    e_last: float | None = None
    residuals = {
        "energy": np.inf,
        "density": np.inf,
        "lambda": np.inf,
        "z": np.inf,
        "optimizer_gradient": np.inf,
    }
    converged = False
    maxiter = int(options.get("maxiter_lf_hf", 1000))
    density_mix = float(options.get("lf_hf_mix", 0.30))
    if not (0.0 < density_mix <= 1.0):
        raise ValueError("lf_hf_mix must be in (0,1].")
    e_thresh = float(options.get("thresh_lf_hf_energy", 1.0e-8))
    p_thresh = float(options.get("thresh_lf_hf_density", 1.0e-7))
    param_thresh = float(options.get("lf_param_thresh", 1.0e-6))
    hf_ntrial = int(options.get("lf_hf_ntrial", 5))
    if hf_ntrial < 1:
        raise ValueError("lf_hf_ntrial must be at least 1.")

    for iteration in range(1, maxiter + 1):
        opt_options = dict(options)
        opt_options["lf_param_mix"] = float(options.get("lf_hf_param_mix", 1.0))
        opt_options["lf_opt_ntrial"] = hf_ntrial if iteration == 1 else 1

        if "lf_hf_max_dlambda" in options:
            opt_options["lf_max_dlambda"] = options["lf_hf_max_dlambda"]
        else:
            opt_options.pop("lf_max_dlambda", None)

        opt = optimize_lf_parameters_fixed_density(context, p, params, opt_options)
        new_params = damp_lf_parameter_update(
            params,
            opt.params,
            opt_options,
            prefix="lf_hf",
        )
        effective = build_lf_effective(context, new_params)
        fock_ao = build_lf_fock(effective.h_ao, p, effective.eri_ao)

        f_orth = context.X.T @ fock_ao @ context.X
        _, u = np.linalg.eigh(_sym(f_orth))
        c_ao = context.X @ u
        c_occ = c_ao[:, :nocc]
        p_build = 2.0 * np.einsum("pi,qi->pq", c_occ, c_occ, optimize=True)
        p_new = _sym((1.0 - density_mix) * p + density_mix * p_build)

        new_params = parameters_from_lambda(context, p_new, new_params.lam)
        effective = build_lf_effective(context, new_params)
        energy = lf_static_energy(context, p_new, new_params)

        residuals = {
            "energy": float(abs(energy - e_last)) if e_last is not None else np.inf,
            "density": float(np.linalg.norm(p_new - p)),
            "lambda": float(np.linalg.norm(new_params.lam - params.lam)),
            "z": float(abs(new_params.z - params.z)),
            "optimizer_gradient": float(opt.gradient_norm),
        }
        p = p_new
        params = new_params
        e_last = energy
        if (
            residuals["energy"] < e_thresh
            and residuals["density"] < p_thresh
            and residuals["lambda"] < param_thresh
            and residuals["z"] < param_thresh
        ):
            converged = True
            break

    params = parameters_from_lambda(context, p, params.lam)
    effective = build_lf_effective(context, params)
    fock_ao = build_lf_fock(effective.h_ao, p, effective.eri_ao)
    energy = lf_static_energy(context, p, params)
    return LFHFResult(
        context=context,
        params=params,
        effective=effective,
        density_ao=p,
        fock_ao=fock_ao,
        energy=float(energy),
        converged=converged,
        iterations=iteration,
        residuals=residuals,
    )
