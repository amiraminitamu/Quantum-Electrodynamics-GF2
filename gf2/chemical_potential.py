# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Chemical-potential tuning so that ``Tr[ρ S] = N_e`` is satisfied."

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.optimize import brentq

from . import greens_function as gf
from . import spline_ft
from .reporting import fortran_real
from .utils import nelec_from_density


def _density_from_state(
    F: np.ndarray,
    S: np.ndarray,
    S_inv_half: np.ndarray,
    Sigma_if: np.ndarray,
    ifreq_grid: np.ndarray,
    B_beta: np.ndarray,
    mu: float,
    beta: float,
    nbf: int,
    orth_state: gf.FockOrthoState,
    spin_factor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    "Helper: given μ, build G(iω), G_0(iω), ρ(τ=β) and the resulting N_e."
    G_iw, G0_iw = gf.build_g_g0_ifreq(nbf, mu, ifreq_grid, S, F, Sigma_if)
    delta_re, _ = spline_ft.giw_to_gtau_beta(B_beta, G_iw, G0_iw, beta)
    G0_tb = gf.build_g0_itime_beta(orth_state, S_inv_half, mu, beta)
    G_tb = G0_tb + delta_re

    # antisymmetric component even when F, S, and Sigma obey their exact
    # symmetries (most visibly for polar, slowly converging systems).  Remove

    # is the symmetric projection; the conjugation keeps the convention valid

    rho = -spin_factor * G_tb
    rho = 0.5 * (rho + rho.conj().T)
    rho = np.real_if_close(rho)
    return G_iw, G0_iw, rho, nelec_from_density(rho, S)


def tune_spin_population(
    F: np.ndarray,
    S: np.ndarray,
    S_inv_half: np.ndarray,
    Sigma_if: np.ndarray,
    ifreq_grid: np.ndarray,
    B_beta: np.ndarray,
    target: int,
    mu_init: float,
    *,
    beta: float,
    thresh: float,
    maxiter: int,
    reporter: Any | None = None,
    use_newton: bool = True,
    damping: float = 1.0,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Tune a one-spin population with an adaptive bracket and safe bisection.

    The separate alpha/beta searches hold N_alpha and N_beta fixed.  The
    bracket covers zero and fully occupied channels without HOMO/LUMO indices.
    """
    nbf = F.shape[0]
    if not 0 <= target <= nbf:
        raise ValueError("Spin population must lie between zero and basis size.")
    state = gf.prepare_fock_orth(F, S_inv_half)

    def evaluate(mu: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        return _density_from_state(
            F, S, S_inv_half, Sigma_if, ifreq_grid, B_beta,
            mu, beta, nbf, state, spin_factor=1.0
        )

    width = max(1.0, float(np.ptp(state.orth_evals)), 20.0 / beta)
    lo = min(mu_init, float(state.orth_evals[0])) - width
    hi = max(mu_init, float(state.orth_evals[-1])) + width
    for _ in range(24):
        if evaluate(lo)[3] <= target + thresh:
            break
        lo -= width
        width *= 2.0
    width = max(1.0, float(np.ptp(state.orth_evals)), 20.0 / beta)
    for _ in range(24):
        if evaluate(hi)[3] >= target - thresh:
            break
        hi += width
        width *= 2.0

    if maxiter < 1:
        raise ValueError("maxiter must be positive.")
    mu = float(np.clip(mu_init, lo, hi))
    result = evaluate(mu)
    for step in range(maxiter):
        delta = result[3] - target
        if abs(delta) < thresh:
            break
        if delta > 0.0:
            hi = mu
        else:
            lo = mu
        candidate = 0.5 * (lo + hi)
        if use_newton:
            g_iw, g0_iw = result[:2]
            dg_iw = -g_iw @ (g_iw @ S)
            dg0_iw = -g0_iw @ (g0_iw @ S)
            delta_beta, _ = spline_ft.giw_to_gtau_beta(
                B_beta, dg_iw, dg0_iw, beta
            )
            dg0_beta = gf.build_g0_itime_beta_derivative(
                state, S_inv_half, mu, beta
            )
            dn = -np.einsum(
                "ij,ji->", dg0_beta + delta_beta, S, optimize=True
            ).real
            if np.isfinite(dn) and dn > 1e-14:
                proposal = mu - damping * delta / dn
                if lo < proposal < hi:
                    candidate = float(proposal)
        mu = candidate
        result = evaluate(mu)
    _log_mu_tune("Spin bisection", step + 1, mu, target - result[3], reporter)
    return mu, result[2], result[0], result[1]



def tune_bisection(
    nbf: int,
    mu_init: float,
    ifreq_grid: np.ndarray,
    S: np.ndarray,
    S_inv_half: np.ndarray,
    F: np.ndarray,
    Sigma_if: np.ndarray,
    Ne: int,
    B_beta: np.ndarray,
    mo_eval: np.ndarray,
    nocc: int,
    *,
    beta: float,
    thresh: float,
    maxiter: int,
    reporter: Any | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Bisection between HOMO and LUMO of ``F`` until ``|N_e^calc - N_e| < thresh``."""
    neg_bound = float(mo_eval[nocc - 1])
    pos_bound = float(mo_eval[nocc])
    mu = mu_init
    rho = G_iw = G0_iw = None
    counter = 0
    delta_ne = 0.0

    orth_state = gf.prepare_fock_orth(F, S_inv_half)

    for step in range(maxiter):
        counter += 1
        G_iw, G0_iw, rho, ne = _density_from_state(
            F,
            S,
            S_inv_half,
            Sigma_if,
            ifreq_grid,
            B_beta,
            mu,
            beta,
            nbf,
            orth_state,
        )
        delta_ne = Ne - ne
        if abs(delta_ne) < thresh:
            break
        if delta_ne < 0:
            pos_bound = mu
        else:
            neg_bound = mu
        mu = neg_bound + (pos_bound - neg_bound) / 2.0

    _log_mu_tune("Bisection", counter, mu, delta_ne, reporter)
    return mu, rho, G_iw, G0_iw


def _log_mu_tune(
    method: str,
    steps: int,
    mu: float,
    delta_ne: float,
    reporter: Any | None,
) -> None:
    msg = (
        f"{method}: {steps:4d} steps   mu = {mu:16.10f}"
        f"   delta Ne = {fortran_real(delta_ne)}"
    )
    if reporter is not None:
        reporter.detail(msg)
    else:
        print(f"        {msg}", flush=True)


def _linear_fit_guess(
    mu: float,
    delta_ne: float,
    mu_init: float,
    F: np.ndarray,
    S: np.ndarray,
    S_inv_half: np.ndarray,
    Sigma_if: np.ndarray,
    ifreq_grid: np.ndarray,
    B_beta: np.ndarray,
    Ne: int,
    mu_Ne: float,
    span: float,
    fit_width: float,
    n_fit_pts: int,
    beta: float,
    nbf: int,
) -> float:
    """Two-point linear fit of ``N_e(μ)`` used to reseed Newton from a bad guess."""
    mu_pts = np.zeros(2)
    ne_pts = np.zeros(2)
    mu_pts[0] = mu
    ne_pts[0] = mu_Ne

    orth_state = gf.prepare_fock_orth(F, S_inv_half)
    mu_new = mu - fit_width if delta_ne > 0.0 else mu + fit_width
    _, _, _, ne_new = _density_from_state(
        F,
        S,
        S_inv_half,
        Sigma_if,
        ifreq_grid,
        B_beta,
        mu_new,
        beta,
        nbf,
        orth_state,
    )
    mu_pts[1] = mu_new
    ne_pts[1] = ne_new

    grid = np.linspace(mu_init - span, mu_init + span, n_fit_pts)
    coeffs = poly.polyfit(mu_pts, ne_pts, 1)
    fit = poly.polyval(grid, coeffs)
    return float(grid[np.argmin(np.abs(fit - Ne))])


def tune_newton(
    nbf: int,
    mu_init: float,
    ifreq_grid: np.ndarray,
    S: np.ndarray,
    S_inv_half: np.ndarray,
    F: np.ndarray,
    Sigma_if: np.ndarray,
    Ne: int,
    B_beta: np.ndarray,
    mo_eval: np.ndarray,
    nocc: int,
    *,
    beta: float,
    thresh: float,
    maxiter: int,
    damping: float,
    mu_guess_thresh: float,
    fit_width: float,
    n_fit_pts: int,
    reporter: Any | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    "Damped Newton search for the chemical potential."
    HOMO = float(mo_eval[nocc - 1])
    LUMO = float(mo_eval[nocc])
    ten_hl_gap = (LUMO - HOMO) * 0.1

    mu = mu_init
    mu_old = mu
    alpha = 1.0
    counter = 0

    delta_ne = 0.0
    delta_ne_old = 0.0
    good_guess = False
    G_iw = G0_iw = rho = None
    orth_state = gf.prepare_fock_orth(F, S_inv_half)

    for step in range(maxiter):
        counter += 1
        G_iw, G0_iw, rho, ne = _density_from_state(
            F,
            S,
            S_inv_half,
            Sigma_if,
            ifreq_grid,
            B_beta,
            mu,
            beta,
            nbf,
            orth_state,
        )
        delta_ne = ne - Ne
        if step == 0:
            delta_ne_old = delta_ne

        if abs(delta_ne) < thresh:
            break

        if step == 0 and abs(delta_ne) < mu_guess_thresh:
            good_guess = True
        elif step == 0 and abs(delta_ne) > mu_guess_thresh:
            if reporter is not None:
                reporter.warn(
                    "poor initial mu "
                    f"(delta Ne = {fortran_real(delta_ne)}); reseeding with linear fit"
                )
            else:
                print(
                    " !!!  poor initial mu "
                    f"(delta Ne = {fortran_real(delta_ne)}); reseeding with linear fit",
                    flush=True,
                )
            mu = _linear_fit_guess(
                mu,
                delta_ne,
                mu_init,
                F,
                S,
                S_inv_half,
                Sigma_if,
                ifreq_grid,
                B_beta,
                Ne,
                ne,
                ten_hl_gap,
                fit_width,
                n_fit_pts,
                beta,
                nbf,
            )

        if step == 1:
            good_guess = True

        if good_guess:
            dG_iw = -np.matmul(G_iw, np.matmul(G_iw, S))
            dG0_iw = -np.matmul(G0_iw, np.matmul(G0_iw, S))

            re_der, _ = spline_ft.giw_to_gtau_beta(B_beta, dG_iw, dG0_iw, beta)
            dG0_tb = gf.build_g0_itime_beta_derivative(orth_state, S_inv_half, mu, beta)
            dG_tb = -2.0 * (dG0_tb + re_der)
            dNe = np.einsum("ij,ji", dG_tb, S, optimize=True)

            if abs(delta_ne) < abs(delta_ne_old):
                mu_old = mu
                delta_ne_old = delta_ne
                mu = mu - alpha * delta_ne / dNe
            else:
                alpha *= damping
                mu = mu_old - alpha * delta_ne_old / dNe

    _log_mu_tune("Newton", counter, mu, delta_ne, reporter)
    return mu, rho, G_iw, G0_iw


def evaluate_restricted_population_at_mu(
    F: np.ndarray, S: np.ndarray, S_inv_half: np.ndarray,
    Sigma_if: np.ndarray, ifreq_grid: np.ndarray, B_beta: np.ndarray,
    mu: float, *, beta: float,
    on_evaluate: Callable[[], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """One Dyson evaluation, spin-summed density, count, and analytic count slope."""
    state = gf.prepare_fock_orth(F, S_inv_half)
    if on_evaluate is not None:
        on_evaluate()
    green, bare, density, count = _density_from_state(
        F, S, S_inv_half, Sigma_if, ifreq_grid, B_beta, mu, beta, F.shape[0], state
    )
    derivative_green = -green @ (green @ S)
    derivative_bare = -bare @ (bare @ S)
    delta_beta, _ = spline_ft.giw_to_gtau_beta(
        B_beta, derivative_green, derivative_bare, beta
    )
    derivative_bare_beta = gf.build_g0_itime_beta_derivative(
        state, S_inv_half, mu, beta
    )
    slope = float(-2.0 * np.einsum(
        "ij,ji->", derivative_bare_beta + delta_beta, S, optimize=True
    ).real)
    return density, green, bare, count, slope


def tune_restricted_population_brent(
    F: np.ndarray, S: np.ndarray, S_inv_half: np.ndarray,
    Sigma_if: np.ndarray, ifreq_grid: np.ndarray, B_beta: np.ndarray,
    target: int, mu_init: float, *, beta: float, thresh: float,
    maxiter: int, reporter: Any | None = None,
    on_evaluate: Callable[[], None] | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Bracket μ near the previous value and verify the final electron count."""
    nbf = F.shape[0]
    if not 0 <= target <= 2 * nbf:
        raise ValueError("Restricted population must lie between zero and twice the basis size.")
    if beta <= 0 or thresh <= 0 or maxiter < 1:
        raise ValueError("beta, thresh, and maxiter must be positive.")
    state = gf.prepare_fock_orth(F, S_inv_half)
    cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}

    def point(value: float) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, float], float]:
        key = float(value)
        if key not in cache:
            if on_evaluate is not None:
                on_evaluate()
            cache[key] = _density_from_state(
                F, S, S_inv_half, Sigma_if, ifreq_grid, B_beta,
                key, beta, nbf, state
            )
        result = cache[key]
        return result, float(result[3] - target)

    current, error = point(mu_init)
    if abs(error) < thresh:
        return float(mu_init), current[2], current[0], current[1]
    direction = -1.0 if error > 0.0 else 1.0
    width = max(0.01, 1.0 / beta)
    bracket = None
    for _ in range(32):
        other_mu = float(mu_init + direction * width)
        other, other_error = point(other_mu)
        if abs(other_error) < thresh:
            return other_mu, other[2], other[0], other[1]
        if (error > 0.0) != (other_error > 0.0):
            bracket = tuple(sorted((float(mu_init), other_mu)))
            break
        width *= 2.0
    if bracket is None:
        raise RuntimeError("Could not bracket chemical potential near the previous value.")

    class PopulationReached(Exception):
        def __init__(self, value: float):
            self.value = value

    def residual(value: float) -> float:
        difference = point(value)[1]
        if abs(difference) < thresh:
            raise PopulationReached(float(value))
        return difference

    try:
        root = float(brentq(
            residual, *bracket, xtol=1e-12,
            rtol=4 * np.finfo(float).eps, maxiter=maxiter
        ))
    except PopulationReached as reached:
        root = reached.value
    result, final_error = point(root)
    if abs(final_error) >= thresh:
        raise RuntimeError(
            "Brent chemical-potential search did not reach the particle-number "
            f"threshold {thresh:.3e}; delta N={final_error:.3e}."
        )
    _log_mu_tune("Restricted Brent", 0, root, -final_error, reporter)
    return root, result[2], result[0], result[1]
