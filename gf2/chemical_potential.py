# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Chemical-potential tuning so that ``Tr[ρ S] = N_e`` is satisfied."

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.polynomial.polynomial as poly

from . import greens_function as gf
from . import spline_ft
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    "Helper: given μ, build G(iω), G_0(iω), ρ(τ=β) and the resulting N_e."
    G_iw, G0_iw = gf.build_g_g0_ifreq(nbf, mu, ifreq_grid, S, F, Sigma_if)
    delta_re, _ = spline_ft.giw_to_gtau_beta(B_beta, G_iw, G0_iw, beta)
    G0_tb = gf.build_g0_itime_beta(orth_state, S_inv_half, mu, beta)
    G_tb = G0_tb + delta_re

    # antisymmetric component even when F, S, and Sigma obey their exact
    # symmetries (most visibly for polar, slowly converging systems).  Remove

    # is the symmetric projection; the conjugation keeps the convention valid

    rho = -2.0 * G_tb
    rho = 0.5 * (rho + rho.conj().T)
    rho = np.real_if_close(rho)
    return G_iw, G0_iw, rho, nelec_from_density(rho, S)


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
    msg = f"{method}: {steps} steps, mu = {mu:.10f}, delta Ne = {delta_ne:.3e}"
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
                    f"poor initial mu (delta Ne = {delta_ne:.3e}); reseeding with linear fit"
                )
            else:
                print(
                    "        Poor initial mu guess - reseeding with linear fit "
                    f"(delta_ne={delta_ne:.3e})",
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
