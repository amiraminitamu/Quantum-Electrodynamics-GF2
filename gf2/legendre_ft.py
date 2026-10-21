# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Legendre-polynomial Fourier transforms for ``Σ(τ) ↔ Σ(iω)``."

from __future__ import annotations

import numpy as np
from scipy import special as bess


def legendre_basis(n_poly: int, cos_grid: np.ndarray) -> np.ndarray:
    """Evaluate Legendre polynomials ``P_l(cos θ)`` on the Chebyshev grid."""
    n_itime = cos_grid.shape[0]
    identity = np.eye(n_poly)
    P = np.zeros((n_poly, n_itime))
    for order in range(n_poly):
        P[order, :] = np.polynomial.Legendre(identity[order, :])(cos_grid)
    return P


def legendre_fit_weights(
    n_poly: int,
    cos_grid: np.ndarray,
    d_itime: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Precompute ``(2l+1)/β × P_l(τ) × dτ``; shape ``(n_poly, n_itime)``."""
    P = legendre_basis(n_poly, cos_grid)
    norms = (2.0 * np.arange(n_poly) + 1.0) / beta
    return norms[:, None] * P * d_itime[None, :]


def legendre_coefficients(
    Sigma_t: np.ndarray,
    fit_weights: np.ndarray,
) -> np.ndarray:
    """Fit ``Σ(τ)`` as a Legendre expansion; returns ``(n_poly, nbf, nbf)``."""
    return np.einsum("lt,tij->lij", fit_weights, Sigma_t, optimize=True)


def analytic_transformation_integrals(
    w_prime: np.ndarray,
    int_list: np.ndarray,
    n_ifreq: int,
    beta: float,
    n_poly: int,
) -> np.ndarray:
    """Analytic FT of the Legendre polynomials; shape ``(n_poly, n_ifreq)`` complex."""
    signs = (-1.0) ** int_list.astype(int)
    phases = (1.0j) ** (np.arange(n_poly)[:, None] + 1)
    jl = np.stack(
        [bess.spherical_jn(order, w_prime) for order in range(n_poly)],
        axis=0,
    )
    return beta * signs[None, :] * phases * jl


def time_to_frequency(
    legendre_trans: np.ndarray,
    c_coeffs: np.ndarray,
) -> np.ndarray:
    """Reconstruct ``Σ(iω_n)`` from Legendre coefficients; shape ``(n_ifreq, nbf, nbf)``."""
    return np.einsum("kij,kn->nij", c_coeffs, legendre_trans, optimize=True)


def e_iwt_beta(full_ifreq_grid: np.ndarray, beta: float) -> np.ndarray:
    """``exp(-i ω_n β)`` on the dense Matsubara frequency grid."""
    wt = full_ifreq_grid * beta
    return np.cos(wt) - 1.0j * np.sin(wt)


def e_iwt_all(full_ifreq_grid: np.ndarray, itime_grid: np.ndarray) -> np.ndarray:
    """``exp(-i ω_n τ_k)`` for all ``(n, k)``; shape ``(n_ifreq_full, n_itime)``."""
    wt = np.outer(full_ifreq_grid, itime_grid)
    return np.cos(wt) - 1.0j * np.sin(wt)
