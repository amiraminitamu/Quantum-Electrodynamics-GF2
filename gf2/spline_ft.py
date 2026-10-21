# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Cubic-spline-based ``G(iω) → G(τ)`` Fourier transform."

from __future__ import annotations

import numpy as np


def _spline_abcd(full_ifreq_grid: np.ndarray, ifreq_grid: np.ndarray):
    """Coefficients for the cubic spline ``f(x) = A y_j + B y_{j+1} + C Z_j + D Z_{j+1}``."""
    full_n = full_ifreq_grid.shape[0]
    n = ifreq_grid.shape[0]

    A = np.zeros((full_n, n))
    B = np.zeros((full_n, n))
    C = np.zeros((full_n, n))
    D = np.zeros((full_n, n))

    for k in range(full_n):
        for j in range(n - 1):
            if ifreq_grid[j] <= full_ifreq_grid[k] < ifreq_grid[j + 1]:
                rj = ifreq_grid[j + 1] - ifreq_grid[j]
                A[k, j] = (ifreq_grid[j + 1] - full_ifreq_grid[k]) / rj
                B[k, j] = 1.0 - A[k, j]
                C[k, j] = (A[k, j] ** 3 - A[k, j]) * rj**2 / 6.0
                D[k, j] = (B[k, j] ** 3 - B[k, j]) * rj**2 / 6.0

    for k in range(full_n):
        if ifreq_grid[n - 1] <= full_ifreq_grid[k]:
            B[k, n - 2] = 1.0
            D[k, n - 2] = 0.0
    return A, B, C, D


def _spline_u_matrix(ifreq_grid: np.ndarray) -> np.ndarray:
    """Spline second-derivative operator ``U = M^{-1} S``."""
    n = ifreq_grid.shape[0]
    M = np.zeros((n, n))
    S = np.zeros((n, n))

    for j in range(1, n - 1):
        r_lo = ifreq_grid[j] - ifreq_grid[j - 1]
        r_hi = ifreq_grid[j + 1] - ifreq_grid[j]

        M[j, j - 1] = r_lo
        M[j, j] = 2.0 * (r_hi + r_lo)
        M[j, j + 1] = r_hi

        S[j, j - 1] = 6.0 / r_lo
        S[j, j] = -6.0 / r_lo - 6.0 / r_hi
        S[j, j + 1] = 6.0 / r_hi

    S[0, 0] = 1.0
    S[-1, -1] = 1.0
    M[0, 1:] = 0.0
    M[0, 0] = 1.0e10
    M[-1, : n - 1] = 0.0
    M[-1, -1] = 1.0e10

    return np.linalg.inv(M) @ S


def spline_v_matrix(full_ifreq_grid: np.ndarray, ifreq_grid: np.ndarray) -> np.ndarray:
    """Matrix ``V`` such that ``g_dense = V @ g_sparse`` for a cubic spline."""
    n_full = full_ifreq_grid.shape[0]
    n = ifreq_grid.shape[0]

    A, B, C, D = _spline_abcd(full_ifreq_grid, ifreq_grid)
    U = _spline_u_matrix(ifreq_grid)

    V = np.zeros((n_full, n))
    V[:, 1:n] = B[:, 0 : n - 1]
    V += A
    V += C @ U
    Ushift = np.zeros((n, n))
    Ushift[0 : n - 1, :] = U[1:n, :]
    V += D @ Ushift
    return V


def spline_vft(V: np.ndarray, e_iwt: np.ndarray) -> np.ndarray:
    """Combine the spline matrix with ``e^{-iω τ}`` to get ``B`` for all τ.

    Result has shape ``(n_itime, n_ifreq)``.
    """
    return np.einsum("kj,ki->ji", e_iwt, V, optimize=True)


def spline_vft_beta(V: np.ndarray, e_iwbeta: np.ndarray) -> np.ndarray:
    """Same as :func:`spline_vft` but for the single time slice ``τ = β``."""
    return np.einsum("k,ki->i", e_iwbeta, V, optimize=True)


def giw_to_gtau(
    B: np.ndarray,
    G_iw: np.ndarray,
    Gnot_iw: np.ndarray,
    Gnot_t: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Reconstruct ``G(τ)`` from ``G(iω)`` and the analytic ``G_0(τ)`` tail.

    ``G(τ) = G_0(τ) + (2/β) Re Σ_n B_{τ,n} [G(iω_n) - G_0(iω_n)]``.
    """
    delta_iw = G_iw - Gnot_iw
    G_t = np.einsum("ji,iab->jab", B, delta_iw, optimize=True)
    return Gnot_t + np.real(G_t) * (2.0 / beta)


def giw_to_gtau_beta(
    B_beta: np.ndarray,
    G_iw: np.ndarray,
    Gnot_iw: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``G(τ=β) - G_0(τ=β)`` from ``G(iω)`` and ``G_0(iω)``.

    Returns the (real, imaginary) AO-basis matrices.
    """
    delta = (2.0 / beta) * np.einsum("i,iab->ab", B_beta, G_iw - Gnot_iw, optimize=True)
    return np.real(delta), np.imag(delta)
