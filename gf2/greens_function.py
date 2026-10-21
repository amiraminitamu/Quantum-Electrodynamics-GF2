# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Non-interacting Green's functions in imaginary time and frequency."

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import linalg as LA


@dataclass(frozen=True)
class FockOrthoState:
    """Cached orthogonal-basis diagonalisation of the Fock matrix."""

    orth_evals: np.ndarray
    evecs: np.ndarray


def prepare_fock_orth(F: np.ndarray, S_inv_half: np.ndarray) -> FockOrthoState:
    """Diagonalise ``S^{-1/2} F S^{-1/2}`` once and cache eigenvectors."""
    Fock_orth = np.einsum("ab,bc,cd->ad", S_inv_half, F, S_inv_half, optimize=True)
    orth_evals, evecs = LA.eigh(Fock_orth)
    return FockOrthoState(orth_evals=orth_evals, evecs=evecs)


def _shifted_eval_blocks(orth_evals: np.ndarray, mu: float) -> tuple[int, np.ndarray, np.ndarray]:
    """Split ``ε - μ`` into negative and positive blocks (ascending order)."""
    shifted = orth_evals - mu
    n_neg = int(np.sum(shifted < 0.0))
    return n_neg, shifted[:n_neg], shifted[n_neg:]


def _diagonal_weights_all_tau(
    itime_grid: np.ndarray,
    beta: float,
    neg: np.ndarray,
    pos: np.ndarray,
    n_neg: int,
    nbf: int,
) -> np.ndarray:
    """Diagonal ``G_0`` eigenvalues for every τ; shape ``(n_itime, nbf)``."""
    n_itime = itime_grid.shape[0]
    D = np.empty((n_itime, nbf))
    if n_neg > 0:
        eb = np.exp(1.0 * beta * neg)
        D[:, :n_neg] = -np.exp((beta - itime_grid[:, None]) * neg) / (1.0 + eb)
    if n_neg < nbf:
        ep = np.exp(-1.0 * beta * pos)
        D[:, n_neg:] = -np.exp(-itime_grid[:, None] * pos) / (1.0 + ep)
    return D


def _g0_from_state(
    state: FockOrthoState,
    mu: float,
    beta: float,
    itime_grid: np.ndarray | None,
    S_inv_half: np.ndarray,
) -> np.ndarray:
    """Build ``G_0`` in the AO basis from a cached orthogonal state."""
    nbf = state.evecs.shape[0]
    n_neg, neg, pos = _shifted_eval_blocks(state.orth_evals, mu)

    if itime_grid is None:
        D = np.empty(nbf)
        if n_neg > 0:
            D[:n_neg] = -1.0 / (1.0 + np.exp(1.0 * beta * neg))
        if n_neg < nbf:
            D[n_neg:] = -np.exp(-1.0 * beta * pos) / (1.0 + np.exp(-1.0 * beta * pos))
        G_orth = np.einsum("ij,j,kj->ik", state.evecs, D, state.evecs, optimize=True)
        return np.einsum("ab,bc,cd->ad", S_inv_half, G_orth, S_inv_half, optimize=True)

    D_all = _diagonal_weights_all_tau(itime_grid, beta, neg, pos, n_neg, nbf)
    G_orth = np.einsum("ij,tj,kj->tik", state.evecs, D_all, state.evecs, optimize=True)
    return np.einsum("ab,tbc,cd->tad", S_inv_half, G_orth, S_inv_half, optimize=True)


def build_g0_itime_all(
    state: FockOrthoState,
    S_inv_half: np.ndarray,
    mu: float,
    itime_grid: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Non-interacting ``G_0(τ)`` on every imaginary-time grid point."""
    return _g0_from_state(state, mu, beta, itime_grid, S_inv_half)


def build_g0_itime_beta(
    state: FockOrthoState,
    S_inv_half: np.ndarray,
    mu: float,
    beta: float,
) -> np.ndarray:
    """Non-interacting ``G_0(τ = β)`` (used during chemical-potential tuning)."""
    return _g0_from_state(state, mu, beta, None, S_inv_half)


def build_g0_itime_beta_derivative(
    state: FockOrthoState,
    S_inv_half: np.ndarray,
    mu: float,
    beta: float,
) -> np.ndarray:
    """Derivative of ``G_0(τ = β)`` with respect to μ."""
    nbf = state.evecs.shape[0]
    n_neg, neg, pos = _shifted_eval_blocks(state.orth_evals, mu)
    D = np.empty(nbf)
    if n_neg > 0:
        eb_n = np.exp(beta * neg)
        D[:n_neg] = -1.0 * beta / (1.0 + eb_n) + beta / (np.power((1.0 + eb_n), 2))
    if n_neg < nbf:
        eb_p = np.exp(-1.0 * beta * pos)
        D[n_neg:] = -1.0 * beta * eb_p / (1.0 + eb_p) + beta * np.exp(-2.0 * beta * pos) / (
            np.power((1.0 + eb_p), 2)
        )
    G_orth = np.einsum("ij,j,kj->ik", state.evecs, D, state.evecs, optimize=True)
    return np.einsum("ab,bc,cd->ad", S_inv_half, G_orth, S_inv_half, optimize=True)


def build_g_g0_ifreq(
    nbf: int,
    mu: float,
    ifreq_grid: np.ndarray,
    S: np.ndarray,
    F: np.ndarray,
    Sigma_if: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interacting and non-interacting Green's functions on the sparse Matsubara grid."""
    n_freq = ifreq_grid.shape[0]
    identity = np.broadcast_to(np.eye(nbf), (n_freq, nbf, nbf)).copy()
    z = (mu + 1.0j * ifreq_grid)[:, None, None]
    denom0 = z * S[None, :, :] - F[None, :, :]
    G0_iw = np.linalg.solve(denom0, identity)
    G_iw = np.linalg.solve(denom0 - Sigma_if, identity)
    return G_iw, G0_iw
