# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Small linear-algebra utilities used throughout the GF2 package."""

from __future__ import annotations

import numpy as np


def sym(A: np.ndarray) -> np.ndarray:
    "Return the Hermitian (symmetric) part of ``A``."
    return 0.5 * (A + A.conj().T)


def s_inv_sqrt(S: np.ndarray, thresh: float = 1.0e-12) -> np.ndarray:
    "Compute ``S^{-1/2}`` from the symmetric positive-definite overlap matrix."
    S = sym(S)
    eigvals, eigvecs = np.linalg.eigh(S)
    if np.any(eigvals <= thresh):
        raise RuntimeError(
            f"Overlap matrix has small/negative eigenvalues (min={eigvals.min():.3e})."
        )
    X = (eigvecs / np.sqrt(eigvals)) @ eigvecs.conj().T
    return sym(X)


def fock_eigs(F: np.ndarray, S: np.ndarray) -> np.ndarray:
    "Return generalised Fock eigenvalues solving ``F C = eps S C``."
    X = s_inv_sqrt(S)
    F_ortho = sym(X @ F @ X)
    return np.linalg.eigvalsh(F_ortho)


def linear_mix(new: np.ndarray, old: np.ndarray, alpha: float) -> np.ndarray:
    """Linear combination ``alpha * new + (1 - alpha) * old``."""
    return alpha * new + (1.0 - alpha) * old


def nelec_from_density(P: np.ndarray, S: np.ndarray):
    "Number of electrons from the AO density matrix: ``tr(P S)``."
    return np.matrix.trace(np.einsum("pq,qr", P, S, optimize=True))
