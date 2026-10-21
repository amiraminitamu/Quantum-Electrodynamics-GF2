# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Electronic Fock matrix construction and core-energy evaluation."

from __future__ import annotations

import numpy as np


def build_jk(eri: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    "Coulomb (``J``) and exchange (``K``) operators from a 4-index tensor."
    J = np.einsum("pqrs,rs->pq", eri, P, optimize=True)
    K = np.einsum("prqs,rs->pq", eri, P, optimize=True)
    return J, K


def build_jk_reduced_symmetry(
    eri: np.ndarray,
    P: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    "J/K operators for a reduced-symmetry two-electron tensor."
    J = np.einsum("pqrs,rs->pq", eri, P, optimize=True)
    K = np.einsum("psrq,rs->pq", eri, P, optimize=True)
    return J, K


def build_jk_df(Qpq: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    "Coulomb and exchange operators from the DF tensor ``Q_{Qpq}``."
    J_int = np.einsum("rs,Qrs->Q", P, Qpq, optimize=True)
    J = np.einsum("Qpq,Q->pq", Qpq, J_int, optimize=True)
    K_int = np.einsum("rs,Qqs->Qrq", P, Qpq)
    K = np.einsum("Qpr,Qrq->pq", Qpq, K_int)
    return J, K


def build_fock(H_core: np.ndarray, P: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """Closed-shell Fock matrix ``F = h + J - K/2`` from a 4-index ERI tensor."""
    J, K = build_jk(eri, P)
    return H_core + J - 0.5 * K


def build_fock_reduced_symmetry(
    H_core: np.ndarray,
    P: np.ndarray,
    eri: np.ndarray,
) -> np.ndarray:
    """Closed-shell Fock matrix for a reduced-symmetry interaction."""
    J, K = build_jk_reduced_symmetry(eri, P)
    F = H_core + J - 0.5 * K
    return 0.5 * (F + F.T)


def build_fock_df(H_core: np.ndarray, P: np.ndarray, Qpq: np.ndarray) -> np.ndarray:
    """Closed-shell Fock matrix from a density-fitted 3-index tensor."""
    J, K = build_jk_df(Qpq, P)
    return H_core + J - 0.5 * K


def core_energy(F: np.ndarray, H_core: np.ndarray, P: np.ndarray, E_nuc: float) -> float:
    "``E_core = (1/2) Tr[(F + h) P] + E_nuc`` (Hartree-Fock-like core energy)."
    return float(np.einsum("pq,pq->", F + H_core, 0.5 * P, optimize=True).real + E_nuc)
