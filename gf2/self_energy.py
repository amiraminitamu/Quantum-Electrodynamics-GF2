# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Second-order (GF2) self-energy in imaginary time."

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from opt_einsum import contract as _contract


def _exact_sigma_tau(
    tau: int,
    G_tau: np.ndarray,
    eri: np.ndarray,
    n_itime: int,
) -> np.ndarray:
    """Evaluate the exact GF2 self-energy for one imaginary-time point."""
    beta_tau = n_itime - tau - 1
    Gt = G_tau[tau]
    Gb = G_tau[beta_tau]

    X = _contract("pq,pnlj->qnlj", Gb, eri, optimize=True)
    D = _contract("mn,qnlj->mqlj", Gt, X, optimize=True)
    E = _contract("kl,mqlj->kmqj", Gt, D, optimize=True)
    direct = 2.0 * _contract("ikmq,kmqj->ij", eri, E, optimize=True)

    X = _contract("pq,plnj->qnjl", Gb, eri, optimize=True)
    D = _contract("mn,qnjl->mqjl", Gt, X, optimize=True)
    E = _contract("kl,mqjl->kmqj", Gt, D, optimize=True)
    exchange = _contract("ikmq,kmqj->ij", eri, E, optimize=True)

    return direct - exchange


def _reduced_sigma_tau(
    tau: int,
    G_tau: np.ndarray,
    eri: np.ndarray,
    n_itime: int,
) -> np.ndarray:
    "Second-Born self-energy for a reduced-symmetry interaction."
    beta_tau = n_itime - tau - 1
    Gt = G_tau[tau]
    Gb = G_tau[beta_tau]

    X = _contract("pq,pnjl->qnjl", Gb, eri, optimize=True)
    D = _contract("mn,qnjl->mqjl", Gt, X, optimize=True)
    E = _contract("kl,mqjl->kmqj", Gt, D, optimize=True)
    direct = 2.0 * _contract(
        "ikqm,kmqj->ij",
        eri,
        E,
        optimize=True,
    )

    X = _contract("pq,pljn->qljn", Gb, eri, optimize=True)
    D = _contract("mn,qljn->mqlj", Gt, X, optimize=True)
    E = _contract("kl,mqlj->kmqj", Gt, D, optimize=True)
    exchange = _contract(
        "ikqm,kmqj->ij",
        eri,
        E,
        optimize=True,
    )

    return direct - exchange


def _df_sigma_tau(
    tau: int,
    G_tau: np.ndarray,
    Qpq: np.ndarray,
    n_itime: int,
) -> np.ndarray:
    """Evaluate the density-fitted GF2 self-energy for one imaginary-time point."""
    beta_tau = n_itime - tau - 1
    Gt = G_tau[tau]
    Gb = G_tau[beta_tau]

    R_Zqn = _contract("pq,Zpn->Zqn", Gb, Qpq, optimize=True)
    E_Qnq = _contract("mn,Qmq->Qnq", Gt, Qpq, optimize=True)
    D_kjZ = _contract("kl,Zlj->kjZ", Gt, Qpq, optimize=True)
    X_QZ = _contract("Qnq,Zqn->QZ", E_Qnq, R_Zqn, optimize=True)
    F_Zik = _contract("Qik,QZ->Zik", Qpq, X_QZ, optimize=True)
    direct = 2.0 * _contract("kjZ,Zik->ij", D_kjZ, F_Zik, optimize=True)

    # Exchange ("K-like") contractions (fused to avoid nbf^4 intermediate)
    R_Zlq = _contract("pq,Zlp->Zlq", Gb, Qpq, optimize=True)
    D_Zmj = _contract("mn,Znj->Zmj", Gt, Qpq, optimize=True)
    X_Qli = _contract("kl,Qik->Qli", Gt, Qpq, optimize=True)
    E_Qjl = _contract("Qmq,Zmj,Zlq->Qjl", Qpq, D_Zmj, R_Zlq, optimize=True)
    exchange = _contract("Qli,Qjl->ij", X_Qli, E_Qjl, optimize=True)

    return direct - exchange


def _validated_worker_count(n_workers: int, n_itime: int) -> int:
    workers = int(n_workers)
    if workers < 1:
        raise ValueError("n_workers must be a positive integer.")
    return min(workers, n_itime)


def _unrestricted_exact_tau(
    tau: int, external: np.ndarray, internal: np.ndarray, eri: np.ndarray,
    include_exchange: bool = True,
) -> np.ndarray:
    """One internal-spin direct term minus optional same-spin exchange."""
    gt = external[tau]
    gi = internal[tau]
    gb = internal[-tau - 1]
    x = _contract("pq,pnlj->qnlj", gb, eri, optimize=True)
    d = _contract("mn,qnlj->mqlj", gi, x, optimize=True)
    e = _contract("kl,mqlj->kmqj", gt, d, optimize=True)
    direct = _contract("ikmq,kmqj->ij", eri, e, optimize=True)

    if not include_exchange:
        return direct

    gb = external[-tau - 1]
    x = _contract("pq,plnj->qnjl", gb, eri, optimize=True)
    d = _contract("mn,qnjl->mqjl", gt, x, optimize=True)
    e = _contract("kl,mqjl->kmqj", gt, d, optimize=True)
    exchange = _contract("ikmq,kmqj->ij", eri, e, optimize=True)
    return direct - exchange


def _unrestricted_df_tau(
    tau: int, external: np.ndarray, internal: np.ndarray, qpq: np.ndarray,
    include_exchange: bool = True,
) -> np.ndarray:
    """Factorized direct and exchange, retaining the restricted index order."""
    gt = external[tau]
    gi = internal[tau]
    gb = internal[-tau - 1]
    r = _contract("pq,Zpn->Zqn", gb, qpq, optimize=True)
    e = _contract("mn,Qmq->Qnq", gi, qpq, optimize=True)
    d = _contract("kl,Zlj->kjZ", gt, qpq, optimize=True)
    x = _contract("Qnq,Zqn->QZ", e, r, optimize=True)
    f = _contract("Qik,QZ->Zik", qpq, x, optimize=True)
    direct = _contract("kjZ,Zik->ij", d, f, optimize=True)

    if not include_exchange:
        return direct

    gb = external[-tau - 1]
    r = _contract("pq,Zlp->Zlq", gb, qpq, optimize=True)
    d = _contract("mn,Znj->Zmj", gt, qpq, optimize=True)
    x = _contract("kl,Qik->Qli", gt, qpq, optimize=True)
    e = _contract("Qmq,Zmj,Zlq->Qjl", qpq, d, r, optimize=True)
    exchange = _contract("Qli,Qjl->ij", x, e, optimize=True)
    return direct - exchange


def _unrestricted_pair(
    alpha: np.ndarray, beta: np.ndarray, interaction: np.ndarray,
    kernel: object, n_workers: int
) -> tuple[np.ndarray, np.ndarray]:
    if alpha.shape != beta.shape or alpha.ndim != 3:
        raise ValueError("Alpha and beta G(tau) must have equal three-dimensional shapes.")
    ntime = alpha.shape[0]
    workers = _validated_worker_count(n_workers, ntime)
    # Each direct line sums over both internal spins.  Subtract the same-spin
    # exchange only once: kernel(a,a) already contains that subtraction.
    def at(tau: int) -> tuple[np.ndarray, np.ndarray]:
        aa = kernel(tau, alpha, alpha, interaction)
        bb = kernel(tau, beta, beta, interaction)
        ab = kernel(tau, alpha, beta, interaction, False)
        ba = kernel(tau, beta, alpha, interaction, False)
        return aa + ab, bb + ba

    if workers == 1:
        values = map(at, range(ntime))
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        values = executor.map(at, range(ntime))
    try:
        pairs = list(values)
    finally:
        if workers != 1:
            executor.shutdown()
    return np.stack([v[0] for v in pairs]), np.stack([v[1] for v in pairs])


def gf2_self_energy_unrestricted(
    alpha: np.ndarray, beta: np.ndarray, eri: np.ndarray, *, n_workers: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Spin-orbital second Born using exact AO integrals."""
    return _unrestricted_pair(alpha, beta, eri, _unrestricted_exact_tau, n_workers)


def gf2_self_energy_unrestricted_df(
    alpha: np.ndarray, beta: np.ndarray, qpq: np.ndarray, *, n_workers: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Spin-orbital second Born using three-index factors only."""
    return _unrestricted_pair(alpha, beta, qpq, _unrestricted_df_tau, n_workers)


def gf2_self_energy(
    nbf: int,
    G_tau: np.ndarray,
    eri: np.ndarray,
    n_itime: int,
    *,
    n_workers: int = 1,
) -> np.ndarray:
    "Build ``Σ(τ)`` for every τ on the Chebyshev imaginary-time grid."
    Sigma = np.zeros((n_itime, nbf, nbf))
    workers = _validated_worker_count(n_workers, n_itime)

    if workers == 1:
        for tau in range(n_itime):
            Sigma[tau] = _exact_sigma_tau(tau, G_tau, eri, n_itime)
        return Sigma

    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = executor.map(
            lambda tau: _exact_sigma_tau(tau, G_tau, eri, n_itime),
            range(n_itime),
        )
        for tau, value in enumerate(values):
            Sigma[tau] = value
    return Sigma


def gf2_self_energy_reduced(
    nbf: int,
    G_tau: np.ndarray,
    eri: np.ndarray,
    n_itime: int,
    *,
    n_workers: int = 1,
) -> np.ndarray:
    "GF2 self-energy for an exact reduced-symmetry interaction."
    Sigma = np.zeros((n_itime, nbf, nbf))
    workers = _validated_worker_count(n_workers, n_itime)

    if workers == 1:
        for tau in range(n_itime):
            Sigma[tau] = _reduced_sigma_tau(
                tau,
                G_tau,
                eri,
                n_itime,
            )
        return Sigma

    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = executor.map(
            lambda tau: _reduced_sigma_tau(
                tau,
                G_tau,
                eri,
                n_itime,
            ),
            range(n_itime),
        )
        for tau, value in enumerate(values):
            Sigma[tau] = value

    return Sigma


def gf2_self_energy_df(
    nbf: int,
    G_tau: np.ndarray,
    Qpq: np.ndarray,
    n_itime: int,
    *,
    n_workers: int = 1,
) -> np.ndarray:
    "Density-fitted GF2 self-energy."
    Sigma = np.zeros((n_itime, nbf, nbf))
    workers = _validated_worker_count(n_workers, n_itime)

    if workers == 1:
        for tau in range(n_itime):
            Sigma[tau] = _df_sigma_tau(tau, G_tau, Qpq, n_itime)
        return Sigma

    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = executor.map(
            lambda tau: _df_sigma_tau(tau, G_tau, Qpq, n_itime),
            range(n_itime),
        )
        for tau, value in enumerate(values):
            Sigma[tau] = value
    return Sigma
