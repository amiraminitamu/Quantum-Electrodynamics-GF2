# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Energy estimators for the GF2 procedure."

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import psi4
except ImportError:  # pragma: no cover - psi4 is required at runtime
    psi4 = None  # type: ignore[assignment]


def _integral_tr_g_sigma(G_tau: np.ndarray, Sigma_tau: np.ndarray, d_itime: np.ndarray) -> float:
    """Compute ``integral Tr[G(beta-tau) Sigma(tau)] d tau`` by quadrature."""
    n_t = G_tau.shape[0]
    beta_idx = np.arange(n_t - 1, -1, -1)
    traces = np.einsum("tij,tji->t", G_tau[beta_idx], Sigma_tau, optimize=True)
    return float(np.dot(traces, d_itime))


def mp2_correlation(G_tau: np.ndarray, Sigma_tau: np.ndarray, d_itime: np.ndarray) -> float:
    """One-shot MP2 correlation energy from ``G_0`` and ``Sigma[G_0]``.

    ``E_MP2 = -(1/2) integral Tr[G_0(beta-tau) Sigma_0(tau)] d tau``.
    """
    return -0.5 * _integral_tr_g_sigma(G_tau, Sigma_tau, d_itime)


def gf2_correlation(G_tau: np.ndarray, Sigma_tau: np.ndarray, d_itime: np.ndarray) -> float:
    "Restricted self-consistent GF2 dynamic energy for a one-spin ``G``."
    return -1.0 * _integral_tr_g_sigma(G_tau, Sigma_tau, d_itime)


def gf2_correlation_frequency_positive(
    G_iw: np.ndarray,
    Sigma_iw: np.ndarray,
    beta: float,
) -> float:
    "Positive-frequency restricted-GF2 dynamic energy."
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    if G_iw.shape != Sigma_iw.shape or G_iw.ndim != 3:
        raise ValueError("G_iw and Sigma_iw must have shape (nfreq,nao,nao).")
    traces = np.einsum("nij,nji->n", Sigma_iw, G_iw, optimize=True)
    return float(2.0 * np.sum(traces.real) / beta)


def coherent_state_mean_field_energy(
    F_cs: np.ndarray,
    H_effective: np.ndarray,
    density: np.ndarray,
    E_nuc: float,
    g: np.ndarray,
    z: np.ndarray,
    omega: np.ndarray,
) -> float:
    "Correct CS mean-field energy for a spin-summed restricted density."
    from .fock import core_energy

    coherent_fock = 2.0 * np.einsum("x,xpq->pq", z, g, optimize=True)
    static_fock = F_cs - coherent_fock
    electronic = core_energy(static_fock, H_effective, density, E_nuc)
    expectation = np.einsum("xij,ji->x", g, density, optimize=True).real
    coherent = np.sum(2.0 * z * expectation + omega * z**2)
    return float(electronic + coherent)


def psi4_mp2_correlation(hf_energy: float, ref_wfn: Any) -> float:
    "Canonical Psi4 MP2 correlation energy ``E_MP2 - E_HF``."
    if psi4 is None:  # pragma: no cover
        raise RuntimeError("psi4 is required to compute the MP2 correlation energy.")
    e_mp2 = psi4.energy("mp2", ref_wfn=ref_wfn)
    return float(e_mp2) - float(hf_energy)


def print_mp2_comparison(
    psi4_corr: float,
    gf2_estimate: float,
    reporter: Any | None = None,
) -> None:
    """Print Psi4 MP2 and the GF2 one-shot MP2 estimate side by side."""
    diff = gf2_estimate - psi4_corr
    rel = diff / psi4_corr if psi4_corr != 0.0 else float("nan")
    if reporter is not None:
        reporter.section("MP2 comparison")
        reporter.kv("Psi4 (canonical)", f"{psi4_corr: .12f}")
        reporter.kv("GF2 estimate (G0)", f"{gf2_estimate: .12f}")
        reporter.kv("Difference", f"{diff: .12f}")
        reporter.kv("Relative", f"{rel:.6e}")
        return
    print("  MP2 comparison", flush=True)
    print(f"    Psi4 (canonical)     {psi4_corr: .12f}", flush=True)
    print(f"    GF2 estimate (G0)  {gf2_estimate: .12f}", flush=True)
    print(f"    Difference         {diff: .12f}", flush=True)
    print(f"    Relative           {rel:.6e}", flush=True)
