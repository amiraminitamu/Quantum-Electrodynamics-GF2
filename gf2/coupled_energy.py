# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Coupled fermion--boson total-energy functional used by QED-GF2."

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .energy import gf2_correlation


@dataclass(frozen=True)
class CoupledEnergyBreakdown:
    """Components of the coupled restricted fermion--boson total energy."""

    static_energy: float
    ee_correlation: float
    eb_fermionic_trace: float
    eb_interaction_sigma_g: float
    eb_interaction_pi_d: float
    photon_coordinate_term: float
    photon_fluctuation_energy: float
    fermionic_only_total: float
    coupled_correlation: float
    coupled_total: float
    coupled_shift_from_fermionic: float
    sigma_g_pi_d_residual: float
    d_equal_time_diagonal: np.ndarray


def _as_real_scalar(
    value: complex | float,
    *,
    name: str,
    atol: float = 1.0e-9,
) -> float:
    z = complex(value)
    if abs(z.imag) > atol:
        raise ValueError(f"{name} has a non-negligible imaginary part: {z.imag:.3e}")
    return float(z.real)


def tail_corrected_boson_equal_time(
    D_iv: np.ndarray,
    D0_iv: np.ndarray,
    omega: np.ndarray,
    beta: float,
) -> np.ndarray:
    "Return ``D(0^-)`` with the analytic bare high-frequency tail restored."
    D_iv = np.asarray(D_iv)
    D0_iv = np.asarray(D0_iv)
    omega = np.asarray(omega, dtype=float)

    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    if D_iv.shape != D0_iv.shape or D_iv.ndim != 3:
        raise ValueError("D_iv and D0_iv must have shape (nnu,nmode,nmode).")
    if D_iv.shape[1:] != (omega.size, omega.size):
        raise ValueError("omega is incompatible with the bosonic propagator.")
    if np.any(omega <= 0.0) or not np.all(np.isfinite(omega)):
        raise ValueError("omega must contain finite positive frequencies.")

    bare_diag = -1.0 / np.tanh(0.5 * beta * omega)
    bare_equal = np.diag(bare_diag)
    correction = np.sum(D_iv - D0_iv, axis=0) / beta
    value = bare_equal + correction

    imag_max = float(np.max(np.abs(np.imag(value))))
    if imag_max > 1.0e-8:
        raise ValueError(
            "Equal-time bosonic propagator has a non-negligible imaginary "
            f"component ({imag_max:.3e})."
        )
    return np.asarray(np.real(value), dtype=float)


def pi_d_interaction_integral(
    Pi_tau: np.ndarray,
    D_tau: np.ndarray,
    d_itime: np.ndarray,
) -> float:
    "Evaluate ``int Tr_b[Pi^T(tau) D(tau)] d tau``."
    Pi_tau = np.asarray(Pi_tau)
    D_tau = np.asarray(D_tau)
    weights = np.asarray(d_itime, dtype=float)
    if Pi_tau.shape != D_tau.shape or Pi_tau.ndim != 3:
        raise ValueError("Pi_tau and D_tau must have shape (ntime,nmode,nmode).")
    if weights.shape != (Pi_tau.shape[0],):
        raise ValueError("d_itime must match the time grid.")
    integrand = np.einsum("txy,txy->t", Pi_tau, D_tau, optimize=True)
    return _as_real_scalar(np.dot(integrand, weights), name="Pi-D interaction integral")


def coupled_energy_from_state(
    *,
    static_energy: float,
    G_tau: np.ndarray,
    Sigma_ee_tau: np.ndarray,
    Sigma_eb_tau: np.ndarray,
    Pi_tau: np.ndarray,
    D_tau: np.ndarray,
    D_iv: np.ndarray,
    D0_iv: np.ndarray,
    d_itime: np.ndarray,
    omega: np.ndarray,
    beta: float,
) -> CoupledEnergyBreakdown:
    """Evaluate the coupled total energy on one synchronized solver state."""
    C_ee = gf2_correlation(G_tau, Sigma_ee_tau, d_itime)
    C_eb = gf2_correlation(G_tau, Sigma_eb_tau, d_itime)

    # Full closed-shell spin sum for a linear electron--boson interaction.
    E_eb_sigma_g = 2.0 * C_eb

    E_eb_pi_d = -pi_d_interaction_integral(Pi_tau, D_tau, d_itime)

    D_equal = tail_corrected_boson_equal_time(D_iv, D0_iv, omega, beta)
    d_diag = np.diag(D_equal)

    photon_coordinate = float(-0.5 * np.sum(np.asarray(omega, dtype=float) * (d_diag + 1.0)))
    photon_energy = photon_coordinate + 0.5 * E_eb_sigma_g

    fermionic_only_total = float(static_energy + C_ee + C_eb)
    coupled_total = float(static_energy + C_ee + E_eb_sigma_g + photon_energy)
    coupled_correlation = float(coupled_total - static_energy)

    return CoupledEnergyBreakdown(
        static_energy=float(static_energy),
        ee_correlation=float(C_ee),
        eb_fermionic_trace=float(C_eb),
        eb_interaction_sigma_g=float(E_eb_sigma_g),
        eb_interaction_pi_d=float(E_eb_pi_d),
        photon_coordinate_term=float(photon_coordinate),
        photon_fluctuation_energy=float(photon_energy),
        fermionic_only_total=fermionic_only_total,
        coupled_correlation=coupled_correlation,
        coupled_total=coupled_total,
        coupled_shift_from_fermionic=float(coupled_total - fermionic_only_total),
        sigma_g_pi_d_residual=float(E_eb_sigma_g - E_eb_pi_d),
        d_equal_time_diagonal=np.asarray(d_diag, dtype=float),
    )
