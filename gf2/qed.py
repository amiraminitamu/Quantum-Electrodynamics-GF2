# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Multimode coherent-state QED building blocks (atomic units)."

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PhotonMode:
    """One bare photon mode; frequency/coupling are in Hartree atomic units."""

    frequency: float
    coupling: float
    polarization: tuple[float, float, float]

    def __post_init__(self) -> None:
        p = np.asarray(self.polarization, dtype=float)
        if not np.isfinite(self.frequency) or self.frequency <= 0:
            raise ValueError("Photon frequency must be finite and positive.")
        if not np.isfinite(self.coupling):
            raise ValueError("Photon coupling must be finite.")
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            raise ValueError("Polarization must contain three finite components.")
        if not np.isclose(np.linalg.norm(p), 1.0, atol=1.0e-12):
            raise ValueError("Polarization must be normalized to one.")
        object.__setattr__(self, "frequency", float(self.frequency))
        object.__setattr__(self, "coupling", float(self.coupling))
        object.__setattr__(self, "polarization", tuple(float(value) for value in p))

    @classmethod
    def from_legacy_lambda(cls, frequency: float, lambda_vec: Iterable[float]) -> "PhotonMode":
        """Explicit adapter for legacy ``lambda_vec = coupling * polarization``."""
        lam = np.asarray(tuple(lambda_vec), dtype=float)
        if lam.shape != (3,) or not np.all(np.isfinite(lam)):
            raise ValueError("legacy lambda_vec must contain three finite components.")
        coupling = float(np.linalg.norm(lam))

        pol = (0.0, 0.0, 1.0) if coupling == 0.0 else tuple(lam / coupling)
        return cls(frequency=float(frequency), coupling=coupling, polarization=pol)


@dataclass(frozen=True)
class CavityConfiguration:
    """Independent bare modes in the dipole-gauge Pauli--Fierz Hamiltonian."""

    modes: tuple[PhotonMode, ...]

    def __init__(self, modes: Iterable[PhotonMode]):
        object.__setattr__(self, "modes", tuple(modes))
        if not self.modes:
            raise ValueError("CavityConfiguration requires at least one mode.")
        if not all(isinstance(mode, PhotonMode) for mode in self.modes):
            raise TypeError("Every cavity mode must be a PhotonMode instance.")


@dataclass(frozen=True)
class QEDIntegralData:
    omega: np.ndarray
    h_c: np.ndarray
    g: np.ndarray
    h_dse: np.ndarray


def build_qed_integrals(
    dipole_ao: np.ndarray, overlap: np.ndarray, cavity: CavityConfiguration
) -> QEDIntegralData:
    """Project AO dipoles and form the sum of independent-mode DSE terms."""
    mu = np.asarray(dipole_ao, dtype=float)
    s = np.asarray(overlap, dtype=float)
    if mu.ndim != 3 or mu.shape[0] != 3 or mu.shape[1:] != s.shape:
        raise ValueError("dipole_ao must have shape (3, nao, nao) compatible with overlap.")
    lam = np.asarray([m.coupling * np.asarray(m.polarization) for m in cavity.modes])
    omega = np.asarray([m.frequency for m in cavity.modes])
    hc = np.einsum("xa,apq->xpq", lam, mu, optimize=True)
    hc = 0.5 * (hc + hc.swapaxes(-1, -2))
    g = -np.sqrt(omega[:, None, None] / 2.0) * hc
    sinv = np.linalg.inv(s)
    hdse = 0.5 * np.einsum("xpi,ij,xjq->pq", hc, sinv, hc, optimize=True)
    return QEDIntegralData(omega=omega, h_c=hc, g=g, h_dse=0.5 * (hdse + hdse.T))


def molecular_dipole_ao(
    electronic_dipole_ao: np.ndarray,
    overlap: np.ndarray,
    nuclear_dipole: np.ndarray,
    electron_count: int,
) -> np.ndarray:
    "Represent the total clamped-nuclei dipole in a fixed-``N`` AO space."
    mu = np.asarray(electronic_dipole_ao, dtype=float)
    s = np.asarray(overlap, dtype=float)
    d_nuc = np.asarray(nuclear_dipole, dtype=float)
    if mu.shape != (3,) + s.shape or d_nuc.shape != (3,):
        raise ValueError("dipoles must have shapes (3,nao,nao) and (3,).")
    if electron_count <= 0:
        raise ValueError("electron_count must be positive.")
    return mu + (d_nuc / float(electron_count))[:, None, None] * s[None, :, :]


def add_multimode_dse(eri: np.ndarray, h_c: np.ndarray) -> np.ndarray:
    """Return ``(pq|rs) + sum_x h_c[x,pq] h_c[x,rs]`` in chemist notation."""
    if h_c.ndim != 3 or eri.shape != (h_c.shape[1],) * 4:
        raise ValueError("Incompatible ERI and h_c shapes.")
    return eri + np.einsum("xpq,xrs->pqrs", h_c, h_c, optimize=True)


def coherent_displacement(g: np.ndarray, density: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """``z_x = -Tr[g_x P]/omega_x`` for a spin-summed restricted density."""
    return -np.einsum("xij,ji->x", g, density, optimize=True).real / omega


def coherent_fock_correction(g: np.ndarray, z: np.ndarray) -> np.ndarray:
    return 2.0 * np.einsum("x,xpq->pq", z, g, optimize=True)


def sigma_eb_tau(G: np.ndarray, D: np.ndarray, g: np.ndarray) -> np.ndarray:
    "Full-mode Fan self-energy for ``D=-<T X X>``."
    if D.shape[:1] != G.shape[:1] or D.shape[1:] != (g.shape[0], g.shape[0]):
        raise ValueError("D must have shape (ntime, nmode, nmode).")
    return -np.einsum("xik,tkl,ylj,txy->tij", g, G, g, D, optimize=True).real


def sigma_eb_tau_reference(G: np.ndarray, D: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Small explicit-loop reference implementation used by tests."""
    out = np.zeros_like(G, dtype=float)
    for t in range(G.shape[0]):
        for x in range(g.shape[0]):
            for y in range(g.shape[0]):
                out[t] -= D[t, x, y] * (g[x] @ G[t] @ g[y])
    return out


def polarization_tau(G: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Full mode-space ``Pi[t,x,y] = -2 Tr[gx G(t) gy G(beta-t)]``."""
    return -2.0 * np.einsum("xij,tjk,ykl,tli->txy", g, G, g, G[::-1], optimize=True).real


def bosonic_frequencies(beta: float, nfreq: int) -> np.ndarray:
    if beta <= 0 or nfreq < 1:
        raise ValueError("beta and nfreq must be positive.")
    return 2.0 * np.pi * np.fft.fftfreq(nfreq, d=beta / nfreq)


def bare_boson_iv(omega: np.ndarray, nu: np.ndarray) -> np.ndarray:
    d = -2.0 * omega[None, :] / (nu[:, None] ** 2 + omega[None, :] ** 2)
    return np.einsum("tx,xy->txy", d, np.eye(len(omega)), optimize=True)


def periodic_time_to_iv(
    values: np.ndarray,
    beta: float,
    time_grid: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    "Periodic forward transform ``D(iv)=integral exp(+iv*tau)D(tau)``."
    ntime = values.shape[0]
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    if time_grid is None and weights is None:
        return beta * np.fft.ifft(values, axis=0)
    if time_grid is None or weights is None:
        raise ValueError("time_grid and weights must be supplied together.")
    tau = np.asarray(time_grid, dtype=float)
    w = np.asarray(weights, dtype=float)
    if tau.shape != (ntime,) or w.shape != (ntime,):
        raise ValueError("time_grid and weights must match the leading time axis.")
    nu = bosonic_frequencies(beta, ntime)
    phase = np.exp(1.0j * np.outer(nu, tau))
    return np.einsum("nt,t...,t->n...", phase, values, w, optimize=True)


def periodic_iv_to_time(
    values: np.ndarray,
    beta: float,
    time_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Periodic inverse transform ``D(tau)=beta^-1 sum exp(-iv*tau)D(iv)``."""
    nfreq = values.shape[0]
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    if time_grid is None:
        result = np.fft.fft(values, axis=0) / beta
    else:
        tau = np.asarray(time_grid, dtype=float)
        if tau.ndim != 1:
            raise ValueError("time_grid must be one-dimensional.")
        nu = bosonic_frequencies(beta, nfreq)
        phase = np.exp(-1.0j * np.outer(tau, nu))
        result = np.einsum("tn,n...->t...", phase, values, optimize=True) / beta
    return np.real_if_close(result, tol=1000)


def bosonic_dyson(d0_iv: np.ndarray, pi_iv: np.ndarray) -> np.ndarray:
    """Matrix Dyson equation independently for each bosonic frequency."""
    if d0_iv.shape != pi_iv.shape or d0_iv.ndim != 3 or d0_iv.shape[1] != d0_iv.shape[2]:
        raise ValueError("d0_iv and pi_iv must have shape (nnu,nmode,nmode).")
    return np.linalg.inv(np.linalg.inv(d0_iv) - pi_iv)
