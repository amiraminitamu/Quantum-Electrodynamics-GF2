# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Imaginary-frequency and imaginary-time grids for Matsubara GF2."

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridBundle:
    """Container for the (immutable) grid arrays used by the GF2 driver."""

    ifreq_grid: np.ndarray
    full_ifreq_grid: np.ndarray
    w_prime: np.ndarray
    int_list: np.ndarray
    theta_grid: np.ndarray
    shift_theta: np.ndarray
    cos_grid: np.ndarray
    itime_grid: np.ndarray
    d_itime: np.ndarray


def matsubara_frequency_grids(
    n_ifreq: int,
    mid_pt: int,
    n_ifreq_full: int,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    "Build the sparse and dense Matsubara frequency grids."
    x = np.arange(mid_pt)
    log_lo = np.log(mid_pt)
    log_hi = np.log(n_ifreq_full)
    y = np.rint(np.exp(np.linspace(log_lo, log_hi, n_ifreq - mid_pt)))
    int_list = np.concatenate([x, y]).astype(int)

    ifreq_grid = np.pi * (2.0 * int_list + 1.0) / beta
    w_prime = np.pi * (2.0 * int_list + 1.0) / 2.0
    full_ifreq_grid = np.pi * (2.0 * np.arange(n_ifreq_full) + 1.0) / beta

    return ifreq_grid, full_ifreq_grid, w_prime, int_list


def theta_grid(n_itime: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform angular grid on ``[0, π]`` used to define the Chebyshev nodes."""
    delta = np.pi / n_itime
    t = np.arange(delta / 2.0, np.pi, delta)
    theta = np.flip(t, 0)
    shift = theta - delta / 2.0
    return theta, shift


def cheb_grid(cos_grid: np.ndarray, beta: float) -> np.ndarray:
    """Chebyshev imaginary-time grid on ``[0, β]`` from the cosine grid."""
    return (beta / 2.0) * (cos_grid + 1.0)


def cheb_dt(shift_theta: np.ndarray, beta: float, n_itime: int) -> np.ndarray:
    """Riemann-sum widths on the Chebyshev imaginary-time grid."""
    dt = np.zeros(n_itime)
    for i in range(n_itime):
        if i == 0:
            dt[i] = abs(np.cos(shift_theta[i]) + 1.0)
        else:
            dt[i] = abs(np.cos(shift_theta[i]) - np.cos(shift_theta[i - 1]))
    return dt * (beta / 2.0)


def build_grids(opts) -> GridBundle:
    """Convenience entry point that builds every grid the driver needs."""
    ifreq_grid, full_ifreq_grid, w_prime, int_list = matsubara_frequency_grids(
        n_ifreq=opts.n_ifreq,
        mid_pt=opts.mid_pt,
        n_ifreq_full=opts.n_ifreq_full,
        beta=opts.beta,
    )
    theta, shift = theta_grid(opts.n_itime)
    cos_grid = np.cos(theta)
    itime = cheb_grid(cos_grid, opts.beta)
    d_it = cheb_dt(shift, opts.beta, opts.n_itime)
    return GridBundle(
        ifreq_grid=ifreq_grid,
        full_ifreq_grid=full_ifreq_grid,
        w_prime=w_prime,
        int_list=int_list,
        theta_grid=theta,
        shift_theta=shift,
        cos_grid=cos_grid,
        itime_grid=itime,
        d_itime=d_it,
    )
