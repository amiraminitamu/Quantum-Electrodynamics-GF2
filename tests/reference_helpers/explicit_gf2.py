# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Slow four-index GF2 references independent of production contractions."""

from __future__ import annotations

import numpy as np


def explicit_gf2_self_energy(green_tau: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """Direct transcription of the restricted GF2 equation (tests only)."""
    out = np.empty_like(green_tau, dtype=float)
    for tau in range(green_tau.shape[0]):
        gt = green_tau[tau]
        gb = green_tau[-tau - 1]
        direct = 2.0 * np.einsum(
            "ikmq,kl,mn,pq,pnlj->ij",
            eri,
            gt,
            gt,
            gb,
            eri,
            optimize=False,
        )
        exchange = np.einsum(
            "ikmq,kl,mn,pq,plnj->ij",
            eri,
            gt,
            gt,
            gb,
            eri,
            optimize=False,
        )
        out[tau] = direct - exchange
    return out
