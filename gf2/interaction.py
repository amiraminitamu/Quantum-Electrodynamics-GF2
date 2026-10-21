# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Electronic interaction representations shared by GF2 solver variants."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from . import fock as fock_mod
from . import self_energy
from .integrals import DFTensors


@dataclass(frozen=True)
class ElectronInteraction:
    "Exact or factorized AO electron-electron interaction."

    mode: str
    orbital_basis: str
    eri: np.ndarray | None = None
    df: DFTensors | None = None
    se_fit_basis: str | None = None
    fock_fit_basis: str | None = None
    configured_qed_channels: int = 0
    active_qed_channels: int = 0
    reduced_symmetry: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"exact", "df"}:
            raise ValueError("interaction mode must be 'exact' or 'df'.")
        if self.reduced_symmetry and self.mode != "exact":
            raise ValueError("Reduced-symmetry interactions require an exact ERI tensor.")
        if self.mode == "exact":
            if self.eri is None or self.df is not None or self.eri.ndim != 4:
                raise ValueError("exact interaction requires one four-index ERI tensor.")
            if self.eri.shape != (self.eri.shape[0],) * 4:
                raise ValueError("exact ERI tensor must have shape (nao, nao, nao, nao).")
            if not np.all(np.isfinite(self.eri)):
                raise ValueError("exact ERI tensor must contain finite values.")
        elif self.df is None or self.eri is not None:
            raise ValueError("DF interaction requires a factor bundle and no ERI tensor.")

    @classmethod
    def exact(
        cls,
        eri: np.ndarray,
        orbital_basis: str,
        configured_qed_channels: int = 0,
        active_qed_channels: int = 0,
        reduced_symmetry: bool = False,
    ) -> "ElectronInteraction":
        return cls(
            mode="exact",
            orbital_basis=orbital_basis,
            eri=eri,
            configured_qed_channels=configured_qed_channels,
            active_qed_channels=active_qed_channels,
            reduced_symmetry=reduced_symmetry,
        )

    @classmethod
    def factorized(
        cls,
        df: DFTensors,
        orbital_basis: str,
        se_fit_basis: str,
        fock_fit_basis: str,
    ) -> "ElectronInteraction":
        return cls(
            mode="df",
            orbital_basis=orbital_basis,
            df=df,
            se_fit_basis=se_fit_basis,
            fock_fit_basis=fock_fit_basis,
            configured_qed_channels=df.self_energy.configured_exact_channels,
            active_qed_channels=df.self_energy.active_exact_channels,
        )

    @property
    def is_factorized(self) -> bool:
        return self.mode == "df"

    def build_fock(self, core: np.ndarray, density: np.ndarray) -> np.ndarray:
        if self.is_factorized:
            assert self.df is not None
            return fock_mod.build_fock_df(core, density, self.df.f_Qpq)
        assert self.eri is not None
        if self.reduced_symmetry:
            return fock_mod.build_fock_reduced_symmetry(
                core,
                density,
                self.eri,
            )
        return fock_mod.build_fock(core, density, self.eri)

    def build_self_energy(
        self,
        nbf: int,
        green_tau: np.ndarray,
        ntime: int,
        n_workers: int | None = None,
    ) -> np.ndarray:

        if n_workers is None:
            n_workers = int(os.environ.get("QED_GF2_SIGMA_WORKERS", "1"))

        if self.is_factorized:
            assert self.df is not None
            return self_energy.gf2_self_energy_df(
                nbf,
                green_tau,
                self.df.Qpq,
                ntime,
                n_workers=n_workers,
            )
        assert self.eri is not None
        if self.reduced_symmetry:
            return self_energy.gf2_self_energy_reduced(
                nbf,
                green_tau,
                self.eri,
                ntime,
                n_workers=n_workers,
            )
        return self_energy.gf2_self_energy(
            nbf,
            green_tau,
            self.eri,
            ntime,
            n_workers=n_workers,
        )

    def metadata(self) -> dict[str, Any]:
        if not self.is_factorized:
            return {
                "interaction_mode": "exact",
                "orbital_basis": self.orbital_basis,
                "se_fit_basis": None,
                "fock_fit_basis": None,
                "electronic_aux_rank": None,
                "electronic_fock_aux_rank": None,
                "qed_low_rank_channels": self.configured_qed_channels,
                "active_qed_low_rank_channels": self.active_qed_channels,
                "effective_factor_rank": None,
                "effective_fock_factor_rank": None,
                "full_four_index_allocated": True,
            }
        assert self.df is not None
        se = self.df.self_energy
        fock = self.df.fock
        return {
            "interaction_mode": "df",
            "orbital_basis": self.orbital_basis,
            "se_fit_basis": self.se_fit_basis,
            "fock_fit_basis": self.fock_fit_basis,
            "electronic_aux_rank": se.electronic_rank,
            "electronic_fock_aux_rank": fock.electronic_rank,
            "qed_low_rank_channels": se.configured_exact_channels,
            "active_qed_low_rank_channels": se.active_exact_channels,
            "effective_factor_rank": se.effective_rank,
            "effective_fock_factor_rank": fock.effective_rank,
            "full_four_index_allocated": False,
        }
