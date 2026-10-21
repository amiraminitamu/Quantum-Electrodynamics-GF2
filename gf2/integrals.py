# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Density-fitted two-electron integrals in the AO basis."

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

try:
    import psi4
except ImportError:  # pragma: no cover - psi4 is required at runtime
    psi4 = None  # type: ignore[assignment]


@dataclass
class FactorizedInteraction:
    "Metric-dressed electronic factors plus exact low-rank channels."

    electronic: np.ndarray
    exact_channels: np.ndarray | None = None
    _factors: np.ndarray = field(init=False, repr=False)
    configured_exact_channels: int = field(init=False)
    active_exact_channels: int = field(init=False)

    def __post_init__(self) -> None:
        electronic = _validate_factors(self.electronic, "electronic")
        self.electronic = electronic
        nao = electronic.shape[1]
        if self.exact_channels is None:
            channels = np.empty((0, nao, nao), dtype=electronic.dtype)
        else:
            channels = _validate_factors(self.exact_channels, "exact_channels", nao)
        self.configured_exact_channels = int(channels.shape[0])
        active = np.any(channels != 0.0, axis=(1, 2))
        channels = channels[active]
        self.active_exact_channels = int(channels.shape[0])
        self.exact_channels = channels
        self._factors = (
            electronic
            if not self.active_exact_channels
            else np.concatenate((electronic, channels), axis=0)
        )

    @property
    def factors(self) -> np.ndarray:
        """Effective factors, shape ``(electronic_rank + active, nao, nao)``."""
        return self._factors

    @property
    def electronic_rank(self) -> int:
        return int(self.electronic.shape[0])

    @property
    def effective_rank(self) -> int:
        return int(self._factors.shape[0])

    @property
    def nao(self) -> int:
        return int(self.electronic.shape[1])

    def reconstruct(self) -> np.ndarray:
        """Explicit four-index reconstruction for tests/reference work only."""
        return np.einsum("Apq,Ars->pqrs", self._factors, self._factors, optimize=True)


def _validate_factors(factors: np.ndarray, name: str, nao: int | None = None) -> np.ndarray:
    value = np.asarray(factors)
    if value.ndim != 3 or value.shape[1] != value.shape[2]:
        raise ValueError(f"{name} must have shape (nchannel, nao, nao).")
    if nao is not None and value.shape[1:] != (nao, nao):
        raise ValueError(f"{name} AO dimensions do not match electronic factors.")
    if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain finite numerical values.")
    if not np.allclose(value, value.swapaxes(1, 2), atol=1.0e-12, rtol=0.0):
        raise ValueError(f"{name} AO matrices must be symmetric.")
    return value


@dataclass
class DFTensors:
    "Bundle of density-fitted tensors used by the GF2 driver."

    self_energy: FactorizedInteraction
    fock: FactorizedInteraction

    @property
    def Qpq(self) -> np.ndarray:
        """Effective factors used by the GF2 self-energy contraction."""
        return self.self_energy.factors

    @property
    def f_Qpq(self) -> np.ndarray:
        """Effective factors used by the Fock contraction."""
        return self.fock.factors

    def with_exact_channels(self, channels: np.ndarray) -> "DFTensors":
        """Return a bundle with identical exact channels appended to each fit."""
        return DFTensors(
            self_energy=FactorizedInteraction(self.self_energy.electronic, channels),
            fock=FactorizedInteraction(self.fock.electronic, channels),
        )


def build_df_tensor(
    mol: "psi4.core.Molecule",
    orbital_basis: "psi4.core.BasisSet",
    aux_basis_name: str,
) -> np.ndarray:
    "Build the symmetric DF tensor ``Q_{Qpq} = (V^{-1/2})_{QP} (P|pq)``."
    if psi4 is None:  # pragma: no cover - defensive guard
        raise RuntimeError("psi4 is required to build density-fitted integrals.")

    aux = psi4.core.BasisSet.build(mol, "", target=aux_basis_name)
    zero = psi4.core.BasisSet.zero_ao_basis_set()
    mints = psi4.core.MintsHelper(orbital_basis)

    Ppq = np.squeeze(np.asarray(mints.ao_eri(aux, zero, orbital_basis, orbital_basis)))

    V_inv_half = mints.ao_eri(aux, zero, aux, zero)
    V_inv_half.power(-0.5, 1.0e-20)
    V_inv_half = np.squeeze(np.asarray(V_inv_half))

    return np.einsum("QP,Ppq->Qpq", V_inv_half, Ppq)


def build_df_eri_bundle(
    mol: "psi4.core.Molecule",
    orbital_basis: "psi4.core.BasisSet",
    se_fit_basis: str,
    fock_fit_basis: str,
) -> DFTensors:
    "Build the two DF tensors needed for a deterministic GF2 calculation."
    Qpq = build_df_tensor(mol, orbital_basis, se_fit_basis)
    f_Qpq = build_df_tensor(mol, orbital_basis, fock_fit_basis)
    return DFTensors(
        self_energy=FactorizedInteraction(Qpq),
        fock=FactorizedInteraction(f_Qpq),
    )
