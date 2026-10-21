# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"DIIS extrapolation for real (Fock) and complex (Green's function) vectors."

from __future__ import annotations

import numpy as np


class _DIISBase:
    """Shared bookkeeping for the real / complex DIIS objects."""

    _dtype: type = np.float64

    def __init__(self, max_vec: int = 6, removal_policy: str = "OLDEST"):
        self.max_vec = max_vec
        self.removal_policy = removal_policy.upper()
        if self.removal_policy not in {"OLDEST", "LARGEST"}:
            raise ValueError("DIIS: removal_policy must be 'OLDEST' or 'LARGEST'.")
        self.state: list[np.ndarray] = []
        self.error: list[np.ndarray] = []

    def add(self, state: np.ndarray, error: np.ndarray) -> None:
        """Push a new (state, error) pair onto the DIIS history."""
        self.state.append(np.array(state, copy=True))
        self.error.append(np.array(error, copy=True))

    def _prune(self) -> int:
        """Drop the oldest / largest entry once the buffer is over capacity."""
        n = len(self.state)
        if n <= self.max_vec:
            return n
        if self.removal_policy == "OLDEST":
            pos = 0
        else:
            rms = [np.sqrt(np.mean(np.abs(e) ** 2)) for e in self.error]
            pos = int(np.argmax(rms))
        del self.state[pos]
        del self.error[pos]
        return n - 1

    def extrapolate(self) -> np.ndarray:
        """Solve the Pulay equations and return the extrapolated state."""
        n = self._prune()
        if n == 0:
            raise RuntimeError("DIIS: no stored vectors to extrapolate from.")
        if n == 1:
            return np.array(self.state[0], copy=True)

        B = np.empty((n + 1, n + 1), dtype=self._dtype)
        B[-1, :] = 1
        B[:, -1] = 1
        B[-1, -1] = 0
        for i, e_i in enumerate(self.error):
            B[i, i] = np.vdot(e_i, e_i)
            for j in range(i):
                val = np.vdot(e_i, self.error[j])
                B[i, j] = val
                B[j, i] = val

        rhs = np.zeros(n + 1, dtype=self._dtype)
        rhs[-1] = 1.0

        diag = np.diag(B)[:-1]
        if np.any(np.real(diag) <= 0.0):
            scale = np.ones(n + 1, dtype=self._dtype)
        else:
            scale = np.empty(n + 1, dtype=self._dtype)
            scale[:-1] = np.real(diag) ** -0.5
            scale[-1] = 1.0

        B_scaled = B * (scale[:, None] * scale[None, :])
        ci = np.linalg.pinv(B_scaled) @ rhs
        ci = ci * scale

        out = np.zeros_like(self.state[0])
        for k, c in enumerate(ci[:-1]):
            out = out + c * self.state[k]
        return out


class RealDIIS(_DIISBase):
    """DIIS for real-valued state / error vectors (used for the Fock matrix)."""

    _dtype = np.float64


class ComplexDIIS(_DIISBase):
    """DIIS for complex-valued state / error vectors (used for ``G(iω)``)."""

    _dtype = np.complex128
