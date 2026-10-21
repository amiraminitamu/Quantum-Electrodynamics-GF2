# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Control Psi4 console and log output."""

from __future__ import annotations

from typing import Any, Mapping

try:
    import psi4
except ImportError:  # pragma: no cover
    psi4 = None  # type: ignore[assignment]


_QUIET_OPTIONS: dict[str, Any] = {
    "print_basis": False,
}


def silence_psi4() -> None:
    """Send Psi4 library output to ``/dev/null``."""
    if psi4 is None:  # pragma: no cover
        return
    psi4.core.be_quiet()


def reopen_psi4_output() -> None:
    """Restore Psi4's normal stdout behaviour (used with ``--verbose``)."""
    if psi4 is None:  # pragma: no cover
        return
    psi4.core.reopen_outfile()


def apply_quiet_psi4_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``options`` with quiet defaults filled in."""
    merged = dict(options)
    for key, val in _QUIET_OPTIONS.items():
        merged.setdefault(key, val)
    return merged
