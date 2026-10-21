# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Canonical public API for electronic, coherent-state, and projected LF-GF2."

from __future__ import annotations

from typing import Any

from ._version import __version__


__all__ = [
    "CavityConfiguration",
    "GF2Options",
    "PhotonMode",
    "RunSummary",
    "run_gf2",
    "__version__",
]


def __getattr__(name: str) -> Any:
    """Resolve only the documented deterministic public API lazily."""
    if name == "run_gf2":
        from .driver_theory import run_gf2

        return run_gf2
    if name == "GF2Options":
        from .options import GF2Options

        return GF2Options
    if name == "RunSummary":
        from .reporting import RunSummary

        return RunSummary
    if name in {"CavityConfiguration", "PhotonMode"}:
        from .qed import CavityConfiguration, PhotonMode

        return {
            "CavityConfiguration": CavityConfiguration,
            "PhotonMode": PhotonMode,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
