# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Default options and a lightweight options object for the GF2 driver."

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


OPTIONAL_FLOAT_KEYS = frozenset({"g_mix", "sigma_mix"})


DEFAULTS: dict[str, Any] = {
    "module": "ft",
    "type": "deterministic",
    "eri_type": "df",
    "se_fit_basis": "cc-pvdz-ri",
    "fock_fit_basis": "cc-pvdz-ri",
    "df_type": "RIFIT",
    "beta": 50.0,
    "n_ifreq": 300,
    "mid_pt": 100,
    "n_ifreq_full": 10000,
    "n_itime": 300,
    "n_poly": 50,
    "grid_changed": True,
    "thresh_ne": 1.0e-6,
    "thresh_fock_rms": 1.0e-6,
    "thresh_energy": 1.0e-6,
    "bisection_on": False,
    "maxiter_bisection": 60,
    "newton_on": True,
    "newton_damping": 0.5,
    "maxiter_newton": 60,
    "mu_guess_thresh": 1.0e-6,
    "fit_width": 0.05,
    "n_fit_pts": 50000,
    "maxiter_fock_diis": 30,
    "maxiter_gf_diis": 30,
    "n_threads": 2,
    "fock_diis_on_off": "on",
    "debug": False,
    "hf": True,
    "mp2": True,
    "g_mix": None,
    "sigma_mix": None,
    "dask": False,
    "dask_scheduler": "",
}


@dataclass
class GF2Options:
    """Strongly-typed view over the user-supplied GF2 options dictionary."""

    module: str = "ft"
    type: str = "deterministic"
    eri_type: str = "df"
    se_fit_basis: str = "cc-pvdz-ri"
    fock_fit_basis: str = "cc-pvdz-ri"
    df_type: str = "RIFIT"
    beta: float = 50.0
    n_ifreq: int = 300
    mid_pt: int = 100
    n_ifreq_full: int = 10000
    n_itime: int = 300
    n_poly: int = 50
    grid_changed: bool = True
    thresh_ne: float = 1.0e-6
    thresh_fock_rms: float = 1.0e-6
    thresh_energy: float = 1.0e-6
    bisection_on: bool = False
    maxiter_bisection: int = 60
    newton_on: bool = True
    newton_damping: float = 0.5
    maxiter_newton: int = 60
    mu_guess_thresh: float = 1.0e-6
    fit_width: float = 0.05
    n_fit_pts: int = 50000
    maxiter_fock_diis: int = 30
    maxiter_gf_diis: int = 30
    n_threads: int = 2
    fock_diis_on_off: str = "on"
    debug: bool = False
    hf: bool = True
    mp2: bool = True
    g_mix: float | None = None
    sigma_mix: float | None = None
    dask: bool = False
    dask_scheduler: str = ""

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "GF2Options":
        """Build a :class:`GF2Options` from a partial user-supplied mapping."""
        normalised: dict[str, Any] = {}
        extra: dict[str, Any] = {}

        known = set(DEFAULTS.keys())

        for key, val in DEFAULTS.items():
            normalised[key] = val

        for key, raw in mapping.items():
            if key in known:
                normalised[key] = _coerce(key, raw)
            else:
                extra[key] = raw

        if normalised["module"] != "ft":
            raise ValueError("Only module='ft' (finite-temperature Matsubara) is supported.")
        if normalised["type"] != "deterministic":
            raise ValueError("Only type='deterministic' is supported.")
        if normalised["eri_type"] not in {"std", "df"}:
            raise ValueError("eri_type must be either 'std' or 'df'.")
        if normalised["dask"]:
            raise ValueError("Distributed execution is not supported.")
        if normalised["n_ifreq"] <= normalised["mid_pt"]:
            raise ValueError("n_ifreq must be larger than mid_pt.")
        if normalised["n_ifreq_full"] <= normalised["n_ifreq"]:
            raise ValueError("n_ifreq_full must be larger than n_ifreq.")
        if normalised["beta"] <= 0.0:
            raise ValueError("beta must be positive.")
        for key in OPTIONAL_FLOAT_KEYS:
            val = normalised[key]
            if val is not None and not (0.0 < val <= 1.0):
                raise ValueError(f"{key} must be in (0, 1] or None (no mixing).")

        return cls(**normalised, extra=extra)


def _coerce(key: str, value: Any) -> Any:
    """Best-effort cast of a raw user value to the type of the default."""
    if key in OPTIONAL_FLOAT_KEYS:
        if value is None:
            return None
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"", "none", "null", "nil"}:
                return None
        return float(value)
    if key not in DEFAULTS:
        return value
    target = DEFAULTS[key]
    if isinstance(target, bool):
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "yes", "on", "t", "1"}:
                return True
            if v in {"false", "no", "off", "f", "0"}:
                return False
            raise ValueError(f"Cannot interpret {value!r} as a boolean for {key!r}.")
        return bool(value)
    if isinstance(target, int) and not isinstance(target, bool):
        return int(value)
    if isinstance(target, float):
        return float(value)
    return str(value) if isinstance(target, str) else value
