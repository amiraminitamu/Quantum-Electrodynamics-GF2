# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Tests for optional G / Sigma linear mixing."""

from __future__ import annotations

import pytest

from gf2.options import GF2Options


def test_mixing_defaults_are_none() -> None:
    opts = GF2Options.from_mapping({})
    assert opts.g_mix is None
    assert opts.sigma_mix is None


@pytest.mark.parametrize("token", ["none", "None", "null", ""])
def test_mixing_none_from_string(token: str) -> None:
    opts = GF2Options.from_mapping({"g_mix": token, "sigma_mix": token})
    assert opts.g_mix is None
    assert opts.sigma_mix is None


def test_mixing_accepts_fraction() -> None:
    opts = GF2Options.from_mapping({"g_mix": 0.5, "sigma_mix": 0.25})
    assert opts.g_mix == 0.5
    assert opts.sigma_mix == 0.25


def test_mixing_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="g_mix"):
        GF2Options.from_mapping({"g_mix": 1.5})
