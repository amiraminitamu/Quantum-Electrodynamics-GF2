"""Spin-trace normalization and coupled-energy collapse."""

import numpy as np

from gf2.energy import (
    gf2_correlation,
    gf2_correlation_unrestricted,
    mp2_correlation,
    mp2_correlation_unrestricted,
    static_energy_unrestricted,
)
from gf2.coupled_energy import coupled_energy_from_state, coupled_energy_unrestricted_from_state


def test_spin_energy_collapse():
    rng = np.random.default_rng(33)
    g = rng.normal(size=(4, 2, 2))
    se = rng.normal(size=g.shape)
    w = np.full(4, 0.25)
    np.testing.assert_allclose(
        gf2_correlation_unrestricted(g, g, se, se, w), gf2_correlation(g, se, w)
    )
    np.testing.assert_allclose(
        mp2_correlation_unrestricted(g, g, se, se, w), mp2_correlation(g, se, w)
    )
    h = np.eye(2)
    f = 2 * np.eye(2)
    p = np.eye(2)
    np.testing.assert_allclose(static_energy_unrestricted(h, f, f, p / 2, p / 2, 1), 4.0)


def test_coupled_energy_component_collapse():
    rng = np.random.default_rng(11)
    g = rng.normal(size=(4, 2, 2))
    se = rng.normal(size=g.shape)
    eb = rng.normal(size=g.shape)
    pi = rng.normal(size=(4, 1, 1))
    d = rng.normal(size=pi.shape)
    d0iv = -np.ones((4, 1, 1), dtype=complex)
    div = d0iv * 0.9
    kw = dict(
        static_energy=-1.2,
        Pi_tau=pi,
        D_tau=d,
        D_iv=div,
        D0_iv=d0iv,
        d_itime=np.full(4, 0.25),
        omega=np.array([0.5]),
        beta=2.0,
    )
    r = coupled_energy_from_state(G_tau=g, Sigma_ee_tau=se, Sigma_eb_tau=eb, **kw)
    u = coupled_energy_unrestricted_from_state(
        G_alpha=g,
        G_beta=g,
        Sigma_ee_alpha=se,
        Sigma_ee_beta=se,
        Sigma_eb_alpha=eb,
        Sigma_eb_beta=eb,
        **kw,
    )
    for field in (
        "ee_correlation",
        "eb_fermionic_trace",
        "eb_interaction_sigma_g",
        "eb_interaction_pi_d",
        "photon_fluctuation_energy",
        "coupled_total",
    ):
        np.testing.assert_allclose(getattr(u, field), getattr(r, field))
