"""Explicit spin sum in coherent displacement, Fan, and photon bubble."""

import numpy as np

from gf2.qed import (
    coherent_displacement,
    polarization_tau,
    polarization_tau_unrestricted,
    sigma_eb_tau,
)


def test_qed_spin_kernels():
    rng = np.random.default_rng(123)
    g = rng.normal(size=(2, 2, 2))
    g = 0.5 * (g + g.swapaxes(1, 2))
    pa = rng.normal(size=(2, 2))
    pb = rng.normal(size=(2, 2))
    omega = np.array([0.4, 0.8])
    z = coherent_displacement(g, pa + pb, omega)
    np.testing.assert_allclose(z, -np.einsum("xij,ji->x", g, pa + pb) / omega)
    a = rng.normal(size=(4, 2, 2))
    b = rng.normal(size=(4, 2, 2))
    d = rng.normal(size=(4, 2, 2))
    np.testing.assert_allclose(
        sigma_eb_tau(a, d, g), -np.einsum("xik,tkl,ylj,txy->tij", g, a, g, d)
    )
    np.testing.assert_allclose(
        sigma_eb_tau(b, d, g), -np.einsum("xik,tkl,ylj,txy->tij", g, b, g, d)
    )
    pi = polarization_tau_unrestricted(a, b, g)
    expected = -sum(np.einsum("xij,tjk,ykl,tli->txy", g, x, g, x[::-1]) for x in (a, b))
    np.testing.assert_allclose(pi, expected)
    np.testing.assert_allclose(polarization_tau_unrestricted(a, a, g), polarization_tau(a, g))
