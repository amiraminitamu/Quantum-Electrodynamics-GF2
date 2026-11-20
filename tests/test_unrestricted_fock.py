"""Spin Fock contractions and closed-shell collapse."""

import numpy as np
import pytest

from gf2.integrals import DFTensors, FactorizedInteraction
from gf2.interaction import ElectronInteraction


@pytest.mark.parametrize("mode", ["std", "df"])
def test_unrestricted_fock_reference_and_collapse(mode):
    rng = np.random.default_rng(2026)
    n = 3
    q = rng.normal(size=(4, n, n))
    q = 0.5 * (q + q.swapaxes(1, 2))
    # The last factor mimics an exact DSE channel.
    q[-1] *= 0.2
    v = np.einsum("Qpq,Qrs->pqrs", q, q)
    h = rng.normal(size=(n, n))
    h = 0.5 * (h + h.T)
    pa = rng.normal(size=(n, n))
    pa = 0.5 * (pa + pa.T)
    pb = rng.normal(size=(n, n))
    pb = 0.5 * (pb + pb.T)
    if mode == "std":
        interaction = ElectronInteraction.exact(v, "test")
    else:
        factors = FactorizedInteraction(q[:-1], q[-1:])
        interaction = ElectronInteraction.factorized(
            DFTensors(factors, factors), "test", "aux", "aux"
        )
    fa, fb = interaction.build_fock_unrestricted(h, pa, pb)
    direct = np.einsum("pqrs,rs->pq", v, pa + pb)
    exa = np.einsum("prqs,rs->pq", v, pa)
    exb = np.einsum("prqs,rs->pq", v, pb)
    np.testing.assert_allclose(fa, h + direct - exa, atol=1e-12)
    np.testing.assert_allclose(fb, h + direct - exb, atol=1e-12)
    p = pa + pb
    ca, cb = interaction.build_fock_unrestricted(h, p / 2, p / 2)
    np.testing.assert_allclose(ca, interaction.build_fock(h, p), atol=1e-12)
    np.testing.assert_allclose(cb, interaction.build_fock(h, p), atol=1e-12)
