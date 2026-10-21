# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
import numpy as np

from gf2 import fock as fock_mod
from gf2.interaction import ElectronInteraction
from gf2.self_energy import (
    gf2_self_energy,
    gf2_self_energy_reduced,
)


def _symmetric_green(rng, ntime, nbf):
    g = rng.normal(size=(ntime, nbf, nbf))
    return 0.5 * (g + g.transpose(0, 2, 1))


def _reduced_eri(rng, nbf):
    """Pair-exchange + simultaneous-reversal symmetry only."""
    v = rng.normal(size=(nbf, nbf, nbf, nbf))

    return 0.25 * (v + v.transpose(2, 3, 0, 1) + v.transpose(1, 0, 3, 2) + v.transpose(3, 2, 1, 0))


def test_reduced_self_energy_matches_literal_einsum():
    rng = np.random.default_rng(1827)

    nbf = 3
    ntime = 5

    g = _symmetric_green(rng, ntime, nbf)
    eri = _reduced_eri(rng, nbf)

    # Make sure this is genuinely not an ordinary Coulomb tensor.
    assert np.max(np.abs(eri - eri.transpose(1, 0, 2, 3))) > 1.0e-4

    sigma = gf2_self_energy_reduced(
        nbf,
        g,
        eri,
        ntime,
        n_workers=1,
    )

    reference = np.empty_like(sigma)

    for tau in range(ntime):
        gt = g[tau]
        gb = g[ntime - tau - 1]

        direct = 2.0 * np.einsum(
            "ikqm,kl,mn,pq,pnjl->ij",
            eri,
            gt,
            gt,
            gb,
            eri,
            optimize=True,
        )

        exchange = np.einsum(
            "ikqm,kl,mn,pq,pljn->ij",
            eri,
            gt,
            gt,
            gb,
            eri,
            optimize=True,
        )

        reference[tau] = direct - exchange

    np.testing.assert_allclose(
        sigma,
        reference,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_reduced_kernel_collapses_to_standard_for_coulomb_eri():
    rng = np.random.default_rng(991)

    nbf = 3
    ntime = 5
    naux = 4

    g = _symmetric_green(rng, ntime, nbf)

    b = rng.normal(size=(naux, nbf, nbf))
    b = 0.5 * (b + b.transpose(0, 2, 1))

    eri = np.einsum(
        "Qpq,Qrs->pqrs",
        b,
        b,
        optimize=True,
    )

    standard = gf2_self_energy(
        nbf,
        g,
        eri,
        ntime,
        n_workers=1,
    )

    reduced = gf2_self_energy_reduced(
        nbf,
        g,
        eri,
        ntime,
        n_workers=1,
    )

    np.testing.assert_allclose(
        reduced,
        standard,
        atol=1.0e-11,
        rtol=1.0e-11,
    )


def test_reduced_interaction_dispatches_fock_and_self_energy():
    rng = np.random.default_rng(331)

    nbf = 3
    ntime = 5

    h = rng.normal(size=(nbf, nbf))
    h = 0.5 * (h + h.T)

    p = rng.normal(size=(nbf, nbf))
    p = 0.5 * (p + p.T)

    g = _symmetric_green(rng, ntime, nbf)
    eri = _reduced_eri(rng, nbf)

    interaction = ElectronInteraction.exact(
        eri,
        "reduced-symmetry-test",
        reduced_symmetry=True,
    )

    f_expected = fock_mod.build_fock_reduced_symmetry(
        h,
        p,
        eri,
    )

    f_actual = interaction.build_fock(
        h,
        p,
    )

    np.testing.assert_allclose(
        f_actual,
        f_expected,
        atol=1.0e-13,
        rtol=1.0e-13,
    )

    sigma_expected = gf2_self_energy_reduced(
        nbf,
        g,
        eri,
        ntime,
        n_workers=1,
    )

    sigma_actual = interaction.build_self_energy(
        nbf,
        g,
        ntime,
        n_workers=1,
    )

    np.testing.assert_allclose(
        sigma_actual,
        sigma_expected,
        atol=1.0e-13,
        rtol=1.0e-13,
    )
