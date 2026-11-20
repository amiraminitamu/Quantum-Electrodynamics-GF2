# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Exact restricted Fock/chemical-potential solver checks."""

import unittest

import numpy as np

from gf2.chemical_potential import (
    evaluate_restricted_population_at_mu,
    tune_restricted_population_brent,
)
from gf2.model_benchmark import (
    DimerParameters, default_gf2_options, run_csgf2, solve_cshf,
)
from gf2.mu_inner import RestrictedFockMuMixer, safeguarded_mu_step
from gf2.options import GF2Options


class RestrictedMuTest(unittest.TestCase):
    def test_options_and_guarded_step(self):
        self.assertEqual(GF2Options.from_mapping({}).restricted_mu_strategy, "auto")
        self.assertEqual(GF2Options.from_mapping({"restricted_mu_strategy": "nested"}).restricted_mu_strategy,
                         "nested")
        with self.assertRaisesRegex(ValueError, "restricted_mu_strategy"):
            GF2Options.from_mapping({"restricted_mu_strategy": "unknown"})
        self.assertAlmostEqual(safeguarded_mu_step(0., -.1, 2., damping=.5,
                               max_step=.02), .02)

    def test_restricted_density_slope_and_strict_root(self):
        fock = np.diag([-.2, .1])
        overlap = np.eye(2)
        sigma = np.zeros((3, 2, 2), dtype=complex)
        frequencies = np.array([.08, .3, .9])
        weights = np.array([.4 + .2j, -.1 + .15j, .08 - .04j])

        def state(mu):
            return evaluate_restricted_population_at_mu(
                fock, overlap, overlap, sigma, frequencies, weights, mu, beta=10.
            )

        finite_difference = (state(1e-5)[3] - state(-1e-5)[3]) / 2e-5
        self.assertAlmostEqual(state(0.)[4], finite_difference, delta=1e-9)
        mu, density, _, _ = tune_restricted_population_brent(
            fock, overlap, overlap, sigma, frequencies, weights,
            2, 0., beta=10., thresh=1e-8, maxiter=60
        )
        self.assertAlmostEqual(mu, -.05, delta=1e-7)
        self.assertAlmostEqual(float(np.trace(density)), 2., delta=1e-8)

    def test_mixer_bounds_mu_and_symmetrizes_fock(self):
        mixer = RestrictedFockMuMixer(2, charge_scale=.01, max_mu_step=.02)
        old = np.diag([0., 1.])
        built = np.array([[.01, .002], [.002, 1.01]])
        new_fock, new_mu, _ = mixer.update(old, built, -.2, -.19, -.001)
        np.testing.assert_allclose(new_fock, new_fock.T)
        self.assertLessEqual(abs(new_mu + .2), .02)

    def test_exact_model_coupled_and_nested_agree(self):
        parameters = DimerParameters(U=2., g=.1, delta=.1, nph=6)
        reference = solve_cshf(parameters)
        options = default_gf2_options(
            beta=20., ntime=40, nfreq=41, nfull=120, npoly=12,
            max_iterations=60,
        )
        options.update(thresh_energy=1e-5, thresh_ne=1e-6, thresh_fock_rms=1e-6)
        nested = run_csgf2(parameters, reference, {
            **options, "restricted_mu_strategy": "nested"
        })
        coupled = run_csgf2(parameters, reference, {
            **options, "restricted_mu_strategy": "coupled"
        })
        self.assertTrue(coupled.converged)
        self.assertLess(coupled.residuals["particle_number"], 1e-6)
        self.assertLess(abs(coupled.energy - nested.energy), 1e-6)
        self.assertLessEqual(coupled.iterations, nested.iterations)


if __name__ == "__main__":
    unittest.main()
