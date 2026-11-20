# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Compare exact CS-GF2 convergence routes on the same two-site model."""

from gf2.model_benchmark import (
    DimerParameters, default_gf2_options, run_csgf2, solve_cshf,
)


def main() -> None:
    parameters = DimerParameters(U=2., g=.1, delta=.1, nph=6)
    reference = solve_cshf(parameters)
    options = default_gf2_options(
        beta=20., ntime=40, nfreq=41, nfull=120, npoly=12,
        max_iterations=60,
    )
    options.update(thresh_energy=1e-5, thresh_ne=1e-6, thresh_fock_rms=1e-6)
    for route in ("nested", "coupled"):
        result = run_csgf2(parameters, reference, {
            **options, "restricted_mu_strategy": route,
        })
        print(
            f"{route:7} converged={result.converged} "
            f"outer={result.iterations:2d} E={result.energy:.12f} Eh "
            f"|delta N|={result.residuals['particle_number']:.3e}"
        )


if __name__ == "__main__":
    main()
