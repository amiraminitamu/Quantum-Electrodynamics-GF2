# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Production DF CS-GF2 must never materialize effective four-index ERIs."""

from __future__ import annotations

import numpy as np

from gf2.integrals import FactorizedInteraction
from tests.cs_df_helpers import ONE_Z, molecule


def test_df_driver_does_not_call_explicit_dse_or_reconstruct(monkeypatch) -> None:
    import gf2.driver_theory as driver

    def forbidden(*args, **kwargs):
        raise AssertionError("four-index effective interaction was requested")

    from gf2 import reference

    monkeypatch.setattr(reference, "add_multimode_dse", forbidden)
    monkeypatch.setattr(FactorizedInteraction, "reconstruct", forbidden)
    options = {
        "eri_type": "df",
        "se_fit_basis": "cc-pvdz-ri",
        "fock_fit_basis": "cc-pvdz-ri",
        "beta": 20.0,
        "n_ifreq": 30,
        "mid_pt": 10,
        "n_ifreq_full": 200,
        "n_itime": 30,
        "n_poly": 20,
        "mp2": False,
        "maxiter_gf_diis": 10,
        "maxiter_fock_diis": 8,
        "n_threads": 1,
        "thresh_energy": 1.0e-5,
        "maxiter_newton": 20,
        "n_fit_pts": 1000,
        "cs_hf_mix": 0.2,
    }
    p4 = {"basis": "sto-3g", "scf_type": "pk"}
    captures = {}
    _, _, summary = driver.run_gf2(
        molecule("h2"),
        p4,
        options,
        method="cs-gf2",
        cavity=driver.CavityConfiguration(ONE_Z),
        diagnostic_callback=lambda stage, data: captures.update({stage: data}),
    )
    reference = captures["reference"]
    assert summary.converged
    assert reference["eri"] is None
    assert reference["eri_bare"] is None
    assert reference["df"].Qpq.ndim == 3
    assert summary.interaction_metadata["full_four_index_allocated"] is False
    nao = reference["nbf"]
    rank = summary.interaction_metadata["effective_factor_rank"]
    assert reference["df"].Qpq.nbytes == rank * nao * nao * np.dtype(float).itemsize
