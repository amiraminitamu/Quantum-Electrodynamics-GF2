# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
from __future__ import annotations
from typing import Any, Mapping
import numpy as np

try:
    import psi4
except ImportError:
    psi4 = None
from . import fock as fock_mod
from . import integrals as df_integrals
from . import legendre_ft, spline_ft
from .grids import GridBundle
from .interaction import ElectronInteraction
from .options import GF2Options
from .psi4_io import apply_quiet_psi4_options
from .qed import (
    CavityConfiguration,
    QEDIntegralData,
    add_multimode_dse,
    build_qed_integrals,
    coherent_displacement,
    coherent_fock_correction,
    molecular_dipole_ao,
)


def _build_cs_fock(
    h_qed: np.ndarray,
    density: np.ndarray,
    eri_qed: np.ndarray,
    qed_data: QEDIntegralData,
) -> np.ndarray:
    """Build the corrected CS Fock matrix with each DSE term included once."""
    z = coherent_displacement(qed_data.g, density, qed_data.omega)
    return fock_mod.build_fock(h_qed, density, eri_qed) + coherent_fock_correction(qed_data.g, z)


def _build_cs_fock_interaction(
    h_qed: np.ndarray,
    density: np.ndarray,
    interaction: ElectronInteraction,
    qed_data: QEDIntegralData,
) -> np.ndarray:
    """CS Fock builder shared by exact and factorized interactions."""
    z = coherent_displacement(qed_data.g, density, qed_data.omega)
    return interaction.build_fock(h_qed, density) + coherent_fock_correction(qed_data.g, z)


def _rhf_reference(
    mol: psi4.core.Molecule,
    psi4_options: Mapping[str, Any],
    opts: GF2Options,
    cavity: CavityConfiguration | None = None,
) -> dict[str, Any]:
    "Run a closed-shell HF calculation and return everything the driver needs."
    psi4_opts = apply_quiet_psi4_options(psi4_options)
    psi4_opts.setdefault("reference", "rhf")
    psi4.set_options(psi4_opts)

    hf_energy, scf_wfn = psi4.energy("scf", return_wfn=True)

    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("basis"))
    basisset = wfn.basisset()
    mints = psi4.core.MintsHelper(basisset)

    S = np.asarray(mints.ao_overlap())
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    H_core = T + V
    H_bare = H_core.copy()

    # same 1e-14 cutoff used by the original deterministic Matsubara driver).
    S_inv_half_mat = mints.ao_overlap()
    S_inv_half_mat.power(-0.5, 1.0e-14)
    S_inv_half_value = np.asarray(S_inv_half_mat)

    nocc = scf_wfn.nalpha()
    nbf = wfn.nso()
    eps = np.asarray(scf_wfn.epsilon_a())

    # as the reference driver: ``ρ = 2 C_occ C_occ^T`` (no symmetrization).
    C = np.asarray(scf_wfn.Ca())
    C_occ = C[:, :nocc]
    P = 2.0 * np.einsum("pi,qi->pq", C_occ, C_occ, optimize=True)

    qed_data: QEDIntegralData | None = None
    dipole = None
    electronic_dipole = None
    nuclear_dipole = None
    eri_bare = None
    eri = None
    df = None
    orbital_basis_name = str(basisset.name())
    interaction: ElectronInteraction
    if cavity is not None:
        electronic_dipole = np.asarray(mints.ao_dipole())
        coordinates = np.asarray(mol.geometry())
        charges = np.asarray([mol.Z(atom) for atom in range(mol.natom())])
        nuclear_dipole = np.einsum("a,ax->x", charges, coordinates, optimize=True)
        dipole = molecular_dipole_ao(electronic_dipole, S, nuclear_dipole, 2 * nocc)
        qed_data = build_qed_integrals(dipole, S, cavity)
        H_core = H_core + qed_data.h_dse
        active_qed = int(np.count_nonzero(np.any(qed_data.h_c != 0.0, axis=(1, 2))))
        if opts.eri_type == "std":
            eri_bare = mints.ao_eri().to_array()
            eri = add_multimode_dse(eri_bare, qed_data.h_c)
            interaction = ElectronInteraction.exact(
                eri,
                orbital_basis_name,
                configured_qed_channels=len(qed_data.h_c),
                active_qed_channels=active_qed,
            )
        else:
            electronic_df = df_integrals.build_df_eri_bundle(
                mol, basisset, opts.se_fit_basis, opts.fock_fit_basis
            )
            df = electronic_df.with_exact_channels(qed_data.h_c)
            interaction = ElectronInteraction.factorized(
                df,
                orbital_basis_name,
                opts.se_fit_basis,
                opts.fock_fit_basis,
            )

        # the GF loop share one interaction builder, so each DSE enters once.
        alpha = float(opts.extra.get("cs_hf_mix", 0.1))
        e_last: float | None = None
        if active_qed:
            for _ in range(int(opts.extra.get("maxiter_qed_scf", 1000))):
                F = _build_cs_fock_interaction(H_core, P, interaction, qed_data)
                f_orth = S_inv_half_value @ F @ S_inv_half_value
                _, u = np.linalg.eigh(0.5 * (f_orth + f_orth.T))
                c = S_inv_half_value @ u
                p_build = 2.0 * np.einsum("pi,qi->pq", c[:, :nocc], c[:, :nocc], optimize=True)
                mixed = (1.0 - alpha) * P + alpha * p_build
                p_new = 0.5 * (mixed + mixed.T)
                z_new = coherent_displacement(qed_data.g, p_new, qed_data.omega)
                a_new = np.einsum("xij,ji->x", qed_data.g, p_new, optimize=True).real
                e_new = fock_mod.core_energy(
                    interaction.build_fock(H_core, p_new),
                    H_core,
                    p_new,
                    float(mol.nuclear_repulsion_energy()),
                ) + float(np.sum(2.0 * z_new * a_new + qed_data.omega * z_new**2))
                dp = np.linalg.norm(p_new - P)
                if e_last is not None and abs(e_new - e_last) < 1.0e-8 and dp < 1.0e-7:
                    P = p_new
                    break
                P, e_last = p_new, e_new

        F = _build_cs_fock_interaction(H_core, P, interaction, qed_data)
    elif opts.eri_type == "std":
        eri = mints.ao_eri().to_array()
        interaction = ElectronInteraction.exact(eri, orbital_basis_name)
        F = interaction.build_fock(H_core, P)
    else:
        df = df_integrals.build_df_eri_bundle(mol, basisset, opts.se_fit_basis, opts.fock_fit_basis)
        interaction = ElectronInteraction.factorized(
            df,
            orbital_basis_name,
            opts.se_fit_basis,
            opts.fock_fit_basis,
        )
        F = interaction.build_fock(H_core, P)

    ref: dict[str, Any] = {
        "hf_energy": float(hf_energy),
        "scf_wfn": scf_wfn,
        "nocc": int(nocc),
        "nbf": int(nbf),
        "Ne": int(2 * nocc),
        "S": S,
        "S_inv_half": S_inv_half_value,
        "H_core": H_core,
        "H_bare": H_bare,
        "eri_bare": eri_bare,
        "dipole_xyz": dipole,
        "electronic_dipole_xyz": electronic_dipole,
        "nuclear_dipole": nuclear_dipole,
        "eri": eri,
        "df": df,
        "eps": eps,
        "P": P,
        "F": F,
        "E_nuc": float(mol.nuclear_repulsion_energy()),
        "qed_data": qed_data,
        "interaction": interaction,
        "interaction_metadata": interaction.metadata(),
    }
    return ref


def _build_transform_objects(grids: GridBundle, opts: GF2Options) -> dict[str, np.ndarray]:
    """Precompute spline / FFT / Legendre objects that are constant in the SCF."""
    e_iwbeta = legendre_ft.e_iwt_beta(grids.full_ifreq_grid, opts.beta)
    e_iwt = legendre_ft.e_iwt_all(grids.full_ifreq_grid, grids.itime_grid)

    V = spline_ft.spline_v_matrix(grids.full_ifreq_grid, grids.ifreq_grid)
    B = spline_ft.spline_vft(V, e_iwt)
    B_beta = spline_ft.spline_vft_beta(V, e_iwbeta)

    legendre_trans = legendre_ft.analytic_transformation_integrals(
        grids.w_prime, grids.int_list, opts.n_ifreq, opts.beta, opts.n_poly
    )
    legendre_weights = legendre_ft.legendre_fit_weights(
        opts.n_poly, grids.cos_grid, grids.d_itime, opts.beta
    )

    return {
        "e_iwbeta": e_iwbeta,
        "e_iwt": e_iwt,
        "V": V,
        "B": B,
        "B_beta": B_beta,
        "legendre_trans": legendre_trans,
        "legendre_weights": legendre_weights,
    }
