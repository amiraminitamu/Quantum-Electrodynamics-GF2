# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"Exactly solvable two-site electron--boson benchmark for CS methods."

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.linalg import eigh


@dataclass(frozen=True)
class DimerParameters:
    """Parameters for the half-filled two-site electron--boson model."""

    t: float = 1.0
    U: float = 2.0
    omega: float = 1.0
    g: float = 0.2
    delta: float = 0.0
    nph: int = 24

    def __post_init__(self) -> None:
        values = (self.t, self.U, self.omega, self.g, self.delta)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("All model parameters must be finite.")
        if self.t <= 0.0 or self.omega <= 0.0:
            raise ValueError("t and omega must be positive.")
        if self.U < 0.0 or self.g < 0.0:
            raise ValueError("U and g must be non-negative.")
        if self.nph < 2:
            raise ValueError("nph must be at least 2.")


@dataclass(frozen=True)
class ExactResult:
    """Exact canonical ground-state or grand-canonical thermal result."""

    energy: float
    one_body: float
    interaction: float
    electron_boson: float
    photon: float
    electron_number: float
    chemical_potential: float | None
    photon_top_probability: float


@dataclass(frozen=True)
class CSHFResult:
    """Lowest converged restricted CS-HF solution among deterministic starts."""

    energy: float
    density: np.ndarray
    fock: np.ndarray
    orbital_energies: np.ndarray
    coefficients: np.ndarray
    xi: float
    residual: float
    iterations: int
    converged: bool
    start: str
    converged_starts: int
    stationary_residual: float


@dataclass(frozen=True)
class CSMP2Result:
    """CS-HF plus zero-photon electronic and one-photon second-order terms."""

    energy: float
    reference_energy: float
    electronic_correlation: float
    electron_boson_correlation: float
    stationary_residual: float


@dataclass(frozen=True)
class CSGF2Result:
    """Energy and convergence metadata returned by the production solver."""

    energy: float
    static_energy: float
    correlation_energy: float
    converged: bool
    iterations: int
    wall_time_s: float
    residuals: Mapping[str, float]
    energy_components: Mapping[str, float]
    iteration_history: tuple[Mapping[str, float], ...]
    final_state: Mapping[str, Any]
    log_lines: tuple[str, ...]


def one_body_integrals(parameters: DimerParameters) -> np.ndarray:
    """Return the two-site one-electron matrix in the site basis."""
    p = parameters
    return np.array([[0.5 * p.delta, -p.t], [-p.t, -0.5 * p.delta]])


def coupling_integrals(parameters: DimerParameters) -> np.ndarray:
    """Return the linear displacement vertex ``g * diag(1,-1)``."""
    return np.diag([parameters.g, -parameters.g])


def interaction_integrals(parameters: DimerParameters) -> np.ndarray:
    """Return chemist-notation ERIs for the on-site Hubbard interaction."""
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = parameters.U
    eri[1, 1, 1, 1] = parameters.U
    return eri


def _annihilation(spin_orbital: int) -> np.ndarray:
    """Fermion annihilator on four spin orbitals.

    The ordering is ``(1 up, 1 down, 2 up, 2 down)``.
    """
    if spin_orbital not in range(4):
        raise ValueError("spin_orbital must be in range(4).")
    operator = np.zeros((16, 16))
    for state in range(16):
        if state & (1 << spin_orbital):
            occupied_below = (state & ((1 << spin_orbital) - 1)).bit_count()
            operator[state ^ (1 << spin_orbital), state] = (-1.0) ** occupied_below
    return operator


def _model_operators(parameters: DimerParameters) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build the full truncated Hamiltonian and its physical contributions."""
    p = parameters
    annihilators = [_annihilation(index) for index in range(4)]
    number = [operator.T @ operator for operator in annihilators]
    hopping = -p.t * sum(
        annihilators[index].T @ annihilators[index + 2]
        + annihilators[index + 2].T @ annihilators[index]
        for index in (0, 1)
    )
    imbalance = number[0] + number[1] - number[2] - number[3]
    bias = 0.5 * p.delta * imbalance
    hubbard = p.U * (number[0] @ number[1] + number[2] @ number[3])

    boson = np.diag(np.sqrt(np.arange(1, p.nph + 1)), 1)
    displacement = boson + boson.T
    identity_b = np.eye(p.nph + 1)
    identity_e = np.eye(16)
    terms = {
        "one_body": np.kron(hopping + bias, identity_b),
        "interaction": np.kron(hubbard, identity_b),
        "photon": np.kron(identity_e, p.omega * (boson.T @ boson)),
        "electron_boson": np.kron(p.g * imbalance, displacement),
    }
    return terms, sum(terms.values())


def _sector_indices(electron_number: int, nph: int) -> np.ndarray:
    return np.asarray(
        [
            electronic * (nph + 1) + photon
            for electronic in range(16)
            if electronic.bit_count() == electron_number
            for photon in range(nph + 1)
        ],
        dtype=int,
    )


def _expectation(state: np.ndarray, operator: np.ndarray) -> float:
    return float(np.real(state.conj() @ operator @ state))


def exact_ground_state(parameters: DimerParameters) -> ExactResult:
    """Diagonalize the canonical two-electron Hamiltonian."""
    terms, hamiltonian = _model_operators(parameters)
    indices = _sector_indices(2, parameters.nph)
    energies, vectors = eigh(hamiltonian[np.ix_(indices, indices)])
    state = np.zeros(hamiltonian.shape[0])
    state[indices] = vectors[:, 0]
    values = {name: _expectation(state, term) for name, term in terms.items()}
    top_probability = float(
        sum(
            abs(state[electronic * (parameters.nph + 1) + parameters.nph]) ** 2
            for electronic in range(16)
        )
    )
    return ExactResult(
        energy=float(energies[0]),
        one_body=values["one_body"],
        interaction=values["interaction"],
        electron_boson=values["electron_boson"],
        photon=values["photon"],
        electron_number=2.0,
        chemical_potential=None,
        photon_top_probability=top_probability,
    )


def _thermal_expectations_at_mu(
    sector_data: Mapping[int, tuple[np.ndarray, Mapping[str, np.ndarray]]],
    *,
    beta: float,
    chemical_potential: float,
) -> tuple[float, dict[str, float]]:
    shifted: list[np.ndarray] = []
    numbers: list[np.ndarray] = []
    observables: dict[str, list[np.ndarray]] = {}
    for electron_number, (energies, values) in sector_data.items():
        shifted.append(energies - chemical_potential * electron_number)
        numbers.append(np.full(energies.shape, float(electron_number)))
        for name, array in values.items():
            observables.setdefault(name, []).append(array)
    grand_energies = np.concatenate(shifted)
    electron_numbers = np.concatenate(numbers)
    offset = float(np.min(grand_energies))
    weights = np.exp(-beta * (grand_energies - offset))
    weights /= np.sum(weights)
    mean_number = float(weights @ electron_numbers)
    means = {name: float(weights @ np.concatenate(chunks)) for name, chunks in observables.items()}
    return mean_number, means


def exact_thermal(
    parameters: DimerParameters,
    beta: float,
    *,
    target_electrons: float = 2.0,
    number_tolerance: float = 1.0e-12,
) -> ExactResult:
    """Grand-canonical ED at the same ``beta`` and mean electron number as GF2."""
    if beta <= 0.0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and positive.")
    terms, hamiltonian = _model_operators(parameters)
    sector_data: dict[int, tuple[np.ndarray, Mapping[str, np.ndarray]]] = {}
    for electron_number in range(5):
        indices = _sector_indices(electron_number, parameters.nph)
        energies, vectors = eigh(hamiltonian[np.ix_(indices, indices)])
        values = {
            name: np.einsum(
                "im,im->m",
                vectors.conj(),
                term[np.ix_(indices, indices)] @ vectors,
                optimize=True,
            ).real
            for name, term in terms.items()
        }
        sector_data[electron_number] = (energies, values)

    scale = max(1.0, abs(parameters.U), abs(parameters.delta), parameters.omega)
    lower = -20.0 * scale
    upper = 20.0 * scale
    chemical_potential = 0.0
    for _ in range(240):
        midpoint = 0.5 * (lower + upper)
        mean_number, _ = _thermal_expectations_at_mu(
            sector_data, beta=beta, chemical_potential=midpoint
        )
        chemical_potential = midpoint
        if abs(mean_number - target_electrons) < number_tolerance:
            break
        if mean_number < target_electrons:
            lower = midpoint
        else:
            upper = midpoint
    mean_number, values = _thermal_expectations_at_mu(
        sector_data, beta=beta, chemical_potential=chemical_potential
    )
    energy = sum(values.values())
    return ExactResult(
        energy=float(energy),
        one_body=values["one_body"],
        interaction=values["interaction"],
        electron_boson=values["electron_boson"],
        photon=values["photon"],
        electron_number=mean_number,
        chemical_potential=chemical_potential,
        photon_top_probability=float("nan"),
    )


def _electronic_fock(h: np.ndarray, eri: np.ndarray, density: np.ndarray) -> np.ndarray:
    coulomb = np.einsum("pqrs,rs->pq", eri, density, optimize=True)
    exchange = np.einsum("prqs,rs->pq", eri, density, optimize=True)
    return h + coulomb - 0.5 * exchange


def _cshf_energy(
    h: np.ndarray,
    eri: np.ndarray,
    vertex: np.ndarray,
    omega: float,
    density: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    static_fock = _electronic_fock(h, eri, density)
    expectation = float(np.einsum("ij,ji->", vertex, density, optimize=True))
    xi = -expectation / omega
    electronic = 0.5 * float(np.einsum("ij,ji->", static_fock + h, density))
    coherent = 2.0 * xi * expectation + omega * xi * xi
    fock = static_fock + 2.0 * xi * vertex
    return electronic + coherent, xi, fock


def _initial_density(h: np.ndarray, label: str) -> np.ndarray:
    if label == "bonding":
        _, coefficients = eigh(h)
        occupied = coefficients[:, :1]
        return 2.0 * occupied @ occupied.T
    if label == "site1":
        return np.diag([2.0, 0.0])
    if label == "site2":
        return np.diag([0.0, 2.0])
    raise ValueError(f"Unknown CS-HF start {label!r}.")


def _solve_cshf_start(
    parameters: DimerParameters,
    *,
    start: str,
    tolerance: float,
    max_iterations: int,
    damping: float,
) -> CSHFResult:
    h = one_body_integrals(parameters)
    eri = interaction_integrals(parameters)
    vertex = coupling_integrals(parameters)
    density = _initial_density(h, start)
    residual = math.inf
    coefficients = np.eye(2)
    orbital_energies = np.zeros(2)
    fock = np.zeros((2, 2))
    xi = 0.0
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        _, xi, fock = _cshf_energy(h, eri, vertex, parameters.omega, density)
        orbital_energies, coefficients = eigh(fock)
        occupied = coefficients[:, :1]
        candidate = 2.0 * occupied @ occupied.T
        residual = float(np.linalg.norm(candidate - density))
        density = damping * candidate + (1.0 - damping) * density
        density = 0.5 * (density + density.T)
        if residual < tolerance:
            density = candidate
            converged = True
            break

    energy, xi, fock = _cshf_energy(h, eri, vertex, parameters.omega, density)
    orbital_energies, coefficients = eigh(fock)
    occupied = coefficients[:, :1]
    idempotent_density = 2.0 * occupied @ occupied.T
    residual = float(np.linalg.norm(idempotent_density - density))
    expectation = float(np.einsum("ij,ji->", vertex, density, optimize=True))
    stationary_residual = abs(parameters.omega * xi + expectation)
    return CSHFResult(
        energy=float(energy),
        density=density,
        fock=fock,
        orbital_energies=orbital_energies,
        coefficients=coefficients,
        xi=float(xi),
        residual=residual,
        iterations=iteration,
        converged=converged,
        start=start,
        converged_starts=int(converged),
        stationary_residual=float(stationary_residual),
    )


def solve_cshf(
    parameters: DimerParameters,
    *,
    starts: Iterable[str] = ("bonding", "site1", "site2"),
    tolerance: float = 1.0e-12,
    max_iterations: int = 500,
    damping: float = 0.5,
) -> CSHFResult:
    """Solve restricted CS-HF and select the lowest converged deterministic branch."""
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("tolerance and max_iterations must be positive.")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0,1].")
    labels = tuple(dict.fromkeys(starts))
    if not labels:
        raise ValueError("At least one CS-HF start is required.")
    solutions = [
        _solve_cshf_start(
            parameters,
            start=label,
            tolerance=tolerance,
            max_iterations=max_iterations,
            damping=damping,
        )
        for label in labels
    ]
    converged = [solution for solution in solutions if solution.converged]
    if not converged:
        details = ", ".join(
            f"{solution.start}: residual={solution.residual:.3e}" for solution in solutions
        )
        raise RuntimeError(f"CS-HF did not converge from any start ({details}).")
    best = min(converged, key=lambda solution: (solution.energy, solution.start))
    return CSHFResult(
        energy=best.energy,
        density=best.density,
        fock=best.fock,
        orbital_energies=best.orbital_energies,
        coefficients=best.coefficients,
        xi=best.xi,
        residual=best.residual,
        iterations=best.iterations,
        converged=True,
        start=best.start,
        converged_starts=len(converged),
        stationary_residual=best.stationary_residual,
    )


def csmp2_energy(parameters: DimerParameters, reference: CSHFResult) -> CSMP2Result:
    """Evaluate the explicit CS-MP2 energy used as the GF2 perturbative reference."""
    coefficients = reference.coefficients
    orbital_energies = reference.orbital_energies
    eri = interaction_integrals(parameters)
    eri_mo = np.einsum(
        "pi,qj,rk,sl,pqrs->ijkl",
        coefficients,
        coefficients,
        coefficients,
        coefficients,
        eri,
        optimize=True,
    )
    occupied = range(1)
    virtual = range(1, 2)
    electronic = 0.0
    for i in occupied:
        for j in occupied:
            for a in virtual:
                for b in virtual:
                    direct = eri_mo[i, a, j, b]
                    exchange = eri_mo[i, b, j, a]
                    denominator = (
                        orbital_energies[a]
                        + orbital_energies[b]
                        - orbital_energies[i]
                        - orbital_energies[j]
                    )
                    if denominator <= 0.0:
                        raise RuntimeError("CS-HF reference has a non-positive MP2 denominator.")
                    electronic -= direct * (2.0 * direct - exchange) / denominator

    vertex_mo = coefficients.T @ coupling_integrals(parameters) @ coefficients
    electron_boson = 0.0
    for i in occupied:
        for a in virtual:
            denominator = orbital_energies[a] - orbital_energies[i] + parameters.omega
            if denominator <= 0.0:
                raise RuntimeError("CS-HF reference has a non-positive one-photon denominator.")
            electron_boson -= 2.0 * abs(vertex_mo[i, a]) ** 2 / denominator

    return CSMP2Result(
        energy=float(reference.energy + electronic + electron_boson),
        reference_energy=reference.energy,
        electronic_correlation=float(electronic),
        electron_boson_correlation=float(electron_boson),
        stationary_residual=reference.stationary_residual,
    )


def model_reference_data(
    parameters: DimerParameters,
    reference: CSHFResult,
) -> dict[str, Any]:
    """Build the integral/reference mapping consumed by ``driver_theory.run_gf2``."""
    from .interaction import ElectronInteraction
    from .qed import QEDIntegralData

    h = one_body_integrals(parameters)
    eri = interaction_integrals(parameters)
    interaction = ElectronInteraction.exact(eri, "two-site model / orthonormal site basis")
    qed_data = QEDIntegralData(
        omega=np.array([parameters.omega]),
        h_c=np.zeros((1, 2, 2)),
        g=coupling_integrals(parameters)[None, :, :],
        h_dse=np.zeros((2, 2)),
    )
    return {
        "nbf": 2,
        "nocc": 1,
        "Ne": 2,
        "S": np.eye(2),
        "S_inv_half": np.eye(2),
        "H_core": h,
        "H_bare": h,
        "eri": eri,
        "eri_bare": eri,
        "P": reference.density.copy(),
        "F": reference.fock.copy(),
        "eps": reference.orbital_energies.copy(),
        "E_nuc": 0.0,
        "hf_energy": reference.energy,
        "qed_data": qed_data,
        "interaction": interaction,
        "interaction_metadata": interaction.metadata(),
    }


def run_csgf2(
    parameters: DimerParameters,
    reference: CSHFResult,
    options: Mapping[str, Any],
    *,
    job_name: str = "two_site_model",
    save_final_state: bool = False,
) -> CSGF2Result:
    """Run the production CS-GF2 fixed-point solver on the model integrals."""
    from .driver_theory import run_gf2
    from .qed import CavityConfiguration, PhotonMode
    from .reporting import Reporter

    reference_data = model_reference_data(parameters, reference)
    reporter = Reporter(job_name=job_name, echo=False)
    history: list[dict[str, float]] = []
    final_state: dict[str, Any] = {}

    def capture(event: str, payload: Mapping[str, Any]) -> None:
        nonlocal final_state
        if not event.startswith("iteration_"):
            return
        residuals = {key: float(value) for key, value in payload["residuals"].items()}
        history.append(
            {
                "iteration": float(payload["iteration"]),
                "total_energy": float(payload["total_energy"]),
                **residuals,
            }
        )
        final_state = dict(payload)

    cavity = CavityConfiguration([PhotonMode(parameters.omega, parameters.g, (0.0, 0.0, 1.0))])
    started = time.perf_counter()
    static, correlation, summary = run_gf2(
        None,
        {},
        options,
        method="cs-gf2",
        cavity=cavity,
        reporter=reporter,
        diagnostic_callback=capture,
        reference_data=reference_data,
    )
    elapsed = time.perf_counter() - started
    components = dict(summary.interaction_metadata.get("coupled_energy_components", {}))
    if not save_final_state:
        final_state = {
            key: value for key, value in final_state.items() if not isinstance(value, np.ndarray)
        }
    return CSGF2Result(
        energy=float(static + correlation),
        static_energy=float(static),
        correlation_energy=float(correlation),
        converged=bool(summary.converged),
        iterations=len(history),
        wall_time_s=float(elapsed),
        residuals={key: float(value) for key, value in summary.convergence_diagnostics.items()},
        energy_components={key: float(value) for key, value in components.items()},
        iteration_history=tuple(history),
        final_state=final_state,
        log_lines=tuple(reporter.lines),
    )


def default_gf2_options(
    *,
    beta: float = 200.0,
    ntime: int = 300,
    nfreq: int = 300,
    nfull: int = 10000,
    npoly: int = 50,
    max_iterations: int = 200,
) -> dict[str, Any]:
    """Publication-oriented model settings matching the molecular calculations."""
    if nfreq < 3:
        raise ValueError("nfreq must be at least 3.")
    return {
        "beta": beta,
        "eri_type": "std",
        "n_itime": ntime,
        "n_poly": npoly,
        "n_ifreq": nfreq,
        "mid_pt": min(100, nfreq - 1),
        "n_ifreq_full": nfull,
        "n_threads": 1,
        "mp2": False,
        "maxiter_gf_diis": max_iterations,
        "maxiter_fock_diis": 100,
        "g_mix": 0.5,
        "sigma_mix": 0.5,
        "dx_mix": 0.5,
        "thresh_energy": 1.0e-9,
        "thresh_ne": 1.0e-9,
        "thresh_fock_rms": 1.0e-9,
        "thresh_sigma_iw": 1.0e-8,
        "thresh_bosonic": 1.0e-8,
    }
