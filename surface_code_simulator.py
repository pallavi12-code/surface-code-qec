"""Monte Carlo simulation utilities for a rotated surface-code memory.

The implementation uses Stim for fast stabilizer-circuit sampling and
PyMatching for minimum-weight perfect matching decoding.  The circuit has one
logical X observable, so a decoding failure is an incorrect prediction of that
observable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pymatching
import stim

NOISE_MODELS = ("depolarizing", "measurement", "reset", "combined")


@dataclass(frozen=True)
class SimulationResult:
    """Summary and raw counts from one Monte Carlo point."""

    distance: int
    physical_error_rate: float
    shots: int
    logical_errors: int
    physical_qubits: int
    detectors: int
    observables: int
    noise_model: str
    qec: bool

    @property
    def logical_error_rate(self) -> float:
        """Return the empirical logical error probability."""

        return self.logical_errors / self.shots

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """Return a 95% Wilson confidence interval for the error rate."""

        rate = self.logical_error_rate
        if self.logical_errors == 0:
            rate = 0.0
        z = 1.959963984540054
        denominator = 1 + z**2 / self.shots
        center = (rate + z**2 / (2 * self.shots)) / denominator
        margin = (
            z
            * np.sqrt(
                rate * (1 - rate) / self.shots
                + z**2 / (4 * self.shots**2)
            )
            / denominator
        )
        lower = 0.0 if self.logical_errors == 0 else max(0.0, center - margin)
        return float(lower), min(1.0, float(center + margin))


class RotatedSurfaceCodeSimulator:
    """Build, compile, decode, and sample a rotated surface-code memory."""

    def __init__(
        self, distance: int, physical_error_rate: float, noise_model: str = "depolarizing"
    ) -> None:
        if distance < 2:
            raise ValueError("distance must be at least 2")
        if not 0 <= physical_error_rate <= 1:
            raise ValueError("physical_error_rate must be between 0 and 1")
        if noise_model not in NOISE_MODELS:
            raise ValueError(f"noise_model must be one of {NOISE_MODELS}")

        self.distance = distance
        self.physical_error_rate = physical_error_rate
        self.noise_model = noise_model
        self.circuit = self._build_circuit()
        self.detector_error_model = self.circuit.detector_error_model(
            decompose_errors=True
        )
        self.decoder = pymatching.Matching.from_detector_error_model(
            self.detector_error_model
        )

    def _build_circuit(self) -> stim.Circuit:
        """Construct the generated rotated X-memory circuit."""

        noise = self.physical_error_rate
        kwargs = {
            "after_clifford_depolarization": noise
            if self.noise_model in ("depolarizing", "combined")
            else 0,
            "before_measure_flip_probability": noise
            if self.noise_model in ("measurement", "combined")
            else 0,
            "after_reset_flip_probability": noise
            if self.noise_model in ("reset", "combined")
            else 0,
        }
        return stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=self.distance,
            rounds=self.distance,
            **kwargs,
        )

    def sample(self, shots: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Sample detector syndromes and logical observables from Stim."""

        if shots <= 0:
            raise ValueError("shots must be positive")
        sampler = self.circuit.compile_detector_sampler(seed=seed)
        sampled = sampler.sample(
            shots=shots,
            separate_observables=True,
        )
        detectors, observables = sampled
        return np.asarray(detectors, dtype=np.uint8), np.asarray(observables, dtype=np.uint8)

    def run(self, shots: int, seed: int | None = None) -> SimulationResult:
        """Run shots, decode in a batch, and calculate the logical error rate."""

        detectors, observables = self.sample(shots, seed=seed)
        predictions = np.asarray(self.decoder.decode_batch(detectors), dtype=np.uint8)
        actual = observables[:, 0]
        logical_errors = int(np.count_nonzero(predictions[:, 0] != actual))
        return SimulationResult(
            distance=self.distance,
            physical_error_rate=self.physical_error_rate,
            shots=shots,
            logical_errors=logical_errors,
            physical_qubits=self.circuit.num_qubits,
            detectors=self.circuit.num_detectors,
            observables=self.circuit.num_observables,
            noise_model=self.noise_model,
            qec=True,
        )


def simulate_no_qec(
    physical_error_rate: float,
    shots: int,
    seed: int | None = None,
    noise_model: str = "depolarizing",
) -> SimulationResult:
    """Simulate one unencoded physical qubit as the no-QEC baseline."""

    if not 0 <= physical_error_rate <= 1:
        raise ValueError("physical_error_rate must be between 0 and 1")
    if shots <= 0:
        raise ValueError("shots must be positive")
    if noise_model not in NOISE_MODELS:
        raise ValueError(f"noise_model must be one of {NOISE_MODELS}")

    # A depolarizing channel flips the computational-basis bit for X or Y.
    flip_probability = (
        2 * physical_error_rate / 3
        if noise_model == "depolarizing"
        else physical_error_rate
    )
    rng = np.random.default_rng(seed)
    logical_errors = int(np.count_nonzero(rng.random(shots) < flip_probability))
    return SimulationResult(
        distance=1,
        physical_error_rate=physical_error_rate,
        shots=shots,
        logical_errors=logical_errors,
        physical_qubits=1,
        detectors=0,
        observables=1,
        noise_model=noise_model,
        qec=False,
    )


def simulate(
    distance: int,
    physical_error_rate: float,
    shots: int,
    seed: int | None = None,
    noise_model: str = "depolarizing",
) -> SimulationResult:
    """Convenience wrapper for a single simulation point."""

    return RotatedSurfaceCodeSimulator(distance, physical_error_rate, noise_model).run(
        shots, seed=seed
    )


def run_matrix(
    distances: Iterable[int],
    physical_error_rates: Iterable[float],
    shots: int,
    seed: int | None = None,
    noise_model: str = "depolarizing",
    include_no_qec: bool = True,
) -> list[SimulationResult]:
    """Run a reproducible matrix, deriving a distinct seed per data point."""

    results: list[SimulationResult] = []
    rates = tuple(physical_error_rates)
    point = 0
    if include_no_qec:
        for rate in rates:
            point_seed = None if seed is None else seed + point
            results.append(simulate_no_qec(rate, shots, point_seed, noise_model))
            point += 1
    for distance in distances:
        for rate in rates:
            point_seed = None if seed is None else seed + point
            results.append(simulate(distance, rate, shots, point_seed, noise_model))
            point += 1
    return results
