"""Public strong-simulation API for TerKet."""

from __future__ import annotations

from .circuits import (
    CircuitSpec,
    bits_to_big_endian_string,
    bits_to_index,
    bits_to_little_endian_string,
    from_qiskit,
    lift_exact_dyadic_precision,
    make_circuit,
    normalize_circuit,
)
from .cubic_arithmetic import CubicFunction, PhaseFunction
from .schur_engine import (
    ProbabilityInfo,
    ScaledAmplitude,
    ScaledProbability,
    SchurState,
    SolverConfig,
    analyze_amplitudes,
    analyze_circuit,
    build_state,
    compute_amplitude,
    compute_amplitudes,
    compute_amplitude_scaled,
    compute_circuit_amplitude,
    compute_circuit_amplitude_scaled,
    compute_circuit_probability,
    compute_circuit_probability_scaled,
    reduce_and_sum,
)

__all__ = [
    "CircuitSpec",
    "CubicFunction",
    "PhaseFunction",
    "ScaledAmplitude",
    "SchurState",
    "SolverConfig",
    "analyze_amplitudes",
    "analyze_circuit",
    "bits_to_big_endian_string",
    "bits_to_index",
    "bits_to_little_endian_string",
    "build_state",
    "compute_amplitude",
    "compute_amplitudes",
    "compute_amplitude_scaled",
    "compute_circuit_amplitude",
    "compute_circuit_amplitude_scaled",
    "compute_circuit_probability",
    "compute_circuit_probability_scaled",
    "from_qiskit",
    "lift_exact_dyadic_precision",
    "make_circuit",
    "normalize_circuit",
    "ProbabilityInfo",
    "reduce_and_sum",
    "ScaledProbability",
]
