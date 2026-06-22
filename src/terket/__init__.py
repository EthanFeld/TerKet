"""Public strong-simulation API for TerKet."""

from __future__ import annotations

from importlib import import_module

from .spec import (
    CircuitSpec,
    bits_to_big_endian_string,
    bits_to_index,
    bits_to_little_endian_string,
    lift_exact_dyadic_precision,
    make_circuit,
    normalize_circuit,
    snap_arbitrary_angles,
)
from .cache import cache_stats, clear_caches
from ._engine_impl import (
    analyze_amplitudes,
    analyze_circuit,
    compute_amplitude,
    compute_amplitudes,
    compute_amplitude_scaled,
    compute_circuit_amplitude,
    compute_circuit_amplitude_scaled,
)
from .phase_function import CubicFunction, PhaseFunction
from .scaling import ScaledAmplitude
from .state import SolverConfig

__all__ = [
    "CircuitSpec",
    "CubicFunction",
    "DoubledFactorProblem",
    "DoubledSumResult",
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
    "cache_stats",
    "clear_caches",
    "compute_amplitude",
    "compute_amplitudes",
    "compute_amplitude_scaled",
    "compute_circuit_amplitude",
    "compute_circuit_amplitude_scaled",
    "compute_circuit_probability_doubled",
    "compute_circuit_pauli_expectation_probabilities_doubled",
    "compute_circuit_pauli_expectations",
    "from_qiskit",
    "lift_exact_dyadic_precision",
    "make_circuit",
    "normalize_circuit",
    "snap_arbitrary_angles",
    "reduce_and_sum",
    "sum_doubled_phase",
    "sum_doubled_factor_problem",
]


_LAZY_EXPORTS = {
    "DoubledFactorProblem": (".doubled", "DoubledFactorProblem"),
    "DoubledSumResult": (".doubled", "DoubledSumResult"),
    "SchurState": (".state", "SchurState"),
    "build_state": (".state", "build_state"),
    "compute_circuit_probability_doubled": (".doubled", "compute_circuit_probability_doubled"),
    "compute_circuit_pauli_expectation_probabilities_doubled": (
        ".doubled",
        "compute_circuit_pauli_expectation_probabilities_doubled",
    ),
    "compute_circuit_pauli_expectations": (".pauli", "compute_circuit_pauli_expectations"),
    "from_qiskit": (".circuits", "from_qiskit"),
    "reduce_and_sum": (".reduction", "reduce_and_sum"),
    "sum_doubled_phase": (".doubled", "sum_doubled_phase"),
    "sum_doubled_factor_problem": (".doubled", "sum_doubled_factor_problem"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is not None:
        module_name, attr_name = target
        return getattr(import_module(module_name, __name__), attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
