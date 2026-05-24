"""Compatibility facade for the TerKet strong-simulation engine."""

from __future__ import annotations

from ._engine_impl import (
    analyze_amplitudes,
    analyze_circuit,
    compute_amplitude,
    compute_amplitude_scaled,
    compute_amplitudes,
    compute_circuit_amplitude,
    compute_circuit_amplitude_scaled,
)
from .native import (
    _get_quimb_tensor_module,
    _load_schur_native_module,
    _quimb_import_reason,
    _schur_native,
)
from .pauli import compute_circuit_pauli_expectations
from .reduction import reduce_and_sum
from .scaling import ScaledAmplitude
from .state import (
    BitSequence,
    CircuitInput,
    ExtendedReductionMode,
    ReducerInfo,
    ReductionInfo,
    SchurState,
    SolverConfig,
    build_state,
)

__all__ = [
    "BitSequence",
    "CircuitInput",
    "ExtendedReductionMode",
    "ReducerInfo",
    "ReductionInfo",
    "ScaledAmplitude",
    "SchurState",
    "SolverConfig",
    "_get_quimb_tensor_module",
    "_load_schur_native_module",
    "_quimb_import_reason",
    "_schur_native",
    "analyze_amplitudes",
    "analyze_circuit",
    "build_state",
    "compute_amplitude",
    "compute_amplitude_scaled",
    "compute_amplitudes",
    "compute_circuit_amplitude",
    "compute_circuit_amplitude_scaled",
    "compute_circuit_pauli_expectations",
    "reduce_and_sum",
]
