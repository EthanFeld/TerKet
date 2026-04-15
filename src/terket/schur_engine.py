"""Compatibility facade for the TerKet strong-simulation engine."""

from __future__ import annotations

from .backends import (
    _get_cupy_module,
    _get_quimb_tensor_module,
    _load_schur_native_module,
    _quimb_import_reason,
    _schur_native,
)
from .engine import (
    BitSequence,
    CircuitInput,
    ExtendedReductionMode,
    ReducerInfo,
    ReductionInfo,
    ScaledAmplitude,
    SchurState,
    analyze_amplitudes,
    analyze_circuit,
    build_state,
    compute_amplitude,
    compute_amplitude_scaled,
    compute_amplitudes,
    compute_circuit_amplitude,
    compute_circuit_amplitude_scaled,
    reduce_and_sum,
)

__all__ = [
    "BitSequence",
    "CircuitInput",
    "ExtendedReductionMode",
    "ReducerInfo",
    "ReductionInfo",
    "ScaledAmplitude",
    "SchurState",
    "_get_cupy_module",
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
    "reduce_and_sum",
]
