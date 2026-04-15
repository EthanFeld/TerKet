"""Facade for circuit normalization and interop in TerKet."""

from __future__ import annotations

from .circuit_io import from_qiskit, parse_openqasm2
from .circuit_spec import (
    CircuitSpec,
    Gate,
    SUPPORTED_GATES,
    _circuit_global_phase_radians,
    _dyadic_phase_from_qiskit_angle,
    _dyadic_phase_to_angle,
    _rewrite_gate_sequence,
    bits_to_big_endian_string,
    bits_to_index,
    bits_to_little_endian_string,
    dyadic_snap,
    iter_bitstrings,
    lift_exact_dyadic_precision,
    make_circuit,
    normalize_circuit,
)

__all__ = [
    "CircuitSpec",
    "Gate",
    "SUPPORTED_GATES",
    "_circuit_global_phase_radians",
    "_dyadic_phase_from_qiskit_angle",
    "_dyadic_phase_to_angle",
    "_rewrite_gate_sequence",
    "bits_to_big_endian_string",
    "bits_to_index",
    "bits_to_little_endian_string",
    "dyadic_snap",
    "from_qiskit",
    "iter_bitstrings",
    "lift_exact_dyadic_precision",
    "make_circuit",
    "normalize_circuit",
    "parse_openqasm2",
]
