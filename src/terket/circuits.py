"""Facade for circuit normalization and interop in TerKet."""

from __future__ import annotations

from .interop.angles import (
    _dyadic_phase_from_qiskit_angle,
    _dyadic_phase_to_angle,
    dyadic_snap,
)
from .interop.qasm2 import parse_openqasm2
from .interop.qiskit_export import to_qiskit
from .interop.qiskit_import import from_qiskit
from .interop.rewrite import _rewrite_gate_sequence
from .spec import (
    CircuitSpec,
    Gate,
    SUPPORTED_GATES,
    _circuit_global_phase_radians,
    bits_to_big_endian_string,
    bits_to_index,
    bits_to_little_endian_string,
    iter_bitstrings,
    lift_exact_dyadic_precision,
    make_circuit,
    normalize_circuit,
    snap_arbitrary_angles,
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
    "snap_arbitrary_angles",
    "parse_openqasm2",
    "to_qiskit",
]
