"""Circuit spec and bit helper facade."""

from __future__ import annotations

from .circuit_spec import (
    CircuitSpec,
    Gate,
    SUPPORTED_GATES,
    _circuit_global_phase_radians,
    bits_to_big_endian_string,
    bits_to_index,
    bits_to_little_endian_string,
    big_endian_string_to_bits,
    iter_bitstrings,
    lift_exact_dyadic_precision,
    little_endian_string_to_bits,
    make_circuit,
    normalize_circuit,
)

__all__ = [
    "CircuitSpec",
    "Gate",
    "SUPPORTED_GATES",
    "_circuit_global_phase_radians",
    "big_endian_string_to_bits",
    "bits_to_big_endian_string",
    "bits_to_index",
    "bits_to_little_endian_string",
    "iter_bitstrings",
    "lift_exact_dyadic_precision",
    "little_endian_string_to_bits",
    "make_circuit",
    "normalize_circuit",
]
