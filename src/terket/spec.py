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

def snap_arbitrary_angles(
    circuit: CircuitSpec,
    *,
    max_level: int = 3,
    max_error: float | None = None,
    max_total_error: float | None = None,
) -> CircuitSpec:
    """Snap arbitrary phase gates to nearest dyadic roots of unity."""
    from .angle_snapping import snap_arbitrary_angles as _snap_arbitrary_angles

    return _snap_arbitrary_angles(
        circuit,
        max_level=max_level,
        max_error=max_error,
        max_total_error=max_total_error,
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
    "snap_arbitrary_angles",
]
