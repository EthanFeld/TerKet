"""Explicit approximate lowering of arbitrary phases to dyadic roots."""

from __future__ import annotations

from .circuit_spec import (
    CircuitSpec,
    Gate,
    _APPROXIMATION_BASIS_SIZE_METADATA_KEY,
    _APPROXIMATION_MAX_ANGLE_ERROR_METADATA_KEY,
    _APPROXIMATION_MODE_METADATA_KEY,
    _APPROXIMATION_PHASE_COUNT_METADATA_KEY,
    _APPROXIMATION_TOLERANCE_METADATA_KEY,
    _APPROXIMATION_TOTAL_ANGLE_ERROR_METADATA_KEY,
    _EXACT_DYADIC_TOLERANCE,
)
from .interop.angles import _dyadic_phase_to_angle, dyadic_snap

_SNAP_DYADIC_MAX_LEVEL_METADATA_KEY = "snap_dyadic_max_level"


def snap_arbitrary_angles(
    circuit: CircuitSpec,
    *,
    max_level: int = 3,
    max_error: float | None = None,
    max_total_error: float | None = None,
) -> CircuitSpec:
    """Snap arbitrary phase gates to nearest dyadic roots of unity."""
    if not isinstance(circuit, CircuitSpec):
        raise TypeError(f"Expected CircuitSpec, received {type(circuit)!r}.")
    max_level = int(max_level)
    if max_level < 1:
        raise ValueError(f"max_level must be positive, received {max_level}.")
    if max_error is not None and float(max_error) < 0.0:
        raise ValueError(f"max_error must be non-negative, received {max_error}.")
    if max_total_error is not None and float(max_total_error) < 0.0:
        raise ValueError(f"max_total_error must be non-negative, received {max_total_error}.")

    snapped_gates: list[Gate] = []
    errors: list[float] = []
    snapped_angles: set[float] = set()
    changed = False
    for gate in circuit.gates:
        if gate[0] not in {"rz_arbitrary", "pauli_expbox"}:
            snapped_gates.append(gate)
            continue
        angle = float(gate[2] if gate[0] == "rz_arbitrary" else gate[3])
        coeff, level, error = dyadic_snap(angle, max_level=max_level, nearest=True)
        if max_error is not None and error > float(max_error):
            raise ValueError(
                f"Nearest level-{max_level} dyadic angle for {angle!r} exceeds "
                f"max_error {float(max_error):.3e}: {error:.3e}."
            )
        snapped_angle = _dyadic_phase_to_angle(coeff, level)
        if gate[0] == "rz_arbitrary":
            snapped_gates.append(("rz_dyadic", int(gate[1]), int(coeff), int(level)))
            changed = True
        else:
            snapped_gate = ("pauli_expbox", gate[1], gate[2], snapped_angle)
            snapped_gates.append(snapped_gate)
            changed |= snapped_gate != gate
        if error > _EXACT_DYADIC_TOLERANCE:
            errors.append(float(error))
            snapped_angles.add(snapped_angle)

    if not changed:
        return circuit
    total_error = sum(errors)
    if max_total_error is not None and total_error > float(max_total_error):
        raise ValueError(
            f"Total dyadic snap error {total_error:.3e} exceeds "
            f"max_total_error {float(max_total_error):.3e}."
        )
    metadata = dict(circuit.metadata)
    metadata[_SNAP_DYADIC_MAX_LEVEL_METADATA_KEY] = max_level
    if errors:
        metadata[_APPROXIMATION_MODE_METADATA_KEY] = "snap_dyadic"
        metadata[_APPROXIMATION_BASIS_SIZE_METADATA_KEY] = len(snapped_angles)
        metadata[_APPROXIMATION_PHASE_COUNT_METADATA_KEY] = len(errors)
        metadata[_APPROXIMATION_TOTAL_ANGLE_ERROR_METADATA_KEY] = total_error
        metadata[_APPROXIMATION_MAX_ANGLE_ERROR_METADATA_KEY] = max(errors)
        if max_error is not None:
            metadata[_APPROXIMATION_TOLERANCE_METADATA_KEY] = float(max_error)
    return CircuitSpec(
        n_qubits=circuit.n_qubits,
        gates=tuple(snapped_gates),
        name=circuit.name,
        metadata=metadata,
    )
