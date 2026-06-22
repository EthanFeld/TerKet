"""Recovered pauli support helpers."""

from __future__ import annotations

import cmath
import math
import os
from typing import Sequence

from ._engine_runtime_core import _configure_extracted_module

_LOCAL_NAMES = {
    '_exact_pauli_expbox_dyadic',
    '_normalize_pauli_expbox_terms',
    'apply_pauli_expbox_to_state',
    '_pauli_expbox_dyadic_snap_level',
}
_LOCAL_IMPLS = {}
_configure_extracted_module(
    globals(),
    local_names=_LOCAL_NAMES,
    local_impls=_LOCAL_IMPLS,
    respect_mock_wraps=True,
)


def _normalize_pauli_expbox_terms(
    paulis: Sequence[str],
    qubits: Sequence[int],
) -> tuple[tuple[str, int], ...]:
    if len(paulis) != len(qubits):
        raise ValueError("PauliExpBox paulis and qubits must have equal length.")

    active: list[tuple[str, int]] = []
    for pauli, qubit in zip(paulis, qubits):
        pauli_char = str(pauli).upper()
        if pauli_char == "I":
            continue
        if pauli_char not in {"X", "Y", "Z"}:
            raise ValueError(f"Unsupported PauliExpBox Pauli {pauli!r}.")
        active.append((pauli_char, int(qubit)))
    return tuple(active)

def _exact_pauli_expbox_dyadic(angle: float, *, max_level: int = 20) -> tuple[int, int, float] | None:
    """Return smallest exact dyadic level/coeff/snapped angle for one rotation."""
    for level in range(1, max_level + 1):
        modulus = 1 << level
        coeff = int(round(angle * modulus / (2.0 * math.pi)))
        snapped = 2.0 * math.pi * coeff / modulus
        if math.isclose(angle, snapped, rel_tol=0.0, abs_tol=1e-12):
            return level, coeff % modulus, snapped
    return None

def apply_pauli_expbox_to_state(
    state: "SchurState",
    paulis: Sequence[str],
    qubits: Sequence[int],
    angle: float,
) -> None:
    """Apply one PauliExpBox to a SchurState without materializing a CNOT ladder."""
    angle_value = float(angle)
    if not math.isfinite(angle_value):
        raise ValueError(f"pauli_expbox angle must be finite, received {angle!r}.")

    active = _normalize_pauli_expbox_terms(paulis, qubits)
    if not active:
        state.scalar *= cmath.exp(-0.5j * angle_value)
        return

    if math.isclose(math.remainder(angle_value, 2.0 * math.pi), 0.0, rel_tol=0.0, abs_tol=1e-15):
        state.scalar *= cmath.exp(-0.5j * angle_value)
        return

    for pauli_char, qubit in active:
        if pauli_char == "X":
            state.h(qubit)
        elif pauli_char == "Y":
            state.sdg(qubit)
            state.h(qubit)

    row_mask = 0
    offset = 0
    for _pauli_char, qubit in active:
        row_mask ^= state.eps[qubit]
        offset ^= state.eps0[qubit] & 1
    state.scalar *= cmath.exp(-0.5j * angle_value)
    snap_level = _pauli_expbox_dyadic_snap_level()
    exact_dyadic = _exact_pauli_expbox_dyadic(angle_value) if snap_level is None else None
    if snap_level is not None or exact_dyadic is not None:
        if exact_dyadic is not None:
            snap_level, coeff, snapped = exact_dyadic
        else:
            assert snap_level is not None
            modulus = 1 << snap_level
            coeff = int(round(angle_value * modulus / (2.0 * math.pi))) % modulus
            snapped = 2.0 * math.pi * coeff / modulus
        modulus = 1 << snap_level
        state.scalar *= cmath.exp(0.5j * (angle_value - snapped))
        if row_mask and coeff:
            state._ensure_phase_precision(snap_level)
            _apply_diag_phase_in_place(
                state.q,
                row_mask,
                offset,
                state._lift_linear_coeff(coeff, snap_level),
            )
            state._invalidate_classification_cache()
        elif offset and coeff:
            state.scalar *= cmath.exp(1j * snapped)
    else:
        if row_mask:
            state._arbitrary_phases.append(_ArbitraryPhaseTerm(row_mask, offset, angle_value))
            state._update_reference_mask(0, row_mask)
        elif offset:
            state.scalar *= cmath.exp(1j * angle_value)

    for pauli_char, qubit in reversed(active):
        if pauli_char == "X":
            state.h(qubit)
        elif pauli_char == "Y":
            state.h(qubit)
            state.s(qubit)

def _pauli_expbox_dyadic_snap_level() -> int | None:
    raw = os.environ.get("TERKET_PAULI_EXPBOX_DYADIC_LEVEL")
    if raw is None or raw == "":
        return None
    try:
        level = int(raw)
    except ValueError:
        return None
    return max(1, level)

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
