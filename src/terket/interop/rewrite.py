"""Gate normalization and local rewrite helpers."""

from __future__ import annotations

import math
from typing import Sequence

from ..circuit_spec import (
    Gate,
    _TEMP_PHASE_GATE,
    _coerce_finite_radians,
    _normalize_global_phase_radians,
)

_SELF_INVERSE_GATES = {"h", "x", "cnot", "cz"}

_NAMED_DYADIC_GATES = {
    (1, 1): ("z",),
    (2, 1): ("s",),
    (2, 3): ("sdg",),
    (5, 1): ("rz_pi_16",),
    (5, 31): ("rz_pi_16_dg",),
    (6, 1): ("rz_pi_32",),
    (6, 63): ("rz_pi_32_dg",),
}

_LEVEL3_PHASE_SEQUENCES = {
    1: ("t",),
    2: ("s",),
    3: ("s", "t"),
    4: ("z",),
    5: ("z", "t"),
    6: ("sdg",),
    7: ("tdg",),
}

def _gate_qubits(gate: Gate) -> tuple[int, ...]:
    name = gate[0]
    if name in {"rz_dyadic", "rz_arbitrary"}:
        return (gate[1],)
    if name == "rzz_dyadic":
        return int(gate[1]), int(gate[2])
    if name == "pauli_expbox":
        return tuple(int(qubit) for qubit in gate[2])
    return tuple(int(qubit) for qubit in gate[1:])

def _diagonal_phase_spec(gate: Gate) -> tuple[int, int, int] | None:
    name = gate[0]
    if name == "t":
        return gate[1], 1, 3
    if name == "tdg":
        return gate[1], 7, 3
    if name == "s":
        return gate[1], 2, 3
    if name == "sdg":
        return gate[1], 6, 3
    if name == "z":
        return gate[1], 4, 3
    if name == "rz_pi_16":
        return gate[1], 1, 5
    if name == "rz_pi_16_dg":
        return gate[1], 31, 5
    if name == "rz_pi_32":
        return gate[1], 1, 6
    if name == "rz_pi_32_dg":
        return gate[1], 63, 6
    if name == "rz_dyadic":
        return gate[1], int(gate[2]), int(gate[3])
    return None

def _normalize_phase_angle(angle: float) -> float:
    return _normalize_global_phase_radians(angle)

def _phase_angle_from_gate(gate: Gate) -> float | None:
    from .angles import _dyadic_phase_to_angle

    name = gate[0]
    if name == _TEMP_PHASE_GATE:
        return _normalize_phase_angle(_coerce_finite_radians(gate[2], source="Unsupported phase angle"))
    if name == "t":
        return math.pi / 4.0
    if name == "tdg":
        return -math.pi / 4.0
    if name == "s":
        return math.pi / 2.0
    if name == "sdg":
        return -math.pi / 2.0
    if name == "z":
        return math.pi
    if name == "rz_dyadic":
        return _dyadic_phase_to_angle(gate[2], gate[3])
    if name == "rz_pi_16":
        return math.pi / 16.0
    if name == "rz_pi_16_dg":
        return -math.pi / 16.0
    if name == "rz_pi_32":
        return math.pi / 32.0
    if name == "rz_pi_32_dg":
        return -math.pi / 32.0
    return None

def _diagonal_phase_angle(gate: Gate) -> tuple[int, float] | None:
    name = gate[0]
    phase = _phase_angle_from_gate(gate)
    if phase is not None:
        return int(gate[1]), _normalize_phase_angle(phase)
    if name == "rz_arbitrary":
        return int(gate[1]), _normalize_phase_angle(
            _coerce_finite_radians(gate[2], source="Unsupported arbitrary phase angle")
        )
    return None

def _emit_exact_phase_gate(qubit: int, angle: float) -> tuple[Gate, ...]:
    from .angles import _exact_phase_gate_from_angle

    normalized = _normalize_phase_angle(angle)
    if normalized == 0.0:
        return ()
    gate, _exact_angle = _exact_phase_gate_from_angle(
        normalized,
        qubit,
        source=f"Unsupported exact phase angle {normalized!r}",
    )
    return () if gate is None else (gate,)

def _gate_can_slide_left_past(previous: Gate, gate: Gate) -> bool:
    previous_qubits = set(_gate_qubits(previous))
    gate_qubits = set(_gate_qubits(gate))
    if gate_qubits.isdisjoint(previous_qubits):
        return True

    diagonal = _diagonal_phase_angle(gate)
    if diagonal is not None:
        qubit = diagonal[0]
        if previous[0] == "cz" and qubit in previous_qubits:
            return True
        if previous[0] == "cnot" and qubit == int(previous[1]):
            return True

    if gate[0] == "x" and previous[0] == "cnot" and int(gate[1]) == int(previous[2]):
        return True

    return False

def _simplify_local_gate_window(rewritten: list[Gate], start: int) -> None:
    idx = max(0, int(start) - 2)
    while idx < len(rewritten):
        if idx + 1 < len(rewritten):
            left = rewritten[idx]
            right = rewritten[idx + 1]
            left_rzz = _rzz_dyadic_spec(left)
            right_rzz = _rzz_dyadic_spec(right)
            if left_rzz is not None and right_rzz is not None:
                if frozenset((left_rzz[0], left_rzz[1])) == frozenset((right_rzz[0], right_rzz[1])):
                    level = max(left_rzz[3], right_rzz[3])
                    coeff = (left_rzz[2] << (level - left_rzz[3])) + (right_rzz[2] << (level - right_rzz[3]))
                    rewritten[idx:idx + 2] = list(_emit_rzz_dyadic_gate(left_rzz[0], left_rzz[1], coeff, level))
                    idx = max(0, idx - 2)
                    continue
            if left[0] == "sx" and right == left:
                rewritten[idx:idx + 2] = [("x", int(left[1]))]
                idx = max(0, idx - 2)
                continue
            if left[0] == "sxdg" and right == left:
                rewritten[idx:idx + 2] = [("x", int(left[1]))]
                idx = max(0, idx - 2)
                continue
            if left[0] == "sx" and right[0] == "sxdg" and int(left[1]) == int(right[1]):
                rewritten[idx:idx + 2] = []
                idx = max(0, idx - 2)
                continue
            if left[0] == "sxdg" and right[0] == "sx" and int(left[1]) == int(right[1]):
                rewritten[idx:idx + 2] = []
                idx = max(0, idx - 2)
                continue

        if idx + 2 < len(rewritten):
            first = rewritten[idx]
            second = rewritten[idx + 1]
            third = rewritten[idx + 2]
            second_phase = _diagonal_phase_spec(second)
            if first == third and first[0] == "cnot" and second_phase is not None and int(first[2]) == int(second_phase[0]):
                rewritten[idx:idx + 3] = list(_emit_rzz_dyadic_gate(int(first[1]), int(first[2]), second_phase[1], second_phase[2]))
                idx = max(0, idx - 2)
                continue
            if (
                first == third
                and first[0] == "h"
                and second[0] in {"x", "z", "s", "sdg"}
                and int(first[1]) == int(second[1])
            ):
                qubit = int(first[1])
                if second[0] == "z":
                    rewritten[idx:idx + 3] = [("x", qubit)]
                    idx = max(0, idx - 2)
                    continue
                if second[0] == "x":
                    rewritten[idx:idx + 3] = [("z", qubit)]
                    idx = max(0, idx - 2)
                    continue
                if second[0] == "s":
                    rewritten[idx:idx + 3] = [("sx", qubit)]
                    idx = max(0, idx - 2)
                    continue
                if second[0] == "sdg":
                    rewritten[idx:idx + 3] = [("sxdg", qubit)]
                    idx = max(0, idx - 2)
                    continue

        idx += 1

def _normalize_dyadic_phase(coeff: int, level: int) -> tuple[int, int]:
    level = int(level)
    if level < 1:
        raise ValueError(f"Dyadic precision level must be positive, received {level}.")

    modulus = 1 << level
    coeff = int(coeff) % modulus
    while level > 1 and coeff % 2 == 0:
        coeff //= 2
        level -= 1
        modulus >>= 1
    return coeff % modulus, level

def _combine_dyadic_phases(left: tuple[int, int, int], right: tuple[int, int, int]) -> tuple[int, int, int]:
    qubit = left[0]
    if qubit != right[0]:
        raise ValueError("Cannot combine diagonal phases on different qubits.")

    level = max(left[2], right[2])
    coeff = (left[1] << (level - left[2])) + (right[1] << (level - right[2]))
    coeff, level = _normalize_dyadic_phase(coeff, level)
    return qubit, coeff, level

def _rzz_dyadic_spec(gate: Gate) -> tuple[int, int, int, int] | None:
    if gate[0] != "rzz_dyadic":
        return None
    return int(gate[1]), int(gate[2]), int(gate[3]), int(gate[4])

def _emit_dyadic_phase_gate(qubit: int, coeff: int, level: int) -> tuple[Gate, ...]:
    coeff, level = _normalize_dyadic_phase(coeff, level)
    modulus = 1 << level
    coeff %= modulus
    if coeff == 0:
        return ()

    if level <= 3:
        sequence = _LEVEL3_PHASE_SEQUENCES.get(coeff << (3 - level))
        if sequence is not None:
            return tuple((name, qubit) for name in sequence)

    named = _NAMED_DYADIC_GATES.get((level, coeff))
    if named is not None:
        return ((named[0], qubit),)
    return (("rz_dyadic", qubit, coeff, level),)

def _emit_rzz_dyadic_gate(q0: int, q1: int, coeff: int, level: int) -> tuple[Gate, ...]:
    coeff, level = _normalize_dyadic_phase(coeff, level)
    modulus = 1 << level
    coeff %= modulus
    if coeff == 0:
        return ()
    return (("rzz_dyadic", int(q0), int(q1), coeff, level),)

def _rewrite_gate_sequence_local(gates: Sequence[Gate]) -> tuple[Gate, ...]:
    """Apply only linear-time local rewrites without commutation search."""
    rewritten: list[Gate] = []
    for raw_gate in gates:
        gate = _normalize_gate(raw_gate)
        if gate[0] in _SELF_INVERSE_GATES and rewritten and rewritten[-1] == gate:
            rewritten.pop()
            _simplify_local_gate_window(rewritten, len(rewritten) - 1)
            continue
        rewritten.append(gate)
        _simplify_local_gate_window(rewritten, len(rewritten) - 1)
    return tuple(rewritten)

def _rewrite_gate_sequence(gates: Sequence[Gate]) -> tuple[Gate, ...]:
    """Apply safe local rewrites before Schur-state construction."""
    rewritten: list[Gate] = []

    for raw_gate in gates:
        gate = _normalize_gate(raw_gate)
        insert_pos = len(rewritten)
        while insert_pos > 0 and _gate_can_slide_left_past(rewritten[insert_pos - 1], gate):
            insert_pos -= 1

        diagonal = _diagonal_phase_angle(gate)
        if diagonal is not None and insert_pos > 0:
            previous = _diagonal_phase_angle(rewritten[insert_pos - 1])
            if previous is not None and previous[0] == diagonal[0]:
                replacement = list(_emit_exact_phase_gate(diagonal[0], previous[1] + diagonal[1]))
                rewritten[insert_pos - 1:insert_pos] = replacement
                _simplify_local_gate_window(rewritten, insert_pos - 1)
                continue

        if gate[0] in _SELF_INVERSE_GATES and insert_pos > 0 and rewritten[insert_pos - 1] == gate:
            del rewritten[insert_pos - 1]
            _simplify_local_gate_window(rewritten, insert_pos - 1)
            continue

        rewritten.insert(insert_pos, gate)
        _simplify_local_gate_window(rewritten, insert_pos)

    return tuple(rewritten)

def _normalize_gate(gate: Gate) -> Gate:
    if not gate:
        raise ValueError("Empty gate tuple is not allowed.")
    name = str(gate[0]).lower()
    if name == "cx":
        name = "cnot"
    if name == "pauli_expbox":
        if len(gate) != 4:
            return (name, *gate[1:])
        paulis = tuple(str(pauli).upper() for pauli in gate[1])
        qubits = tuple(int(qubit) for qubit in gate[2])
        return (name, paulis, qubits, gate[3])
    return (name, *gate[1:])
