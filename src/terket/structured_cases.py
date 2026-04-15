"""Structured exact-strong showcase families shipped with TerKet."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Callable, Sequence

from qiskit import QuantumCircuit, transpile

from .circuits import from_qiskit


SUPPORTED_BASIS = ["h", "sx", "x", "rz", "cx", "cz"]


@dataclass(frozen=True)
class StructuredQuery:
    circuit: object
    input_bits: tuple[int, ...]
    output_bits: tuple[int, ...]
    wrong_output_bits: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuredCase:
    name: str
    family: str
    size: int
    build_query_fn: Callable[[int], StructuredQuery]

    def build_query(self) -> StructuredQuery:
        return self.build_query_fn(self.size)


def transpile_to_supported_basis(qc: QuantumCircuit, optimization_level: int = 2):
    transpiled = transpile(qc, basis_gates=SUPPORTED_BASIS, optimization_level=optimization_level)
    return from_qiskit(transpiled, rz_compile_mode="dyadic")


def _disjoint_triples(width: int) -> tuple[tuple[int, int, int], ...]:
    triples = []
    for start in range(0, width - 2, 3):
        triples.append((start, start + 1, start + 2))
    return tuple(triples)


def _apply_ccz(qc: QuantumCircuit, a: int, b: int, c: int) -> None:
    qc.h(c)
    qc.ccx(a, b, c)
    qc.h(c)


def _apply_mm_cross_terms(qc: QuantumCircuit, half_qubits: int) -> None:
    for left in range(half_qubits):
        qc.cz(left, half_qubits + left)


def _apply_cubic_terms(qc: QuantumCircuit, qubits: Sequence[int]) -> None:
    for a, b, c in _disjoint_triples(len(qubits)):
        _apply_ccz(qc, qubits[a], qubits[b], qubits[c])


def build_mm_hidden_shift_logical(half_qubits: int, shift_bits: Sequence[int]) -> QuantumCircuit:
    if half_qubits < 3:
        raise ValueError("half_qubits must be at least 3.")
    if len(shift_bits) != half_qubits:
        raise ValueError("shift_bits must match half_qubits.")

    total_qubits = 2 * half_qubits
    left = tuple(range(half_qubits))
    right = tuple(range(half_qubits, total_qubits))
    qc = QuantumCircuit(total_qubits, name=f"mm_hidden_shift_{total_qubits}")

    qc.h(range(total_qubits))
    _apply_mm_cross_terms(qc, half_qubits)
    _apply_cubic_terms(qc, right)
    qc.h(range(total_qubits))

    for bit, qubit in zip(shift_bits, right):
        if int(bit) & 1:
            qc.x(qubit)
    _apply_mm_cross_terms(qc, half_qubits)
    _apply_cubic_terms(qc, left)
    for bit, qubit in zip(shift_bits, right):
        if int(bit) & 1:
            qc.x(qubit)

    qc.h(range(total_qubits))
    return qc


def build_mm_hidden_shift_supported(
    half_qubits: int,
    shift_bits: Sequence[int],
    *,
    optimization_level: int = 2,
):
    logical = build_mm_hidden_shift_logical(half_qubits, shift_bits)
    return transpile_to_supported_basis(logical, optimization_level=optimization_level)


def build_mm_hidden_shift_query(half_qubits: int, *, shift_seed: int = 7) -> StructuredQuery:
    rng = random.Random(shift_seed + half_qubits)
    shift_bits = tuple(rng.randrange(2) for _ in range(half_qubits))
    target_bits = tuple(int(bit) for bit in shift_bits) + (0,) * half_qubits
    wrong_bits = list(target_bits)
    wrong_bits[0] ^= 1
    circuit = build_mm_hidden_shift_supported(half_qubits, shift_bits, optimization_level=2)
    zero_bits = (0,) * circuit.n_qubits
    return StructuredQuery(
        circuit=circuit,
        input_bits=zero_bits,
        output_bits=target_bits,
        wrong_output_bits=tuple(wrong_bits),
        metadata={"shift_bits": shift_bits},
    )


def _build_mm_hidden_shift_case(half_qubits: int) -> StructuredQuery:
    return build_mm_hidden_shift_query(half_qubits)


CASES: dict[str, StructuredCase] = {
    "mm_hidden_shift24": StructuredCase("mm_hidden_shift24", "mm_hidden_shift", 24, _build_mm_hidden_shift_case),
    "mm_hidden_shift192": StructuredCase(
        "mm_hidden_shift192",
        "mm_hidden_shift",
        192,
        _build_mm_hidden_shift_case,
    ),
    "mm_hidden_shift384": StructuredCase(
        "mm_hidden_shift384",
        "mm_hidden_shift",
        384,
        _build_mm_hidden_shift_case,
    ),
    "mm_hidden_shift1024": StructuredCase(
        "mm_hidden_shift1024",
        "mm_hidden_shift",
        1024,
        _build_mm_hidden_shift_case,
    ),
}

SUITES: dict[str, tuple[str, ...]] = {
    "smoke": ("mm_hidden_shift24",),
    "large": ("mm_hidden_shift192", "mm_hidden_shift384"),
    "xlarge": ("mm_hidden_shift1024",),
}


def get_case(name: str) -> StructuredCase:
    try:
        return CASES[name]
    except KeyError as exc:
        known = ", ".join(sorted(CASES))
        raise KeyError(f"Unknown structured showcase case {name!r}. Known cases: {known}") from exc


def resolve_cases(case_names: Sequence[str] | None = None, *, suite: str = "large") -> list[StructuredCase]:
    if case_names:
        return [get_case(name) for name in case_names]
    try:
        suite_case_names = SUITES[suite]
    except KeyError as exc:
        known = ", ".join(sorted(SUITES))
        raise KeyError(f"Unknown structured showcase suite {suite!r}. Known suites: {known}") from exc
    return [CASES[name] for name in suite_case_names]


__all__ = [
    "CASES",
    "SUPPORTED_BASIS",
    "SUITES",
    "StructuredCase",
    "StructuredQuery",
    "build_mm_hidden_shift_logical",
    "build_mm_hidden_shift_query",
    "build_mm_hidden_shift_supported",
    "get_case",
    "resolve_cases",
    "transpile_to_supported_basis",
]
