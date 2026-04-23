"""Standalone exact-strong benchmark families for quimb-vs-TerKet runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Sequence

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DraperQFTAdder

from ..circuits import from_qiskit, make_circuit


SUPPORTED_BASIS = ["h", "sx", "x", "rz", "cx", "cz"]


@dataclass(frozen=True)
class AmplitudeQuery:
    circuit: object
    input_bits: tuple[int, ...]
    output_bits: tuple[int, ...]
    wrong_output_bits: tuple[int, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    family: str
    size: int
    build_query_fn: Callable[[int], AmplitudeQuery]

    def build_query(self) -> AmplitudeQuery:
        return self.build_query_fn(self.size)


def transpile_to_supported_basis(qc: QuantumCircuit, optimization_level: int = 0):
    transpiled = transpile(qc, basis_gates=SUPPORTED_BASIS, optimization_level=optimization_level)
    return from_qiskit(transpiled, rz_compile_mode="dyadic")


def _append_controlled_s(qc: QuantumCircuit, control: int, target: int):
    qc.t(control)
    qc.t(target)
    qc.cx(control, target)
    qc.tdg(target)
    qc.cx(control, target)


def build_qaoa_ring_logical(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name=f"qaoa_ring_p1_{n_qubits}")
    for qubit in range(n_qubits):
        qc.h(qubit)
    for qubit in range(n_qubits):
        qc.rzz(math.pi / 4.0, qubit, (qubit + 1) % n_qubits)
    for qubit in range(n_qubits):
        qc.rx(math.pi / 4.0, qubit)
    return qc


def build_qaoa_ring(n_qubits: int):
    return transpile_to_supported_basis(build_qaoa_ring_logical(n_qubits), optimization_level=0)


def build_toffoli_chain_logical(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name=f"toffoli_chain_{n_qubits}")
    for idx in range(n_qubits - 2):
        qc.ccx(idx, idx + 1, idx + 2)
    return qc


def build_toffoli_ladder(n_qubits: int):
    return transpile_to_supported_basis(build_toffoli_chain_logical(n_qubits), optimization_level=0)


def build_approximate_qft_logical(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name=f"approximate_qft_{n_qubits}")
    for target in range(n_qubits):
        if target + 1 < n_qubits:
            _append_controlled_s(qc, target + 1, target)
        qc.h(target)
    return qc


def build_approximate_qft(n_qubits: int):
    gates = []
    for target in range(n_qubits):
        if target + 1 < n_qubits:
            control = target + 1
            gates.extend(
                (
                    ("t", control),
                    ("t", target),
                    ("cnot", control, target),
                    ("tdg", target),
                    ("cnot", control, target),
                )
            )
        gates.append(("h", target))
    return make_circuit(n_qubits, gates, name=f"approximate_qft_{n_qubits}")


def _grover_logical_qubits(total_qubits: int) -> int:
    if total_qubits < 2:
        raise ValueError("grover_iteration requires at least 2 qubits.")
    return max(2, (total_qubits + 2) // 2)


def _append_multi_controlled_x_ladder(qc: QuantumCircuit, controls: list[int], target: int, ancillas: list[int]):
    n_controls = len(controls)
    if n_controls == 0:
        qc.x(target)
        return
    if n_controls == 1:
        qc.cx(controls[0], target)
        return
    if n_controls == 2:
        qc.ccx(controls[0], controls[1], target)
        return

    required_ancillas = n_controls - 2
    if len(ancillas) < required_ancillas:
        raise ValueError(
            f"Need at least {required_ancillas} clean ancillas for {n_controls} controls, got {len(ancillas)}."
        )

    qc.ccx(controls[0], controls[1], ancillas[0])
    for idx in range(2, n_controls - 1):
        qc.ccx(controls[idx], ancillas[idx - 2], ancillas[idx - 1])
    qc.ccx(controls[-1], ancillas[n_controls - 3], target)
    for idx in range(n_controls - 2, 1, -1):
        qc.ccx(controls[idx], ancillas[idx - 2], ancillas[idx - 1])
    qc.ccx(controls[0], controls[1], ancillas[0])


def _append_multi_controlled_z_ladder(qc: QuantumCircuit, qubits: list[int], ancillas: list[int]):
    if len(qubits) == 1:
        qc.z(qubits[0])
        return
    if len(qubits) == 2:
        qc.cz(qubits[0], qubits[1])
        return

    target = qubits[-1]
    qc.h(target)
    _append_multi_controlled_x_ladder(qc, qubits[:-1], target, ancillas)
    qc.h(target)


def build_grover_iteration_logical(n_qubits: int) -> QuantumCircuit:
    logical_qubits = _grover_logical_qubits(n_qubits)
    search_qubits = list(range(logical_qubits))
    ancillas = list(range(logical_qubits, n_qubits))

    qc = QuantumCircuit(n_qubits, name=f"grover_iteration_{n_qubits}")
    _append_multi_controlled_z_ladder(qc, search_qubits, ancillas)
    for qubit in search_qubits:
        qc.h(qubit)
        qc.x(qubit)
    _append_multi_controlled_z_ladder(qc, search_qubits, ancillas)
    for qubit in search_qubits:
        qc.x(qubit)
        qc.h(qubit)
    return qc


def build_grover_iteration(n_qubits: int):
    return transpile_to_supported_basis(build_grover_iteration_logical(n_qubits), optimization_level=0)


def build_repetition_magic_round(distance: int):
    data_qubits = distance
    ancillas = max(1, distance - 1)
    qc = QuantumCircuit(data_qubits + ancillas, name=f"repetition_magic_round_{distance}")
    for qubit in range(data_qubits):
        qc.h(qubit)
    for ancilla in range(ancillas):
        ancilla_idx = data_qubits + ancilla
        qc.cx(ancilla, ancilla_idx)
        qc.cx((ancilla + 1) % data_qubits, ancilla_idx)
        qc.t(ancilla_idx)
        qc.cx(ancilla, ancilla_idx)
        qc.cx((ancilla + 1) % data_qubits, ancilla_idx)
    return from_qiskit(qc, rz_compile_mode="dyadic")


def _bits_le_from_int(value: int, width: int) -> tuple[int, ...]:
    return tuple((int(value) >> idx) & 1 for idx in range(width))


def _int_from_bits_le(bits: Sequence[int]) -> int:
    value = 0
    for idx, bit in enumerate(bits):
        value |= (int(bit) & 1) << idx
    return value


def _select_register_values(problem_size: int) -> tuple[int, int]:
    a_bits = [1 if ((3 * idx + 1) % 5) < 2 else 0 for idx in range(problem_size)]
    b_bits = [1 if ((5 * idx + 2) % 7) < 3 else 0 for idx in range(problem_size)]
    a = _int_from_bits_le(a_bits)
    b = _int_from_bits_le(b_bits)
    if a == 0:
        a = 1
    if b == 0:
        b = 1 if problem_size > 1 else 0
    return a, b


def draper_fixed_input_output(
    problem_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int, int, int]:
    modulus = 1 << problem_size
    a_value, b_value = _select_register_values(problem_size)
    sum_value = (a_value + b_value) % modulus

    a_bits = _bits_le_from_int(a_value, problem_size)
    b_bits = _bits_le_from_int(b_value, problem_size)
    sum_bits = _bits_le_from_int(sum_value, problem_size)

    input_bits = a_bits + b_bits
    target_bits = a_bits + sum_bits

    wrong_bits = list(target_bits)
    wrong_index = problem_size if problem_size < len(wrong_bits) else len(wrong_bits) - 1
    wrong_bits[wrong_index] ^= 1
    return input_bits, tuple(target_bits), tuple(wrong_bits), a_value, b_value, sum_value


def build_draper_qft_adder_supported(problem_size: int, *, optimization_level: int = 1):
    logical = DraperQFTAdder(problem_size, kind="fixed")
    lowered = transpile(logical, basis_gates=["h", "p", "cx"], optimization_level=optimization_level)
    return from_qiskit(lowered, rz_compile_mode="dyadic")


def _zero_query(builder: Callable[[int], object], n_qubits: int, *, metadata: dict[str, Any] | None = None) -> AmplitudeQuery:
    zero_bits = (0,) * n_qubits
    return AmplitudeQuery(
        circuit=builder(n_qubits),
        input_bits=zero_bits,
        output_bits=zero_bits,
        metadata={} if metadata is None else dict(metadata),
    )


def build_qaoa_query(n_qubits: int) -> AmplitudeQuery:
    return _zero_query(build_qaoa_ring, n_qubits)


def build_grover_query(n_qubits: int) -> AmplitudeQuery:
    return _zero_query(build_grover_iteration, n_qubits)


def build_approximate_qft_query(n_qubits: int) -> AmplitudeQuery:
    return _zero_query(build_approximate_qft, n_qubits, metadata={"expected_log2_abs": -0.5 * n_qubits})


def build_toffoli_ladder_query(n_qubits: int) -> AmplitudeQuery:
    return _zero_query(build_toffoli_ladder, n_qubits)


def build_qec_repetition_magic_round_query(distance: int) -> AmplitudeQuery:
    circuit = build_repetition_magic_round(distance)
    zero_bits = (0,) * circuit.n_qubits
    return AmplitudeQuery(circuit=circuit, input_bits=zero_bits, output_bits=zero_bits)


def build_draper_query(problem_size: int) -> AmplitudeQuery:
    circuit = build_draper_qft_adder_supported(problem_size, optimization_level=1)
    input_bits, target_bits, wrong_bits, a_value, b_value, sum_value = draper_fixed_input_output(problem_size)
    return AmplitudeQuery(
        circuit=circuit,
        input_bits=input_bits,
        output_bits=target_bits,
        wrong_output_bits=wrong_bits,
        metadata={"input_a": a_value, "input_b": b_value, "target_sum": sum_value},
    )


CASES: dict[str, BenchmarkCase] = {
    "grover16": BenchmarkCase("grover16", "grover_iteration", 16, build_grover_query),
    "grover48": BenchmarkCase("grover48", "grover_iteration", 48, build_grover_query),
    "qaoa16": BenchmarkCase("qaoa16", "qaoa_ring", 16, build_qaoa_query),
    "qaoa64": BenchmarkCase("qaoa64", "qaoa_ring", 64, build_qaoa_query),
    "approximate_qft32": BenchmarkCase("approximate_qft32", "approximate_qft", 32, build_approximate_qft_query),
    "approximate_qft4096": BenchmarkCase(
        "approximate_qft4096",
        "approximate_qft",
        4096,
        build_approximate_qft_query,
    ),
    "toffoli_ladder16": BenchmarkCase("toffoli_ladder16", "toffoli_ladder", 16, build_toffoli_ladder_query),
    "toffoli_ladder512": BenchmarkCase("toffoli_ladder512", "toffoli_ladder", 512, build_toffoli_ladder_query),
    "draper8": BenchmarkCase("draper8", "draper_qft_adder", 8, build_draper_query),
    "draper32": BenchmarkCase("draper32", "draper_qft_adder", 32, build_draper_query),
    "qec_repetition_magic_round8": BenchmarkCase(
        "qec_repetition_magic_round8",
        "qec_repetition_magic_round",
        8,
        build_qec_repetition_magic_round_query,
    ),
    "qec_repetition_magic_round64": BenchmarkCase(
        "qec_repetition_magic_round64",
        "qec_repetition_magic_round",
        64,
        build_qec_repetition_magic_round_query,
    ),
    "qec_repetition_magic_round128": BenchmarkCase(
        "qec_repetition_magic_round128",
        "qec_repetition_magic_round",
        128,
        build_qec_repetition_magic_round_query,
    ),
}

CASE_ALIASES = {
    "toffoli_chain16": "toffoli_ladder16",
    "toffoli_chain512": "toffoli_ladder512",
}

SUITES: dict[str, tuple[str, ...]] = {
    "hero": ("grover48", "qaoa64"),
    "smoke": (
        "grover16",
        "qaoa16",
        "approximate_qft32",
        "toffoli_ladder16",
        "draper8",
        "qec_repetition_magic_round8",
    ),
    "expanded": (
        "grover48",
        "qaoa64",
        "approximate_qft4096",
        "toffoli_ladder512",
        "draper32",
        "qec_repetition_magic_round64",
        "qec_repetition_magic_round128",
    ),
}


def get_case(name: str) -> BenchmarkCase:
    resolved_name = CASE_ALIASES.get(name, name)
    try:
        return CASES[resolved_name]
    except KeyError as exc:
        known = ", ".join(sorted({*CASES, *CASE_ALIASES}))
        raise KeyError(f"Unknown benchmark case {name!r}. Known cases: {known}") from exc


def resolve_cases(case_names: Sequence[str] | None = None, *, suite: str = "hero") -> list[BenchmarkCase]:
    if case_names:
        return [get_case(name) for name in case_names]
    try:
        suite_case_names = SUITES[suite]
    except KeyError as exc:
        known = ", ".join(sorted(SUITES))
        raise KeyError(f"Unknown benchmark suite {suite!r}. Known suites: {known}") from exc
    return [CASES[name] for name in suite_case_names]


__all__ = [
    "CASES",
    "CASE_ALIASES",
    "SUPPORTED_BASIS",
    "SUITES",
    "AmplitudeQuery",
    "BenchmarkCase",
    "_append_multi_controlled_z_ladder",
    "_grover_logical_qubits",
    "build_approximate_qft",
    "build_approximate_qft_logical",
    "build_draper_query",
    "build_draper_qft_adder_supported",
    "build_grover_iteration",
    "build_grover_iteration_logical",
    "build_qaoa_query",
    "build_qaoa_ring",
    "build_qaoa_ring_logical",
    "build_repetition_magic_round",
    "build_toffoli_ladder",
    "build_toffoli_chain_logical",
    "draper_fixed_input_output",
    "get_case",
    "resolve_cases",
    "transpile_to_supported_basis",
]
