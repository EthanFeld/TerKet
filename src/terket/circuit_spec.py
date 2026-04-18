"""Circuit input helpers for the TerKet Clifford+T simulator."""

from __future__ import annotations

import ast
import cmath
from dataclasses import dataclass, field
from functools import lru_cache
import math
import numpy as np
from pathlib import Path
import re
from typing import Any, Iterable, Sequence


Gate = tuple[Any, ...]
SUPPORTED_GATES = {
    "h",
    "sx",
    "sxdg",
    "x",
    "t",
    "tdg",
    "s",
    "sdg",
    "z",
    "cnot",
    "cz",
    "rzz_dyadic",
    "rz_arbitrary",
    "rz_dyadic",
    "rz_pi_16",
    "rz_pi_16_dg",
    "rz_pi_32",
    "rz_pi_32_dg",
}
_QASM_GATE_MAP = {
    "cx": "cnot",
    "cnot": "cnot",
    "h": "h",
    "sx": "sx",
    "sxdg": "sxdg",
    "x": "x",
    "t": "t",
    "tdg": "tdg",
    "s": "s",
    "sdg": "sdg",
    "z": "z",
    "cz": "cz",
}
_QASM_QUBIT = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]")
_QASM_SUPPORTED_GATE_SET = "{h, sx, sxdg, x, t, tdg, s, sdg, z, cnot, cz, rz(theta)}"
_MAX_RZ_TOLERANCE = 1e-5
_EXACT_DYADIC_TOLERANCE = 1e-12
_EXACT_DYADIC_MAX_LEVEL = 20
_GLOBAL_PHASE_METADATA_KEY = "global_phase_radians"
_ROSS_SELINGER_SUBPROCESS_ONLY = False
_TEMP_PHASE_GATE = "phase_angle"
_RZ_COMPILE_MODE_CLIFFORD_T = "clifford_t"
_RZ_COMPILE_MODE_DYADIC = "dyadic"
_RZ_COMPILE_MODES = {_RZ_COMPILE_MODE_CLIFFORD_T, _RZ_COMPILE_MODE_DYADIC}
_FAST_IMPORT_NATIVE_GATES = frozenset({"h", "t", "tdg", "cnot"})
_FAST_IMPORT_GATE_COUNT_THRESHOLD = 4096


@dataclass(frozen=True)
class CircuitSpec:
    """Normalized Clifford+T circuit accepted by the TerKet simulator."""

    n_qubits: int
    gates: tuple[Gate, ...]
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "gates", tuple(_normalize_gate(g) for g in self.gates))
        object.__setattr__(self, "metadata", _normalize_circuit_metadata(self.metadata))
        _validate_gates(self.n_qubits, self.gates)


@dataclass
class _ImportCompileStats:
    global_phase_radians: float = 0.0
    exact_dyadic_phase_count: int = 0
    approximated_phase_count: int = 0
    total_angle_error: float = 0.0
    max_angle_error: float = 0.0

    def absorb(self, other: "_ImportCompileStats") -> None:
        self.global_phase_radians = _normalize_global_phase_radians(
            self.global_phase_radians + other.global_phase_radians
        )
        self.exact_dyadic_phase_count += other.exact_dyadic_phase_count
        self.approximated_phase_count += other.approximated_phase_count
        self.total_angle_error += other.total_angle_error
        self.max_angle_error = max(self.max_angle_error, other.max_angle_error)


def make_circuit(
    n_qubits: int,
    gates: Iterable[Gate],
    name: str | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> CircuitSpec:
    """Create a normalized circuit specification from a gate iterable."""
    return CircuitSpec(
        n_qubits=n_qubits,
        gates=tuple(gates),
        name=name,
        metadata={} if metadata is None else dict(metadata),
    )


def lift_exact_dyadic_precision(
    circuit: CircuitSpec,
    *,
    min_level: int,
) -> CircuitSpec:
    """Return an equivalent circuit with exact dyadic single-qubit phases lifted.

    This is an exact representation change used for structural experiments.
    Any diagonal single-qubit gate with dyadic precision below ``min_level`` is
    rewritten as an explicit ``rz_dyadic`` gate at ``min_level`` instead of the
    shortest equivalent representation. The represented circuit is unchanged,
    but Schur-state construction can see a higher retained phase precision.
    """
    if not isinstance(circuit, CircuitSpec):
        raise TypeError(f"Expected CircuitSpec, received {type(circuit)!r}.")

    min_level = int(min_level)
    if min_level < 1:
        raise ValueError(f"min_level must be positive, received {min_level}.")

    lifted_gates: list[Gate] = []
    changed = False
    for gate in circuit.gates:
        diagonal = _diagonal_phase_spec(gate)
        if diagonal is None:
            lifted_gates.append(gate)
            continue

        qubit, coeff, level = diagonal
        if level >= min_level:
            lifted_gates.append(gate)
            continue

        changed = True
        lifted_coeff = int(coeff) << (min_level - level)
        lifted_gates.append(("rz_dyadic", int(qubit), lifted_coeff, min_level))

    if not changed:
        return circuit
    return CircuitSpec(
        n_qubits=circuit.n_qubits,
        gates=tuple(lifted_gates),
        name=circuit.name,
        metadata=dict(circuit.metadata),
    )


def normalize_circuit(
    circuit: Any,
    gates: Iterable[Gate] | None = None,
    *,
    rz_tolerance: float = 1e-5,
    rz_compile_mode: str | None = _RZ_COMPILE_MODE_DYADIC,
) -> CircuitSpec:
    """
    Normalize supported circuit inputs.

    Accepted forms are:
    - ``CircuitSpec``
    - ``(n_qubits, gates)`` via ``normalize_circuit(n, gates)``
    - a TerKet Stim-style ``Circuit``
    - a raw ``stim.Circuit`` using the supported unitary/final-measurement subset
    - an OpenQASM 2 string or path
    - a Qiskit ``QuantumCircuit``
    """
    if isinstance(circuit, CircuitSpec):
        if gates is not None:
            raise TypeError("Do not pass gates separately when providing CircuitSpec.")
        return circuit

    if isinstance(circuit, int):
        if gates is None:
            raise TypeError("Gate list required when normalizing from an integer size.")
        return make_circuit(circuit, gates)

    if gates is not None:
        raise TypeError("Second positional gates argument is only valid with an integer size.")

    converter = getattr(circuit, "to_terket_circuit_spec", None)
    if callable(converter):
        return converter()

    circuit_type = type(circuit)
    if circuit_type.__name__ == "Circuit" and (
        circuit_type.__module__.startswith("stim.") or circuit_type.__module__.startswith("tsim.")
    ):
        from .circuit import Circuit as StimLikeCircuit

        return StimLikeCircuit(circuit).to_terket_circuit_spec()

    if isinstance(circuit, str):
        qasm_path = Path(circuit)
        if qasm_path.exists():
            source = qasm_path.read_text(encoding="utf-8")
            if _looks_like_openqasm3(source):
                return _parse_openqasm3_via_qiskit(
                    qasm_path,
                    name=qasm_path.stem,
                    rz_tolerance=rz_tolerance,
                    rz_compile_mode=rz_compile_mode,
                )
            return parse_openqasm2(
                source,
                name=qasm_path.stem,
                rz_tolerance=rz_tolerance,
                rz_compile_mode=rz_compile_mode,
            )
        if _looks_like_openqasm3(circuit):
            return _parse_openqasm3_via_qiskit(
                circuit,
                rz_tolerance=rz_tolerance,
                rz_compile_mode=rz_compile_mode,
            )
        return parse_openqasm2(circuit, rz_tolerance=rz_tolerance, rz_compile_mode=rz_compile_mode)

    if hasattr(circuit, "num_qubits") and hasattr(circuit, "data"):
        return from_qiskit(circuit, rz_tolerance=rz_tolerance, rz_compile_mode=rz_compile_mode)

    raise TypeError(f"Unsupported circuit input: {type(circuit)!r}")


def parse_openqasm2(
    source: str,
    name: str | None = None,
    *,
    rz_tolerance: float = 1e-5,
    rz_compile_mode: str | None = _RZ_COMPILE_MODE_DYADIC,
) -> CircuitSpec:
    """
    Parse a supported Clifford+T subset of OpenQASM 2 into a ``CircuitSpec``.

    The parser accepts only the TerKet gate set and ignores ``creg``,
    ``barrier``, ``measure``, and ``reset`` statements. ``rz`` gates stay in
    TerKet's native dyadic/arbitrary form unless ``rz_compile_mode`` requests
    explicit Clifford+T synthesis.
    """
    qregs: dict[str, int] = {}
    offsets: dict[str, int] = {}
    raw_gates: list[Gate] = []
    n_qubits = 0
    global_phase_radians = 0.0
    rz_tolerance = _validated_rz_tolerance(rz_tolerance)
    compile_mode = _normalize_rz_compile_mode(rz_compile_mode)

    for raw_line in source.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        if line.endswith(";"):
            line = line[:-1].strip()
        if not line:
            continue
        if line.lower().startswith("openqasm") or line.lower().startswith("include"):
            continue
        if line.lower().startswith("qreg "):
            reg_decl = line[5:].strip()
            match = _QASM_QUBIT.fullmatch(reg_decl)
            if match is None:
                raise ValueError(f"Unsupported qreg declaration: {raw_line!r}")
            reg_name = match.group(1)
            size = int(match.group(2))
            qregs[reg_name] = size
            offsets[reg_name] = n_qubits
            n_qubits += size
            continue
        if line.lower().startswith(("creg ", "barrier ", "measure ", "reset ")):
            continue

        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"Unsupported OpenQASM statement: {raw_line!r}")
        gate_token = parts[0].lower()
        gate_name = _QASM_GATE_MAP.get(gate_token)
        gate_angle_expr: str | None = None
        if gate_name is None and gate_token.startswith("rz(") and gate_token.endswith(")"):
            gate_name = "rz"
            gate_angle_expr = gate_token[3:-1]
        if gate_name is None:
            raise ValueError(
                f"Unsupported OpenQASM gate: {parts[0]!r}. "
                f"TerKet supports only Clifford+T gates {_QASM_SUPPORTED_GATE_SET}. "
                "Consider transpiling to this gate set first."
            )
        operands = [_parse_qasm_qubit(token.strip(), offsets, qregs) for token in parts[1].split(",")]
        if gate_name == "rz":
            if len(operands) != 1:
                raise ValueError(f"OpenQASM gate {parts[0]!r} expects one qubit.")
            if gate_angle_expr is None:  # pragma: no cover - internal guard
                raise ValueError("Missing OpenQASM rz angle.")
            try:
                angle = _evaluate_qasm_angle_expr(gate_angle_expr)
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported rz angle {gate_angle_expr!r}. Only numeric expressions over pi are supported."
                ) from exc
            if compile_mode == _RZ_COMPILE_MODE_DYADIC:
                phase_gate, exact_angle = _exact_phase_gate_from_angle(
                    angle,
                    operands[0],
                    source=f"Unsupported rz angle {gate_angle_expr!r}",
                )
                if phase_gate is not None:
                    raw_gates.append(phase_gate)
            else:
                raw_gates.append((_TEMP_PHASE_GATE, operands[0], angle))
        else:
            raw_gates.append((gate_name, *operands))

    fast_import = _fast_import_gate_sequence_if_supported(raw_gates)
    if fast_import is None:
        compiled_gates, compile_stats = _compile_import_gate_sequence(raw_gates, tolerance=rz_tolerance)
    else:
        compiled_gates = fast_import
        compile_stats = _ImportCompileStats()
    global_phase_radians = _normalize_global_phase_radians(
        global_phase_radians + compile_stats.global_phase_radians
    )

    return make_circuit(
        n_qubits,
        compiled_gates,
        name=name,
        metadata=_metadata_with_global_phase(global_phase_radians),
    )


def _looks_like_openqasm3(source: str) -> bool:
    for raw_line in source.splitlines():
        line = raw_line.split("//", 1)[0].strip().lower()
        if not line:
            continue
        return line.startswith("openqasm 3")
    return False


def _parse_openqasm3_via_qiskit(
    source: str | Path,
    *,
    name: str | None = None,
    rz_tolerance: float = 1e-5,
    rz_compile_mode: str | None = _RZ_COMPILE_MODE_DYADIC,
) -> CircuitSpec:
    try:
        import qiskit.qasm3
    except ImportError as exc:  # pragma: no cover - depends on optional qiskit install
        raise RuntimeError("Qiskit is required to import OpenQASM 3 circuits.") from exc

    circuit = qiskit.qasm3.load(source) if isinstance(source, Path) else qiskit.qasm3.loads(source)
    if name is not None:
        circuit.name = name
    compile_mode = _RZ_COMPILE_MODE_DYADIC if rz_compile_mode is None else rz_compile_mode
    return from_qiskit(circuit, rz_tolerance=rz_tolerance, rz_compile_mode=compile_mode)


def from_qiskit(
    circuit: Any,
    *,
    rz_tolerance: float = 1e-5,
    rz_compile_mode: str | None = _RZ_COMPILE_MODE_DYADIC,
) -> CircuitSpec:
    """
    Convert a Qiskit ``QuantumCircuit`` into a normalized ``CircuitSpec``.

    By default, ``rz`` and phase gates stay in TerKet's native
    dyadic/arbitrary representation instead of being synthesized into
    Clifford+T.
    """
    working_circuit = circuit
    if hasattr(circuit, "remove_final_measurements"):
        removed = circuit.remove_final_measurements(inplace=False)
        if removed is not None:
            working_circuit = removed

    raw_gates: list[Gate] = []
    rz_tolerance = _validated_rz_tolerance(rz_tolerance)
    compile_mode = _normalize_rz_compile_mode(rz_compile_mode)
    global_phase_radians = _normalize_global_phase_radians(
        _coerce_finite_radians(
            getattr(working_circuit, "global_phase", 0.0),
            source="Unsupported Qiskit circuit global phase",
        )
    )
    qubit_indices = {qubit: idx for idx, qubit in enumerate(working_circuit.qubits)}
    for instruction in working_circuit.data:
        operation = instruction.operation
        name = _QASM_GATE_MAP.get(operation.name.lower())
        qubits = [qubit_indices[qubit] for qubit in instruction.qubits]
        if name is not None:
            raw_gates.append((name, *qubits))
            continue
        op_name = operation.name.lower()
        if op_name in {"barrier", "delay", "id"}:
            continue
        if op_name == "sx":
            if len(qubits) != 1:
                raise ValueError(f"Unsupported Qiskit gate arity for {operation.name!r}.")
            raw_gates.append(("sx", qubits[0]))
            continue
        if op_name == "rz":
            if len(qubits) != 1:
                raise ValueError(f"Unsupported Qiskit gate arity for {operation.name!r}.")
            angle = _coerce_finite_radians(
                operation.params[0],
                source=f"Unsupported Qiskit rz angle {operation.params[0]!r}",
            )
            if compile_mode == _RZ_COMPILE_MODE_DYADIC:
                phase_gate, exact_angle = _exact_phase_gate_from_angle(
                    angle,
                    qubits[0],
                    source=f"Unsupported Qiskit rz angle {operation.params[0]!r}",
                )
                if phase_gate is not None:
                    raw_gates.append(phase_gate)
                global_phase_radians += _normalize_global_phase_radians(-0.5 * exact_angle)
            else:
                raw_gates.append((_TEMP_PHASE_GATE, qubits[0], angle))
                global_phase_radians += _normalize_global_phase_radians(-0.5 * angle)
            continue
        if op_name in {"p", "u1"}:
            if len(qubits) != 1:
                raise ValueError(f"Unsupported Qiskit gate arity for {operation.name!r}.")
            angle = _coerce_finite_radians(
                operation.params[0],
                source=f"Unsupported Qiskit phase angle {operation.params[0]!r}",
            )
            if compile_mode == _RZ_COMPILE_MODE_DYADIC:
                phase_gate, _ = _exact_phase_gate_from_angle(
                    angle,
                    qubits[0],
                    source=f"Unsupported Qiskit phase angle {operation.params[0]!r}",
                )
                if phase_gate is not None:
                    raw_gates.append(phase_gate)
            else:
                raw_gates.append((_TEMP_PHASE_GATE, qubits[0], angle))
            continue
        if op_name == "rzz":
            if len(qubits) != 2:
                raise ValueError(f"Unsupported Qiskit gate arity for {operation.name!r}.")
            angle = _coerce_finite_radians(
                operation.params[0],
                source=f"Unsupported Qiskit rzz angle {operation.params[0]!r}",
            )
            if compile_mode == _RZ_COMPILE_MODE_DYADIC:
                exact = _exact_dyadic_phase_from_angle(angle)
                if exact is not None:
                    coeff, precision_level = exact
                    raw_gates.append(("rzz_dyadic", qubits[0], qubits[1], coeff, precision_level))
                    global_phase_radians += _normalize_global_phase_radians(-0.5 * angle)
                    continue
            raw_gates.extend(
                (
                    ("cnot", qubits[0], qubits[1]),
                    (_TEMP_PHASE_GATE, qubits[1], angle),
                    ("cnot", qubits[0], qubits[1]),
                )
            )
            global_phase_radians += _normalize_global_phase_radians(-0.5 * angle)
            continue
        raise ValueError(f"Unsupported Qiskit gate: {operation.name!r}")
    fast_import = _fast_import_gate_sequence_if_supported(raw_gates)
    if fast_import is None:
        compiled_gates, compile_stats = _compile_import_gate_sequence(raw_gates, tolerance=rz_tolerance)
    else:
        compiled_gates = fast_import
        compile_stats = _ImportCompileStats()
    global_phase_radians = _normalize_global_phase_radians(
        global_phase_radians + compile_stats.global_phase_radians
    )
    return make_circuit(
        working_circuit.num_qubits,
        compiled_gates,
        name=getattr(working_circuit, "name", None),
        metadata=_metadata_with_global_phase(global_phase_radians),
    )


def _fast_import_gate_sequence_if_supported(raw_gates: Sequence[Gate]) -> tuple[Gate, ...] | None:
    """Bypass the generic import compiler for very large already-native streams."""

    if len(raw_gates) < _FAST_IMPORT_GATE_COUNT_THRESHOLD:
        return None
    normalized: list[Gate] = []
    for raw_gate in raw_gates:
        gate = _normalize_gate(raw_gate)
        if gate[0] not in _FAST_IMPORT_NATIVE_GATES:
            return None
        normalized.append(gate)
    return tuple(normalized)


def to_qiskit(circuit: Any):
    """Convert a supported circuit input into a Qiskit ``QuantumCircuit``."""
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise RuntimeError("Qiskit is required for to_qiskit().") from exc

    spec = normalize_circuit(circuit)
    qc = QuantumCircuit(spec.n_qubits, name=spec.name)
    qc.global_phase = _circuit_global_phase_radians(spec)
    for gate in spec.gates:
        name = gate[0]
        if name == "cnot":
            qc.cx(gate[1], gate[2])
        elif name == "sx":
            qc.sx(gate[1])
        elif name == "sxdg":
            qc.sxdg(gate[1])
        elif name == "rzz_dyadic":
            qc.cx(gate[1], gate[2])
            qc.p(_dyadic_phase_to_angle(gate[3], gate[4]), gate[2])
            qc.cx(gate[1], gate[2])
        elif name == "rz_arbitrary":
            qc.p(_coerce_finite_radians(gate[2], source="Unsupported arbitrary phase angle"), gate[1])
        elif name == "rz_dyadic":
            qc.p(_dyadic_phase_to_angle(gate[2], gate[3]), gate[1])
        elif name == "rz_pi_16":
            qc.p(math.pi / 16.0, gate[1])
        elif name == "rz_pi_16_dg":
            qc.p(-math.pi / 16.0, gate[1])
        elif name == "rz_pi_32":
            qc.p(math.pi / 32.0, gate[1])
        elif name == "rz_pi_32_dg":
            qc.p(-math.pi / 32.0, gate[1])
        else:
            getattr(qc, name)(*gate[1:])
    return qc


def bits_to_index(bits: Sequence[int]) -> int:
    """Encode a little-endian bit string as a basis-state index."""
    return sum((int(bit) & 1) << idx for idx, bit in enumerate(bits))


def bits_to_little_endian_string(bits: Sequence[int]) -> str:
    """Serialize a little-endian bit sequence as ``q0 q1 ...`` order."""
    return "".join(str(int(bit) & 1) for bit in bits)


def bits_to_big_endian_string(bits: Sequence[int]) -> str:
    """Serialize a little-endian bit sequence as a human-readable bitstring."""
    return bits_to_little_endian_string(reversed(tuple(bits)))


def little_endian_string_to_bits(bitstring: str) -> tuple[int, ...]:
    """Parse a ``q0 q1 ...`` bitstring into TerKet's internal little-endian tuple."""
    if any(char not in {"0", "1"} for char in bitstring):
        raise ValueError(f"Bitstring must contain only 0/1 characters, received {bitstring!r}.")
    return tuple(int(char) for char in bitstring)


def big_endian_string_to_bits(bitstring: str) -> tuple[int, ...]:
    """Parse a human-readable bitstring into TerKet's internal little-endian tuple."""
    return little_endian_string_to_bits(bitstring[::-1])


def iter_bitstrings(n_qubits: int):
    """Yield all little-endian basis strings of length ``n_qubits``."""
    for value in range(1 << n_qubits):
        yield [(value >> idx) & 1 for idx in range(n_qubits)]


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
_ONE_QUBIT_IDENTITY = np.eye(2, dtype=complex)
_ONE_QUBIT_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)
_ONE_QUBIT_SX = np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=complex)
_ONE_QUBIT_SXDG = np.array([[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]], dtype=complex)
_ONE_QUBIT_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def _gate_qubits(gate: Gate) -> tuple[int, ...]:
    name = gate[0]
    if name in {"rz_dyadic", "rz_arbitrary"}:
        return (gate[1],)
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


def _gate_can_slide_left_past(previous: Gate, gate: Gate) -> bool:
    previous_qubits = set(_gate_qubits(previous))
    gate_qubits = set(_gate_qubits(gate))
    if gate_qubits.isdisjoint(previous_qubits):
        return True

    diagonal = _diagonal_phase_spec(gate)
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
            if first == third and first[0] == "h" and int(first[1]) == int(second[1]):
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


def _rewrite_gate_sequence(gates: Sequence[Gate]) -> tuple[Gate, ...]:
    """Apply safe local rewrites before Schur-state construction."""
    rewritten: list[Gate] = []

    for raw_gate in gates:
        gate = _normalize_gate(raw_gate)
        insert_pos = len(rewritten)
        while insert_pos > 0 and _gate_can_slide_left_past(rewritten[insert_pos - 1], gate):
            insert_pos -= 1

        diagonal = _diagonal_phase_spec(gate)
        if diagonal is not None and insert_pos > 0:
            previous = _diagonal_phase_spec(rewritten[insert_pos - 1])
            if previous is not None and previous[0] == diagonal[0]:
                combined = _combine_dyadic_phases(previous, diagonal)
                replacement = list(_emit_dyadic_phase_gate(*combined))
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
    return (name, *gate[1:])


def _validate_gates(n_qubits: int, gates: Sequence[Gate]) -> None:
    for gate in gates:
        name = gate[0]
        if name not in SUPPORTED_GATES:
            raise ValueError(f"Unsupported gate: {name!r}")
        if name in {"cnot", "cz"}:
            arity = 2
        elif name == "rzz_dyadic":
            arity = 4
        elif name == "rz_dyadic":
            arity = 3
        elif name == "rz_arbitrary":
            arity = 2
        else:
            arity = 1
        if len(gate) != arity + 1:
            raise ValueError(f"Gate {gate!r} has the wrong arity.")
        qubits = gate[1:2] if name in {"rz_dyadic", "rz_arbitrary"} else gate[1:3] if name == "rzz_dyadic" else gate[1:]
        for qubit in qubits:
            if not isinstance(qubit, int):
                raise TypeError(f"Gate {gate!r} uses a non-integer qubit index.")
            if not 0 <= qubit < n_qubits:
                raise ValueError(f"Gate {gate!r} targets qubit outside 0..{n_qubits - 1}.")
        if name == "rzz_dyadic":
            coeff, precision_level = gate[3], gate[4]
            if not isinstance(coeff, int):
                raise TypeError(f"Gate {gate!r} uses a non-integer phase coefficient.")
            if not isinstance(precision_level, int):
                raise TypeError(f"Gate {gate!r} uses a non-integer precision level.")
            if precision_level < 1:
                raise ValueError(f"Gate {gate!r} uses invalid precision level {precision_level}.")
        if name == "rz_dyadic":
            coeff, precision_level = gate[2], gate[3]
            if not isinstance(coeff, int):
                raise TypeError(f"Gate {gate!r} uses a non-integer phase coefficient.")
            if not isinstance(precision_level, int):
                raise TypeError(f"Gate {gate!r} uses a non-integer precision level.")
            if precision_level < 1:
                raise ValueError(f"Gate {gate!r} uses invalid precision level {precision_level}.")
        if name == "rz_arbitrary":
            _coerce_finite_radians(gate[2], source="Unsupported arbitrary phase angle")


def _parse_qasm_qubit(token: str, offsets: dict[str, int], qregs: dict[str, int]) -> int:
    match = _QASM_QUBIT.fullmatch(token)
    if match is None:
        raise ValueError(f"Unsupported qubit reference: {token!r}")
    reg_name = match.group(1)
    if reg_name not in qregs:
        raise ValueError(f"Unknown qreg {reg_name!r}.")
    offset = int(match.group(2))
    if not 0 <= offset < qregs[reg_name]:
        raise ValueError(f"Qubit index out of range in {token!r}.")
    return offsets[reg_name] + offset


def _normalize_global_phase_radians(value: float) -> float:
    normalized = math.remainder(float(value), 2.0 * math.pi)
    return 0.0 if math.isclose(normalized, 0.0, rel_tol=0.0, abs_tol=1e-15) else normalized


def _normalize_circuit_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    normalized = {} if metadata is None else dict(metadata)
    if _GLOBAL_PHASE_METADATA_KEY in normalized:
        normalized[_GLOBAL_PHASE_METADATA_KEY] = _normalize_global_phase_radians(_coerce_finite_radians(
            normalized[_GLOBAL_PHASE_METADATA_KEY],
            source="Unsupported circuit global phase",
        ))
    return normalized


def _metadata_with_global_phase(global_phase_radians: float) -> dict[str, Any]:
    normalized = _normalize_global_phase_radians(global_phase_radians)
    if normalized == 0.0:
        return {}
    return {_GLOBAL_PHASE_METADATA_KEY: normalized}


def _circuit_global_phase_radians(spec: CircuitSpec) -> float:
    return float(spec.metadata.get(_GLOBAL_PHASE_METADATA_KEY, 0.0))


def _validated_rz_tolerance(tolerance: float) -> float:
    tolerance = float(tolerance)
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, received {tolerance}.")
    if tolerance > _MAX_RZ_TOLERANCE:
        raise ValueError(f"rz_tolerance must be <= {_MAX_RZ_TOLERANCE:.3e}, received {tolerance:.3e}.")
    return tolerance


def _normalize_rz_compile_mode(
    mode: str | None,
    *,
    default: str = _RZ_COMPILE_MODE_DYADIC,
) -> str:
    if mode is None:
        mode = default
    normalized = str(mode).strip().lower()
    if normalized not in _RZ_COMPILE_MODES:
        supported = ", ".join(sorted(_RZ_COMPILE_MODES))
        raise ValueError(f"rz_compile_mode must be one of {{{supported}}}, received {mode!r}.")
    return normalized


def _coerce_finite_radians(angle: Any, *, source: str) -> float:
    try:
        value = float(angle)
    except Exception as exc:  # pragma: no cover - depends on optional qiskit parameter types
        raise ValueError(f"{source} {angle!r}. A numeric value is required.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{source} {angle!r}. Finite numeric values are required.")
    return value


def _exact_dyadic_phase_from_angle(angle: float) -> tuple[int, int] | None:
    try:
        coeff, precision_level, _ = dyadic_snap(
            angle,
            max_level=_EXACT_DYADIC_MAX_LEVEL,
            tolerance=_EXACT_DYADIC_TOLERANCE,
        )
    except ValueError:
        return None
    return coeff, precision_level


def _exact_phase_gate_from_angle(
    angle: Any,
    qubit: int,
    *,
    source: str,
) -> tuple[Gate | None, float]:
    value = _coerce_finite_radians(angle, source=source)
    exact = _exact_dyadic_phase_from_angle(value)
    if exact is None:
        return ("rz_arbitrary", qubit, value), value
    coeff, precision_level = _normalize_dyadic_phase(exact[0], exact[1])
    if coeff == 0:
        return None, value
    return ("rz_dyadic", qubit, coeff, precision_level), value


def _dyadic_phase_gate_from_angle(
    angle: Any,
    qubit: int,
    *,
    tolerance: float,
    source: str,
) -> tuple[Gate | None, float]:
    try:
        coeff, precision_level, _ = dyadic_snap(angle, tolerance=tolerance)
    except ValueError as exc:
        raise ValueError(f"{source}. {exc}") from exc
    coeff, precision_level = _normalize_dyadic_phase(coeff, precision_level)
    snapped_angle = _dyadic_phase_to_angle(coeff, precision_level)
    if coeff == 0:
        return None, snapped_angle
    return ("rz_dyadic", qubit, coeff, precision_level), snapped_angle


def _normalize_phase_angle(angle: float) -> float:
    return _normalize_global_phase_radians(angle)


def _phase_angle_from_gate(gate: Gate) -> float | None:
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


def _gate_qubits_import(gate: Gate) -> tuple[int, ...]:
    if gate[0] in {"rz_arbitrary", "rz_dyadic", _TEMP_PHASE_GATE}:
        return (int(gate[1]),)
    return tuple(int(qubit) for qubit in gate[1:] if isinstance(qubit, int))


def _is_single_qubit_import_gate(gate: Gate) -> bool:
    return len(_gate_qubits_import(gate)) == 1


def _merge_import_diagonal_phases(gates: Sequence[Gate]) -> tuple[Gate, ...]:
    pending_angles: dict[int, float] = {}
    merged: list[Gate] = []

    def add_pending(qubit: int, angle: float) -> None:
        combined = _normalize_phase_angle(pending_angles.get(qubit, 0.0) + angle)
        if combined == 0.0:
            pending_angles.pop(qubit, None)
        else:
            pending_angles[qubit] = combined

    def flush_qubits(qubits: Iterable[int]) -> None:
        for qubit in sorted(set(int(qubit) for qubit in qubits)):
            angle = pending_angles.pop(qubit, 0.0)
            angle = _normalize_phase_angle(angle)
            if angle != 0.0:
                merged.append((_TEMP_PHASE_GATE, qubit, angle))

    for raw_gate in gates:
        gate = raw_gate if raw_gate and raw_gate[0] == _TEMP_PHASE_GATE else _normalize_gate(raw_gate)
        phase_angle = _phase_angle_from_gate(gate)
        if phase_angle is not None:
            add_pending(int(gate[1]), phase_angle)
            continue

        if gate[0] == "cz":
            merged.append(gate)
            continue

        flush_qubits(_gate_qubits_import(gate))
        merged.append(gate)

    flush_qubits(tuple(sorted(pending_angles)))
    return tuple(merged)


def _exact_single_qubit_run(run: Sequence[Gate]) -> tuple[Gate, ...] | None:
    exact_gates: list[Gate] = []
    for gate in run:
        if gate[0] == _TEMP_PHASE_GATE:
            exact = _exact_dyadic_phase_from_angle(_coerce_finite_radians(gate[2], source="Unsupported phase angle"))
            if exact is None:
                return None
            exact_gates.extend(_emit_dyadic_phase_gate(int(gate[1]), exact[0], exact[1]))
        else:
            exact_gates.append(_normalize_gate(gate))
    return _rewrite_gate_sequence(exact_gates)


def _phase_gate_matrix(angle: float) -> np.ndarray:
    return np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, cmath.exp(1j * angle)]], dtype=complex)


def _one_qubit_gate_matrix(gate: Gate) -> np.ndarray:
    phase_angle = _phase_angle_from_gate(gate)
    if phase_angle is not None:
        return _phase_gate_matrix(phase_angle)
    if gate[0] == "rz_arbitrary":
        return _phase_gate_matrix(_coerce_finite_radians(gate[2], source="Unsupported arbitrary phase angle"))
    if gate[0] == "h":
        return _ONE_QUBIT_H
    if gate[0] == "sx":
        return _ONE_QUBIT_SX
    if gate[0] == "sxdg":
        return _ONE_QUBIT_SXDG
    if gate[0] == "x":
        return _ONE_QUBIT_X
    raise ValueError(f"Unsupported one-qubit run gate {gate!r}.")


def _unitary_key(matrix: np.ndarray) -> tuple[complex, ...]:
    return tuple(complex(value) for value in np.asarray(matrix, dtype=complex).reshape(-1))


def _matrix_from_key(matrix_key: tuple[complex, ...]) -> np.ndarray:
    return np.array(matrix_key, dtype=complex).reshape(2, 2)


def _one_qubit_run_unitary(run: Sequence[Gate]) -> np.ndarray:
    unitary = _ONE_QUBIT_IDENTITY.copy()
    for gate in run:
        unitary = _one_qubit_gate_matrix(gate) @ unitary
    return unitary


def _compile_one_qubit_run(
    qubit: int,
    run: Sequence[Gate],
    *,
    tolerance: float,
) -> tuple[tuple[Gate, ...], _ImportCompileStats]:
    if not run:
        return (), _ImportCompileStats()

    normalized_run = tuple(
        gate if gate[0] == _TEMP_PHASE_GATE else _normalize_gate(gate)
        for gate in run
    )
    phase_gate_count = sum(1 for gate in normalized_run if gate[0] == _TEMP_PHASE_GATE)
    if phase_gate_count == 0:
        return _rewrite_gate_sequence(normalized_run), _ImportCompileStats()

    exact_run = _exact_single_qubit_run(normalized_run)
    if exact_run is not None:
        return exact_run, _ImportCompileStats(exact_dyadic_phase_count=phase_gate_count)

    if all(_phase_angle_from_gate(gate) is not None for gate in normalized_run):
        total_angle = 0.0
        for gate in normalized_run:
            phase_angle = _phase_angle_from_gate(gate)
            if phase_angle is None:  # pragma: no cover - guarded above
                raise ValueError(f"Unsupported diagonal gate {gate!r}.")
            total_angle = _normalize_phase_angle(total_angle + phase_angle)
        compiled_gates, gate_global_phase, is_exact_dyadic, angle_error = _compile_phase_gate(
            total_angle,
            qubit,
            tolerance=tolerance,
            source=f"Unsupported diagonal phase run on qubit {qubit}",
        )
        stats = _ImportCompileStats(global_phase_radians=gate_global_phase)
        if is_exact_dyadic:
            stats.exact_dyadic_phase_count = phase_gate_count
        else:
            stats.approximated_phase_count = phase_gate_count
            stats.total_angle_error = angle_error
            stats.max_angle_error = angle_error
        return _rewrite_gate_sequence(compiled_gates), stats

    # Mixed one-qubit runs are still compiled through their residual phase gates.
    # In practice, gridsynth_unitary on these short-but-unique runs costs far more
    # compile time than it saves in gate count.
    compiled: list[Gate] = []
    stats = _ImportCompileStats()
    for gate in normalized_run:
        if gate[0] != _TEMP_PHASE_GATE:
            compiled.append(gate)
            continue
        compiled_gates, gate_global_phase, is_exact_dyadic, angle_error = _compile_phase_gate(
            gate[2],
            qubit,
            tolerance=tolerance,
            source=f"Unsupported phase angle {gate[2]!r}",
        )
        compiled.extend(compiled_gates)
        gate_stats = _ImportCompileStats(global_phase_radians=gate_global_phase)
        if is_exact_dyadic:
            gate_stats.exact_dyadic_phase_count = 1
        else:
            gate_stats.approximated_phase_count = 1
            gate_stats.total_angle_error = angle_error
            gate_stats.max_angle_error = angle_error
        stats.absorb(gate_stats)
    return _rewrite_gate_sequence(compiled), stats


def _compile_import_gate_sequence(
    raw_gates: Sequence[Gate],
    *,
    tolerance: float,
) -> tuple[tuple[Gate, ...], _ImportCompileStats]:
    merged_gates = _merge_import_diagonal_phases(raw_gates)
    compiled: list[Gate] = []
    stats = _ImportCompileStats()
    pending_runs: dict[int, list[Gate]] = {}

    def flush_qubit(qubit: int) -> None:
        run = pending_runs.pop(qubit, None)
        if not run:
            return
        compiled_run, run_stats = _compile_one_qubit_run(qubit, tuple(run), tolerance=tolerance)
        compiled.extend(compiled_run)
        stats.absorb(run_stats)

    for gate in merged_gates:
        if _is_single_qubit_import_gate(gate):
            qubit = _gate_qubits_import(gate)[0]
            pending_runs.setdefault(qubit, []).append(gate)
            continue

        for qubit in sorted(_gate_qubits_import(gate)):
            flush_qubit(qubit)
        compiled.append(_normalize_gate(gate))

    for qubit in sorted(pending_runs):
        flush_qubit(qubit)

    return _rewrite_gate_sequence(compiled), stats


def _reduced_synthesis_angle(angle: float, *, kind: str) -> tuple[float, float]:
    reduced = math.remainder(angle, 2.0 * math.pi)
    if kind != "rz":
        return reduced, 0.0
    turns = int(round((angle - reduced) / (2.0 * math.pi)))
    return reduced, _normalize_global_phase_radians(-math.pi * turns)


def _translate_qiskit_single_qubit_circuit(qc: Any) -> tuple[Gate, ...]:
    translated: list[Gate] = []
    for instruction in qc.data:
        operation = instruction.operation
        name = operation.name.lower()
        if name == "id":
            continue
        normalized = _normalize_gate((name, 0))
        if normalized[0] not in {"h", "x", "t", "tdg", "s", "sdg", "z"}:
            raise ValueError(f"Ross-Selinger synthesis emitted unsupported gate {operation.name!r}.")
        translated.append(normalized)
    return tuple(translated)


def _translate_gate_names(names: Sequence[str]) -> tuple[Gate, ...]:
    translated: list[Gate] = []
    for name in names:
        normalized = _normalize_gate((name, 0))
        if normalized[0] not in {"h", "x", "t", "tdg", "s", "sdg", "z"}:
            raise ValueError(f"Ross-Selinger synthesis emitted unsupported gate {name!r}.")
        translated.append(normalized)
    return tuple(translated)


def _ross_selinger_template_subprocess(kind: str, angle: float, epsilon: float) -> tuple[tuple[Gate, ...], float]:
    import json
    import subprocess
    import sys

    script = """
import cmath
import json
import sys

import numpy as np
from qiskit.synthesis import gridsynth_rz, gridsynth_unitary

kind = sys.argv[1]
angle = float(sys.argv[2])
epsilon = float(sys.argv[3])

if kind == "rz":
    try:
        qc = gridsynth_rz(angle, epsilon=epsilon)
    except BaseException:
        qc = gridsynth_unitary(
            np.array(
                [
                    [cmath.exp(-0.5j * angle), 0.0 + 0.0j],
                    [0.0 + 0.0j, cmath.exp(0.5j * angle)],
                ],
                dtype=complex,
            ),
            epsilon=epsilon,
        )
elif kind == "phase":
    qc = gridsynth_unitary(
        np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, cmath.exp(1j * angle)]], dtype=complex),
        epsilon=epsilon,
    )
else:
    raise ValueError(f"unsupported kind {kind!r}")

payload = {
    "names": [instruction.operation.name.lower() for instruction in qc.data if instruction.operation.name.lower() != "id"],
    "global_phase": float(qc.global_phase),
}
print(json.dumps(payload))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, kind, repr(angle), repr(epsilon)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or f"subprocess exited with status {completed.returncode}"
        raise RuntimeError(
            f"Ross-Selinger synthesis subprocess failed for {kind} angle {angle!r} with tolerance {epsilon:.3e}: "
            f"{detail}"
        )
    payload = json.loads(completed.stdout)
    rewritten = _rewrite_gate_sequence(_translate_gate_names(tuple(payload["names"])))
    global_phase = _normalize_global_phase_radians(
        _coerce_finite_radians(payload["global_phase"], source="Unsupported Ross-Selinger global phase")
    )
    return rewritten, global_phase


def _ross_selinger_unitary_template_subprocess(
    matrix_key: tuple[complex, ...],
    epsilon: float,
) -> tuple[tuple[Gate, ...], float]:
    import json
    import subprocess
    import sys

    matrix_payload = json.dumps(
        [[float(value.real), float(value.imag)] for value in matrix_key],
        separators=(",", ":"),
    )
    script = """
import json
import sys

import numpy as np
from qiskit.synthesis import gridsynth_unitary

matrix_payload = json.loads(sys.argv[1])
epsilon = float(sys.argv[2])
matrix = np.array(
    [complex(real, imag) for real, imag in matrix_payload],
    dtype=complex,
).reshape(2, 2)
qc = gridsynth_unitary(matrix, epsilon=epsilon)
payload = {
    "names": [instruction.operation.name.lower() for instruction in qc.data if instruction.operation.name.lower() != "id"],
    "global_phase": float(qc.global_phase),
}
print(json.dumps(payload))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, matrix_payload, repr(epsilon)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or f"subprocess exited with status {completed.returncode}"
        raise RuntimeError(
            "Ross-Selinger unitary synthesis subprocess failed "
            f"for matrix {matrix_key!r} with tolerance {epsilon:.3e}: {detail}"
        )
    payload = json.loads(completed.stdout)
    rewritten = _rewrite_gate_sequence(_translate_gate_names(tuple(payload["names"])))
    global_phase = _normalize_global_phase_radians(
        _coerce_finite_radians(payload["global_phase"], source="Unsupported Ross-Selinger global phase")
    )
    return rewritten, global_phase


@lru_cache(maxsize=4096)
def _ross_selinger_template(kind: str, angle: float, epsilon: float) -> tuple[tuple[Gate, ...], float]:
    global _ROSS_SELINGER_SUBPROCESS_ONLY
    if epsilon <= 0.0:
        raise ValueError("Ross-Selinger synthesis requires positive rz_tolerance.")
    if _ROSS_SELINGER_SUBPROCESS_ONLY:
        return _ross_selinger_template_subprocess(kind, angle, epsilon)

    try:
        import numpy as np
        from qiskit.synthesis import gridsynth_rz, gridsynth_unitary
    except ImportError as exc:  # pragma: no cover - depends on optional qiskit install
        raise RuntimeError(
            "Qiskit with Ross-Selinger gridsynth support is required to synthesize non-dyadic rz gates."
        ) from exc

    try:
        if kind == "rz":
            try:
                qc = gridsynth_rz(angle, epsilon=epsilon)
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                qc = gridsynth_unitary(
                    np.array(
                        [
                            [cmath.exp(-0.5j * angle), 0.0 + 0.0j],
                            [0.0 + 0.0j, cmath.exp(0.5j * angle)],
                        ],
                        dtype=complex,
                    ),
                    epsilon=epsilon,
                )
        elif kind == "phase":
            qc = gridsynth_unitary(
                np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, cmath.exp(1j * angle)]], dtype=complex),
                epsilon=epsilon,
            )
        else:  # pragma: no cover - internal guard
            raise ValueError(f"Unsupported Ross-Selinger synthesis kind {kind!r}.")
    except BaseException as exc:  # pragma: no cover - depends on qiskit/pyo3 failure modes
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return _ross_selinger_template_subprocess(kind, angle, epsilon)

    rewritten = _rewrite_gate_sequence(_translate_qiskit_single_qubit_circuit(qc))
    global_phase = _normalize_global_phase_radians(_coerce_finite_radians(
        getattr(qc, "global_phase", 0.0),
        source="Unsupported Ross-Selinger global phase",
    ))
    return rewritten, global_phase


@lru_cache(maxsize=4096)
def _ross_selinger_unitary_template(
    matrix_key: tuple[complex, ...],
    epsilon: float,
) -> tuple[tuple[Gate, ...], float]:
    global _ROSS_SELINGER_SUBPROCESS_ONLY
    if epsilon <= 0.0:
        raise ValueError("Ross-Selinger synthesis requires positive rz_tolerance.")
    if _ROSS_SELINGER_SUBPROCESS_ONLY:
        return _ross_selinger_unitary_template_subprocess(matrix_key, epsilon)

    try:
        from qiskit.synthesis import gridsynth_unitary
    except ImportError as exc:  # pragma: no cover - depends on optional qiskit install
        raise RuntimeError(
            "Qiskit with Ross-Selinger gridsynth support is required to synthesize non-dyadic rz gates."
        ) from exc

    try:
        qc = gridsynth_unitary(_matrix_from_key(matrix_key), epsilon=epsilon)
    except BaseException as exc:  # pragma: no cover - depends on qiskit/pyo3 failure modes
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return _ross_selinger_unitary_template_subprocess(matrix_key, epsilon)

    rewritten = _rewrite_gate_sequence(_translate_qiskit_single_qubit_circuit(qc))
    global_phase = _normalize_global_phase_radians(_coerce_finite_radians(
        getattr(qc, "global_phase", 0.0),
        source="Unsupported Ross-Selinger global phase",
    ))
    return rewritten, global_phase


def _retarget_single_qubit_gates(gates: Sequence[Gate], qubit: int) -> tuple[Gate, ...]:
    retargeted: list[Gate] = []
    for gate in gates:
        if gate[0] == "rz_dyadic":
            retargeted.append((gate[0], qubit, gate[2], gate[3]))
        elif gate[0] == "rz_arbitrary":
            retargeted.append((gate[0], qubit, gate[2]))
        else:
            retargeted.append((gate[0], qubit))
    return tuple(retargeted)


def _compile_single_qubit_rotation(
    angle: Any,
    qubit: int,
    *,
    tolerance: float,
    kind: str,
    source: str,
) -> tuple[tuple[Gate, ...], float, bool, float]:
    value = _coerce_finite_radians(angle, source=source)
    exact = _exact_dyadic_phase_from_angle(value)
    if exact is not None:
        coeff, precision_level = exact
        compiled_gates = _emit_dyadic_phase_gate(qubit, coeff, precision_level)
        gate_global_phase = -0.5 * value if kind == "rz" else 0.0
        return compiled_gates, _normalize_global_phase_radians(gate_global_phase), True, 0.0

    if tolerance == 0.0:
        raise ValueError(f"{source}. Non-dyadic angles require positive rz_tolerance for Ross-Selinger synthesis.")

    synth_angle, periodic_global_phase = _reduced_synthesis_angle(value, kind=kind)
    template_gates, gate_global_phase = _ross_selinger_template(kind, synth_angle, tolerance)
    compiled_gates = _retarget_single_qubit_gates(template_gates, qubit)
    return (
        compiled_gates,
        _normalize_global_phase_radians(gate_global_phase + periodic_global_phase),
        False,
        tolerance,
    )


def _compile_rz_gate(
    angle: Any,
    qubit: int,
    *,
    tolerance: float,
    source: str,
) -> tuple[tuple[Gate, ...], float, bool, float]:
    return _compile_single_qubit_rotation(
        angle,
        qubit,
        tolerance=tolerance,
        kind="rz",
        source=source,
    )


def _compile_phase_gate(
    angle: Any,
    qubit: int,
    *,
    tolerance: float,
    source: str,
) -> tuple[tuple[Gate, ...], float, bool, float]:
    value = _coerce_finite_radians(angle, source=source)
    compiled_gates, gate_global_phase, is_exact_dyadic, angle_error = _compile_single_qubit_rotation(
        value,
        qubit,
        tolerance=tolerance,
        kind="rz",
        source=source,
    )
    return (
        compiled_gates,
        _normalize_global_phase_radians(gate_global_phase + 0.5 * value),
        is_exact_dyadic,
        angle_error,
    )


def _compile_qasm_rz_gate(
    expr: str | None,
    qubit: int,
    *,
    tolerance: float,
) -> tuple[tuple[Gate, ...], float, bool, float]:
    if expr is None:  # pragma: no cover - internal guard
        raise ValueError("Missing OpenQASM rz angle.")
    try:
        value = _evaluate_qasm_angle_expr(expr)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported rz angle {expr!r}. Only numeric expressions over pi are supported."
        ) from exc
    return _compile_rz_gate(
        value,
        qubit,
        tolerance=tolerance,
        source=f"Unsupported rz angle {expr!r}",
    )


def dyadic_snap(
    angle: Any,
    max_level: int = 20,
    tolerance: float = 1e-5,
    *,
    nearest: bool = False,
) -> tuple[int, int, float]:
    """Snap ``angle`` to the dyadic lattice ``coeff * pi / 2**(level - 1)``."""
    if max_level < 1:
        raise ValueError(f"max_level must be positive, received {max_level}.")
    tolerance = _validated_rz_tolerance(tolerance)

    try:
        value = float(angle)
    except Exception as exc:  # pragma: no cover - depends on optional qiskit parameter types
        raise ValueError(
            f"Unsupported rz angle {angle!r}. A numeric value is required."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"Unsupported rz angle {angle!r}. Finite numeric values are required.")

    best_error = math.inf
    best_coeff = 0
    best_level = 1

    for level in range(1, max_level + 1):
        denom = 1 << (level - 1)
        k = int(round(value * denom / math.pi))
        reconstructed = k * math.pi / denom
        error = abs(value - reconstructed)
        coeff = k % (1 << level)
        if error < best_error:
            best_error = error
            best_coeff = coeff
            best_level = level
        if not nearest and error <= tolerance:
            return coeff, level, error

    if nearest:
        return best_coeff, best_level, best_error

    raise ValueError(
        f"Only dyadic multiples of pi are supported within tolerance {tolerance:.3e}. "
        f"Nearest dyadic: level={best_level}, coeff={best_coeff}, error={best_error:.3e}."
    )


def _evaluate_qasm_angle_expr(expr: str) -> float:
    """Safely evaluate a simple OpenQASM numeric angle expression."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Unsupported rz angle expression {expr!r}.") from exc

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id.lower() == "pi":
            return math.pi
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = eval_node(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0.0:
                raise ValueError("Division by zero in rz angle expression.")
            return left / right
        raise ValueError(f"Unsupported rz angle expression {expr!r}.")

    value = eval_node(tree)
    if not math.isfinite(value):
        raise ValueError(f"Unsupported rz angle expression {expr!r}.")
    return value


def _parse_dyadic_pi_expr(expr: str, *, tolerance: float = 1e-5) -> tuple[int, int]:
    try:
        value = _evaluate_qasm_angle_expr(expr)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported rz angle {expr!r}. Only numeric expressions over pi are supported."
        ) from exc

    try:
        coeff, precision_level, _ = dyadic_snap(value, tolerance=tolerance)
    except ValueError as exc:
        raise ValueError(f"Unsupported rz angle {expr!r}. {exc}") from exc
    return coeff, precision_level


def _dyadic_phase_from_qiskit_angle(angle: Any, *, tolerance: float = 1e-5) -> tuple[int, int]:
    try:
        coeff, precision_level, _ = dyadic_snap(angle, tolerance=tolerance)
    except ValueError as exc:
        raise ValueError(f"Unsupported Qiskit rz angle {angle!r}. {exc}") from exc
    return coeff, precision_level


def _dyadic_phase_to_angle(coeff: int, precision_level: int) -> float:
    modulus = 1 << precision_level
    residue = coeff % modulus
    if residue > modulus // 2:
        residue -= modulus
    return math.pi * residue / (1 << (precision_level - 1))
