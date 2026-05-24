"""OpenQASM 2 parsing and lowering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..circuit_spec import (
    CircuitSpec,
    Gate,
    _QASM_GATE_MAP,
    _QASM_QUBIT,
    _QASM_SUPPORTED_GATE_SET,
    _RZ_COMPILE_MODE_DYADIC,
    _TEMP_PHASE_GATE,
    _metadata_with_import_stats,
    _normalize_global_phase_radians,
    _normalize_rz_compile_mode,
    _validated_rz_tolerance,
    make_circuit,
)
from .angles import (
    _ImportCompileStats,
    _compile_import_gate_sequence,
    _evaluate_qasm_angle_expr,
    _exact_dyadic_phase_from_angle,
    _exact_phase_gate_from_angle,
)
from .qiskit_import import (
    _fast_import_gate_sequence_if_supported,
    _qiskit_operation_to_raw_gates,
    _qiskit_u_gate,
)

@dataclass(frozen=True, slots=True)
class _OpenQasm2Statement:
    text: str
    line: int

def parse_openqasm2(
    source: str,
    name: str | None = None,
    *,
    rz_tolerance: float = 1e-5,
    rz_compile_mode: str | None = _RZ_COMPILE_MODE_DYADIC,
) -> CircuitSpec:
    """Parse the supported OpenQASM 2 subset into a ``CircuitSpec``."""
    qregs: dict[str, int] = {}
    offsets: dict[str, int] = {}
    raw_gates: list[Gate] = []
    n_qubits = 0
    global_phase_radians = 0.0
    rz_tolerance = _validated_rz_tolerance(rz_tolerance)
    compile_mode = _normalize_rz_compile_mode(rz_compile_mode)

    statements = _openqasm2_unitary_statements(tuple(_iter_openqasm2_statements(source)))
    for statement in statements:
        line = statement.text
        if line.lower().startswith("openqasm") or line.lower().startswith("include"):
            continue
        if line.lower().startswith("qreg "):
            reg_decl = line[5:].strip()
            match = _QASM_QUBIT.fullmatch(reg_decl)
            if match is None:
                raise ValueError(f"Unsupported qreg declaration on line {statement.line}: {line!r}")
            reg_name = match.group(1)
            size = int(match.group(2))
            qregs[reg_name] = size
            offsets[reg_name] = n_qubits
            n_qubits += size
            continue
        if line.lower().startswith(("creg ", "barrier ")):
            continue

        gate_token_raw, operand_text = _split_qasm_gate_statement(line)
        if not operand_text:
            raise ValueError(f"Unsupported OpenQASM statement on line {statement.line}: {line!r}")
        gate_token = gate_token_raw.lower()
        gate_name = _QASM_GATE_MAP.get(gate_token)
        gate_angle_expr: str | None = None
        if gate_name is None and gate_token.startswith("rz(") and gate_token.endswith(")"):
            gate_name = "rz"
            gate_angle_expr = gate_token[3:-1]
        if gate_name is None and gate_token.startswith("rx(") and gate_token.endswith(")"):
            gate_name = "rx"
            gate_angle_expr = gate_token[3:-1]
        if gate_name is None and gate_token.startswith("rzz(") and gate_token.endswith(")"):
            gate_name = "rzz"
            gate_angle_expr = gate_token[4:-1]
        if gate_name is None and gate_token.startswith("u3(") and gate_token.endswith(")"):
            angle_parts = gate_token[3:-1].split(",")
            if len(angle_parts) != 3:
                raise ValueError(f"OpenQASM u3 gate expects three angle parameters: {gate_token_raw!r}.")
            try:
                theta = _evaluate_qasm_angle_expr(angle_parts[0].strip())
                phi = _evaluate_qasm_angle_expr(angle_parts[1].strip())
                lam = _evaluate_qasm_angle_expr(angle_parts[2].strip())
            except ValueError as exc:
                raise ValueError(f"Unsupported u3 angle in {gate_token_raw!r}.") from exc
            qubit_tokens = [t.strip() for t in operand_text.split(",")]
            if len(qubit_tokens) != 1:
                raise ValueError(f"OpenQASM u3 gate expects one qubit, got {len(qubit_tokens)}.")
            qubit = _parse_qasm_qubit(qubit_tokens[0], offsets, qregs)
            op_gates, op_phase = _qiskit_operation_to_raw_gates(
                _qiskit_u_gate(theta, phi, lam),
                [qubit],
                compile_mode=compile_mode,
                tolerance=rz_tolerance,
            )
            raw_gates.extend(op_gates)
            global_phase_radians = _normalize_global_phase_radians(
                global_phase_radians + op_phase
            )
            continue
        if gate_name is None:
            raise ValueError(
                f"Unsupported OpenQASM gate on line {statement.line}: {gate_token_raw!r}. "
                f"TerKet supports only Clifford+T gates {_QASM_SUPPORTED_GATE_SET}. "
                "Consider transpiling to this gate set first."
            )
        operands = [_parse_qasm_qubit(token.strip(), offsets, qregs) for token in operand_text.split(",")]
        if gate_name == "rz":
            if len(operands) != 1:
                raise ValueError(f"OpenQASM gate {gate_token_raw!r} expects one qubit.")
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
            global_phase_radians = _normalize_global_phase_radians(
                global_phase_radians - 0.5 * angle
            )
        elif gate_name == "rx":
            if len(operands) != 1:
                raise ValueError(f"OpenQASM gate {gate_token_raw!r} expects one qubit.")
            if gate_angle_expr is None:  # pragma: no cover - internal guard
                raise ValueError("Missing OpenQASM rx angle.")
            try:
                angle = _evaluate_qasm_angle_expr(gate_angle_expr)
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported rx angle {gate_angle_expr!r}. Only numeric expressions over pi are supported."
                ) from exc
            raw_gates.append(("h", operands[0]))
            if compile_mode == _RZ_COMPILE_MODE_DYADIC:
                phase_gate, exact_angle = _exact_phase_gate_from_angle(
                    angle,
                    operands[0],
                    source=f"Unsupported rx angle {gate_angle_expr!r}",
                )
                if phase_gate is not None:
                    raw_gates.append(phase_gate)
            else:
                raw_gates.append((_TEMP_PHASE_GATE, operands[0], angle))
            raw_gates.append(("h", operands[0]))
            global_phase_radians = _normalize_global_phase_radians(
                global_phase_radians - 0.5 * angle
            )
        elif gate_name == "rzz":
            if len(operands) != 2:
                raise ValueError(f"OpenQASM gate {gate_token_raw!r} expects two qubits.")
            if gate_angle_expr is None:  # pragma: no cover - internal guard
                raise ValueError("Missing OpenQASM rzz angle.")
            try:
                angle = _evaluate_qasm_angle_expr(gate_angle_expr)
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported rzz angle {gate_angle_expr!r}. Only numeric expressions over pi are supported."
                ) from exc
            if compile_mode == _RZ_COMPILE_MODE_DYADIC:
                exact = _exact_dyadic_phase_from_angle(angle)
                if exact is not None:
                    coeff, precision_level = exact
                    raw_gates.append(("rzz_dyadic", operands[0], operands[1], coeff, precision_level))
                else:
                    raw_gates.append(("cnot", operands[0], operands[1]))
                    raw_gates.append((_TEMP_PHASE_GATE, operands[1], angle))
                    raw_gates.append(("cnot", operands[0], operands[1]))
            else:
                raw_gates.append(("cnot", operands[0], operands[1]))
                raw_gates.append((_TEMP_PHASE_GATE, operands[1], angle))
                raw_gates.append(("cnot", operands[0], operands[1]))
            global_phase_radians = _normalize_global_phase_radians(
                global_phase_radians - 0.5 * angle
            )
        else:
            raw_gates.append((gate_name, *operands))

    fast_import = _fast_import_gate_sequence_if_supported(raw_gates)
    if fast_import is None:
        compiled_gates, compile_stats = _compile_import_gate_sequence(
            raw_gates,
            tolerance=rz_tolerance,
            compile_mode=compile_mode,
        )
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
        metadata=_metadata_with_import_stats(
            global_phase_radians,
            compile_stats,
            compile_mode=compile_mode,
            tolerance=rz_tolerance,
        ),
    )

def _strip_qasm_line_comment(line: str) -> str:
    in_string = False
    escaped = False
    for idx, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string and char == "/" and idx + 1 < len(line) and line[idx + 1] == "/":
            return line[:idx]
    return line

def _iter_openqasm2_statements(source: str) -> Iterable[_OpenQasm2Statement]:
    pending: list[str] = []
    start_line = 1
    for line_no, raw_line in enumerate(source.splitlines(), start=1):
        line = _strip_qasm_line_comment(raw_line)
        if not pending and not line.strip():
            continue
        if not pending:
            start_line = line_no

        start = 0
        for idx, char in enumerate(line):
            if char != ";":
                continue
            pending.append(line[start:idx])
            text = " ".join(part.strip() for part in pending if part.strip()).strip()
            if text:
                yield _OpenQasm2Statement(text=text, line=start_line)
            pending = []
            start = idx + 1
            start_line = line_no
        tail = line[start:].strip()
        if tail:
            pending.append(tail)

    text = " ".join(part.strip() for part in pending if part.strip()).strip()
    if text:
        raise ValueError(f"OpenQASM statement starting on line {start_line} is missing a terminating ';'.")

def _openqasm2_unitary_statements(
    statements: Sequence[_OpenQasm2Statement],
) -> tuple[_OpenQasm2Statement, ...]:
    end = len(statements)
    removed_measurement = False
    while end > 0:
        lower = statements[end - 1].text.lower()
        if lower.startswith("measure "):
            removed_measurement = True
            end -= 1
            continue
        if removed_measurement and lower.startswith("barrier "):
            end -= 1
            continue
        break

    kept = tuple(statements[:end])
    for idx, statement in enumerate(kept):
        lower = statement.text.lower()
        if lower.startswith("measure "):
            raise ValueError(
                "Unsupported OpenQASM 2 circuit: mid-circuit measurement at statement "
                f"{idx} on line {statement.line}. TerKet import supports only unitary "
                "circuits plus optional trailing measurements."
            )
        if lower.startswith("reset "):
            raise ValueError(
                "Unsupported OpenQASM 2 circuit: reset at statement "
                f"{idx} on line {statement.line}. TerKet import supports only unitary "
                "circuits plus optional trailing measurements."
            )
    return kept

def _split_qasm_gate_statement(line: str) -> tuple[str, str]:
    depth = 0
    for idx, char in enumerate(line):
        if char == "(":
            depth += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            continue
        if char.isspace() and depth == 0:
            return line[:idx], line[idx:].strip()
    return line, ""

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
