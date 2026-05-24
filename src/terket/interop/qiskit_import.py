"""Qiskit import helpers and lowering tables."""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

from ..circuit_spec import (
    CircuitSpec,
    Gate,
    SUPPORTED_GATES,
    _QASM_GATE_MAP,
    _RZ_COMPILE_MODE_CLIFFORD_T,
    _RZ_COMPILE_MODE_DYADIC,
    _TEMP_PHASE_GATE,
    _coerce_finite_radians,
    _metadata_with_import_stats,
    _normalize_global_phase_radians,
    _normalize_rz_compile_mode,
    _validated_rz_tolerance,
    make_circuit,
)
from .angles import (
    _FAST_IMPORT_GATE_COUNT_THRESHOLD,
    _ImportCompileStats,
    _compile_import_gate_sequence,
    _exact_dyadic_phase_from_angle,
    _exact_phase_gate_from_angle,
)
from .rewrite import _normalize_gate, _rewrite_gate_sequence_local
from .ross_selinger import _compile_u3_gate

_QiskitRawTemplate = tuple[tuple[Gate, ...], float]

_QiskitOperationTemplateCache = dict[tuple[object, ...], _QiskitRawTemplate]
_FAST_IMPORT_NATIVE_GATES = frozenset(SUPPORTED_GATES)

def from_qiskit(
    circuit: Any,
    *,
    rz_tolerance: float = 1e-5,
    rz_compile_mode: str | None = _RZ_COMPILE_MODE_DYADIC,
) -> CircuitSpec:
    """Convert a Qiskit circuit into a normalized ``CircuitSpec``."""
    raw_gates: list[Gate] = []
    rz_tolerance = _validated_rz_tolerance(rz_tolerance)
    compile_mode = _normalize_rz_compile_mode(rz_compile_mode)
    global_phase_radians = _normalize_global_phase_radians(
        _coerce_finite_radians(
            getattr(circuit, "global_phase", 0.0),
            source="Unsupported Qiskit circuit global phase",
        )
    )
    qubit_indices = {qubit: idx for idx, qubit in enumerate(circuit.qubits)}
    template_cache: _QiskitOperationTemplateCache = {}
    for instruction in _qiskit_unitary_data(circuit):
        qubits = [qubit_indices[qubit] for qubit in instruction.qubits]
        op_gates, op_phase = _qiskit_operation_to_raw_gates(
            instruction.operation,
            qubits,
            compile_mode=compile_mode,
            tolerance=rz_tolerance,
            template_cache=template_cache,
        )
        raw_gates.extend(op_gates)
        global_phase_radians = _normalize_global_phase_radians(global_phase_radians + op_phase)
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
        circuit.num_qubits,
        compiled_gates,
        name=getattr(circuit, "name", None),
        metadata=_metadata_with_import_stats(
            global_phase_radians,
            compile_stats,
            compile_mode=compile_mode,
            tolerance=rz_tolerance,
        ),
    )

def _qiskit_unitary_data(circuit: Any) -> Sequence[Any]:
    data = circuit.data
    end = len(data)
    removed_measurement = False
    while end > 0:
        name = data[end - 1].operation.name.lower()
        if name == "measure":
            removed_measurement = True
            end -= 1
            continue
        if removed_measurement and name == "barrier":
            end -= 1
            continue
        break
    kept = data[:end]
    for idx, instruction in enumerate(kept):
        name = instruction.operation.name.lower()
        if name == "measure":
            raise ValueError(
                "Unsupported Qiskit circuit: mid-circuit measurement at instruction "
                f"{idx}. TerKet import supports only unitary circuits plus optional "
                "trailing measurements."
            )
        if name == "reset":
            raise ValueError(
                "Unsupported Qiskit circuit: reset at instruction "
                f"{idx}. TerKet import supports only unitary circuits plus optional "
                "trailing measurements."
            )
        if name == "if_else" or getattr(instruction.operation, "condition", None) is not None or instruction.clbits:
            raise ValueError(
                "Unsupported Qiskit circuit: classical control at instruction "
                f"{idx} ({instruction.operation.name!r}). TerKet import supports only "
                "unitary circuits plus optional trailing measurements."
            )
    return kept

def _retarget_qiskit_raw_gate_template(
    template_gates: Sequence[Gate],
    qubits: Sequence[int],
) -> list[Gate]:
    retargeted: list[Gate] = []
    for gate in template_gates:
        name = gate[0]
        if name == "rzz_dyadic":
            retargeted.append(
                ("rzz_dyadic", int(qubits[int(gate[1])]), int(qubits[int(gate[2])]), int(gate[3]), int(gate[4]))
            )
            continue
        if name in {"cnot", "cz"}:
            retargeted.append((name, int(qubits[int(gate[1])]), int(qubits[int(gate[2])])))
            continue

        qubit = int(qubits[int(gate[1])])
        if name == "rz_dyadic":
            retargeted.append(("rz_dyadic", qubit, int(gate[2]), int(gate[3])))
        elif name in {"rz_arbitrary", _TEMP_PHASE_GATE}:
            retargeted.append((name, qubit, gate[2]))
        else:
            retargeted.append((name, qubit))
    return retargeted

def _compile_qiskit_circuit_template(
    circuit: Any,
    *,
    compile_mode: str,
    tolerance: float,
    template_cache: _QiskitOperationTemplateCache,
) -> _QiskitRawTemplate:
    return _qiskit_circuit_to_raw_gates(
        circuit,
        qubits=tuple(range(len(circuit.qubits))),
        compile_mode=compile_mode,
        tolerance=tolerance,
        template_cache=template_cache,
    )

def _qiskit_cached_circuit_template(
    cache_key: tuple[object, ...],
    circuit: Any,
    *,
    compile_mode: str,
    tolerance: float,
    template_cache: _QiskitOperationTemplateCache | None,
) -> _QiskitRawTemplate:
    if template_cache is None:
        return _compile_qiskit_circuit_template(
            circuit,
            compile_mode=compile_mode,
            tolerance=tolerance,
            template_cache={},
        )

    cached = template_cache.get(cache_key)
    if cached is not None:
        return cached

    cached = _compile_qiskit_circuit_template(
        circuit,
        compile_mode=compile_mode,
        tolerance=tolerance,
        template_cache=template_cache,
    )
    template_cache[cache_key] = cached
    return cached

def _phase_gate_raw_gates(
    angle: Any,
    qubit: int,
    *,
    compile_mode: str,
    source: str,
    rz_global_phase: bool,
) -> tuple[list[Gate], float]:
    value = _coerce_finite_radians(angle, source=source)
    if compile_mode == _RZ_COMPILE_MODE_DYADIC:
        phase_gate, exact_angle = _exact_phase_gate_from_angle(
            value,
            qubit,
            source=source,
        )
        global_phase = -0.5 * exact_angle if rz_global_phase else 0.0
        return ([] if phase_gate is None else [phase_gate]), _normalize_global_phase_radians(global_phase)
    global_phase = -0.5 * value if rz_global_phase else 0.0
    return [(_TEMP_PHASE_GATE, qubit, value)], _normalize_global_phase_radians(global_phase)

def _extend_phase_gate(
    raw_gates: list[Gate],
    angle: Any,
    qubit: int,
    *,
    compile_mode: str,
    source: str,
    rz_global_phase: bool,
) -> float:
    gates, phase = _phase_gate_raw_gates(
        angle,
        qubit,
        compile_mode=compile_mode,
        source=source,
        rz_global_phase=rz_global_phase,
    )
    raw_gates.extend(gates)
    return phase

def _controlled_phase_raw_gates(
    angle: Any,
    control: int,
    target: int,
    *,
    compile_mode: str,
    source: str,
) -> tuple[list[Gate], float]:
    value = _coerce_finite_radians(angle, source=source)
    half = 0.5 * value
    raw_gates: list[Gate] = []
    phase = 0.0
    phase += _extend_phase_gate(
        raw_gates,
        half,
        control,
        compile_mode=compile_mode,
        source=source,
        rz_global_phase=False,
    )
    raw_gates.append(("cnot", control, target))
    phase += _extend_phase_gate(
        raw_gates,
        -half,
        target,
        compile_mode=compile_mode,
        source=source,
        rz_global_phase=False,
    )
    raw_gates.append(("cnot", control, target))
    phase += _extend_phase_gate(
        raw_gates,
        half,
        target,
        compile_mode=compile_mode,
        source=source,
        rz_global_phase=False,
    )
    return raw_gates, _normalize_global_phase_radians(phase)

def _controlled_rz_raw_gates(
    angle: Any,
    control: int,
    target: int,
    *,
    compile_mode: str,
    source: str,
) -> tuple[list[Gate], float]:
    value = _coerce_finite_radians(angle, source=source)
    half = 0.5 * value
    raw_gates: list[Gate] = []
    phase = 0.0
    phase += _extend_phase_gate(
        raw_gates,
        half,
        target,
        compile_mode=compile_mode,
        source=source,
        rz_global_phase=True,
    )
    raw_gates.append(("cnot", control, target))
    phase += _extend_phase_gate(
        raw_gates,
        -half,
        target,
        compile_mode=compile_mode,
        source=source,
        rz_global_phase=True,
    )
    raw_gates.append(("cnot", control, target))
    return raw_gates, _normalize_global_phase_radians(phase)

def _mcphase_2_control_raw_gates(
    angle: Any,
    control0: int,
    control1: int,
    target: int,
    *,
    compile_mode: str,
    source: str,
) -> tuple[list[Gate], float]:
    value = _coerce_finite_radians(angle, source=source)
    quarter = 0.25 * value
    raw_gates: list[Gate] = []
    phase = 0.0
    raw_gates.append(("cnot", control0, target))
    phase += _extend_phase_gate(
        raw_gates,
        -quarter,
        target,
        compile_mode=compile_mode,
        source=source,
        rz_global_phase=True,
    )
    raw_gates.append(("cnot", control1, target))
    phase += _extend_phase_gate(
        raw_gates,
        quarter,
        target,
        compile_mode=compile_mode,
        source=source,
        rz_global_phase=True,
    )
    raw_gates.append(("cnot", control0, target))
    phase += _extend_phase_gate(
        raw_gates,
        -quarter,
        target,
        compile_mode=compile_mode,
        source=source,
        rz_global_phase=True,
    )
    raw_gates.append(("cnot", control1, target))
    phase += _extend_phase_gate(
        raw_gates,
        quarter,
        target,
        compile_mode=compile_mode,
        source=source,
        rz_global_phase=True,
    )
    crz_gates, crz_phase = _controlled_rz_raw_gates(
        0.5 * value,
        control0,
        control1,
        compile_mode=compile_mode,
        source=source,
    )
    raw_gates.extend(crz_gates)
    phase += crz_phase
    phase += _extend_phase_gate(
        raw_gates,
        quarter,
        control0,
        compile_mode=compile_mode,
        source=source,
        rz_global_phase=False,
    )
    return raw_gates, _normalize_global_phase_radians(phase)

_QiskitOperationHandler = Callable[
    [Any, Sequence[int], str, float, _QiskitOperationTemplateCache | None],
    tuple[list[Gate], float],
]

def _qiskit_direct_native_raw_gates(
    operation: Any,
    qubits: Sequence[int],
    compile_mode: str,
    tolerance: float,
    template_cache: _QiskitOperationTemplateCache | None,
) -> tuple[list[Gate], float]:
    del compile_mode, tolerance, template_cache
    op_name = operation.name.lower()
    name = _QASM_GATE_MAP.get(op_name)
    if name is None:  # pragma: no cover - registry guard
        raise ValueError(f"Unsupported direct Qiskit gate {operation.name!r}.")
    return [(name, *qubits)], 0.0

def _qiskit_ignored_raw_gates(
    operation: Any,
    qubits: Sequence[int],
    compile_mode: str,
    tolerance: float,
    template_cache: _QiskitOperationTemplateCache | None,
) -> tuple[list[Gate], float]:
    del operation, qubits, compile_mode, tolerance, template_cache
    return [], 0.0


def _require_qiskit_arity(operation: Any, qubits: Sequence[int], expected: int) -> None:
    if len(qubits) != expected:
        raise ValueError(f"Unsupported Qiskit gate arity for {operation.name!r}.")

def _qiskit_sx_raw_gates(
    operation: Any,
    qubits: Sequence[int],
    compile_mode: str,
    tolerance: float,
    template_cache: _QiskitOperationTemplateCache | None,
) -> tuple[list[Gate], float]:
    del compile_mode, tolerance, template_cache
    _require_qiskit_arity(operation, qubits, 1)
    return [("sx", qubits[0])], 0.0

def _qiskit_rz_raw_gates(
    operation: Any,
    qubits: Sequence[int],
    compile_mode: str,
    tolerance: float,
    template_cache: _QiskitOperationTemplateCache | None,
) -> tuple[list[Gate], float]:
    del tolerance, template_cache
    _require_qiskit_arity(operation, qubits, 1)
    return _phase_gate_raw_gates(
        operation.params[0],
        qubits[0],
        compile_mode=compile_mode,
        source=f"Unsupported Qiskit rz angle {operation.params[0]!r}",
        rz_global_phase=True,
    )

def _qiskit_phase_raw_gates(
    operation: Any,
    qubits: Sequence[int],
    compile_mode: str,
    tolerance: float,
    template_cache: _QiskitOperationTemplateCache | None,
) -> tuple[list[Gate], float]:
    del tolerance, template_cache
    _require_qiskit_arity(operation, qubits, 1)
    return _phase_gate_raw_gates(
        operation.params[0],
        qubits[0],
        compile_mode=compile_mode,
        source=f"Unsupported Qiskit phase angle {operation.params[0]!r}",
        rz_global_phase=False,
    )

def _qiskit_rx_raw_gates(
    operation: Any,
    qubits: Sequence[int],
    compile_mode: str,
    tolerance: float,
    template_cache: _QiskitOperationTemplateCache | None,
) -> tuple[list[Gate], float]:
    del tolerance, template_cache
    _require_qiskit_arity(operation, qubits, 1)
    raw_gates: list[Gate] = [("h", qubits[0])]
    phase = _extend_phase_gate(
        raw_gates,
        operation.params[0],
        qubits[0],
        compile_mode=compile_mode,
        source=f"Unsupported Qiskit rx angle {operation.params[0]!r}",
        rz_global_phase=True,
    )
    raw_gates.append(("h", qubits[0]))
    return raw_gates, phase

def _qiskit_controlled_phase_raw_gates(
    operation: Any,
    qubits: Sequence[int],
    compile_mode: str,
    tolerance: float,
    template_cache: _QiskitOperationTemplateCache | None,
) -> tuple[list[Gate], float]:
    del tolerance, template_cache
    _require_qiskit_arity(operation, qubits, 2)
    return _controlled_phase_raw_gates(
        operation.params[0],
        qubits[0],
        qubits[1],
        compile_mode=compile_mode,
        source=f"Unsupported Qiskit controlled-phase angle {operation.params[0]!r}",
    )

def _qiskit_crz_raw_gates(
    operation: Any,
    qubits: Sequence[int],
    compile_mode: str,
    tolerance: float,
    template_cache: _QiskitOperationTemplateCache | None,
) -> tuple[list[Gate], float]:
    del tolerance, template_cache
    _require_qiskit_arity(operation, qubits, 2)
    return _controlled_rz_raw_gates(
        operation.params[0],
        qubits[0],
        qubits[1],
        compile_mode=compile_mode,
        source=f"Unsupported Qiskit crz angle {operation.params[0]!r}",
    )

def _qiskit_swap_raw_gates(
    operation: Any,
    qubits: Sequence[int],
    compile_mode: str,
    tolerance: float,
    template_cache: _QiskitOperationTemplateCache | None,
) -> tuple[list[Gate], float]:
    del compile_mode, tolerance, template_cache
    _require_qiskit_arity(operation, qubits, 2)
    return [
        ("cnot", qubits[0], qubits[1]),
        ("cnot", qubits[1], qubits[0]),
        ("cnot", qubits[0], qubits[1]),
    ], 0.0

def _qiskit_mcphase_raw_gates(
    operation: Any,
    qubits: Sequence[int],
    compile_mode: str,
    tolerance: float,
    template_cache: _QiskitOperationTemplateCache | None,
) -> tuple[list[Gate], float]:
    if len(qubits) != 3:
        return _qiskit_definition_or_synthesis_raw_gates(operation, qubits, compile_mode, tolerance, template_cache)
    return _mcphase_2_control_raw_gates(
        operation.params[0],
        qubits[0],
        qubits[1],
        qubits[2],
        compile_mode=compile_mode,
        source=f"Unsupported Qiskit mcphase angle {operation.params[0]!r}",
    )

def _qiskit_u_raw_gates(
    operation: Any,
    qubits: Sequence[int],
    compile_mode: str,
    tolerance: float,
    template_cache: _QiskitOperationTemplateCache | None,
) -> tuple[list[Gate], float]:
    op_name = operation.name.lower()
    _require_qiskit_arity(operation, qubits, 1)
    if compile_mode == _RZ_COMPILE_MODE_CLIFFORD_T:
        if op_name == "u2":
            theta = math.pi / 2.0
            phi = _coerce_finite_radians(
                operation.params[0],
                source=f"Unsupported Qiskit u2 angle {operation.params[0]!r}",
            )
            lam = _coerce_finite_radians(
                operation.params[1],
                source=f"Unsupported Qiskit u2 angle {operation.params[1]!r}",
            )
        else:
            theta = _coerce_finite_radians(
                operation.params[0],
                source=f"Unsupported Qiskit {op_name} angle {operation.params[0]!r}",
            )
            phi = _coerce_finite_radians(
                operation.params[1],
                source=f"Unsupported Qiskit {op_name} angle {operation.params[1]!r}",
            )
            lam = _coerce_finite_radians(
                operation.params[2],
                source=f"Unsupported Qiskit {op_name} angle {operation.params[2]!r}",
            )
        compiled_gates, compiled_phase = _compile_u3_gate(
            theta,
            phi,
            lam,
            qubits[0],
            tolerance=tolerance,
        )
        return list(compiled_gates), compiled_phase
    template_gates, template_phase = _qiskit_cached_circuit_template(
        ("qiskit_psx_decompose", id(operation), compile_mode, tolerance),
        _qiskit_one_qubit_psx_decomposer()(operation),
        compile_mode=compile_mode,
        tolerance=tolerance,
        template_cache=template_cache,
    )
    return _retarget_qiskit_raw_gate_template(template_gates, qubits), template_phase

def _qiskit_rzz_raw_gates(
    operation: Any,
    qubits: Sequence[int],
    compile_mode: str,
    tolerance: float,
    template_cache: _QiskitOperationTemplateCache | None,
) -> tuple[list[Gate], float]:
    del tolerance, template_cache
    _require_qiskit_arity(operation, qubits, 2)
    angle = _coerce_finite_radians(
        operation.params[0],
        source=f"Unsupported Qiskit rzz angle {operation.params[0]!r}",
    )
    phase = _normalize_global_phase_radians(-0.5 * angle)
    if compile_mode == _RZ_COMPILE_MODE_DYADIC:
        exact = _exact_dyadic_phase_from_angle(angle)
        if exact is not None:
            coeff, precision_level = exact
            return [("rzz_dyadic", qubits[0], qubits[1], coeff, precision_level)], phase
    return [
        ("cnot", qubits[0], qubits[1]),
        (_TEMP_PHASE_GATE, qubits[1], angle),
        ("cnot", qubits[0], qubits[1]),
    ], phase

def _qiskit_definition_or_synthesis_raw_gates(
    operation: Any,
    qubits: Sequence[int],
    compile_mode: str,
    tolerance: float,
    template_cache: _QiskitOperationTemplateCache | None,
) -> tuple[list[Gate], float]:
    definition = getattr(operation, "definition", None)
    if definition is not None:
        template_gates, template_phase = _qiskit_cached_circuit_template(
            ("qiskit_definition", id(definition), compile_mode, tolerance),
            definition,
            compile_mode=compile_mode,
            tolerance=tolerance,
            template_cache=template_cache,
        )
        return _retarget_qiskit_raw_gate_template(template_gates, qubits), template_phase

    synthesized = _synthesize_qiskit_operation(operation, len(qubits))
    template_gates, template_phase = _qiskit_cached_circuit_template(
        ("qiskit_synthesized", id(operation), compile_mode, tolerance),
        synthesized,
        compile_mode=compile_mode,
        tolerance=tolerance,
        template_cache=template_cache,
    )
    return _retarget_qiskit_raw_gate_template(template_gates, qubits), template_phase

_QISKIT_OPERATION_HANDLERS: dict[str, _QiskitOperationHandler] = {
    **{name: _qiskit_direct_native_raw_gates for name in _QASM_GATE_MAP},
    "barrier": _qiskit_ignored_raw_gates,
    "delay": _qiskit_ignored_raw_gates,
    "id": _qiskit_ignored_raw_gates,
    "sx": _qiskit_sx_raw_gates,
    "rz": _qiskit_rz_raw_gates,
    "p": _qiskit_phase_raw_gates,
    "u1": _qiskit_phase_raw_gates,
    "rx": _qiskit_rx_raw_gates,
    "cp": _qiskit_controlled_phase_raw_gates,
    "cu1": _qiskit_controlled_phase_raw_gates,
    "crz": _qiskit_crz_raw_gates,
    "swap": _qiskit_swap_raw_gates,
    "mcphase": _qiskit_mcphase_raw_gates,
    "mcp": _qiskit_mcphase_raw_gates,
    "u": _qiskit_u_raw_gates,
    "u2": _qiskit_u_raw_gates,
    "u3": _qiskit_u_raw_gates,
    "rzz": _qiskit_rzz_raw_gates,
}

def _qiskit_operation_to_raw_gates(
    operation: Any,
    qubits: Sequence[int],
    *,
    compile_mode: str,
    tolerance: float,
    template_cache: _QiskitOperationTemplateCache | None = None,
) -> tuple[list[Gate], float]:
    op_name = operation.name.lower()
    handler = _QISKIT_OPERATION_HANDLERS.get(op_name)
    if handler is not None:
        return handler(operation, qubits, compile_mode, tolerance, template_cache)
    return _qiskit_definition_or_synthesis_raw_gates(
        operation,
        qubits,
        compile_mode,
        tolerance,
        template_cache,
    )

def _qiskit_circuit_to_raw_gates(
    circuit: Any,
    *,
    qubits: Sequence[int],
    compile_mode: str,
    tolerance: float,
    template_cache: _QiskitOperationTemplateCache | None = None,
) -> tuple[list[Gate], float]:
    qubit_indices = {qubit: idx for idx, qubit in enumerate(circuit.qubits)}
    raw_gates: list[Gate] = []
    global_phase_radians = _normalize_global_phase_radians(
        _coerce_finite_radians(
            getattr(circuit, "global_phase", 0.0),
            source="Unsupported Qiskit circuit global phase",
        )
    )
    for instruction in circuit.data:
        mapped_qubits = [qubits[qubit_indices[qubit]] for qubit in instruction.qubits]
        op_gates, op_phase = _qiskit_operation_to_raw_gates(
            instruction.operation,
            mapped_qubits,
            compile_mode=compile_mode,
            tolerance=tolerance,
            template_cache=template_cache,
        )
        raw_gates.extend(op_gates)
        global_phase_radians = _normalize_global_phase_radians(global_phase_radians + op_phase)
    return raw_gates, global_phase_radians

def _synthesize_qiskit_operation(operation: Any, n_qubits: int):
    try:
        from qiskit import QuantumCircuit
        from qiskit.compiler import transpile
    except ImportError as exc:  # pragma: no cover - depends on optional qiskit install
        raise RuntimeError("Qiskit is required to synthesize unsupported operations.") from exc

    circuit = QuantumCircuit(n_qubits)
    circuit.append(operation, range(n_qubits))
    try:
        return transpile(
            circuit,
            basis_gates=["rz", "sx", "x", "cx", "cz"],
            optimization_level=0,
        )
    except Exception as exc:
        raise ValueError(f"Unsupported Qiskit gate: {operation.name!r}") from exc

def _qiskit_u_gate(theta: float, phi: float, lam: float):
    try:
        from qiskit.circuit.library import UGate
    except ImportError as exc:  # pragma: no cover - depends on optional qiskit install
        raise RuntimeError("Qiskit is required to import OpenQASM u3 gates.") from exc
    return UGate(theta, phi, lam)

def _qiskit_one_qubit_psx_decomposer():
    try:
        from qiskit.synthesis.one_qubit.one_qubit_decompose import OneQubitEulerDecomposer
    except ImportError as exc:  # pragma: no cover - depends on optional qiskit install
        raise RuntimeError("Qiskit is required to decompose unsupported single-qubit operations.") from exc

    return OneQubitEulerDecomposer("PSX")

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
    return _rewrite_gate_sequence_local(tuple(normalized))
