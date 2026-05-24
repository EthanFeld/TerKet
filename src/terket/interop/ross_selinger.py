"""Ross-Selinger synthesis helpers for circuit interop."""

from __future__ import annotations

import cmath
import math
from typing import Any, Sequence

from ..circuit_spec import (
    Gate,
    _ROSS_SELINGER_SUBPROCESS_ONLY,
    _coerce_finite_radians,
    _normalize_global_phase_radians,
)
from .angles import (
    _evaluate_qasm_angle_expr,
    _exact_dyadic_phase_from_angle,
    _matrix_from_key,
    _unitary_key,
    _u3_matrix,
)
from .rewrite import _emit_dyadic_phase_gate, _normalize_gate, _rewrite_gate_sequence

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
    def _run_subprocess(a: float):
        return subprocess.run(
            [sys.executable, "-c", script, kind, repr(a), repr(epsilon)],
            capture_output=True,
            text=True,
            check=False,
        )

    completed = _run_subprocess(angle)
    if completed.returncode != 0 and "panicked" in (completed.stderr or ""):
        # rsgridsynth sometimes aborts on degenerate angles; retry slightly nudged.
        completed = _run_subprocess(angle + 1e-10)
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

def _compile_u3_gate_via_psx(
    theta: float,
    phi: float,
    lam: float,
    qubit: int,
    *,
    tolerance: float,
) -> tuple[tuple[Gate, ...], float]:
    compiled: list[Gate] = []
    global_phase = 0.0
    for rz_angle, add_sx in [
        (lam - math.pi / 2, True),
        (math.pi - theta, True),
        (phi + math.pi / 2, False),
    ]:
        phase_gates, gate_phase, _is_exact_dyadic, _angle_error = _compile_phase_gate(
            rz_angle,
            qubit,
            tolerance=tolerance,
            source="Unsupported u3 angle",
        )
        compiled.extend(phase_gates)
        global_phase = _normalize_global_phase_radians(global_phase + gate_phase)
        if add_sx:
            compiled.append(("sx", qubit))
    return _rewrite_gate_sequence(tuple(compiled)), global_phase

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

def _compile_u3_gate(
    theta: float,
    phi: float,
    lam: float,
    qubit: int,
    *,
    tolerance: float,
) -> tuple[tuple[Gate, ...], float]:
    try:
        template_gates, template_phase = _ross_selinger_unitary_template(
            _unitary_key(_u3_matrix(theta, phi, lam)),
            tolerance,
        )
        return _retarget_single_qubit_gates(template_gates, qubit), template_phase
    except RuntimeError:
        return _compile_u3_gate_via_psx(theta, phi, lam, qubit, tolerance=tolerance)
