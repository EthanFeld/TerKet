"""Dyadic-angle parsing and import-compilation helpers."""

from __future__ import annotations

import ast
import cmath
from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np

from ..circuit_spec import (
    Gate,
    SUPPORTED_GATES,
    _EXACT_DYADIC_MAX_LEVEL,
    _EXACT_DYADIC_TOLERANCE,
    _RZ_COMPILE_MODE_APPROX_DYADIC,
    _TEMP_PHASE_GATE,
    _coerce_finite_radians,
    _normalize_global_phase_radians,
    _validated_rz_tolerance,
)
from .rewrite import (
    _emit_dyadic_phase_gate,
    _normalize_dyadic_phase,
    _normalize_gate,
    _normalize_phase_angle,
    _phase_angle_from_gate,
    _rewrite_gate_sequence,
    _rewrite_gate_sequence_local,
)

_FAST_IMPORT_NATIVE_GATES = frozenset(SUPPORTED_GATES)
_FAST_IMPORT_GATE_COUNT_THRESHOLD = 4096

@dataclass(slots=True)
class _ImportCompileStats:
    global_phase_radians: float = 0.0
    exact_dyadic_phase_count: int = 0
    approximated_phase_count: int = 0
    total_angle_error: float = 0.0
    max_angle_error: float = 0.0
    approximation_basis_size: int = 0
    approximation_run_count: int = 0
    total_run_fro_error: float = 0.0
    max_run_fro_error: float = 0.0

    def absorb(self, other: "_ImportCompileStats") -> None:
        self.global_phase_radians = _normalize_global_phase_radians(
            self.global_phase_radians + other.global_phase_radians
        )
        self.exact_dyadic_phase_count += other.exact_dyadic_phase_count
        self.approximated_phase_count += other.approximated_phase_count
        self.total_angle_error += other.total_angle_error
        self.max_angle_error = max(self.max_angle_error, other.max_angle_error)
        self.approximation_basis_size = max(self.approximation_basis_size, other.approximation_basis_size)
        self.approximation_run_count += other.approximation_run_count
        self.total_run_fro_error += other.total_run_fro_error
        self.max_run_fro_error = max(self.max_run_fro_error, other.max_run_fro_error)

@dataclass(frozen=True, slots=True)
class _ApproximationRunTask:
    qubit: int
    exact_unitary: tuple[complex, ...]
    skeleton: tuple[Gate, ...]
    global_phase_radians: float
    approximated_phase_count: int

@dataclass(frozen=True, slots=True)
class _ApproximationRunPlan:
    compiled_gates: tuple[Gate, ...]
    global_phase_radians: float
    approximated_phase_count: int
    total_angle_error: float
    max_angle_error: float
    run_fro_error: float

@dataclass(frozen=True, slots=True)
class _ApproximationAngleAssignment:
    snapped_angle: float
    angle_error: float

@dataclass(frozen=True, slots=True)
class _ApproximationClusterPlan:
    assignments: tuple[_ApproximationAngleAssignment, ...]
    basis_size: int

_ONE_QUBIT_IDENTITY = np.eye(2, dtype=complex)

_ONE_QUBIT_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)

_ONE_QUBIT_SX = np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=complex)

_ONE_QUBIT_SXDG = np.array([[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]], dtype=complex)

_ONE_QUBIT_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

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

def _u3_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    half_theta = 0.5 * theta
    cos_half = math.cos(half_theta)
    sin_half = math.sin(half_theta)
    phi_phase = cmath.exp(1j * phi)
    lam_phase = cmath.exp(1j * lam)
    return np.array(
        [
            [cos_half, -lam_phase * sin_half],
            [phi_phase * sin_half, phi_phase * lam_phase * cos_half],
        ],
        dtype=complex,
    )

def _one_qubit_run_unitary(run: Sequence[Gate]) -> np.ndarray:
    unitary = _ONE_QUBIT_IDENTITY.copy()
    for gate in run:
        unitary = _one_qubit_gate_matrix(gate) @ unitary
    return unitary

def _periodic_angle_distance(angle: float, other: float) -> float:
    return abs(_normalize_phase_angle(angle - other))

def _circular_mean(angles: Sequence[float]) -> float:
    if not angles:
        return 0.0
    vector = sum(cmath.exp(1j * angle) for angle in angles)
    if abs(vector) < 1e-15:
        return _normalize_phase_angle(float(angles[0]))
    return _normalize_phase_angle(cmath.phase(vector))

def _nearest_dyadic_phase(angle: float) -> tuple[float, float]:
    coeff, level, error = dyadic_snap(angle, max_level=_EXACT_DYADIC_MAX_LEVEL, nearest=True)
    return _normalize_phase_angle(_dyadic_phase_to_angle(coeff, level)), float(error)

def _phase_gates_from_snapped_angle(qubit: int, angle: float) -> tuple[Gate, ...]:
    gate, _ = _exact_phase_gate_from_angle(angle, qubit, source="Internal snapped dyadic angle")
    if gate is None:
        return ()
    return (gate,)


def _normalized_import_run(
    run: Sequence[Gate],
) -> tuple[tuple[Gate, ...], int, tuple[Gate, ...] | None, float | None]:
    normalized_run = tuple(
        gate if gate[0] == _TEMP_PHASE_GATE else _normalize_gate(gate)
        for gate in run
    )
    phase_gate_count = sum(1 for gate in normalized_run if gate[0] == _TEMP_PHASE_GATE)
    exact_run = _exact_single_qubit_run(normalized_run)
    if exact_run is not None:
        return normalized_run, phase_gate_count, exact_run, None
    if all(_phase_angle_from_gate(gate) is not None for gate in normalized_run):
        total_angle = 0.0
        for gate in normalized_run:
            phase_angle = _phase_angle_from_gate(gate)
            if phase_angle is None:  # pragma: no cover - guarded above
                raise ValueError(f"Unsupported diagonal gate {gate!r}.")
            total_angle = _normalize_phase_angle(total_angle + phase_angle)
        return normalized_run, phase_gate_count, None, total_angle
    return normalized_run, phase_gate_count, None, None

def _approximation_run_unitary(compiled_gates: Sequence[Gate], global_phase_radians: float) -> np.ndarray:
    return cmath.exp(1j * global_phase_radians) * _one_qubit_run_unitary(compiled_gates)

def _prepare_approximation_run_task(
    qubit: int,
    run: Sequence[Gate],
    *,
    tolerance: float,
) -> tuple[tuple[Gate, ...], _ImportCompileStats] | _ApproximationRunTask:
    from .qiskit_import import _qiskit_circuit_to_raw_gates, _qiskit_one_qubit_psx_decomposer

    if not run:
        return (), _ImportCompileStats()

    normalized_run, phase_gate_count, exact_run, total_angle = _normalized_import_run(run)
    if phase_gate_count == 0:
        return _rewrite_gate_sequence(normalized_run), _ImportCompileStats()
    if exact_run is not None:
        return exact_run, _ImportCompileStats(exact_dyadic_phase_count=phase_gate_count)
    if total_angle is not None:
        return _ApproximationRunTask(
            qubit=qubit,
            exact_unitary=_unitary_key(_phase_gate_matrix(total_angle)),
            skeleton=((_TEMP_PHASE_GATE, qubit, total_angle),),
            global_phase_radians=0.0,
            approximated_phase_count=1,
        )

    unitary = _one_qubit_run_unitary(normalized_run)
    decomposed = _qiskit_one_qubit_psx_decomposer()(unitary)
    raw_gates, phase = _qiskit_circuit_to_raw_gates(
        decomposed,
        qubits=(qubit,),
        compile_mode=_RZ_COMPILE_MODE_APPROX_DYADIC,
        tolerance=tolerance,
    )
    skeleton = tuple(
        gate if gate[0] == _TEMP_PHASE_GATE else _normalize_gate(gate)
        for gate in raw_gates
    )
    approximated_phase_count = sum(1 for gate in skeleton if gate[0] == _TEMP_PHASE_GATE)
    return _ApproximationRunTask(
        qubit=qubit,
        exact_unitary=_unitary_key(unitary),
        skeleton=skeleton,
        global_phase_radians=_normalize_global_phase_radians(float(phase)),
        approximated_phase_count=approximated_phase_count,
    )

def _cluster_approximation_angles(
    angles: Sequence[float],
    *,
    tolerance: float,
) -> _ApproximationClusterPlan:
    if not angles:
        return _ApproximationClusterPlan(assignments=(), basis_size=0)
    if tolerance <= 0.0:
        raise ValueError(
            "approx_dyadic mode requires positive rz_tolerance for non-dyadic single-qubit runs."
        )

    cluster_radius = 0.5 * tolerance
    clusters: list[list[int]] = []
    centers: list[float] = []

    for idx, angle in enumerate(angles):
        best_cluster = -1
        best_distance = math.inf
        for cluster_idx, center in enumerate(centers):
            distance = _periodic_angle_distance(angle, center)
            if distance <= cluster_radius and distance < best_distance:
                best_cluster = cluster_idx
                best_distance = distance
        if best_cluster < 0:
            clusters.append([idx])
            centers.append(angle)
        else:
            clusters[best_cluster].append(idx)
            centers[best_cluster] = _circular_mean([angles[item] for item in clusters[best_cluster]])

    assignments: list[_ApproximationAngleAssignment | None] = [None] * len(angles)
    pending_singletons: list[int] = []

    for cluster in clusters:
        center = _circular_mean([angles[idx] for idx in cluster])
        snapped_center, _center_error = _nearest_dyadic_phase(center)
        member_errors = [_periodic_angle_distance(angles[idx], snapped_center) for idx in cluster]
        if any(error > tolerance for error in member_errors):
            pending_singletons.extend(cluster)
            continue
        for idx, error in zip(cluster, member_errors):
            assignments[idx] = _ApproximationAngleAssignment(snapped_angle=snapped_center, angle_error=error)

    for idx in pending_singletons:
        snapped_angle, error = _nearest_dyadic_phase(angles[idx])
        if error > tolerance:
            raise ValueError(
                f"Nearest dyadic phase for angle {angles[idx]!r} exceeds tolerance {tolerance:.3e}: {error:.3e}."
            )
        assignments[idx] = _ApproximationAngleAssignment(snapped_angle=snapped_angle, angle_error=error)

    finalized = tuple(
        assignment if assignment is not None else _ApproximationAngleAssignment(0.0, 0.0)
        for assignment in assignments
    )
    basis_size = len({assignment.snapped_angle for assignment in finalized})
    return _ApproximationClusterPlan(assignments=finalized, basis_size=basis_size)

def _compile_approximation_runs(
    tasks: Sequence[_ApproximationRunTask],
    *,
    tolerance: float,
) -> tuple[tuple[tuple[Gate, ...], ...], _ImportCompileStats]:
    all_angles = tuple(
        float(gate[2])
        for task in tasks
        for gate in task.skeleton
        if gate[0] == _TEMP_PHASE_GATE
    )
    cluster_plan = _cluster_approximation_angles(all_angles, tolerance=tolerance)
    compiled_runs: list[tuple[Gate, ...]] = []
    stats = _ImportCompileStats(approximation_basis_size=cluster_plan.basis_size)
    assignment_idx = 0

    for task in tasks:
        compiled: list[Gate] = []
        angle_errors: list[float] = []
        for gate in task.skeleton:
            if gate[0] != _TEMP_PHASE_GATE:
                compiled.append(gate)
                continue
            assignment = cluster_plan.assignments[assignment_idx]
            assignment_idx += 1
            compiled.extend(_phase_gates_from_snapped_angle(task.qubit, assignment.snapped_angle))
            angle_errors.append(assignment.angle_error)
        compiled_run = _rewrite_gate_sequence(tuple(compiled))
        approx_unitary = _approximation_run_unitary(compiled_run, task.global_phase_radians)
        exact_unitary = _matrix_from_key(task.exact_unitary)
        run_fro_error = float(np.linalg.norm(exact_unitary - approx_unitary, ord="fro"))
        compiled_runs.append(compiled_run)
        stats.absorb(
            _ImportCompileStats(
                global_phase_radians=task.global_phase_radians,
                approximated_phase_count=task.approximated_phase_count,
                total_angle_error=sum(angle_errors),
                max_angle_error=max(angle_errors, default=0.0),
                approximation_run_count=1,
                total_run_fro_error=run_fro_error,
                max_run_fro_error=run_fro_error,
            )
        )
    return tuple(compiled_runs), stats

def _compile_one_qubit_run(
    qubit: int,
    run: Sequence[Gate],
    *,
    tolerance: float,
) -> tuple[tuple[Gate, ...], _ImportCompileStats]:
    from .ross_selinger import _compile_phase_gate

    if not run:
        return (), _ImportCompileStats()

    normalized_run, phase_gate_count, exact_run, total_angle = _normalized_import_run(run)
    if phase_gate_count == 0:
        return _rewrite_gate_sequence(normalized_run), _ImportCompileStats()
    if exact_run is not None:
        return exact_run, _ImportCompileStats(exact_dyadic_phase_count=phase_gate_count)
    if total_angle is not None:
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

    # Mixed one-qubit runs still compile residual phase gates directly.
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
    compile_mode: str,
) -> tuple[tuple[Gate, ...], _ImportCompileStats]:
    merged_gates = _merge_import_diagonal_phases(raw_gates)
    if compile_mode == _RZ_COMPILE_MODE_APPROX_DYADIC:
        return _compile_import_gate_sequence_approx_dyadic(merged_gates, tolerance=tolerance)

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

    # Large imports keep the cheaper local rewrite pass.
    if len(compiled) >= _FAST_IMPORT_GATE_COUNT_THRESHOLD:
        return _rewrite_gate_sequence_local(compiled), stats
    return _rewrite_gate_sequence(compiled), stats

def _compile_import_gate_sequence_approx_dyadic(
    merged_gates: Sequence[Gate],
    *,
    tolerance: float,
) -> tuple[tuple[Gate, ...], _ImportCompileStats]:
    sequence: list[Gate | _ApproximationRunTask | tuple[tuple[Gate, ...], _ImportCompileStats]] = []
    pending_runs: dict[int, list[Gate]] = {}

    def flush_qubit(qubit: int) -> None:
        run = pending_runs.pop(qubit, None)
        if not run:
            return
        sequence.append(_prepare_approximation_run_task(qubit, tuple(run), tolerance=tolerance))

    for gate in merged_gates:
        if _is_single_qubit_import_gate(gate):
            qubit = _gate_qubits_import(gate)[0]
            pending_runs.setdefault(qubit, []).append(gate)
            continue
        for qubit in sorted(_gate_qubits_import(gate)):
            flush_qubit(qubit)
        sequence.append(_normalize_gate(gate))

    for qubit in sorted(pending_runs):
        flush_qubit(qubit)

    approx_tasks = [entry for entry in sequence if isinstance(entry, _ApproximationRunTask)]
    compiled_task_runs: tuple[tuple[Gate, ...], ...] = ()
    approx_stats = _ImportCompileStats()
    if approx_tasks:
        compiled_task_runs, approx_stats = _compile_approximation_runs(approx_tasks, tolerance=tolerance)
    compiled: list[Gate] = []
    stats = _ImportCompileStats()
    task_idx = 0

    for entry in sequence:
        if isinstance(entry, tuple) and entry and isinstance(entry[0], tuple):
            compiled_run, run_stats = entry
            compiled.extend(compiled_run)
            stats.absorb(run_stats)
            continue
        if isinstance(entry, _ApproximationRunTask):
            compiled.extend(compiled_task_runs[task_idx])
            task_idx += 1
            continue
        compiled.append(entry)

    stats.absorb(approx_stats)
    if len(compiled) >= _FAST_IMPORT_GATE_COUNT_THRESHOLD:
        return _rewrite_gate_sequence_local(compiled), stats
    return _rewrite_gate_sequence(compiled), stats

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
