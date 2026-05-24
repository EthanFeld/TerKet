"""Circuit spec types and thin frontend normalization dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
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
    "pauli_expbox",
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
_QASM_SUPPORTED_GATE_SET = "{h, sx, sxdg, x, t, tdg, s, sdg, z, cnot, cz, rz(theta), rx(theta), rzz(theta), u3(theta, phi, lambda)}"
_MAX_RZ_TOLERANCE = 1e-5
_EXACT_DYADIC_TOLERANCE = 1e-12
_EXACT_DYADIC_MAX_LEVEL = 20
_GLOBAL_PHASE_METADATA_KEY = "global_phase_radians"
_APPROXIMATION_MODE_METADATA_KEY = "approximation_mode"
_APPROXIMATION_BASIS_SIZE_METADATA_KEY = "approximation_basis_size"
_APPROXIMATION_PHASE_COUNT_METADATA_KEY = "approximation_phase_count"
_APPROXIMATION_RUN_COUNT_METADATA_KEY = "approximation_run_count"
_APPROXIMATION_TOTAL_RUN_FRO_ERROR_METADATA_KEY = "approximation_total_run_fro_error"
_APPROXIMATION_MAX_RUN_FRO_ERROR_METADATA_KEY = "approximation_max_run_fro_error"
_APPROXIMATION_TOTAL_ANGLE_ERROR_METADATA_KEY = "approximation_total_angle_error"
_APPROXIMATION_MAX_ANGLE_ERROR_METADATA_KEY = "approximation_max_angle_error"
_APPROXIMATION_TOLERANCE_METADATA_KEY = "approximation_tolerance"
_ROSS_SELINGER_SUBPROCESS_ONLY = False
_TEMP_PHASE_GATE = "phase_angle"
_RZ_COMPILE_MODE_APPROX_DYADIC = "approx_dyadic"
_RZ_COMPILE_MODE_CLIFFORD_T = "clifford_t"
_RZ_COMPILE_MODE_DYADIC = "dyadic"
_RZ_COMPILE_MODES = {
    _RZ_COMPILE_MODE_APPROX_DYADIC,
    _RZ_COMPILE_MODE_CLIFFORD_T,
    _RZ_COMPILE_MODE_DYADIC,
}

@dataclass(frozen=True, slots=True)
class CircuitSpec:
    """Normalized Clifford+T circuit accepted by the TerKet simulator."""

    n_qubits: int
    gates: tuple[Gate, ...]
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        from .interop.rewrite import _normalize_gate

        object.__setattr__(self, "gates", tuple(_normalize_gate(g) for g in self.gates))
        object.__setattr__(self, "metadata", _normalize_circuit_metadata(self.metadata))
        _validate_gates(self.n_qubits, self.gates)

def make_circuit(
    n_qubits: int,
    gates: Iterable[Gate],
    name: str | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> CircuitSpec:
    return CircuitSpec(
        n_qubits=n_qubits,
        gates=tuple(gates),
        name=name,
        metadata={} if metadata is None else dict(metadata),
    )

def lift_exact_dyadic_precision(circuit: CircuitSpec, *, min_level: int) -> CircuitSpec:
    from .interop.rewrite import _diagonal_phase_spec

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
        from .interop.qasm2 import parse_openqasm2
        from .interop.qasm3 import _looks_like_openqasm3, _parse_openqasm3_via_qiskit

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
            return parse_openqasm2(source, name=qasm_path.stem, rz_tolerance=rz_tolerance, rz_compile_mode=rz_compile_mode)
        if _looks_like_openqasm3(circuit):
            return _parse_openqasm3_via_qiskit(circuit, rz_tolerance=rz_tolerance, rz_compile_mode=rz_compile_mode)
        return parse_openqasm2(circuit, rz_tolerance=rz_tolerance, rz_compile_mode=rz_compile_mode)
    if hasattr(circuit, "num_qubits") and hasattr(circuit, "data"):
        from .interop.qiskit_import import from_qiskit

        return from_qiskit(circuit, rz_tolerance=rz_tolerance, rz_compile_mode=rz_compile_mode)
    raise TypeError(f"Unsupported circuit input: {type(circuit)!r}")

def bits_to_index(bits: Sequence[int]) -> int:
    value = 0
    for idx, bit in enumerate(bits):
        value |= (int(bit) & 1) << idx
    return value

def bits_to_little_endian_string(bits: Sequence[int]) -> str:
    return "".join(str(int(bit) & 1) for bit in bits)

def bits_to_big_endian_string(bits: Sequence[int]) -> str:
    return bits_to_little_endian_string(reversed(tuple(bits)))

def little_endian_string_to_bits(bitstring: str) -> tuple[int, ...]:
    if any(char not in {"0", "1"} for char in bitstring):
        raise ValueError(f"Bitstring must contain only 0/1 characters, received {bitstring!r}.")
    return tuple(int(char) for char in bitstring)

def big_endian_string_to_bits(bitstring: str) -> tuple[int, ...]:
    return little_endian_string_to_bits(bitstring[::-1])

def iter_bitstrings(n_qubits: int):
    for value in range(1 << n_qubits):
        yield [(value >> idx) & 1 for idx in range(n_qubits)]

def _validate_gates(n_qubits: int, gates: Sequence[Gate]) -> None:
    for gate in gates:
        name = gate[0]
        if name not in SUPPORTED_GATES:
            raise ValueError(f"Unsupported gate: {name!r}")
        arity = 2 if name in {"cnot", "cz"} else 4 if name == "rzz_dyadic" else 3 if name in {"pauli_expbox", "rz_dyadic"} else 2 if name == "rz_arbitrary" else 1
        if len(gate) != arity + 1:
            raise ValueError(f"Gate {gate!r} has the wrong arity.")
        qubits = gate[1:2] if name in {"rz_dyadic", "rz_arbitrary"} else gate[1:3] if name == "rzz_dyadic" else gate[2] if name == "pauli_expbox" else gate[1:]
        if name == "pauli_expbox":
            paulis = gate[1]
            if not isinstance(paulis, tuple):
                raise TypeError(f"Gate {gate!r} uses a non-tuple Pauli string.")
            if len(paulis) != len(gate[2]):
                raise ValueError(f"Gate {gate!r} has mismatched Pauli/qubit lengths.")
            for pauli in paulis:
                if pauli not in {"I", "X", "Y", "Z"}:
                    raise ValueError(f"Gate {gate!r} uses unsupported Pauli {pauli!r}.")
            _coerce_finite_radians(gate[3], source="Unsupported PauliExpBox angle")
        for qubit in qubits:
            if not isinstance(qubit, int):
                raise TypeError(f"Gate {gate!r} uses a non-integer qubit index.")
            if not 0 <= qubit < n_qubits:
                raise ValueError(f"Gate {gate!r} targets qubit outside 0..{n_qubits - 1}.")
        if name in {"rzz_dyadic", "rz_dyadic"}:
            coeff, precision_level = gate[-2], gate[-1]
            if not isinstance(coeff, int):
                raise TypeError(f"Gate {gate!r} uses a non-integer phase coefficient.")
            if not isinstance(precision_level, int):
                raise TypeError(f"Gate {gate!r} uses a non-integer precision level.")
            if precision_level < 1:
                raise ValueError(f"Gate {gate!r} uses invalid precision level {precision_level}.")
        if name == "rz_arbitrary":
            _coerce_finite_radians(gate[2], source="Unsupported arbitrary phase angle")

def _normalize_global_phase_radians(value: float) -> float:
    return math.remainder(float(value), 2.0 * math.pi)

def _normalize_circuit_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(metadata or {})
    global_phase_radians = _normalize_global_phase_radians(normalized.get(_GLOBAL_PHASE_METADATA_KEY, 0.0))
    normalized[_GLOBAL_PHASE_METADATA_KEY] = global_phase_radians
    return normalized

def _metadata_with_global_phase(global_phase_radians: float) -> dict[str, Any]:
    return {_GLOBAL_PHASE_METADATA_KEY: _normalize_global_phase_radians(global_phase_radians)}

def _metadata_with_import_stats(global_phase_radians: float, compile_stats: _ImportCompileStats, *, compile_mode: str, tolerance: float) -> dict[str, Any]:
    metadata = _metadata_with_global_phase(global_phase_radians)
    if compile_mode == _RZ_COMPILE_MODE_APPROX_DYADIC:
        metadata[_APPROXIMATION_MODE_METADATA_KEY] = compile_mode
        metadata[_APPROXIMATION_BASIS_SIZE_METADATA_KEY] = int(compile_stats.approximation_basis_size)
        metadata[_APPROXIMATION_PHASE_COUNT_METADATA_KEY] = int(compile_stats.approximated_phase_count)
        metadata[_APPROXIMATION_RUN_COUNT_METADATA_KEY] = int(compile_stats.approximation_run_count)
        metadata[_APPROXIMATION_TOTAL_RUN_FRO_ERROR_METADATA_KEY] = float(compile_stats.total_run_fro_error)
        metadata[_APPROXIMATION_MAX_RUN_FRO_ERROR_METADATA_KEY] = float(compile_stats.max_run_fro_error)
        metadata[_APPROXIMATION_TOTAL_ANGLE_ERROR_METADATA_KEY] = float(compile_stats.total_angle_error)
        metadata[_APPROXIMATION_MAX_ANGLE_ERROR_METADATA_KEY] = float(compile_stats.max_angle_error)
        metadata[_APPROXIMATION_TOLERANCE_METADATA_KEY] = float(tolerance)
    return metadata

def _circuit_global_phase_radians(spec: CircuitSpec) -> float:
    return float(spec.metadata.get(_GLOBAL_PHASE_METADATA_KEY, 0.0))

def _validated_rz_tolerance(tolerance: float) -> float:
    tolerance = float(tolerance)
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, received {tolerance}.")
    if tolerance > _MAX_RZ_TOLERANCE:
        raise ValueError(f"rz_tolerance must be <= {_MAX_RZ_TOLERANCE:.3e}, received {tolerance:.3e}.")
    return tolerance

def _normalize_rz_compile_mode(mode: str | None, *, default: str = _RZ_COMPILE_MODE_DYADIC) -> str:
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
    except Exception as exc:
        raise ValueError(f"{source} {angle!r}. A numeric value is required.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{source} {angle!r}. Finite numeric values are required.")
    return value
