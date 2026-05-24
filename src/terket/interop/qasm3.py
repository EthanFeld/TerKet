"""OpenQASM 3 detection and Qiskit-backed import helpers."""

from __future__ import annotations

from pathlib import Path

from ..circuit_spec import CircuitSpec, _RZ_COMPILE_MODE_DYADIC
from .qiskit_import import from_qiskit

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
