"""Circuit interop facades."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "_parse_openqasm3_via_qiskit",
    "_rewrite_gate_sequence",
    "dyadic_snap",
    "from_qiskit",
    "parse_openqasm2",
    "to_qiskit",
]


def __getattr__(name: str):
    if name == "dyadic_snap":
        return getattr(import_module(".angles", __name__), name)
    if name == "parse_openqasm2":
        return getattr(import_module(".qasm2", __name__), name)
    if name == "_parse_openqasm3_via_qiskit":
        return getattr(import_module(".qasm3", __name__), name)
    if name == "to_qiskit":
        return getattr(import_module(".qiskit_export", __name__), name)
    if name in {"from_qiskit"}:
        return getattr(import_module(".qiskit_import", __name__), name)
    if name == "_rewrite_gate_sequence":
        return getattr(import_module(".rewrite", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
