"""Qiskit export helpers."""

from __future__ import annotations

import math
from typing import Any, Sequence

from ..circuit_spec import _circuit_global_phase_radians, _coerce_finite_radians, normalize_circuit
from .angles import _dyadic_phase_to_angle

def to_qiskit(circuit: Any):
    """Convert a supported circuit input into a Qiskit ``QuantumCircuit``."""
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise RuntimeError("Qiskit is required for to_qiskit().") from exc

    spec = normalize_circuit(circuit)
    qc = QuantumCircuit(spec.n_qubits, name=spec.name)
    qc.global_phase = _circuit_global_phase_radians(spec)

    def apply_pauli_expbox(paulis: Sequence[str], qubits: Sequence[int], angle: float) -> None:
        angle_value = _coerce_finite_radians(angle, source="Unsupported PauliExpBox angle")
        active = [
            (str(pauli).upper(), int(qubit))
            for pauli, qubit in zip(paulis, qubits)
            if str(pauli).upper() != "I"
        ]
        qc.global_phase += -0.5 * angle_value
        if not active:
            return

        for pauli_char, qubit in active:
            if pauli_char == "X":
                qc.h(qubit)
            elif pauli_char == "Y":
                qc.sdg(qubit)
                qc.h(qubit)
            elif pauli_char != "Z":
                raise ValueError(f"Unsupported PauliExpBox Pauli {pauli_char!r}.")

        ordered_qubits = [qubit for _pauli_char, qubit in active]
        target = ordered_qubits[-1]
        for control, target_qubit in zip(ordered_qubits, ordered_qubits[1:]):
            qc.cx(control, target_qubit)
        qc.p(angle_value, target)
        for control, target_qubit in reversed(tuple(zip(ordered_qubits, ordered_qubits[1:]))):
            qc.cx(control, target_qubit)

        for pauli_char, qubit in reversed(active):
            if pauli_char == "X":
                qc.h(qubit)
            elif pauli_char == "Y":
                qc.h(qubit)
                qc.s(qubit)

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
        elif name == "pauli_expbox":
            apply_pauli_expbox(gate[1], gate[2], gate[3])
        else:
            getattr(qc, name)(*gate[1:])
    return qc
