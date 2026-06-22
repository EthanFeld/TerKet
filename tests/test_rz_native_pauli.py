"""Tests for exact Pauli expectation behavior on native-enabled RZ paths."""

from __future__ import annotations

import math
from unittest import mock

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from terket import compute_circuit_amplitude, make_circuit
from terket.circuits import from_qiskit, to_qiskit
from terket import engine
from terket.interop.rewrite import _rewrite_gate_sequence


def _bits_to_index(bits: tuple[int, ...]) -> int:
    return sum((int(bit) & 1) << idx for idx, bit in enumerate(bits))


def test_pauli_expbox_single_pauli_matches_rotation_definition():
    theta = 0.37
    spec = make_circuit(1, [("pauli_expbox", ("Y",), (0,), theta)])

    amp0, _ = compute_circuit_amplitude(spec, [0], [0], as_complex=True)
    amp1, _ = compute_circuit_amplitude(spec, [0], [1], as_complex=True)

    assert abs(amp0 - math.cos(0.5 * theta)) < 1e-12
    assert abs(amp1 - math.sin(0.5 * theta)) < 1e-12


def test_pauli_expbox_active_two_pi_keeps_global_sign():
    spec = make_circuit(1, [("pauli_expbox", ("Z",), (0,), 2.0 * math.pi)])

    amp0, _ = compute_circuit_amplitude(spec, [0], [0], as_complex=True)
    amp1, _ = compute_circuit_amplitude(spec, [1], [1], as_complex=True)

    assert abs(amp0 + 1.0) < 1e-12
    assert abs(amp1 + 1.0) < 1e-12


def test_pauli_expbox_multi_pauli_matches_direct_matrix():
    theta = -0.91
    spec = make_circuit(2, [("pauli_expbox", ("X", "Z"), (0, 1), theta)])

    amp00, _ = compute_circuit_amplitude(spec, [0, 1], [0, 1], as_complex=True)
    amp10, _ = compute_circuit_amplitude(spec, [0, 1], [1, 1], as_complex=True)

    assert abs(amp00 - math.cos(0.5 * theta)) < 1e-12
    assert abs(amp10 - 1.0j * math.sin(0.5 * theta)) < 1e-12


def test_rewrite_does_not_crash_on_h_pauli_expbox_h():
    gates = (("h", 0), ("pauli_expbox", ("Z",), (0,), 0.37), ("h", 0))
    assert _rewrite_gate_sequence(gates) == gates


def test_to_qiskit_lowers_pauli_expbox():
    theta = 0.43
    spec = make_circuit(2, [("h", 0), ("pauli_expbox", ("X", "Z"), (0, 1), theta)])
    statevector = Statevector.from_instruction(to_qiskit(spec)).data

    for bits in ((0, 0), (1, 0), (0, 1), (1, 1)):
        actual, _ = compute_circuit_amplitude(spec, [0, 0], bits, as_complex=True)
        assert abs(actual - complex(statevector[_bits_to_index(bits)])) < 1e-12


def test_pauli_expbox_schur_path_does_not_materialize_cnot_ladder():
    spec = make_circuit(3, [("pauli_expbox", ("X", "Z", "Y"), (0, 1, 2), 0.43)])

    with mock.patch.object(engine.SchurState, "cnot", side_effect=AssertionError("unexpected cnot ladder")):
        amp, _ = compute_circuit_amplitude(spec, [0, 0, 0], [0, 0, 0], as_complex=True)

    assert abs(amp - math.cos(0.5 * 0.43)) < 1e-12


def test_pauli_expbox_dyadic_snap_uses_phase_polynomial(monkeypatch):
    spec = make_circuit(2, [("pauli_expbox", ("X", "Z"), (0, 1), math.pi / 4.0)])
    monkeypatch.setenv("TERKET_PAULI_EXPBOX_DYADIC_LEVEL", "3")

    state = engine.build_state(spec.n_qubits, spec.gates, [0, 1])
    amp, _ = compute_circuit_amplitude(spec, [0, 1], [0, 1], as_complex=True)

    assert not state._arbitrary_phases
    assert abs(amp - math.cos(math.pi / 8.0)) < 1e-12


def test_sx_on_entangled_support_matches_qiskit_statevector():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.sx(2)
    qc.rz(math.pi / 16.0, 2)
    qc.cz(0, 2)
    qc.sx(1)
    qc.rz(-math.pi / 32.0, 1)
    qc.cx(2, 0)
    qc.sx(0)
    qc.h(2)

    spec = from_qiskit(qc)
    statevector = Statevector.from_instruction(qc).data
    for bits in ((0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)):
        actual, _ = compute_circuit_amplitude(spec, [0, 0, 0], bits, as_complex=True)
        assert abs(actual - complex(statevector[_bits_to_index(bits)])) < 1e-12
