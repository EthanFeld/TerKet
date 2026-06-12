"""Tests for Qiskit import/export round-trip behavior."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.library import QFTGate, U2Gate, U3Gate
from qiskit.quantum_info import Statevector

import terket.interop.qiskit_import as qiskit_import
from terket import compute_circuit_amplitude
from terket.circuits import from_qiskit, normalize_circuit, parse_openqasm2, to_qiskit
from terket.interop.qiskit_import import (
    _FAST_IMPORT_GATE_COUNT_THRESHOLD,
    _QISKIT_OPERATION_HANDLERS,
    _fast_import_gate_sequence_if_supported,
    _qiskit_one_qubit_psx_decomposer,
)
from terket.interop.rewrite import _rewrite_gate_sequence
from terket.benchmarking.head_to_head_cases import build_approximate_qft_logical

def _bits_to_index(bits: tuple[int, ...]) -> int:
    return sum((int(bit) & 1) << idx for idx, bit in enumerate(bits))

class QiskitNativeImportRoundtripTests(unittest.TestCase):

    def test_to_qiskit_from_qiskit_roundtrip_supported_gates(self):
        spec = normalize_circuit(
            3,
            (
                ("h", 0),
                ("sx", 1),
                ("sxdg", 1),
                ("x", 2),
                ("t", 0),
                ("tdg", 0),
                ("s", 1),
                ("sdg", 1),
                ("z", 2),
                ("cnot", 0, 1),
                ("cz", 1, 2),
                ("rz_dyadic", 0, 1, 5),
                ("rz_arbitrary", 1, math.pi / 7.0),
                ("rzz_dyadic", 0, 2, 1, 4),
                ("pauli_expbox", ("X", "Z"), (0, 2), 0.31),
                ("rz_pi_16", 1),
                ("rz_pi_16_dg", 1),
                ("rz_pi_32", 2),
                ("rz_pi_32_dg", 2),
            ),
        )

        roundtrip = from_qiskit(to_qiskit(spec))

        for bits in ((0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1)):
            left, _ = compute_circuit_amplitude(spec, [0, 0, 0], bits, as_complex=True)
            right, _ = compute_circuit_amplitude(roundtrip, [0, 0, 0], bits, as_complex=True)
            self.assertAlmostEqual(left.real, right.real, places=12)
            self.assertAlmostEqual(left.imag, right.imag, places=12)

    def test_qiskit_rzz_imports_natively_and_matches_statevector(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rzz(math.pi / 8.0, 0, 1)
        qc.sx(1)
        qc.rzz(-math.pi / 4.0, 0, 1)

        spec = from_qiskit(qc)
        counts = Counter(gate[0] for gate in spec.gates)

        self.assertEqual(counts["rzz_dyadic"], 2)
        self.assertEqual(counts["rz_arbitrary"], 0)

        statevector = Statevector.from_instruction(qc).data
        for bits in ((0, 0), (1, 0), (0, 1), (1, 1)):
            amplitude, _info = compute_circuit_amplitude(spec, [0, 0], bits, as_complex=True)
            expected = complex(statevector[_bits_to_index(bits)])
            self.assertAlmostEqual(amplitude.real, expected.real, places=12)
            self.assertAlmostEqual(amplitude.imag, expected.imag, places=12)

    def test_qiskit_rx_import_matches_statevector(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.rx(0.37, 1)
        qc.rz(-0.21, 2)
        qc.rzz(0.19, 0, 2)
        qc.cx(1, 2)
        qc.sdg(0)

        spec = from_qiskit(qc, rz_compile_mode="approx_dyadic", rz_tolerance=1e-5)

        statevector = Statevector.from_instruction(qc).data
        for index in range(1 << 3):
            bits = tuple((index >> qubit) & 1 for qubit in range(3))
            amplitude, _info = compute_circuit_amplitude(spec, [0, 0, 0], bits, as_complex=True)
            expected = complex(statevector[_bits_to_index(bits)])
            self.assertAlmostEqual(amplitude.real, expected.real, places=5)
            self.assertAlmostEqual(amplitude.imag, expected.imag, places=5)

    def test_openqasm2_rx_rzz_import_matches_statevector(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.rx(0.37, 1)
        qc.rz(-0.21, 2)
        qc.rzz(0.19, 0, 2)
        qc.cx(1, 2)
        qc.sdg(0)

        spec = normalize_circuit(qasm2.dumps(qc), rz_compile_mode="approx_dyadic", rz_tolerance=1e-5)

        statevector = Statevector.from_instruction(qc).data
        for index in range(1 << 3):
            bits = tuple((index >> qubit) & 1 for qubit in range(3))
            amplitude, _info = compute_circuit_amplitude(spec, [0, 0, 0], bits, as_complex=True)
            expected = complex(statevector[_bits_to_index(bits)])
            self.assertAlmostEqual(amplitude.real, expected.real, places=5)
            self.assertAlmostEqual(amplitude.imag, expected.imag, places=5)

    def test_controlled_phase_family_imports_directly_and_matches_statevector(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.cp(math.pi / 7.0, 0, 1)
        qc.crz(-math.pi / 5.0, 1, 2)
        qc.mcp(math.pi / 3.0, [0, 1], 2)

        with patch("terket.interop.qiskit_import._synthesize_qiskit_operation", side_effect=AssertionError("unexpected")):
            spec = from_qiskit(qc)

        statevector = Statevector.from_instruction(qc).data
        for index in range(1 << 3):
            bits = tuple((index >> qubit) & 1 for qubit in range(3))
            amplitude, _info = compute_circuit_amplitude(spec, [0, 0, 0], bits, as_complex=True)
            expected = complex(statevector[_bits_to_index(bits)])
            self.assertAlmostEqual(amplitude.real, expected.real, places=12)
            self.assertAlmostEqual(amplitude.imag, expected.imag, places=12)

    def test_qft_inverse_imports_cp_and_swap_natively(self):
        qc = QuantumCircuit(4)
        qc.x(0)
        qc.h(1)
        qc.append(QFTGate(4).inverse(), range(4))

        spec = from_qiskit(qc)

        self.assertEqual(sum(gate[0] == "rz_arbitrary" for gate in spec.gates), 0)
        self.assertGreater(sum(gate[0] == "rz_dyadic" for gate in spec.gates), 0)
        self.assertGreater(sum(gate[0] == "cnot" for gate in spec.gates), 0)

        statevector = Statevector.from_instruction(qc).data
        for bits in ((0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 1)):
            amplitude, _info = compute_circuit_amplitude(spec, [0, 0, 0, 0], bits, as_complex=True)
            expected = complex(statevector[_bits_to_index(bits)])
            self.assertAlmostEqual(amplitude.real, expected.real, places=12)
            self.assertAlmostEqual(amplitude.imag, expected.imag, places=12)

    def test_u_gate_shor_patterns_import_natively(self):
        qc = QuantumCircuit(1)
        qc.u(0.0, 0.0, math.pi / 8.0, 0)
        qc.u(math.pi / 2.0, 0.0, math.pi, 0)
        qc.u(math.pi, 0.0, math.pi, 0)

        spec = from_qiskit(qc)
        self.assertGreater(sum(gate[0] == "rz_dyadic" for gate in spec.gates), 0)
        self.assertEqual(sum(gate[0] == "rz_arbitrary" for gate in spec.gates), 0)
        self.assertGreater(sum(gate[0] == "x" for gate in spec.gates), 0)

        amplitude, _info = compute_circuit_amplitude(spec, [0], [0], as_complex=True)
        expected = complex(Statevector.from_instruction(qc).data[0])
        self.assertAlmostEqual(amplitude.real, expected.real, places=12)
        self.assertAlmostEqual(amplitude.imag, expected.imag, places=12)

    def test_u_family_import_avoids_generic_synthesis(self):
        qc = QuantumCircuit(1)
        qc.append(U3Gate(0.73717103712378, 2.1071933827064955, -3.0042834966413654), [0])
        qc.append(U2Gate(0.5, -0.7), [0])
        qc.u(0.2, 0.3, 0.4, 0)

        with patch("terket.interop.qiskit_import._synthesize_qiskit_operation", side_effect=AssertionError("unexpected")):
            spec = from_qiskit(qc)

        self.assertGreater(sum(gate[0] == "sx" for gate in spec.gates), 0)
        self.assertGreater(sum(gate[0].startswith("rz") for gate in spec.gates), 0)

        amplitude, _info = compute_circuit_amplitude(spec, [0], [0], as_complex=True)
        expected = complex(Statevector.from_instruction(qc).data[0])
        self.assertAlmostEqual(amplitude.real, expected.real, places=12)
        self.assertAlmostEqual(amplitude.imag, expected.imag, places=12)
