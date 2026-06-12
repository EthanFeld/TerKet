"""Tests for Qiskit import template caching behavior."""

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

class QiskitNativeImportTemplateTests(unittest.TestCase):

    def test_repeated_custom_subcircuit_definition_reuses_import_template(self):
        block = QuantumCircuit(2, name="block")
        block.h(0)
        block.cp(math.pi / 8.0, 0, 1)
        block.rz(-math.pi / 16.0, 1)
        block.cx(0, 1)
        gate = block.to_gate()

        qc = QuantumCircuit(4)
        qc.append(gate, [0, 1])
        qc.append(gate, [2, 3])
        qc.append(gate, [1, 2])

        with patch(
            "terket.interop.qiskit_import._compile_qiskit_circuit_template",
            wraps=qiskit_import._compile_qiskit_circuit_template,
        ) as compile_template:
            spec = from_qiskit(qc)

        self.assertEqual(compile_template.call_count, 1)

        statevector = Statevector.from_instruction(qc).data
        for bits in ((0, 0, 0, 0), (1, 0, 1, 0), (1, 1, 1, 1)):
            amplitude, _info = compute_circuit_amplitude(spec, [0, 0, 0, 0], bits, as_complex=True)
            expected = complex(statevector[_bits_to_index(bits)])
            self.assertAlmostEqual(amplitude.real, expected.real, places=12)
            self.assertAlmostEqual(amplitude.imag, expected.imag, places=12)

    def test_u3_clifford_t_import_uses_direct_unitary_synthesis(self):
        gate = U3Gate(0.73717103712378, 2.1071933827064955, -3.0042834966413654)
        qc = QuantumCircuit(1)
        qc.append(gate, [0])

        decomposed_qc = _qiskit_one_qubit_psx_decomposer()(gate)

        direct_spec = from_qiskit(qc, rz_compile_mode="clifford_t")
        decomposed_spec = from_qiskit(decomposed_qc, rz_compile_mode="clifford_t")

        self.assertEqual(sum(gate[0] == "rz_arbitrary" for gate in direct_spec.gates), 0)
        self.assertEqual(sum(gate[0] == "sx" for gate in direct_spec.gates), 0)
        self.assertLess(len(direct_spec.gates), len(decomposed_spec.gates))

        statevector = Statevector.from_instruction(qc).data
        for bits in ((0,), (1,)):
            amplitude, _info = compute_circuit_amplitude(direct_spec, [0], bits, as_complex=True)
            expected = complex(statevector[_bits_to_index(bits)])
            self.assertAlmostEqual(amplitude.real, expected.real, places=4)
            self.assertAlmostEqual(amplitude.imag, expected.imag, places=4)

    def test_approx_dyadic_clusters_close_phase_basis_angles(self):
        qc = QuantumCircuit(2)
        qc.p(0.111111, 0)
        qc.cx(0, 1)
        qc.p(0.111114, 0)
        qc.cx(0, 1)
        qc.p(0.111116, 1)

        spec = from_qiskit(qc, rz_compile_mode="approx_dyadic", rz_tolerance=1e-5)

        self.assertEqual(sum(gate[0] == "rz_arbitrary" for gate in spec.gates), 0)
        self.assertEqual(spec.metadata["approximation_mode"], "approx_dyadic")
        self.assertLess(spec.metadata["approximation_basis_size"], spec.metadata["approximation_phase_count"])
        self.assertGreater(spec.metadata["approximation_total_angle_error"], 0.0)

        statevector = Statevector.from_instruction(qc).data
        for bits in ((0, 0), (1, 0), (0, 1), (1, 1)):
            amplitude, _info = compute_circuit_amplitude(spec, [0, 0], bits, as_complex=True)
            expected = complex(statevector[_bits_to_index(bits)])
            self.assertAlmostEqual(amplitude.real, expected.real, places=4)
            self.assertAlmostEqual(amplitude.imag, expected.imag, places=4)

    def test_from_qiskit_rejects_mid_circuit_measurement_cleanly(self):
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        qc.x(0)

        with self.assertRaisesRegex(
            ValueError,
            "mid-circuit measurement.*optional trailing measurements",
        ):
            from_qiskit(qc)
