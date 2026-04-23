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

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, U2Gate, U3Gate
from qiskit.quantum_info import Statevector

from terket import compute_circuit_amplitude
from terket.circuit_spec import (
    _FAST_IMPORT_GATE_COUNT_THRESHOLD,
    _fast_import_gate_sequence_if_supported,
    _rewrite_gate_sequence,
    from_qiskit,
)
from terket.benchmarking.head_to_head_cases import build_approximate_qft_logical


def _bits_to_index(bits: tuple[int, ...]) -> int:
    return sum((int(bit) & 1) << idx for idx, bit in enumerate(bits))


class QiskitNativeImportTests(unittest.TestCase):
    def test_rewrite_cancels_arbitrary_rz_across_cz(self):
        rewritten = _rewrite_gate_sequence(
            (
                ("rz_arbitrary", 0, 0.37),
                ("cz", 0, 1),
                ("rz_arbitrary", 0, -0.37),
            )
        )

        self.assertEqual(rewritten, (("cz", 0, 1),))

    def test_rewrite_fuses_mixed_phase_gates_exactly(self):
        rewritten = _rewrite_gate_sequence(
            (
                ("rz_arbitrary", 0, math.pi / 4.0),
                ("t", 0),
                ("tdg", 0),
                ("rz_arbitrary", 0, -math.pi / 4.0),
            )
        )

        self.assertEqual(rewritten, ())

    def test_large_native_clifford_t_stream_uses_fast_import_path(self):
        qc = build_approximate_qft_logical(1024)
        raw_gates = []
        for instruction in qc.data:
            name = instruction.operation.name.lower()
            qubits = [qc.find_bit(qubit).index for qubit in instruction.qubits]
            raw_gates.append((("cnot" if name == "cx" else name), *qubits))

        self.assertGreater(len(raw_gates), _FAST_IMPORT_GATE_COUNT_THRESHOLD)
        fast = _fast_import_gate_sequence_if_supported(raw_gates)

        self.assertIsNotNone(fast)
        self.assertEqual(fast, tuple(raw_gates))

    def test_large_native_rz_stream_uses_fast_import_path(self):
        qc = QuantumCircuit(2)
        for _ in range(_FAST_IMPORT_GATE_COUNT_THRESHOLD + 1):
            qc.rz(math.pi / 16.0, 0)
            qc.rz(math.pi / 7.0, 1)
            qc.cx(0, 1)

        with patch(
            "terket.circuit_spec._compile_import_gate_sequence",
            side_effect=AssertionError("unexpected slow import compiler"),
        ):
            spec = from_qiskit(qc)

        counts = Counter(gate[0] for gate in spec.gates)
        self.assertGreater(counts["rz_dyadic"], 0)
        self.assertGreater(counts["rz_arbitrary"], 0)
        self.assertGreater(counts["cnot"], 0)

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

    def test_controlled_phase_family_imports_directly_and_matches_statevector(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.cp(math.pi / 7.0, 0, 1)
        qc.crz(-math.pi / 5.0, 1, 2)
        qc.mcp(math.pi / 3.0, [0, 1], 2)

        with patch("terket.circuit_spec._synthesize_qiskit_operation", side_effect=AssertionError("unexpected")):
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

        with patch("terket.circuit_spec._synthesize_qiskit_operation", side_effect=AssertionError("unexpected")):
            spec = from_qiskit(qc)

        self.assertGreater(sum(gate[0] == "sx" for gate in spec.gates), 0)
        self.assertGreater(sum(gate[0].startswith("rz") for gate in spec.gates), 0)

        amplitude, _info = compute_circuit_amplitude(spec, [0], [0], as_complex=True)
        expected = complex(Statevector.from_instruction(qc).data[0])
        self.assertAlmostEqual(amplitude.real, expected.real, places=12)
        self.assertAlmostEqual(amplitude.imag, expected.imag, places=12)

if __name__ == "__main__":
    unittest.main()
