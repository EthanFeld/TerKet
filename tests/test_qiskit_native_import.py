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

from terket import compute_circuit_amplitude
from terket.circuit_spec import (
    _FAST_IMPORT_GATE_COUNT_THRESHOLD,
    _fast_import_gate_sequence_if_supported,
    _qiskit_one_qubit_psx_decomposer,
    _rewrite_gate_sequence,
    from_qiskit,
    normalize_circuit,
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

    def test_rewrite_fuses_cnot_phase_cnot_into_rzz(self):
        rewritten = _rewrite_gate_sequence(
            (
                ("cnot", 0, 1),
                ("rz_dyadic", 1, 3, 5),
                ("cnot", 0, 1),
            )
        )

        self.assertEqual(rewritten, (("rzz_dyadic", 0, 1, 3, 5),))

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
        self.assertLessEqual(len(fast), len(raw_gates))
        self.assertGreater(Counter(gate[0] for gate in fast)["rzz_dyadic"], 0)

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

    def test_large_native_fast_import_still_applies_local_cnot_phase_fusion(self):
        raw_gates = [
            gate
            for _ in range((_FAST_IMPORT_GATE_COUNT_THRESHOLD // 3) + 8)
            for gate in (("cnot", 0, 1), ("rz_dyadic", 1, 1, 3), ("cnot", 0, 1))
        ]

        fast = _fast_import_gate_sequence_if_supported(raw_gates)

        self.assertIsNotNone(fast)
        self.assertEqual(Counter(gate[0] for gate in fast)["cnot"], 0)
        self.assertEqual(Counter(gate[0] for gate in fast)["rzz_dyadic"], 1)

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

if __name__ == "__main__":
    unittest.main()
