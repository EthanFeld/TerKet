from __future__ import annotations

from collections import Counter
import importlib.util
import math
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector

from terket import compute_circuit_amplitude
from terket.circuit_spec import (
    _FAST_IMPORT_GATE_COUNT_THRESHOLD,
    _fast_import_gate_sequence_if_supported,
    from_qiskit,
    to_qiskit,
)
from terket.head_to_head_cases import build_approximate_qft_logical


def _bits_to_index(bits: tuple[int, ...]) -> int:
    return sum((int(bit) & 1) << idx for idx, bit in enumerate(bits))


class QiskitNativeImportTests(unittest.TestCase):
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

    def test_qft_inverse_imports_cp_and_swap_natively(self):
        qc = QuantumCircuit(4)
        qc.x(0)
        qc.h(1)
        qc.append(QFTGate(4).inverse(), range(4))

        spec = from_qiskit(qc)
        counts = Counter(gate[0] for gate in spec.gates)

        self.assertEqual(counts["cp_dyadic"], 6)
        self.assertEqual(counts["swap"], 2)
        self.assertEqual(counts["rz_arbitrary"], 0)

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
        self.assertEqual(spec.gates, (("rz_dyadic", 0, 1, 4), ("h", 0), ("x", 0)))

        amplitude, _info = compute_circuit_amplitude(spec, [0], [0], as_complex=True)
        expected = complex(Statevector.from_instruction(qc).data[0])
        self.assertAlmostEqual(amplitude.real, expected.real, places=12)
        self.assertAlmostEqual(amplitude.imag, expected.imag, places=12)


@unittest.skipUnless(importlib.util.find_spec("mqt.bench") is not None, "mqt.bench is not installed")
class MQTBenchShorImportTests(unittest.TestCase):
    def test_mqtbench_shor_imports_without_transpilation(self):
        from mqt.bench.benchmarks.shor import create_circuit_from_num_and_coprime

        qc = create_circuit_from_num_and_coprime(15, 4)
        spec = from_qiskit(qc)
        counts = Counter(gate[0] for gate in spec.gates)

        self.assertEqual(spec.n_qubits, 18)
        self.assertEqual(counts["cp_dyadic"], 28)
        self.assertEqual(counts["swap"], 4)
        self.assertEqual(counts["rz_arbitrary"], 0)
        self.assertGreater(counts["rz_dyadic"], 0)


@unittest.skipUnless(importlib.util.find_spec("mqt.bench") is not None, "mqt.bench is not installed")
class MQTBenchAEImportTests(unittest.TestCase):
    def test_mqtbench_ae_roundtrips_small_instances(self):
        from mqt.bench.benchmarks.ae import create_circuit

        for n_qubits in (2, 3):
            with self.subTest(n_qubits=n_qubits):
                qc = create_circuit(n_qubits).remove_final_measurements(inplace=False)
                spec = from_qiskit(qc)
                roundtrip = to_qiskit(spec)
                expected = Statevector.from_instruction(qc).data
                actual = Statevector.from_instruction(roundtrip).data

                for got, want in zip(actual, expected):
                    self.assertAlmostEqual(complex(got).real, complex(want).real, places=12)
                    self.assertAlmostEqual(complex(got).imag, complex(want).imag, places=12)

                for bits in ((0,) * n_qubits, (1,) * n_qubits):
                    amplitude, _info = compute_circuit_amplitude(spec, [0] * n_qubits, bits, as_complex=True)
                    expected_amplitude = complex(expected[_bits_to_index(bits)])
                    self.assertAlmostEqual(amplitude.real, expected_amplitude.real, places=12)
                    self.assertAlmostEqual(amplitude.imag, expected_amplitude.imag, places=12)

    def test_mqtbench_ae_direct_import_stays_smaller_than_forced_basis(self):
        from mqt.bench.benchmarks.ae import create_circuit

        qc = create_circuit(6)
        direct = from_qiskit(qc)
        transpiled = transpile(
            qc.remove_final_measurements(inplace=False),
            basis_gates=["h", "sx", "x", "rz", "cx", "cz"],
            optimization_level=0,
        )
        forced_basis = from_qiskit(transpiled)

        self.assertLess(len(direct.gates), len(forced_basis.gates))
        self.assertGreater(sum(gate[0] == "rz_arbitrary" for gate in direct.gates), 0)


if __name__ == "__main__":
    unittest.main()
