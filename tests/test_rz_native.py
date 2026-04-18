from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from terket import compute_circuit_amplitude
from terket.circuit_spec import from_qiskit, parse_openqasm2
from terket.head_to_head_cases import (
    SUPPORTED_BASIS as HEAD_TO_HEAD_SUPPORTED_BASIS,
    transpile_to_supported_basis as transpile_head_to_head,
)
from terket.structured_cases import (
    SUPPORTED_BASIS as STRUCTURED_SUPPORTED_BASIS,
    transpile_to_supported_basis as transpile_structured,
)


def _bits_to_index(bits: tuple[int, ...]) -> int:
    return sum((int(bit) & 1) << idx for idx, bit in enumerate(bits))


class NativeRZImportTests(unittest.TestCase):
    def test_from_qiskit_defaults_to_native_rz_mode(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rz(math.pi / 16.0, 0)
        qc.cx(0, 1)
        qc.rz(math.pi / 7.0, 1)
        qc.sx(0)
        qc.rz(-math.pi / 32.0, 0)

        default_spec = from_qiskit(qc)
        explicit_dyadic_spec = from_qiskit(qc, rz_compile_mode="dyadic")

        self.assertEqual(default_spec, explicit_dyadic_spec)
        self.assertTrue(any(gate[0].startswith("rz") for gate in default_spec.gates))

    def test_parse_openqasm2_defaults_to_native_rz_mode(self):
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        rz(pi/16) q[0];
        cx q[0], q[1];
        rz(pi/7) q[1];
        rz(-pi/32) q[0];
        """

        default_spec = parse_openqasm2(qasm)
        explicit_dyadic_spec = parse_openqasm2(qasm, rz_compile_mode="dyadic")

        self.assertEqual(default_spec, explicit_dyadic_spec)
        self.assertTrue(any(gate[0].startswith("rz") for gate in default_spec.gates))

    def test_native_rz_import_stays_small_vs_clifford_t_synthesis(self):
        qc = QuantumCircuit(4)
        for qubit in range(4):
            qc.h(qubit)
            qc.rz(math.pi / 7.0, qubit)
        for qubit in range(3):
            qc.cx(qubit, qubit + 1)
        for qubit in range(4):
            qc.rz(math.pi / 16.0, qubit)

        native_spec = from_qiskit(qc)
        synthesized_spec = from_qiskit(qc, rz_compile_mode="clifford_t")

        self.assertTrue(any(gate[0].startswith("rz") for gate in native_spec.gates))
        self.assertLess(len(native_spec.gates) * 10, len(synthesized_spec.gates))


class NativeRZCorrectnessTests(unittest.TestCase):
    def test_native_rz_amplitudes_match_qiskit_statevector(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rz(math.pi / 16.0, 0)
        qc.cx(0, 1)
        qc.rz(math.pi / 7.0, 1)
        qc.sx(0)
        qc.rz(-math.pi / 32.0, 1)
        qc.h(1)

        spec = from_qiskit(qc)
        statevector = Statevector.from_instruction(qc).data

        for bits in ((0, 0), (1, 0), (0, 1), (1, 1)):
            amplitude, _info = compute_circuit_amplitude(spec, [0, 0], bits, as_complex=True)
            expected = complex(statevector[_bits_to_index(bits)])
            self.assertAlmostEqual(amplitude.real, expected.real, places=12)
            self.assertAlmostEqual(amplitude.imag, expected.imag, places=12)

    def test_sx_on_entangled_support_matches_qiskit_statevector(self):
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

        for bits in (
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 1, 0),
            (1, 1, 1),
        ):
            amplitude, _info = compute_circuit_amplitude(spec, [0, 0, 0], bits, as_complex=True)
            expected = complex(statevector[_bits_to_index(bits)])
            self.assertAlmostEqual(amplitude.real, expected.real, places=12)
            self.assertAlmostEqual(amplitude.imag, expected.imag, places=12)


class NativeRZBenchmarkTranspileTests(unittest.TestCase):
    def test_benchmark_transpile_helpers_preserve_rz_gates(self):
        qc = QuantumCircuit(2)
        qc.rzz(math.pi / 7.0, 0, 1)
        qc.rx(math.pi / 7.0, 0)
        qc.rx(math.pi / 5.0, 1)

        self.assertIn("rz", HEAD_TO_HEAD_SUPPORTED_BASIS)
        self.assertIn("sx", HEAD_TO_HEAD_SUPPORTED_BASIS)
        self.assertIn("rz", STRUCTURED_SUPPORTED_BASIS)
        self.assertIn("sx", STRUCTURED_SUPPORTED_BASIS)

        for spec in (transpile_head_to_head(qc), transpile_structured(qc)):
            self.assertTrue(any(gate[0].startswith("rz") for gate in spec.gates))
            self.assertLess(len(spec.gates), 100)


if __name__ == "__main__":
    unittest.main()
