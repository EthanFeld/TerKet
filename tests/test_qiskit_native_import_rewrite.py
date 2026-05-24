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

class QiskitNativeImportRewriteTests(unittest.TestCase):

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
            "terket.interop.angles._compile_import_gate_sequence",
            side_effect=AssertionError("unexpected slow import compiler"),
        ):
            spec = from_qiskit(qc)

        counts = Counter(gate[0] for gate in spec.gates)
        self.assertGreater(counts["rz_dyadic"], 0)
        self.assertGreater(counts["rz_arbitrary"], 0)
        self.assertGreater(counts["cnot"], 0)

    def test_openqasm2_statement_lexer_handles_multiline_and_trailing_measure(self):
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        rz(pi/16)
            q[0];
        cx q[0],
            q[1];
        barrier q;
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        """

        spec = parse_openqasm2(qasm)

        self.assertEqual(spec.n_qubits, 2)
        self.assertIn(("cnot", 0, 1), spec.gates)

    def test_openqasm2_rejects_mid_measure_and_reset(self):
        with self.assertRaisesRegex(ValueError, "mid-circuit measurement"):
            parse_openqasm2(
                """
                OPENQASM 2.0;
                qreg q[1];
                creg c[1];
                measure q[0] -> c[0];
                x q[0];
                """
            )

        with self.assertRaisesRegex(ValueError, "reset"):
            parse_openqasm2(
                """
                OPENQASM 2.0;
                qreg q[1];
                reset q[0];
                """
            )

    def test_qiskit_import_uses_handler_registry(self):
        for name in ("cx", "rz", "p", "cp", "crz", "swap", "mcphase", "u3", "rzz"):
            self.assertIn(name, _QISKIT_OPERATION_HANDLERS)

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
