"""Tests for exact dyadic RZ import behavior."""

from __future__ import annotations

import cmath
import math
import os
from pathlib import Path
import sys
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from terket import compute_circuit_amplitude, make_circuit
from terket.circuits import from_qiskit, parse_openqasm2, to_qiskit
from terket.cubic_arithmetic import PhaseFunction
from terket import engine
from terket.interop.rewrite import _rewrite_gate_sequence
from terket.benchmarking.head_to_head_cases import (
    SUPPORTED_BASIS as HEAD_TO_HEAD_SUPPORTED_BASIS,
    build_approximate_qft,
    build_qaoa_ring_logical,
    build_repetition_magic_round,
    transpile_to_supported_basis as transpile_head_to_head,
)
from terket.benchmarking.structured_cases import (
    SUPPORTED_BASIS as STRUCTURED_SUPPORTED_BASIS,
    build_mm_hidden_shift_logical,
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

    def test_parse_openqasm2_clifford_t_synthesizes_u3_directly(self):
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        u3(0.73717103712378, 2.1071933827064955, -3.0042834966413654) q[0];
        """
        qc = QuantumCircuit(1)
        qc.u(0.73717103712378, 2.1071933827064955, -3.0042834966413654, 0)

        spec = parse_openqasm2(qasm, rz_compile_mode="clifford_t")

        self.assertEqual(sum(gate[0] == "rz_arbitrary" for gate in spec.gates), 0)
        self.assertEqual(sum(gate[0] == "sx" for gate in spec.gates), 0)

        statevector = Statevector.from_instruction(qc).data
        for bits in ((0,), (1,)):
            amplitude, _info = compute_circuit_amplitude(spec, [0], bits, as_complex=True)
            expected = complex(statevector[_bits_to_index(bits)])
            self.assertAlmostEqual(amplitude.real, expected.real, places=4)
            self.assertAlmostEqual(amplitude.imag, expected.imag, places=4)

    def test_parse_openqasm2_approx_dyadic_reports_run_level_error(self):
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        u3(0.73717103712378, 2.1071933827064955, -3.0042834966413654) q[0];
        """
        qc = QuantumCircuit(1)
        qc.u(0.73717103712378, 2.1071933827064955, -3.0042834966413654, 0)

        spec = parse_openqasm2(qasm, rz_compile_mode="approx_dyadic", rz_tolerance=1e-5)

        self.assertEqual(sum(gate[0] == "rz_arbitrary" for gate in spec.gates), 0)
        self.assertEqual(spec.metadata["approximation_mode"], "approx_dyadic")
        self.assertGreater(spec.metadata["approximation_phase_count"], 0)
        self.assertGreater(spec.metadata["approximation_run_count"], 0)
        self.assertGreaterEqual(spec.metadata["approximation_total_run_fro_error"], 0.0)

        statevector = Statevector.from_instruction(qc).data
        for bits in ((0,), (1,)):
            amplitude, _info = compute_circuit_amplitude(spec, [0], bits, as_complex=True)
            expected = complex(statevector[_bits_to_index(bits)])
            self.assertAlmostEqual(amplitude.real, expected.real, places=4)
            self.assertAlmostEqual(amplitude.imag, expected.imag, places=4)
