from __future__ import annotations

import cmath
import math
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

from terket import compute_circuit_amplitude
from terket.circuit_spec import from_qiskit, parse_openqasm2
from terket.cubic_arithmetic import PhaseFunction
from terket import engine
from terket.benchmarking.head_to_head_cases import (
    SUPPORTED_BASIS as HEAD_TO_HEAD_SUPPORTED_BASIS,
    build_approximate_qft,
    build_repetition_magic_round,
    transpile_to_supported_basis as transpile_head_to_head,
)
from terket.benchmarking.structured_cases import (
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

    def test_q3_free_unary_arbitrary_phase_helper_matches_bruteforce(self):
        q = PhaseFunction(
            4,
            level=3,
            q0=0,
            q1=[1, 4, 7, 2],
            q2={
                (0, 1): 1,
                (1, 2): 2,
                (2, 3): 1,
            },
            q3={},
        )
        terms = (
            engine._ArbitraryPhaseTerm(1 << 0, 0, math.pi / 7.0),
            engine._ArbitraryPhaseTerm(1 << 1, 1, -math.pi / 5.0),
            engine._ArbitraryPhaseTerm(1 << 3, 0, 0.37),
        )

        expected = 0j
        for assignment in range(1 << q.n):
            bits = tuple((assignment >> idx) & 1 for idx in range(q.n))
            weight = cmath.exp(2j * math.pi * float(q.evaluate(bits)))
            for term in terms:
                bit = bits[term.row_mask.bit_length() - 1] ^ int(term.offset)
                if bit:
                    weight *= cmath.exp(1j * float(term.angle))
            expected += weight

        actual = engine._scaled_to_complex(
            engine._sum_q3_free_with_unary_arbitrary_phases_scaled(q, terms)
        )
        self.assertAlmostEqual(actual.real, expected.real, places=12)
        self.assertAlmostEqual(actual.imag, expected.imag, places=12)

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


class SubcircuitMacroTests(unittest.TestCase):
    def _state_signature(self, state: engine.SchurState):
        return (
            state.n,
            state.m,
            tuple(state.eps),
            tuple(state.eps0),
            state.q.level,
            state.q.q0,
            tuple(state.q.q1),
            tuple(sorted(state.q.q2.items())),
            tuple(sorted(state.q.q3.items())),
            state.scalar,
            state.scalar_half_pow2,
            tuple(state.output_refcount),
            tuple(state._arbitrary_phases),
        )

    def _build_replayed_states(self, spec):
        macro_state = engine.SchurState(spec.n_qubits)
        linear_state = engine.SchurState(spec.n_qubits)
        engine._SUBCIRCUIT_MACRO_PLAN_CACHE.clear()
        with mock.patch.object(engine, "_SUBCIRCUIT_MACRO_MIN_TOTAL_GATES", 0):
            seed_state = engine.SchurState(spec.n_qubits)
            engine._apply_gate_sequence_to_state(seed_state, spec.gates)
            engine._apply_gate_sequence_to_state(macro_state, spec.gates)
        engine._apply_gate_sequence_to_state_linear(linear_state, spec.gates)
        macro_state._flush_pending_dead_variables()
        linear_state._flush_pending_dead_variables()
        return macro_state, linear_state

    def test_repeated_allocation_free_subcircuits_match_linear_replay(self):
        spec = build_repetition_magic_round(64)
        macro_state, linear_state = self._build_replayed_states(spec)
        self.assertEqual(self._state_signature(macro_state), self._state_signature(linear_state))

    def test_repeated_fresh_variable_subcircuits_match_linear_replay(self):
        spec = build_approximate_qft(64)
        macro_state, linear_state = self._build_replayed_states(spec)
        self.assertEqual(self._state_signature(macro_state), self._state_signature(linear_state))

    def test_repeated_benchmark_subcircuits_compile_macro(self):
        spec = build_approximate_qft(64)
        state = engine.SchurState(spec.n_qubits)
        engine._SUBCIRCUIT_MACRO_PLAN_CACHE.clear()
        with mock.patch.object(engine, "_SUBCIRCUIT_MACRO_MIN_TOTAL_GATES", 0):
            seed_state = engine.SchurState(spec.n_qubits)
            engine._apply_gate_sequence_to_state(seed_state, spec.gates)
        with mock.patch.object(engine, "_SUBCIRCUIT_MACRO_MIN_TOTAL_GATES", 0), mock.patch.object(
            engine,
            "_compile_subcircuit_macro",
            wraps=engine._compile_subcircuit_macro,
        ) as compile_macro:
            engine._apply_gate_sequence_to_state(state, spec.gates)
        self.assertGreater(compile_macro.call_count, 0)


if __name__ == "__main__":
    unittest.main()
