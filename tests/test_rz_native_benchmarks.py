"""Tests for RZ-native benchmark helper behavior."""

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
            self.assertLess(len(spec.gates), 128)

    def test_benchmark_transpile_helpers_prefer_direct_import(self):
        head_qc = build_qaoa_ring_logical(8)
        structured_qc = build_mm_hidden_shift_logical(3, (0, 1, 0))

        with mock.patch("terket.benchmarking.head_to_head_cases.transpile", side_effect=AssertionError("unexpected")):
            head_spec = transpile_head_to_head(head_qc)
        with mock.patch("terket.benchmarking.structured_cases.transpile", side_effect=AssertionError("unexpected")):
            structured_spec = transpile_structured(structured_qc)

        self.assertEqual(head_spec, from_qiskit(head_qc, rz_compile_mode="dyadic"))
        self.assertEqual(structured_spec, from_qiskit(structured_qc, rz_compile_mode="dyadic"))

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
