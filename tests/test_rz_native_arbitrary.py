"""Tests for arbitrary-angle native and fallback behavior."""

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

class NativeRZArbitraryTests(unittest.TestCase):

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

    def test_nonunary_arbitrary_phases_use_path_sum_not_branching(self):
        q = PhaseFunction(
            3,
            level=3,
            q0=0,
            q1=[1, 0, 2],
            q2={(0, 2): 1},
            q3={},
        )
        terms = (
            engine._ArbitraryPhaseTerm((1 << 0) | (1 << 1), 0, math.pi / 7.0),
            engine._ArbitraryPhaseTerm((1 << 1) | (1 << 2), 1, -0.37),
        )

        expected = 0j
        for assignment in range(1 << q.n):
            bits = tuple((assignment >> idx) & 1 for idx in range(q.n))
            weight = cmath.exp(2j * math.pi * float(q.evaluate(bits)))
            for term in terms:
                bit = (int(term.row_mask) & assignment).bit_count() & 1
                if bit ^ int(term.offset):
                    weight *= cmath.exp(1j * float(term.angle))
            expected += weight

        actual, width, backend, metadata = engine._sum_with_arbitrary_phases_scaled(q, terms)
        self.assertLessEqual(width, 3)
        self.assertEqual(backend, "arbitrary_path_sum")
        self.assertEqual(metadata, {})
        actual_complex = engine._scaled_to_complex(actual)
        self.assertAlmostEqual(actual_complex.real, expected.real, places=12)
        self.assertAlmostEqual(actual_complex.imag, expected.imag, places=12)

    def test_arbitrary_path_sum_width_limit_is_native_aware(self):
        q = PhaseFunction(4, level=3, q1=[0] * 4, q2={}, q3={})
        terms = (
            engine._ArbitraryPhaseTerm((1 << 0) | (1 << 1), 0, math.pi / 11.0),
            engine._ArbitraryPhaseTerm((1 << 2) | (1 << 3), 1, -0.29),
        )

        class NativeStub:
            @staticmethod
            def support_from_mask(mask):
                return tuple(idx for idx in range(int(mask).bit_length()) if (int(mask) >> idx) & 1)

            def sum_factor_tables_scaled(self):
                raise AssertionError("not called")

        with mock.patch.object(
            engine,
            "_schur_native",
            NativeStub(),
        ), mock.patch.object(
            engine,
            "_factor_scope_order",
            return_value=(list(range(q.n)), 25),
        ), mock.patch.object(
            engine,
            "_estimate_factor_table_dp_cost",
            return_value=(1_000, 1 << 20),
        ), mock.patch.object(
            engine,
            "_sum_factor_tables_scaled",
            return_value=((3.0 + 0.0j, 0), 25),
        ):
            result, width, backend, _metadata = engine._sum_with_arbitrary_phases_scaled(q, terms)

        self.assertEqual(width, 25)
        self.assertEqual(result, (3.0 + 0.0j, 0))
        self.assertEqual(backend, "arbitrary_path_sum")

    def test_arbitrary_path_sum_requires_native_when_available(self):
        q = PhaseFunction(2, level=3, q1=[0, 0], q2={}, q3={})
        terms = (engine._ArbitraryPhaseTerm((1 << 0) | (1 << 1), 0, math.pi / 5.0),)

        class NativeStub:
            @staticmethod
            def support_from_mask(mask):
                return tuple(idx for idx in range(int(mask).bit_length()) if (int(mask) >> idx) & 1)

            @staticmethod
            def sum_factor_tables_scaled(*_args, **_kwargs):
                raise ValueError("native failed")

        with mock.patch.object(engine, "_schur_native", NativeStub()):
            with self.assertRaisesRegex(RuntimeError, "Native factor-table path-sum backend failed"):
                engine._sum_with_arbitrary_phases_scaled(q, terms)

    def test_arbitrary_factor_cutset_sum_matches_bruteforce(self):
        q = PhaseFunction(
            4,
            level=3,
            q1=[1, 0, 2, 3],
            q2={(0, 1): 1, (1, 2): 1, (2, 3): 1},
            q3={},
        )
        terms = (
            engine._ArbitraryPhaseTerm((1 << 0) | (1 << 2), 0, math.pi / 9.0),
            engine._ArbitraryPhaseTerm((1 << 1) | (1 << 3), 1, -0.41),
        )
        scalar, factors = engine._build_cubic_factors_scaled(q)
        scalar = engine._mul_scaled_complex(
            scalar,
            engine._add_arbitrary_phase_factors_scaled(factors, terms),
        )
        plan = engine._ArbitraryFactorCutsetPlan(
            cutset=(1,),
            residual_order=(0, 1, 2),
            residual_width=3,
            residual_work=1,
            residual_table_entries=4,
        )

        actual, _width = engine._sum_factor_tables_with_cutset_scaled(
            q.n,
            factors,
            plan,
            scalar=scalar,
            require_native=False,
        )

        expected = 0j
        for assignment in range(1 << q.n):
            bits = tuple((assignment >> idx) & 1 for idx in range(q.n))
            weight = cmath.exp(2j * math.pi * float(q.evaluate(bits)))
            for term in terms:
                if ((int(term.row_mask) & assignment).bit_count() & 1) ^ int(term.offset):
                    weight *= cmath.exp(1j * float(term.angle))
            expected += weight
        self.assertAlmostEqual(abs(engine._scaled_to_complex(actual) - expected), 0.0, places=12)

    def test_solve_arbitrary_exact_rejects_over_limit_without_cutset(self):
        q = PhaseFunction(
            3,
            level=3,
            q1=[0, 0, 0],
            q2={(0, 1): 1, (1, 2): 1, (0, 2): 1},
            q3={},
        )
        terms = (engine._ArbitraryPhaseTerm(1 << 0, 0, math.pi / 7.0),)

        with mock.patch.object(
            engine,
            "_factor_scope_order",
            return_value=([0, 1, 2], 99),
        ), mock.patch.object(
            engine,
            "_estimate_factor_table_dp_cost",
            return_value=(engine._MAX_ARBITRARY_PATH_SUM_WORK + 1, 1 << 20),
        ), mock.patch.object(
            engine,
            "_find_arbitrary_factor_cutset_plan",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "Cannot compute amplitude directly"):
                engine.solve_arbitrary_exact(q, terms)

    def test_arbitrary_rz_pauli_expectation_not_forced_to_one(self):
        target = 0.6
        theta = math.acos(target)
        spec = make_circuit(1, [("h", 0), ("rz_arbitrary", 0, theta)])

        [(value, info)] = engine.compute_circuit_pauli_expectations(
            spec,
            ["X"],
            as_complex=True,
            allow_tensor_contraction=True,
        )

        self.assertAlmostEqual(value.real, target, places=12)
        self.assertAlmostEqual(value.imag, 0.0, places=12)
        self.assertNotAlmostEqual(value.real, 1.0, places=6)
