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
        self.assertIs(metadata["is_approximate"], False)
        self.assertEqual(metadata["approx_validation"], "exact")
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

    def test_arbitrary_bethe_bp_is_exact_on_pairwise_tree(self):
        q = PhaseFunction(
            4,
            level=3,
            q1=[1, 2, 0, 3],
            q2={(0, 1): 1, (1, 2): 2, (1, 3): 1},
            q3={},
        )
        terms = (
            engine._ArbitraryPhaseTerm(1 << 0, 0, math.pi / 7.0),
            engine._ArbitraryPhaseTerm(1 << 3, 1, -0.25),
        )
        scalar, factors = engine._build_cubic_factors_scaled(q)
        scalar = engine._mul_scaled_complex(
            scalar,
            engine._add_arbitrary_phase_factors_scaled(factors, terms),
        )

        actual, _width = engine._sum_pairwise_factor_graph_bethe_scaled(q.n, factors, scalar=scalar)

        expected = 0j
        for assignment in range(1 << q.n):
            bits = tuple((assignment >> idx) & 1 for idx in range(q.n))
            weight = cmath.exp(2j * math.pi * float(q.evaluate(bits)))
            for term in terms:
                if ((int(term.row_mask) & assignment).bit_count() & 1) ^ int(term.offset):
                    weight *= cmath.exp(1j * float(term.angle))
            expected += weight
        self.assertAlmostEqual(abs(engine._scaled_to_complex(actual) - expected), 0.0, places=9)

    def test_arbitrary_factor_bethe_bp_converges_on_factor_tree(self):
        def table(scope_size: int, offset: float):
            return [
                engine._make_scaled_complex(cmath.exp(1j * (offset + 0.17 * assignment)))
                for assignment in range(1 << scope_size)
            ]

        n_vars = 6
        factors = {
            (0, 1, 2): table(3, 0.11),
            (2, 3, 4): table(3, -0.23),
            (4, 5): table(2, 0.41),
        }
        scalar = engine._make_scaled_complex(0.7 + 0.2j)
        actual, _width = engine._sum_factor_graph_bethe_scaled(n_vars, factors, scalar=scalar)

        expected = 0j
        for assignment in range(1 << n_vars):
            weight = engine._scaled_to_complex(scalar)
            for scope, factor_table in factors.items():
                factor_assignment = 0
                for pos, var in enumerate(scope):
                    factor_assignment |= ((assignment >> var) & 1) << pos
                weight *= engine._scaled_to_complex(factor_table[factor_assignment])
            expected += weight

        self.assertAlmostEqual(abs(engine._scaled_to_complex(actual) - expected), 0.0, places=10)

    def test_sparse_parity_bethe_handles_wide_arbitrary_factor_without_dense_table(self):
        n_vars = 9
        q = PhaseFunction(n_vars, level=3, q1=[0] * n_vars, q2={}, q3={})
        term = engine._ArbitraryPhaseTerm((1 << n_vars) - 1, 0, math.pi / 7.0)

        with mock.patch.object(engine, "_MAX_ARBITRARY_PHASE_FACTOR_SCOPE", 8):
            actual, width, backend, _metadata = engine._sum_with_arbitrary_phases_scaled(
                q,
                (term,),
                allow_approximate=True,
            )

        phase = cmath.exp(1j * float(term.angle))
        expected = (1 << (n_vars - 1)) * (1.0 + phase)
        self.assertEqual(width, n_vars)
        self.assertEqual(backend, "arbitrary_sparse_parity_bethe_bp")
        self.assertIs(_metadata["is_approximate"], True)
        self.assertEqual(_metadata["approx_backend"], backend)
        self.assertEqual(_metadata["approx_validation"], "factor_graph_forest_exact")
        self.assertAlmostEqual(abs(engine._scaled_to_complex(actual) - expected), 0.0, places=9)

    def test_solve_arbitrary_exact_never_calls_bp_fallback(self):
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
        ), mock.patch.object(
            engine,
            "_sum_pairwise_factor_graph_bethe_scaled",
            side_effect=AssertionError("exact path must not call BP"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Cannot compute amplitude directly"):
                engine.solve_arbitrary_exact(q, terms)

    def test_arbitrary_bethe_bp_heuristic_accepts_stable_loopy_ensemble(self):
        q = PhaseFunction(
            3,
            level=3,
            q1=[0, 0, 0],
            q2={(0, 1): 1, (1, 2): 1, (0, 2): 1},
            q3={},
        )
        terms = (engine._ArbitraryPhaseTerm(1 << 0, 0, math.pi / 7.0),)

        def stable_bp(*_args, **kwargs):
            self.assertFalse(kwargs["require_forest"])
            return engine._make_scaled_complex(0.25 + 0.0j), 3

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
            "_sum_pairwise_factor_graph_bethe_scaled",
            side_effect=stable_bp,
        ):
            actual, width, backend, metadata = engine._sum_with_arbitrary_phases_scaled(
                q,
                terms,
                allow_approximate=True,
            )

        self.assertEqual(width, 3)
        self.assertEqual(backend, "arbitrary_bethe_bp_heuristic")
        self.assertIs(metadata["is_approximate"], True)
        self.assertEqual(metadata["approx_backend"], backend)
        self.assertEqual(metadata["approx_validation"], "loopy_ensemble_thresholds")
        self.assertEqual(metadata["bp_heuristic_ensemble_size"], len(engine._ARBITRARY_BP_HEURISTIC_SCHEDULES))
        self.assertAlmostEqual(abs(engine._scaled_to_complex(actual) - 0.25), 0.0, places=12)

    def test_arbitrary_bethe_bp_heuristic_rejects_unstable_loopy_ensemble(self):
        q = PhaseFunction(
            3,
            level=3,
            q1=[0, 0, 0],
            q2={(0, 1): 1, (1, 2): 1, (0, 2): 1},
            q3={},
        )
        terms = (engine._ArbitraryPhaseTerm(1 << 0, 0, math.pi / 7.0),)
        values = iter((
            engine._make_scaled_complex(0.25 + 0.0j),
            engine._make_scaled_complex(0.001 + 0.0j),
            engine._make_scaled_complex(0.25j),
        ))

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
            "_sum_pairwise_factor_graph_bethe_scaled",
            side_effect=lambda *_args, **_kwargs: (next(values), 3),
        ):
            with self.assertRaisesRegex(RuntimeError, "Cannot compute amplitude directly"):
                engine._sum_with_arbitrary_phases_scaled(q, terms, allow_approximate=True)

    def test_arbitrary_bp_direct_amplitude_rejects_impossible_scale(self):
        info = engine._info(1, 0, 0, 0, 1, phase3_backend="arbitrary_bethe_bp")

        with self.assertRaisesRegex(RuntimeError, "implied output probability"):
            engine._raise_if_invalid_arbitrary_bp_amplitude(info, (1.0 + 0.0j, 20))

    def test_analyze_arbitrary_bp_marks_invalid_scale_without_raising(self):
        spec = make_circuit(1, [("h", 0), ("rz_arbitrary", 0, 0.37), ("h", 0)])

        def fake_sum(_q, _terms, *, allow_approximate=False, **_kwargs):
            return (1.0 + 0.0j, 20), 1, "arbitrary_bethe_bp", {}

        with mock.patch.object(engine, "_sum_with_arbitrary_phases_scaled", side_effect=fake_sum):
            [info] = engine.analyze_amplitudes(
                spec,
                [0],
                [[0]],
                allow_tensor_contraction=True,
                solver_config=engine.SolverConfig(allow_approximate=True),
            )

        self.assertEqual(info["phase3_backend"], "arbitrary_bethe_bp_invalid_scale")
        self.assertEqual(info["bp_invalid_reason"], "implied_probability_exceeds_one")
        self.assertGreater(info["bp_log2_probability"], 0.0)

    def test_invalid_bp_public_amplitude_rejects_instead_of_retrying(self):
        spec = make_circuit(1, [("h", 0), ("rz_arbitrary", 0, 0.37), ("h", 0)])

        def fake_sum(_q, _terms, *, allow_approximate=False, **_kwargs):
            return (1.0 + 0.0j, 20), 1, "arbitrary_bethe_bp", {}

        with mock.patch.object(engine, "_sum_with_arbitrary_phases_scaled", side_effect=fake_sum):
            with self.assertRaisesRegex(RuntimeError, "implied output probability"):
                compute_circuit_amplitude(
                    spec,
                    [0],
                    [0],
                    as_complex=True,
                    allow_tensor_contraction=True,
                    solver_config=engine.SolverConfig(allow_approximate=True),
                )

    def test_arbitrary_bp_requires_solver_config_opt_in(self):
        spec = make_circuit(1, [("h", 0), ("rz_arbitrary", 0, 0.37), ("h", 0)])
        seen: list[bool] = []

        def fake_sum(_q, _terms, *, allow_approximate=False):
            seen.append(bool(allow_approximate))
            return engine._ONE_SCALED, 0, "arbitrary_path_sum", {}

        with mock.patch.object(engine, "_sum_with_arbitrary_phases_scaled", side_effect=fake_sum):
            compute_circuit_amplitude(spec, [0], [0], as_complex=True, allow_tensor_contraction=True)
        self.assertEqual(seen, [False])

        seen.clear()
        with mock.patch.object(engine, "_sum_with_arbitrary_phases_scaled", side_effect=fake_sum):
            compute_circuit_amplitude(
                spec,
                [0],
                [0],
                as_complex=True,
                allow_tensor_contraction=True,
                solver_config=engine.SolverConfig(allow_approximate=True),
            )
        self.assertEqual(seen, [True])

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
