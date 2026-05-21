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
from terket.circuit_spec import from_qiskit, parse_openqasm2
from terket.cubic_arithmetic import PhaseFunction
from terket import engine
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

        actual, width, backend, _metadata = engine._sum_with_arbitrary_phases_scaled(q, terms)
        self.assertLessEqual(width, 3)
        self.assertEqual(backend, "arbitrary_path_sum")
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
        self.assertAlmostEqual(abs(engine._scaled_to_complex(actual) - expected), 0.0, places=9)

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

    def test_native_mps_approx_handles_nonadjacent_cnot_order(self):
        spec = make_circuit(3, [("h", 2), ("cnot", 2, 0)])

        values = engine._native_mps_approx_pauli_expectations(
            spec,
            [0, 0, 0],
            ["ZIZ", "XIX"],
            max_bond=4,
        )

        self.assertIsNotNone(values)
        assert values is not None
        self.assertAlmostEqual(values[0].real, 1.0, places=12)
        self.assertAlmostEqual(values[0].imag, 0.0, places=12)
        self.assertAlmostEqual(values[1].real, 1.0, places=12)
        self.assertAlmostEqual(values[1].imag, 0.0, places=12)

    def test_native_mps_approx_handles_rzz_dyadic(self):
        spec = make_circuit(2, [("h", 0), ("h", 1), ("rzz_dyadic", 0, 1, 1, 3)])
        observables = ["XX", "ZZ"]

        values = engine._native_mps_approx_pauli_expectations(
            spec,
            [0, 0],
            observables,
            max_bond=4,
        )
        exact = engine.compute_circuit_pauli_expectations(
            spec,
            observables,
            as_complex=True,
            allow_tensor_contraction=False,
        )

        self.assertIsNotNone(values)
        assert values is not None
        for actual, (expected, _info) in zip(values, exact):
            self.assertAlmostEqual(actual.real, expected.real, places=12)
            self.assertAlmostEqual(actual.imag, expected.imag, places=12)

    def test_pauli_expbox_single_pauli_matches_rotation_definition(self):
        theta = 0.37
        spec = make_circuit(1, [("pauli_expbox", ("Y",), (0,), theta)])

        amp0, _info0 = compute_circuit_amplitude(spec, [0], [0], as_complex=True)
        amp1, _info1 = compute_circuit_amplitude(spec, [0], [1], as_complex=True)

        self.assertAlmostEqual(amp0.real, math.cos(0.5 * theta), places=12)
        self.assertAlmostEqual(amp0.imag, 0.0, places=12)
        self.assertAlmostEqual(amp1.real, math.sin(0.5 * theta), places=12)
        self.assertAlmostEqual(amp1.imag, 0.0, places=12)

    def test_pauli_expbox_multi_pauli_matches_direct_matrix(self):
        theta = -0.91
        spec = make_circuit(2, [("pauli_expbox", ("X", "Z"), (0, 1), theta)])

        amp00, _ = compute_circuit_amplitude(spec, [0, 1], [0, 1], as_complex=True)
        amp10, _ = compute_circuit_amplitude(spec, [0, 1], [1, 1], as_complex=True)

        self.assertAlmostEqual(amp00.real, math.cos(0.5 * theta), places=12)
        self.assertAlmostEqual(amp00.imag, 0.0, places=12)
        self.assertAlmostEqual(amp10.real, 0.0, places=12)
        self.assertAlmostEqual(amp10.imag, 1.0 * math.sin(0.5 * theta), places=12)

    def test_pauli_expbox_schur_path_does_not_materialize_cnot_ladder(self):
        spec = make_circuit(3, [("pauli_expbox", ("X", "Z", "Y"), (0, 1, 2), 0.43)])

        with mock.patch.object(engine.SchurState, "cnot", side_effect=AssertionError("unexpected cnot ladder")):
            amp, _info = compute_circuit_amplitude(spec, [0, 0, 0], [0, 0, 0], as_complex=True)

        self.assertAlmostEqual(amp.real, math.cos(0.5 * 0.43), places=12)
        self.assertAlmostEqual(amp.imag, 0.0, places=12)

    def test_pauli_expbox_dyadic_snap_uses_phase_polynomial(self):
        spec = make_circuit(2, [("pauli_expbox", ("X", "Z"), (0, 1), math.pi / 4.0)])

        with mock.patch.dict(os.environ, {"TERKET_PAULI_EXPBOX_DYADIC_LEVEL": "3"}):
            state = engine.build_state(spec.n_qubits, spec.gates, [0, 1])
            amp, _info = compute_circuit_amplitude(spec, [0, 1], [0, 1], as_complex=True)

        self.assertFalse(state._arbitrary_phases)
        self.assertAlmostEqual(amp.real, math.cos(math.pi / 8.0), places=12)
        self.assertAlmostEqual(amp.imag, 0.0, places=12)

    def test_native_mps_approx_handles_pauli_expbox(self):
        theta = 0.43
        spec = make_circuit(2, [("pauli_expbox", ("X", "Z"), (0, 1), theta)])

        values = engine._native_mps_approx_pauli_expectations(
            spec,
            [0, 0],
            ["XI", "IZ"],
            max_bond=4,
        )
        exact = engine.compute_circuit_pauli_expectations(
            spec,
            ["XI", "IZ"],
            input_bits=[0, 0],
            as_complex=True,
            allow_tensor_contraction=False,
        )

        self.assertIsNotNone(values)
        assert values is not None
        for actual, (expected, _info) in zip(values, exact):
            self.assertAlmostEqual(actual.real, expected.real, places=12)
            self.assertAlmostEqual(actual.imag, expected.imag, places=12)

    def test_native_mps_approx_mirror_fidelity_exact_when_untruncated(self):
        spec = make_circuit(
            2,
            [
                ("h", 0),
                ("cnot", 0, 1),
                ("pauli_expbox", ("Z", "Z"), (0, 1), 0.37),
            ],
        )
        dagger = make_circuit(
            2,
            [
                ("pauli_expbox", ("Z", "Z"), (0, 1), -0.37),
                ("cnot", 0, 1),
                ("h", 0),
            ],
        )

        fidelity = engine._native_mps_approx_mirror_fidelity(spec, dagger, [0, 0], max_bond=4)

        self.assertIsNotNone(fidelity)
        assert fidelity is not None
        self.assertAlmostEqual(fidelity, 1.0, places=12)

    def test_pauli_beam_approx_is_exact_without_pruning(self):
        spec = make_circuit(
            3,
            [
                ("x", 0),
                ("pauli_expbox", ("X", "Z", "Y"), (0, 1, 2), 0.43),
                ("pauli_expbox", ("Z", "Z", "Z"), (0, 1, 2), -0.2),
            ],
        )
        observables = ["ZZZ", "XII", "IYY"]

        values = engine._pauli_beam_approx_pauli_expectations(
            spec,
            [0, 0, 0],
            observables,
            max_terms=10000,
        )
        exact = engine.compute_circuit_pauli_expectations(
            spec,
            observables,
            input_bits=[0, 0, 0],
            as_complex=True,
            allow_tensor_contraction=False,
        )

        self.assertIsNotNone(values)
        assert values is not None
        for actual, (expected, _info) in zip(values, exact):
            self.assertAlmostEqual(actual.real, expected.real, places=12)
            self.assertAlmostEqual(actual.imag, expected.imag, places=12)

    def test_pauli_product_phase_fast_path_matches_table(self):
        pauli_masks = {
            "I": (0, 0),
            "X": (1, 0),
            "Z": (0, 1),
            "Y": (1, 1),
        }
        for left_name, left_masks in pauli_masks.items():
            for right_name, right_masks in pauli_masks.items():
                left_code = (1 if "X" in left_name or "Y" in left_name else 0) | (
                    2 if "Z" in left_name or "Y" in left_name else 0
                )
                right_code = (1 if "X" in right_name or "Y" in right_name else 0) | (
                    2 if "Z" in right_name or "Y" in right_name else 0
                )
                expected = engine._PAULI_PRODUCT_PHASE[(left_code, right_code)]
                self.assertEqual(engine._pauli_product_phase(left_masks, right_masks), expected)

    def test_pauli_beam_processes_circuit_once_for_many_observables(self):
        spec = make_circuit(
            3,
            [
                ("x", 0),
                ("pauli_expbox", ("X", "Z", "Y"), (0, 1, 2), 0.43),
                ("pauli_expbox", ("Z", "Z", "Z"), (0, 1, 2), -0.2),
            ],
        )
        observables = ["ZZZ", "XII", "IYY"]

        expected = [
            engine._pauli_beam_approx_pauli_expectations(spec, [0, 0, 0], [observable], max_terms=2)[0]
            for observable in observables
        ]
        with mock.patch.object(
            engine,
            "_pauli_masks_from_sparse",
            wraps=engine._pauli_masks_from_sparse,
        ) as masks_from_sparse:
            actual = engine._pauli_beam_approx_pauli_expectations(
                spec,
                [0, 0, 0],
                observables,
                max_terms=2,
            )

        self.assertEqual(masks_from_sparse.call_count, 2)
        self.assertEqual(actual, expected)

    def test_pauli_beam_handles_clifford_scaffolding(self):
        spec = make_circuit(
            2,
            [
                ("h", 0),
                ("s", 1),
                ("cnot", 0, 1),
                ("pauli_expbox", ("Z",), (0,), 0.37),
                ("pauli_expbox", ("X",), (1,), -0.19),
                ("sdg", 1),
            ],
        )
        observables = ["XI", "YZ", "ZZ"]

        values = engine._pauli_beam_approx_pauli_expectations(
            spec,
            [0, 0],
            observables,
            max_terms=64,
        )
        exact = engine.compute_circuit_pauli_expectations(
            spec,
            observables,
            as_complex=True,
            allow_tensor_contraction=False,
        )

        self.assertIsNotNone(values)
        assert values is not None
        for actual, (expected, _info) in zip(values, exact):
            self.assertAlmostEqual(actual.real, expected.real, places=12)
            self.assertAlmostEqual(actual.imag, expected.imag, places=12)

    def test_pauli_expbox_approx_fast_path_skips_schur_build(self):
        spec = make_circuit(1, [("pauli_expbox", ("Y",), (0,), 0.37)])

        with mock.patch.object(engine, "build_state", side_effect=AssertionError("unexpected")):
            [(value, info)] = engine.compute_circuit_pauli_expectations(
                spec,
                ["Z"],
                as_complex=True,
                solver_config=engine.SolverConfig(allow_approximate=True),
            )

        self.assertAlmostEqual(value.real, math.cos(0.37), places=12)
        self.assertEqual(info["phase3_backend"], "pauli_beam_approx")

    def test_pauli_expbox_beam_handles_chemistry_x_prefix(self):
        spec = make_circuit(
            2,
            [
                ("x", 0),
                ("pauli_expbox", ("X", "Z"), (0, 1), 0.37),
                ("pauli_expbox", ("Z", "Z"), (0, 1), -0.11),
            ],
        )

        with mock.patch.object(engine, "build_state", side_effect=AssertionError("unexpected")):
            [(value, info)] = engine.compute_circuit_pauli_expectations(
                spec,
                ["ZZ"],
                as_complex=True,
                solver_config=engine.SolverConfig(allow_approximate=True),
            )

        self.assertTrue(math.isfinite(value.real))
        self.assertAlmostEqual(value.imag, 0.0, places=12)
        self.assertEqual(info["phase3_backend"], "pauli_beam_approx")

    def test_arbitrary_pauli_approx_uses_bp_path_for_xy_observable(self):
        spec = make_circuit(2, [("h", 0), ("rz_arbitrary", 0, 0.37), ("cnot", 0, 1)])
        seen: list[bool] = []

        def fake_sum(_q, _terms, *, allow_approximate=False):
            seen.append(bool(allow_approximate))
            return engine._ONE_SCALED, 0, "arbitrary_bethe_bp", {}

        with mock.patch.object(engine, "_sum_with_arbitrary_phases_scaled", side_effect=fake_sum):
            [(value, info)] = engine.compute_circuit_pauli_expectations(
                spec,
                ["XY"],
                as_complex=True,
                solver_config=engine.SolverConfig(allow_approximate=True),
            )

        self.assertEqual(seen, [True])
        self.assertAlmostEqual(value.real, 0.5, places=12)
        self.assertEqual(info["phase3_backend"], "arbitrary_bethe_bp_normalized")

    def test_pauli_beam_miss_falls_through_to_bp_before_native_mps(self):
        spec = make_circuit(1, [("h", 0), ("pauli_expbox", ("Z",), (0,), 0.37)])
        seen: list[bool] = []

        def fake_sum(_q, _terms, *, allow_approximate=False):
            seen.append(bool(allow_approximate))
            return engine._ONE_SCALED, 0, "arbitrary_bethe_bp", {}

        with mock.patch.object(
            engine,
            "_pauli_beam_approx_pauli_expectations",
            return_value=None,
        ) as beam, mock.patch.object(
            engine,
            "_native_mps_approx_pauli_expectations",
            side_effect=AssertionError("mps should wait for bp rejection"),
        ), mock.patch.object(
            engine,
            "_sum_with_arbitrary_phases_scaled",
            side_effect=fake_sum,
        ):
            [(value, info)] = engine.compute_circuit_pauli_expectations(
                spec,
                ["X"],
                as_complex=True,
                solver_config=engine.SolverConfig(allow_approximate=True),
            )

        beam.assert_called_once()
        self.assertEqual(seen, [True])
        self.assertAlmostEqual(value.real, 0.5, places=12)
        self.assertEqual(info["phase3_backend"], "arbitrary_bethe_bp_normalized")

    def test_arbitrary_pauli_approx_falls_back_to_native_mps_after_bp_rejects(self):
        spec = make_circuit(2, [("h", 0), ("rz_arbitrary", 0, 0.37), ("cnot", 0, 1)])
        seen: list[bool] = []

        def fake_sum(_q, _terms, *, allow_approximate=False):
            seen.append(bool(allow_approximate))
            raise RuntimeError("Loopy BP heuristic failed acceptance thresholds.")

        with mock.patch.object(
            engine,
            "_sum_with_arbitrary_phases_scaled",
            side_effect=fake_sum,
        ), mock.patch.object(
            engine,
            "_native_mps_approx_pauli_expectations",
            return_value=[0.25 + 0.0j],
        ) as native_mps:
            [(value, info)] = engine.compute_circuit_pauli_expectations(
                spec,
                ["XY"],
                as_complex=True,
                solver_config=engine.SolverConfig(allow_approximate=True),
            )

        self.assertEqual(seen, [True])
        native_mps.assert_called_once()
        self.assertAlmostEqual(value.real, 0.25, places=12)
        self.assertEqual(info["phase3_backend"], "native_mps_approx")
        self.assertEqual(info["fallback_reason"], "arbitrary_bp_unavailable")

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


if __name__ == "__main__":
    unittest.main()
