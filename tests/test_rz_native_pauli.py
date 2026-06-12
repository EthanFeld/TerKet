"""Tests for Pauli expectation behavior on native-enabled RZ paths."""

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

class NativeRZPauliTests(unittest.TestCase):

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

    def test_pauli_expbox_active_two_pi_keeps_global_sign(self):
        spec = make_circuit(1, [("pauli_expbox", ("Z",), (0,), 2.0 * math.pi)])

        amp0, _info0 = compute_circuit_amplitude(spec, [0], [0], as_complex=True)
        amp1, _info1 = compute_circuit_amplitude(spec, [1], [1], as_complex=True)

        self.assertAlmostEqual(amp0.real, -1.0, places=12)
        self.assertAlmostEqual(amp0.imag, 0.0, places=12)
        self.assertAlmostEqual(amp1.real, -1.0, places=12)
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

    def test_rewrite_does_not_crash_on_h_pauli_expbox_h(self):
        gates = (("h", 0), ("pauli_expbox", ("Z",), (0,), 0.37), ("h", 0))

        rewritten = _rewrite_gate_sequence(gates)

        self.assertEqual(rewritten, gates)

    def test_to_qiskit_lowers_pauli_expbox(self):
        theta = 0.43
        spec = make_circuit(2, [("h", 0), ("pauli_expbox", ("X", "Z"), (0, 1), theta)])

        qc = to_qiskit(spec)
        statevector = Statevector.from_instruction(qc).data

        for bits in ((0, 0), (1, 0), (0, 1), (1, 1)):
            actual, _info = compute_circuit_amplitude(spec, [0, 0], bits, as_complex=True)
            expected = complex(statevector[_bits_to_index(bits)])
            self.assertAlmostEqual(actual.real, expected.real, places=12)
            self.assertAlmostEqual(actual.imag, expected.imag, places=12)

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

    def test_pauli_beam_approx_has_first_class_entrypoint(self):
        spec = make_circuit(1, [("pauli_expbox", ("Y",), (0,), 0.37)])

        result = engine.compute_circuit_pauli_expectations_approx(
            spec,
            ["Z"],
            as_complex=True,
            backend="pauli_beam",
        )

        self.assertIsNotNone(result)
        assert result is not None
        value, info = result[0]
        self.assertAlmostEqual(value.real, math.cos(0.37), places=12)
        self.assertEqual(info["phase3_backend"], "pauli_beam_approx")
        self.assertIs(info["is_approximate"], True)

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
