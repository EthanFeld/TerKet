"""Tests for batched Phase-3 treewidth execution behavior."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import terket
from terket import engine


class Phase3TreewidthBatchTests(unittest.TestCase):
    def test_shared_support_batch_matches_individual_treewidth(self):
        q0 = engine._phase_function_from_parts(
            4,
            level=4,
            q0=0,
            q1=[1, 2, 0, 4],
            q2={(0, 1): 1, (1, 2): 2, (2, 3): 1},
            q3={(0, 1, 2): 1, (1, 2, 3): 1},
        )
        q1 = engine._phase_function_from_parts(
            4,
            level=4,
            q0=1 / 8,
            q1=[5, 0, 6, 1],
            q2={(0, 1): 3, (1, 2): 1, (2, 3): 2},
            q3={(0, 1, 2): 1, (1, 2, 3): 2},
        )
        order = [0, 1, 2, 3]

        batch_totals, batch_width = engine._sum_via_treewidth_dp_scaled_batch_shared_support(
            [q0, q1],
            order,
        )
        expected0, width0 = engine._sum_via_treewidth_dp_scaled(q0, order)
        expected1, width1 = engine._sum_via_treewidth_dp_scaled(q1, order)

        self.assertEqual(batch_width, width0)
        self.assertEqual(batch_width, width1)
        self.assertAlmostEqual(
            abs(engine._scaled_to_complex(batch_totals[0]) - engine._scaled_to_complex(expected0)),
            0.0,
        )
        self.assertAlmostEqual(
            abs(engine._scaled_to_complex(batch_totals[1]) - engine._scaled_to_complex(expected1)),
            0.0,
        )

    @unittest.skipIf(engine._schur_native is None, "native accelerator unavailable")
    def test_native_shared_support_batch_matches_individual_treewidth(self):
        q0 = engine._phase_function_from_parts(
            5,
            level=3,
            q0=0,
            q1=[1, 2, 3, 4, 5],
            q2={(0, 1): 1, (1, 2): 2, (2, 3): 3},
            q3={(0, 2, 4): 1, (1, 3, 4): 1},
        )
        q1 = engine._phase_function_from_parts(
            5,
            level=3,
            q0=1 / 8,
            q1=[7, 0, 6, 1, 2],
            q2={(0, 1): 3, (1, 2): 1, (2, 3): 2},
            q3={(0, 2, 4): 1, (1, 3, 4): 1},
        )
        order = [0, 1, 2, 3, 4]

        batch = engine._sum_native_level3_phase3_treewidth_batch_shared_support([q0, q1], order)
        self.assertIsNotNone(batch)
        assert batch is not None
        batch_totals, batch_width = batch
        expected0, width0 = engine._sum_via_treewidth_dp_scaled(q0, order)
        expected1, width1 = engine._sum_via_treewidth_dp_scaled(q1, order)

        self.assertEqual(batch_width, width0)
        self.assertEqual(batch_width, width1)
        self.assertAlmostEqual(
            abs(engine._scaled_to_complex(batch_totals[0]) - engine._scaled_to_complex(expected0)),
            0.0,
        )
        self.assertAlmostEqual(
            abs(engine._scaled_to_complex(batch_totals[1]) - engine._scaled_to_complex(expected1)),
            0.0,
        )

    @unittest.skipIf(engine._schur_native is None, "native accelerator unavailable")
    def test_native_generic_shared_support_batch_matches_individual_treewidth(self):
        q0 = engine._phase_function_from_parts(
            4,
            level=5,
            q0=0,
            q1=[1, 7, 0, 9],
            q2={(0, 1): 3, (1, 2): 5, (2, 3): 1},
            q3={(0, 1, 2): 1, (1, 2, 3): 3},
        )
        q1 = engine._phase_function_from_parts(
            4,
            level=5,
            q0=1 / 32,
            q1=[11, 0, 14, 2],
            q2={(0, 1): 7, (1, 2): 1, (2, 3): 6},
            q3={(0, 1, 2): 2, (1, 2, 3): 1},
        )
        order = [0, 1, 2, 3]

        batch = engine._sum_native_phase_function_treewidth_batch_shared_support([q0, q1], order)
        self.assertIsNotNone(batch)
        assert batch is not None
        batch_totals, batch_width = batch
        expected0, width0 = engine._sum_via_treewidth_dp_scaled(q0, order)
        expected1, width1 = engine._sum_via_treewidth_dp_scaled(q1, order)

        self.assertEqual(batch_width, width0)
        self.assertEqual(batch_width, width1)
        self.assertAlmostEqual(
            abs(engine._scaled_to_complex(batch_totals[0]) - engine._scaled_to_complex(expected0)),
            0.0,
        )
        self.assertAlmostEqual(
            abs(engine._scaled_to_complex(batch_totals[1]) - engine._scaled_to_complex(expected1)),
            0.0,
        )

    @unittest.skipIf(engine._schur_native is None, "native accelerator unavailable")
    def test_native_generic_shared_support_batch_handles_odd_batch_size(self):
        q0 = engine._phase_function_from_parts(
            4,
            level=5,
            q0=0,
            q1=[1, 7, 0, 9],
            q2={(0, 1): 3, (1, 2): 5, (2, 3): 1},
            q3={(0, 1, 2): 1, (1, 2, 3): 3},
        )
        q1 = engine._phase_function_from_parts(
            4,
            level=5,
            q0=1 / 32,
            q1=[11, 0, 14, 2],
            q2={(0, 1): 7, (1, 2): 1, (2, 3): 6},
            q3={(0, 1, 2): 2, (1, 2, 3): 1},
        )
        q2 = engine._phase_function_from_parts(
            4,
            level=5,
            q0=3 / 32,
            q1=[4, 5, 3, 10],
            q2={(0, 1): 1, (1, 2): 4, (2, 3): 2},
            q3={(0, 1, 2): 3, (1, 2, 3): 2},
        )
        order = [0, 1, 2, 3]

        batch = engine._sum_native_phase_function_treewidth_batch_shared_support([q0, q1, q2], order)
        self.assertIsNotNone(batch)
        assert batch is not None
        batch_totals, batch_width = batch
        expected_rows = [
            engine._sum_via_treewidth_dp_scaled(q, order)
            for q in (q0, q1, q2)
        ]

        self.assertEqual(batch_width, expected_rows[0][1])
        for batch_total, (expected_total, expected_width) in zip(batch_totals, expected_rows):
            self.assertEqual(batch_width, expected_width)
            self.assertAlmostEqual(
                abs(engine._scaled_to_complex(batch_total) - engine._scaled_to_complex(expected_total)),
                0.0,
            )

    def test_compute_circuit_pauli_expectations_matches_simple_known_values(self):
        circuit = terket.make_circuit(1, [("h", 0)])

        results = terket.compute_circuit_pauli_expectations(
            circuit,
            ["X", "Z", "Y"],
            input_bits=[0],
            as_complex=True,
        )
        amplitudes = [complex(amp) for amp, _info in results]

        self.assertAlmostEqual(abs(amplitudes[0] - (1.0 + 0.0j)), 0.0)
        self.assertAlmostEqual(abs(amplitudes[1] - 0.0j), 0.0)
        self.assertAlmostEqual(abs(amplitudes[2] - 0.0j), 0.0)

    def test_compute_circuit_pauli_expectations_builds_prefix_once(self):
        circuit = terket.make_circuit(2, [("h", 0), ("cnot", 0, 1), ("t", 1)])
        original_build_state = engine.build_state
        build_count = 0

        def counted_build_state(*args, **kwargs):
            nonlocal build_count
            build_count += 1
            return original_build_state(*args, **kwargs)

        with mock.patch.object(engine, "build_state", side_effect=counted_build_state):
            results = terket.compute_circuit_pauli_expectations(
                circuit,
                ["XX", "ZZ", "IZ"],
                input_bits=[0, 0],
                as_complex=True,
            )

        self.assertEqual(len(results), 3)
        self.assertEqual(build_count, 1)

    def test_compute_circuit_pauli_expectations_reuses_identical_suffix_query_cache(self):
        circuit = terket.make_circuit(2, [("h", 0), ("cnot", 0, 1), ("t", 1)])
        original_prepare = engine.SchurState._prepare_echelon
        prepare_count = 0

        def counted_prepare(state):
            nonlocal prepare_count
            prepare_count += 1
            return original_prepare(state)

        with mock.patch.object(engine.SchurState, "_prepare_echelon", new=counted_prepare):
            results = terket.compute_circuit_pauli_expectations(
                circuit,
                ["XX", "ZZ", "IZ"],
                input_bits=[0, 0],
                as_complex=True,
            )

        self.assertEqual(len(results), 3)
        self.assertEqual(prepare_count, 2)

    def test_direct_post_replay_payload_matches_slow_suffix_replay(self):
        circuit = terket.make_circuit(
            2,
            [
                ("sx", 0),
                ("cnot", 0, 1),
                ("t", 1),
                ("rz_pi_16", 0),
                ("s", 0),
            ],
        )
        base_state = engine.build_state(
            circuit.n_qubits,
            circuit.gates,
            [0, 0],
            global_phase_radians=0.0,
        )
        inverse_gates = engine._invert_native_gates(circuit.gates)
        with mock.patch.object(engine, "_DIRECT_POST_REPLAY_MIN_SUFFIX_GATES", 1):
            template = engine._build_direct_post_replay_template(base_state, inverse_gates, 8)

        self.assertIsNotNone(template)
        assert template is not None

        for observable in ("XX", "YZ", "IZ"):
            slow_state = engine._build_post_replay_state(
                base_state,
                engine._pauli_string_gates(observable),
                inverse_gates,
            )
            fast_payload = engine._construct_direct_post_replay_payload(
                base_state,
                observable,
                template,
            )
            self.assertTrue(
                engine._direct_post_replay_payload_matches_state(
                    fast_payload,
                    slow_state,
                    template,
                )
            )

    def test_shared_support_batch_reuses_pre_exact_phase3_escape(self):
        q0 = engine._phase_function_from_parts(
            256,
            level=4,
            q0=0,
            q1=[1] * 256,
            q2={(0, 1): 1},
            q3={(0, 1, 2): 1},
        )
        q1 = engine._phase_function_from_parts(
            256,
            level=4,
            q0=1 / 16,
            q1=[2] * 256,
            q2={(0, 1): 3},
            q3={(0, 1, 2): 1},
        )
        fake_total = engine._make_scaled_complex(1.0 + 0.0j)
        native_symbol = engine._native_symbol

        def fake_native_symbol(name):
            if name in {
                "sum_level3_treewidth_preplanned_batch_array",
                "build_phase_function_treewidth_support_plan",
                "sum_phase_function_treewidth_preplanned_batch_scaled_array",
            }:
                return None
            return native_symbol(name)

        with mock.patch.object(
            engine,
            "_pre_exact_phase3_treewidth_escape",
            return_value=([0], [0, 1, 2], 2, 1, "treewidth_dp"),
        ) as pre_exact, mock.patch.object(
            engine,
            "_native_symbol",
            side_effect=fake_native_symbol,
        ), mock.patch.object(
            engine,
            "_sum_via_treewidth_dp_scaled_batch_shared_support",
            return_value=([fake_total, fake_total], 2),
        ) as batch_sum, mock.patch.object(
            engine,
            "_reduce_and_sum_scaled",
            side_effect=AssertionError("shared-support batch should not fall back"),
        ):
            rows = engine._reduce_and_sum_scaled_batch([q0, q1], context=engine._ReductionContext())

        self.assertEqual(len(rows), 2)
        self.assertEqual(pre_exact.call_count, 1)
        self.assertEqual(batch_sum.call_count, 1)

    def test_wide_non_level3_batch_falls_back_to_individual_reduction(self):
        q0 = engine._phase_function_from_parts(
            3,
            level=4,
            q0=0,
            q1=[1, 2, 3],
            q2={(0, 1): 1},
            q3={(0, 1, 2): 1},
        )
        q1 = engine._phase_function_from_parts(
            3,
            level=4,
            q0=1 / 16,
            q1=[4, 5, 6],
            q2={(0, 1): 1},
            q3={(0, 1, 2): 1},
        )
        context = engine._ReductionContext()
        fake_total = engine._make_scaled_complex(1.0 + 0.0j)
        fake_info = {
            "quad": 0,
            "constraint": 0,
            "branched": 0,
            "remaining": 16,
            "structural_obstruction": 5,
            "gauss_obstruction": 5,
            "cost_r": 16,
            "phase_states": 0,
            "phase_splits": 0,
            "phase3_backend": "treewidth_dp",
        }

        native_symbol = engine._native_symbol

        def fake_native_symbol(name):
            if name in {
                "sum_level3_treewidth_preplanned_batch_array",
                "build_phase_function_treewidth_support_plan",
                "sum_phase_function_treewidth_preplanned_batch_scaled_array",
            }:
                return None
            return native_symbol(name)

        with mock.patch.object(
            engine,
            "_pre_exact_phase3_treewidth_escape",
            return_value=([0], [0, 1, 2], 16, 5, "treewidth_dp"),
        ), mock.patch.object(
            engine,
            "_native_symbol",
            side_effect=fake_native_symbol,
        ), mock.patch.object(
            engine,
            "_sum_via_treewidth_dp_scaled_batch_shared_support",
            side_effect=AssertionError("wide non-level3 batch should not run"),
        ), mock.patch.object(
            engine,
            "_reduce_and_sum_scaled",
            return_value=(fake_total, fake_info),
        ) as reduce_one:
            rows = engine._reduce_and_sum_scaled_batch([q0, q1], context=context)

        self.assertEqual(len(rows), 2)
        self.assertEqual(reduce_one.call_count, 2)


if __name__ == "__main__":
    unittest.main()
