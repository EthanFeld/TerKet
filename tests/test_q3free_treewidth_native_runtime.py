"""Tests for native q3-free treewidth runtime execution."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket import engine

class Q3FreeTreewidthNativeRuntimeTests(unittest.TestCase):

    def test_generic_mediator_plan_precomputes_assignment_shifts(self):
        q = engine._phase_function_from_parts(
            4,
            level=3,
            q0=0,
            q1=[1, 0, 0, 0],
            q2={
                (0, 1): 1,
                (0, 2): 1,
                (1, 3): 1,
            },
            q3={},
        )
        plan = engine._build_generic_q2_mediator_plan(q)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertTrue(plan.mediators)
        self.assertTrue(plan.mediators[0].assignment_residue_shifts)

    def test_cluster_plan_builds_native_artifacts_up_front(self):
        q = engine._phase_function_from_parts(
            4,
            level=3,
            q0=0,
            q1=[1, 1, 0, 0],
            q2={
                (0, 1): 2,
                (0, 2): 2,
                (1, 3): 2,
            },
            q3={},
        )
        plan = engine._build_half_phase_cluster_plan(q)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertTrue(plan.clusters)
        spec = plan.clusters[0]
        self.assertTrue(spec.cluster_order)
        self.assertIsNotNone(spec.native_treewidth_plan)
        self.assertIsNotNone(spec.boundary_shift_table)

    def test_cluster_plan_single_row_phase_folding_deduplicates_boundary_shifts(self):
        q = engine._phase_function_from_parts(
            3,
            level=3,
            q0=0,
            q1=[1, 0, 0],
            q2={
                (0, 1): 2,
                (0, 2): 2,
            },
            q3={},
        )
        plan = engine._build_half_phase_cluster_plan(q)
        self.assertIsNotNone(plan)
        assert plan is not None

        captured_batches: list[np.ndarray] = []

        def fake_treewidth_batch(*, n_vars, level, q1_batch, q2, order, native_plan=None):
            del n_vars, level, q2, order, native_plan
            captured_batches.append(np.asarray(q1_batch, dtype=np.int64).copy())
            return [((idx + 1) + 0.0j, 0) for idx in range(len(q1_batch))]

        with mock.patch.object(
            engine,
            "_sum_q3_free_treewidth_dp_scaled_batch",
            side_effect=fake_treewidth_batch,
        ):
            total = engine._evaluate_half_phase_cluster_plan_scaled(plan, [1, 0, 0])

        self.assertEqual(len(captured_batches), 1)
        np.testing.assert_array_equal(
            captured_batches[0],
            np.asarray([[1], [5]], dtype=np.int64),
        )
        self.assertIsInstance(total, tuple)

    def test_cluster_plan_batch_phase_folding_deduplicates_boundary_shifts(self):
        q = engine._phase_function_from_parts(
            3,
            level=3,
            q0=0,
            q1=[1, 0, 0],
            q2={
                (0, 1): 2,
                (0, 2): 2,
            },
            q3={},
        )
        plan = engine._build_half_phase_cluster_plan(q)
        self.assertIsNotNone(plan)
        assert plan is not None

        captured_batches: list[np.ndarray] = []

        def fake_treewidth_batch(*, n_vars, level, q1_batch, q2, order, native_plan=None):
            del n_vars, level, q2, order, native_plan
            captured_batches.append(np.asarray(q1_batch, dtype=np.int64).copy())
            return [((idx + 1) + 0.0j, 0) for idx in range(len(q1_batch))]

        with mock.patch.object(
            engine,
            "_sum_q3_free_treewidth_dp_scaled_batch",
            side_effect=fake_treewidth_batch,
        ):
            totals = engine._evaluate_half_phase_cluster_plan_scaled_batch(
                plan,
                np.asarray([[1, 0, 0]], dtype=np.int64),
            )

        self.assertEqual(len(captured_batches), 1)
        np.testing.assert_array_equal(
            captured_batches[0],
            np.asarray([[1], [5]], dtype=np.int64),
        )
        self.assertEqual(len(totals), 1)

    def test_cached_peeled_treewidth_factor_plan_is_reused(self):
        q = engine._phase_function_from_parts(
            4,
            level=3,
            q0=0,
            q1=[1, 0, 2, 0],
            q2={(0, 1): 1, (2, 3): 2},
            q3={(0, 2, 3): 1},
        )
        engine._STRUCTURE_PHASE3_TREEWIDTH_FACTOR_CACHE.clear()

        scalar0, factors0 = engine._build_cached_phase3_treewidth_factor_plan_scaled(q)
        scalar1, factors1 = engine._build_cached_phase3_treewidth_factor_plan_scaled(q)

        self.assertEqual(scalar0, scalar1)
        self.assertIs(factors0, factors1)

    def test_treewidth_dp_scaled_prefers_preplanned_level3_native_path(self):
        q = engine._phase_function_from_parts(
            3,
            level=3,
            q0=0,
            q1=[1, 2, 3],
            q2={(0, 1): 2},
            q3={(0, 1, 2): 1},
        )
        order = [0, 1, 2]
        expected = engine._make_scaled_complex(1.25 + 0.5j)

        class FailNative:
            @staticmethod
            def sum_treewidth_dp_level3(*args, **kwargs):
                raise AssertionError("direct native kernel should not run when preplanned path hits")

        with mock.patch.object(engine, "_native_level3_enabled", return_value=True), \
            mock.patch.object(
                engine,
                "_sum_native_level3_phase3_treewidth_preplanned",
                return_value=(1.25 + 0.5j, 7),
            ) as planned_sum, \
            mock.patch.object(engine, "_schur_native", FailNative()):
            total, width = engine._sum_via_treewidth_dp_scaled(q, order)

        self.assertEqual(total, expected)
        self.assertEqual(width, 7)
        planned_sum.assert_called_once_with(q=q, order=order)

    def test_q3_free_planner_prefers_native_treewidth_over_cluster(self):
        q = engine._phase_function_from_parts(
            10,
            level=3,
            q0=0,
            q1=[1] * 10,
            q2={(idx, (idx + 1) % 10): 1 for idx in range(10)},
            q3={},
        )
        cluster_plan = engine._HalfPhaseClusterPlan(
            level=3,
            core_vars=(0, 1),
            core_q2={},
            order=(0, 1),
            width=8,
            clusters=(),
        )
        native_plan = object()

        with (
            mock.patch.object(engine, "_Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS", 4),
            mock.patch.object(engine, "_q3_free_treewidth_order", return_value=list(range(10))),
            mock.patch.object(engine, "_finalize_q3_free_treewidth_order", return_value=(list(range(10)), 20)),
            mock.patch.object(engine, "_build_native_q3_free_treewidth_plan", return_value=native_plan),
            mock.patch.object(engine, "_build_q1_cluster_plan", return_value=cluster_plan),
            mock.patch.object(
                engine,
                "_q3_free_one_shot_cutset_conditioning_plan",
                side_effect=AssertionError("native treewidth should skip one-shot cutset planning"),
            ),
        ):
            isolated_vars, component_plans = engine._plan_q3_free_constraint_components(
                q,
                0,
                prefer_one_shot_slicing=True,
            )

        self.assertEqual(isolated_vars, ())
        self.assertEqual(len(component_plans), 1)
        self.assertEqual(component_plans[0].backend, "treewidth")
        self.assertIs(component_plans[0].native_treewidth_plan, native_plan)
        self.assertIsNone(component_plans[0].cluster_plan)

    def test_bad_q2_cover_prefers_high_degree_equal_size_cover(self):
        q = engine._phase_function_from_parts(
            6,
            level=3,
            q0=0,
            q1=[0] * 6,
            q2={
                (0, 1): 1,
                (1, 2): 1,
                (2, 3): 1,
                (3, 0): 1,
                (1, 4): 2,
                (1, 5): 2,
                (3, 4): 2,
                (3, 5): 2,
            },
            q3={},
        )

        cover = engine._minimum_bad_q2_vertex_cover_uncached(q)

        self.assertEqual(cover, [1, 3])

    def test_q3_cover_template_uses_compact_storage_dtypes(self):
        q = engine._phase_function_from_parts(
            4,
            level=3,
            q0=0,
            q1=[1, 2, 3, 4],
            q2={(0, 2): 1, (1, 3): 1},
            q3={(0, 1, 2): 1},
        )

        template = engine._build_q3_free_branch_template(q, [0, 1])

        self.assertEqual(template.base_q1_residue.dtype, np.uint8)
        self.assertEqual(template.base_q2_residue.dtype, np.uint8)
        self.assertEqual(template.cover_remaining_q2_residue.dtype, np.uint8)
        self.assertEqual(template.pair_left.dtype, np.uint8)
        self.assertEqual(template.cubic_pair_index.dtype, np.uint8)

    def test_q3_free_execution_plan_uses_compact_sequence_storage(self):
        q = engine._phase_function_from_parts(
            3,
            level=3,
            q0=0,
            q1=[1, 2, 3],
            q2={(0, 1): 2},
            q3={},
        )

        plan = engine._build_q3_free_execution_plan(
            q=q,
            allow_tensor_contraction=True,
        )

        self.assertIsInstance(plan.q1, np.ndarray)
        self.assertEqual(plan.q1.dtype, np.uint8)
        self.assertIsInstance(plan.isolated_vars, np.ndarray)
        self.assertEqual(list(plan.isolated_vars), [])
        self.assertIsInstance(plan.components[0].variables, np.ndarray)
        self.assertEqual(plan.components[0].variables.dtype, np.uint8)

    def test_q3_free_constraint_plan_scaled_handles_array_backed_isolated_vars(self):
        state = engine.build_state(2, [], [0, 0])
        cache = state._prepare_constraint_echelon()
        plan = engine._build_q3_free_constraint_plan(state, cache)

        total = engine._evaluate_q3_free_constraint_plan_scaled(
            plan,
            [0, 0],
            allow_tensor_contraction=True,
        )
        batch_total = engine._evaluate_q3_free_constraint_plan_scaled_batch(
            plan,
            [[0, 0]],
        )[0]

        self.assertIsInstance(plan.isolated_vars, np.ndarray)
        self.assertEqual(total, batch_total)
