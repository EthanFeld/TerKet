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


class Q3FreeTreewidthNativeTests(unittest.TestCase):
    def test_factorized_components_route_q3_free_components_directly(self):
        q = engine._phase_function_from_parts(
            2,
            level=3,
            q0=0,
            q1=[1, 2],
            q2={},
            q3={},
        )

        with mock.patch.object(
            engine,
            "_gauss_sum_q3_free_scaled",
            return_value=((1.0 + 0.0j, 0), {"phase_states": 0, "phase_splits": 0}),
        ) as q3_free_sum, mock.patch.object(
            engine,
            "_reduce_and_sum_scaled",
            side_effect=AssertionError("should not re-enter full reducer for q3-free component"),
        ):
            total, info = engine._sum_factorized_components_scaled(q, [{0}, {1}])

        self.assertEqual(total[0], 0.5 + 0.0j)
        self.assertEqual(total[1], 2)
        self.assertTrue(info["phase3_backend"] in {"q3_free", "quadratic_tensor", "mixed"})
        self.assertEqual(q3_free_sum.call_count, 2)

    def test_q3_cover_routes_q3_free_branches_directly(self):
        q = engine._phase_function_from_parts(
            3,
            level=3,
            q0=0,
            q1=[0, 0, 0],
            q2={},
            q3={(0, 1, 2): 1},
        )

        with mock.patch.object(
            engine,
            "_fix_variables",
            side_effect=AssertionError("q3_cover should use branch template batch, not rebuild branches"),
        ), mock.patch.object(
            engine,
            "_reduce_and_sum_scaled",
            side_effect=AssertionError("should not re-enter full reducer on q3-cover leaves"),
        ):
            total, info = engine._sum_via_q3_cover(q)

        self.assertEqual(total[0], 0.75 + 1.5308084989341915e-17j)
        self.assertEqual(total[1], 6)
        self.assertEqual(info["phase3_backend"], "q3_cover")

    def test_q3_separator_routes_q3_free_branches_directly(self):
        q = engine._phase_function_from_parts(
            3,
            level=3,
            q0=0,
            q1=[0, 0, 0],
            q2={},
            q3={(0, 1, 2): 1},
        )

        with mock.patch.object(
            engine,
            "_gauss_sum_q3_free_scaled",
            return_value=((1.0 + 0.0j, 0), {"phase_states": 0, "phase_splits": 0}),
        ) as q3_free_sum, mock.patch.object(
            engine,
            "_reduce_and_sum_scaled",
            side_effect=AssertionError("should not re-enter full reducer on q3-free separator branches"),
        ):
            total, info = engine._sum_via_q3_separator(q, [0])

        self.assertEqual(total, (2.0 + 0.0j, 0))
        self.assertEqual(info["phase3_backend"], "q3_separator")
        self.assertEqual(q3_free_sum.call_count, 2)

    def test_phase3_plan_prefers_peeled_treewidth_when_core_is_empty(self):
        q = engine._phase_function_from_parts(
            5,
            level=3,
            q0=0,
            q1=[0] * 5,
            q2={(0, 1): 2},
            q3={(0, 1, 2): 1},
        )
        engine._STRUCTURE_PHASE3_PLAN_CACHE.clear()

        with mock.patch.object(engine, "_minimum_q3_vertex_cover", return_value=[0, 1, 2]), \
            mock.patch.object(engine, "_min_fill_cubic_order", return_value=([0, 1, 2, 3, 4], 20)), \
            mock.patch.object(engine, "_q3_hypergraph_2core", return_value=(set(), [0, 1, 2, 3, 4])), \
            mock.patch.object(engine, "_q3_core_cover_size", return_value=0), \
            mock.patch.object(engine, "_treewidth_order_width", return_value=20), \
            mock.patch.object(engine, "_estimate_treewidth_dp_work", return_value=1_451_928_362):
            _cover, _order, width, _obstruction, backend = engine._phase3_plan(
                q,
                allow_tensor_contraction=False,
            )

        self.assertEqual(width, 20)
        self.assertEqual(backend, "treewidth_dp_peeled")

    def test_phase3_plan_prefers_peeled_treewidth_at_width_24_when_work_is_acceptable(self):
        q = engine._phase_function_from_parts(
            5,
            level=3,
            q0=0,
            q1=[0] * 5,
            q2={(0, 1): 2},
            q3={(0, 1, 2): 1},
        )
        engine._STRUCTURE_PHASE3_PLAN_CACHE.clear()

        with mock.patch.object(engine, "_minimum_q3_vertex_cover", return_value=[0, 1, 2]), \
            mock.patch.object(engine, "_min_fill_cubic_order", return_value=([0, 1, 2, 3, 4], 24)), \
            mock.patch.object(engine, "_q3_hypergraph_2core", return_value=(set(), [0, 1, 2, 3, 4])), \
            mock.patch.object(engine, "_q3_core_cover_size", return_value=0), \
            mock.patch.object(engine, "_treewidth_order_width", return_value=24), \
            mock.patch.object(engine, "_estimate_treewidth_dp_work", return_value=28_666_669_866):
            _cover, _order, width, _obstruction, backend = engine._phase3_plan(
                q,
                allow_tensor_contraction=False,
            )

        self.assertEqual(width, 24)
        self.assertEqual(backend, "treewidth_dp_peeled")

    def test_phase3_plan_prefers_separator_when_score_beats_cover(self):
        q = engine._phase_function_from_parts(
            6,
            level=3,
            q0=0,
            q1=[0] * 6,
            q2={(0, 1): 2},
            q3={(0, 1, 2): 1, (2, 3, 4): 1},
        )
        engine._STRUCTURE_PHASE3_PLAN_CACHE.clear()

        with mock.patch.object(engine, "_minimum_q3_vertex_cover", return_value=[0, 1, 2, 3]), \
            mock.patch.object(engine, "_min_fill_cubic_order", return_value=([0, 1, 2, 3, 4, 5], 25)), \
            mock.patch.object(engine, "_q3_hypergraph_2core", return_value=({0, 1, 2, 3}, [])), \
            mock.patch.object(engine, "_q3_core_cover_size", return_value=2), \
            mock.patch.object(engine, "_prefer_treewidth_phase3", return_value=False), \
            mock.patch.object(engine, "_prefer_cubic_contraction_phase3", return_value=False), \
            mock.patch.object(engine, "_should_apply_extended_q3_reductions", return_value=True), \
            mock.patch.object(engine, "_find_small_q3_separator", return_value=(2,)), \
            mock.patch.object(engine, "_estimate_q3_cover_work", return_value=1_000_000), \
            mock.patch.object(engine, "_estimate_q3_separator_work", return_value=10_000):
            _cover, _order, _width, _obstruction, backend = engine._phase3_plan(
                q,
                allow_tensor_contraction=False,
            )

        self.assertEqual(backend, "q3_separator")

    def test_phase3_plan_prefers_cover_when_separator_score_is_worse(self):
        q = engine._phase_function_from_parts(
            6,
            level=3,
            q0=0,
            q1=[0] * 6,
            q2={(0, 1): 2},
            q3={(0, 1, 2): 1, (2, 3, 4): 1},
        )
        engine._STRUCTURE_PHASE3_PLAN_CACHE.clear()

        with mock.patch.object(engine, "_minimum_q3_vertex_cover", return_value=[0, 1, 2]), \
            mock.patch.object(engine, "_min_fill_cubic_order", return_value=([0, 1, 2, 3, 4, 5], 25)), \
            mock.patch.object(engine, "_q3_hypergraph_2core", return_value=({0, 1, 2}, [])), \
            mock.patch.object(engine, "_q3_core_cover_size", return_value=2), \
            mock.patch.object(engine, "_prefer_treewidth_phase3", return_value=False), \
            mock.patch.object(engine, "_prefer_cubic_contraction_phase3", return_value=False), \
            mock.patch.object(engine, "_should_apply_extended_q3_reductions", return_value=True), \
            mock.patch.object(engine, "_find_small_q3_separator", return_value=(2,)), \
            mock.patch.object(engine, "_estimate_q3_cover_work", return_value=10_000), \
            mock.patch.object(engine, "_estimate_q3_separator_work", return_value=1_000_000):
            _cover, _order, _width, _obstruction, backend = engine._phase3_plan(
                q,
                allow_tensor_contraction=False,
            )

        self.assertEqual(backend, "q3_cover")

    def test_local_refinement_can_reduce_work_without_hurting_width(self):
        q = engine._phase_function_from_parts(
            8,
            level=3,
            q0=0,
            q1=[0] * 8,
            q2={
                (2, 3): 2,
                (3, 6): 2,
                (5, 6): 2,
                (5, 7): 2,
            },
            q3={},
        )

        base_order = [0, 1, 4, 2, 3, 6, 5, 7]
        base_width = engine._treewidth_order_width(q, base_order)
        refined_order, refined_width = engine._refine_q3_free_treewidth_order_locally(
            q,
            base_order,
            base_width,
        )

        base_score = (
            int(base_width),
            int(engine._estimate_treewidth_dp_work(q, base_order)),
        )
        refined_score = (
            int(refined_width),
            int(engine._estimate_treewidth_dp_work(q, refined_order)),
        )

        self.assertLess(refined_score, base_score)

    @unittest.skipIf(engine._schur_native is None, "native accelerator unavailable")
    def test_native_preplanned_treewidth_batch_matches_exact_rows(self):
        q = engine._phase_function_from_parts(
            6,
            level=3,
            q0=0,
            q1=[0] * 6,
            q2={
                (0, 1): 2,
                (1, 2): 2,
                (2, 3): 2,
                (3, 4): 2,
                (1, 4): 2,
                (4, 5): 2,
            },
            q3={},
        )
        order, _width = engine._min_fill_cubic_order(q)
        native_plan = engine._build_native_q3_free_treewidth_plan(
            n_vars=q.n,
            level=q.level,
            q2=q.q2,
            order=order,
        )
        self.assertIsNotNone(native_plan)

        q1_batch = np.asarray(
            [
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
            ],
            dtype=np.int64,
        )

        native_totals = engine._sum_q3_free_treewidth_dp_scaled_batch(
            n_vars=q.n,
            level=q.level,
            q1_batch=q1_batch,
            q2=q.q2,
            order=order,
            native_plan=native_plan,
        )
        exact_totals = []
        for row in q1_batch:
            row_q = engine._phase_function_from_parts(
                q.n,
                level=q.level,
                q0=0,
                q1=row.tolist(),
                q2=q.q2,
                q3={},
            )
            exact_totals.append(engine._sum_via_treewidth_dp_scaled(row_q, order)[0])

        native_complex = [engine._scaled_to_complex(total) for total in native_totals]
        exact_complex = [engine._scaled_to_complex(total) for total in exact_totals]
        self.assertEqual(native_complex, exact_complex)

    @unittest.skipIf(engine._schur_native is None, "native accelerator unavailable")
    def test_native_preplanned_treewidth_array_entry_matches_list_entry(self):
        q2 = {
            (0, 1): 2,
            (1, 2): 2,
            (2, 3): 2,
            (0, 3): 2,
        }
        order = [0, 1, 2, 3]
        native_plan = engine._build_native_q3_free_treewidth_plan(
            n_vars=4,
            level=3,
            q2=q2,
            order=order,
        )
        self.assertIsNotNone(native_plan)

        q1_batch = np.asarray(
            [
                [0, 2, 4, 6],
                [6, 4, 2, 0],
            ],
            dtype=np.int64,
        )

        list_rows = engine._schur_native.sum_q3_free_treewidth_preplanned_batch_scaled(
            native_plan,
            q1_batch.tolist(),
        )
        array_rows = engine._schur_native.sum_q3_free_treewidth_preplanned_batch_scaled_array(
            native_plan,
            q1_batch,
        )

        self.assertEqual(list_rows, array_rows)

    @unittest.skipIf(engine._schur_native is None, "native accelerator unavailable")
    def test_native_fixed_factor_treewidth_plan_matches_generic_factor_sum(self):
        q = engine._phase_function_from_parts(
            5,
            level=3,
            q0=0,
            q1=[1, 0, 2, 0, 0],
            q2={(0, 1): 2, (1, 2): 2, (2, 3): 2},
            q3={(1, 3, 4): 1},
        )
        order, _width = engine._min_fill_cubic_order(q)
        scalar, factors = engine._build_cached_phase3_treewidth_factor_plan_scaled(q)
        native_plan = engine._build_native_phase3_treewidth_plan(q=q, order=order)
        self.assertIsNotNone(native_plan)
        assert native_plan is not None

        native_total, native_width = engine._schur_native.sum_scaled_factor_treewidth_preplanned(native_plan)
        expected_total, expected_width = engine._sum_factor_tables_scaled(
            q.n,
            dict(factors),
            order,
            scalar=scalar,
        )

        self.assertEqual(engine._scaled_to_complex((complex(native_total[0]), int(native_total[1]))), engine._scaled_to_complex(expected_total))
        self.assertEqual(int(native_width), int(expected_width))

    @unittest.skipIf(engine._schur_native is None, "native accelerator unavailable")
    def test_native_level3_treewidth_plan_matches_direct_kernel(self):
        q = engine._phase_function_from_parts(
            6,
            level=3,
            q0=0,
            q1=[1, 0, 2, 0, 3, 0],
            q2={(0, 1): 2, (1, 2): 2, (2, 3): 2, (3, 4): 2},
            q3={(1, 3, 5): 1, (0, 2, 4): 1},
        )
        order, _width = engine._min_fill_cubic_order(q)
        native_plan = engine._build_native_level3_phase3_treewidth_plan(q=q, order=order)
        self.assertIsNotNone(native_plan)
        assert native_plan is not None

        planned_total, planned_width = engine._schur_native.sum_level3_treewidth_preplanned(native_plan)
        direct_total, direct_width = engine._schur_native.sum_treewidth_dp_level3(
            q.n,
            q.q1,
            q.q2,
            q.q3,
            order,
        )
        self.assertEqual(complex(planned_total), complex(direct_total))
        self.assertEqual(int(planned_width), int(direct_width))

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


if __name__ == "__main__":
    unittest.main()
