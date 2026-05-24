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

class Q3FreeTreewidthNativeRoutingTests(unittest.TestCase):

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
        engine._STRUCTURE_PHASE3_REFINED_ORDER_CACHE.clear()

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
        engine._STRUCTURE_PHASE3_REFINED_ORDER_CACHE.clear()

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
        engine._STRUCTURE_PHASE3_REFINED_ORDER_CACHE.clear()

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
        engine._STRUCTURE_PHASE3_REFINED_ORDER_CACHE.clear()

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

    def test_phase3_plan_refines_treewidth_order_before_backend_choice(self):
        q = engine._phase_function_from_parts(
            5,
            level=3,
            q0=0,
            q1=[0] * 5,
            q2={(0, 1): 2},
            q3={(0, 1, 2): 1},
        )
        engine._STRUCTURE_PHASE3_PLAN_CACHE.clear()
        engine._STRUCTURE_PHASE3_REFINED_ORDER_CACHE.clear()
        captured: dict[str, object] = {}

        def fake_choose_phase3_backend(
            _q,
            cover,
            order,
            width,
            structural_obstruction,
            **kwargs,
        ):
            del _q, cover, kwargs
            captured["order"] = list(order)
            captured["width"] = int(width)
            captured["obstruction"] = int(structural_obstruction)
            return "treewidth_dp", (0,), None

        with mock.patch.object(engine, "_minimum_q3_vertex_cover", return_value=[0, 1, 2]), \
            mock.patch.object(engine, "_min_fill_cubic_order", return_value=([0, 1, 2, 3, 4], 26)), \
            mock.patch.object(engine, "_q3_hypergraph_2core", return_value=({0, 1, 2}, [])), \
            mock.patch.object(engine, "_q3_core_cover_size", return_value=4), \
            mock.patch.object(
                engine,
                "_finalize_phase3_treewidth_order",
                return_value=([2, 0, 1, 3, 4], 18),
            ), \
            mock.patch.object(
                engine,
                "_choose_phase3_backend",
                side_effect=fake_choose_phase3_backend,
            ):
            _cover, order, width, obstruction, backend = engine._phase3_plan(
                q,
                allow_tensor_contraction=False,
            )

        self.assertEqual(order, [2, 0, 1, 3, 4])
        self.assertEqual(width, 18)
        self.assertEqual(obstruction, 4)
        self.assertEqual(backend, "treewidth_dp")
        self.assertEqual(captured["order"], [2, 0, 1, 3, 4])
        self.assertEqual(captured["width"], 18)
        self.assertEqual(captured["obstruction"], 4)

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
