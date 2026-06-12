"""Tests for q3-free one-shot slicing heuristic selection behavior."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket import compute_circuit_amplitude, make_circuit
from terket import engine
from terket.cubic_arithmetic import PhaseFunction

class Q3FreeOneShotSlicingHeuristicTests(unittest.TestCase):

    def test_one_shot_cutset_planner_uses_contracted_core_candidate_pool(self):
        q = PhaseFunction(
            6,
            level=3,
            q1=[1] * 6,
            q2={
                (0, 1): 1,
                (0, 2): 1,
                (0, 3): 1,
                (1, 2): 1,
                (1, 3): 1,
                (2, 3): 1,
                (3, 4): 1,
                (4, 5): 1,
            },
            q3={},
        )
        core_plan = engine._Q3FreeCutsetConditioningPlan(
            level=3,
            cutset_vars=(1,),
            remaining_vars=(0, 2, 3),
            remaining_backend="treewidth",
            remaining_q2={},
            remaining_order=(0, 1, 2),
            cutset_remaining_q2_residue=np.zeros((1, 3), dtype=np.int64),
            cutset_cutset_left=np.zeros(0, dtype=np.int64),
            cutset_cutset_right=np.zeros(0, dtype=np.int64),
            cutset_cutset_residue=np.zeros(0, dtype=np.int64),
            remaining_width=2,
            estimated_total_work=4,
        )
        pool_plan = engine._Q3FreeCutsetConditioningPlan(
            level=3,
            cutset_vars=(1,),
            remaining_vars=(0, 2, 3, 4, 5),
            remaining_backend="treewidth",
            remaining_q2={},
            remaining_order=(0, 1, 2, 3, 4),
            cutset_remaining_q2_residue=np.zeros((1, 5), dtype=np.int64),
            cutset_cutset_left=np.zeros(0, dtype=np.int64),
            cutset_cutset_right=np.zeros(0, dtype=np.int64),
            cutset_cutset_residue=np.zeros(0, dtype=np.int64),
            remaining_width=3,
            estimated_total_work=8,
        )
        pool_calls: list[tuple[int, ...]] = []

        def fake_uncached(base_q, **kwargs):
            candidate_override = kwargs.get("candidate_override")
            if base_q.n == 4 and candidate_override is None:
                return core_plan
            if base_q.n == 6 and candidate_override is not None:
                pool_calls.append(tuple(candidate_override))
                if len(candidate_override) > 1:
                    return pool_plan
            return None

        with (
            patch.object(engine, "_q3_free_cutset_conditioning_plan", return_value=None),
            patch.object(engine, "_separator_ranked_q3_free_cutset_vertices", return_value=(0,)),
            patch.object(engine, "_candidate_q3_free_cutset_vertices", return_value=(1,)),
            patch.object(engine, "_order_guided_q3_free_cutset_vertices", return_value=(2,)),
            patch.object(engine, "_Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS", 4),
            patch.object(engine, "_Q3_FREE_SERIES_CORE_RECURSE_MIN_SHRINK", 1),
            patch.object(engine, "_build_q3_free_cutset_conditioning_plan_uncached", side_effect=fake_uncached),
        ):
            plan = engine._q3_free_one_shot_cutset_conditioning_plan(q)

        self.assertIs(plan, pool_plan)
        self.assertTrue(any(set(call) >= {0, 1, 2} for call in pool_calls))

    def test_one_shot_giant_path_skips_expensive_core_seed_search_when_core_pool_rich(self):
        q = PhaseFunction(
            10,
            level=3,
            q1=[1] * 10,
            q2={(idx, (idx + 1) % 10): 1 for idx in range(10)},
            q3={},
        )
        final_plan = engine._Q3FreeCutsetConditioningPlan(
            level=3,
            cutset_vars=(2,),
            remaining_vars=tuple(var for var in range(10) if var != 2),
            remaining_backend="treewidth",
            remaining_q2={},
            remaining_order=tuple(range(9)),
            cutset_remaining_q2_residue=np.zeros((1, 9), dtype=np.int64),
            cutset_cutset_left=np.zeros(0, dtype=np.int64),
            cutset_cutset_right=np.zeros(0, dtype=np.int64),
            cutset_cutset_residue=np.zeros(0, dtype=np.int64),
            remaining_width=3,
            estimated_total_work=8,
        )
        uncached_calls: list[int] = []

        def fake_uncached(base_q, **kwargs):
            del kwargs
            uncached_calls.append(base_q.n)
            return final_plan if base_q.n == 10 else None

        with (
            patch.object(engine, "_q3_free_cutset_conditioning_plan", return_value=None),
            patch.object(engine, "_Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS", 4),
            patch.object(engine, "_Q3_FREE_SERIES_CORE_RECURSE_MIN_SHRINK", 1),
            patch.object(engine, "_direct_order_guided_q3_free_cutset_plan", return_value=None),
            patch.object(engine, "_separator_ranked_q3_free_cutset_vertices", side_effect=[(), (0, 1), ()]),
            patch.object(engine, "_candidate_q3_free_cutset_vertices", side_effect=[(), (2, 3), ()]),
            patch.object(engine, "_order_guided_q3_free_cutset_vertices", side_effect=[(), (1, 2), (), ()]),
            patch.object(engine, "_build_q3_free_cutset_conditioning_plan_uncached", side_effect=fake_uncached),
        ):
            plan = engine._q3_free_one_shot_cutset_conditioning_plan(q)

        self.assertIs(plan, final_plan)
        self.assertEqual(uncached_calls, [10])

    def test_one_shot_cutset_planner_dedupes_identical_order_guided_variants(self):
        q = PhaseFunction(
            6,
            level=3,
            q1=[1] * 6,
            q2={
                (0, 1): 1,
                (1, 2): 1,
                (2, 3): 1,
                (3, 4): 1,
                (4, 5): 1,
            },
            q3={},
        )
        variant_calls: list[tuple[tuple[int, ...] | None, tuple[int, ...] | None]] = []
        variant_plan = engine._Q3FreeCutsetConditioningPlan(
            level=3,
            cutset_vars=(2,),
            remaining_vars=(0, 1, 3, 4, 5),
            remaining_backend="treewidth",
            remaining_q2={},
            remaining_order=(0, 1, 2, 3, 4),
            cutset_remaining_q2_residue=np.zeros((1, 5), dtype=np.int64),
            cutset_cutset_left=np.zeros(0, dtype=np.int64),
            cutset_cutset_right=np.zeros(0, dtype=np.int64),
            cutset_cutset_residue=np.zeros(0, dtype=np.int64),
            remaining_width=3,
            estimated_total_work=8,
        )

        def fake_uncached(base_q, **kwargs):
            del base_q
            key = (
                None if kwargs.get("candidate_override") is None else tuple(kwargs["candidate_override"]),
                None if kwargs.get("remaining_order_hint") is None else tuple(kwargs["remaining_order_hint"]),
            )
            variant_calls.append(key)
            return variant_plan

        with (
            patch.object(engine, "_q3_free_cutset_conditioning_plan", return_value=None),
            patch.object(engine, "_separator_ranked_q3_free_cutset_vertices", return_value=()),
            patch.object(engine, "_candidate_q3_free_cutset_vertices", return_value=()),
            patch.object(engine, "_order_guided_q3_free_cutset_vertices", return_value=(2, 3, 4)),
            patch.object(engine, "_build_q3_free_cutset_conditioning_plan_uncached", side_effect=fake_uncached),
        ):
            plan = engine._q3_free_one_shot_cutset_conditioning_plan(q)

        self.assertIs(plan, variant_plan)
        self.assertEqual(len(variant_calls), 1)
        self.assertEqual(variant_calls[0][0], (2, 3, 4))

    def test_native_shortlist_guides_greedy_cutset_search(self):
        q = PhaseFunction(
            10,
            level=3,
            q1=[1] * 10,
            q2={(idx, (idx + 1) % 10): 1 for idx in range(10)},
            q3={},
        )
        seen_cutsets: list[tuple[int, ...]] = []

        def fake_eval(base_q, cutset_vars, **kwargs):
            del base_q, kwargs
            seen_cutsets.append(tuple(cutset_vars))
            remaining_vars = tuple(var for var in range(q.n) if var not in set(cutset_vars))
            plan = engine._Q3FreeCutsetConditioningPlan(
                level=3,
                cutset_vars=tuple(cutset_vars),
                remaining_vars=remaining_vars,
                remaining_backend="treewidth",
                remaining_q2={},
                remaining_order=tuple(range(len(remaining_vars))),
                cutset_remaining_q2_residue=np.zeros((len(cutset_vars), len(remaining_vars)), dtype=np.int64),
                cutset_cutset_left=np.zeros(0, dtype=np.int64),
                cutset_cutset_right=np.zeros(0, dtype=np.int64),
                cutset_cutset_residue=np.zeros(0, dtype=np.int64),
                native_treewidth_plan=object(),
                remaining_width=1,
                estimated_total_work=1,
            )
            return engine._Q3FreeCutsetCandidateEvaluation(
                cutset_vars=tuple(cutset_vars),
                plan=plan,
                viable=True,
                score=(0, len(cutset_vars), cutset_vars),
            )

        with (
            patch.object(engine, "_Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS", 4),
            patch.object(engine, "_native_rank_q3_free_cutset_extensions", return_value=((7, 1, 1), (5, 2, 2))),
            patch.object(engine, "_evaluate_q3_free_cutset_candidate", side_effect=fake_eval),
            patch.object(engine, "_finalize_q3_free_treewidth_order", side_effect=lambda q_obj, order: (list(order), engine._treewidth_order_width(q_obj, order))),
        ):
            plan = engine._build_q3_free_cutset_conditioning_plan_uncached(
                q,
                max_size=1,
                candidate_pool=6,
                beam_width=2,
                branches_per_state=2,
                prioritize_width=True,
                target_remaining_width=1,
                allow_generic_remaining=True,
                prefer_one_shot_slicing=True,
                candidate_override=(0, 1, 5, 7, 8, 9),
            )

        self.assertIsNotNone(plan)
        self.assertEqual(seen_cutsets, [(7,), (5,)])

    def test_giant_surrogate_greedy_treewidth_result_skips_beam(self):
        q = PhaseFunction(
            10,
            level=3,
            q1=[1] * 10,
            q2={(idx, idx + 1): 1 for idx in range(9)},
            q3={},
        )
        seen_cutsets: list[tuple[int, ...]] = []

        def fake_eval(base_q, cutset_vars, **kwargs):
            del base_q, kwargs
            seen_cutsets.append(tuple(cutset_vars))
            remaining_vars = tuple(var for var in range(q.n) if var not in set(cutset_vars))
            plan = engine._Q3FreeCutsetConditioningPlan(
                level=3,
                cutset_vars=tuple(cutset_vars),
                remaining_vars=remaining_vars,
                remaining_backend="treewidth",
                remaining_q2={},
                remaining_order=tuple(range(len(remaining_vars))),
                cutset_remaining_q2_residue=np.zeros((len(cutset_vars), len(remaining_vars)), dtype=np.int64),
                cutset_cutset_left=np.zeros(0, dtype=np.int64),
                cutset_cutset_right=np.zeros(0, dtype=np.int64),
                cutset_cutset_residue=np.zeros(0, dtype=np.int64),
                native_treewidth_plan=object(),
                remaining_width=5,
                estimated_total_work=1,
            )
            return engine._Q3FreeCutsetCandidateEvaluation(
                cutset_vars=tuple(cutset_vars),
                plan=plan,
                viable=True,
                score=(0, len(cutset_vars), cutset_vars),
            )

        with (
            patch.object(engine, "_Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS", 4),
            patch.object(engine, "_native_rank_q3_free_cutset_extensions", return_value=((7, 1, 1), (5, 2, 2))),
            patch.object(engine, "_evaluate_q3_free_cutset_candidate", side_effect=fake_eval),
            patch.object(
                engine,
                "_finalize_q3_free_treewidth_order",
                side_effect=AssertionError("giant greedy treewidth result should skip beam/refinement path"),
            ),
        ):
            plan = engine._build_q3_free_cutset_conditioning_plan_uncached(
                q,
                max_size=1,
                candidate_pool=6,
                beam_width=2,
                branches_per_state=2,
                prioritize_width=True,
                target_remaining_width=1,
                allow_generic_remaining=True,
                prefer_one_shot_slicing=True,
                candidate_override=(0, 1, 5, 7, 8, 9),
            )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.remaining_backend, "treewidth")
        self.assertEqual(seen_cutsets, [(7,), (5,)])

    def test_giant_surrogate_beam_collapse_keeps_top_frontier_only_without_width_gain(self):
        q = PhaseFunction(
            10,
            level=3,
            q1=[1] * 10,
            q2={(idx, idx + 1): 1 for idx in range(9)},
            q3={},
        )
        seen_cutsets: list[tuple[int, ...]] = []

        def fake_eval(base_q, cutset_vars, **kwargs):
            del base_q, kwargs
            seen_cutsets.append(tuple(cutset_vars))
            remaining_vars = tuple(var for var in range(q.n) if var not in set(cutset_vars))
            plan = engine._Q3FreeCutsetConditioningPlan(
                level=3,
                cutset_vars=tuple(cutset_vars),
                remaining_vars=remaining_vars,
                remaining_backend="generic",
                remaining_q2={},
                remaining_order=(),
                cutset_remaining_q2_residue=np.zeros((len(cutset_vars), len(remaining_vars)), dtype=np.int64),
                cutset_cutset_left=np.zeros(0, dtype=np.int64),
                cutset_cutset_right=np.zeros(0, dtype=np.int64),
                cutset_cutset_residue=np.zeros(0, dtype=np.int64),
                remaining_width=5,
                estimated_total_work=1,
            )
            return engine._Q3FreeCutsetCandidateEvaluation(
                cutset_vars=tuple(cutset_vars),
                plan=plan,
                viable=True,
                score=(0, len(cutset_vars), cutset_vars),
            )

        with (
            patch.object(engine, "_Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS", 4),
            patch.object(engine, "_native_rank_q3_free_cutset_extensions", return_value=None),
            patch.object(engine, "_evaluate_q3_free_cutset_candidate", side_effect=fake_eval),
            patch.object(engine, "_finalize_q3_free_cutset_conditioning_plan", side_effect=lambda plan, **kwargs: plan),
        ):
            plan = engine._build_q3_free_cutset_conditioning_plan_uncached(
                q,
                max_size=2,
                candidate_pool=3,
                beam_width=2,
                branches_per_state=2,
                prioritize_width=True,
                target_remaining_width=1,
                allow_generic_remaining=True,
                prefer_one_shot_slicing=True,
                candidate_override=(0, 1, 2),
            )

        self.assertIsNotNone(plan)
        self.assertIn((0, 1), seen_cutsets)
        self.assertIn((0, 2), seen_cutsets)
        self.assertNotIn((1, 2), seen_cutsets)

    def test_giant_surrogate_bypasses_one_shot_when_cheap_width_already_viable(self):
        q = PhaseFunction(
            10,
            level=3,
            q1=[1] * 10,
            q2={(idx, idx + 1): 1 for idx in range(9)},
            q3={},
        )

        with (
            patch.object(engine, "_Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS", 4),
            patch.object(engine, "_best_cheap_q3_free_order", return_value=(tuple(range(10)), 3)),
            patch.object(engine, "_q3_free_treewidth_candidate_is_viable", return_value=True),
            patch.object(
                engine,
                "_separator_ranked_q3_free_cutset_vertices",
                side_effect=AssertionError("viable cheap order should bypass one-shot candidate generation"),
            ),
        ):
            plan = engine._q3_free_one_shot_cutset_conditioning_plan(q)

        self.assertIsNone(plan)

    def test_best_cheap_q3_free_order_can_use_separator_hint(self):
        q = PhaseFunction(
            8,
            level=3,
            q1=[0] * 8,
            q2={(idx, idx + 1): 1 for idx in range(7)},
            q3={},
        )
        separator_order = [7, 5, 3, 1, 0, 2, 4, 6]

        def fake_width(_q, order):
            order = tuple(int(var) for var in order)
            if order == tuple(separator_order):
                return 4
            if order == tuple(range(8)):
                return 8
            if order == tuple(range(7, -1, -1)):
                return 7
            raise AssertionError(f"unexpected order {order}")

        with (
            patch.object(engine, "_pair_graph_separator_order", return_value=(separator_order, 4)),
            patch.object(engine, "_cubic_order_width", side_effect=fake_width),
        ):
            order, width = engine._best_cheap_q3_free_order(q)

        self.assertEqual(order, tuple(separator_order))
        self.assertEqual(width, 4)

    def test_giant_surrogate_uses_direct_order_guided_cutset_before_broad_search(self):
        q = PhaseFunction(
            10,
            level=3,
            q1=[1] * 10,
            q2={(idx, idx + 1): 1 for idx in range(9)},
            q3={},
        )
        direct_plan = engine._Q3FreeCutsetConditioningPlan(
            level=3,
            cutset_vars=(1, 2),
            remaining_vars=tuple(idx for idx in range(10) if idx not in (1, 2)),
            remaining_backend="treewidth",
            remaining_q2={},
            remaining_order=tuple(range(8)),
            cutset_remaining_q2_residue=np.zeros((0, 0), dtype=np.int64),
            cutset_cutset_left=np.zeros(0, dtype=np.int64),
            cutset_cutset_right=np.zeros(0, dtype=np.int64),
            cutset_cutset_residue=np.zeros(0, dtype=np.int64),
            remaining_width=5,
            estimated_total_work=64,
        )

        with (
            patch.object(engine, "_Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS", 4),
            patch.object(engine, "_best_cheap_q3_free_order", return_value=(tuple(range(10)), 8)),
            patch.object(engine, "_direct_order_guided_q3_free_cutset_plan", return_value=direct_plan),
            patch.object(
                engine,
                "_q3_free_series_reduction_core",
                side_effect=AssertionError("direct giant-surrogate cutset should bypass broad one-shot search"),
            ),
        ):
            plan = engine._q3_free_one_shot_cutset_conditioning_plan(q)

        self.assertIs(plan, direct_plan)

    def test_gauss_reduction_prefers_cluster_plan_before_one_shot_cutset(self):
        q = PhaseFunction(
            6,
            level=3,
            q1=[1] * 6,
            q2={(idx, idx + 1): 1 for idx in range(5)},
            q3={},
        )
        cluster_plan = engine._HalfPhaseClusterPlan(
            level=3,
            core_vars=(0, 1),
            core_q2={},
            order=(0, 1),
            width=2,
            clusters=(),
        )

        with (
            patch.object(engine, "_qubit_quadratic_tensor_obstruction", return_value=1),
            patch.object(engine, "_sum_half_phase_q2_unary_expansion_scaled", return_value=None),
            patch.object(engine, "_build_half_phase_mediator_plan", return_value=None),
            patch.object(engine, "_build_q1_cluster_plan", return_value=cluster_plan),
            patch.object(engine, "_evaluate_half_phase_cluster_plan_scaled", return_value=((1+0j), 0)),
            patch.object(
                engine,
                "_sum_q3_free_via_one_shot_cutset_scaled",
                side_effect=AssertionError("cluster plan should run before one-shot cutset"),
            ),
        ):
            total = engine._sum_q3_free_via_gauss_reduction_scaled(q)

        self.assertEqual(total, ((1+0j), 0))

    def test_sparse_low_degree_component_prefers_cluster_plan_over_one_shot_treewidth(self):
        q = PhaseFunction(
            10,
            level=3,
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

        with (
            patch.object(engine, "_Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS", 4),
            patch.object(engine, "_q3_free_treewidth_order", return_value=list(range(10))),
            patch.object(engine, "_finalize_q3_free_treewidth_order", return_value=(list(range(10)), 20)),
            patch.object(engine, "_treewidth_order_width", return_value=20),
            patch.object(engine, "_build_native_q3_free_treewidth_plan", return_value=None),
            patch.object(engine, "_build_q1_cluster_plan", return_value=cluster_plan),
            patch.object(
                engine,
                "_q3_free_one_shot_cutset_conditioning_plan",
                side_effect=AssertionError("better cluster plan should avoid one-shot cutset planning"),
            ),
        ):
            isolated_vars, component_plans = engine._plan_q3_free_constraint_components(
                q,
                0,
                prefer_one_shot_slicing=True,
            )

        self.assertEqual(isolated_vars, ())
        self.assertEqual(len(component_plans), 1)
        self.assertEqual(component_plans[0].backend, "generic")
        self.assertIs(component_plans[0].cluster_plan, cluster_plan)
        self.assertFalse(component_plans[0].prefer_cutset_backend)

    def test_one_shot_planner_skips_base_search_on_giant_low_degree_kernel(self):
        q = PhaseFunction(
            10,
            level=3,
            q1=[1] * 10,
            q2={(idx, idx + 1): 1 for idx in range(9)},
            q3={},
        )
        variant_plan = engine._Q3FreeCutsetConditioningPlan(
            level=3,
            cutset_vars=(2,),
            remaining_vars=tuple(var for var in range(10) if var != 2),
            remaining_backend="treewidth",
            remaining_q2={},
            remaining_order=tuple(range(9)),
            cutset_remaining_q2_residue=np.zeros((1, 9), dtype=np.int64),
            cutset_cutset_left=np.zeros(0, dtype=np.int64),
            cutset_cutset_right=np.zeros(0, dtype=np.int64),
            cutset_cutset_residue=np.zeros(0, dtype=np.int64),
            remaining_width=3,
            estimated_total_work=8,
        )

        with (
            patch.object(engine, "_Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS", 4),
            patch.object(
                engine,
                "_q3_free_cutset_conditioning_plan",
                side_effect=AssertionError("giant low-degree one-shot path should skip base search"),
            ),
            patch.object(engine, "_separator_ranked_q3_free_cutset_vertices", return_value=(2, 3, 4)),
            patch.object(engine, "_candidate_q3_free_cutset_vertices", return_value=(2, 3, 4)),
            patch.object(engine, "_order_guided_q3_free_cutset_vertices", return_value=()),
            patch.object(engine, "_build_q3_free_cutset_conditioning_plan_uncached", return_value=variant_plan),
        ):
            plan = engine._q3_free_one_shot_cutset_conditioning_plan(q)

        self.assertIs(plan, variant_plan)

    def test_gauss_reduction_keeps_generic_path_when_one_shot_cutset_plan_is_too_wide(self):
        q2 = {}
        edge_count = 0
        for left in range(90):
            for right in range(left + 1, 90):
                q2[(left, right)] = 1
                edge_count += 1
                if edge_count >= 270:
                    break
            if edge_count >= 270:
                break
        q = PhaseFunction(90, level=3, q2=q2, q3={})
        plan = engine._Q3FreeCutsetConditioningPlan(
            level=3,
            cutset_vars=(0, 1, 2, 3, 4, 5, 6, 7),
            remaining_vars=tuple(range(8, 90)),
            remaining_backend="treewidth",
            remaining_q2={},
            remaining_order=tuple(range(82)),
            cutset_remaining_q2_residue=np.zeros((8, 82), dtype=np.int64),
            cutset_cutset_left=np.zeros(0, dtype=np.int64),
            cutset_cutset_right=np.zeros(0, dtype=np.int64),
            cutset_cutset_residue=np.zeros(0, dtype=np.int64),
            remaining_width=17,
            estimated_total_work=1024,
        )
        mediator_plan = object()

        with (
            patch.object(engine, "_min_fill_cubic_order", return_value=(tuple(range(q.n)), 24)),
            patch.object(engine, "_q3_free_one_shot_cutset_conditioning_plan", return_value=plan),
            patch.object(engine, "_build_generic_q2_mediator_plan", return_value=mediator_plan) as generic_plan,
            patch.object(
                engine,
                "_evaluate_generic_q2_mediator_plan_scaled",
                return_value=engine._ONE_SCALED,
            ) as evaluate_generic,
        ):
            total = engine._sum_q3_free_via_gauss_reduction_scaled(q)

        self.assertEqual(total, engine._ONE_SCALED)
        generic_plan.assert_called_once()
        evaluate_generic.assert_called_once_with(mediator_plan, q.q1)
