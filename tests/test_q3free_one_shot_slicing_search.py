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

class Q3FreeOneShotSlicingSearchTests(unittest.TestCase):

    def test_one_shot_cutset_candidate_does_not_replan_generic_remainder_during_search(self):
        q = PhaseFunction(
            4,
            level=3,
            q2={
                (0, 1): 1,
                (0, 2): 1,
                (0, 3): 1,
                (1, 2): 1,
                (1, 3): 1,
                (2, 3): 1,
            },
            q3={},
        )

        with (
            patch.object(engine, "_q3_free_treewidth_order", return_value=None),
            patch.object(engine, "_plan_q3_free_constraint_components") as planner,
        ):
            evaluation = engine._evaluate_q3_free_cutset_candidate(
                q,
                (0,),
                allow_generic_remaining=True,
                prefer_one_shot_slicing=True,
            )

        self.assertIsNotNone(evaluation)
        self.assertTrue(evaluation.viable)
        self.assertIsNotNone(evaluation.plan)
        self.assertEqual(evaluation.plan.remaining_backend, "generic")
        planner.assert_not_called()

    def test_gauss_reduction_routes_giant_dense_q2_to_one_shot_cutset(self):
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
            remaining_width=14,
            estimated_total_work=1024,
        )

        with (
            patch.object(engine, "_min_fill_cubic_order", return_value=(tuple(range(q.n)), 24)),
            patch.object(engine, "_q3_free_one_shot_cutset_conditioning_plan", return_value=plan),
            patch.object(
                engine,
                "_evaluate_q3_free_cutset_conditioning_plan_scaled",
                return_value=engine._ONE_SCALED,
            ) as evaluate_cutset,
            patch.object(engine, "_build_generic_q2_mediator_plan") as generic_plan,
        ):
            total = engine._sum_q3_free_via_gauss_reduction_scaled(q)

        self.assertEqual(total, engine._ONE_SCALED)
        evaluate_cutset.assert_called_once()
        generic_plan.assert_not_called()

    def test_sparse_large_q2_skips_dense_one_shot_direct_probe(self):
        n = max(90, engine._Q3_FREE_ONE_SHOT_DIRECT_MIN_VARS)
        q = PhaseFunction(
            n,
            level=3,
            q1=[1] * n,
            q2={(idx, idx + 1): 1 for idx in range(n - 1)},
            q3={},
        )

        with patch.object(
            engine,
            "_min_fill_cubic_order",
            side_effect=AssertionError("sparse giant q2 kernel should skip dense one-shot probe"),
        ):
            total = engine._sum_q3_free_via_one_shot_cutset_scaled(q)

        self.assertIsNone(total)

    def test_finalize_q3_free_treewidth_order_skips_local_refinement_on_giant_kernel(self):
        n = engine._Q3_FREE_OPTIONAL_REWRITE_MAX_VARS + 1
        q = PhaseFunction(
            n,
            level=3,
            q1=[1] * n,
            q2={(idx, idx + 1): 1 for idx in range(n - 1)},
            q3={},
        )
        order = tuple(range(n))

        with patch.object(
            engine,
            "_refine_q3_free_treewidth_order_locally",
            side_effect=AssertionError("giant q3-free kernel should skip local order refinement"),
        ):
            refined_order, refined_width = engine._finalize_q3_free_treewidth_order(q, order)

        self.assertEqual(refined_order, list(order))
        self.assertEqual(refined_width, engine._treewidth_order_width(q, order))

    def test_q3_free_treewidth_order_uses_cheap_chronological_hints_for_giant_low_degree_kernel(self):
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

        with (
            patch.object(engine, "_Q3_FREE_CHEAP_ORDER_HINT_MIN_VARS", 4),
            patch.object(engine, "_q3_free_treewidth_candidate_is_viable", return_value=True),
            patch.object(
                engine,
                "_min_degree_cubic_order_uncached",
                side_effect=AssertionError("cheap hint should bypass min-degree order search"),
            ),
            patch.object(
                engine,
                "_min_fill_cubic_order",
                side_effect=AssertionError("cheap hint should bypass min-fill order search"),
            ),
        ):
            order = engine._q3_free_treewidth_order(q, feedback_size=2, max_degree=2)

        self.assertEqual(order, list(range(q.n)))

    def test_q3_free_treewidth_order_peels_degree_two_shell_before_core_search(self):
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
        seen_sizes: list[int] = []

        def fake_min_degree(subq):
            seen_sizes.append(subq.n)
            return tuple(range(subq.n)), 3

        with (
            patch.object(engine, "_Q3_FREE_CHEAP_ORDER_HINT_MIN_VARS", 99),
            patch.object(engine, "_q3_free_treewidth_candidate_is_viable", return_value=True),
            patch.object(engine, "_min_degree_cubic_order_uncached", side_effect=fake_min_degree),
            patch.object(
                engine,
                "_min_fill_cubic_order",
                side_effect=AssertionError("series-reduced core should finish before min-fill"),
            ),
        ):
            order = engine._q3_free_treewidth_order(q, feedback_size=5, max_degree=3)

        self.assertEqual(seen_sizes, [4])
        self.assertEqual(order[:2], [4, 5])
        self.assertEqual(sorted(order), list(range(q.n)))

    def test_build_q1_cluster_plan_falls_back_to_small_boundary_regions(self):
        q = PhaseFunction(
            6,
            level=3,
            q1=[0] * 6,
            q2={
                (0, 1): 1,
                (1, 2): 1,
                (2, 3): 1,
                (3, 4): 1,
                (4, 5): 1,
            },
            q3={},
        )

        with patch.object(engine, "_Q3_FREE_SMALL_BOUNDARY_REGION_MIN_SIZE", 2):
            plan = engine._build_q1_cluster_plan(q)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertGreaterEqual(len(plan.clusters), 1)

    def test_order_guided_cutset_vertices_focus_peak_frontier(self):
        adjacency = [
            {1},
            {0, 2},
            {1, 3},
            {2, 4},
            {3, 5},
            {4},
        ]

        ranked = engine._order_guided_q3_free_cutset_vertices(
            adjacency,
            candidate_orders=(tuple(range(6)),),
            max_candidates=4,
        )

        self.assertSetEqual(set(ranked[:2]), {2, 3})

    def test_residual_projection_can_descend_from_parent_remainder(self):
        q = PhaseFunction(
            5,
            level=3,
            q1=[0] * 5,
            q2={
                (0, 1): 1,
                (1, 2): 1,
                (2, 3): 1,
                (3, 4): 1,
            },
            q3={},
        )
        parent = engine._build_q3_free_residual_projection(q, (1,))
        self.assertIsNotNone(parent)
        assert parent is not None

        child = engine._build_q3_free_residual_projection(
            q,
            (1, 3),
            parent_projection=parent,
        )

        self.assertIsNotNone(child)
        assert child is not None
        self.assertEqual(child.remaining_vars, (0, 2, 4))
        self.assertEqual(child.remaining_q.n, 3)
        self.assertEqual(child.remaining_q.q2, {})

    def test_cutset_search_reuses_projected_residual_order_in_greedy_growth(self):
        q = PhaseFunction(
            4,
            level=3,
            q1=[1] * 4,
            q2={
                (0, 1): 1,
                (1, 2): 1,
                (2, 3): 1,
            },
            q3={},
        )
        seen_hints: list[tuple[tuple[int, ...], tuple[int, ...] | None]] = []

        def fake_eval(base_q, cutset_vars, **kwargs):
            remaining_vars = tuple(var for var in range(base_q.n) if var not in set(cutset_vars))
            seen_hints.append(
                (
                    tuple(cutset_vars),
                    None if kwargs.get("remaining_order_hint") is None else tuple(kwargs["remaining_order_hint"]),
                )
            )
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
                remaining_width=max(0, 4 - len(cutset_vars)),
                estimated_total_work=max(1, len(remaining_vars)),
            )
            return engine._Q3FreeCutsetCandidateEvaluation(
                cutset_vars=tuple(cutset_vars),
                plan=plan,
                viable=True,
                score=(0, len(cutset_vars), cutset_vars),
            )

        with (
            patch.object(engine, "_evaluate_q3_free_cutset_candidate", side_effect=fake_eval),
            patch.object(engine, "_finalize_q3_free_treewidth_order", side_effect=lambda q_obj, order: (list(order), engine._treewidth_order_width(q_obj, order))),
        ):
            plan = engine._build_q3_free_cutset_conditioning_plan_uncached(
                q,
                max_size=2,
                candidate_pool=3,
                beam_width=2,
                branches_per_state=2,
                prioritize_width=True,
                target_remaining_width=2,
                candidate_override=(0, 1, 2),
            )

        self.assertIsNotNone(plan)
        hinted_calls = [hint for cutset, hint in seen_hints if len(cutset) == 2]
        self.assertIn((1, 2, 3), hinted_calls)

    def test_one_shot_generic_surrogate_skips_min_fill_on_giant_low_degree_residual(self):
        n = 10
        q = PhaseFunction(
            n,
            level=3,
            q1=[1] * n,
            q2={(idx, idx + 1): 1 for idx in range(n - 1)},
            q3={},
        )

        with (
            patch.object(engine, "_Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS", 4),
            patch.object(engine, "_q3_free_treewidth_order", return_value=None),
            patch.object(
                engine,
                "_min_fill_cubic_order",
                side_effect=AssertionError("surrogate generic scoring should skip min-fill"),
            ),
        ):
            evaluation = engine._evaluate_q3_free_cutset_candidate(
                q,
                (0,),
                allow_generic_remaining=True,
                prefer_one_shot_slicing=True,
            )

        self.assertIsNotNone(evaluation)
        assert evaluation is not None
        self.assertTrue(evaluation.viable)
        assert evaluation.plan is not None
        self.assertEqual(evaluation.plan.remaining_backend, "generic")

    def test_one_shot_surrogate_skips_exact_treewidth_probe_when_width_far_too_large(self):
        n = 10
        q = PhaseFunction(
            n,
            level=3,
            q1=[1] * n,
            q2={(idx, idx + 1): 1 for idx in range(n - 1)},
            q3={},
        )

        with (
            patch.object(engine, "_Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS", 4),
            patch.object(engine, "_best_cheap_q3_free_order", return_value=(tuple(range(n - 1)), 50)),
            patch.object(
                engine,
                "_q3_free_treewidth_order",
                side_effect=AssertionError("surrogate search should skip exact treewidth probe"),
            ),
        ):
            evaluation = engine._evaluate_q3_free_cutset_candidate(
                q,
                (0,),
                allow_generic_remaining=True,
                prefer_one_shot_slicing=True,
                target_remaining_width=14,
            )

        self.assertIsNotNone(evaluation)
        assert evaluation is not None
        self.assertTrue(evaluation.viable)
        assert evaluation.plan is not None
        self.assertEqual(evaluation.plan.remaining_backend, "generic")

    def test_one_shot_surrogate_generic_scoring_skips_factorization(self):
        n = 10
        q = PhaseFunction(
            n,
            level=3,
            q1=[1] * n,
            q2={(idx, idx + 1): 1 for idx in range(n - 1)},
            q3={},
        )

        with (
            patch.object(engine, "_Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS", 4),
            patch.object(engine, "_best_cheap_q3_free_order", return_value=(tuple(range(n - 1)), 50)),
            patch.object(
                engine,
                "_q3_free_treewidth_order",
                return_value=None,
            ),
            patch.object(
                engine,
                "detect_factorization",
                side_effect=AssertionError("surrogate generic scoring should skip factorization"),
            ),
        ):
            evaluation = engine._evaluate_q3_free_cutset_candidate(
                q,
                (0,),
                allow_generic_remaining=True,
                prefer_one_shot_slicing=True,
                target_remaining_width=14,
            )

        self.assertIsNotNone(evaluation)
        assert evaluation is not None
        self.assertTrue(evaluation.viable)
        assert evaluation.plan is not None
        self.assertEqual(evaluation.plan.remaining_backend, "generic")

    def test_one_shot_surrogate_skips_exact_work_estimate(self):
        n = 10
        q = PhaseFunction(
            n,
            level=3,
            q1=[1] * n,
            q2={(idx, idx + 1): 1 for idx in range(n - 1)},
            q3={},
        )

        with (
            patch.object(engine, "_Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS", 4),
            patch.object(
                engine,
                "_estimate_treewidth_dp_work",
                side_effect=AssertionError("surrogate candidate scoring should skip exact work estimate"),
            ),
        ):
            evaluation = engine._evaluate_q3_free_cutset_candidate(
                q,
                (0,),
                allow_generic_remaining=True,
                prefer_one_shot_slicing=True,
            )

        self.assertIsNotNone(evaluation)
        assert evaluation is not None
        self.assertTrue(evaluation.viable)
        assert evaluation.plan is not None
        self.assertGreater(evaluation.plan.estimated_total_work, 0)

    def test_one_shot_cutset_planner_tries_order_guided_candidates(self):
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
        base_plan = engine._Q3FreeCutsetConditioningPlan(
            level=3,
            cutset_vars=(0,),
            remaining_vars=(1, 2, 3, 4, 5),
            remaining_backend="treewidth",
            remaining_q2={},
            remaining_order=(0, 1, 2, 3, 4),
            cutset_remaining_q2_residue=np.zeros((1, 5), dtype=np.int64),
            cutset_cutset_left=np.zeros(0, dtype=np.int64),
            cutset_cutset_right=np.zeros(0, dtype=np.int64),
            cutset_cutset_residue=np.zeros(0, dtype=np.int64),
            remaining_width=99,
            estimated_total_work=999,
        )
        guided_plan = engine._Q3FreeCutsetConditioningPlan(
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
        uncached_calls: list[dict[str, object]] = []

        def fake_uncached(*args, **kwargs):
            uncached_calls.append(kwargs)
            if kwargs.get("remaining_order_hint") is not None:
                return guided_plan
            return None

        with (
            patch.object(engine, "_q3_free_cutset_conditioning_plan", return_value=base_plan),
            patch.object(engine, "_separator_ranked_q3_free_cutset_vertices", return_value=()),
            patch.object(engine, "_candidate_q3_free_cutset_vertices", return_value=()),
            patch.object(engine, "_build_q3_free_cutset_conditioning_plan_uncached", side_effect=fake_uncached),
        ):
            plan = engine._q3_free_one_shot_cutset_conditioning_plan(q)

        self.assertIs(plan, guided_plan)
        self.assertTrue(any(call.get("remaining_order_hint") is not None for call in uncached_calls))
        self.assertEqual(len(uncached_calls), 1)

    def test_one_shot_cutset_planner_recurses_on_series_reduced_core(self):
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
        mapped_plan = engine._Q3FreeCutsetConditioningPlan(
            level=3,
            cutset_vars=(0,),
            remaining_vars=(1, 2, 3, 4, 5),
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
        core_plan = engine._Q3FreeCutsetConditioningPlan(
            level=3,
            cutset_vars=(0,),
            remaining_vars=(1, 2, 3),
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
        uncached_calls: list[tuple[int, tuple[int, ...] | None]] = []

        def fake_uncached(base_q, **kwargs):
            uncached_calls.append((base_q.n, None if kwargs.get("candidate_override") is None else tuple(kwargs["candidate_override"])))
            if base_q.n == 4 and kwargs.get("candidate_override") is None:
                return core_plan
            if base_q.n == 6 and kwargs.get("candidate_override") is not None and 0 in tuple(kwargs["candidate_override"]):
                return mapped_plan
            return None

        with (
            patch.object(engine, "_q3_free_cutset_conditioning_plan", return_value=None),
            patch.object(engine, "_separator_ranked_q3_free_cutset_vertices", return_value=()),
            patch.object(engine, "_candidate_q3_free_cutset_vertices", return_value=()),
            patch.object(engine, "_order_guided_q3_free_cutset_vertices", return_value=()),
            patch.object(engine, "_Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS", 4),
            patch.object(engine, "_Q3_FREE_SERIES_CORE_RECURSE_MIN_SHRINK", 1),
            patch.object(engine, "_build_q3_free_cutset_conditioning_plan_uncached", side_effect=fake_uncached),
        ):
            plan = engine._q3_free_one_shot_cutset_conditioning_plan(q)

        self.assertIs(plan, mapped_plan)
        self.assertIn((4, None), uncached_calls)
        self.assertTrue(any(n == 6 and cand is not None and 0 in cand for n, cand in uncached_calls))
