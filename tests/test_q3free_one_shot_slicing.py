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


class Q3FreeOneShotSlicingTests(unittest.TestCase):
    def test_select_feedback_vertices_covers_all_chords(self):
        chords = [
            (0, 2, 1),
            (1, 2, 1),
            (1, 3, 1),
            (2, 4, 1),
        ]
        depth = [0, 1, 2, 3, 4]

        chosen = engine._select_feedback_vertices(5, chords, depth)

        self.assertEqual(chosen, [2, 3])
        covered = {idx for idx, (left, right, _phase) in enumerate(chords) if left in chosen or right in chosen}
        self.assertEqual(covered, set(range(len(chords))))

    def test_articulation_boundary_region_candidates_find_lobe(self):
        adjacency = [
            {1},            # 0
            {0, 2, 5},      # 1 articulation into cycle lobe
            {1, 3},         # 2
            {2, 4},         # 3
            {3, 5},         # 4
            {4, 1},         # 5
        ]

        candidates = engine._articulation_boundary_region_candidates(
            adjacency,
            min_region_size=4,
            max_region_size=8,
            max_boundary=2,
        )

        self.assertIn((2, 3, 4, 5), candidates)

    def test_selected_boundary_region_plan_uses_articulation_candidates(self):
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
                (1, 5): 1,
            },
            q3={},
        )
        adjacency = engine._build_q2_adjacency(q)
        candidates = engine._articulation_boundary_region_candidates(
            adjacency,
            min_region_size=4,
            max_region_size=4,
            max_boundary=2,
        )
        plan = engine._build_selected_boundary_region_plan(
            q,
            adjacency=adjacency,
            candidate_regions=candidates,
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        cluster_sets = {tuple(sorted(spec.cluster_vars)) for spec in plan.clusters}
        self.assertIn((2, 3, 4, 5), cluster_sets)

    def test_block_cut_boundary_region_candidates_find_two_boundary_lobe(self):
        adjacency = [
            {1, 5, 6},      # 0 boundary articulation with tail 6
            {0, 2},         # 1
            {1, 3, 7},      # 2 boundary articulation with tail 7
            {2, 4},         # 3
            {3, 5},         # 4
            {4, 0},         # 5
            {0},            # 6 external tail
            {2},            # 7 external tail
        ]

        candidates = engine._block_cut_boundary_region_candidates(
            adjacency,
            min_region_size=4,
            max_region_size=4,
            max_boundary=2,
        )

        self.assertIn((1, 3, 4, 5), candidates)

    def test_q2_block_cut_decomposition_finds_cycle_block_and_articulations(self):
        adjacency = [
            {1, 5, 6},
            {0, 2},
            {1, 3, 7},
            {2, 4},
            {3, 5},
            {4, 0},
            {0},
            {2},
        ]

        blocks, articulation = engine._q2_block_cut_decomposition(adjacency)

        self.assertIn((0, 1, 2, 3, 4, 5), blocks)
        self.assertEqual(articulation, frozenset({0, 2}))

    def test_selected_boundary_region_plan_uses_block_cut_candidates(self):
        q = PhaseFunction(
            9,
            level=3,
            q1=[1] * 9,
            q2={
                (0, 1): 1,
                (1, 2): 1,
                (2, 3): 1,
                (3, 4): 1,
                (4, 5): 1,
                (5, 0): 1,
                (0, 6): 1,
                (2, 7): 1,
                (0, 8): 1,
            },
            q3={},
        )
        adjacency = engine._build_q2_adjacency(q)
        candidates = engine._block_cut_boundary_region_candidates(
            adjacency,
            min_region_size=4,
            max_region_size=4,
            max_boundary=2,
        )
        plan = engine._build_selected_boundary_region_plan(
            q,
            adjacency=adjacency,
            candidate_regions=candidates,
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        cluster_sets = {tuple(sorted(spec.cluster_vars)) for spec in plan.clusters}
        self.assertIn((1, 3, 4, 5), cluster_sets)

    def test_build_block_cut_tree_region_plan_contracts_cycle_block(self):
        q = PhaseFunction(
            8,
            level=3,
            q1=[1] * 8,
            q2={
                (0, 1): 1,
                (1, 2): 1,
                (2, 3): 1,
                (3, 4): 1,
                (4, 5): 1,
                (5, 0): 1,
                (0, 6): 1,
                (2, 7): 1,
            },
            q3={},
        )

        plan = engine._build_block_cut_tree_region_plan(q)

        self.assertIsNotNone(plan)
        assert plan is not None
        cluster_sets = {tuple(sorted(spec.cluster_vars)) for spec in plan.clusters}
        self.assertIn((1, 3, 4, 5), cluster_sets)
        self.assertTrue({0, 2}.issubset(set(plan.core_vars)))
        self.assertNotIn(1, set(plan.core_vars))

    def test_build_block_cut_tree_region_plan_uses_cache(self):
        q = PhaseFunction(
            8,
            level=3,
            q1=[1] * 8,
            q2={
                (0, 1): 1,
                (1, 2): 1,
                (2, 3): 1,
                (3, 4): 1,
                (4, 5): 1,
                (5, 0): 1,
                (0, 6): 1,
                (2, 7): 1,
            },
            q3={},
        )

        plan1 = engine._build_block_cut_tree_region_plan(q)
        plan2 = engine._build_block_cut_tree_region_plan(q)

        self.assertIs(plan1, plan2)

    def test_sum_acyclic_factor_tables_matches_bucket_elimination(self):
        omega = engine._omega_scaled_table(3)
        factors = {
            (0,): [engine._ONE_SCALED, omega[1]],
            (0, 1): [engine._ONE_SCALED, engine._ONE_SCALED, engine._ONE_SCALED, omega[2]],
            (1, 2): [engine._ONE_SCALED, engine._ONE_SCALED, engine._ONE_SCALED, omega[3]],
        }

        tree_total = engine._sum_acyclic_factor_tables_scaled(3, factors, scalar=engine._ONE_SCALED)
        bucket_total, _ = engine._sum_factor_tables_scaled(3, factors, (0, 1, 2), scalar=engine._ONE_SCALED)

        self.assertEqual(engine._scaled_to_complex(tree_total), engine._scaled_to_complex(bucket_total))

    def test_cluster_plan_eval_uses_acyclic_tree_sum_before_bucket_elimination(self):
        q = PhaseFunction(
            8,
            level=3,
            q1=[1] * 8,
            q2={
                (0, 1): 1,
                (1, 2): 1,
                (2, 3): 1,
                (3, 4): 1,
                (4, 5): 1,
                (5, 0): 1,
                (0, 6): 1,
                (2, 7): 1,
            },
            q3={},
        )
        plan = engine._build_block_cut_tree_region_plan(q)

        self.assertIsNotNone(plan)
        assert plan is not None
        with patch.object(
            engine,
            "_sum_factor_tables_scaled",
            side_effect=AssertionError("acyclic block-cut factor graph should use tree sum"),
        ):
            total = engine._evaluate_half_phase_cluster_plan_scaled(plan, q.q1)

        self.assertIsInstance(total, tuple)

    def test_sum_acyclic_factor_tables_batch_matches_scalar_rows(self):
        omega = engine._omega_scaled_table(3)
        scalar_values = np.asarray([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        scalar_exponents = np.asarray([0, 0], dtype=np.int64)
        factors = {
            (0,): (
                np.asarray([[1.0 + 0j, omega[1][0]], [1.0 + 0j, omega[2][0]]], dtype=np.complex128),
                np.asarray([[0, omega[1][1]], [0, omega[2][1]]], dtype=np.int64),
            ),
            (0, 1): (
                np.asarray(
                    [
                        [1.0 + 0j, 1.0 + 0j, 1.0 + 0j, omega[3][0]],
                        [1.0 + 0j, 1.0 + 0j, 1.0 + 0j, omega[4][0]],
                    ],
                    dtype=np.complex128,
                ),
                np.asarray(
                    [
                        [0, 0, 0, omega[3][1]],
                        [0, 0, 0, omega[4][1]],
                    ],
                    dtype=np.int64,
                ),
            ),
        }

        batch_totals = engine._sum_acyclic_factor_tables_scaled_batch(
            2,
            factors,
            scalar=(scalar_values, scalar_exponents),
        )

        self.assertIsNotNone(batch_totals)
        assert batch_totals is not None
        for row_idx, total in enumerate(batch_totals):
            row_factors = {
                scope: [
                    (complex(values[row_idx, col]), int(exponents[row_idx, col]))
                    for col in range(values.shape[1])
                ]
                for scope, (values, exponents) in factors.items()
            }
            scalar_total = engine._sum_acyclic_factor_tables_scaled(
                2,
                row_factors,
                scalar=(complex(scalar_values[row_idx]), int(scalar_exponents[row_idx])),
            )
            self.assertEqual(engine._scaled_to_complex(total), engine._scaled_to_complex(scalar_total))

    def test_cluster_plan_eval_batch_uses_acyclic_tree_sum_before_bucket_elimination(self):
        q = PhaseFunction(
            8,
            level=3,
            q1=[1] * 8,
            q2={
                (0, 1): 1,
                (1, 2): 1,
                (2, 3): 1,
                (3, 4): 1,
                (4, 5): 1,
                (5, 0): 1,
                (0, 6): 1,
                (2, 7): 1,
            },
            q3={},
        )
        plan = engine._build_block_cut_tree_region_plan(q)

        self.assertIsNotNone(plan)
        assert plan is not None
        with patch.object(
            engine,
            "_sum_factor_tables_scaled_batch",
            side_effect=AssertionError("acyclic block-cut factor graph batch should use tree sum"),
        ):
            totals = engine._evaluate_half_phase_cluster_plan_scaled_batch(
                plan,
                np.asarray([q.q1, q.q1], dtype=np.int64),
            )

        self.assertEqual(len(totals), 2)

    def test_sum_q3_free_component_scaled_prefers_block_cut_tree_route_early(self):
        q = PhaseFunction(
            9,
            level=3,
            q1=[1] * 9,
            q2={
                (0, 1): 1,
                (1, 2): 1,
                (2, 3): 1,
                (3, 4): 1,
                (4, 5): 1,
                (5, 0): 1,
                (0, 6): 1,
                (2, 7): 1,
                (0, 8): 1,
            },
            q3={},
        )
        plan = engine._build_block_cut_tree_region_plan(q)
        self.assertIsNotNone(plan)
        assert plan is not None

        with (
            patch.object(engine, "_sum_q3_free_via_gauss_reduction_scaled", return_value=None),
            patch.object(engine, "_build_block_cut_tree_region_plan", return_value=plan),
            patch.object(engine, "_evaluate_half_phase_cluster_plan_scaled", return_value=((1+0j), 0)),
            patch.object(
                engine,
                "_sum_binary_phase_quadratic_scaled",
                side_effect=AssertionError("block-cut direct route should run before deeper residual solve"),
            ),
        ):
            total = engine._sum_q3_free_component_scaled(q)

        self.assertEqual(total, ((1+0j), 0))

    def test_amplitude_path_does_not_use_raw_constraint_shortcut(self):
        circuit = make_circuit(1, [("h", 0)])
        with (
            patch.object(engine, "_build_q3_free_raw_constraint_plan") as build_plan,
            patch.object(engine, "_restrict_q3_free_raw_constraint_plan") as restrict_plan,
            patch.object(engine, "_evaluate_q3_free_raw_constraint_plan_scaled") as evaluate_plan,
        ):
            compute_circuit_amplitude(circuit, [0], [0], as_complex=True)

        build_plan.assert_not_called()
        restrict_plan.assert_not_called()
        evaluate_plan.assert_not_called()

    def test_cutset_plan_finalizer_propagates_nested_one_shot_slicing(self):
        q = PhaseFunction(
            3,
            level=3,
            q2={
                (0, 1): 1,
                (1, 2): 1,
            },
        )
        nested_plan = engine._Q3FreeConstraintComponentPlan(
            variables=(0, 1),
            level=3,
            q2={(0, 1): 1},
            backend="treewidth",
            order=(0, 1),
        )
        requested: dict[str, object] = {}

        def fake_nested_plan(base_q, lambda_offset, **kwargs):
            del base_q, lambda_offset
            requested.update(kwargs)
            return (), (nested_plan,)

        plan = engine._Q3FreeCutsetConditioningPlan(
            level=3,
            cutset_vars=(0,),
            remaining_vars=(1, 2),
            remaining_backend="generic",
            remaining_q2={(0, 1): 1},
            remaining_order=(),
            cutset_remaining_q2_residue=np.zeros((1, 2), dtype=np.int64),
            cutset_cutset_left=np.zeros(0, dtype=np.int64),
            cutset_cutset_right=np.zeros(0, dtype=np.int64),
            cutset_cutset_residue=np.zeros(0, dtype=np.int64),
            remaining_isolated_vars=(),
            remaining_components=(),
            remaining_width=2,
            estimated_total_work=4,
        )

        with patch.object(engine, "_plan_q3_free_constraint_components", side_effect=fake_nested_plan):
            finalized = engine._finalize_q3_free_cutset_conditioning_plan(
                plan,
                prefer_one_shot_slicing=True,
            )

        self.assertIs(requested.get("prefer_one_shot_slicing"), True)
        self.assertEqual(finalized.remaining_components, (nested_plan,))

    def test_dense_conditioned_component_prefers_cutset_backend_in_one_shot_mode(self):
        base_q = PhaseFunction(
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
        )
        dummy_cutset_plan = engine._Q3FreeCutsetConditioningPlan(
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
            estimated_total_work=8,
        )

        with (
            patch.object(engine, "_Q3_FREE_DENSE_PLAN_MIN_DEGREE", 1),
            patch.object(engine, "_Q3_FREE_DENSE_PLAN_MIN_DENSITY", 0.0),
            patch.object(engine, "_build_half_phase_mediator_plan", return_value=None),
            patch.object(engine, "_build_generic_q2_mediator_plan", return_value=None),
            patch.object(engine, "_build_q1_cluster_plan", return_value=None),
            patch.object(engine, "_supports_exact_dense_schur", return_value=False),
            patch.object(engine, "_q3_free_treewidth_order", return_value=None),
            patch.object(engine, "_q3_free_prefers_locality_preserving_cutset", return_value=False),
            patch.object(engine, "_q3_free_one_shot_cutset_conditioning_plan", return_value=dummy_cutset_plan),
        ):
            isolated_vars, component_plans = engine._plan_q3_free_constraint_components(
                base_q,
                lambda_offset=3,
                prefer_one_shot_slicing=True,
            )

        self.assertEqual(isolated_vars, ())
        self.assertEqual(len(component_plans), 1)
        self.assertIs(component_plans[0].cutset_plan, dummy_cutset_plan)
        self.assertTrue(component_plans[0].prefer_cutset_backend)

    def test_constraint_plan_applies_phase_optimizer_before_backends(self):
        cache = engine.EchelonCache(
            n=1,
            m=1,
            echelon_rows=(0,),
            pivot_col=(-1,),
            used_mask=0,
            row_ops=(0,),
            free_vars=(0,),
            gamma_masks=(1,),
            n_free=1,
        )
        plan = engine._Q3FreeConstraintPlan(
            cache=cache,
            eps0=(0,),
            level=3,
            q0=0,
            base_q1=(0,),
            base_q2={},
            lambda_offset=1,
            rank=0,
            n_free_after_constraints=1,
            rhs_linear_coeff=4,
            isolated_vars=(),
            components=(),
        )
        optimized_q = PhaseFunction(1, level=3, q1=[4], q2={}, q3={})
        execution_plan = engine._Q3FreeExecutionPlan(
            level=3,
            q0=0,
            q1=(4,),
            isolated_vars=(0,),
            components=(),
        )

        with (
            patch.object(engine, "_optimize_q3_free_phase", return_value=(optimized_q, True)) as optimize,
            patch.object(engine, "_build_q3_free_execution_plan", return_value=execution_plan) as build_plan,
            patch.object(engine, "_evaluate_q3_free_execution_plan_scaled", return_value=engine._ONE_SCALED) as execute_plan,
        ):
            total = engine._evaluate_q3_free_constraint_plan_scaled(plan, [0])

        self.assertEqual(total, engine._ONE_SCALED)
        optimize.assert_called_once()
        build_plan.assert_called_once()
        execute_plan.assert_called_once()

    def test_raw_constraint_plan_applies_phase_optimizer_before_backends(self):
        plan = engine._Q3FreeRawConstraintPlan(
            eps0=(0,),
            level=3,
            q0=0,
            base_q1=(0,),
            base_q2={},
            lambda_offset=1,
            constraint_count=1,
            rhs_linear_coeff=4,
            isolated_vars=(),
            components=(),
        )
        restricted = engine._Q3FreeRawConstraintRestrictedPlan(
            active_count=0,
            isolated_vars=(),
            components=(),
        )
        optimized_q = PhaseFunction(1, level=3, q1=[4], q2={}, q3={})
        execution_plan = engine._Q3FreeExecutionPlan(
            level=3,
            q0=0,
            q1=(4,),
            isolated_vars=(0,),
            components=(),
        )

        with (
            patch.object(engine, "_optimize_q3_free_phase", return_value=(optimized_q, True)) as optimize,
            patch.object(engine, "_build_q3_free_execution_plan", return_value=execution_plan) as build_plan,
            patch.object(engine, "_evaluate_q3_free_execution_plan_scaled", return_value=engine._ONE_SCALED) as execute_plan,
        ):
            total = engine._evaluate_q3_free_raw_constraint_plan_scaled(plan, restricted, [])

        self.assertEqual(total, engine._ONE_SCALED)
        optimize.assert_called_once()
        build_plan.assert_called_once()
        execute_plan.assert_called_once()

    def test_q3_free_execution_plan_is_cached_by_structure(self):
        q = PhaseFunction(
            2,
            level=3,
            q1=[4, 0],
            q2={(0, 1): 2},
            q3={},
        )

        with patch.object(
            engine,
            "_plan_q3_free_constraint_components",
            return_value=((), ()),
        ) as planner:
            first = engine._build_q3_free_execution_plan(
                q=q,
                allow_tensor_contraction=False,
                prefer_one_shot_slicing=True,
            )
            second = engine._build_q3_free_execution_plan(
                q=q,
                allow_tensor_contraction=False,
                prefer_one_shot_slicing=True,
            )

        self.assertIs(first, second)
        planner.assert_called_once()

    def test_q3_free_execution_plan_reuses_reusable_structure_for_shifted_q1(self):
        q0 = PhaseFunction(
            4,
            level=3,
            q1=[1, 3, 0, 4],
            q2={(0, 1): 2, (1, 2): 2, (2, 3): 2},
            q3={},
        )
        q1 = PhaseFunction(
            4,
            level=3,
            q1=[5, 7, 0, 4],
            q2={(0, 1): 2, (1, 2): 2, (2, 3): 2},
            q3={},
        )
        q2 = PhaseFunction(
            4,
            level=3,
            q1=[7, 1, 0, 4],
            q2={(0, 1): 2, (1, 2): 2, (2, 3): 2},
            q3={},
        )

        planner_modes: list[bool] = []

        def fake_plan(*args, **kwargs):
            planner_modes.append(bool(kwargs.get("prefer_reusable_decomposition")))
            return (), ()

        engine._STRUCTURE_Q3_FREE_EXECUTION_PLAN_CACHE.clear()
        engine._STRUCTURE_Q3_FREE_REUSABLE_EXECUTION_PLAN_CACHE.clear()
        with (
            patch.object(engine, "_Q3_FREE_REUSABLE_EXECUTION_PLAN_MIN_VARS", 0),
            patch.object(engine, "_plan_q3_free_constraint_components", side_effect=fake_plan),
        ):
            first = engine._build_q3_free_execution_plan(
                q=q0,
                allow_tensor_contraction=False,
            )
            second = engine._build_q3_free_execution_plan(
                q=q1,
                allow_tensor_contraction=False,
            )
            third = engine._build_q3_free_execution_plan(
                q=q2,
                allow_tensor_contraction=False,
            )

        self.assertEqual(planner_modes, [False, True])
        self.assertEqual(tuple(first.q1), (1, 3, 0, 4))
        self.assertEqual(tuple(second.q1), (5, 7, 0, 4))
        self.assertEqual(tuple(third.q1), (7, 1, 0, 4))
        self.assertEqual(first.components, second.components)
        self.assertEqual(second.components, third.components)

    def test_q3_free_reusable_execution_plan_matches_direct_sum_for_shifted_q1(self):
        q0 = PhaseFunction(
            6,
            level=3,
            q1=[1, 3, 0, 4, 5, 0],
            q2={(0, 1): 2, (1, 2): 2, (3, 4): 2, (4, 5): 2},
            q3={},
        )
        q1 = PhaseFunction(
            6,
            level=3,
            q1=[5, 7, 0, 4, 1, 0],
            q2={(0, 1): 2, (1, 2): 2, (3, 4): 2, (4, 5): 2},
            q3={},
        )

        engine._STRUCTURE_Q3_FREE_EXECUTION_PLAN_CACHE.clear()
        engine._STRUCTURE_Q3_FREE_REUSABLE_EXECUTION_PLAN_CACHE.clear()
        with patch.object(engine, "_Q3_FREE_REUSABLE_EXECUTION_PLAN_MIN_VARS", 0):
            engine._build_q3_free_execution_plan(
                q=q0,
                allow_tensor_contraction=False,
            )
            reusable_plan = engine._build_q3_free_execution_plan(
                q=q1,
                allow_tensor_contraction=False,
            )

        expected = engine._sum_q3_free_component_scaled(q1)
        actual = engine._evaluate_q3_free_execution_plan_scaled(reusable_plan)
        self.assertAlmostEqual(abs(engine._scaled_to_complex(actual) - engine._scaled_to_complex(expected)), 0.0)

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

if __name__ == "__main__":
    unittest.main()
