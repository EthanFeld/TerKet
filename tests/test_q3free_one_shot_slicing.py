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
