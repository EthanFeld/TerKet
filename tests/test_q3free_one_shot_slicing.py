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
    def test_one_shot_amplitude_requests_raw_constraint_slicing(self):
        circuit = make_circuit(1, [("h", 0)])
        requested: dict[str, object] = {}
        restricted_plan = engine._Q3FreeRawConstraintRestrictedPlan(
            active_count=1,
            isolated_vars=(),
            components=(),
        )

        def fake_build(state, **kwargs):
            del state
            requested.update(kwargs)
            return object()

        with (
            patch.object(engine, "_build_q3_free_raw_constraint_plan", side_effect=fake_build),
            patch.object(engine, "_restrict_q3_free_raw_constraint_plan", return_value=restricted_plan),
            patch.object(engine, "_evaluate_q3_free_raw_constraint_plan_scaled", return_value=engine._ONE_SCALED),
        ):
            compute_circuit_amplitude(circuit, [0], [0], as_complex=True)

        self.assertIs(requested.get("prefer_one_shot_slicing"), True)

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


if __name__ == "__main__":
    unittest.main()
