"""Tests for phase-structure optimizer behavior on q3-free paths."""

from __future__ import annotations

import cmath
import sys
from pathlib import Path
import unittest
import unittest.mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket import engine
from terket import _reduction_classify as reduction_classify
from terket import _reduction_runtime as reduction_runtime
from terket.cubic_arithmetic import PhaseFunction

def _bruteforce_phase_sum(q: PhaseFunction) -> complex:
    total = 0j
    for mask in range(1 << q.n):
        bits = [(mask >> bit) & 1 for bit in range(q.n)]
        total += cmath.exp(2j * cmath.pi * float(q.evaluate(bits)))
    return total

class PhaseStructureQ3FreeOptimizerTests(unittest.TestCase):

    def test_exact_elimination_batches_many_sparse_quadratic_pivots(self):
        n = 96
        q = PhaseFunction(
            n,
            level=3,
            q1=[2] * n,
            q2={(idx, idx + 1): 2 for idx in range(n - 1)},
            q3={},
        )

        with unittest.mock.patch.object(
            reduction_runtime,
            "_elim_sparse_dead_quadratics_batch",
            wraps=reduction_classify._elim_sparse_dead_quadratics_batch,
        ) as batch_eliminate:
            reduced_q, half_pow2, info, blocked = reduction_runtime._apply_exact_eliminations(q)

        self.assertIsNotNone(reduced_q)
        self.assertEqual(reduced_q.n, 0)
        self.assertEqual(half_pow2, n)
        self.assertGreater(info["quad"], 0)
        self.assertGreater(info["constraint"], 0)
        self.assertFalse(blocked)
        batch_eliminate.assert_called()

    def test_sparse_quadratic_batch_rechecks_adjacent_candidate_residue(self):
        q = PhaseFunction(
            3,
            level=3,
            q1=[2, 2, 2],
            q2={(0, 1): 2, (1, 2): 2},
            q3={},
        )

        reduced_q, half_pow2, removed = reduction_classify._elim_sparse_dead_quadratics_batch(
            q,
            (0, 1, 2),
        )

        self.assertEqual(removed, (0, 2))
        self.assertEqual(half_pow2, 2)
        self.assertAlmostEqual(
            abs(
                _bruteforce_phase_sum(q)
                - (2 ** (half_pow2 / 2)) * _bruteforce_phase_sum(reduced_q)
            ),
            0.0,
            places=12,
        )

    def test_q3_free_optimizer_rejects_structural_rewrite_with_worse_runtime_plan(self):
        q = PhaseFunction(2, level=3, q1=[4, 0], q2={(0, 1): 2}, q3={})
        structurally_better = PhaseFunction(2, level=3, q1=[0, 4], q2={(0, 1): 1}, q3={})

        baseline_plan = engine._Q3FreeExecutionPlan(
            level=3,
            q0=0,
            q1=(4, 0),
            isolated_vars=(),
            components=(
                engine._Q3FreeConstraintComponentPlan(
                    variables=(0, 1),
                    level=3,
                    q2={(0, 1): 2},
                    backend="generic",
                    cutset_plan=engine._Q3FreeCutsetConditioningPlan(
                        level=3,
                        cutset_vars=(0,),
                        remaining_vars=(1,),
                        remaining_backend="treewidth",
                        remaining_q2={},
                        remaining_order=(0,),
                        cutset_remaining_q2_residue=engine.np.zeros((1, 1), dtype=engine.np.int64),
                        cutset_cutset_left=engine.np.zeros(0, dtype=engine.np.int64),
                        cutset_cutset_right=engine.np.zeros(0, dtype=engine.np.int64),
                        cutset_cutset_residue=engine.np.zeros(0, dtype=engine.np.int64),
                        remaining_width=4,
                        estimated_total_work=10,
                    ),
                    prefer_cutset_backend=True,
                ),
            ),
        )
        worse_plan = engine._Q3FreeExecutionPlan(
            level=3,
            q0=0,
            q1=(0, 4),
            isolated_vars=(),
            components=(
                engine._Q3FreeConstraintComponentPlan(
                    variables=(0, 1),
                    level=3,
                    q2={(0, 1): 1},
                    backend="treewidth",
                    order=(0, 1),
                ),
            ),
        )

        with unittest.mock.patch.object(
            engine,
            "_optimize_phase_function_structure",
            return_value=(structurally_better, True),
        ), unittest.mock.patch.object(
            engine,
            "_build_q3_free_execution_plan",
            side_effect=[baseline_plan, worse_plan],
        ), unittest.mock.patch.object(
            engine,
            "_q3_free_execution_plan_runtime_score",
            side_effect=[(10, 4, 0, 0, 1), (20, 6, 0, 1, 1)],
        ):
            optimized_q, changed = engine._optimize_q3_free_phase(q, allow_tensor_contraction=False)

        self.assertFalse(changed)
        self.assertIs(optimized_q, q)

    def test_q3_free_optimizer_accepts_structural_rewrite_with_better_runtime_plan(self):
        q = PhaseFunction(2, level=3, q1=[4, 0], q2={(0, 1): 2}, q3={})
        structurally_better = PhaseFunction(2, level=3, q1=[0, 4], q2={(0, 1): 1}, q3={})

        baseline_plan = engine._Q3FreeExecutionPlan(
            level=3,
            q0=0,
            q1=(4, 0),
            isolated_vars=(),
            components=(
                engine._Q3FreeConstraintComponentPlan(
                    variables=(0, 1),
                    level=3,
                    q2={(0, 1): 2},
                    backend="treewidth",
                    order=(0, 1),
                ),
            ),
        )
        better_plan = engine._Q3FreeExecutionPlan(
            level=3,
            q0=0,
            q1=(0, 4),
            isolated_vars=(),
            components=(
                engine._Q3FreeConstraintComponentPlan(
                    variables=(0, 1),
                    level=3,
                    q2={(0, 1): 1},
                    backend="generic",
                    cutset_plan=engine._Q3FreeCutsetConditioningPlan(
                        level=3,
                        cutset_vars=(0,),
                        remaining_vars=(1,),
                        remaining_backend="treewidth",
                        remaining_q2={},
                        remaining_order=(0,),
                        cutset_remaining_q2_residue=engine.np.zeros((1, 1), dtype=engine.np.int64),
                        cutset_cutset_left=engine.np.zeros(0, dtype=engine.np.int64),
                        cutset_cutset_right=engine.np.zeros(0, dtype=engine.np.int64),
                        cutset_cutset_residue=engine.np.zeros(0, dtype=engine.np.int64),
                        remaining_width=3,
                        estimated_total_work=8,
                    ),
                    prefer_cutset_backend=True,
                ),
            ),
        )

        with unittest.mock.patch.object(
            engine,
            "_optimize_phase_function_structure",
            return_value=(structurally_better, True),
        ), unittest.mock.patch.object(
            engine,
            "_build_q3_free_execution_plan",
            side_effect=[baseline_plan, better_plan],
        ), unittest.mock.patch.object(
            engine,
            "_q3_free_execution_plan_runtime_score",
            side_effect=[(20, 6, 0, 1, 1), (10, 4, 0, 0, 1)],
        ):
            optimized_q, changed = engine._optimize_q3_free_phase(q, allow_tensor_contraction=False)

        self.assertTrue(changed)
        self.assertIs(optimized_q, structurally_better)

    def test_q3_free_optimizer_skips_optional_structure_search_on_giant_kernel(self):
        n = engine._Q3_FREE_OPTIONAL_REWRITE_MAX_VARS + 1
        q = PhaseFunction(
            n,
            level=3,
            q1=[4] * n,
            q2={(idx, idx + 1): 2 for idx in range(n - 1)},
            q3={},
        )

        with unittest.mock.patch.object(
            engine,
            "_optimize_phase_function_structure",
            side_effect=AssertionError("giant q3-free kernel should skip optional structure search"),
        ):
            optimized_q, changed = engine._optimize_q3_free_phase(q, allow_tensor_contraction=False)

        self.assertFalse(changed)
        self.assertIs(optimized_q, q)

    def test_half_phase_mediator_plan_skips_giant_kernel(self):
        n = engine._Q3_FREE_OPTIONAL_REWRITE_MAX_VARS + 1
        half_q1 = (1 << 3) // 2
        q = PhaseFunction(
            n,
            level=3,
            q1=[half_q1 + 1] * n,
            q2={(idx, idx + 1): 2 for idx in range(n - 1)},
            q3={},
        )

        with unittest.mock.patch.object(
            engine,
            "_min_fill_cubic_order",
            side_effect=AssertionError("giant half-phase kernel should skip mediator ordering"),
        ):
            plan = engine._build_half_phase_mediator_plan(q)

        self.assertIsNone(plan)

    def test_q3_free_optimizer_skips_when_one_shot_baseline_is_already_good(self):
        q = PhaseFunction(2, level=3, q1=[4, 0], q2={(0, 1): 2}, q3={})

        with unittest.mock.patch.object(
            engine,
            "_optimize_phase_function_structure",
        ) as optimize:
            optimized_q, changed = engine._optimize_q3_free_phase(
                q,
                allow_tensor_contraction=False,
                prefer_one_shot_slicing=True,
                baseline_runtime_score=(10, 4, 0, 0, 1),
            )

        self.assertFalse(changed)
        self.assertIs(optimized_q, q)
        optimize.assert_not_called()
