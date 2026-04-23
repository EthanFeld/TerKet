from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket.cubic_arithmetic import PhaseFunction
from terket.engine import (
    _build_q3_free_execution_plan,
    _evaluate_q3_free_execution_plan_scaled,
    _q3_free_execution_plan_runtime_score,
    _rewrite_q3_free_phase_to_normal_form,
    _scaled_to_complex,
)


class Q3FreeNormalFormRewriteTests(unittest.TestCase):
    def test_normal_form_rewrite_preserves_sum_and_improves_runtime_score(self):
        q = PhaseFunction(
            4,
            level=3,
            q0=Fraction(0),
            q1=[2, 0, 1, 6],
            q2={(1, 2): 1},
            q3={},
        )
        baseline_plan = _build_q3_free_execution_plan(
            q=q,
            allow_tensor_contraction=False,
            prefer_one_shot_slicing=True,
        )
        baseline_score = _q3_free_execution_plan_runtime_score(baseline_plan)

        rewritten_q, rewrite_scale, changed, rewritten_plan, rewritten_score = _rewrite_q3_free_phase_to_normal_form(
            q,
            allow_tensor_contraction=False,
            prefer_one_shot_slicing=True,
            baseline_runtime_score=baseline_score,
        )

        self.assertTrue(changed)
        self.assertIsNotNone(rewritten_q)
        self.assertIsNotNone(rewritten_plan)
        self.assertIsNotNone(rewritten_score)
        self.assertLess(rewritten_score, baseline_score)

        baseline_total = _scaled_to_complex(
            _evaluate_q3_free_execution_plan_scaled(baseline_plan)
        )
        rewritten_total = _scaled_to_complex(
            _evaluate_q3_free_execution_plan_scaled(
                rewritten_plan,
                output_scale_half_pow2=rewrite_scale,
            )
        )

        self.assertAlmostEqual(baseline_total.real, rewritten_total.real, places=12)
        self.assertAlmostEqual(baseline_total.imag, rewritten_total.imag, places=12)

    def test_high_precision_parity_rewrite_preserves_sum_and_improves_runtime_score(self):
        q = PhaseFunction(
            8,
            level=5,
            q0=Fraction(0),
            q1=[0, 3, 5, 0, 11, 0, 13, 7],
            q2={
                (0, 1): 8,
                (0, 2): 8,
                (3, 2): 8,
                (3, 4): 8,
                (5, 4): 8,
                (5, 6): 8,
                (1, 7): 3,
                (6, 7): 5,
                (2, 4): 6,
            },
            q3={},
        )
        baseline_plan = _build_q3_free_execution_plan(
            q=q,
            allow_tensor_contraction=False,
            prefer_one_shot_slicing=True,
        )
        baseline_score = _q3_free_execution_plan_runtime_score(baseline_plan)

        rewritten_q, rewrite_scale, changed, rewritten_plan, rewritten_score = _rewrite_q3_free_phase_to_normal_form(
            q,
            allow_tensor_contraction=False,
            prefer_one_shot_slicing=True,
            baseline_runtime_score=baseline_score,
        )

        self.assertTrue(changed)
        self.assertIsNotNone(rewritten_q)
        self.assertIsNotNone(rewritten_plan)
        self.assertIsNotNone(rewritten_score)
        self.assertLess(rewritten_q.n, q.n)
        self.assertLess(rewritten_score, baseline_score)

        baseline_total = _scaled_to_complex(
            _evaluate_q3_free_execution_plan_scaled(baseline_plan)
        )
        rewritten_total = _scaled_to_complex(
            _evaluate_q3_free_execution_plan_scaled(
                rewritten_plan,
                output_scale_half_pow2=rewrite_scale,
            )
        )

        self.assertAlmostEqual(baseline_total.real, rewritten_total.real, places=12)
        self.assertAlmostEqual(baseline_total.imag, rewritten_total.imag, places=12)


if __name__ == "__main__":
    unittest.main()
