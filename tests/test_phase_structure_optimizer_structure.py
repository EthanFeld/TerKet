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
from terket.cubic_arithmetic import PhaseFunction

def _bruteforce_phase_sum(q: PhaseFunction) -> complex:
    total = 0j
    for mask in range(1 << q.n):
        bits = [(mask >> bit) & 1 for bit in range(q.n)]
        total += cmath.exp(2j * cmath.pi * float(q.evaluate(bits)))
    return total

class PhaseStructureStructureTests(unittest.TestCase):

    def test_optimizer_reduces_cubic_core_without_changing_sum(self):
        q = PhaseFunction(
            4,
            level=3,
            q1=[1, 0, 0, 0],
            q2={(0, 1): 1, (1, 2): 1},
            q3={(0, 1, 2): 1, (1, 2, 3): 1},
        )

        optimized_q, changed = engine._optimize_phase_function_structure(q)

        self.assertTrue(changed)
        self.assertLess(
            engine._phase_function_structure_score(optimized_q),
            engine._phase_function_structure_score(q),
        )
        self.assertAlmostEqual(abs(_bruteforce_phase_sum(q) - _bruteforce_phase_sum(optimized_q)), 0.0)

    def test_optimizer_reduces_q3_free_dense_core_risk_without_changing_sum(self):
        q = PhaseFunction(
            4,
            level=3,
            q1=[5, 5, 5, 7],
            q2={
                (0, 1): 1,
                (0, 2): 2,
                (0, 3): 2,
                (1, 2): 2,
                (1, 3): 3,
                (2, 3): 3,
            },
            q3={},
        )

        optimized_q, changed = engine._optimize_phase_function_structure(q)

        self.assertTrue(changed)
        self.assertFalse(optimized_q.q3)
        self.assertLess(
            engine._phase_function_structure_score(optimized_q),
            engine._phase_function_structure_score(q),
        )
        self.assertAlmostEqual(abs(_bruteforce_phase_sum(q) - _bruteforce_phase_sum(optimized_q)), 0.0)

    def test_local_optimizer_handles_large_phase_function_by_subregion(self):
        q = PhaseFunction(
            49,
            level=3,
            q1=[5, 5, 5, 7] + ([0] * 45),
            q2={
                (0, 1): 1,
                (0, 2): 2,
                (0, 3): 2,
                (1, 2): 2,
                (1, 3): 3,
                (2, 3): 3,
            },
            q3={},
        )

        optimized_q, changed = engine._optimize_phase_function_structure(q)

        self.assertTrue(changed)
        self.assertLess(
            engine._phase_function_structure_score(optimized_q),
            engine._phase_function_structure_score(q),
        )
        original_total, _ = engine._gauss_sum_q3_free_scaled(q, allow_tensor_contraction=False)
        optimized_total, _ = engine._gauss_sum_q3_free_scaled(optimized_q, allow_tensor_contraction=False)
        self.assertAlmostEqual(abs(original_total[0] - optimized_total[0]), 0.0, places=12)
        self.assertEqual(original_total[1], optimized_total[1])
