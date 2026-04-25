from __future__ import annotations

import sys
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.targeted.rcs.amplitude_post_elimination_tensor_rcs import run_case


class PostEliminationTensorRcsTests(unittest.TestCase):
    def test_smoke_case_plans_post_elimination_tensor_residual(self):
        row = run_case(
            "smoke",
            2,
            seed=7,
            run_contraction_up_to_vars=0,
            include_full_amplitude=False,
        )
        self.assertEqual(row.case, "smoke")
        self.assertIn(row.status, {"planned", "schur_solved"})
        self.assertGreaterEqual(row.component_var_count, row.residual_var_count)
        self.assertGreaterEqual(row.residual_factor_count, 0)
        if row.status == "planned":
            self.assertGreater(row.residual_var_count, 0)
            self.assertIsNotNone(row.greedy_plan_width)
            self.assertIsNotNone(row.greedy_plan_log2_max_tensor_size)
            self.assertIsNotNone(row.greedy_plan_log10_flops)


if __name__ == "__main__":
    unittest.main()
