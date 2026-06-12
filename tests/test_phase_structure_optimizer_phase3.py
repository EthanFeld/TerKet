"""Tests for phase-structure optimizer behavior on Phase-3 paths."""

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

class PhaseStructurePhase3Tests(unittest.TestCase):

    def test_cubic_optimizer_rejects_structural_rewrite_with_worse_phase3_runtime(self):
        q = PhaseFunction(3, level=3, q1=[1, 0, 0], q2={(0, 1): 1}, q3={(0, 1, 2): 1})
        structurally_better = PhaseFunction(3, level=3, q1=[0, 1, 0], q2={(0, 2): 1}, q3={(0, 1, 2): 1})
        seen = []

        with unittest.mock.patch.object(
            engine,
            "_optimize_phase_function_structure",
            return_value=(structurally_better, True),
        ), unittest.mock.patch.object(
            engine,
            "_phase3_execution_plan_runtime_score",
            side_effect=[(0, 10, 20, 1, 0), (1, 20, 22, 1, 0)],
        ), unittest.mock.patch.object(
            engine,
            "_apply_exact_eliminations",
            return_value=(q, 0, {"quad": 0, "constraint": 0}, ()),
        ), unittest.mock.patch.object(
            engine,
            "_sum_irreducible_cubic_core",
            side_effect=lambda q_arg, **kwargs: (
                seen.append(q_arg) or ((1.0 + 0.0j, 0), {"quad": 0, "constraint": 0, "branched": 0, "remaining": 0})
            ),
        ):
            engine._reduce_and_sum_scaled(
                q,
                context=engine._ReductionContext(
                    preserve_scale=False,
                    allow_tensor_contraction=False,
                    extended_reductions="auto",
                ),
            )

        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], q)

    def test_cubic_optimizer_skips_when_peeled_treewidth_baseline_is_already_good(self):
        q = PhaseFunction(3, level=3, q1=[1, 0, 0], q2={(0, 1): 1}, q3={(0, 1, 2): 1})

        with unittest.mock.patch.object(
            engine,
            "_optimize_phase_function_structure",
        ) as optimize, unittest.mock.patch.object(
            engine,
            "_phase3_execution_plan_runtime_score",
            return_value=(0, 10, 22, 1, 0),
        ), unittest.mock.patch.object(
            engine,
            "_apply_exact_eliminations",
            return_value=(q, 0, {"quad": 0, "constraint": 0}, ()),
        ), unittest.mock.patch.object(
            engine,
            "_sum_irreducible_cubic_core",
            return_value=((1.0 + 0.0j, 0), {"quad": 0, "constraint": 0, "branched": 0, "remaining": 0}),
        ):
            engine._reduce_and_sum_scaled(
                q,
                context=engine._ReductionContext(
                    preserve_scale=False,
                    allow_tensor_contraction=False,
                    extended_reductions="auto",
                ),
            )

        optimize.assert_not_called()

    def test_large_sparse_cubic_kernel_can_skip_exact_eliminations_for_direct_treewidth(self):
        q = PhaseFunction(
            300,
            level=3,
            q1=[0] * 300,
            q2={(3, 4): 1, (4, 5): 1},
            q3={(0, 1, 2): 1},
        )

        with unittest.mock.patch.object(
            engine,
            "_phase3_plan",
            return_value=([0], list(range(300)), 16, 1, "treewidth_dp"),
        ), unittest.mock.patch.object(
            engine,
            "_estimate_treewidth_dp_work",
            return_value=2_375_618,
        ), unittest.mock.patch.object(
            engine,
            "_apply_exact_eliminations",
            side_effect=AssertionError("direct treewidth escape should skip exact eliminations"),
        ), unittest.mock.patch.object(
            engine,
            "_sum_irreducible_cubic_core",
            return_value=((1.0 + 0.0j, 0), {
                "quad": 0,
                "constraint": 0,
                "branched": 0,
                "remaining": 16,
                "structural_obstruction": 1,
                "gauss_obstruction": 1,
                "cost_r": 16,
                "phase_states": 0,
                "phase_splits": 0,
                "phase3_backend": "treewidth_dp",
            }),
        ) as sum_core:
            total, info = engine._reduce_and_sum_scaled(
                q,
                context=engine._ReductionContext(
                    preserve_scale=False,
                    allow_tensor_contraction=False,
                    extended_reductions="auto",
                ),
            )

        self.assertEqual(total, (1.0 + 0.0j, 0))
        self.assertEqual(info["phase3_backend"], "treewidth_dp")
        sum_core.assert_called_once()

    def test_two_partner_constraint_elimination_preserves_sum_for_target_zero(self):
        q = PhaseFunction(
            4,
            level=3,
            q1=[0, 3, 0, 0],
            q2={
                (0, 1): 2,
                (0, 2): 2,
                (1, 3): 1,
            },
            q3={
                (1, 2, 3): 1,
            },
        )

        reduced_q, half_pow2 = engine._elim_two_partner_constraint(q, 0, 2, 1, 0)

        self.assertIsNotNone(reduced_q)
        self.assertEqual(half_pow2, 2)
        self.assertEqual(reduced_q.n, 2)
        self.assertFalse(reduced_q.q3)
        self.assertAlmostEqual(
            abs(_bruteforce_phase_sum(q) - (2 ** (half_pow2 / 2)) * _bruteforce_phase_sum(reduced_q)),
            0.0,
            places=12,
        )

    def test_two_partner_constraint_elimination_preserves_sum_for_target_one(self):
        q = PhaseFunction(
            5,
            level=3,
            q1=[4, 3, 0, 0, 0],
            q2={
                (0, 1): 2,
                (0, 2): 2,
                (1, 3): 1,
            },
            q3={
                (1, 3, 4): 1,
            },
        )

        reduced_q, half_pow2 = engine._elim_two_partner_constraint(q, 0, 2, 1, 1)

        self.assertIsNotNone(reduced_q)
        self.assertEqual(half_pow2, 2)
        self.assertEqual(reduced_q.n, 3)
        self.assertTrue(reduced_q.q3)
        self.assertAlmostEqual(
            abs(_bruteforce_phase_sum(q) - (2 ** (half_pow2 / 2)) * _bruteforce_phase_sum(reduced_q)),
            0.0,
            places=12,
        )

    def test_exact_elimination_uses_direct_two_partner_constraint_path(self):
        q = PhaseFunction(
            4,
            level=3,
            q1=[0, 3, 0, 0],
            q2={
                (0, 1): 2,
                (0, 2): 2,
                (1, 3): 1,
            },
            q3={
                (1, 2, 3): 1,
            },
        )

        with unittest.mock.patch.object(
            engine,
            "_aff_compose_cached",
            side_effect=AssertionError("two-partner constraints should not route through affine compose"),
        ):
            reduced_q, half_pow2, info, blocked = engine._apply_exact_eliminations(q)

        self.assertIsNotNone(reduced_q)
        self.assertEqual(half_pow2, 2)
        self.assertEqual(info["constraint"], 1)
        self.assertFalse(blocked)

    def test_phase3_treewidth_keeps_scaled_totals_without_overflowing(self):
        q = PhaseFunction(
            3,
            level=3,
            q1=[0, 0, 0],
            q2={},
            q3={(0, 1, 2): 1},
        )

        with unittest.mock.patch.object(
            engine,
            "_sum_via_treewidth_dp_peeled",
            side_effect=AssertionError("treewidth phase3 should stay on scaled totals"),
        ), unittest.mock.patch.object(
            engine,
            "_sum_via_treewidth_dp_peeled_scaled",
            return_value=((1.0 + 0.0j, 4000), 7),
        ):
            total, info = engine._sum_irreducible_cubic_core(
                q,
                context=engine._ReductionContext(preserve_scale=False, allow_tensor_contraction=False),
                cover=[0],
                order=[0, 1, 2],
                width=7,
                structural_obstruction=1,
                backend="treewidth_dp_peeled",
                allow_tensor_contraction=False,
            )

        self.assertEqual(total, (1.0 + 0.0j, 4000))
        self.assertEqual(info["remaining"], 7)
        self.assertEqual(info["phase3_backend"], "treewidth_dp_peeled")

    def test_phase3_backend_can_choose_treewidth_cutset_for_high_width_core(self):
        q = PhaseFunction(
            6,
            level=3,
            q1=[0] * 6,
            q2={},
            q3={
                (0, 1, 2): 1,
                (0, 2, 3): 1,
                (0, 3, 4): 1,
                (0, 4, 5): 1,
                (1, 3, 5): 1,
            },
        )

        with unittest.mock.patch.object(
            engine,
            "_find_q3_treewidth_cutset",
            return_value=((0, 1), [0, 1, 2, 3], 10, 50),
        ):
            backend, score, separator = engine._choose_phase3_backend(
                q,
                cover=[0, 1, 2, 3, 4],
                order=list(range(6)),
                width=30,
                structural_obstruction=5,
                allow_tensor_contraction=False,
                fully_peeled=False,
            )

        self.assertEqual(backend, "q3_treewidth_cutset")
        self.assertIsNone(separator)
        self.assertLess(score[1], engine._estimate_q3_cover_work(q, 5))

    def test_phase3_treewidth_cutset_rejects_bad_assignment_width(self):
        q = PhaseFunction(
            4,
            level=3,
            q1=[0] * 4,
            q2={},
            q3={(0, 1, 2): 1},
        )

        with unittest.mock.patch.object(
            engine,
            "_phase3_residual_after_cutset",
            return_value=(q, tuple(range(q.n))),
        ), unittest.mock.patch.object(
            engine,
            "_phase3_treewidth_cutset_candidates",
            side_effect=[(0,), ()],
        ), unittest.mock.patch.object(
            engine,
            "_min_fill_cubic_order",
            return_value=(list(range(q.n)), 1),
        ), unittest.mock.patch.object(
            engine,
            "_finalize_phase3_treewidth_order",
            return_value=(list(range(q.n)), 1),
        ), unittest.mock.patch.object(
            engine,
            "_estimate_treewidth_dp_work",
            return_value=1,
        ), unittest.mock.patch.object(
            engine,
            "_phase3_cutset_worst_residual",
            return_value=(list(range(q.n)), 19, 1),
        ):
            plan = engine._find_q3_treewidth_cutset(
                q,
                order=list(range(q.n)),
                width=30,
                fully_peeled=False,
            )

        self.assertIsNone(plan)

    def test_phase3_treewidth_cutset_preserves_exact_sum(self):
        q = PhaseFunction(
            5,
            level=3,
            q1=[1, 2, 3, 0, 1],
            q2={(0, 1): 1, (2, 3): 1},
            q3={(0, 1, 2): 1, (0, 3, 4): 1},
        )

        total, info = engine._sum_via_q3_treewidth_cutset(
            q,
            (0,),
            context=engine._ReductionContext(preserve_scale=True, allow_tensor_contraction=False),
            structural_obstruction=1,
        )

        actual = engine._scaled_to_complex(total)
        expected = _bruteforce_phase_sum(q)
        self.assertAlmostEqual(abs(actual - expected), 0.0, places=12)
        self.assertEqual(info["phase3_backend"], "q3_treewidth_cutset")
        self.assertGreaterEqual(info["branched"], 1)
