"""Tests for q3-free one-shot slicing region construction behavior."""

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

class Q3FreeOneShotSlicingRegionTests(unittest.TestCase):

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
