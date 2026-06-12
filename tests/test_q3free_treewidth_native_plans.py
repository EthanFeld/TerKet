"""Tests for native q3-free treewidth plan construction."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket import engine

class Q3FreeTreewidthNativePlanTests(unittest.TestCase):

    def test_native_preplanned_treewidth_batch_matches_exact_rows(self):
        q = engine._phase_function_from_parts(
            6,
            level=3,
            q0=0,
            q1=[0] * 6,
            q2={
                (0, 1): 2,
                (1, 2): 2,
                (2, 3): 2,
                (3, 4): 2,
                (1, 4): 2,
                (4, 5): 2,
            },
            q3={},
        )
        order, _width = engine._min_fill_cubic_order(q)
        native_plan = engine._build_native_q3_free_treewidth_plan(
            n_vars=q.n,
            level=q.level,
            q2=q.q2,
            order=order,
        )
        self.assertIsNotNone(native_plan)

        q1_batch = np.asarray(
            [
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
            ],
            dtype=np.int64,
        )

        native_totals = engine._sum_q3_free_treewidth_dp_scaled_batch(
            n_vars=q.n,
            level=q.level,
            q1_batch=q1_batch,
            q2=q.q2,
            order=order,
            native_plan=native_plan,
        )
        exact_totals = []
        for row in q1_batch:
            row_q = engine._phase_function_from_parts(
                q.n,
                level=q.level,
                q0=0,
                q1=row.tolist(),
                q2=q.q2,
                q3={},
            )
            exact_totals.append(engine._sum_via_treewidth_dp_scaled(row_q, order)[0])

        native_complex = [engine._scaled_to_complex(total) for total in native_totals]
        exact_complex = [engine._scaled_to_complex(total) for total in exact_totals]
        self.assertEqual(native_complex, exact_complex)

    def test_native_preplanned_treewidth_array_entry_matches_list_entry(self):
        q2 = {
            (0, 1): 2,
            (1, 2): 2,
            (2, 3): 2,
            (0, 3): 2,
        }
        order = [0, 1, 2, 3]
        native_plan = engine._build_native_q3_free_treewidth_plan(
            n_vars=4,
            level=3,
            q2=q2,
            order=order,
        )
        self.assertIsNotNone(native_plan)

        q1_batch = np.asarray(
            [
                [0, 2, 4, 6],
                [6, 4, 2, 0],
            ],
            dtype=np.int64,
        )

        list_rows = engine._schur_native.sum_q3_free_treewidth_preplanned_batch_scaled(
            native_plan,
            q1_batch.tolist(),
        )
        array_rows = engine._schur_native.sum_q3_free_treewidth_preplanned_batch_scaled_array(
            native_plan,
            q1_batch,
        )

        self.assertEqual(list_rows, array_rows)

    def test_native_fixed_factor_treewidth_plan_matches_generic_factor_sum(self):
        q = engine._phase_function_from_parts(
            5,
            level=3,
            q0=0,
            q1=[1, 0, 2, 0, 0],
            q2={(0, 1): 2, (1, 2): 2, (2, 3): 2},
            q3={(1, 3, 4): 1},
        )
        order, _width = engine._min_fill_cubic_order(q)
        scalar, factors = engine._build_cached_phase3_treewidth_factor_plan_scaled(q)
        native_plan = engine._build_native_phase3_treewidth_plan(q=q, order=order)
        self.assertIsNotNone(native_plan)
        assert native_plan is not None

        native_total, native_width = engine._schur_native.sum_scaled_factor_treewidth_preplanned(native_plan)
        expected_total, expected_width = engine._sum_factor_tables_scaled(
            q.n,
            dict(factors),
            order,
            scalar=scalar,
        )

        self.assertEqual(engine._scaled_to_complex((complex(native_total[0]), int(native_total[1]))), engine._scaled_to_complex(expected_total))
        self.assertEqual(int(native_width), int(expected_width))

    def test_native_level3_treewidth_plan_matches_direct_kernel(self):
        q = engine._phase_function_from_parts(
            6,
            level=3,
            q0=0,
            q1=[1, 0, 2, 0, 3, 0],
            q2={(0, 1): 2, (1, 2): 2, (2, 3): 2, (3, 4): 2},
            q3={(1, 3, 5): 1, (0, 2, 4): 1},
        )
        order, _width = engine._min_fill_cubic_order(q)
        native_plan = engine._build_native_level3_phase3_treewidth_plan(q=q, order=order)
        self.assertIsNotNone(native_plan)
        assert native_plan is not None

        planned_total, planned_width = engine._schur_native.sum_level3_treewidth_preplanned(native_plan)
        direct_total, direct_width = engine._schur_native.sum_treewidth_dp_level3(
            q.n,
            q.q1,
            q.q2,
            q.q3,
            order,
        )
        self.assertEqual(complex(planned_total), complex(direct_total))
        self.assertEqual(int(planned_width), int(direct_width))
