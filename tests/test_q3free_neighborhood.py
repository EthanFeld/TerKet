"""Tests for bounded-neighborhood-diversity q3-free summation."""

from __future__ import annotations

from fractions import Fraction
import random

import numpy as np

from terket.cubic_arithmetic import PhaseFunction
from terket import engine


def _complete_q(n: int, q1) -> PhaseFunction:
    return PhaseFunction(
        n,
        level=3,
        q1=q1,
        q2={(left, right): 2 for left in range(n) for right in range(left + 1, n)},
        q3={},
    )


def _blown_up_path_q(class_count: int, class_size: int, q1) -> PhaseFunction:
    q2 = {}
    for class_idx in range(class_count - 1):
        left_class = range(class_idx * class_size, (class_idx + 1) * class_size)
        right_class = range((class_idx + 1) * class_size, (class_idx + 2) * class_size)
        q2.update({(left, right): 2 for left in left_class for right in right_class})
    return PhaseFunction(class_count * class_size, level=3, q1=q1, q2=q2, q3={})


def test_neighborhood_sum_matches_bruteforce_with_arbitrary_unary_roots():
    rng = random.Random(1847)
    for n in range(2, 9):
        q = _complete_q(n, [rng.randrange(8) for _ in range(n)])
        actual = engine._scaled_to_complex(engine._sum_q3_free_via_neighborhood_scaled(q))
        expected = engine._bruteforce_q3_free_sum(q)
        assert abs(actual - expected) < 1e-9


def test_direct_neighborhood_sum_preserves_constant_phase():
    q = _complete_q(7, [1, 2, 3, 4, 5, 6, 7])
    q.q0 = Fraction(3, 16)
    actual = engine._scaled_to_complex(engine._sum_q3_free_via_neighborhood_scaled(q))
    assert abs(actual - engine._bruteforce_q3_free_sum(q)) < 1e-9


def test_neighborhood_sum_matches_bruteforce_at_higher_dyadic_precision():
    rng = random.Random(772)
    for level in (4, 5, 6):
        modulus = 1 << level
        q = PhaseFunction(
            7,
            level=level,
            q1=[rng.randrange(modulus) for _ in range(7)],
            q2={(left, right): modulus // 4 for left in range(7) for right in range(left + 1, 7)},
            q3={},
        )
        actual = engine._scaled_to_complex(engine._sum_q3_free_via_neighborhood_scaled(q))
        assert abs(actual - engine._bruteforce_q3_free_sum(q)) < 1e-9


def test_neighborhood_sum_matches_bruteforce_on_mixed_twin_classes():
    # Clique {0,1,2}, independent twins {3,4,5}, complete join between classes.
    q2 = {(left, right): 2 for left in range(3) for right in range(left + 1, 3)}
    q2.update({(left, right): 2 for left in range(3) for right in range(3, 6)})
    q = PhaseFunction(6, level=3, q1=[1, 2, 3, 4, 5, 6], q2=q2, q3={})
    plan = engine._build_q3_free_neighborhood_plan(q)
    assert plan is not None
    assert len(plan.classes) == 2
    actual = engine._scaled_to_complex(engine._evaluate_q3_free_neighborhood_plan_scaled(plan, q.q1, level=3))
    assert abs(actual - engine._bruteforce_q3_free_sum(q)) < 1e-9


def test_planner_routes_complete_t_phase_kernel_to_neighborhood_backend():
    q = _complete_q(128, [1] * 128)
    plan = engine._build_q3_free_execution_plan(q=q, allow_tensor_contraction=False)
    assert [component.backend for component in plan.components] == ["neighborhood"]
    assert len(plan.components[0].neighborhood_plan.classes) == 1
    total = engine._evaluate_q3_free_execution_plan_scaled(plan)
    assert total[0] != 0j


def test_neighborhood_plan_supports_batch_and_restriction():
    q = _complete_q(8, [1] * 8)
    execution_plan = engine._build_q3_free_execution_plan(q=q, allow_tensor_contraction=False)
    component = execution_plan.components[0]
    rows = np.asarray([[1] * 8, [3] * 8], dtype=np.int64)
    totals = engine._evaluate_q3_free_component_plan_scaled_batch(component, rows, level=3)
    for row, total in zip(rows, totals):
        row_q = _complete_q(8, row.tolist())
        assert abs(engine._scaled_to_complex(total) - engine._bruteforce_q3_free_sum(row_q)) < 1e-9

    restricted = engine._restrict_q3_free_component_plan(component, [0, 1, 2, 3])
    assert restricted.backend == "neighborhood"
    restricted_total = engine._evaluate_q3_free_component_plan_scaled(restricted, [1, 2, 3, 4], level=3)
    assert abs(
        engine._scaled_to_complex(restricted_total)
        - engine._bruteforce_q3_free_sum(_complete_q(4, [1, 2, 3, 4]))
    ) < 1e-9


def test_neighborhood_backend_does_not_hijack_long_path():
    n = 20
    q = PhaseFunction(
        n,
        level=3,
        q1=[1] * n,
        q2={(idx, idx + 1): 2 for idx in range(n - 1)},
        q3={},
    )
    assert engine._build_q3_free_neighborhood_plan(q) is None
    plan = engine._build_q3_free_execution_plan(q=q, allow_tensor_contraction=False)
    assert [component.backend for component in plan.components] == ["constant"]


def test_non_half_phase_q2_rejected():
    q = PhaseFunction(6, level=3, q1=[0] * 6, q2={(0, 1): 1}, q3={})
    assert engine._build_q3_free_neighborhood_plan(q) is None


def test_neighborhood_treewidth_matches_bruteforce_on_blown_up_path():
    q = _blown_up_path_q(4, 2, [1, 2, 3, 4, 5, 6, 7, 0])
    plan = engine._build_q3_free_neighborhood_treewidth_plan(q)
    assert plan is not None
    assert len(plan.classes) == 4
    assert plan.width <= 2
    actual = engine._scaled_to_complex(
        engine._evaluate_q3_free_neighborhood_treewidth_plan_scaled(plan, q.q1, level=3)
    )
    assert abs(actual - engine._bruteforce_q3_free_sum(q)) < 1e-9


def test_planner_routes_large_blown_up_path_to_neighborhood_treewidth():
    q = _blown_up_path_q(14, 3, [1] * 42)
    assert engine._build_q3_free_neighborhood_plan(q) is None
    plan = engine._build_q3_free_execution_plan(q=q, allow_tensor_contraction=False)
    assert [component.backend for component in plan.components] == ["neighborhood_treewidth"]
    assert plan.components[0].neighborhood_treewidth_plan.width <= 2
    assert engine._evaluate_q3_free_execution_plan_scaled(plan)[0] != 0j


def test_neighborhood_treewidth_plan_supports_batch_and_restriction():
    q = _blown_up_path_q(4, 2, [1] * 8)
    execution_plan = engine._build_q3_free_execution_plan(q=q, allow_tensor_contraction=False)
    component = execution_plan.components[0]
    assert component.backend == "neighborhood_treewidth"
    rows = np.asarray([[1] * 8, [3] * 8], dtype=np.int64)
    totals = engine._evaluate_q3_free_component_plan_scaled_batch(component, rows, level=3)
    for row, total in zip(rows, totals):
        row_q = _blown_up_path_q(4, 2, row.tolist())
        assert abs(engine._scaled_to_complex(total) - engine._bruteforce_q3_free_sum(row_q)) < 1e-9

    restricted = engine._restrict_q3_free_component_plan(component, [0, 1, 2, 3, 4, 5])
    restricted_total = engine._evaluate_q3_free_component_plan_scaled(restricted, [1] * 6, level=3)
    expected = engine._bruteforce_q3_free_sum(_blown_up_path_q(3, 2, [1] * 6))
    assert abs(engine._scaled_to_complex(restricted_total) - expected) < 1e-9
