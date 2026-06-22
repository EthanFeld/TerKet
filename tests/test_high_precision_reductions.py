"""Exact high-precision constraint-reduction tests."""

from __future__ import annotations

import cmath

from terket._reduction_high_precision import _apply_safe_high_precision_eliminations
from terket.cubic_arithmetic import PhaseFunction


def _bruteforce_phase_sum(q: PhaseFunction) -> complex:
    return sum(
        cmath.exp(
            2j
            * cmath.pi
            * float(q.evaluate([(mask >> bit) & 1 for bit in range(q.n)]))
        )
        for mask in range(1 << q.n)
    )


def test_batches_only_cubic_safe_low_arity_constraints() -> None:
    q = PhaseFunction(
        6,
        level=5,
        q1=[0, 3, 5, 16, 7, 11],
        q2={(0, 1): 8, (0, 2): 8, (1, 5): 3, (3, 4): 8, (4, 5): 6},
        q3={(1, 2, 5): 3, (2, 4, 5): 5},
    )

    reduced_q, half_pow2, info = _apply_safe_high_precision_eliminations(q)

    assert reduced_q is not None
    assert reduced_q.n == 2
    assert half_pow2 == 4
    assert info == {"quad": 0, "constraint": 2}
    assert abs(
        _bruteforce_phase_sum(q)
        - (2 ** (half_pow2 / 2)) * _bruteforce_phase_sum(reduced_q)
    ) < 1e-10


def test_rejects_nonparity_even_coupling() -> None:
    q = PhaseFunction(
        3,
        level=5,
        q1=[0, 1, 3],
        q2={(0, 1): 8, (0, 2): 2},
        q3={},
    )

    reduced_q, half_pow2, info = _apply_safe_high_precision_eliminations(q)

    assert reduced_q is q
    assert half_pow2 == 0
    assert info == {"quad": 0, "constraint": 0}
