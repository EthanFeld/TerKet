"""Circuit-state conversion helpers for doubled factor problems."""

from __future__ import annotations

import cmath
from typing import Sequence

from ._doubled_factors import DoubledFactorProblem
from .cubic_arithmetic import PhaseFunction

_MAX_CONTOUR_FACTOR_SCOPE = 24


def _difference_contour_phase(q: PhaseFunction) -> PhaseFunction:
    k = q.n
    return PhaseFunction(
        2 * k,
        level=q.level,
        q1=list(q.q1) + [-coeff for coeff in q.q1],
        q2={
            **dict(q.q2),
            **{(k + left, k + right): -coeff for (left, right), coeff in q.q2.items()},
        },
        q3={
            **dict(q.q3),
            **{
                (k + left, k + middle, k + right): -coeff
                for (left, middle, right), coeff in q.q3.items()
            },
        },
    )


def _combine_plain_factor(
    factors: dict[tuple[int, ...], list[complex]],
    scope: tuple[int, ...],
    table: Sequence[complex],
) -> None:
    existing = factors.get(scope)
    if existing is None:
        factors[scope] = list(table)
    else:
        factors[scope] = [left * right for left, right in zip(existing, table)]


def _arbitrary_contour_factors(terms, contour_variables: int) -> dict[tuple[int, ...], list[complex]]:
    factors: dict[tuple[int, ...], list[complex]] = {}
    for term in terms:
        scope = tuple(idx for idx in range(contour_variables) if (int(term.row_mask) >> idx) & 1)
        if len(scope) > _MAX_CONTOUR_FACTOR_SCOPE:
            raise RuntimeError(
                f"Arbitrary-angle doubled factor has scope {len(scope)}, "
                f"above limit {_MAX_CONTOUR_FACTOR_SCOPE}."
            )
        phase = cmath.exp(1j * float(term.angle))
        offset = int(term.offset) & 1
        table = [
            phase if (assignment.bit_count() & 1) ^ offset else 1.0 + 0j
            for assignment in range(1 << len(scope))
        ]
        _combine_plain_factor(factors, scope, table)
        _combine_plain_factor(
            factors,
            tuple(contour_variables + idx for idx in scope),
            [value.conjugate() for value in table],
        )
    return factors


def _arbitrary_doubled_problem(q: PhaseFunction, terms) -> DoubledFactorProblem:
    return DoubledFactorProblem(
        contour_variables=q.n,
        phase=_difference_contour_phase(q),
        factors=_arbitrary_contour_factors(terms, q.n),
    )


__all__ = ["_arbitrary_doubled_problem"]
