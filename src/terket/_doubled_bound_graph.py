"""General local-factor bounds over doubled difference variables."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
import math
from typing import TYPE_CHECKING, Iterator, Sequence

from ._factor_tables import _combine_factor_scaled, _sum_factor_tables_scaled
from ._q3free.factor_plans import _estimate_factor_table_dp_cost, _factor_scope_order
from .scaling import ScaledComplex, _make_scaled_complex, _mul_scaled_complex

if TYPE_CHECKING:
    from ._doubled_factors import DoubledFactorProblem

BoundFactor = tuple[tuple[int, ...], tuple[float, ...]]
_MAX_BOUND_SEARCH_FRONTIER = 1_000_000
_MAX_BOUND_SEARCH_EXPANDED = 5_000_000
_MAX_BOUND_PARTITION_WORK = 200_000_000
_MAX_BOUND_PARTITION_TABLE_ENTRIES = 8_388_608


@dataclass(slots=True)
class _DifferenceBoundSearch:
    factors: Sequence[BoundFactor]
    order: list[int]
    incidents: list[list[int]]
    assigned_local_masks: list[list[int]]
    partial_max_caches: list[dict[tuple[int, int], float]]

    def partial_max(self, factor_idx: int, depth: int, value_mask: int) -> float:
        scope, table = self.factors[factor_idx]
        local_assigned = self.assigned_local_masks[factor_idx][depth]
        local_value = sum(
            ((value_mask >> var) & 1) << pos for pos, var in enumerate(scope)
        )
        key = (local_assigned, local_value & local_assigned)
        cache = self.partial_max_caches[factor_idx]
        result = cache.get(key)
        if result is None:
            result = max(
                value
                for assignment, value in enumerate(table)
                if (assignment & local_assigned) == key[1]
            )
            cache[key] = result
        return result


def _factor_difference_scope(scope: Sequence[int], contour_variables: int) -> tuple[int, ...]:
    return tuple(sorted({
        var if var < contour_variables else var - contour_variables
        for var in scope
        if var < 2 * contour_variables
    }))


def _factor_difference_bound(
    scope: tuple[int, ...],
    table: Sequence[complex],
    contour_variables: int,
) -> BoundFactor:
    d_scope = _factor_difference_scope(scope, contour_variables)
    d_positions = {var: pos for pos, var in enumerate(d_scope)}
    bounds = [0.0] * (1 << len(d_scope))
    for assignment, value in enumerate(table):
        d_assignment = 0
        x_bits: dict[int, int] = {}
        y_bits: dict[int, int] = {}
        for pos, var in enumerate(scope):
            bit = (assignment >> pos) & 1
            if var < contour_variables:
                x_bits[var] = bit
            elif var < 2 * contour_variables:
                y_bits[var - contour_variables] = bit
        for var in d_scope:
            difference = x_bits.get(var, 0) ^ y_bits.get(var, 0)
            if var not in x_bits or var not in y_bits:
                difference = 0
            d_assignment |= difference << d_positions[var]
        magnitude = abs(value)
        if magnitude > bounds[d_assignment]:
            bounds[d_assignment] = magnitude

    # A contour variable present on only one side leaves its difference free.
    one_sided = [
        var
        for var in d_scope
        if not (
            var in scope
            and contour_variables + var in scope
        )
    ]
    for var in one_sided:
        pos = d_positions[var]
        for assignment in range(1 << len(d_scope)):
            alternate = assignment ^ (1 << pos)
            bound = max(bounds[assignment], bounds[alternate])
            bounds[assignment] = bound
            bounds[alternate] = bound
    return d_scope, tuple(bounds)


def compile_difference_bound_graph(
    problem: DoubledFactorProblem,
) -> tuple[float, tuple[BoundFactor, ...]]:
    """Compile arbitrary local factors into a positive factor graph over ``d``."""
    scalar = abs(problem.scalar)
    factors: dict[tuple[int, ...], list[float]] = {}
    for scope, table in problem.factors.items():
        d_scope, bound_table = _factor_difference_bound(
            scope,
            table,
            problem.contour_variables,
        )
        if not d_scope or all(value == bound_table[0] for value in bound_table):
            scalar *= bound_table[0]
            continue
        existing = factors.get(d_scope)
        if existing is None:
            factors[d_scope] = list(bound_table)
        else:
            factors[d_scope] = [
                left * right for left, right in zip(existing, bound_table)
            ]
    return scalar, tuple((scope, tuple(table)) for scope, table in sorted(factors.items()))


def _build_difference_bound_search(
    n_vars: int,
    factors: Sequence[BoundFactor],
) -> _DifferenceBoundSearch:
    incidence = [0] * n_vars
    for scope, table in factors:
        spread = max(table) - min(table)
        if spread:
            for var in scope:
                incidence[var] += 1
    order = sorted(range(n_vars), key=lambda var: (-incidence[var], var))
    incidents: list[list[int]] = [[] for _ in range(n_vars)]
    for factor_idx, (scope, _table) in enumerate(factors):
        for var in scope:
            incidents[var].append(factor_idx)
    rank = {var: depth for depth, var in enumerate(order)}
    assigned_local_masks = [
        [
            sum((rank[var] < depth) << pos for pos, var in enumerate(scope))
            for depth in range(n_vars + 1)
        ]
        for scope, _table in factors
    ]
    partial_max_caches: list[dict[tuple[int, int], float]] = [
        {(0, 0): max(table)} for _scope, table in factors
    ]
    return _DifferenceBoundSearch(
        factors,
        order,
        incidents,
        assigned_local_masks,
        partial_max_caches,
    )


def _top_correlated_difference_bound_masks(
    n_vars: int,
    factors: Sequence[BoundFactor],
    max_sectors: int,
) -> Iterator[tuple[int, float]]:
    search = _build_difference_bound_search(n_vars, factors)
    full_mask = (1 << n_vars) - 1
    initial_bound = math.prod(cache[(0, 0)] for cache in search.partial_max_caches)
    if initial_bound == 0.0:
        return
    heap: list[tuple[float, int, int, int]] = [(-initial_bound, 0, 0, 0)]
    yielded = 0
    expanded = 0
    while heap and yielded < max_sectors:
        negative_bound, negative_depth, assigned_mask, value_mask = heappop(heap)
        bound = -negative_bound
        depth = -negative_depth
        if assigned_mask == full_mask:
            yield value_mask, bound
            yielded += 1
            continue
        expanded += 1
        if expanded > _MAX_BOUND_SEARCH_EXPANDED or len(heap) > _MAX_BOUND_SEARCH_FRONTIER:
            raise RuntimeError(
                "General difference-bound branch-and-bound exceeded search limits. "
                "Use factor_bound for independent pair factors or lower max_sectors."
            )
        var = search.order[depth]
        next_assigned = assigned_mask | (1 << var)
        for bit in (0, 1):
            next_value = (value_mask & ~(1 << var)) | (bit << var)
            child_bound = bound
            for factor_idx in search.incidents[var]:
                old_max = search.partial_max(factor_idx, depth, value_mask)
                new_max = search.partial_max(factor_idx, depth + 1, next_value)
                child_bound *= new_max / old_max
            if child_bound:
                heappush(
                    heap,
                    (-child_bound, -(depth + 1), next_assigned, next_value),
                )


def top_difference_bound_masks(
    n_vars: int,
    factors: Sequence[BoundFactor],
    *,
    max_sectors: int,
) -> Iterator[tuple[int, float]]:
    """Yield exact top bound assignments using best-first branch-and-bound."""
    if not all(len(scope) <= 1 for scope, _table in factors):
        yield from _top_correlated_difference_bound_masks(n_vars, factors, max_sectors)
        return

    from ._doubled_strategy import _factor_bound_masks

    bounds = [[1.0, 1.0] for _ in range(n_vars)]
    for scope, table in factors:
        if scope:
            var = scope[0]
            bounds[var][0] *= table[0]
            bounds[var][1] *= table[1]
    frozen_bounds = tuple((zero, one) for zero, one in bounds)
    for mask in _factor_bound_masks(
        frozen_bounds,
        max_weight=n_vars,
        max_sectors=max_sectors,
    ):
        bound = math.prod(frozen_bounds[var][(mask >> var) & 1] for var in range(n_vars))
        yield mask, bound


def difference_bound_partition_sum(
    n_vars: int,
    scalar: float,
    factors: Sequence[BoundFactor],
    *,
    require_native: bool,
) -> ScaledComplex:
    """Return exact positive partition sum of compiled difference bounds."""
    scaled_factors: dict[tuple[int, ...], list[ScaledComplex]] = {}
    scaled_scalar = _make_scaled_complex(scalar)
    for scope, table in factors:
        scaled_scalar = _mul_scaled_complex(
            scaled_scalar,
            _combine_factor_scaled(
                scaled_factors,
                scope,
                [_make_scaled_complex(value) for value in table],
            ),
        )
    order, _width = _factor_scope_order(n_vars, tuple(scaled_factors))
    estimated_work, max_table_entries = _estimate_factor_table_dp_cost(
        n_vars,
        tuple(scaled_factors),
        order,
    )
    if (
        estimated_work > _MAX_BOUND_PARTITION_WORK
        or max_table_entries > _MAX_BOUND_PARTITION_TABLE_ENTRIES
    ):
        raise RuntimeError(
            "General difference-bound partition sum exceeds factor-table limits "
            f"(work {estimated_work}, table entries {max_table_entries}). "
            "Use factor_bound when bounds are independent, or simplify bound factors."
        )
    total, _max_scope = _sum_factor_tables_scaled(
        n_vars,
        scaled_factors,
        order,
        scalar=scaled_scalar,
        require_native=require_native,
    )
    return total


__all__ = [
    "compile_difference_bound_graph",
    "difference_bound_partition_sum",
    "top_difference_bound_masks",
]
