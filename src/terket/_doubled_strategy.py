"""Difference-sector ordering strategies for doubled factor problems."""

from __future__ import annotations

from heapq import heappop, heappush
from itertools import islice
import math
from operator import index
from typing import TYPE_CHECKING, Iterator, Sequence

from ._doubled_core import _chunked_masks, _validate_options

if TYPE_CHECKING:
    from ._doubled_factors import DoubledFactorProblem


def _validate_factor_sum_options(
    problem: DoubledFactorProblem,
    max_difference_weight: int,
    sector_batch_size: int,
    max_sectors: int | None,
    difference_strategy: str,
    omitted_magnitude_tolerance: float | None,
) -> tuple[int, int | None, float | None]:
    k = problem.contour_variables
    max_weight = _validate_options(k, max_difference_weight, sector_batch_size)
    try:
        sector_limit = None if max_sectors is None else index(max_sectors)
    except TypeError as exc:
        raise TypeError("max_sectors must be an integer.") from exc
    if sector_limit is not None and sector_limit <= 0:
        raise ValueError("max_sectors must be positive.")
    if difference_strategy not in {"hamming", "factor_bound", "general_bound"}:
        raise ValueError(
            "difference_strategy must be 'hamming', 'factor_bound', or 'general_bound'."
        )
    if difference_strategy in {"factor_bound", "general_bound"}:
        if max_weight != k:
            raise ValueError("Bound strategies require the full difference-weight range.")
        if sector_limit is None:
            raise ValueError("Bound strategies require max_sectors.")
    if omitted_magnitude_tolerance is not None:
        omitted_magnitude_tolerance = float(omitted_magnitude_tolerance)
        if omitted_magnitude_tolerance < 0.0 or not math.isfinite(omitted_magnitude_tolerance):
            raise ValueError("omitted_magnitude_tolerance must be finite and nonnegative.")
        if difference_strategy != "general_bound":
            raise ValueError("omitted_magnitude_tolerance requires general_bound strategy.")
    return max_weight, sector_limit, omitted_magnitude_tolerance


def _pair_difference_bounds(problem: DoubledFactorProblem) -> tuple[tuple[float, float], ...]:
    """Return cheap per-bit sector bounds from direct ``(x_i, y_i)`` factors."""
    k = problem.contour_variables
    bounds: list[tuple[float, float]] = []
    for idx in range(k):
        table = problem.factors.get((idx, k + idx))
        if table is None:
            bounds.append((1.0, 1.0))
        else:
            bounds.append((
                max(abs(table[0]), abs(table[3])),
                max(abs(table[1]), abs(table[2])),
            ))
    return tuple(bounds)


def _factor_bound_masks(
    bounds: Sequence[tuple[float, float]],
    *,
    max_weight: int,
    max_sectors: int,
) -> Iterator[int]:
    """Yield masks by descending independent pair-factor magnitude bound."""
    base_mask = 0
    deviations: list[tuple[float, int]] = []
    for bit, (zero_bound, one_bound) in enumerate(bounds):
        if zero_bound == 0.0 and one_bound == 0.0:
            return
        if one_bound > zero_bound:
            base_mask |= 1 << bit
            best, alternate = one_bound, zero_bound
        else:
            best, alternate = zero_bound, one_bound
        if alternate > 0.0:
            deviations.append((math.log(best / alternate), bit))
    deviations.sort()

    yielded = 0
    if base_mask.bit_count() <= max_weight:
        yield base_mask
        yielded += 1
        if yielded == max_sectors:
            return
    if not deviations:
        return

    first_penalty, first_bit = deviations[0]
    heap: list[tuple[float, int, int]] = [(first_penalty, 0, 1 << first_bit)]
    while heap and yielded < max_sectors:
        penalty, position, deviation_mask = heappop(heap)
        mask = base_mask ^ deviation_mask
        if mask.bit_count() <= max_weight:
            yield mask
            yielded += 1
        next_position = position + 1
        if next_position >= len(deviations):
            continue
        next_penalty, next_bit = deviations[next_position]
        heappush(
            heap,
            (penalty + next_penalty, next_position, deviation_mask | (1 << next_bit)),
        )
        position_penalty, position_bit = deviations[position]
        heappush(
            heap,
            (
                penalty - position_penalty + next_penalty,
                next_position,
                (deviation_mask ^ (1 << position_bit)) | (1 << next_bit),
            ),
        )


def _chunk_masks(masks: Iterator[int], chunk_size: int) -> Iterator[list[int]]:
    chunk: list[int] = []
    for mask in masks:
        chunk.append(mask)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _build_mask_chunks(
    problem: DoubledFactorProblem,
    *,
    max_weight: int,
    sector_limit: int | None,
    sector_batch_size: int,
    difference_strategy: str,
) -> tuple[Iterator[list[int]], int | None, int, bool, bool]:
    k = problem.contour_variables
    pair_bounds = _pair_difference_bounds(problem)
    informative_bounds = any(left != right for left, right in pair_bounds)
    viable_pair_sectors = math.prod(
        int(zero_bound > 0.0) + int(one_bound > 0.0)
        for zero_bound, one_bound in pair_bounds
    )
    hard_pair_constraints = any(
        zero_bound == 0.0 or one_bound == 0.0
        for zero_bound, one_bound in pair_bounds
    )
    auto_hard_constraint_order = (
        difference_strategy == "hamming"
        and sector_limit is None
        and max_weight == k
        and hard_pair_constraints
    )
    if difference_strategy == "factor_bound":
        if max_weight != k:
            raise ValueError("factor_bound strategy requires the full difference-weight range.")
        if not informative_bounds:
            raise ValueError("factor_bound strategy requires informative direct contour-pair factors.")
        if sector_limit is None:
            raise ValueError("factor_bound strategy requires max_sectors.")
    if (difference_strategy == "factor_bound" and informative_bounds) or auto_hard_constraint_order:
        if sector_limit is None:
            sector_limit = viable_pair_sectors
        chunks = _chunk_masks(
            _factor_bound_masks(pair_bounds, max_weight=max_weight, max_sectors=sector_limit),
            sector_batch_size,
        )
        return chunks, sector_limit, viable_pair_sectors, informative_bounds, auto_hard_constraint_order

    masks = (
        mask
        for chunk in _chunked_masks(k, max_weight, sector_batch_size, include_zero=True)
        for mask in chunk
    )
    if sector_limit is not None:
        masks = islice(masks, sector_limit)
    return (
        _chunk_masks(masks, sector_batch_size),
        sector_limit,
        viable_pair_sectors,
        informative_bounds,
        auto_hard_constraint_order,
    )


__all__ = [
    "_build_mask_chunks",
    "_factor_bound_masks",
    "_pair_difference_bounds",
    "_validate_factor_sum_options",
]
