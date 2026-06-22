"""General difference-bound sector planning and certificates."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable

from ._doubled_bound_graph import (
    compile_difference_bound_graph,
    difference_bound_partition_sum,
    top_difference_bound_masks,
)
from .scaling import (
    ScaledAmplitude,
    ScaledComplex,
    _add_scaled_complex,
    _make_scaled_complex,
    _scale_scaled_complex,
    _scaled_log2_abs,
)

if TYPE_CHECKING:
    from ._doubled_factors import DoubledFactorProblem


def prepare_general_bound_sectors(
    problem: DoubledFactorProblem,
    *,
    max_sectors: int,
    sector_batch_size: int,
    omitted_magnitude_tolerance: float | None,
    require_native: bool,
) -> tuple[Iterable[list[int]], ScaledAmplitude]:
    """Rank sectors and return their rigorous omitted-magnitude certificate."""
    k = problem.contour_variables
    bound_scalar, bound_factors = compile_difference_bound_graph(problem)
    if bound_scalar == 0.0:
        return (), ScaledAmplitude(0j)
    bound_total = difference_bound_partition_sum(
        k,
        bound_scalar,
        bound_factors,
        require_native=require_native,
    )
    ranked: list[int] = []
    kept_bound: ScaledComplex = (0j, 0)
    tolerance_log2 = (
        -math.inf
        if omitted_magnitude_tolerance == 0.0
        else math.log2(omitted_magnitude_tolerance)
        if omitted_magnitude_tolerance is not None
        else None
    )
    residual = bound_total
    for mask, bound in top_difference_bound_masks(k, bound_factors, max_sectors=max_sectors):
        ranked.append(mask)
        kept_bound = _add_scaled_complex(
            kept_bound,
            _make_scaled_complex(bound_scalar * bound),
        )
        residual = _add_scaled_complex(bound_total, (-kept_bound[0], kept_bound[1]))
        scaled_residual = _scale_scaled_complex(
            residual,
            2 * (k + problem.auxiliary_variables),
        )
        if tolerance_log2 is not None and _scaled_log2_abs(scaled_residual) <= tolerance_log2:
            break
    if residual[0].real < 0.0 and abs(residual[0].real) < 1e-12:
        residual = (0j, 0)
    mask_chunks = (
        ranked[offset : offset + sector_batch_size]
        for offset in range(0, len(ranked), sector_batch_size)
    )
    omitted_bound = ScaledAmplitude.from_tuple(
        _scale_scaled_complex(residual, 2 * (k + problem.auxiliary_variables))
    )
    return mask_chunks, omitted_bound


__all__ = ["prepare_general_bound_sectors"]
