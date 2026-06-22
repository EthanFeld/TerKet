"""Direct exact routes for doubled factor problems."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

from ._doubled_core import DoubledSumResult
from ._factor_tables import _combine_factor_scaled, _sum_factor_tables_scaled
from ._phase3.factors import _build_cubic_factors_scaled
from ._q3free.factor_plans import _estimate_factor_table_dp_cost, _factor_scope_order
from .native import _schur_native
from .scaling import ScaledAmplitude, ScaledComplex, _ZERO_SCALED, _mul_scaled_complex

if TYPE_CHECKING:
    from ._doubled_factors import DoubledFactorProblem

_DIRECT_EXACT_MAX_WORK = 200_000_000
_DIRECT_EXACT_MAX_TABLE_ENTRIES = 8_388_608
_DIRECT_EXACT_MAX_SECTOR_FALLBACK = 1 << 20


def _build_original_factors(
    problem: DoubledFactorProblem,
    scaled_problem_factors: Mapping[tuple[int, ...], Sequence[ScaledComplex]],
    problem_scalar: ScaledComplex,
) -> tuple[ScaledComplex, dict[tuple[int, ...], list[ScaledComplex]]]:
    factors: dict[tuple[int, ...], list[ScaledComplex]] = {}
    scalar = problem_scalar
    for scope, table in scaled_problem_factors.items():
        if all(value[0] == 0j for value in table):
            return _ZERO_SCALED, {}
        scalar = _mul_scaled_complex(
            scalar,
            _combine_factor_scaled(factors, scope, list(table)),
        )
    if problem.phase is not None:
        phase_scalar, phase_factors = _build_cubic_factors_scaled(problem.phase)
        scalar = _mul_scaled_complex(scalar, phase_scalar)
        for scope, table in phase_factors.items():
            scalar = _mul_scaled_complex(
                scalar,
                _combine_factor_scaled(factors, scope, table),
            )
    return scalar, factors


def _try_direct_exact(
    problem: DoubledFactorProblem,
    *,
    max_weight: int,
    sector_limit: int | None,
    require_native: bool,
    scaled_problem_factors: Mapping[tuple[int, ...], Sequence[ScaledComplex]],
    problem_scalar: ScaledComplex,
) -> DoubledSumResult | None:
    k = problem.contour_variables
    if max_weight != k or sector_limit is not None:
        return None
    scalar, factors = _build_original_factors(problem, scaled_problem_factors, problem_scalar)
    if scalar[0] == 0j:
        return DoubledSumResult(
            estimate=ScaledAmplitude(0j),
            path_variables=k,
            max_difference_weight=max_weight,
            sectors_evaluated=0,
            sectors_total=1 << k,
            exact=True,
            max_reducer_remaining=0,
            phase3_backends=(),
        )
    order, estimated_width = _factor_scope_order(
        2 * k + problem.auxiliary_variables,
        tuple(factors),
    )
    estimated_work, max_table_entries = _estimate_factor_table_dp_cost(
        2 * k + problem.auxiliary_variables,
        tuple(factors),
        order,
    )
    if (
        estimated_work > _DIRECT_EXACT_MAX_WORK
        or max_table_entries > _DIRECT_EXACT_MAX_TABLE_ENTRIES
    ):
        if 1 << k > _DIRECT_EXACT_MAX_SECTOR_FALLBACK:
            raise RuntimeError(
                "Exact doubled factor sum exceeds direct factor-table limits "
                f"(width {estimated_width}, work {estimated_work}, table entries "
                f"{max_table_entries}) and sector enumeration has {1 << k} sectors. "
                "Use max_sectors with difference_strategy='factor_bound' when "
                "direct contour-pair factors provide useful bounds."
            )
        return None
    total, max_scope = _sum_factor_tables_scaled(
        2 * k + problem.auxiliary_variables,
        factors,
        order,
        scalar=scalar,
        require_native=require_native,
    )
    backend = "factor_tables_native" if require_native and _schur_native is not None else "factor_tables"
    return DoubledSumResult(
        estimate=ScaledAmplitude.from_tuple(total),
        path_variables=k,
        max_difference_weight=max_weight,
        sectors_evaluated=1,
        sectors_total=1 << k,
        exact=True,
        max_reducer_remaining=max_scope,
        phase3_backends=(backend,),
    )


__all__ = ["_try_direct_exact"]
