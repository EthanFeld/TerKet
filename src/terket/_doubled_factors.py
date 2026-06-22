"""General factor-table problems on doubled contours."""

from __future__ import annotations

from dataclasses import dataclass
import math
from operator import index
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

from ._doubled_bound_runtime import prepare_general_bound_sectors
from ._doubled_core import DoubledSumResult, _validate_options
from ._doubled_direct import _try_direct_exact
from ._doubled_strategy import _build_mask_chunks, _validate_factor_sum_options
from ._engine_runtime_state import _aff_compose_cached
from ._factor_tables import _combine_factor_scaled, _sum_factor_tables_scaled
from ._phase3.factors import _build_cubic_factors_scaled
from ._q3free.factor_plans import _factor_scope_order
from ._reduction_support import _ReductionContext
from .cubic_arithmetic import PhaseFunction
from .native import _schur_native
from .scaling import (
    ScaledAmplitude,
    ScaledComplex,
    _ZERO_SCALED,
    _add_scaled_complex,
    _make_scaled_complex,
    _mul_scaled_complex,
)


@dataclass(frozen=True, slots=True)
class DoubledFactorProblem:
    """Coupled doubled-contour sum with arbitrary local complex factors.

    Variables are ordered as ``(x..., y..., auxiliary...)``. Factor-table bit
    positions follow their sorted scope positions.
    """

    contour_variables: int
    auxiliary_variables: int = 0
    phase: PhaseFunction | None = None
    factors: Mapping[tuple[int, ...], Sequence[complex]] | None = None
    scalar: complex = 1.0 + 0j

    def __post_init__(self) -> None:
        try:
            contour_variables = index(self.contour_variables)
            auxiliary_variables = index(self.auxiliary_variables)
        except TypeError as exc:
            raise TypeError("Variable counts must be integers.") from exc
        if contour_variables < 0 or auxiliary_variables < 0:
            raise ValueError("Variable counts must be nonnegative.")
        object.__setattr__(self, "contour_variables", contour_variables)
        object.__setattr__(self, "auxiliary_variables", auxiliary_variables)

        n_vars = 2 * contour_variables + auxiliary_variables
        if self.phase is not None and self.phase.n != n_vars:
            raise ValueError("phase must contain 2 * contour_variables + auxiliary_variables variables.")

        frozen_factors: dict[tuple[int, ...], tuple[complex, ...]] = {}
        for raw_scope, raw_table in (self.factors or {}).items():
            try:
                scope = tuple(index(var) for var in raw_scope)
            except TypeError as exc:
                raise TypeError("Factor scope variables must be integers.") from exc
            if tuple(sorted(set(scope))) != scope:
                raise ValueError("Factor scopes must be strictly increasing.")
            if any(var < 0 or var >= n_vars for var in scope):
                raise ValueError("Factor scope contains an out-of-range variable.")
            table = tuple(complex(value) for value in raw_table)
            if len(table) != 1 << len(scope):
                raise ValueError("Factor table length must equal 2 ** len(scope).")
            if any(not (math.isfinite(value.real) and math.isfinite(value.imag)) for value in table):
                raise ValueError("Factor table entries must be finite.")
            frozen_factors[scope] = table
        object.__setattr__(self, "factors", MappingProxyType(frozen_factors))
        scalar = complex(self.scalar)
        if not (math.isfinite(scalar.real) and math.isfinite(scalar.imag)):
            raise ValueError("scalar must be finite.")
        object.__setattr__(self, "scalar", scalar)


def _sector_variable_map(problem: DoubledFactorProblem, difference_mask: int) -> tuple[tuple[int, int], ...]:
    k = problem.contour_variables
    return (
        tuple((idx, 0) for idx in range(k))
        + tuple((idx, (difference_mask >> idx) & 1) for idx in range(k))
        + tuple((k + idx, 0) for idx in range(problem.auxiliary_variables))
    )


def _restrict_factor(
    scope: tuple[int, ...],
    table: Sequence[ScaledComplex],
    variable_map: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, ...], list[ScaledComplex]]:
    reduced_scope = tuple(sorted({variable_map[var][0] for var in scope}))
    reduced_positions = {var: pos for pos, var in enumerate(reduced_scope)}
    restricted: list[ScaledComplex] = []
    for assignment in range(1 << len(reduced_scope)):
        original_assignment = 0
        for original_pos, var in enumerate(scope):
            reduced_var, flip = variable_map[var]
            bit = ((assignment >> reduced_positions[reduced_var]) & 1) ^ flip
            original_assignment |= bit << original_pos
        restricted.append(table[original_assignment])
    return reduced_scope, restricted


def _build_sector_factors(
    problem: DoubledFactorProblem,
    difference_mask: int,
    context: _ReductionContext,
    scaled_problem_factors: Mapping[tuple[int, ...], Sequence[ScaledComplex]],
    problem_scalar: ScaledComplex,
) -> tuple[ScaledComplex, dict[tuple[int, ...], list[ScaledComplex]]]:
    variable_map = _sector_variable_map(problem, difference_mask)
    factors: dict[tuple[int, ...], list[ScaledComplex]] = {}
    scalar = problem_scalar

    for scope, table in scaled_problem_factors.items():
        reduced_scope, restricted = _restrict_factor(scope, table, variable_map)
        if all(value[0] == 0j for value in restricted):
            return _ZERO_SCALED, {}
        scalar = _mul_scaled_complex(
            scalar,
            _combine_factor_scaled(factors, reduced_scope, restricted),
        )
        if scalar[0] == 0j:
            return scalar, {}

    if problem.phase is not None:
        row_masks = tuple(1 << reduced_var for reduced_var, _flip in variable_map)
        shift_mask = sum(flip << idx for idx, (_reduced_var, flip) in enumerate(variable_map))
        reduced_phase = _aff_compose_cached(
            problem.phase,
            shift_mask,
            row_masks,
            problem.contour_variables + problem.auxiliary_variables,
            context=context,
        )
        phase_scalar, phase_factors = _build_cubic_factors_scaled(reduced_phase)
        scalar = _mul_scaled_complex(scalar, phase_scalar)
        for scope, table in phase_factors.items():
            scalar = _mul_scaled_complex(
                scalar,
                _combine_factor_scaled(factors, scope, table),
            )
    return scalar, factors


def sum_doubled_factor_problem(
    problem: DoubledFactorProblem,
    *,
    max_difference_weight: int,
    sector_batch_size: int = 128,
    require_native: bool = False,
    max_sectors: int | None = None,
    difference_strategy: Literal["hamming", "factor_bound", "general_bound"] = "hamming",
    omitted_magnitude_tolerance: float | None = None,
) -> DoubledSumResult:
    """Truncate a general coupled doubled-contour factor sum by ``|x xor y|``."""
    k = problem.contour_variables
    max_weight, sector_limit, omitted_magnitude_tolerance = _validate_factor_sum_options(
        problem,
        max_difference_weight,
        sector_batch_size,
        max_sectors,
        difference_strategy,
        omitted_magnitude_tolerance,
    )
    context = _ReductionContext(preserve_scale=True)
    total: ScaledComplex = (0j, 0)
    max_scope = 0
    sectors_evaluated = 0
    order_cache: dict[tuple[tuple[int, ...], ...], list[int]] = {}
    problem_scalar = _make_scaled_complex(problem.scalar)
    scaled_problem_factors = {
        scope: tuple(_make_scaled_complex(value) for value in table)
        for scope, table in problem.factors.items()
    }
    direct = _try_direct_exact(
        problem,
        max_weight=max_weight,
        sector_limit=sector_limit,
        require_native=require_native,
        scaled_problem_factors=scaled_problem_factors,
        problem_scalar=problem_scalar,
    )
    if direct is not None:
        return direct
    omitted_bound = None
    if difference_strategy == "general_bound":
        mask_chunks, omitted_bound = prepare_general_bound_sectors(
            problem,
            max_sectors=sector_limit,
            sector_batch_size=index(sector_batch_size),
            omitted_magnitude_tolerance=omitted_magnitude_tolerance,
            require_native=require_native,
        )
        viable_pair_sectors = 1 << k
        informative_bounds = True
        auto_hard_constraint_order = False
    else:
        (
            mask_chunks,
            sector_limit,
            viable_pair_sectors,
            informative_bounds,
            auto_hard_constraint_order,
        ) = _build_mask_chunks(
            problem,
            max_weight=max_weight,
            sector_limit=sector_limit,
            sector_batch_size=index(sector_batch_size),
            difference_strategy=difference_strategy,
        )

    for masks in mask_chunks:
        for difference_mask in masks:
            scalar, factors = _build_sector_factors(
                problem,
                difference_mask,
                context,
                scaled_problem_factors,
                problem_scalar,
            )
            if scalar[0] == 0j:
                continue
            scope_key = tuple(sorted(factors))
            order = order_cache.get(scope_key)
            if order is None:
                order, _estimated_width = _factor_scope_order(
                    k + problem.auxiliary_variables,
                    scope_key,
                )
                order_cache[scope_key] = order
            sector_total, sector_scope = _sum_factor_tables_scaled(
                k + problem.auxiliary_variables,
                factors,
                order,
                scalar=scalar,
                require_native=require_native,
            )
            total = _add_scaled_complex(total, sector_total)
            max_scope = max(max_scope, sector_scope)
            sectors_evaluated += 1
        context.affine_compose_cache.clear()

    if auto_hard_constraint_order:
        exact = True
    elif sector_limit is None:
        exact = max_weight == k
    elif difference_strategy == "general_bound":
        exact = omitted_bound is not None and omitted_bound.mantissa == 0j
    elif difference_strategy == "factor_bound" and informative_bounds:
        exact = max_weight == k and sector_limit >= viable_pair_sectors
    else:
        exact = max_weight == k and sector_limit >= 1 << k
    backend = "factor_tables_native" if require_native and _schur_native is not None else "factor_tables"
    return DoubledSumResult(
        estimate=ScaledAmplitude.from_tuple(total),
        path_variables=k,
        max_difference_weight=max_weight,
        sectors_evaluated=sectors_evaluated,
        sectors_total=1 << k,
        exact=exact,
        max_reducer_remaining=max_scope,
        phase3_backends=(backend,) if sectors_evaluated else (),
        omitted_magnitude_bound=omitted_bound,
    )


__all__ = ["DoubledFactorProblem", "sum_doubled_factor_problem"]
