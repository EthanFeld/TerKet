"""Exact arbitrary-angle runtime helpers."""

from __future__ import annotations

import importlib
import sys
from typing import Mapping, Sequence

from ._engine_runtime_core import _configure_extracted_module
from .cubic_arithmetic import PhaseFunction
from .scaling import ScaledComplex
from .state import BitSequence

_LOCAL_NAMES = {
    '_arbitrary_factor_graph_for_state_output',
    'solve_arbitrary_exact',
    '_sum_with_arbitrary_phases_scaled',
}
_LOCAL_IMPLS = {}
_FORCE_ENGINE_BINDINGS_REFRESH = "pytest" in sys.modules
_configure_extracted_module(globals(), local_names=_LOCAL_NAMES, local_impls=_LOCAL_IMPLS)


def _refresh_engine_bindings() -> None:
    if not _FORCE_ENGINE_BINDINGS_REFRESH:
        return
    _sync_from_engine(importlib.import_module("terket._engine_impl"))


def _path_sum_plan_metrics(
    q: PhaseFunction,
    factors: Mapping[tuple[int, ...], Sequence[ScaledComplex]],
) -> tuple[tuple[tuple[int, ...], ...], list[int], int, int, int, int]:
    factor_scopes = tuple(factors)
    if factor_scopes:
        order, width = _factor_scope_order(q.n, factor_scopes)
    else:
        order = list(range(q.n))
        width = 1 if q.n else 0
    work, max_table_entries = _estimate_factor_table_dp_cost(q.n, factor_scopes, order)
    width_limit = (
        _MAX_ARBITRARY_PATH_SUM_NATIVE_WIDTH
        if _schur_native is not None and hasattr(_schur_native, "sum_factor_tables_scaled")
        else _MAX_ARBITRARY_PATH_SUM_PY_WIDTH
    )
    return factor_scopes, order, width, work, max_table_entries, width_limit


def _arbitrary_factor_graph_for_state_output(
    state: "SchurState",
    output_bits: BitSequence,
    context: "_ReductionContext",
) -> tuple[int, dict[tuple[int, ...], Sequence[ScaledComplex]], ScaledComplex] | None:
    cache = state._prepare_echelon()
    solved = state._solve_for_output(cache, output_bits)
    if solved is None:
        return None
    shift_mask, _, gamma, n_free = solved
    q_free = _aff_compose_cached(state.q, shift_mask, gamma, n_free, context=context)
    arbitrary_scalar, arbitrary_terms = state._transform_arbitrary_phases(shift_mask, gamma)
    scalar, factors = _build_cubic_factors_scaled(q_free)
    scalar = _mul_scaled_complex(
        scalar,
        _make_scaled_complex(complex(state.scalar) * arbitrary_scalar),
    )
    scalar = _scale_scaled_complex(scalar, int(state.scalar_half_pow2))
    scalar = _mul_scaled_complex(scalar, _add_arbitrary_phase_factors_scaled(factors, arbitrary_terms))
    return q_free.n, factors, scalar


def solve_arbitrary_exact(
    q: PhaseFunction,
    terms: Sequence[_ArbitraryPhaseTerm],
) -> tuple[ScaledComplex, int, str, dict[str, object]]:
    """Exact arbitrary-angle path-sum."""
    _refresh_engine_bindings()
    scalar, factors = _build_cubic_factors_scaled(q)
    wide_arbitrary = any(
        int(term.row_mask).bit_count() > _MAX_ARBITRARY_PHASE_FACTOR_SCOPE
        for term in terms
    )
    if wide_arbitrary:
        max_term_scope = max(int(term.row_mask).bit_count() for term in terms)
        raise RuntimeError(
            f"Cannot compute amplitude directly: arbitrary-angle factor has scope {max_term_scope}, "
            f"above limit {_MAX_ARBITRARY_PHASE_FACTOR_SCOPE}."
        )

    scalar = _mul_scaled_complex(scalar, _add_arbitrary_phase_factors_scaled(factors, terms))
    factor_scopes, order, width, work, max_table_entries, width_limit = _path_sum_plan_metrics(q, factors)
    over_limit = (
        width > width_limit
        or work > _MAX_ARBITRARY_PATH_SUM_WORK
        or max_table_entries > _MAX_ARBITRARY_PATH_SUM_TABLE_ENTRIES
    )
    require_native = _schur_native is not None and hasattr(_schur_native, "sum_factor_tables_scaled")
    if over_limit:
        cutset_plan = None
        if width - _MAX_ARBITRARY_PATH_SUM_CUTSET_SIZE <= width_limit:
            cutset_plan = _find_arbitrary_factor_cutset_plan(
                q.n,
                factor_scopes,
                width_limit=width_limit,
            )
        if cutset_plan is not None:
            total, max_scope = _sum_factor_tables_with_cutset_scaled(
                q.n,
                factors,
                cutset_plan,
                scalar=scalar,
                require_native=require_native,
            )
            return total, int(max_scope), "arbitrary_path_sum_cutset", {}
        raise RuntimeError(
            "Cannot compute amplitude directly: arbitrary-angle path-sum "
            f"width {width}, work {work}, table entries {max_table_entries} exceed "
            f"limits width {width_limit}, work {_MAX_ARBITRARY_PATH_SUM_WORK}, "
            f"table entries {_MAX_ARBITRARY_PATH_SUM_TABLE_ENTRIES}; no exact cutset "
            f"of size <= {_MAX_ARBITRARY_PATH_SUM_CUTSET_SIZE} lowered total work enough."
        )

    total, max_scope = _sum_factor_tables_scaled(
        q.n,
        factors,
        order,
        scalar=scalar,
        require_native=require_native,
    )
    return total, int(max_scope), "arbitrary_path_sum", {}


_sum_with_arbitrary_phases_exact_scaled = solve_arbitrary_exact
_sum_with_arbitrary_phases_scaled = solve_arbitrary_exact

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
