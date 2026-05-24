"""Extracted arbitrary-angle runtime helpers."""

from __future__ import annotations

import importlib
import math
import sys
from typing import Mapping, Sequence

from ._engine_runtime_core import _configure_extracted_module
from .cubic_arithmetic import PhaseFunction
from .scaling import ScaledComplex, _scaled_log2_abs, _scaled_phase
from .state import BitSequence

_LOCAL_NAMES = {
    '_factor_graph_is_forest',
    '_scaled_log2_abs',
    '_scaled_phase',
    '_phase_distance',
    '_arbitrary_bp_heuristic_candidate',
    '_sum_arbitrary_bp_heuristic_ensemble_scaled',
    '_arbitrary_factor_graph_for_state_output',
    'solve_arbitrary_exact',
    'solve_arbitrary_approx',
    '_sum_with_arbitrary_phases_scaled'
}
_LOCAL_IMPLS = {}
_FORCE_ENGINE_BINDINGS_REFRESH = "pytest" in sys.modules
_configure_extracted_module(globals(), local_names=_LOCAL_NAMES, local_impls=_LOCAL_IMPLS)


def _refresh_engine_bindings() -> None:
    if not _FORCE_ENGINE_BINDINGS_REFRESH:
        return
    _sync_from_engine(importlib.import_module("terket._engine_impl"))


def _factor_graph_is_forest(n_vars: int, factor_scopes: Sequence[Sequence[int]]) -> bool:
    """Return true when the bipartite factor graph is acyclic, where BP is exact."""
    parent = list(range(max(0, int(n_vars)) + len(factor_scopes)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> bool:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return False
        parent[root_right] = root_left
        return True

    for factor_idx, scope in enumerate(factor_scopes):
        factor_node = int(n_vars) + factor_idx
        for var in scope:
            if not union(int(var), factor_node):
                return False
    return True


def _phase_distance(left: float, right: float) -> float:
    return abs(math.atan2(math.sin(left - right), math.cos(left - right)))


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


def _arbitrary_bp_heuristic_candidate(
    n_vars: int,
    dense_factors: Mapping[tuple[int, ...], Sequence[ScaledComplex]] | None,
    dense_scalar: ScaledComplex,
    base_factors: Mapping[tuple[int, ...], Sequence[ScaledComplex]],
    terms: Sequence[_ArbitraryPhaseTerm],
    base_scalar: ScaledComplex,
    *,
    max_iters: int,
    damping: float,
) -> tuple[ScaledComplex, int, str] | None:
    if dense_factors is not None:
        candidate = _sum_pairwise_factor_graph_bethe_scaled(
            n_vars,
            dense_factors,
            scalar=dense_scalar,
            max_iters=max_iters,
            damping=damping,
            require_forest=False,
        )
        if candidate is not None:
            total, max_scope = candidate
            return total, max_scope, "arbitrary_bethe_bp_heuristic"
        candidate = _sum_factor_graph_bethe_scaled(
            n_vars,
            dense_factors,
            scalar=dense_scalar,
            max_iters=max_iters,
            damping=damping,
            require_forest=False,
        )
        if candidate is not None:
            total, max_scope = candidate
            return total, max_scope, "arbitrary_factor_bethe_bp_heuristic"

    candidate = _sum_factor_graph_with_sparse_parity_bethe_scaled(
        n_vars,
        base_factors,
        terms,
        scalar=base_scalar,
        max_iters=max_iters,
        damping=damping,
        require_forest=False,
    )
    if candidate is None:
        return None
    total, max_scope = candidate
    return total, max_scope, "arbitrary_sparse_parity_bethe_bp_heuristic"


def _sum_arbitrary_bp_heuristic_ensemble_scaled(
    n_vars: int,
    dense_factors: Mapping[tuple[int, ...], Sequence[ScaledComplex]] | None,
    dense_scalar: ScaledComplex,
    base_factors: Mapping[tuple[int, ...], Sequence[ScaledComplex]],
    terms: Sequence[_ArbitraryPhaseTerm],
    base_scalar: ScaledComplex,
) -> tuple[ScaledComplex, int, str, dict[str, object]] | None:
    config = _get_solver_config()
    candidates: list[tuple[ScaledComplex, int, str, int, float]] = []
    for max_iters, damping in _ARBITRARY_BP_HEURISTIC_SCHEDULES:
        candidate = _arbitrary_bp_heuristic_candidate(
            n_vars,
            dense_factors,
            dense_scalar,
            base_factors,
            terms,
            base_scalar,
            max_iters=max_iters,
            damping=damping,
        )
        if candidate is not None:
            total, max_scope, backend = candidate
            candidates.append((total, max_scope, backend, int(max_iters), float(damping)))

    if len(candidates) < 2:
        return None

    log2_probabilities = [_scaled_probability_log2(total) for total, _scope, _backend, _iters, _damping in candidates]
    max_log2_probability = max(log2_probabilities)
    if max_log2_probability > float(config.bp_heuristic_bound_log2_tol):
        return None

    log2_abs_values = [_scaled_log2_abs(total) for total, _scope, _backend, _iters, _damping in candidates]
    finite_log2_abs = [value for value in log2_abs_values if math.isfinite(value)]
    if len(finite_log2_abs) != len(log2_abs_values):
        if finite_log2_abs:
            return None
        log2_abs_spread = 0.0
    else:
        log2_abs_spread = max(finite_log2_abs) - min(finite_log2_abs)
    if log2_abs_spread > float(config.bp_heuristic_max_log2_abs_spread):
        return None

    phases = [phase for total, _scope, _backend, _iters, _damping in candidates if (phase := _scaled_phase(total)) is not None]
    phase_spread = 0.0
    for idx, left in enumerate(phases):
        for right in phases[idx + 1:]:
            phase_spread = max(phase_spread, _phase_distance(left, right))
    if phase_spread > float(config.bp_heuristic_max_phase_spread):
        return None

    if finite_log2_abs:
        median_log2_abs = sorted(finite_log2_abs)[len(finite_log2_abs) // 2]
        selected_idx = min(
            range(len(candidates)),
            key=lambda idx: abs(log2_abs_values[idx] - median_log2_abs),
        )
    else:
        selected_idx = 0
    total, max_scope, backend, _iters, _damping = candidates[selected_idx]
    metadata: dict[str, object] = {
        "bp_heuristic_ensemble_size": len(candidates),
        "bp_heuristic_log2_abs_spread": float(log2_abs_spread),
        "bp_heuristic_phase_spread": float(phase_spread),
        "bp_heuristic_max_log2_probability": float(max_log2_probability),
    }
    return total, max_scope, backend, metadata


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
    """Exact arbitrary-angle path-sum. Never falls back to BP/MPS."""
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
            return total, int(max_scope), "arbitrary_path_sum_cutset", _arbitrary_exact_metadata()
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
    return total, int(max_scope), "arbitrary_path_sum", _arbitrary_exact_metadata()


_sum_with_arbitrary_phases_exact_scaled = solve_arbitrary_exact


def solve_arbitrary_approx(
    q: PhaseFunction,
    terms: Sequence[_ArbitraryPhaseTerm],
) -> tuple[ScaledComplex, int, str, dict[str, object]] | None:
    """Opt-in arbitrary-angle approximate fallback. Returns None on rejection."""
    _refresh_engine_bindings()
    scalar, factors = _build_cubic_factors_scaled(q)
    base_scalar = scalar
    base_factors = dict(factors)
    wide_arbitrary = any(
        int(term.row_mask).bit_count() > _MAX_ARBITRARY_PHASE_FACTOR_SCOPE
        for term in terms
    )

    if wide_arbitrary:
        sparse_factor_scopes = tuple(factors) + tuple(
            _support_from_mask(int(term.row_mask))
            for term in terms
            if int(term.row_mask)
        )
        if _factor_graph_is_forest(q.n, sparse_factor_scopes):
            approximate = _sum_factor_graph_with_sparse_parity_bethe_scaled(
                q.n,
                factors,
                terms,
                scalar=scalar,
            )
            if approximate is not None:
                total, max_scope = approximate
                backend = "arbitrary_sparse_parity_bethe_bp"
                return (
                    total,
                    int(max_scope),
                    backend,
                    _arbitrary_approx_metadata(backend, "factor_graph_forest_exact"),
                )
        heuristic = _sum_arbitrary_bp_heuristic_ensemble_scaled(
            q.n,
            None,
            scalar,
            base_factors,
            terms,
            base_scalar,
        )
        if heuristic is None:
            return None
        total, max_scope, backend, metadata = heuristic
        return total, int(max_scope), backend, _arbitrary_approx_metadata(
            backend,
            "loopy_ensemble_thresholds",
            metadata,
        )

    scalar = _mul_scaled_complex(scalar, _add_arbitrary_phase_factors_scaled(factors, terms))
    factor_scopes, _order, width, work, max_table_entries, width_limit = _path_sum_plan_metrics(q, factors)
    over_limit = (
        width > width_limit
        or work > _MAX_ARBITRARY_PATH_SUM_WORK
        or max_table_entries > _MAX_ARBITRARY_PATH_SUM_TABLE_ENTRIES
    )
    if not over_limit:
        return None

    approximate = None
    approximate_backend = "arbitrary_bethe_bp"
    if _factor_graph_is_forest(q.n, factor_scopes):
        approximate = _sum_pairwise_factor_graph_bethe_scaled(
            q.n,
            factors,
            scalar=scalar,
        )
        if approximate is None:
            approximate = _sum_factor_graph_bethe_scaled(
                q.n,
                factors,
                scalar=scalar,
            )
            approximate_backend = "arbitrary_factor_bethe_bp"
    if approximate is not None:
        total, max_scope = approximate
        return (
            total,
            int(max_scope),
            approximate_backend,
            _arbitrary_approx_metadata(approximate_backend, "factor_graph_forest_exact"),
        )

    sparse_factor_scopes = tuple(base_factors) + tuple(
        _support_from_mask(int(term.row_mask))
        for term in terms
        if int(term.row_mask)
    )
    if _factor_graph_is_forest(q.n, sparse_factor_scopes):
        approximate = _sum_factor_graph_with_sparse_parity_bethe_scaled(
            q.n,
            base_factors,
            terms,
            scalar=base_scalar,
        )
        if approximate is not None:
            total, max_scope = approximate
            backend = "arbitrary_sparse_parity_bethe_bp"
            return (
                total,
                int(max_scope),
                backend,
                _arbitrary_approx_metadata(backend, "factor_graph_forest_exact"),
            )

    heuristic = _sum_arbitrary_bp_heuristic_ensemble_scaled(
        q.n,
        factors,
        scalar,
        base_factors,
        terms,
        base_scalar,
    )
    if heuristic is None:
        return None
    total, max_scope, backend, metadata = heuristic
    return total, int(max_scope), backend, _arbitrary_approx_metadata(
        backend,
        "loopy_ensemble_thresholds",
        metadata,
    )


_sum_with_arbitrary_phases_approx_scaled = solve_arbitrary_approx


def _sum_with_arbitrary_phases_scaled(
    q: PhaseFunction,
    terms: Sequence[_ArbitraryPhaseTerm],
    *,
    allow_approximate: bool = False,
) -> tuple[ScaledComplex, int, str, dict[str, object]]:
    """Exact arbitrary-angle sum, with explicit opt-in approximate fallback."""
    _refresh_engine_bindings()
    try:
        return solve_arbitrary_exact(q, terms)
    except RuntimeError as exact_error:
        if not allow_approximate:
            raise
        approximate = solve_arbitrary_approx(q, terms)
        if approximate is not None:
            return approximate
        raise RuntimeError(
            f"{exact_error} Approximate arbitrary-angle fallback failed acceptance thresholds."
        ) from exact_error

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
