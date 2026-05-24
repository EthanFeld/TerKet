"""Extracted q3-free factor-scope planning helpers."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from .._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals

_LOCAL_NAMES = {
    "_factor_scope_order",
    "_estimate_factor_table_dp_cost",
    "_factor_order_scope_sets",
    "_factor_cutset_residual_scopes",
    "_factor_cutset_candidates",
    "_find_arbitrary_factor_cutset_plan",
    "_factor_scope_degeneracy",
}

_MISSING = object()
_ENGINE_LOCAL_BASELINES = {}
_LOCAL_IMPLS = {}


def _sync_from_engine(engine) -> None:
    _sync_extracted_globals(
        globals(),
        engine,
        local_names=_LOCAL_NAMES,
        local_impls=_LOCAL_IMPLS,
        baselines=_ENGINE_LOCAL_BASELINES,
        missing=_MISSING,
        respect_mock_wraps=True,
    )


_MISSING, _ENGINE_LOCAL_BASELINES = _bootstrap_extracted_globals(
    globals(),
    local_names=_LOCAL_NAMES,
    local_impls=_LOCAL_IMPLS,
    respect_mock_wraps=True,
)


def _factor_scope_order(n_vars: int, factor_scopes) -> tuple[list[int], int]:
    if n_vars == 0:
        return [], 0

    dummy_q2: dict[tuple[int, int], int] = {}
    for scope in factor_scopes:
        ordered_scope = tuple(sorted({int(var) for var in scope}))
        for left, right in combinations(ordered_scope, 2):
            dummy_q2.setdefault((left, right), 1)

    dummy_q = _phase_function_from_parts(
        n_vars,
        level=3,
        q0=Fraction(0),
        q1=[0] * n_vars,
        q2=dummy_q2,
        q3={},
    )
    if n_vars >= _Q2_SEPARATOR_ORDER_MIN_VARS and dummy_q2:
        separator_order = _pair_graph_separator_order(dummy_q)
        if separator_order is not None:
            order, width = separator_order
            return list(order), int(width)
    order, width = _min_fill_cubic_order(dummy_q)
    separator_order = _pair_graph_separator_order(dummy_q)
    if separator_order is not None:
        candidate_order, candidate_width = separator_order
        if candidate_width < width:
            order, width = candidate_order, candidate_width
    return list(order), int(width)


def _estimate_factor_table_dp_cost(
    n_vars: int,
    factor_scopes,
    order,
) -> tuple[int, int]:
    """Estimate generic factor bucket-elimination work and largest new table."""
    del n_vars
    factors = {tuple(sorted({int(var) for var in scope})) for scope in factor_scopes if scope}
    work = 0
    max_table_entries = 1

    for var in order:
        bucket_scopes = [scope for scope in factors if var in scope]
        if not bucket_scopes:
            work += 1
            continue

        for scope in bucket_scopes:
            factors.remove(scope)
        union_scope = tuple(sorted({vertex for scope in bucket_scopes for vertex in scope}))
        new_scope = tuple(vertex for vertex in union_scope if vertex != var)
        table_entries = 1 << len(new_scope)
        max_table_entries = max(max_table_entries, table_entries)
        work += max(1, len(bucket_scopes)) * (1 << len(union_scope))
        if new_scope:
            factors.add(new_scope)

    return int(work), int(max_table_entries)


def _factor_order_scope_sets(n_vars: int, factor_scopes, order) -> list[tuple[int, ...]]:
    del n_vars
    factors = {tuple(sorted({int(var) for var in scope})) for scope in factor_scopes if scope}
    scopes: list[tuple[int, ...]] = []

    for var in order:
        bucket_scopes = [scope for scope in factors if var in scope]
        if not bucket_scopes:
            scopes.append((int(var),))
            continue

        for scope in bucket_scopes:
            factors.remove(scope)
        union_scope = tuple(sorted({vertex for scope in bucket_scopes for vertex in scope}))
        scopes.append(union_scope)
        new_scope = tuple(vertex for vertex in union_scope if vertex != var)
        if new_scope:
            factors.add(new_scope)

    return scopes


def _factor_cutset_residual_scopes(n_vars: int, factor_scopes, cutset) -> tuple[int, tuple[int, ...], tuple[tuple[int, ...], ...]]:
    cutset_set = {int(var) for var in cutset}
    remaining_original = tuple(var for var in range(n_vars) if var not in cutset_set)
    remap = {var: idx for idx, var in enumerate(remaining_original)}
    residual_scopes = []
    for scope in factor_scopes:
        residual_scope = tuple(remap[int(var)] for var in scope if int(var) in remap)
        if residual_scope:
            residual_scopes.append(residual_scope)
    return len(remaining_original), remaining_original, tuple(residual_scopes)


def _factor_cutset_candidates(n_vars: int, factor_scopes, residual_order, remaining_original) -> tuple[int, ...]:
    del n_vars
    scopes = _factor_order_scope_sets(len(remaining_original), factor_scopes, residual_order)
    hotspot_scopes = sorted(scopes, key=lambda scope: (len(scope), -sum(scope)), reverse=True)[
        :_MAX_ARBITRARY_PATH_SUM_CUTSET_CANDIDATES
    ]
    counts: dict[int, int] = {}
    degrees: dict[int, int] = {}
    for scope in factor_scopes:
        unique_scope = tuple(sorted({int(var) for var in scope}))
        for residual_var in unique_scope:
            original_var = int(remaining_original[residual_var])
            degrees[original_var] = degrees.get(original_var, 0) + max(0, len(unique_scope) - 1)
    for scope in hotspot_scopes:
        if len(scope) <= 1:
            continue
        for residual_var in scope:
            original_var = int(remaining_original[residual_var])
            counts[original_var] = counts.get(original_var, 0) + 1
    if not counts:
        return ()
    ranked = sorted(
        counts,
        key=lambda var: (counts[var], degrees.get(var, 0), -var),
        reverse=True,
    )
    return tuple(ranked[:_MAX_ARBITRARY_PATH_SUM_CUTSET_CANDIDATES])


def _find_arbitrary_factor_cutset_plan(
    n_vars: int,
    factor_scopes,
    *,
    width_limit: int,
) -> _ArbitraryFactorCutsetPlan | None:
    selected: list[int] = []
    best_plan: _ArbitraryFactorCutsetPlan | None = None

    for _ in range(_MAX_ARBITRARY_PATH_SUM_CUTSET_SIZE + 1):
        residual_n, remaining_original, residual_scopes = _factor_cutset_residual_scopes(
            n_vars,
            factor_scopes,
            selected,
        )
        residual_order, residual_width = _factor_scope_order(residual_n, residual_scopes)
        residual_work, residual_table_entries = _estimate_factor_table_dp_cost(
            residual_n,
            residual_scopes,
            residual_order,
        )
        total_work = (1 << len(selected)) * max(1, int(residual_work))
        if (
            selected
            and residual_width <= width_limit
            and total_work <= _MAX_ARBITRARY_PATH_SUM_WORK
            and residual_table_entries <= _MAX_ARBITRARY_PATH_SUM_TABLE_ENTRIES
        ):
            return _ArbitraryFactorCutsetPlan(
                cutset=tuple(selected),
                residual_order=tuple(residual_order),
                residual_width=int(residual_width),
                residual_work=int(residual_work),
                residual_table_entries=int(residual_table_entries),
            )

        candidate_plan = _ArbitraryFactorCutsetPlan(
            cutset=tuple(selected),
            residual_order=tuple(residual_order),
            residual_width=int(residual_width),
            residual_work=int(residual_work),
            residual_table_entries=int(residual_table_entries),
        )
        if best_plan is None or (
            candidate_plan.residual_width,
            (1 << len(candidate_plan.cutset)) * max(1, candidate_plan.residual_work),
        ) < (
            best_plan.residual_width,
            (1 << len(best_plan.cutset)) * max(1, best_plan.residual_work),
        ):
            best_plan = candidate_plan

        if len(selected) >= _MAX_ARBITRARY_PATH_SUM_CUTSET_SIZE:
            break

        candidates = _factor_cutset_candidates(
            n_vars,
            residual_scopes,
            residual_order,
            remaining_original,
        )
        best_candidate = None
        best_score = None
        for candidate in candidates:
            if candidate in selected:
                continue
            trial_cutset = selected + [candidate]
            trial_n, _trial_remaining, trial_scopes = _factor_cutset_residual_scopes(
                n_vars,
                factor_scopes,
                trial_cutset,
            )
            trial_order, trial_width = _factor_scope_order(trial_n, trial_scopes)
            trial_work, trial_table_entries = _estimate_factor_table_dp_cost(
                trial_n,
                trial_scopes,
                trial_order,
            )
            trial_total_work = (1 << len(trial_cutset)) * max(1, int(trial_work))
            score = (
                int(trial_width),
                int(trial_total_work),
                int(trial_table_entries),
                int(candidate),
            )
            if best_score is None or score < best_score:
                best_candidate = int(candidate)
                best_score = score
        if best_candidate is None:
            break
        selected.append(best_candidate)

    return None


def _factor_scope_degeneracy(n_vars: int, factor_scopes) -> int:
    """Return the degeneracy lower bound of the factor-induced pair graph."""
    adjacency = [set() for _ in range(n_vars)]
    for scope in factor_scopes:
        ordered_scope = tuple(sorted({int(var) for var in scope}))
        for left, right in combinations(ordered_scope, 2):
            adjacency[left].add(right)
            adjacency[right].add(left)
    return _pair_graph_degeneracy(adjacency)


_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
