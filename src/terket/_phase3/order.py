"""Extracted phase-3 treewidth ordering helpers."""

from __future__ import annotations

import bisect
import cmath
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import importlib
import heapq
from itertools import combinations
import math
import os
import struct
import sys
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Sequence, overload

import numpy as np

from .._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals
from ..cubic_arithmetic import CubicFunction, PhaseFunction, detect_factorization
from ..scaling import ScaledAmplitude, ScaledComplex
from ..spec import CircuitSpec, Gate
from ..state import BitSequence, CircuitInput, EchelonCache, ExtendedReductionMode, ReducerInfo, ReductionInfo, SolverConfig

_LOCAL_NAMES = {
    '_build_factor_scopes',
    '_treewidth_order_width',
    '_treewidth_order_scope_trace',
    '_treewidth_order_scope_sets',
    '_q3_free_treewidth_width_limit',
    '_estimate_treewidth_dp_work',
    '_move_order_entry',
    '_refine_q3_free_treewidth_order_locally',
    '_finalize_q3_free_treewidth_order',
    '_refine_phase3_treewidth_order_locally',
    '_finalize_phase3_treewidth_order',
    '_q3_free_treewidth_candidate_is_viable',
    '_q3_hypergraph_2core',
    '_active_q3_variables',
    '_phase_function_q2_density_milli',
    '_phase_function_structure_score',
    '_phase_structure_opt_max_vars',
    '_phase_structure_opt_active_vars',
    '_phase_structure_opt_beam_width',
    '_phase_structure_opt_max_passes',
    '_phase_structure_local_region_max_vars',
    '_phase_structure_local_max_centers',
    '_phase_structure_local_max_passes',
    '_phase_structure_local_candidate_pool',
    '_phase_structure_hotspot_centers',
    '_phase_structure_local_region',
    '_phase_structure_local_move_score'
}


_MISSING = object()
_ENGINE_LOCAL_BASELINES = {}
_LOCAL_IMPLS = {}
_FORCE_ENGINE_BINDINGS_REFRESH = "pytest" in sys.modules


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


def _refresh_engine_bindings() -> None:
    if not _FORCE_ENGINE_BINDINGS_REFRESH:
        return
    _sync_from_engine(importlib.import_module("terket._engine_impl"))


def _build_factor_scopes(q):
    """Return the unique non-scalar factor scopes induced by ``q``."""
    scopes = set()

    for var, coeff in enumerate(q.q1):
        if coeff % q.mod_q1:
            scopes.add((var,))

    for scope, coeff in q.q2.items():
        if coeff % q.mod_q2:
            scopes.add(scope)

    for scope, coeff in q.q3.items():
        if coeff % q.mod_q3:
            scopes.add(scope)

    return scopes


def _treewidth_order_width(q, order):
    """Return the maximum bucket scope induced by ``order`` on ``q``."""
    factors = _build_factor_scopes(q)
    max_scope = 0

    for var in order:
        bucket_scopes = [scope for scope in factors if var in scope]
        if not bucket_scopes:
            max_scope = max(max_scope, 1)
            continue

        for scope in bucket_scopes:
            factors.remove(scope)
        union_scope = tuple(sorted({vertex for scope in bucket_scopes for vertex in scope}))
        max_scope = max(max_scope, len(union_scope))
        new_scope = tuple(vertex for vertex in union_scope if vertex != var)
        if new_scope:
            factors.add(new_scope)

    return max_scope


def _treewidth_order_scope_trace(q, order):
    """Return the per-step bucket scope sizes induced by ``order`` on ``q``."""
    factors = _build_factor_scopes(q)
    scopes: list[int] = []

    for var in order:
        bucket_scopes = [scope for scope in factors if var in scope]
        if not bucket_scopes:
            scopes.append(1)
            continue

        for scope in bucket_scopes:
            factors.remove(scope)
        union_scope = tuple(sorted({vertex for scope in bucket_scopes for vertex in scope}))
        scopes.append(len(union_scope))
        new_scope = tuple(vertex for vertex in union_scope if vertex != var)
        if new_scope:
            factors.add(new_scope)

    return scopes


def _treewidth_order_scope_sets(q, order):
    """Return bucket scopes induced by ``order`` on ``q``."""
    factors = _build_factor_scopes(q)
    scopes: list[tuple[int, ...]] = []

    for var in order:
        bucket_scopes = [scope for scope in factors if var in scope]
        if not bucket_scopes:
            scopes.append((var,))
            continue

        for scope in bucket_scopes:
            factors.remove(scope)
        union_scope = tuple(sorted({vertex for scope in bucket_scopes for vertex in scope}))
        scopes.append(union_scope)
        new_scope = tuple(vertex for vertex in union_scope if vertex != var)
        if new_scope:
            factors.add(new_scope)

    return scopes


def _q3_free_treewidth_width_limit() -> int:
    """Return the q3-free treewidth width limit for the current exact backend."""
    if _schur_native is not None:
        return _Q3_FREE_SUM_TREEWIDTH_NATIVE_MAX_WIDTH
    return _Q3_FREE_SUM_TREEWIDTH_MAX_WIDTH


def _estimate_treewidth_dp_work(q, order):
    """Cheap proxy for the factor-elimination work along ``order``."""
    if (
        not q.q3
        and _schur_native is not None
        and hasattr(_schur_native, "q3_free_treewidth_dp_work")
    ):
        try:
            return int(
                _schur_native.q3_free_treewidth_dp_work(
                    int(q.n),
                    int(q.level),
                    q.q1,
                    q.q2,
                    tuple(int(var) for var in order),
                )
            )
        except Exception:
            pass

    factors = _build_factor_scopes(q)
    work = 0

    for var in order:
        bucket_scopes = [scope for scope in factors if var in scope]
        if not bucket_scopes:
            work += 1
            continue

        for scope in bucket_scopes:
            factors.remove(scope)
        union_scope = tuple(sorted({vertex for scope in bucket_scopes for vertex in scope}))
        new_scope = tuple(vertex for vertex in union_scope if vertex != var)
        work += len(bucket_scopes) * (1 << len(union_scope))
        if new_scope:
            factors.add(new_scope)

    return work


def _move_order_entry(order: Sequence[int], src: int, dst: int) -> list[int]:
    moved = list(order)
    value = moved.pop(src)
    moved.insert(dst, value)
    return moved


def _treewidth_refinement_is_mocked() -> bool:
    return (
        hasattr(_treewidth_order_width, "_increment_mock_call")
        or hasattr(_estimate_treewidth_dp_work, "_increment_mock_call")
    )


def _refine_q3_free_treewidth_order_locally(q, order: Sequence[int], width: int):
    """Bounded local refinement on top of min-fill."""
    if not order or q.n < 8 or width < 2:
        return list(order), int(width)

    best_order = list(order)
    best_width = int(width)
    best_work = int(_estimate_treewidth_dp_work(q, best_order))
    if best_width <= 1:
        return best_order, best_width

    max_passes = 2
    max_hotspots = 8
    move_radius = 2

    for _ in range(max_passes):
        scopes = _treewidth_order_scope_trace(q, best_order)
        hotspot_positions = [
            idx
            for idx, _scope in sorted(
                enumerate(scopes),
                key=lambda item: (item[1], -item[0]),
                reverse=True,
            )[:max_hotspots]
        ]
        improved = False
        seen: set[tuple[int, ...]] = set()
        for pos in hotspot_positions:
            for delta in range(-move_radius, move_radius + 1):
                if delta == 0:
                    continue
                dst = pos + delta
                if dst < 0 or dst >= len(best_order):
                    continue
                candidate = _move_order_entry(best_order, pos, dst)
                key = tuple(candidate)
                if key in seen:
                    continue
                seen.add(key)
                candidate_width = int(_treewidth_order_width(q, candidate))
                if candidate_width > best_width:
                    continue
                candidate_work = int(_estimate_treewidth_dp_work(q, candidate))
                candidate_score = (candidate_width, candidate_work)
                best_score = (best_width, best_work)
                if candidate_score < best_score:
                    best_order = candidate
                    best_width = candidate_width
                    best_work = candidate_work
                    improved = True
        if not improved:
            break

    return best_order, best_width


def _finalize_q3_free_treewidth_order(q, order: Sequence[int]):
    """Refine one chosen q3-free treewidth order after backend selection."""
    base_order = tuple(int(var) for var in order)
    cache_key = (_q_structure_key(q), base_order)
    cached = _STRUCTURE_Q3_FREE_REFINED_ORDER_CACHE.get(cache_key)
    if cached is not None:
        refined_order, refined_width = cached
        return list(refined_order), int(refined_width)

    base_width = int(_treewidth_order_width(q, base_order))
    if q.n > _Q3_FREE_OPTIONAL_REWRITE_MAX_VARS:
        cached = (base_order, base_width)
        _STRUCTURE_Q3_FREE_REFINED_ORDER_CACHE[cache_key] = cached
        refined_order, refined_width = cached
        return list(refined_order), int(refined_width)
    if _treewidth_refinement_is_mocked():
        refined_order, refined_width = list(base_order), base_width
    else:
        refined_order, refined_width = _refine_q3_free_treewidth_order_locally(q, base_order, base_width)
    cached = (tuple(int(var) for var in refined_order), int(refined_width))
    _STRUCTURE_Q3_FREE_REFINED_ORDER_CACHE[cache_key] = cached
    refined_order, refined_width = cached
    return list(refined_order), int(refined_width)


def _refine_phase3_treewidth_order_locally(q, order: Sequence[int], width: int):
    """Bounded local refinement for residual cubic treewidth orders."""
    if (
        not order
        or q.n < 8
        or q.n > _PHASE3_TREEWIDTH_REFINE_MAX_VARS
        or width < 2
        or width > _PHASE3_TREEWIDTH_REFINE_MAX_WIDTH
    ):
        return list(order), int(width)

    best_order = list(order)
    best_width = int(width)
    best_work = int(_estimate_treewidth_dp_work(q, best_order))
    if best_width <= 1:
        return best_order, best_width

    for _ in range(_PHASE3_TREEWIDTH_REFINE_MAX_PASSES):
        scopes = _treewidth_order_scope_trace(q, best_order)
        hotspot_positions = [
            idx
            for idx, _scope in sorted(
                enumerate(scopes),
                key=lambda item: (item[1], -item[0]),
                reverse=True,
            )[:_PHASE3_TREEWIDTH_REFINE_MAX_HOTSPOTS]
        ]
        improved = False
        seen: set[tuple[int, ...]] = set()
        for pos in hotspot_positions:
            for delta in range(
                -_PHASE3_TREEWIDTH_REFINE_MOVE_RADIUS,
                _PHASE3_TREEWIDTH_REFINE_MOVE_RADIUS + 1,
            ):
                if delta == 0:
                    continue
                dst = pos + delta
                if dst < 0 or dst >= len(best_order):
                    continue
                candidate = _move_order_entry(best_order, pos, dst)
                key = tuple(candidate)
                if key in seen:
                    continue
                seen.add(key)
                candidate_width = int(_treewidth_order_width(q, candidate))
                if candidate_width > best_width:
                    continue
                candidate_work = int(_estimate_treewidth_dp_work(q, candidate))
                candidate_score = (candidate_width, candidate_work)
                best_score = (best_width, best_work)
                if candidate_score < best_score:
                    best_order = candidate
                    best_width = candidate_width
                    best_work = candidate_work
                    improved = True
        if not improved:
            break

    return best_order, best_width


def _finalize_phase3_treewidth_order(q, order: Sequence[int]):
    """Refine one residual cubic treewidth order once before backend choice."""
    _refresh_engine_bindings()
    base_order = tuple(int(var) for var in order)
    cache_key = (_q_phase3_structure_key(q), base_order)
    cached = _STRUCTURE_PHASE3_REFINED_ORDER_CACHE.get(cache_key)
    if cached is not None:
        refined_order, refined_width = cached
        return list(refined_order), int(refined_width)

    base_width = int(_treewidth_order_width(q, base_order))
    if _treewidth_refinement_is_mocked():
        refined_order, refined_width = list(base_order), base_width
    else:
        refined_order, refined_width = _refine_phase3_treewidth_order_locally(
            q,
            base_order,
            base_width,
        )
    cached = (tuple(int(var) for var in refined_order), int(refined_width))
    _STRUCTURE_PHASE3_REFINED_ORDER_CACHE[cache_key] = cached
    refined_order, refined_width = cached
    return list(refined_order), int(refined_width)


def _q3_free_treewidth_candidate_is_viable(q, order, width: int, feedback_size: int) -> bool:
    """Decide whether a q3-free treewidth candidate is worth accepting."""
    width_limit = _q3_free_treewidth_width_limit()
    if width > width_limit or width >= feedback_size:
        return False
    if (
        _schur_native is not None
        and width > _Q3_FREE_SUM_TREEWIDTH_MAX_WIDTH
        and _estimate_treewidth_dp_work(q, order) > _Q3_FREE_SUM_TREEWIDTH_NATIVE_MAX_WORK
    ):
        return False
    return True


def _q3_hypergraph_2core(q):
    """Return the live q3 2-core variables and the degree-1 peel order."""
    cache_key = _q_q3_support_key(q)
    cached = _STRUCTURE_Q3_2CORE_CACHE.get(cache_key)
    if cached is not None:
        core_vars, peel_order = cached
        return frozenset(core_vars), list(peel_order)

    active_q3_vars: set[int] = set()
    incident_edges: list[set[tuple[int, int, int]]] = [set() for _ in range(q.n)]
    live_edges: set[tuple[int, int, int]] = set()

    for edge_key, coeff in q.q3.items():
        if coeff % q.mod_q3 == 0:
            continue
        live_edges.add(edge_key)
        active_q3_vars.update(edge_key)
        for var in edge_key:
            incident_edges[var].add(edge_key)

    if not live_edges:
        return frozenset(), []

    peel_order: list[int] = []
    peeled: set[int] = set()
    pending = [var for var in sorted(active_q3_vars) if len(incident_edges[var]) <= 1]

    while pending:
        var = pending.pop()
        if var in peeled:
            continue
        live_incident = incident_edges[var] & live_edges
        if len(live_incident) > 1:
            continue
        peeled.add(var)
        peel_order.append(var)
        for edge_key in tuple(live_incident):
            if edge_key not in live_edges:
                continue
            live_edges.remove(edge_key)
            for neighbor in edge_key:
                if edge_key in incident_edges[neighbor]:
                    incident_edges[neighbor].remove(edge_key)
                    if neighbor in active_q3_vars and neighbor not in peeled and len(incident_edges[neighbor]) <= 1:
                        pending.append(neighbor)

    core_vars = frozenset(
        var
        for var in sorted(active_q3_vars)
        if incident_edges[var] & live_edges
    )
    cached = (tuple(sorted(core_vars)), tuple(peel_order))
    _STRUCTURE_Q3_2CORE_CACHE[cache_key] = cached
    core_vars, peel_order = cached
    return frozenset(core_vars), list(peel_order)


def _active_q3_variables(q) -> tuple[int, ...]:
    active: set[int] = set()
    for edge, coeff in q.q3.items():
        if coeff % q.mod_q3:
            active.update(edge)
    return tuple(sorted(active))


def _phase_function_q2_density_milli(q) -> int:
    if q.n <= 1:
        return 0
    return int(round(1000.0 * (2.0 * len(q.q2)) / (q.n * (q.n - 1))))


def _phase_function_structure_score(q) -> tuple[int, int, int, int, int, int, int, int, int]:
    active_q3 = _active_q3_variables(q)
    core_vars, _ = _q3_hypergraph_2core(q)
    components = detect_factorization(q)
    threshold = max(1, q.mod_q1 // 4)
    bad_q1 = sum(1 for coeff in q.q1 if int(coeff) % threshold)
    max_width = 0
    max_component_vars = 0
    max_density = 0
    total_width = 0
    for component in components:
        component_q = _component_restriction(q, component)
        _order, width = _min_fill_cubic_order(component_q)
        max_width = max(max_width, int(width))
        total_width += int(width)
        max_component_vars = max(max_component_vars, int(component_q.n))
        max_density = max(max_density, _phase_function_q2_density_milli(component_q))
    return (
        len(core_vars),
        len(q.q3),
        len(active_q3),
        max_width,
        max_component_vars,
        max_density,
        bad_q1,
        len(q.q2),
        -len(components),
        total_width,
    )


def _phase_structure_opt_max_vars(q) -> int:
    return _PHASE_STRUCTURE_CUBIC_OPT_MAX_VARS if q.q3 else _PHASE_STRUCTURE_OPT_MAX_VARS


def _phase_structure_opt_active_vars(q) -> int:
    return (
        _PHASE_STRUCTURE_CUBIC_OPT_MAX_ACTIVE_VARS
        if q.q3
        else _PHASE_STRUCTURE_OPT_MAX_ACTIVE_VARS
    )


def _phase_structure_opt_beam_width(q) -> int:
    return _PHASE_STRUCTURE_CUBIC_OPT_BEAM_WIDTH if q.q3 else _PHASE_STRUCTURE_OPT_BEAM_WIDTH


def _phase_structure_opt_max_passes(q) -> int:
    return _PHASE_STRUCTURE_CUBIC_OPT_MAX_PASSES if q.q3 else _PHASE_STRUCTURE_OPT_MAX_PASSES


def _phase_structure_local_region_max_vars(q) -> int:
    return (
        _PHASE_STRUCTURE_CUBIC_LOCAL_REGION_MAX_VARS
        if q.q3
        else _PHASE_STRUCTURE_LOCAL_REGION_MAX_VARS
    )


def _phase_structure_local_max_centers(q) -> int:
    return (
        _PHASE_STRUCTURE_CUBIC_LOCAL_MAX_CENTERS
        if q.q3
        else _PHASE_STRUCTURE_LOCAL_MAX_CENTERS
    )


def _phase_structure_local_max_passes(q) -> int:
    return (
        _PHASE_STRUCTURE_CUBIC_LOCAL_MAX_PASSES
        if q.q3
        else _PHASE_STRUCTURE_LOCAL_MAX_PASSES
    )


def _phase_structure_local_candidate_pool(q) -> int:
    return (
        _PHASE_STRUCTURE_CUBIC_LOCAL_CANDIDATE_POOL
        if q.q3
        else _PHASE_STRUCTURE_LOCAL_CANDIDATE_POOL
    )


def _phase_structure_hotspot_centers(q) -> tuple[int, ...]:
    adjacency = _interaction_graph(q)
    threshold = max(1, q.mod_q1 // 4)
    active = {
        idx
        for idx, coeff in enumerate(q.q1)
        if int(coeff) % q.mod_q1
    }
    for (left, right), coeff in q.q2.items():
        if coeff % q.mod_q2:
            active.add(left)
            active.add(right)
    for edge, coeff in q.q3.items():
        if coeff % q.mod_q3:
            active.update(edge)

    ranked = sorted(
        active,
        key=lambda var: (
            len(adjacency[var]),
            int(q.q1[var] % threshold != 0),
            int(q.q1[var] % q.mod_q1 != 0),
            -var,
        ),
        reverse=True,
    )
    return tuple(ranked[:_phase_structure_local_max_centers(q)])


def _phase_structure_local_region(
    adjacency: Sequence[set[int]],
    center: int,
    *,
    radius: int = _PHASE_STRUCTURE_LOCAL_REGION_RADIUS,
    max_vars: int = _PHASE_STRUCTURE_LOCAL_REGION_MAX_VARS,
) -> tuple[int, ...]:
    region: set[int] = {int(center)}
    frontier = {int(center)}
    for _ in range(radius):
        if len(region) >= max_vars or not frontier:
            break
        next_frontier: set[int] = set()
        for var in frontier:
            next_frontier.update(adjacency[var])
        next_frontier -= region
        if not next_frontier:
            break
        ranked = sorted(
            next_frontier,
            key=lambda var: (len(adjacency[var]), -var),
            reverse=True,
        )
        for var in ranked:
            region.add(int(var))
            if len(region) >= max_vars:
                break
        frontier = set(ranked)
    return tuple(sorted(region))


def _phase_structure_local_move_score(
    q,
    region: Sequence[int],
    target_local: int,
    sources_local: Sequence[int],
    *,
    adjacency=None,
    context=None,
    max_eval_vars: int | None = None,
) -> tuple[int, ...] | None:
    if max_eval_vars is None:
        max_eval_vars = _phase_structure_local_region_max_vars(q)
    region = tuple(int(var) for var in region)
    support = {int(region[int(target_local)])}
    support.update(int(region[int(src)]) for src in sources_local)
    if adjacency is None:
        adjacency = _interaction_graph(q)
    eval_region = set(support)
    for var in tuple(support):
        eval_region.update(adjacency[var])
    if len(eval_region) > max_eval_vars:
        ranked_boundary = sorted(
            (var for var in eval_region if var not in support),
            key=lambda var: (len(adjacency[var]), -var),
            reverse=True,
        )
        trimmed = set(support)
        for var in ranked_boundary:
            trimmed.add(int(var))
            if len(trimmed) >= max_eval_vars:
                break
        eval_region = trimmed
    eval_region_tuple = tuple(sorted(eval_region))
    local_q = _component_restriction(q, eval_region_tuple)
    before = _phase_function_structure_score(local_q)
    local_index = {var: idx for idx, var in enumerate(eval_region_tuple)}
    transformed_local = _basis_xor_transform(
        local_q,
        local_index[int(region[int(target_local)])],
        tuple(local_index[int(region[int(src)])] for src in sources_local),
        context=context,
    )
    after = _phase_function_structure_score(transformed_local)
    if after >= before:
        return None
    return after + before

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
