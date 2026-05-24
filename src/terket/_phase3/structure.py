"""Extracted phase-3 structure optimization helpers."""

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
    '_optimize_phase_function_structure_locally',
    '_basis_xor_transform',
    '_phase_function_basis_candidate_variables',
    '_phase_function_basis_transform_candidates',
    '_optimize_phase_function_structure',
    '_simplify_q3_basis',
    '_projected_components_after_fixing',
    '_find_small_q3_separator',
    '_q3_core_cover_size',
    '_estimate_q3_cover_work',
    '_estimate_q3_separator_work',
    '_phase3_treewidth_cutset_width_limit',
    '_phase3_residual_after_cutset',
    '_phase3_cutset_worst_residual',
    '_phase3_treewidth_cutset_candidates',
    '_find_q3_treewidth_cutset',
    '_estimate_q3_treewidth_cutset_work',
    '_prefer_treewidth_phase3'
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


def _optimize_phase_function_structure_locally(q, context=None):
    """Search exact XOR basis moves only inside dense/hard local neighborhoods."""
    current = q
    changed = False
    current_score = _phase_function_structure_score(current)

    for _ in range(_phase_structure_local_max_passes(current)):
        adjacency = _interaction_graph(current)
        region_max_vars = _phase_structure_local_region_max_vars(current)
        regions: list[tuple[int, ...]] = []
        seen_regions: set[tuple[int, ...]] = set()
        for center in _phase_structure_hotspot_centers(current):
            region = _phase_structure_local_region(
                adjacency,
                center,
                max_vars=region_max_vars,
            )
            if len(region) <= 1 or region in seen_regions:
                continue
            seen_regions.add(region)
            regions.append(region)

        candidate_moves: dict[tuple[int, tuple[int, ...]], tuple[int, ...]] = {}

        for region in regions:
            local_q = _component_restriction(current, region)
            for target_local, sources_local in _phase_function_basis_transform_candidates(local_q):
                score = _phase_structure_local_move_score(
                    current,
                    region,
                    target_local,
                    sources_local,
                    adjacency=adjacency,
                    context=context,
                    max_eval_vars=region_max_vars,
                )
                if score is None:
                    continue
                global_move = (
                    int(region[target_local]),
                    tuple(int(region[src]) for src in sources_local),
                )
                existing = candidate_moves.get(global_move)
                if existing is None or score < existing:
                    candidate_moves[global_move] = score

        if not candidate_moves:
            break

        ranked_candidates = sorted(
            candidate_moves.items(),
            key=lambda item: item[1],
        )[:_phase_structure_local_candidate_pool(current)]
        best_global_q = None
        best_global_score = current_score
        for (target, sources), _local_score in ranked_candidates:
            candidate_q = _basis_xor_transform(current, target, sources, context=context)
            candidate_score = _phase_function_structure_score(candidate_q)
            if candidate_score < best_global_score:
                best_global_q = candidate_q
                best_global_score = candidate_score

        if best_global_q is None:
            break
        current = best_global_q
        current_score = best_global_score
        changed = True

    return current, changed


def _basis_xor_transform(q, target: int, sources: Sequence[int], context=None):
    gamma = [1 << idx for idx in range(q.n)]
    mask = 1 << target
    for source in sources:
        if source == target:
            raise ValueError("Basis XOR transforms require distinct source variables.")
        mask |= 1 << source
    gamma[target] = mask
    return _aff_compose_cached(q, 0, gamma, q.n, context=context)


def _phase_function_basis_candidate_variables(q) -> tuple[int, ...]:
    active: set[int]
    if q.q3:
        active = set(_active_q3_variables(q))
    else:
        active = {
            idx
            for idx, coeff in enumerate(q.q1)
            if int(coeff) % q.mod_q1
        }
        for (left, right), coeff in q.q2.items():
            if coeff % q.mod_q2:
                active.add(left)
                active.add(right)

    if len(active) <= 1:
        return ()

    q3_degree = {var: 0 for var in active}
    q2_degree = {var: 0 for var in active}
    for edge, coeff in q.q3.items():
        if coeff % q.mod_q3 == 0:
            continue
        for var in edge:
            if var in q3_degree:
                q3_degree[var] += 1
    adjacency = [set() for _ in range(q.n)]
    for (left, right), coeff in q.q2.items():
        if coeff % q.mod_q2 == 0:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
        if left in q2_degree:
            q2_degree[left] += 1
        if right in q2_degree:
            q2_degree[right] += 1

    threshold = max(1, q.mod_q1 // 4)
    ranked = sorted(
        active,
        key=lambda var: (
            q3_degree[var],
            q2_degree[var],
            int(q.q1[var] % threshold != 0),
            int(q.q1[var] % q.mod_q1 != 0),
            len(adjacency[var] & active),
            -var,
        ),
        reverse=True,
    )
    return tuple(ranked[:_phase_structure_opt_active_vars(q)])


def _phase_function_basis_transform_candidates(q) -> list[tuple[int, tuple[int, ...]]]:
    candidate_vars = _phase_function_basis_candidate_variables(q)
    if len(candidate_vars) <= 1:
        return []

    q3_degree = {var: 0 for var in candidate_vars}
    q2_degree = {var: 0 for var in candidate_vars}
    adjacency = [set() for _ in range(q.n)]
    for edge, coeff in q.q3.items():
        if coeff % q.mod_q3 == 0:
            continue
        for var in edge:
            if var in q3_degree:
                q3_degree[var] += 1
    for (left, right), coeff in q.q2.items():
        if coeff % q.mod_q2 == 0:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
        if left in q2_degree:
            q2_degree[left] += 1
        if right in q2_degree:
            q2_degree[right] += 1

    moves: list[tuple[int, tuple[int, ...]]] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for target in candidate_vars:
        for source in candidate_vars:
            if source == target:
                continue
            move = (target, (source,))
            if move not in seen:
                seen.add(move)
                moves.append(move)

    if q.q3:
        for edge, coeff in q.q3.items():
            if coeff % q.mod_q3 == 0:
                continue
            if not all(var in candidate_vars for var in edge):
                continue
            a, b, c = edge
            for target, sources in (
                (a, (b, c)),
                (b, (a, c)),
                (c, (a, b)),
            ):
                move = (target, tuple(sorted(sources)))
                if move not in seen:
                    seen.add(move)
                    moves.append(move)
    return moves


def _optimize_phase_function_structure(q, context=None):
    """Beam-search exact XOR basis changes scored for TerKet's solver."""
    cache_key = _q_key(q)
    cached = _STRUCTURE_PHASE3_OPT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if q.n > _phase_structure_opt_max_vars(q):
        cached = _optimize_phase_function_structure_locally(q, context=context)
        _STRUCTURE_PHASE3_OPT_CACHE[cache_key] = cached
        return cached
    if not q.q2 and not q.q3:
        cached = (q, False)
        _STRUCTURE_PHASE3_OPT_CACHE[cache_key] = cached
        return cached

    baseline_score = _phase_function_structure_score(q)
    best_q = q
    best_score = baseline_score
    changed = False
    beam: list[tuple[tuple[int, ...], PhaseFunction]] = [(baseline_score, q)]

    for _ in range(_phase_structure_opt_max_passes(q)):
        pool: dict[tuple[Fraction, tuple[int, ...], tuple[tuple[int, int], int], tuple[tuple[int, int, int], int]], tuple[tuple[int, ...], PhaseFunction]] = {
            _q_key(best_q): (best_score, best_q)
        }
        for _score, candidate_q in beam:
            for target, sources in _phase_function_basis_transform_candidates(candidate_q):
                transformed = _basis_xor_transform(candidate_q, target, sources, context=context)
                key = _q_key(transformed)
                score = _phase_function_structure_score(transformed)
                existing = pool.get(key)
                if existing is None or score < existing[0]:
                    pool[key] = (score, transformed)
                if score < best_score:
                    best_q = transformed
                    best_score = score
                    changed = True

        ranked = sorted(pool.values(), key=lambda item: item[0])
        if not ranked or ranked[0][0] >= beam[0][0]:
            break
        beam = ranked[:_phase_structure_opt_beam_width(q)]

    cached = (best_q, changed)
    _STRUCTURE_PHASE3_OPT_CACHE[cache_key] = cached
    return cached


def _simplify_q3_basis(q, context=None):
    """Backward-compatible wrapper around the general structural optimizer."""
    if not q.q3:
        return q, False
    return _optimize_phase_function_structure(q, context=context)


def _projected_components_after_fixing(q, separator: Sequence[int]) -> list[set[int]]:
    """Connected components guaranteed to survive after fixing ``separator``."""
    removed = set(separator)
    adjacency = [set() for _ in range(q.n)]
    active_vars: set[int] = {
        idx for idx, coeff in enumerate(q.q1) if coeff % q.mod_q1 and idx not in removed
    }

    for (i, j), coeff in q.q2.items():
        if coeff % q.mod_q2 == 0 or i in removed or j in removed:
            continue
        adjacency[i].add(j)
        adjacency[j].add(i)
        active_vars.update((i, j))

    for edge, coeff in q.q3.items():
        if coeff % q.mod_q3 == 0:
            continue
        survivors = [var for var in edge if var not in removed]
        if not survivors:
            continue
        active_vars.update(survivors)
        if len(survivors) < 2:
            continue
        for left in range(len(survivors)):
            for right in range(left + 1, len(survivors)):
                a = survivors[left]
                b = survivors[right]
                adjacency[a].add(b)
                adjacency[b].add(a)

    components: list[set[int]] = []
    visited: set[int] = set()
    for start in sorted(active_vars):
        if start in visited:
            continue
        stack = [start]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            stack.extend(adjacency[node] - visited)
        components.append(component)
    return components


def _find_small_q3_separator(q) -> tuple[int, ...] | None:
    """Return a small separator whose fixing disconnects the residual kernel."""
    cache_key = _q_phase3_structure_key(q)
    cached = _STRUCTURE_Q3_SEPARATOR_CACHE.get(cache_key)
    if cached is not None:
        return None if cached == () else tuple(cached)

    active_q3 = _active_q3_variables(q)
    if len(active_q3) <= 2:
        _STRUCTURE_Q3_SEPARATOR_CACHE[cache_key] = ()
        return None

    q3_degree = {var: 0 for var in active_q3}
    for edge, coeff in q.q3.items():
        if coeff % q.mod_q3 == 0:
            continue
        for var in edge:
            q3_degree[var] += 1

    candidates = sorted(
        active_q3,
        key=lambda var: (q3_degree[var], -var),
        reverse=True,
    )[:_Q3_SEPARATOR_MAX_CANDIDATES]

    best_separator = None
    best_score = None
    max_size = min(_Q3_SEPARATOR_MAX_SIZE, len(candidates) - 1)
    for size in range(1, max_size + 1):
        for separator in combinations(candidates, size):
            components = _projected_components_after_fixing(q, separator)
            if len(components) < 2:
                continue
            largest = max(len(component) for component in components)
            score = (size, largest, -len(components))
            if best_score is None or score < best_score:
                best_separator = tuple(separator)
                best_score = score
        if best_separator is not None:
            break
    _STRUCTURE_Q3_SEPARATOR_CACHE[cache_key] = () if best_separator is None else tuple(best_separator)
    return best_separator


def _q3_core_cover_size(q, core_vars) -> int:
    """Return the exact q3-cover size on the surviving 2-core."""
    if not core_vars:
        return 0
    ordered_core = tuple(sorted(core_vars))
    remap = {var: idx for idx, var in enumerate(ordered_core)}
    core_q3 = {
        tuple(sorted(remap[var] for var in edge)): coeff
        for edge, coeff in q.q3.items()
        if coeff % q.mod_q3 and all(var in remap for var in edge)
    }
    if not core_q3:
        return 0
    core_phase = _phase_function_from_parts(
        len(ordered_core),
        level=q.level,
        q0=Fraction(0),
        q1=[0] * len(ordered_core),
        q2={},
        q3=core_q3,
    )
    return len(_minimum_q3_vertex_cover(core_phase))


def _estimate_q3_cover_work(q, cover_size):
    """Cheap proxy for q3-cover branching work."""
    term_count = sum(1 for coeff in q.q1 if coeff % q.mod_q1) + len(q.q2) + len(q.q3)
    residual_vars = max(1, q.n - cover_size)
    per_leaf_work = residual_vars * max(1, q.n + term_count)
    return (1 << cover_size) * per_leaf_work


def _estimate_q3_separator_work(q, separator: Sequence[int]) -> int:
    """Cheap proxy for separator branching work."""
    if not separator:
        return max(1, q.n)
    components = _projected_components_after_fixing(q, separator)
    if len(components) < 2:
        return max(1, _estimate_q3_cover_work(q, len(separator)))
    term_count = sum(1 for coeff in q.q1 if coeff % q.mod_q1) + len(q.q2) + len(q.q3)
    branch_cost = 0
    for component in components:
        size = len(component)
        branch_cost += max(1, size * max(1, size + term_count))
    return (1 << len(separator)) * max(1, branch_cost)


def _phase3_treewidth_cutset_width_limit(fully_peeled: bool) -> int:
    return (
        _Q3_TREEWIDTH_DP_PEELED_MAX_WIDTH
        if fully_peeled
        else _Q3_TREEWIDTH_DP_MAX_WIDTH
    )


def _phase3_residual_after_cutset(q, cutset: Sequence[int]):
    if not cutset:
        return q, tuple(range(q.n))
    cutset_set = set(int(var) for var in cutset)
    remaining_original = tuple(var for var in range(q.n) if var not in cutset_set)
    residual_q = _fix_variables(q, tuple(sorted(cutset_set)), [1] * len(cutset_set))
    return residual_q, remaining_original


def _phase3_cutset_worst_residual(q, cutset: Sequence[int]) -> tuple[list[int], int, int]:
    """Return the worst residual treewidth/work over every cutset assignment."""
    cutset = tuple(int(var) for var in cutset)
    if not cutset:
        order, width = _min_fill_cubic_order(q)
        if q.q3:
            order, width = _finalize_phase3_treewidth_order(q, order)
        return list(order), int(width), max(1, int(_estimate_treewidth_dp_work(q, order)))

    worst_order: list[int] = []
    worst_width = -1
    worst_work = -1
    for mask in range(1 << len(cutset)):
        fixed_values = [(mask >> idx) & 1 for idx in range(len(cutset))]
        residual_q = _fix_variables(q, cutset, fixed_values)
        residual_order, residual_width = _min_fill_cubic_order(residual_q)
        if residual_q.q3:
            residual_order, residual_width = _finalize_phase3_treewidth_order(
                residual_q,
                residual_order,
            )
        residual_work = max(1, int(_estimate_treewidth_dp_work(residual_q, residual_order)))
        if (int(residual_width), int(residual_work)) > (worst_width, worst_work):
            worst_order = list(residual_order)
            worst_width = int(residual_width)
            worst_work = int(residual_work)

    return worst_order, worst_width, worst_work


def _phase3_treewidth_cutset_candidates(q, residual_q, residual_order, remaining_original):
    scopes = _treewidth_order_scope_sets(residual_q, residual_order)
    hotspot_scopes = sorted(scopes, key=lambda scope: (len(scope), -sum(scope)), reverse=True)[
        :_Q3_TREEWIDTH_CUTSET_MAX_CANDIDATES
    ]
    counts: dict[int, int] = {}
    for scope in hotspot_scopes:
        if len(scope) <= 1:
            continue
        for residual_var in scope:
            original_var = remaining_original[residual_var]
            counts[original_var] = counts.get(original_var, 0) + 1
    if not counts:
        return ()

    adjacency = _interaction_graph(q)
    ranked = sorted(
        counts,
        key=lambda var: (counts[var], len(adjacency[var]), -var),
        reverse=True,
    )
    return tuple(ranked[:_Q3_TREEWIDTH_CUTSET_MAX_CANDIDATES])


def _find_q3_treewidth_cutset(
    q,
    *,
    order: Sequence[int],
    width: int,
    fully_peeled: bool,
) -> tuple[tuple[int, ...], list[int], int, int] | None:
    """Greedily find variables whose conditioning lowers Phase-3 DP width."""
    cache_key = (
        _q_phase3_structure_key(q),
        tuple(int(var) for var in order),
        int(width),
        bool(fully_peeled),
    )
    cached = _STRUCTURE_PHASE3_TREEWIDTH_CUTSET_CACHE.get(cache_key)
    if cached is not None:
        if cached == ():
            return None
        cutset, residual_order, residual_width, residual_work = cached
        return tuple(cutset), list(residual_order), int(residual_width), int(residual_work)

    def cache_result(result):
        if result is None:
            _STRUCTURE_PHASE3_TREEWIDTH_CUTSET_CACHE[cache_key] = ()
            return None
        cutset, residual_order, residual_width, residual_work = result
        cached_result = (
            tuple(int(var) for var in cutset),
            tuple(int(var) for var in residual_order),
            int(residual_width),
            int(residual_work),
        )
        _STRUCTURE_PHASE3_TREEWIDTH_CUTSET_CACHE[cache_key] = cached_result
        return cached_result[0], list(cached_result[1]), cached_result[2], cached_result[3]

    if not q.q3:
        return cache_result(None)
    width_limit = _phase3_treewidth_cutset_width_limit(fully_peeled)
    if width <= width_limit:
        return cache_result(None)

    selected: list[int] = []
    best_residual_order = list(order)
    best_residual_width = int(width)
    best_residual_work = max(1, int(_estimate_treewidth_dp_work(q, order)))

    for _ in range(_Q3_TREEWIDTH_CUTSET_MAX_SIZE):
        residual_q, remaining_original = _phase3_residual_after_cutset(q, selected)
        if not residual_q.q3:
            residual_order, residual_width = _min_fill_cubic_order(residual_q)
        else:
            residual_order, residual_width = _min_fill_cubic_order(residual_q)
            residual_order, residual_width = _finalize_phase3_treewidth_order(
                residual_q,
                residual_order,
            )
        residual_work = max(1, int(_estimate_treewidth_dp_work(residual_q, residual_order)))
        if residual_width < best_residual_width or (
            residual_width == best_residual_width and residual_work < best_residual_work
        ):
            best_residual_order = list(residual_order)
            best_residual_width = int(residual_width)
            best_residual_work = int(residual_work)
        if residual_width <= width_limit and selected:
            worst_order, worst_width, worst_work = _phase3_cutset_worst_residual(q, selected)
            if worst_width <= width_limit:
                return cache_result(
                    (tuple(selected), list(worst_order), int(worst_width), int(worst_work))
                )

        candidates = _phase3_treewidth_cutset_candidates(
            q,
            residual_q,
            residual_order,
            remaining_original,
        )
        if not candidates:
            break

        best_candidate = None
        best_candidate_score = None
        for candidate in candidates:
            if candidate in selected:
                continue
            trial_cutset = selected + [candidate]
            trial_q, _trial_remaining = _phase3_residual_after_cutset(q, trial_cutset)
            trial_order, trial_width = _min_fill_cubic_order(trial_q)
            if trial_q.q3:
                trial_order, trial_width = _finalize_phase3_treewidth_order(trial_q, trial_order)
            trial_work = max(1, int(_estimate_treewidth_dp_work(trial_q, trial_order)))
            score = (int(trial_width), int(trial_work), candidate)
            if best_candidate_score is None or score < best_candidate_score:
                best_candidate = candidate
                best_candidate_score = score
        if best_candidate is None:
            break
        selected.append(best_candidate)

    if selected and best_residual_width <= width_limit:
        worst_order, worst_width, worst_work = _phase3_cutset_worst_residual(q, selected)
        if worst_width <= width_limit:
            return cache_result((tuple(selected), worst_order, worst_width, worst_work))
    return cache_result(None)


def _estimate_q3_treewidth_cutset_work(q, cutset_plan) -> int:
    if cutset_plan is None:
        return 1 << 62
    cutset, _residual_order, _residual_width, residual_work = cutset_plan
    return (1 << len(cutset)) * max(1, int(residual_work))


def _prefer_treewidth_phase3(
    q,
    cover,
    order,
    width,
    *,
    fully_peeled: bool = False,
    treewidth_work: int | None = None,
):
    """Decide whether Phase 3 should use treewidth DP on ``q``."""
    _refresh_engine_bindings()
    width_limit = (
        _Q3_TREEWIDTH_DP_PEELED_MAX_WIDTH
        if fully_peeled
        else _Q3_TREEWIDTH_DP_MAX_WIDTH
    )
    if width > width_limit:
        return False
    if treewidth_work is None:
        treewidth_work = max(1, int(_estimate_treewidth_dp_work(q, order)))
    if fully_peeled and treewidth_work <= _Q3_TREEWIDTH_DP_PEELED_MAX_WORK:
        return True
    if width <= max(1, len(cover)):
        return True
    cover_work = max(1, int(_estimate_q3_cover_work(q, len(cover))))
    return treewidth_work <= cover_work


_CUBIC_CONTRACTION_MAX_WIDTH = 12  # numpy bucket elim beats quimb up to this width
_Q3_COVER_BRANCH_CHUNK_MAX = 128
_Q3_COVER_ASSIGNMENT_CHUNK_LOG2 = 13

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
