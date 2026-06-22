"""Extracted q3-free treewidth and exact component runtime."""

from __future__ import annotations

import bisect
import cmath
from collections import deque
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

from .neighborhood import (
    _sum_q3_free_via_neighborhood_composed_scaled,
)
from .approx_tensor import (
    _clear_q3_free_approx_diagnostics,
    _get_q3_free_approx_diagnostics,
    _sum_q3_free_approx_tensor_scaled,
)
from ..cubic_arithmetic import CubicFunction, PhaseFunction, detect_factorization
from .._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals
from ..scaling import ScaledAmplitude, ScaledComplex
from ..spec import CircuitSpec, Gate
from ..state import BitSequence, CircuitInput, EchelonCache, ExtendedReductionMode, ReducerInfo, ReductionInfo, SolverConfig, _get_solver_config

_LOCAL_NAMES = {
    '_q3_free_series_reduction_core',
    '_q3_free_treewidth_order',
    '_sum_q3_free_component',
    '_iter_q3_free_cheap_order_hints',
    '_best_cheap_q3_free_order',
    '_cheap_q3_free_work_surrogate',
    '_native_rank_q3_free_cutset_extensions',
    '_sum_q3_free_component_scaled',
    '_gauss_sum_q3_free',
    '_gauss_sum_q3_free_scaled',
    '_fix_variables',
    '_fix_variable',
    '_interaction_graph',
    '_connected_components_on_vertices',
    '_pair_graph_degeneracy',
    '_bfs_layers_on_vertices',
    '_farthest_vertex_on_vertices',
    '_min_fill_order_on_subgraph',
    '_choose_pair_graph_layer_separator',
    '_nested_dissection_pair_order_from_adjacency',
    '_pair_graph_separator_order',
    '_min_fill_cubic_order_uncached',
    '_min_degree_cubic_order_uncached',
    '_min_fill_cubic_order',
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


def _q3_free_series_reduction_core(
    adjacency: Sequence[Sequence[int] | dict[int, int] | set[int]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return exact degree-<=2 peel order plus surviving core vertices."""
    n = len(adjacency)
    working = [set(int(neighbor) for neighbor in neighbors) for neighbors in adjacency]
    active = [True] * n
    pending = deque(
        idx for idx, neighbors in enumerate(working) if len(neighbors) <= 2
    )
    peel_order: list[int] = []

    while pending:
        var = pending.popleft()
        if not active[var]:
            continue
        neighbors = tuple(sorted(working[var]))
        if len(neighbors) > 2:
            continue
        active[var] = False
        peel_order.append(int(var))
        if len(neighbors) == 2:
            left, right = neighbors
            if left != right and active[left] and active[right]:
                working[left].add(right)
                working[right].add(left)
        for neighbor in neighbors:
            if active[neighbor]:
                working[neighbor].discard(var)
                if len(working[neighbor]) <= 2:
                    pending.append(int(neighbor))
        working[var].clear()

    core_vars = tuple(idx for idx, is_active in enumerate(active) if is_active)
    return tuple(peel_order), core_vars

def _q3_free_treewidth_order(q, feedback_size, order_hint=None, max_degree=None):
    """
    Return a favorable elimination order for a q3-free component, if any.

    The existing q3-free reducer is exponential in the chosen feedback-set
    size. For strip-like families such as deeper QAOA rings, the q2 graph can
    still have small treewidth even when the feedback set grows linearly.
    """
    if feedback_size <= 1:
        return None

    width_limit = _q3_free_treewidth_width_limit()

    if order_hint is not None:
        hint_order = list(order_hint)
        if q.n <= _Q3_FREE_TREEWIDTH_HEURISTIC_MAX_ORDER_VARS:
            try:
                width = _cubic_order_width(q, hint_order)
            except ValueError:
                width = width_limit + 1
            if (
                width <= min(_Q3_FREE_ORDER_HINT_MAX_WIDTH, width_limit)
                and _q3_free_treewidth_candidate_is_viable(q, hint_order, width, feedback_size)
            ):
                return hint_order

    adjacency = None
    degeneracy_lower_bound = None
    if q.q2:
        adjacency = [set() for _ in range(q.n)]
        for left, right in q.q2:
            adjacency[left].add(right)
            adjacency[right].add(left)
        degeneracy_lower_bound = _pair_graph_degeneracy(adjacency)
        if (
            degeneracy_lower_bound > width_limit
            or degeneracy_lower_bound >= feedback_size
        ):
            return None

    if not q.q3 and adjacency is not None and q.n > 3:
        peel_order, core_vars = _q3_free_series_reduction_core(adjacency)
        if not core_vars:
            if _q3_free_treewidth_candidate_is_viable(q, peel_order, 3, feedback_size):
                return list(peel_order)
            return None
        if len(core_vars) < q.n:
            core_q = _component_restriction(q, core_vars)
            core_adjacency = [set() for _ in range(core_q.n)]
            for left, right in core_q.q2:
                core_adjacency[left].add(right)
                core_adjacency[right].add(left)
            core_hint = None
            if order_hint is not None:
                core_remap = {var: idx for idx, var in enumerate(core_vars)}
                filtered_hint = [core_remap[var] for var in order_hint if var in core_remap]
                if len(filtered_hint) == core_q.n:
                    core_hint = filtered_hint
            core_order = _q3_free_treewidth_order(
                core_q,
                min(feedback_size, max(2, core_q.n)),
                order_hint=core_hint,
                max_degree=max((len(neighbors) for neighbors in core_adjacency), default=0),
            )
            if core_order is not None:
                lifted_order = list(peel_order)
                lifted_order.extend(int(core_vars[idx]) for idx in core_order)
                lifted_width = _cubic_order_width(q, lifted_order)
                if _q3_free_treewidth_candidate_is_viable(
                    q,
                    lifted_order,
                    lifted_width,
                    feedback_size,
                ):
                    return lifted_order

    if q.n > _Q3_FREE_TREEWIDTH_HEURISTIC_MAX_ORDER_VARS:
        return None

    if (
        not q.q3
        and q.n >= _Q3_FREE_CHEAP_ORDER_HINT_MIN_VARS
        and max_degree is not None
        and max_degree <= 4
    ):
        for cheap_order in _iter_q3_free_cheap_order_hints(q.n, q=q):
            try:
                width = _cubic_order_width(q, cheap_order)
            except ValueError:
                continue
            if (
                width <= width_limit
                and _q3_free_treewidth_candidate_is_viable(q, cheap_order, width, feedback_size)
            ):
                return cheap_order

    if max_degree is not None and max_degree <= 4:
        # On sparse strip-like q3-free graphs, min-degree usually lands within
        # one bucket of min-fill at a fraction of the planning cost. This keeps
        # deeper ring-QAOA style amplitudes from spending most of their time on
        # order search when the eventual treewidth DP is already cheap.
        order, width = _min_degree_cubic_order_uncached(q)
        if _q3_free_treewidth_candidate_is_viable(q, order, width, feedback_size):
            return order

    order, width = _min_fill_cubic_order(q)
    if _q3_free_treewidth_candidate_is_viable(q, order, width, feedback_size):
        return order
    separator_order = _pair_graph_separator_order(q)
    if separator_order is not None:
        order, width = separator_order
        if _q3_free_treewidth_candidate_is_viable(q, order, width, feedback_size):
            return order
    return None

def _sum_q3_free_component(
    q,
    *,
    allow_schur_complement: bool = True,
    allow_tensor_contraction: bool = True,
):
    """Sum a connected q3-free component by exact backends on its q2 graph."""
    scaled_total = _sum_q3_free_component_scaled(
        q,
        allow_schur_complement=allow_schur_complement,
        allow_tensor_contraction=allow_tensor_contraction,
    )
    if scaled_total is None:
        raise RuntimeError("No viable exact q3-free backend for component.")
    return _scaled_to_complex(scaled_total)

def _iter_q3_free_cheap_order_hints(
    n_vars: int,
    *,
    q: PhaseFunction | None = None,
) -> tuple[list[int], ...]:
    """Return cheap structural order hints for giant low-degree q3-free kernels."""
    engine = importlib.import_module("terket._engine_impl")
    if _FORCE_ENGINE_BINDINGS_REFRESH:
        _sync_from_engine(engine)
    forward = list(range(int(n_vars)))
    reverse = list(range(int(n_vars) - 1, -1, -1))
    hints: list[list[int]] = [forward, reverse]
    if q is not None and q.n <= _Q3_FREE_TREEWIDTH_HEURISTIC_MAX_ORDER_VARS:
        separator_order = engine._pair_graph_separator_order(q)
        if separator_order is not None:
            order, _width = separator_order
            order_key = tuple(int(var) for var in order)
            if (
                len(order_key) == int(n_vars)
                and len(set(order_key)) == int(n_vars)
                and order_key not in {tuple(forward), tuple(reverse)}
            ):
                hints.append(list(order_key))
    return tuple(hints)

def _best_cheap_q3_free_order(
    q: PhaseFunction,
    *,
    order_hint: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], int]:
    """Return lowest-width cheap order available for a q3-free kernel."""
    engine = importlib.import_module("terket._engine_impl")
    if _FORCE_ENGINE_BINDINGS_REFRESH:
        _sync_from_engine(engine)
    if q.n > _Q3_FREE_TREEWIDTH_HEURISTIC_MAX_ORDER_VARS:
        degrees = [0] * q.n
        for left, right in q.q2:
            degrees[int(left)] += 1
            degrees[int(right)] += 1
        for i, j, k in q.q3:
            degrees[int(i)] += 2
            degrees[int(j)] += 2
            degrees[int(k)] += 2
        return tuple(range(q.n)), max(1, max(degrees, default=0) + 1)
    candidate_orders: list[tuple[int, ...]] = []
    if order_hint is not None:
        hint_order = tuple(int(var) for var in order_hint)
        if len(hint_order) == q.n and len(set(hint_order)) == q.n:
            candidate_orders.append(hint_order)
    candidate_orders.extend(
        tuple(order) for order in _iter_q3_free_cheap_order_hints(q.n, q=q)
    )
    if not candidate_orders:
        raise ValueError("Expected at least one cheap q3-free order candidate.")

    best_order = candidate_orders[0]
    best_width = engine._cubic_order_width(q, best_order)
    for order in candidate_orders[1:]:
        width = engine._cubic_order_width(q, order)
        if width < best_width:
            best_order = order
            best_width = width
    return tuple(best_order), int(best_width)

def _cheap_q3_free_work_surrogate(q: PhaseFunction, width: int) -> int:
    """
    Cheap structural proxy for q3-free DP work on giant surrogate searches.

    One-shot giant-kernel ranking already uses surrogate widths and skips exact
    residual planning. Keep work scoring equally cheap so candidate evaluation
    does not spend most of its time in exact bucket-work estimation.
    """
    effective_width = max(0, int(width))
    scope = min(effective_width + 1, 62)
    structural_mass = max(1, int(q.n) + int(len(q.q2)))
    return max(1, structural_mass * (1 << scope))

def _native_rank_q3_free_cutset_extensions(
    q: PhaseFunction,
    *,
    selected_vars: Sequence[int],
    candidate_vars: Sequence[int],
    remaining_order_hint: Sequence[int] | None = None,
) -> tuple[tuple[int, int, int], ...] | None:
    native_rank = _native_symbol("rank_q3_free_cutset_extensions")
    if native_rank is None or q.q3:
        return None
    try:
        ranked = native_rank(
            int(q.n),
            q.q2,
            tuple(int(var) for var in selected_vars),
            tuple(int(var) for var in candidate_vars),
            None if remaining_order_hint is None else tuple(int(var) for var in remaining_order_hint),
        )
    except Exception:
        return None
    result: list[tuple[int, int, int]] = []
    for candidate, width, work in ranked:
        result.append((int(candidate), int(width), int(work)))
    return tuple(result)

def _sum_q3_free_component_scaled(
    q,
    *,
    allow_schur_complement: bool = True,
    allow_tensor_contraction: bool = True,
):
    """Scaled-complex companion to ``_sum_q3_free_component``."""
    if q.n == 0:
        return _ONE_SCALED
    if q.n <= _Q3_FREE_BRUTE_FORCE_CUTOFF:
        return _make_scaled_complex(_bruteforce_q3_free_sum(q))
    if not q.q2:
        return _product_q1_sum_scaled(q.q1, level=q.level)
    component_sets = detect_factorization(q)
    if len(component_sets) > 1:
        covered = set().union(*component_sets)
        total = _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q.q0)))
        if len(covered) < q.n:
            total = _scale_scaled_complex(total, 2 * (q.n - len(covered)))
        for component in component_sets:
            component_total = _sum_q3_free_component_scaled(
                _component_restriction(q, component),
                allow_schur_complement=allow_schur_complement,
                allow_tensor_contraction=allow_tensor_contraction,
            )
            if component_total is None:
                return None
            total = _mul_scaled_complex(total, component_total)
        return total
    neighborhood_total = _sum_q3_free_via_neighborhood_composed_scaled(q)
    if neighborhood_total is not None:
        return neighborhood_total
    if allow_tensor_contraction and _get_solver_config().approx_q3_free_tensor:
        binary_total = _sum_binary_phase_quadratic_scaled(q)
        if binary_total is not None:
            return binary_total
        approx_total = _sum_q3_free_approx_tensor_scaled(q)
        if approx_total is not None:
            return approx_total
    gauss_reduced_total = _sum_q3_free_via_gauss_reduction_scaled(q)
    if gauss_reduced_total is not None:
        return gauss_reduced_total
    block_cut_plan = _build_block_cut_tree_region_plan(q)
    if block_cut_plan is not None:
        return _evaluate_half_phase_cluster_plan_scaled(
            block_cut_plan,
            q.q1,
        )
    binary_total = _sum_binary_phase_quadratic_scaled(q)
    if binary_total is not None:
        return binary_total

    adjacency, edges = _q3_free_graph(q)
    depth, chords = _q3_free_spanning_data(adjacency, edges)
    if not chords:
        return _forest_transfer_sum_scaled(q.q1, adjacency, level=q.level)

    feedback_vars = _select_feedback_vertices(q.n, chords, depth)
    max_degree = max((len(neighbors) for neighbors in adjacency), default=0)
    treewidth_order = _q3_free_treewidth_order(
        q,
        len(feedback_vars),
        max_degree=max_degree,
    )
    if treewidth_order is not None:
        total, _ = _sum_via_treewidth_dp_scaled(q, treewidth_order)
        return total
    prefer_cutset = _q3_free_prefers_locality_preserving_cutset(
        q,
        feedback_size=len(feedback_vars),
        max_degree=max_degree,
        edge_density=_q3_free_edge_density(q),
        allow_tensor_contraction=allow_tensor_contraction,
    )
    if prefer_cutset:
        cutset_conditioned_total = _sum_q3_free_via_cutset_conditioning_scaled(q)
        if cutset_conditioned_total is not None:
            return cutset_conditioned_total
    if (
        allow_schur_complement
        and len(feedback_vars) > _SCHUR_COMPLEMENT_CROSSOVER_FVS
        and _supports_exact_dense_schur(q)
    ):
        schur_total = _schur_complement_q3_free_sum_scaled(q)
        if schur_total is not None:
            return schur_total
    cutset_conditioned_total = None if prefer_cutset else _sum_q3_free_via_cutset_conditioning_scaled(q)
    if cutset_conditioned_total is not None:
        return cutset_conditioned_total
    if allow_tensor_contraction and _get_solver_config().approx_q3_free_tensor:
        approx_total = _sum_q3_free_approx_tensor_scaled(q)
        if approx_total is not None:
            return approx_total
    if len(feedback_vars) > _Q3_FREE_FEEDBACK_FOREST_MAX_BRANCH_VARS:
        return None

    fixed_pos = {var: idx for idx, var in enumerate(feedback_vars)}
    free_vars = [var for var in range(q.n) if var not in fixed_pos]
    free_index = {var: idx for idx, var in enumerate(free_vars)}
    free_adjacency = [dict() for _ in range(len(free_vars))]
    base_q1 = [q.q1[var] for var in free_vars]

    fixed_linear = [
        (1 << bit, q.q1[var])
        for var, bit in fixed_pos.items()
        if q.q1[var]
    ]
    fixed_to_free = []
    fixed_to_fixed = []

    for i, j, phase in edges:
        bit_i = fixed_pos.get(i)
        bit_j = fixed_pos.get(j)
        if bit_i is not None and bit_j is not None:
            fixed_to_fixed.append((((1 << bit_i) | (1 << bit_j)), phase))
            continue
        if bit_i is not None:
            fixed_to_free.append((1 << bit_i, free_index[j], phase))
            continue
        if bit_j is not None:
            fixed_to_free.append((1 << bit_j, free_index[i], phase))
            continue
        a = free_index[i]
        b = free_index[j]
        free_adjacency[a][b] = phase
        free_adjacency[b][a] = phase

    forest_memo = {}
    forest_components = _forest_postorder_components(free_adjacency)
    total = _ZERO_SCALED
    omega_scaled = _omega_scaled_table(q.level)
    for mask in range(1 << len(feedback_vars)):
        q1_shifted = base_q1[:]
        const_phase = 0

        for bitmask, coeff in fixed_linear:
            if mask & bitmask:
                const_phase = (const_phase + coeff) % q.mod_q1
        for bitmask, idx, phase in fixed_to_free:
            if mask & bitmask:
                q1_shifted[idx] = (q1_shifted[idx] + phase) % q.mod_q1
        for bitmask, phase in fixed_to_fixed:
            if (mask & bitmask) == bitmask:
                const_phase = (const_phase + phase) % q.mod_q1

        key = tuple(q1_shifted)
        forest_total = forest_memo.get(key)
        if forest_total is None:
            forest_total = _forest_transfer_sum_scaled(
                q1_shifted,
                free_adjacency,
                level=q.level,
                components=forest_components,
            )
            forest_memo[key] = forest_total
        total = _add_scaled_complex(
            total,
            _mul_scaled_complex(omega_scaled[const_phase % q.mod_q1], forest_total),
        )

    return total

def _gauss_sum_q3_free(q, *, allow_tensor_contraction: bool = True):
    scaled_total, phase_info = _gauss_sum_q3_free_scaled(
        q,
        allow_tensor_contraction=allow_tensor_contraction,
    )
    return _scaled_to_complex(scaled_total), phase_info

def _gauss_sum_q3_free_scaled(q, *, allow_tensor_contraction: bool = True):
    """Scaled-complex companion to ``_gauss_sum_q3_free``."""
    assert not q.q3, "This function requires a q3-free kernel."
    _clear_q3_free_approx_diagnostics()

    neighborhood_total = _sum_q3_free_via_neighborhood_composed_scaled(q)
    if neighborhood_total is not None:
        return neighborhood_total, {
            'phase_states': 0,
            'phase_splits': 0,
        }

    optimized_q, _changed = _optimize_q3_free_phase(
        q,
        allow_tensor_contraction=allow_tensor_contraction,
    )
    execution_plan = _build_q3_free_execution_plan(
        q=optimized_q,
        allow_tensor_contraction=allow_tensor_contraction,
    )
    total = _evaluate_q3_free_execution_plan_scaled(execution_plan)
    phase_info = {
        'phase_states': 0,
        'phase_splits': 0,
    }
    approx_info = _get_q3_free_approx_diagnostics()
    if approx_info is not None:
        phase_info.update(approx_info)
    return total, phase_info

def _fix_variables(q, fixed_vars, fixed_values, context=None):
    """Fix multiple variables at once and restrict to the remaining free ones."""
    assert len(fixed_vars) == len(fixed_values)
    if not fixed_vars:
        return q

    fixed_pairs = tuple(sorted((var, value % 2) for var, value in zip(fixed_vars, fixed_values)))
    fixed = dict(fixed_pairs)
    assert len(fixed) == len(fixed_pairs)

    if context is not None:
        cache_key = (_q_key(q), fixed_pairs)
        cached = context.fix_variables_cache.get(cache_key)
        if cached is not None:
            return cached

    nf = q.n
    fixed_var_tuple = tuple(int(var) for var, _value in fixed_pairs)
    template_cache_key = (_q_key(q), fixed_var_tuple)
    template = _STRUCTURE_FIX_VARIABLE_TEMPLATE_CACHE.get(template_cache_key)
    if template is None:
        nn = nf - len(fixed)
        gamma = [0] * nf
        free_idx = 0
        for j in range(nf):
            if j in fixed:
                continue
            gamma[j] = 1 << free_idx
            free_idx += 1
        template = (nn, tuple(gamma))
        _STRUCTURE_FIX_VARIABLE_TEMPLATE_CACHE[template_cache_key] = template
    else:
        nn, gamma = template

    shift_mask = 0
    for j, value in fixed_pairs:
        if value:
            shift_mask |= 1 << j
    reduced = _aff_compose_cached(q, shift_mask, list(gamma), nn, context=context)
    if context is not None:
        context.fix_variables_cache[cache_key] = reduced
        return reduced
    return reduced

def _fix_variable(q, k, val, context=None):
    """
    Fix variable k to value val in {0,1}.
    Returns CubicFunction on (q.n - 1) variables.
    """
    return _fix_variables(q, [k], [val], context=context)

def _interaction_graph(q):
    """Primal interaction graph induced by q2 edges and q3 hyperedges."""
    cache_key = _q_phase3_structure_key(q)
    cached = _STRUCTURE_INTERACTION_GRAPH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    adjacency = [set() for _ in range(q.n)]
    for i, j in q.q2:
        adjacency[i].add(j)
        adjacency[j].add(i)
    for i, j, k in q.q3:
        adjacency[i].update([j, k])
        adjacency[j].update([i, k])
        adjacency[k].update([i, j])
    cached = tuple(frozenset(neighbors) for neighbors in adjacency)
    _STRUCTURE_INTERACTION_GRAPH_CACHE[cache_key] = cached
    return cached

def _connected_components_on_vertices(
    adjacency: Sequence[set[int]],
    vertices: Sequence[int],
) -> list[tuple[int, ...]]:
    remaining = set(int(vertex) for vertex in vertices)
    components: list[tuple[int, ...]] = []

    while remaining:
        root = remaining.pop()
        stack = [root]
        component = [root]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(tuple(sorted(component)))

    return components

def _pair_graph_degeneracy(adjacency: Sequence[Sequence[int] | dict[int, int] | set[int]]) -> int:
    """Return the exact degeneracy of an undirected pair graph."""
    n = len(adjacency)
    if n == 0:
        return 0

    degrees = [len(neighbors) for neighbors in adjacency]
    heap = [(degree, idx) for idx, degree in enumerate(degrees)]
    heapq.heapify(heap)
    active = [True] * n
    degeneracy = 0

    while heap:
        degree, idx = heapq.heappop(heap)
        if not active[idx] or degree != degrees[idx]:
            continue
        active[idx] = False
        degeneracy = max(degeneracy, degree)
        for neighbor in adjacency[idx]:
            if active[neighbor]:
                degrees[neighbor] -= 1
                heapq.heappush(heap, (degrees[neighbor], neighbor))
    return degeneracy

def _bfs_layers_on_vertices(
    adjacency: Sequence[set[int]],
    start: int,
    vertices: set[int],
) -> tuple[tuple[tuple[int, ...], ...], dict[int, int]]:
    visited = {int(start)}
    current_layer = [int(start)]
    layers: list[tuple[int, ...]] = []
    distances = {int(start): 0}
    depth = 0

    while current_layer:
        layers.append(tuple(sorted(current_layer)))
        next_layer: list[int] = []
        for current in current_layer:
            for neighbor in adjacency[current]:
                if neighbor not in vertices or neighbor in visited:
                    continue
                visited.add(neighbor)
                distances[neighbor] = depth + 1
                next_layer.append(neighbor)
        current_layer = next_layer
        depth += 1

    return tuple(layers), distances

def _farthest_vertex_on_vertices(
    adjacency: Sequence[set[int]],
    start: int,
    vertices: set[int],
) -> int:
    layers, _ = _bfs_layers_on_vertices(adjacency, start, vertices)
    if not layers:
        return int(start)
    return max(layers[-1])

def _min_fill_order_on_subgraph(
    adjacency: Sequence[set[int]],
    vertices: Sequence[int],
) -> list[int]:
    ordered_vertices = tuple(sorted(int(vertex) for vertex in vertices))
    if not ordered_vertices:
        return []
    remap = {vertex: idx for idx, vertex in enumerate(ordered_vertices)}
    q2 = {
        (remap[left], remap[right]): 1
        for left in ordered_vertices
        for right in adjacency[left]
        if left < right and right in remap
    }
    dummy_q = _phase_function_from_parts(
        len(ordered_vertices),
        level=3,
        q0=Fraction(0),
        q1=[0] * len(ordered_vertices),
        q2=q2,
        q3={},
    )
    order, _ = _min_fill_cubic_order(dummy_q)
    return [ordered_vertices[idx] for idx in order]

def _choose_pair_graph_layer_separator(
    adjacency: Sequence[set[int]],
    vertices: Sequence[int],
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]] | None:
    ordered_vertices = tuple(sorted(int(vertex) for vertex in vertices))
    if len(ordered_vertices) < _Q2_SEPARATOR_ORDER_MIN_VARS:
        return None

    vertex_set = set(ordered_vertices)
    seed = min(ordered_vertices, key=lambda vertex: (len(adjacency[vertex]), vertex))
    left = _farthest_vertex_on_vertices(adjacency, seed, vertex_set)
    right = _farthest_vertex_on_vertices(adjacency, left, vertex_set)
    layers, _ = _bfs_layers_on_vertices(adjacency, left, vertex_set)
    if len(layers) < 3:
        return None

    prefix_sizes = [0]
    for layer in layers:
        prefix_sizes.append(prefix_sizes[-1] + len(layer))

    best: tuple[tuple[float, int, int, int], tuple[int, ...], tuple[tuple[int, ...], ...]] | None = None
    total_size = len(ordered_vertices)
    max_separator_size = min(_Q2_SEPARATOR_ORDER_MAX_SEPARATOR, max(8, total_size // 3))

    for span in range(1, _Q2_SEPARATOR_ORDER_MAX_LAYER_SPAN + 1):
        for start_idx in range(1, len(layers) - span):
            stop_idx = start_idx + span
            separator = tuple(sorted(vertex for layer in layers[start_idx:stop_idx] for vertex in layer))
            if not separator or len(separator) > max_separator_size:
                continue
            separator_set = set(separator)
            remaining = tuple(vertex for vertex in ordered_vertices if vertex not in separator_set)
            components = _connected_components_on_vertices(adjacency, remaining)
            if len(components) < 2:
                continue
            largest = max(len(component) for component in components)
            if largest >= total_size:
                continue
            balance = largest / total_size
            if balance > _Q2_SEPARATOR_ORDER_MAX_BALANCE:
                continue
            left_size = prefix_sizes[start_idx]
            right_size = total_size - prefix_sizes[stop_idx]
            score = (
                balance,
                len(separator),
                abs(left_size - right_size),
                max(len(component) for component in components),
            )
            candidate = (score, separator, tuple(sorted(components, key=lambda component: (len(component), component))))
            if best is None or candidate[0] < best[0]:
                best = candidate

    if best is None:
        return None
    return best[1], best[2]

def _nested_dissection_pair_order_from_adjacency(
    adjacency: Sequence[set[int]],
    vertices: Sequence[int],
) -> list[int]:
    ordered_vertices = tuple(sorted(int(vertex) for vertex in vertices))
    if len(ordered_vertices) <= _Q2_SEPARATOR_ORDER_BASE_CASE:
        return _min_fill_order_on_subgraph(adjacency, ordered_vertices)

    separator_info = _choose_pair_graph_layer_separator(adjacency, ordered_vertices)
    if separator_info is None:
        return _min_fill_order_on_subgraph(adjacency, ordered_vertices)

    separator, components = separator_info
    order: list[int] = []
    for component in components:
        order.extend(_nested_dissection_pair_order_from_adjacency(adjacency, component))
    order.extend(separator)
    return order

def _pair_graph_separator_order(q) -> tuple[list[int], int] | None:
    if q.q3 or len(q.q2) == 0 or q.n < _Q2_SEPARATOR_ORDER_MIN_VARS:
        return None

    adjacency = [set() for _ in range(q.n)]
    for left, right in q.q2:
        adjacency[left].add(right)
        adjacency[right].add(left)

    components = _connected_components_on_vertices(adjacency, range(q.n))
    order: list[int] = []
    for component in components:
        order.extend(_nested_dissection_pair_order_from_adjacency(adjacency, component))
    if len(order) != q.n or len(set(order)) != q.n:
        return None
    width = _cubic_order_width(q, order)
    return order, width

def _min_fill_cubic_order_uncached(q):
    """
    Heuristic elimination order for low-treewidth cubic DP.

    Returns the order and the maximum factor scope size encountered by the
    corresponding variable elimination schedule.
    """
    if q.n == 0:
        return [], 0
    native_min_fill_cubic_order = _native_symbol("min_fill_cubic_order")
    if native_min_fill_cubic_order is not None:
        try:
            return native_min_fill_cubic_order(q.n, q.q2, q.q3)
        except MemoryError:
            pass

    adjacency_masks = [0] * q.n
    for i, j in q.q2:
        bit_i = 1 << i
        bit_j = 1 << j
        adjacency_masks[i] |= bit_j
        adjacency_masks[j] |= bit_i
    for i, j, k in q.q3:
        bit_i = 1 << i
        bit_j = 1 << j
        bit_k = 1 << k
        adjacency_masks[i] |= bit_j | bit_k
        adjacency_masks[j] |= bit_i | bit_k
        adjacency_masks[k] |= bit_i | bit_j

    remaining_mask = (1 << q.n) - 1
    order = []
    max_scope = 1

    while remaining_mask:
        best_var = -1
        best_score = None
        best_neighbors_mask = 0
        for var in _iter_mask_bits(remaining_mask):
            neighbors_mask = adjacency_masks[var] & remaining_mask
            remaining_neighbors = neighbors_mask
            fill = 0
            while remaining_neighbors:
                left_bit = remaining_neighbors & -remaining_neighbors
                left = left_bit.bit_length() - 1
                remaining_neighbors ^= left_bit
                fill += (remaining_neighbors & ~adjacency_masks[left]).bit_count()
            score = (fill, neighbors_mask.bit_count(), var)
            if best_score is None or score < best_score:
                best_var = var
                best_score = score
                best_neighbors_mask = neighbors_mask

        order.append(best_var)
        max_scope = max(max_scope, best_neighbors_mask.bit_count() + 1)

        neighbor_bits = tuple(_iter_mask_bits(best_neighbors_mask))
        remove_var_mask = ~(1 << best_var)
        for left in neighbor_bits:
            adjacency_masks[left] = (
                adjacency_masks[left]
                | (best_neighbors_mask & ~(1 << left))
            ) & remove_var_mask
        adjacency_masks[best_var] = 0
        remaining_mask &= remove_var_mask

    return order, max_scope

def _min_degree_cubic_order_uncached(q):
    """Cheap elimination order based only on the current graph degree."""
    if q.n == 0:
        return [], 0
    native_min_degree_cubic_order = _native_symbol("min_degree_cubic_order")
    if native_min_degree_cubic_order is not None:
        try:
            return native_min_degree_cubic_order(q.n, q.q2, q.q3)
        except MemoryError:
            pass

    adjacency_masks = [0] * q.n
    for i, j in q.q2:
        bit_i = 1 << i
        bit_j = 1 << j
        adjacency_masks[i] |= bit_j
        adjacency_masks[j] |= bit_i
    for i, j, k in q.q3:
        bit_i = 1 << i
        bit_j = 1 << j
        bit_k = 1 << k
        adjacency_masks[i] |= bit_j | bit_k
        adjacency_masks[j] |= bit_i | bit_k
        adjacency_masks[k] |= bit_i | bit_j

    remaining_mask = (1 << q.n) - 1
    order = []
    max_scope = 1

    while remaining_mask:
        best_var = -1
        best_degree = None
        best_neighbors_mask = 0
        for var in _iter_mask_bits(remaining_mask):
            neighbors_mask = adjacency_masks[var] & remaining_mask
            degree = neighbors_mask.bit_count()
            if best_degree is None or degree < best_degree or (degree == best_degree and var < best_var):
                best_var = var
                best_degree = degree
                best_neighbors_mask = neighbors_mask

        order.append(best_var)
        max_scope = max(max_scope, best_neighbors_mask.bit_count() + 1)

        neighbor_bits = tuple(_iter_mask_bits(best_neighbors_mask))
        remove_var_mask = ~(1 << best_var)
        for left in neighbor_bits:
            adjacency_masks[left] = (
                adjacency_masks[left]
                | (best_neighbors_mask & ~(1 << left))
            ) & remove_var_mask
        adjacency_masks[best_var] = 0
        remaining_mask &= remove_var_mask

    return order, max_scope

def _min_fill_cubic_order(q):
    cache_key = _q_structure_key(q)
    cached = _STRUCTURE_MIN_FILL_CACHE.get(cache_key)
    if cached is not None:
        order, width = cached
        return list(order), width
    order, width = _min_fill_cubic_order_uncached(q)
    cached = (tuple(order), width)
    _STRUCTURE_MIN_FILL_CACHE[cache_key] = cached
    order, width = cached
    return list(order), width

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
