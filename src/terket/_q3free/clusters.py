"""Recovered _q3free_clusters.py helpers from monolith worktree."""

from __future__ import annotations

import bisect
import cmath
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import heapq
from itertools import combinations
import math
import os
import struct
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Sequence, overload

import numpy as np

from ..cubic_arithmetic import CubicFunction, PhaseFunction, detect_factorization
from .._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals
from ..scaling import ScaledAmplitude, ScaledComplex
from ..spec import CircuitSpec, Gate
from ..state import BitSequence, CircuitInput, EchelonCache, ExtendedReductionMode, ReducerInfo, ReductionInfo, SolverConfig

_LOCAL_NAMES = {
    '_build_cluster_boundary_shift_table',
    '_build_q2_adjacency',
    '_build_selected_boundary_region_plan',
    '_small_boundary_region_candidates',
    '_articulation_boundary_region_candidates',
    '_q2_block_cut_decomposition',
    '_block_cut_boundary_region_candidates',
    '_build_small_boundary_region_plan',
    '_build_block_cut_tree_region_plan',
    '_build_half_phase_cluster_plan'
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

def _build_cluster_boundary_shift_table(
    *,
    cluster_size: int,
    boundary_size: int,
    boundary_couplings: Sequence[tuple[int, int, int]],
    q2_lift: int,
    mod_q1: int,
) -> np.ndarray:
    shift_table = np.zeros((1 << boundary_size, cluster_size), dtype=np.int64)
    for cluster_idx, boundary_idx, coeff in boundary_couplings:
        active_assignments = ((np.arange(1 << boundary_size, dtype=np.int64) >> int(boundary_idx)) & 1).astype(np.bool_)
        if not np.any(active_assignments):
            continue
        shift_table[active_assignments, int(cluster_idx)] = (
            shift_table[active_assignments, int(cluster_idx)] + (q2_lift * int(coeff))
        ) % mod_q1
    return shift_table

def _build_q2_adjacency(q: PhaseFunction) -> list[set[int]]:
    adjacency = [set() for _ in range(q.n)]
    for (left, right), coeff in q.q2.items():
        if coeff % q.mod_q2:
            adjacency[left].add(right)
            adjacency[right].add(left)
    return adjacency

def _build_selected_boundary_region_plan(
    q: PhaseFunction,
    *,
    adjacency: Sequence[set[int]] | None = None,
    candidate_regions: Sequence[Sequence[int]],
) -> _HalfPhaseClusterPlan | None:
    if q.q3 or not q.q2:
        return None
    adjacency = _build_q2_adjacency(q) if adjacency is None else [set(neighbors) for neighbors in adjacency]

    selected_regions: list[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            dict[tuple[int, int], int],
            tuple[tuple[int, int, int], ...],
        ]
    ] = []
    selected_cluster_vars: set[int] = set()

    scored_regions: list[tuple[tuple[int, int, int], tuple[int, ...]]] = []
    for region in candidate_regions:
        region_vars = tuple(sorted({int(var) for var in region}))
        if not region_vars:
            continue
        boundary_vars = tuple(
            sorted(
                {
                    int(neighbor)
                    for var in region_vars
                    for neighbor in adjacency[var]
                    if neighbor not in region_vars
                }
            )
        )
        if not boundary_vars:
            continue
        scored_regions.append(
            ((len(region_vars), -len(boundary_vars), -region_vars[0]), region_vars)
        )
    scored_regions.sort(reverse=True)

    for _score, region_vars in scored_regions:
        cluster_set = set(region_vars)
        if cluster_set & selected_cluster_vars:
            continue
        boundary_vars = tuple(
            sorted(
                {
                    int(neighbor)
                    for var in region_vars
                    for neighbor in adjacency[var]
                    if neighbor not in cluster_set
                }
            )
        )
        if (
            not boundary_vars
            or set(boundary_vars) & selected_cluster_vars
        ):
            continue
        cluster_remap = {var: idx for idx, var in enumerate(region_vars)}
        boundary_remap = {var: idx for idx, var in enumerate(boundary_vars)}
        boundary_set = set(boundary_vars)
        internal_q2 = {
            (cluster_remap[i], cluster_remap[j]): coeff
            for (i, j), coeff in q.q2.items()
            if i in cluster_set and j in cluster_set
        }
        boundary_couplings: list[tuple[int, int, int]] = []
        for (left, right), coeff in q.q2.items():
            if coeff % q.mod_q2 == 0:
                continue
            if left in cluster_set and right in boundary_set:
                boundary_couplings.append((cluster_remap[left], boundary_remap[right], int(coeff)))
            elif right in cluster_set and left in boundary_set:
                boundary_couplings.append((cluster_remap[right], boundary_remap[left], int(coeff)))
        if not boundary_couplings:
            continue
        selected_regions.append(
            (
                region_vars,
                boundary_vars,
                internal_q2,
                tuple(boundary_couplings),
            )
        )
        selected_cluster_vars.update(cluster_set)
        if len(selected_regions) >= _Q3_FREE_SMALL_BOUNDARY_REGION_MAX_REGIONS:
            break

    if not selected_regions:
        return None

    core_vars = tuple(var for var in range(q.n) if var not in selected_cluster_vars)
    core_remap = {var: idx for idx, var in enumerate(core_vars)}
    core_q2 = {
        (core_remap[i], core_remap[j]): coeff
        for (i, j), coeff in q.q2.items()
        if i in core_remap and j in core_remap
    }
    mod_q1 = 1 << q.level
    mod_q2 = max(1, 1 << (q.level - 1))
    q2_lift = mod_q1 // mod_q2 if mod_q2 else 0

    factor_scopes: list[tuple[int, ...]] = [edge for edge in core_q2]
    cluster_specs: list[_HalfPhaseClusterSpec] = []
    for cluster_vars, boundary_vars, internal_q2, boundary_couplings in selected_regions:
        if not all(var in core_remap for var in boundary_vars):
            return None
        boundary_core = tuple(core_remap[var] for var in boundary_vars)
        factor_scopes.append(boundary_core)
        cluster_order, _cluster_width = _factor_scope_order(
            len(cluster_vars),
            list(internal_q2),
        )
        native_treewidth_plan = _build_native_q3_free_treewidth_plan(
            n_vars=len(cluster_vars),
            level=q.level,
            q2=internal_q2,
            order=cluster_order,
        )
        cluster_specs.append(
            _HalfPhaseClusterSpec(
                cluster_vars=cluster_vars,
                boundary_vars=boundary_core,
                internal_q2=internal_q2,
                boundary_couplings=boundary_couplings,
                boundary_shift_table=_build_cluster_boundary_shift_table(
                    cluster_size=len(cluster_vars),
                    boundary_size=len(boundary_vars),
                    boundary_couplings=boundary_couplings,
                    q2_lift=q2_lift,
                    mod_q1=mod_q1,
                ),
                cluster_order=tuple(cluster_order),
                native_treewidth_plan=native_treewidth_plan,
            )
        )

    width_limit = _q3_free_treewidth_width_limit()
    degeneracy_lower_bound = _factor_scope_degeneracy(len(core_vars), factor_scopes)
    if degeneracy_lower_bound > width_limit:
        return None

    order, width = _factor_scope_order(len(core_vars), factor_scopes)
    if width > width_limit:
        return None

    return _HalfPhaseClusterPlan(
        level=q.level,
        core_vars=core_vars,
        core_q2=core_q2,
        order=tuple(order),
        width=width,
        clusters=tuple(cluster_specs),
    )

def _small_boundary_region_candidates(
    adjacency: Sequence[set[int]],
    *,
    min_region_size: int = _Q3_FREE_SMALL_BOUNDARY_REGION_MIN_SIZE,
    max_region_size: int = _Q3_FREE_SMALL_BOUNDARY_REGION_MAX_SIZE,
    max_boundary: int = _Q3_FREE_SMALL_BOUNDARY_REGION_MAX_BOUNDARY,
) -> tuple[tuple[int, ...], ...]:
    if not adjacency:
        return ()
    leaves = [idx for idx, neighbors in enumerate(adjacency) if len(neighbors) <= 1]
    roots = list(leaves[:2])
    if leaves:
        last_leaf = leaves[-1]
        if last_leaf not in roots:
            roots.append(last_leaf)
    for fallback_root in (0, len(adjacency) - 1):
        if 0 <= fallback_root < len(adjacency) and fallback_root not in roots:
            roots.append(fallback_root)

    candidates: dict[tuple[int, ...], tuple[int, int]] = {}

    for root in roots:
        seen = [False] * len(adjacency)
        parent = [-1] * len(adjacency)
        children = [[] for _ in range(len(adjacency))]
        order: list[int] = []
        stack = [int(root)]
        seen[int(root)] = True
        while stack:
            node = stack.pop()
            order.append(node)
            for neighbor in sorted(adjacency[node], reverse=True):
                if seen[neighbor]:
                    continue
                seen[neighbor] = True
                parent[neighbor] = node
                children[node].append(neighbor)
                stack.append(neighbor)
        for start in range(len(adjacency)):
            if seen[start]:
                continue
            seen[start] = True
            stack = [start]
            while stack:
                node = stack.pop()
                order.append(node)
                for neighbor in sorted(adjacency[node], reverse=True):
                    if seen[neighbor]:
                        continue
                    seen[neighbor] = True
                    parent[neighbor] = node
                    children[node].append(neighbor)
                    stack.append(neighbor)

        tin = [-1] * len(adjacency)
        tout = [-1] * len(adjacency)
        for idx, node in enumerate(order):
            tin[node] = idx
        subtree_sizes = [1] * len(adjacency)
        for node in reversed(order):
            size = 1
            for child in children[node]:
                size += subtree_sizes[child]
            subtree_sizes[node] = size
            tout[node] = tin[node] + size

        for node in order:
            size = subtree_sizes[node]
            if size < int(min_region_size) or size > int(max_region_size):
                continue
            region = tuple(sorted(order[tin[node] : tout[node]]))
            region_set = set(region)
            boundary: set[int] = set()
            valid = True
            for var in region:
                for neighbor in adjacency[var]:
                    if neighbor in region_set:
                        continue
                    boundary.add(int(neighbor))
                    if len(boundary) > int(max_boundary):
                        valid = False
                        break
                if not valid:
                    break
            if not valid or not boundary:
                continue
            score = (len(region), -len(boundary))
            existing = candidates.get(region)
            if existing is None or score > existing:
                candidates[region] = score

    ranked = sorted(
        candidates.items(),
        key=lambda item: (item[1][0], item[1][1], -item[0][0]),
        reverse=True,
    )
    return tuple(region for region, _score in ranked)

def _articulation_boundary_region_candidates(
    adjacency: Sequence[set[int]],
    *,
    min_region_size: int = _Q3_FREE_SMALL_BOUNDARY_REGION_MIN_SIZE,
    max_region_size: int = _Q3_FREE_SMALL_BOUNDARY_REGION_MAX_SIZE,
    max_boundary: int = _Q3_FREE_SMALL_BOUNDARY_REGION_MAX_BOUNDARY,
) -> tuple[tuple[int, ...], ...]:
    if not adjacency:
        return ()

    n = len(adjacency)
    disc = [-1] * n
    low = [0] * n
    parent = [-1] * n
    time = 0
    articulation: set[int] = set()

    for root in range(n):
        if disc[root] != -1:
            continue
        stack: list[tuple[int, int, bool]] = [(root, 0, False)]
        root_children = 0
        while stack:
            node, idx, returning = stack.pop()
            if not returning:
                if disc[node] == -1:
                    disc[node] = time
                    low[node] = time
                    time += 1
                neighbors = sorted(adjacency[node])
                if idx < len(neighbors):
                    neighbor = neighbors[idx]
                    stack.append((node, idx + 1, False))
                    if disc[neighbor] == -1:
                        parent[neighbor] = node
                        if node == root:
                            root_children += 1
                        stack.append((node, neighbor, True))
                        stack.append((neighbor, 0, False))
                    elif neighbor != parent[node]:
                        low[node] = min(low[node], disc[neighbor])
                continue

            child = idx
            low[node] = min(low[node], low[child])
            if parent[node] != -1 and low[child] >= disc[node]:
                articulation.add(node)
        if root_children > 1:
            articulation.add(root)

    candidates: dict[tuple[int, ...], tuple[int, int]] = {}
    for cut in articulation:
        seen = {cut}
        for start in sorted(adjacency[cut]):
            if start in seen:
                continue
            stack = [start]
            component: list[int] = []
            seen.add(start)
            while stack:
                node = stack.pop()
                component.append(node)
                for neighbor in adjacency[node]:
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    stack.append(neighbor)
            region = tuple(sorted(component))
            if (
                len(region) < int(min_region_size)
                or len(region) > int(max_region_size)
            ):
                continue
            region_set = set(region)
            boundary = {
                int(neighbor)
                for var in region
                for neighbor in adjacency[var]
                if neighbor not in region_set
            }
            if not boundary or len(boundary) > int(max_boundary):
                continue
            score = (len(region), -len(boundary))
            existing = candidates.get(region)
            if existing is None or score > existing:
                candidates[region] = score

    ranked = sorted(
        candidates.items(),
        key=lambda item: (item[1][0], item[1][1], -item[0][0]),
        reverse=True,
    )
    return tuple(region for region, _score in ranked)

def _q2_block_cut_decomposition(
    adjacency: Sequence[set[int]],
) -> tuple[tuple[tuple[int, ...], ...], frozenset[int]]:
    """Return biconnected blocks plus articulation vertices of q2 graph."""
    if not adjacency:
        return (), frozenset()

    n = len(adjacency)
    disc = [-1] * n
    low = [0] * n
    parent = [-1] * n
    time = 0
    articulation: set[int] = set()
    blocks: list[tuple[int, ...]] = []
    edge_stack: list[tuple[int, int]] = []

    for root in range(n):
        if disc[root] != -1:
            continue
        disc[root] = time
        low[root] = time
        time += 1
        root_children = 0
        stack: list[tuple[int, list[int], int]] = [(root, sorted(adjacency[root]), 0)]
        while stack:
            node, neighbors, idx = stack[-1]
            if idx < len(neighbors):
                neighbor = neighbors[idx]
                stack[-1] = (node, neighbors, idx + 1)
                if disc[neighbor] == -1:
                    parent[neighbor] = node
                    if node == root:
                        root_children += 1
                    edge_stack.append((node, neighbor))
                    disc[neighbor] = time
                    low[neighbor] = time
                    time += 1
                    stack.append((neighbor, sorted(adjacency[neighbor]), 0))
                    continue
                if neighbor != parent[node] and disc[neighbor] < disc[node]:
                    edge_stack.append((node, neighbor))
                    low[node] = min(low[node], disc[neighbor])
                continue

            stack.pop()
            if parent[node] != -1:
                parent_node = parent[node]
                low[parent_node] = min(low[parent_node], low[node])
                if low[node] >= disc[parent_node]:
                    articulation.add(parent_node)
                    block_vertices: set[int] = set()
                    while edge_stack:
                        left, right = edge_stack.pop()
                        block_vertices.add(left)
                        block_vertices.add(right)
                        if (left == parent_node and right == node) or (left == node and right == parent_node):
                            break
                    if block_vertices:
                        blocks.append(tuple(sorted(block_vertices)))
            elif root_children <= 1:
                articulation.discard(root)
            if parent[node] == -1 and edge_stack:
                block_vertices = set()
                while edge_stack:
                    left, right = edge_stack.pop()
                    block_vertices.add(left)
                    block_vertices.add(right)
                if block_vertices:
                    blocks.append(tuple(sorted(block_vertices)))

    unique_blocks: list[tuple[int, ...]] = []
    seen_blocks: set[tuple[int, ...]] = set()
    for block in blocks:
        if block not in seen_blocks:
            seen_blocks.add(block)
            unique_blocks.append(block)
    return tuple(unique_blocks), frozenset(articulation)

def _block_cut_boundary_region_candidates(
    adjacency: Sequence[set[int]],
    *,
    min_region_size: int = _Q3_FREE_SMALL_BOUNDARY_REGION_MIN_SIZE,
    max_region_size: int = _Q3_FREE_SMALL_BOUNDARY_REGION_MAX_SIZE,
    max_boundary: int = _Q3_FREE_SMALL_BOUNDARY_REGION_MAX_BOUNDARY,
) -> tuple[tuple[int, ...], ...]:
    if not adjacency:
        return ()
    blocks, articulation = _q2_block_cut_decomposition(adjacency)

    if not blocks:
        return ()

    block_neighbors: list[set[int]] = [set() for _ in range(len(blocks))]
    articulation_blocks: dict[int, set[int]] = {}
    for block_idx, block in enumerate(blocks):
        block_set = set(block)
        for var in block:
            if var in articulation:
                articulation_blocks.setdefault(var, set()).add(block_idx)
                block_neighbors[block_idx].add(var)

    candidates: dict[tuple[int, ...], tuple[int, int]] = {}

    def add_region(region_vars: set[int], boundary_vars: set[int]) -> None:
        region = tuple(sorted(int(var) for var in region_vars))
        boundary = tuple(sorted(int(var) for var in boundary_vars if var not in region_vars))
        if (
            len(region) < int(min_region_size)
            or len(region) > int(max_region_size)
            or not boundary
            or len(boundary) > int(max_boundary)
        ):
            return
        score = (len(region), -len(boundary))
        existing = candidates.get(region)
        if existing is None or score > existing:
            candidates[region] = score

    for start_block_idx, block in enumerate(blocks):
        block_set = set(block)
        external_articulations = {var for var in block if var in articulation}
        add_region(block_set - external_articulations, external_articulations)

        for root_articulation in sorted(external_articulations):
            seen_blocks = {start_block_idx}
            seen_articulations = {root_articulation}
            branch_blocks = [start_block_idx]
            frontier_blocks = [start_block_idx]
            while frontier_blocks:
                block_idx = frontier_blocks.pop()
                for art in block_neighbors[block_idx]:
                    if art == root_articulation:
                        continue
                    if art in seen_articulations:
                        continue
                    seen_articulations.add(art)
                    for next_block_idx in articulation_blocks.get(art, ()):
                        if next_block_idx in seen_blocks:
                            continue
                        seen_blocks.add(next_block_idx)
                        branch_blocks.append(next_block_idx)
                        frontier_blocks.append(next_block_idx)

            branch_vertices: set[int] = set()
            for block_idx in branch_blocks:
                branch_vertices.update(blocks[block_idx])

            boundary_vars: set[int] = {root_articulation}
            for art in sorted(branch_vertices & articulation):
                attached = articulation_blocks.get(art, set())
                if any(block_idx not in seen_blocks for block_idx in attached):
                    boundary_vars.add(art)

            add_region(branch_vertices - boundary_vars, boundary_vars)

    ranked = sorted(
        candidates.items(),
        key=lambda item: (item[1][0], item[1][1], -item[0][0]),
        reverse=True,
    )
    return tuple(region for region, _score in ranked)

def _build_small_boundary_region_plan(q) -> _HalfPhaseClusterPlan | None:
    """Collapse exact small-boundary regions onto a remaining q2 core."""
    if q.q3 or not q.q2:
        return None
    adjacency = _build_q2_adjacency(q)
    candidate_regions: list[tuple[int, ...]] = []
    seen_regions: set[tuple[int, ...]] = set()
    for region in _small_boundary_region_candidates(adjacency):
        if region not in seen_regions:
            seen_regions.add(region)
            candidate_regions.append(region)
    for region in _articulation_boundary_region_candidates(adjacency):
        if region not in seen_regions:
            seen_regions.add(region)
            candidate_regions.append(region)
    for region in _block_cut_boundary_region_candidates(adjacency):
        if region not in seen_regions:
            seen_regions.add(region)
            candidate_regions.append(region)
    candidates = tuple(candidate_regions[: _Q3_FREE_SMALL_BOUNDARY_REGION_MAX_REGIONS * 4])
    if not candidates:
        return None
    return _build_selected_boundary_region_plan(
        q,
        adjacency=adjacency,
        candidate_regions=candidates,
    )

def _build_block_cut_tree_region_plan(q) -> _HalfPhaseClusterPlan | None:
    """Exactly contract block-cut tree lobes onto articulation-variable core."""
    if q.q3 or not q.q2:
        return None
    cache_key = _q_structure_key(q)
    cached = _STRUCTURE_Q3_FREE_BLOCK_CUT_PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    adjacency = _build_q2_adjacency(q)
    blocks, articulation = _q2_block_cut_decomposition(adjacency)
    if not blocks or not articulation:
        _STRUCTURE_Q3_FREE_BLOCK_CUT_PLAN_CACHE[cache_key] = None
        return None

    selected_clusters: list[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            dict[tuple[int, int], int],
            tuple[tuple[int, int, int], ...],
        ]
    ] = []
    selected_cluster_vars: set[int] = set()
    for block in blocks:
        boundary_vars = tuple(sorted(int(var) for var in block if var in articulation))
        cluster_vars = tuple(sorted(int(var) for var in block if var not in articulation))
        if (
            not cluster_vars
            or not boundary_vars
            or len(boundary_vars) > _Q3_FREE_SMALL_BOUNDARY_REGION_MAX_BOUNDARY
        ):
            continue
        cluster_set = set(cluster_vars)
        boundary_set = set(boundary_vars)
        cluster_remap = {var: idx for idx, var in enumerate(cluster_vars)}
        boundary_remap = {var: idx for idx, var in enumerate(boundary_vars)}
        internal_q2 = {
            (cluster_remap[i], cluster_remap[j]): coeff
            for (i, j), coeff in q.q2.items()
            if i in cluster_set and j in cluster_set
        }
        boundary_couplings: list[tuple[int, int, int]] = []
        for (left, right), coeff in q.q2.items():
            if coeff % q.mod_q2 == 0:
                continue
            if left in cluster_set and right in boundary_set:
                boundary_couplings.append((cluster_remap[left], boundary_remap[right], int(coeff)))
            elif right in cluster_set and left in boundary_set:
                boundary_couplings.append((cluster_remap[right], boundary_remap[left], int(coeff)))
        if not boundary_couplings:
            continue
        selected_clusters.append(
            (
                cluster_vars,
                boundary_vars,
                internal_q2,
                tuple(boundary_couplings),
            )
        )
        selected_cluster_vars.update(cluster_vars)

    if not selected_clusters:
        _STRUCTURE_Q3_FREE_BLOCK_CUT_PLAN_CACHE[cache_key] = None
        return None
    core_vars = tuple(var for var in range(q.n) if var not in selected_cluster_vars)
    core_remap = {var: idx for idx, var in enumerate(core_vars)}
    core_q2 = {
        (core_remap[i], core_remap[j]): coeff
        for (i, j), coeff in q.q2.items()
        if i in core_remap and j in core_remap
    }
    mod_q1 = 1 << q.level
    mod_q2 = max(1, 1 << (q.level - 1))
    q2_lift = mod_q1 // mod_q2 if mod_q2 else 0

    factor_scopes: list[tuple[int, ...]] = [edge for edge in core_q2]
    cluster_specs: list[_HalfPhaseClusterSpec] = []
    for cluster_vars, boundary_vars, internal_q2, boundary_couplings in selected_clusters:
        if not all(var in core_remap for var in boundary_vars):
            _STRUCTURE_Q3_FREE_BLOCK_CUT_PLAN_CACHE[cache_key] = None
            return None
        boundary_core = tuple(core_remap[var] for var in boundary_vars)
        factor_scopes.append(boundary_core)
        cluster_order, cluster_width = _factor_scope_order(
            len(cluster_vars),
            list(internal_q2),
        )
        if cluster_width > _q3_free_treewidth_width_limit():
            _STRUCTURE_Q3_FREE_BLOCK_CUT_PLAN_CACHE[cache_key] = None
            return None
        native_treewidth_plan = _build_native_q3_free_treewidth_plan(
            n_vars=len(cluster_vars),
            level=q.level,
            q2=internal_q2,
            order=cluster_order,
        )
        cluster_specs.append(
            _HalfPhaseClusterSpec(
                cluster_vars=cluster_vars,
                boundary_vars=boundary_core,
                internal_q2=internal_q2,
                boundary_couplings=boundary_couplings,
                boundary_shift_table=_build_cluster_boundary_shift_table(
                    cluster_size=len(cluster_vars),
                    boundary_size=len(boundary_vars),
                    boundary_couplings=boundary_couplings,
                    q2_lift=q2_lift,
                    mod_q1=mod_q1,
                ),
                cluster_order=tuple(cluster_order),
                native_treewidth_plan=native_treewidth_plan,
            )
        )

    width_limit = _q3_free_treewidth_width_limit()
    degeneracy_lower_bound = _factor_scope_degeneracy(len(core_vars), factor_scopes)
    if degeneracy_lower_bound > width_limit:
        _STRUCTURE_Q3_FREE_BLOCK_CUT_PLAN_CACHE[cache_key] = None
        return None

    order, width = _factor_scope_order(len(core_vars), factor_scopes)
    if width > width_limit:
        _STRUCTURE_Q3_FREE_BLOCK_CUT_PLAN_CACHE[cache_key] = None
        return None

    plan = _HalfPhaseClusterPlan(
        level=q.level,
        core_vars=core_vars,
        core_q2=core_q2,
        order=tuple(order),
        width=width,
        clusters=tuple(cluster_specs),
    )
    _STRUCTURE_Q3_FREE_BLOCK_CUT_PLAN_CACHE[cache_key] = plan
    return plan

def _build_half_phase_cluster_plan(q) -> _HalfPhaseClusterPlan | None:
    """Plan exact elimination of small hard-support clusters onto the remaining core."""
    if not _is_half_phase_q2(q):
        return None

    support = _qubit_quadratic_tensor_obstruction_support(q)
    if not support:
        return None

    adjacency = [set() for _ in range(q.n)]
    for (left, right), coeff in q.q2.items():
        if coeff % q.mod_q2:
            adjacency[left].add(right)
            adjacency[right].add(left)

    support_components = _connected_components_on_vertices(adjacency, support)
    selected_clusters: list[tuple[tuple[int, ...], tuple[int, ...], dict[tuple[int, int], int], tuple[tuple[int, int, int], ...]]] = []
    selected_cluster_vars: set[int] = set()

    for component in support_components:
        cluster_vars = tuple(sorted(int(var) for var in component))
        if (
            not cluster_vars
            or len(cluster_vars) > _Q3_FREE_HALF_PHASE_CLUSTER_MAX_CLUSTER_SIZE
        ):
            continue
        boundary_vars = tuple(
            sorted(
                {
                    int(neighbor)
                    for var in cluster_vars
                    for neighbor in adjacency[var]
                    if neighbor not in component
                }
            )
        )
        if (
            not boundary_vars
            or len(boundary_vars) > _Q3_FREE_HALF_PHASE_CLUSTER_MAX_BOUNDARY
        ):
            continue

        cluster_set = set(cluster_vars)
        boundary_set = set(boundary_vars)
        cluster_remap = {var: idx for idx, var in enumerate(cluster_vars)}
        boundary_remap = {var: idx for idx, var in enumerate(boundary_vars)}
        internal_q2 = {
            (cluster_remap[i], cluster_remap[j]): coeff
            for (i, j), coeff in q.q2.items()
            if i in cluster_set and j in cluster_set
        }
        boundary_couplings: list[tuple[int, int, int]] = []
        for (left, right), coeff in q.q2.items():
            if coeff % q.mod_q2 == 0:
                continue
            if left in cluster_set and right in boundary_set:
                boundary_couplings.append((cluster_remap[left], boundary_remap[right], int(coeff)))
            elif right in cluster_set and left in boundary_set:
                boundary_couplings.append((cluster_remap[right], boundary_remap[left], int(coeff)))

        if not boundary_couplings:
            continue

        selected_clusters.append(
            (
                cluster_vars,
                boundary_vars,
                internal_q2,
                tuple(boundary_couplings),
            )
        )
        selected_cluster_vars.update(cluster_vars)

    if not selected_clusters:
        return None

    core_vars = tuple(var for var in range(q.n) if var not in selected_cluster_vars)
    core_remap = {var: idx for idx, var in enumerate(core_vars)}
    core_q2 = {
        (core_remap[i], core_remap[j]): coeff
        for (i, j), coeff in q.q2.items()
        if i in core_remap and j in core_remap
    }
    mod_q1 = 1 << q.level
    mod_q2 = max(1, 1 << (q.level - 1))
    q2_lift = mod_q1 // mod_q2 if mod_q2 else 0

    factor_scopes: list[tuple[int, ...]] = [edge for edge in core_q2]
    cluster_specs: list[_HalfPhaseClusterSpec] = []
    for cluster_vars, boundary_vars, internal_q2, boundary_couplings in selected_clusters:
        if not all(var in core_remap for var in boundary_vars):
            return None
        boundary_core = tuple(core_remap[var] for var in boundary_vars)
        factor_scopes.append(boundary_core)
        cluster_order, _cluster_width = _factor_scope_order(
            len(cluster_vars),
            list(internal_q2),
        )
        native_treewidth_plan = _build_native_q3_free_treewidth_plan(
            n_vars=len(cluster_vars),
            level=q.level,
            q2=internal_q2,
            order=cluster_order,
        )
        cluster_specs.append(
            _HalfPhaseClusterSpec(
                cluster_vars=cluster_vars,
                boundary_vars=boundary_core,
                internal_q2=internal_q2,
                boundary_couplings=boundary_couplings,
                boundary_shift_table=_build_cluster_boundary_shift_table(
                    cluster_size=len(cluster_vars),
                    boundary_size=len(boundary_vars),
                    boundary_couplings=boundary_couplings,
                    q2_lift=q2_lift,
                    mod_q1=mod_q1,
                ),
                cluster_order=tuple(cluster_order),
                native_treewidth_plan=native_treewidth_plan,
            )
        )

    width_limit = _q3_free_treewidth_width_limit()
    degeneracy_lower_bound = _factor_scope_degeneracy(len(core_vars), factor_scopes)
    if degeneracy_lower_bound > width_limit:
        return None

    order, width = _factor_scope_order(len(core_vars), factor_scopes)
    if width > width_limit:
        return None

    return _HalfPhaseClusterPlan(
        level=q.level,
        core_vars=core_vars,
        core_q2=core_q2,
        order=tuple(order),
        width=width,
        clusters=tuple(cluster_specs),
    )

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
