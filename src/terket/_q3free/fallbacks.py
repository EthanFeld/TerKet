"""Extracted q3-free fallback heuristics."""

from __future__ import annotations

import bisect
import cmath
from collections import deque
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
    '_non_half_phase_q2_edge_masks',
    '_refine_bad_q2_vertex_cover',
    '_minimum_bad_q2_vertex_cover_uncached',
    '_minimum_bad_q2_vertex_cover',
    '_bad_q2_cover_dispatch_allowed',
    '_sum_q3_free_via_bad_q2_cover_scaled',
    '_gauss_obstruction',
    '_sum_q3_free_via_nonquadratic_support_scaled',
    '_sum_q3_free_via_nonquadratic_support',
    '_cubic_order_width',
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


def _non_half_phase_q2_edge_masks(q) -> list[int]:
    """Return the graph edges whose q2 residue blocks the half-phase backend."""
    if q.q3 or not q.q2:
        return []
    half_q2 = q.mod_q2 // 2 if q.mod_q2 else 0
    edge_masks: list[int] = []
    for (left, right), coeff in q.q2.items():
        residue = int(coeff) % q.mod_q2
        if residue not in (0, half_q2):
            edge_masks.append((1 << left) | (1 << right))
    return edge_masks

def _refine_bad_q2_vertex_cover(q, cover: Sequence[int], edge_masks: Sequence[int] | None = None) -> list[int]:
    """Prefer equal-size covers that eliminate more total q2 structure."""
    if not cover:
        return []
    if edge_masks is None:
        edge_masks = _non_half_phase_q2_edge_masks(q)
    if not edge_masks:
        return list(cover)

    half_q2 = q.mod_q2 // 2 if q.mod_q2 else 0
    total_q2_degree = [0] * q.n
    bad_q2_degree = [0] * q.n
    for (left, right), coeff in q.q2.items():
        residue = int(coeff) % q.mod_q2
        if not residue:
            continue
        total_q2_degree[left] += 1
        total_q2_degree[right] += 1
        if residue not in (0, half_q2):
            bad_q2_degree[left] += 1
            bad_q2_degree[right] += 1

    all_edges_mask = (1 << len(edge_masks)) - 1

    def covers_all(vertices: set[int]) -> bool:
        covered_edges_mask = 0
        for var in vertices:
            vertex_bit = 1 << int(var)
            for edge_idx, edge_mask in enumerate(edge_masks):
                if edge_mask & vertex_bit:
                    covered_edges_mask |= 1 << edge_idx
        return covered_edges_mask == all_edges_mask

    def score(vertices: set[int]) -> tuple[int, int, int, tuple[int, ...]]:
        ordered = tuple(sorted(int(var) for var in vertices))
        return (
            sum(total_q2_degree[var] for var in ordered),
            sum(bad_q2_degree[var] for var in ordered),
            sum(var * var for var in ordered),
            tuple(-var for var in ordered),
        )

    support = tuple(sorted(idx for idx, degree in enumerate(bad_q2_degree) if degree))
    best = set(int(var) for var in cover)
    best_score = score(best)

    improved = True
    while improved:
        improved = False
        for remove_var in tuple(sorted(best)):
            for add_var in support:
                if add_var in best:
                    continue
                candidate = set(best)
                candidate.remove(remove_var)
                candidate.add(int(add_var))
                if not covers_all(candidate):
                    continue
                candidate_score = score(candidate)
                if candidate_score > best_score:
                    best = candidate
                    best_score = candidate_score
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        best_ordered = tuple(sorted(best))
        for remove_idx, remove_left in enumerate(best_ordered):
            for remove_right in best_ordered[remove_idx + 1:]:
                for add_idx, add_left in enumerate(support):
                    if add_left in best:
                        continue
                    for add_right in support[add_idx + 1:]:
                        if add_right in best:
                            continue
                        candidate = set(best)
                        candidate.remove(remove_left)
                        candidate.remove(remove_right)
                        candidate.add(int(add_left))
                        candidate.add(int(add_right))
                        if not covers_all(candidate):
                            continue
                        candidate_score = score(candidate)
                        if candidate_score > best_score:
                            best = candidate
                            best_score = candidate_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return sorted(best)

def _minimum_bad_q2_vertex_cover_uncached(q) -> list[int]:
    """Exact minimum cover of the non-half-phase q2 edges, heuristic otherwise."""
    edge_masks = _non_half_phase_q2_edge_masks(q)
    if not edge_masks:
        return []
    cover = _minimum_vertex_cover_from_edge_masks(
        q.n,
        edge_masks,
        exact_size_cutoff=_Q3_VERTEX_COVER_EXACT_SIZE_CUTOFF,
        exact_edge_cutoff=_Q3_VERTEX_COVER_EXACT_EDGE_CUTOFF,
    )
    return _refine_bad_q2_vertex_cover(q, cover, edge_masks=edge_masks)

def _minimum_bad_q2_vertex_cover(q) -> list[int]:
    cache_key = _q_structure_key(q)
    cached = _STRUCTURE_BAD_Q2_COVER_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)
    cover = tuple(_minimum_bad_q2_vertex_cover_uncached(q))
    _STRUCTURE_BAD_Q2_COVER_CACHE[cache_key] = cover
    return list(cover)

def _bad_q2_cover_dispatch_allowed(q, cover: Sequence[int] | None = None) -> bool:
    """Return whether the bad-q2 cover branch is structurally promising."""
    if q.q3 or _is_half_phase_q2(q):
        return False
    if q.n < _Q3_FREE_BAD_Q2_COVER_DISPATCH_MIN_VARS:
        return False
    if cover is None:
        cover = _minimum_bad_q2_vertex_cover(q)
    if not cover or len(cover) > _Q3_FREE_BAD_Q2_COVER_MAX_SIZE:
        return False

    max_edges = q.n * (q.n - 1) // 2
    if max_edges <= 0:
        return False
    if len(q.q2) < int(math.ceil(_Q3_FREE_BAD_Q2_COVER_DISPATCH_MIN_DENSITY * max_edges)):
        return False

    bad_support = set()
    for edge_mask in _non_half_phase_q2_edge_masks(q):
        vertices = edge_mask
        while vertices:
            vertex_bit = vertices & -vertices
            bad_support.add(vertex_bit.bit_length() - 1)
            vertices ^= vertex_bit
    return len(bad_support) >= _Q3_FREE_BAD_Q2_COVER_DISPATCH_MIN_SUPPORT_FACTOR * len(cover)

def _sum_q3_free_via_bad_q2_cover_scaled(q, *, cover: Sequence[int] | None = None):
    """Fix a small cover of bad q2 edges, then solve each half-phase branch exactly."""
    if q.q3 or _is_half_phase_q2(q):
        return None

    if cover is None:
        cover = _minimum_bad_q2_vertex_cover(q)
    if not cover or len(cover) > _Q3_FREE_BAD_Q2_COVER_MAX_SIZE:
        return None

    total = _ZERO_SCALED
    for fixed_mask in range(1 << len(cover)):
        fixed_values = [(fixed_mask >> idx) & 1 for idx in range(len(cover))]
        branch_q = _fix_variables(q, cover, fixed_values)
        if not _is_half_phase_q2(branch_q):
            return None
        total = _add_scaled_complex(total, _sum_q3_free_component_scaled(branch_q))
    return total

def _gauss_obstruction(q, structural_obstruction: int = 0) -> int:
    """Return the BL26-style obstruction combining q3 and nonquadratic q1/q2."""
    return max(structural_obstruction, _qubit_quadratic_tensor_obstruction(q))

def _sum_q3_free_via_nonquadratic_support_scaled(q):
    """Branch on a small nonquadratic support, then solve each branch exactly."""
    support = _qubit_quadratic_tensor_obstruction_support(q)
    if not support or len(support) > _Q3_FREE_NONQUADRATIC_BRANCH_MAX_SUPPORT:
        return None

    total = _ZERO_SCALED
    for fixed_mask in range(1 << len(support)):
        fixed_values = [(fixed_mask >> idx) & 1 for idx in range(len(support))]
        branch_q = _fix_variables(q, support, fixed_values)
        if branch_q.q3:
            return None
        if _qubit_quadratic_tensor_obstruction(branch_q):
            return None
        branch_total, _phase_info = _gauss_sum_q3_free_scaled(branch_q)
        total = _add_scaled_complex(total, branch_total)
    return total

def _sum_q3_free_via_nonquadratic_support(q):
    scaled = _sum_q3_free_via_nonquadratic_support_scaled(q)
    if scaled is None:
        return None
    return _scaled_to_complex(scaled)

def _cubic_order_width(q, order):
    if len(order) != q.n:
        raise ValueError(f"Expected elimination order of length {q.n}, received {len(order)}.")
    native_cubic_order_width = _native_symbol("cubic_order_width")
    if native_cubic_order_width is not None:
        try:
            return native_cubic_order_width(q.n, q.q2, q.q3, order)
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
    max_scope = 0
    for var in order:
        var_bit = 1 << var
        if not (remaining_mask & var_bit):
            raise ValueError("Elimination order must contain each variable exactly once.")
        neighbors_mask = adjacency_masks[var] & remaining_mask
        max_scope = max(max_scope, neighbors_mask.bit_count() + 1)

        neighbor_bits = tuple(_iter_mask_bits(neighbors_mask))
        remove_var_mask = ~var_bit
        for left in neighbor_bits:
            adjacency_masks[left] = (
                adjacency_masks[left]
                | (neighbors_mask & ~(1 << left))
            ) & remove_var_mask
        adjacency_masks[var] = 0
        remaining_mask &= remove_var_mask
    return max_scope

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
