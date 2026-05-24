"""Extracted phase-3 cover-search helpers."""

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

from .._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals
from ..cubic_arithmetic import CubicFunction, PhaseFunction, detect_factorization
from ..scaling import ScaledAmplitude, ScaledComplex
from ..spec import CircuitSpec, Gate
from ..state import BitSequence, CircuitInput, EchelonCache, ExtendedReductionMode, ReducerInfo, ReductionInfo, SolverConfig

_LOCAL_NAMES = {
    '_evaluate_half_phase_mediator_plan_scaled',
    '_evaluate_generic_q2_mediator_plan_scaled',
    '_greedy_q3_vertex_cover',
    '_approximate_q3_vertex_cover',
    '_q3_packing_lower_bound',
    '_pick_q3_branch_edge',
    '_minimum_q3_vertex_cover_uncached',
    '_minimum_vertex_cover_from_edge_masks',
    '_minimum_q3_vertex_cover'
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


def _evaluate_half_phase_mediator_plan_scaled(
    mediator_plan: _HalfPhaseMediatorPlan,
    q1_local: Sequence[int],
) -> ScaledComplex:
    """Evaluate one exact mediator-eliminated component under a concrete q1."""
    if len(q1_local) != len(mediator_plan.core_vars) + len(mediator_plan.mediators):
        # The local q1 vector still uses the original component indexing.
        expected = max(
            (max(mediator_plan.core_vars, default=-1) + 1),
            (max((spec.mediator_var for spec in mediator_plan.mediators), default=-1) + 1),
        )
        if len(q1_local) < expected:
            raise ValueError(
                f"Expected q1_local to cover mediator-plan indices through {expected - 1}, "
                f"received length {len(q1_local)}."
            )

    core_q = _phase_function_from_parts(
        len(mediator_plan.core_vars),
        level=mediator_plan.level,
        q0=Fraction(0),
        q1=[q1_local[var] for var in mediator_plan.core_vars],
        q2=mediator_plan.core_q2,
        q3={},
    )
    scalar, factors = _build_cubic_factors_scaled(core_q)
    omega = _omega_table(mediator_plan.level)

    for spec in mediator_plan.mediators:
        residue = int(q1_local[spec.mediator_var]) % (1 << mediator_plan.level)
        even_value = _make_scaled_complex(1.0 + omega[residue])
        odd_value = _make_scaled_complex(1.0 - omega[residue])
        if len(spec.neighbor_vars) == 0:
            scalar = _mul_scaled_complex(scalar, even_value)
            continue
        if len(spec.neighbor_vars) == 1:
            scalar = _mul_scaled_complex(
                scalar,
                _combine_factor_scaled(
                    factors,
                    spec.neighbor_vars,
                    [even_value, odd_value],
                ),
            )
            continue
        scalar = _mul_scaled_complex(
            scalar,
            _combine_factor_scaled(
                factors,
                spec.neighbor_vars,
                [even_value, odd_value, odd_value, even_value],
            ),
        )

    total, _ = _sum_factor_tables_scaled(
        len(mediator_plan.core_vars),
        factors,
        mediator_plan.order,
        scalar=scalar,
    )
    return total


def _evaluate_generic_q2_mediator_plan_scaled(
    mediator_plan: _GenericQ2MediatorPlan,
    q1_local: Sequence[int],
) -> ScaledComplex:
    """Evaluate one exact arbitrary-q2 mediator-eliminated component."""
    if len(q1_local) != len(mediator_plan.core_vars) + len(mediator_plan.mediators):
        expected = max(
            (max(mediator_plan.core_vars, default=-1) + 1),
            (max((spec.mediator_var for spec in mediator_plan.mediators), default=-1) + 1),
        )
        if len(q1_local) < expected:
            raise ValueError(
                f"Expected q1_local to cover mediator-plan indices through {expected - 1}, "
                f"received length {len(q1_local)}."
            )

    core_q = _phase_function_from_parts(
        len(mediator_plan.core_vars),
        level=mediator_plan.level,
        q0=Fraction(0),
        q1=[q1_local[var] for var in mediator_plan.core_vars],
        q2=mediator_plan.core_q2,
        q3={},
    )
    scalar, factors = _build_cubic_factors_scaled(core_q)
    omega_scaled = _omega_scaled_table(mediator_plan.level)
    mod_q1 = 1 << mediator_plan.level
    mod_q2 = max(1, 1 << (mediator_plan.level - 1))
    q2_lift = mod_q1 // mod_q2 if mod_q2 else 0

    for spec in mediator_plan.mediators:
        base_residue = int(q1_local[spec.mediator_var]) % mod_q1
        table: list[ScaledComplex] = []
        for assignment in range(1 << len(spec.neighbor_vars)):
            residue = base_residue
            for neighbor_idx, coeff in enumerate(spec.neighbor_couplings):
                if (assignment >> neighbor_idx) & 1:
                    residue = (residue + (q2_lift * int(coeff))) % mod_q1
            table.append(_add_scaled_complex(_ONE_SCALED, omega_scaled[residue]))
        scalar = _mul_scaled_complex(
            scalar,
            _combine_factor_scaled(factors, spec.neighbor_vars, table),
        )

    total, _ = _sum_factor_tables_scaled(
        len(mediator_plan.core_vars),
        factors,
        mediator_plan.order,
        scalar=scalar,
    )
    return total


def _greedy_q3_vertex_cover(n_vars, edge_masks, edge_cover_masks, remaining_edges_mask=None):
    """Max-degree greedy cover used as an incumbent and large-instance fallback."""
    if remaining_edges_mask is None:
        remaining_edges_mask = (1 << len(edge_masks)) - 1

    chosen = []
    chosen_mask = 0
    while remaining_edges_mask:
        best_var = -1
        best_score = None
        for var in range(n_vars):
            if chosen_mask & (1 << var):
                continue
            covered = edge_cover_masks[var] & remaining_edges_mask
            if not covered:
                continue
            score = (covered.bit_count(), -var)
            if best_score is None or score > best_score:
                best_var = var
                best_score = score
        if best_var < 0:
            raise RuntimeError("Failed to build a q3 vertex cover.")
        chosen.append(best_var)
        chosen_mask |= 1 << best_var
        remaining_edges_mask &= ~edge_cover_masks[best_var]
    return sorted(chosen)


def _approximate_q3_vertex_cover(n_vars, edge_masks, edge_cover_masks):
    """Return the better of a greedy cover and a trivial 3-approximation."""
    greedy = _greedy_q3_vertex_cover(n_vars, edge_masks, edge_cover_masks)

    remaining_edges_mask = (1 << len(edge_masks)) - 1
    chosen_mask = 0
    while remaining_edges_mask:
        edge_bit = remaining_edges_mask & -remaining_edges_mask
        edge_idx = edge_bit.bit_length() - 1
        edge_mask = edge_masks[edge_idx]
        chosen_mask |= edge_mask

        covered_edges = 0
        vertices = edge_mask
        while vertices:
            vertex_bit = vertices & -vertices
            var = vertex_bit.bit_length() - 1
            covered_edges |= edge_cover_masks[var]
            vertices ^= vertex_bit
        remaining_edges_mask &= ~covered_edges

    three_approx = [var for var in range(n_vars) if chosen_mask & (1 << var)]
    return greedy if len(greedy) <= len(three_approx) else three_approx


def _q3_packing_lower_bound(remaining_edges_mask, edge_conflicts):
    """Lower-bound the cover size via a greedy packing of disjoint hyperedges."""
    packing = 0
    remaining = remaining_edges_mask
    while remaining:
        best_edge = -1
        best_conflicts = None
        probe = remaining
        while probe:
            edge_bit = probe & -probe
            edge_idx = edge_bit.bit_length() - 1
            conflicts = (edge_conflicts[edge_idx] & remaining).bit_count()
            if best_conflicts is None or conflicts < best_conflicts:
                best_edge = edge_idx
                best_conflicts = conflicts
            probe ^= edge_bit
        remaining &= ~edge_conflicts[best_edge]
        packing += 1
    return packing


def _pick_q3_branch_edge(remaining_edges_mask, edge_masks, edge_cover_masks):
    """Choose an uncovered hyperedge whose endpoints cover many remaining edges."""
    best_edge = -1
    best_score = None
    probe = remaining_edges_mask
    while probe:
        edge_bit = probe & -probe
        edge_idx = edge_bit.bit_length() - 1
        counts = []
        vertices = edge_masks[edge_idx]
        while vertices:
            vertex_bit = vertices & -vertices
            var = vertex_bit.bit_length() - 1
            counts.append((edge_cover_masks[var] & remaining_edges_mask).bit_count())
            vertices ^= vertex_bit
        counts.sort(reverse=True)
        score = tuple(counts)
        if best_score is None or score > best_score:
            best_edge = edge_idx
            best_score = score
        probe ^= edge_bit
    return best_edge


def _minimum_q3_vertex_cover_uncached(q):
    """Exact minimum q3-hypergraph cover on small cores, heuristic otherwise."""
    if not q.q3:
        return []

    return _minimum_vertex_cover_from_edge_masks(
        q.n,
        [((1 << i) | (1 << j) | (1 << k)) for i, j, k in q.q3],
        exact_size_cutoff=_Q3_VERTEX_COVER_EXACT_SIZE_CUTOFF,
        exact_edge_cutoff=_Q3_VERTEX_COVER_EXACT_EDGE_CUTOFF,
    )


def _minimum_vertex_cover_from_edge_masks(
    n_vars: int,
    edge_masks: Sequence[int],
    *,
    exact_size_cutoff: int,
    exact_edge_cutoff: int,
) -> list[int]:
    """Exact minimum vertex cover on small hypergraphs, heuristic otherwise."""
    if not edge_masks:
        return []

    edge_cover_masks = [0] * n_vars
    for edge_idx, edge_mask in enumerate(edge_masks):
        vertices = edge_mask
        while vertices:
            vertex_bit = vertices & -vertices
            var = vertex_bit.bit_length() - 1
            edge_cover_masks[var] |= 1 << edge_idx
            vertices ^= vertex_bit

    greedy_cover = _approximate_q3_vertex_cover(n_vars, edge_masks, edge_cover_masks)
    if (
        len(greedy_cover) > exact_size_cutoff
        or len(edge_masks) > exact_edge_cutoff
    ):
        return greedy_cover

    edge_conflicts = [0] * len(edge_masks)
    for edge_idx, edge_mask in enumerate(edge_masks):
        edge_conflicts[edge_idx] |= 1 << edge_idx
        for other_idx in range(edge_idx):
            if edge_mask & edge_masks[other_idx]:
                edge_conflicts[edge_idx] |= 1 << other_idx
                edge_conflicts[other_idx] |= 1 << edge_idx

    all_edges_mask = (1 << len(edge_masks)) - 1
    lower_bound = _q3_packing_lower_bound(all_edges_mask, edge_conflicts)
    if lower_bound == len(greedy_cover):
        return greedy_cover

    budget_bits = max(1, int(len(greedy_cover)).bit_length())
    failed_states: set[int] = set()

    def pack_failed_state(remaining_edges_mask: int, budget: int) -> int:
        return (int(remaining_edges_mask) << budget_bits) | int(budget)

    def search(remaining_edges_mask, budget):
        if not remaining_edges_mask:
            return ()
        if budget == 0:
            return None
        if len(failed_states) >= _Q3_VERTEX_COVER_EXACT_FAILED_STATE_CUTOFF:
            return None
        state = pack_failed_state(remaining_edges_mask, budget)
        if state in failed_states:
            return None
        if _q3_packing_lower_bound(remaining_edges_mask, edge_conflicts) > budget:
            failed_states.add(state)
            return None

        edge_idx = _pick_q3_branch_edge(remaining_edges_mask, edge_masks, edge_cover_masks)
        vertices = []
        vertex_mask = edge_masks[edge_idx]
        while vertex_mask:
            vertex_bit = vertex_mask & -vertex_mask
            var = vertex_bit.bit_length() - 1
            vertices.append(var)
            vertex_mask ^= vertex_bit
        vertices.sort(
            key=lambda var: ((edge_cover_masks[var] & remaining_edges_mask).bit_count(), -var),
            reverse=True,
        )

        for var in vertices:
            result = search(remaining_edges_mask & ~edge_cover_masks[var], budget - 1)
            if result is not None:
                return (var,) + result

        failed_states.add(state)
        return None

    for budget in range(lower_bound, len(greedy_cover)):
        result = search(all_edges_mask, budget)
        if result is not None:
            return sorted(result)
        if len(failed_states) >= _Q3_VERTEX_COVER_EXACT_FAILED_STATE_CUTOFF:
            break
    return greedy_cover


def _minimum_q3_vertex_cover(q):
    cache_key = _q_structure_key(q)
    cached = _STRUCTURE_Q3_COVER_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)
    cover = tuple(_minimum_q3_vertex_cover_uncached(q))
    _STRUCTURE_Q3_COVER_CACHE[cache_key] = cover
    return list(cover)

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
