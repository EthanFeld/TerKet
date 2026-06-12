"""q3-free cutset residue-data builders for exact reuse paths."""

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

from .batch import _as_int64_array
from ..cubic_arithmetic import CubicFunction, PhaseFunction, detect_factorization
from .._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals
from ..scaling import ScaledAmplitude, ScaledComplex
from ..spec import CircuitSpec, Gate
from ..state import BitSequence, CircuitInput, EchelonCache, ExtendedReductionMode, ReducerInfo, ReductionInfo, SolverConfig

_LOCAL_NAMES = {
    '_build_q3_free_cutset_residue_data',
    '_build_q3_free_residual_projection',
    '_evaluate_q3_free_cutset_candidate',
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


def _build_q3_free_cutset_residue_data(
    q: PhaseFunction,
    cutset_vars: tuple[int, ...],
    remaining_vars: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q2_lift = q.mod_q1 // q.mod_q2 if q.mod_q2 else 0
    cutset_index = {var: idx for idx, var in enumerate(cutset_vars)}
    remaining_index = {var: idx for idx, var in enumerate(remaining_vars)}
    cutset_remaining = np.zeros((len(cutset_vars), len(remaining_vars)), dtype=np.int64)
    cutset_cutset_left: list[int] = []
    cutset_cutset_right: list[int] = []
    cutset_cutset_residue: list[int] = []

    for (left, right), coeff in q.q2.items():
        residue = (q2_lift * coeff) % q.mod_q1
        if not residue:
            continue
        if left in cutset_index and right in remaining_index:
            cutset_remaining[cutset_index[left], remaining_index[right]] = (
                cutset_remaining[cutset_index[left], remaining_index[right]] + residue
            ) % q.mod_q1
        elif right in cutset_index and left in remaining_index:
            cutset_remaining[cutset_index[right], remaining_index[left]] = (
                cutset_remaining[cutset_index[right], remaining_index[left]] + residue
            ) % q.mod_q1
        elif left in cutset_index and right in cutset_index:
            cutset_cutset_left.append(cutset_index[left])
            cutset_cutset_right.append(cutset_index[right])
            cutset_cutset_residue.append(residue)

    return (
        cutset_remaining,
        _as_int64_array(cutset_cutset_left),
        _as_int64_array(cutset_cutset_right),
        _as_int64_array(cutset_cutset_residue),
    )

def _build_q3_free_residual_projection(
    q: PhaseFunction,
    cutset_vars: tuple[int, ...],
    *,
    remaining_universe: tuple[int, ...] | None = None,
    parent_projection: _Q3FreeResidualProjection | None = None,
) -> _Q3FreeResidualProjection | None:
    cutset_set = set(int(var) for var in cutset_vars)
    if parent_projection is not None:
        parent_remaining = tuple(int(var) for var in parent_projection.remaining_vars)
        if cutset_set <= set(parent_remaining):
            child_remaining_vars = tuple(
                int(var) for var in parent_remaining if var not in cutset_set
            )
            removed_vars = [var for var in parent_remaining if var in cutset_set]
            if len(child_remaining_vars) + len(removed_vars) == len(parent_remaining):
                local_keep = [
                    idx for idx, var in enumerate(parent_remaining) if var not in cutset_set
                ]
                child_q = _component_restriction(parent_projection.remaining_q, local_keep)
                return _Q3FreeResidualProjection(
                    remaining_vars=child_remaining_vars,
                    remaining_q=child_q,
                )
    if remaining_universe is None:
        remaining_universe = tuple(range(q.n))
    remaining_vars = tuple(var for var in remaining_universe if var not in cutset_set)
    if not remaining_vars:
        return None
    return _Q3FreeResidualProjection(
        remaining_vars=remaining_vars,
        remaining_q=_component_restriction(q, remaining_vars),
    )


def _cutset_candidate_score(
    *,
    cutset_size: int,
    prioritize_width: bool,
    prefer_one_shot_slicing: bool,
    viable_flag: int,
    target_miss: int,
    width: int,
    work: int,
    generic_penalty: int = 0,
) -> tuple[int, ...]:
    if prioritize_width:
        if prefer_one_shot_slicing:
            return (viable_flag, generic_penalty, target_miss, width, work, cutset_size)
        return (viable_flag, target_miss, width, work, cutset_size)
    if prefer_one_shot_slicing:
        return (viable_flag, generic_penalty, work, width, cutset_size)
    return (viable_flag, work, width, cutset_size)


def _make_cutset_conditioning_plan(
    q: PhaseFunction,
    cutset_vars: tuple[int, ...],
    remaining_vars: tuple[int, ...],
    *,
    remaining_backend: Literal["product", "generic", "treewidth"],
    remaining_q2: Mapping[tuple[int, int], int],
    remaining_order: Sequence[int],
    residue_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    remaining_width: int,
    estimated_total_work: int,
    remaining_isolated_vars: Sequence[int] = (),
    remaining_components: Sequence[_Q3FreeConstraintComponentPlan] = (),
) -> _Q3FreeCutsetConditioningPlan:
    cutset_remaining, cutset_cutset_left, cutset_cutset_right, cutset_cutset_residue = residue_data
    return _Q3FreeCutsetConditioningPlan(
        level=q.level,
        cutset_vars=cutset_vars,
        remaining_vars=remaining_vars,
        remaining_backend=remaining_backend,
        remaining_q2=remaining_q2,
        remaining_order=tuple(int(var) for var in remaining_order),
        cutset_remaining_q2_residue=cutset_remaining,
        cutset_cutset_left=cutset_cutset_left,
        cutset_cutset_right=cutset_cutset_right,
        cutset_cutset_residue=cutset_cutset_residue,
        remaining_isolated_vars=tuple(int(var) for var in remaining_isolated_vars),
        remaining_components=tuple(remaining_components),
        remaining_width=int(remaining_width),
        estimated_total_work=max(1, int(estimated_total_work)),
    )


def _summarize_generic_remaining(
    remaining_q: PhaseFunction,
    *,
    use_one_shot_surrogate: bool,
    surrogate_width: int,
    surrogate_work: int,
) -> tuple[tuple[int, ...], int, int, int]:
    if use_one_shot_surrogate:
        return (), int(surrogate_width), max(1, int(surrogate_work)), 1

    component_sets = detect_factorization(remaining_q)
    covered = set().union(*component_sets) if component_sets else set()
    isolated_vars = tuple(sorted(set(range(remaining_q.n)) - covered))
    component_width = 0
    component_work = 0
    generic_penalty = 1
    for component in component_sets:
        component_q = _component_restriction(remaining_q, component)
        component_adjacency, component_edges = _q3_free_graph(component_q)
        component_max_degree = max((len(neighbors) for neighbors in component_adjacency), default=0)
        if (
            use_one_shot_surrogate
            and component_q.n >= _Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS
            and component_max_degree <= 4
        ):
            order, width = _best_cheap_q3_free_order(component_q)
            component_width = max(component_width, int(width))
            component_work += max(1, int(_estimate_treewidth_dp_work(component_q, order)))
            generic_penalty += 1
            continue
        order, width = _min_fill_cubic_order(component_q)
        component_width = max(component_width, int(width))
        component_work += max(1, int(_estimate_treewidth_dp_work(component_q, order)))
        component_depth, component_chords = _q3_free_spanning_data(component_adjacency, component_edges)
        if _q3_free_treewidth_order(
            component_q,
            len(_select_feedback_vertices(component_q.n, component_chords, component_depth)),
            order_hint=order,
            max_degree=component_max_degree,
        ) is None:
            generic_penalty += 1
    return isolated_vars, component_width, max(1, component_work), generic_penalty


def _initial_cutset_candidate_order(
    remaining_q: PhaseFunction,
    *,
    remaining_order_hint: Sequence[int] | None,
    remaining_index: Mapping[int, int],
    use_one_shot_surrogate: bool,
) -> tuple[tuple[int, ...], int, int, tuple[int, ...] | None]:
    local_hint_order = None
    if remaining_order_hint is not None:
        local_hint = [remaining_index[var] for var in remaining_order_hint if var in remaining_index]
        if len(local_hint) == remaining_q.n:
            local_hint_order = tuple(int(var) for var in local_hint)
    if local_hint_order is not None:
        candidate_order = local_hint_order
        width = _cubic_order_width(remaining_q, candidate_order)
    elif use_one_shot_surrogate:
        candidate_order, width = _best_cheap_q3_free_order(remaining_q)
    else:
        candidate_order, width = _min_fill_cubic_order(remaining_q)
    work = (
        _cheap_q3_free_work_surrogate(remaining_q, width)
        if use_one_shot_surrogate
        else _estimate_treewidth_dp_work(remaining_q, candidate_order)
    )
    return tuple(int(var) for var in candidate_order), int(width), max(1, int(work)), local_hint_order


def _resolve_viable_treewidth_order(
    remaining_q: PhaseFunction,
    *,
    candidate_order: tuple[int, ...],
    width: int,
    work: int,
    remaining_adjacency,
    remaining_edges,
    remaining_max_degree: int,
    allow_generic_remaining: bool,
    target_remaining_width: int | None,
    use_one_shot_surrogate: bool,
    local_hint_order: tuple[int, ...] | None,
) -> tuple[tuple[int, ...] | None, tuple[int, ...], int, int]:
    reduced_depth, reduced_chords = _q3_free_spanning_data(remaining_adjacency, remaining_edges)
    reduced_feedback = _select_feedback_vertices(remaining_q.n, reduced_chords, reduced_depth)
    skip_exact_treewidth_search = (
        use_one_shot_surrogate
        and allow_generic_remaining
        and width > max(
            _q3_free_treewidth_width_limit() + 4,
            (
                int(target_remaining_width) + 8
                if target_remaining_width is not None
                else _q3_free_treewidth_width_limit() + 4
            ),
        )
    )
    viable_order = None if skip_exact_treewidth_search else _q3_free_treewidth_order(
        remaining_q,
        len(reduced_feedback),
        order_hint=candidate_order,
        max_degree=remaining_max_degree,
    )
    if viable_order is not None or local_hint_order is None or use_one_shot_surrogate:
        return viable_order, candidate_order, width, work

    candidate_order, width = _min_fill_cubic_order(remaining_q)
    work = max(1, int(_estimate_treewidth_dp_work(remaining_q, candidate_order)))
    viable_order = _q3_free_treewidth_order(
        remaining_q,
        len(reduced_feedback),
        order_hint=candidate_order,
        max_degree=remaining_max_degree,
    )
    return viable_order, tuple(int(var) for var in candidate_order), int(width), int(work)


def _product_cutset_candidate(
    q: PhaseFunction,
    cutset_vars: tuple[int, ...],
    remaining_vars: tuple[int, ...],
    *,
    residue_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    prioritize_width: bool,
    prefer_one_shot_slicing: bool,
    branch_count: int,
) -> _Q3FreeCutsetCandidateEvaluation:
    plan = _make_cutset_conditioning_plan(
        q,
        cutset_vars,
        remaining_vars,
        remaining_backend="product",
        remaining_q2={},
        remaining_order=(),
        residue_data=residue_data,
        remaining_width=0,
        estimated_total_work=branch_count,
    )
    return _Q3FreeCutsetCandidateEvaluation(
        cutset_vars=cutset_vars,
        plan=plan,
        viable=True,
        score=_cutset_candidate_score(
            cutset_size=len(cutset_vars),
            prioritize_width=prioritize_width,
            prefer_one_shot_slicing=prefer_one_shot_slicing,
            viable_flag=0,
            target_miss=0,
            width=0,
            work=branch_count,
        ),
    )


def _generic_cutset_candidate(
    q: PhaseFunction,
    cutset_vars: tuple[int, ...],
    remaining_vars: tuple[int, ...],
    *,
    remaining_q: PhaseFunction,
    residue_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    prioritize_width: bool,
    prefer_one_shot_slicing: bool,
    target_remaining_width: int | None,
    branch_count: int,
    use_one_shot_surrogate: bool,
    surrogate_width: int,
    surrogate_work: int,
) -> _Q3FreeCutsetCandidateEvaluation:
    isolated_vars, component_width, component_work, generic_penalty = _summarize_generic_remaining(
        remaining_q,
        use_one_shot_surrogate=use_one_shot_surrogate,
        surrogate_width=surrogate_width,
        surrogate_work=surrogate_work,
    )
    generic_work = branch_count * component_work
    plan = _make_cutset_conditioning_plan(
        q,
        cutset_vars,
        remaining_vars,
        remaining_backend="generic",
        remaining_q2=remaining_q.q2,
        remaining_order=(),
        residue_data=residue_data,
        remaining_width=component_width,
        estimated_total_work=generic_work,
        remaining_isolated_vars=isolated_vars,
    )
    target_miss = int(
        target_remaining_width is not None
        and component_width > int(target_remaining_width)
    )
    return _Q3FreeCutsetCandidateEvaluation(
        cutset_vars=cutset_vars,
        plan=plan,
        viable=True,
        score=_cutset_candidate_score(
            cutset_size=len(cutset_vars),
            prioritize_width=prioritize_width,
            prefer_one_shot_slicing=prefer_one_shot_slicing,
            viable_flag=0,
            target_miss=target_miss,
            width=component_width,
            work=generic_work,
            generic_penalty=generic_penalty,
        ),
    )


def _infeasible_cutset_candidate(
    cutset_vars: tuple[int, ...],
    *,
    prioritize_width: bool,
    prefer_one_shot_slicing: bool,
    target_remaining_width: int | None,
    width: int,
    effective_work: int,
) -> _Q3FreeCutsetCandidateEvaluation:
    miss_width = max(width, int(target_remaining_width or 0)) if prioritize_width else width
    return _Q3FreeCutsetCandidateEvaluation(
        cutset_vars=cutset_vars,
        plan=None,
        viable=False,
        score=_cutset_candidate_score(
            cutset_size=len(cutset_vars),
            prioritize_width=prioritize_width,
            prefer_one_shot_slicing=prefer_one_shot_slicing,
            viable_flag=1,
            target_miss=1,
            width=miss_width,
            work=effective_work,
        ),
    )


def _treewidth_cutset_candidate(
    q: PhaseFunction,
    cutset_vars: tuple[int, ...],
    remaining_vars: tuple[int, ...],
    *,
    remaining_q: PhaseFunction,
    viable_order: Sequence[int],
    residue_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    prioritize_width: bool,
    prefer_one_shot_slicing: bool,
    target_remaining_width: int | None,
    effective_work: int,
) -> _Q3FreeCutsetCandidateEvaluation:
    viable_width = _treewidth_order_width(remaining_q, viable_order)
    target_miss = int(
        target_remaining_width is not None
        and viable_width > int(target_remaining_width)
    )
    plan = _make_cutset_conditioning_plan(
        q,
        cutset_vars,
        remaining_vars,
        remaining_backend="treewidth",
        remaining_q2=remaining_q.q2,
        remaining_order=viable_order,
        residue_data=residue_data,
        remaining_width=viable_width,
        estimated_total_work=effective_work,
    )
    return _Q3FreeCutsetCandidateEvaluation(
        cutset_vars=cutset_vars,
        plan=plan,
        viable=True,
        score=_cutset_candidate_score(
            cutset_size=len(cutset_vars),
            prioritize_width=prioritize_width,
            prefer_one_shot_slicing=prefer_one_shot_slicing,
            viable_flag=0,
            target_miss=target_miss,
            width=viable_width,
            work=effective_work,
        ),
    )


def _evaluate_q3_free_cutset_candidate(
    q: PhaseFunction,
    cutset_vars: tuple[int, ...],
    *,
    remaining_universe: tuple[int, ...] | None = None,
    residual_projection: _Q3FreeResidualProjection | None = None,
    remaining_order_hint: Sequence[int] | None = None,
    prioritize_width: bool = False,
    target_remaining_width: int | None = None,
    allow_generic_remaining: bool = False,
    prefer_one_shot_slicing: bool = False,
) -> _Q3FreeCutsetCandidateEvaluation | None:
    if remaining_universe is None:
        remaining_universe = tuple(range(q.n))
    if residual_projection is None:
        residual_projection = _build_q3_free_residual_projection(
            q,
            cutset_vars,
            remaining_universe=remaining_universe,
        )
    if residual_projection is None:
        return None
    remaining_vars = tuple(int(var) for var in residual_projection.remaining_vars)
    if not remaining_vars:
        return None
    remaining_index = {var: idx for idx, var in enumerate(remaining_vars)}
    residue_data = _build_q3_free_cutset_residue_data(q, cutset_vars, remaining_vars)
    branch_count = 1 << len(cutset_vars)
    remaining_q = residual_projection.remaining_q
    remaining_adjacency, remaining_edges = _q3_free_graph(remaining_q)
    remaining_max_degree = max((len(neighbors) for neighbors in remaining_adjacency), default=0)
    use_one_shot_surrogate = (
        prefer_one_shot_slicing
        and remaining_q.n >= _Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS
        and remaining_max_degree <= 4
    )

    if not remaining_q.q2:
        return _product_cutset_candidate(
            q,
            cutset_vars,
            remaining_vars,
            residue_data=residue_data,
            prioritize_width=prioritize_width,
            prefer_one_shot_slicing=prefer_one_shot_slicing,
            branch_count=branch_count,
        )

    candidate_order, width, work, local_hint_order = _initial_cutset_candidate_order(
        remaining_q,
        remaining_order_hint=remaining_order_hint,
        remaining_index=remaining_index,
        use_one_shot_surrogate=use_one_shot_surrogate,
    )
    viable_order, candidate_order, width, work = _resolve_viable_treewidth_order(
        remaining_q,
        candidate_order=candidate_order,
        width=width,
        work=work,
        remaining_adjacency=remaining_adjacency,
        remaining_edges=remaining_edges,
        remaining_max_degree=remaining_max_degree,
        allow_generic_remaining=allow_generic_remaining,
        target_remaining_width=target_remaining_width,
        use_one_shot_surrogate=use_one_shot_surrogate,
        local_hint_order=local_hint_order,
    )
    effective_work = branch_count * work
    if viable_order is None:
        if allow_generic_remaining:
            return _generic_cutset_candidate(
                q,
                cutset_vars,
                remaining_vars,
                remaining_q=remaining_q,
                residue_data=residue_data,
                prioritize_width=prioritize_width,
                prefer_one_shot_slicing=prefer_one_shot_slicing,
                target_remaining_width=target_remaining_width,
                branch_count=branch_count,
                use_one_shot_surrogate=use_one_shot_surrogate,
                surrogate_width=width,
                surrogate_work=work,
            )
        return _infeasible_cutset_candidate(
            cutset_vars,
            prioritize_width=prioritize_width,
            prefer_one_shot_slicing=prefer_one_shot_slicing,
            target_remaining_width=target_remaining_width,
            width=width,
            effective_work=effective_work,
        )
    return _treewidth_cutset_candidate(
        q,
        cutset_vars,
        remaining_vars,
        remaining_q=remaining_q,
        viable_order=viable_order,
        residue_data=residue_data,
        prioritize_width=prioritize_width,
        prefer_one_shot_slicing=prefer_one_shot_slicing,
        target_remaining_width=target_remaining_width,
        effective_work=effective_work,
    )

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
