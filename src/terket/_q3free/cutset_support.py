"""Extracted q3-free cutset scoring helpers."""

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
    '_q3_free_component_plan_width_hint',
    '_q3_free_component_plan_work_hint',
    '_q3_free_cutset_plan_generic_penalty',
    '_q3_free_component_plan_generic_penalty',
    '_q3_free_tensor_slice_hint',
    '_direct_order_guided_q3_free_cutset_plan',
    '_candidate_q3_free_cutset_vertices',
    '_order_guided_q3_free_cutset_vertices',
    '_merge_q3_free_cutset_candidate_orders',
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


def _q3_free_component_plan_width_hint(component_plan: _Q3FreeConstraintComponentPlan) -> int:
    if component_plan.cutset_plan is not None:
        return int(component_plan.cutset_plan.remaining_width)
    if component_plan.backend == "constant":
        return 0
    if component_plan.backend == "forest":
        return 1 if len(component_plan.variables) else 0
    if component_plan.backend == "treewidth" and component_plan.order is not None:
        dummy_q = _phase_function_from_parts(
            len(component_plan.variables),
            level=component_plan.level,
            q0=Fraction(0),
            q1=[0] * len(component_plan.variables),
            q2=component_plan.q2,
            q3={},
        )
        return int(_treewidth_order_width(dummy_q, component_plan.order))
    dummy_q = _phase_function_from_parts(
        len(component_plan.variables),
        level=component_plan.level,
        q0=Fraction(0),
        q1=[0] * len(component_plan.variables),
        q2=component_plan.q2,
        q3={},
    )
    _order, width = _min_fill_cubic_order(dummy_q)
    return int(width)

def _q3_free_component_plan_work_hint(component_plan: _Q3FreeConstraintComponentPlan) -> int:
    if component_plan.cutset_plan is not None:
        return int(component_plan.cutset_plan.estimated_total_work)
    if component_plan.backend == "constant":
        return 1
    if component_plan.backend == "forest":
        return max(1, len(component_plan.variables))
    if component_plan.backend == "treewidth" and component_plan.order is not None:
        dummy_q = _phase_function_from_parts(
            len(component_plan.variables),
            level=component_plan.level,
            q0=Fraction(0),
            q1=[0] * len(component_plan.variables),
            q2=component_plan.q2,
            q3={},
        )
        return max(1, _estimate_treewidth_dp_work(dummy_q, component_plan.order))
    dummy_q = _phase_function_from_parts(
        len(component_plan.variables),
        level=component_plan.level,
        q0=Fraction(0),
        q1=[0] * len(component_plan.variables),
        q2=component_plan.q2,
        q3={},
    )
    order, _width = _min_fill_cubic_order(dummy_q)
    return max(1, _estimate_treewidth_dp_work(dummy_q, order))

def _q3_free_cutset_plan_generic_penalty(
    plan: _Q3FreeCutsetConditioningPlan | None,
) -> int:
    if plan is None:
        return 1 << 20
    penalty = int(plan.remaining_backend == "generic")
    for component_plan in plan.remaining_components:
        penalty += _q3_free_component_plan_generic_penalty(component_plan)
    return penalty

def _q3_free_component_plan_generic_penalty(
    component_plan: _Q3FreeConstraintComponentPlan,
) -> int:
    penalty = int(component_plan.backend == "generic")
    if component_plan.cutset_plan is not None:
        penalty += _q3_free_cutset_plan_generic_penalty(component_plan.cutset_plan)
    return penalty

def _q3_free_tensor_slice_hint(q: PhaseFunction) -> tuple[int, ...]:
    """Return preferred cutset variables from a sliced tensor-contraction plan."""
    _cfg = _get_solver_config()
    if (
        q.n < _cfg.tensor_hint_min_vars
        or q.n > _cfg.tensor_hint_max_vars
        or not q.q2
        or not _kahypar_available()
    ):
        return ()

    cache_key = (
        _q_structure_key(q),
        int(_cfg.tensor_hint_target_width),
        int(_cfg.tensor_hint_max_repeats),
        float(_cfg.tensor_hint_max_time),
        bool(_kahypar_available()),
    )
    cached = _STRUCTURE_Q3_FREE_TENSOR_HINT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    qtn = _get_quimb_tensor_module()
    if qtn is None:
        return ()
    try:
        import cotengra as ctg
    except Exception:
        return ()

    _scalar, factors = _build_cubic_factors(q)
    if not factors:
        return ()

    del qtn
    ordered_scopes = [scope for scope, _table in sorted(factors.items())]
    inputs = [tuple(f"v{var}" for var in scope) for scope in ordered_scopes]
    size_dict = {f"v{var}": 2 for var in range(q.n)}

    optimizer = ctg.HyperOptimizer(
        methods=["kahypar"],
        minimize="flops",
        max_repeats=int(_cfg.tensor_hint_max_repeats),
        max_time=float(_cfg.tensor_hint_max_time),
        parallel=False,
        slicing_reconf_opts={
            "target_size": 1 << int(_cfg.tensor_hint_target_width),
        },
        reconf_opts={},
        progbar=False,
    )

    try:
        tree = ctg.array_contract_tree(
            inputs,
            output=(),
            size_dict=size_dict,
            optimize=optimizer,
            canonicalize=False,
        )
    except Exception:
        return ()

    sliced_inds = getattr(tree, "sliced_inds", None) or {}
    hint: list[int] = []
    for ind in sliced_inds:
        if isinstance(ind, str) and ind.startswith("v") and ind[1:].isdigit():
            hint.append(int(ind[1:]))
    result = tuple(sorted(set(hint)))
    _STRUCTURE_Q3_FREE_TENSOR_HINT_CACHE[cache_key] = result
    return result

def _direct_order_guided_q3_free_cutset_plan(
    q: PhaseFunction,
    adjacency: Sequence[set[int]],
    *,
    preferred: set[int] | None = None,
    max_size: int,
    target_remaining_width: int | None = None,
    allow_generic_remaining: bool = False,
) -> _Q3FreeCutsetConditioningPlan | None:
    """Try a tiny fixed-budget cutset plan from cheap order frontier peaks.

    Giant low-degree q3-free kernels spend most of their time in search
    orchestration, not one candidate evaluation. Before launching a broader
    one-shot search, try prefixes of the order-guided peak ranking induced by
    cheap chronological orders. This keeps the cost linear in ``max_size``.
    """
    max_size = max(0, int(max_size))
    if q.q3 or max_size <= 0 or q.n <= 1:
        return None

    preferred = set() if preferred is None else {int(var) for var in preferred}
    best_eval: _Q3FreeCutsetCandidateEvaluation | None = None
    seen_cutsets: set[tuple[int, ...]] = set()

    for cheap_order in _iter_q3_free_cheap_order_hints(q.n, q=q):
        order_guided = _order_guided_q3_free_cutset_vertices(
            adjacency,
            candidate_orders=(cheap_order,),
            preferred=preferred,
            max_candidates=max_size,
        )
        if not order_guided:
            continue
        for size in range(1, min(len(order_guided), max_size) + 1):
            cutset = tuple(int(var) for var in order_guided[:size])
            if cutset in seen_cutsets:
                continue
            seen_cutsets.add(cutset)
            evaluation = _evaluate_q3_free_cutset_candidate(
                q,
                cutset,
                remaining_order_hint=cheap_order,
                prioritize_width=True,
                target_remaining_width=target_remaining_width,
                allow_generic_remaining=allow_generic_remaining,
                prefer_one_shot_slicing=True,
            )
            if evaluation is None or not evaluation.viable or evaluation.plan is None:
                continue
            if best_eval is None or evaluation.score < best_eval.score:
                best_eval = evaluation
                if (
                    evaluation.plan.remaining_backend == "treewidth"
                    and evaluation.plan.remaining_width <= _q3_free_treewidth_width_limit()
                ):
                    return evaluation.plan
                if (
                    target_remaining_width is not None
                    and evaluation.plan.remaining_width <= int(target_remaining_width)
                    and _q3_free_cutset_plan_generic_penalty(evaluation.plan) == 0
                ):
                    return evaluation.plan

    return None if best_eval is None else best_eval.plan

def _candidate_q3_free_cutset_vertices(
    adjacency: Sequence[set[int]],
    *,
    preferred: set[int] | None = None,
    max_candidates: int = _Q3_FREE_CUTSET_CANDIDATE_POOL,
) -> tuple[int, ...]:
    preferred = set() if preferred is None else preferred
    scored: list[tuple[int, int, int, int, int]] = []
    for var, neighbors in enumerate(adjacency):
        triangle_count = 0
        ordered_neighbors = tuple(sorted(neighbors))
        for idx, left in enumerate(ordered_neighbors):
            left_neighbors = adjacency[left]
            for right in ordered_neighbors[idx + 1 :]:
                if right in left_neighbors:
                    triangle_count += 1
        scored.append(
            (
                int(var in preferred),
                len(neighbors),
                sum(len(adjacency[neighbor]) for neighbor in neighbors),
                triangle_count,
                var,
            )
        )
    scored.sort(reverse=True)
    return tuple(var for *_score, var in scored[: min(len(scored), int(max_candidates))])

def _order_guided_q3_free_cutset_vertices(
    adjacency: Sequence[set[int]],
    *,
    candidate_orders: Sequence[Sequence[int]],
    preferred: set[int] | None = None,
    max_candidates: int = _Q3_FREE_ONE_SHOT_CUTSET_CANDIDATE_POOL,
) -> tuple[int, ...]:
    preferred = set() if preferred is None else {int(var) for var in preferred}
    n_vars = len(adjacency)
    if n_vars <= 1 or not candidate_orders:
        return ()

    aggregate_scores: dict[int, list[int]] = {}
    peak_cap = max(1, min(int(max_candidates), _Q3_FREE_ORDER_GUIDED_CUTSET_MAX_PEAKS))

    for order in candidate_orders:
        order_list = [int(var) for var in order]
        if len(order_list) != n_vars:
            continue
        positions = {var: idx for idx, var in enumerate(order_list)}
        if len(positions) != n_vars:
            continue

        diff = [0] * n_vars
        interval_lo = [positions[var] for var in range(n_vars)]
        interval_hi = [positions[var] for var in range(n_vars)]

        for left, neighbors in enumerate(adjacency):
            left_pos = positions[left]
            for right in neighbors:
                if right <= left:
                    continue
                right_pos = positions[right]
                lo = min(left_pos, right_pos)
                hi = max(left_pos, right_pos)
                if lo < hi:
                    diff[lo] += 1
                    diff[hi] -= 1
                interval_lo[left] = min(interval_lo[left], right_pos)
                interval_hi[left] = max(interval_hi[left], right_pos)
                interval_lo[right] = min(interval_lo[right], left_pos)
                interval_hi[right] = max(interval_hi[right], left_pos)

        running = 0
        cut_widths: list[int] = []
        for cut in range(n_vars - 1):
            running += diff[cut]
            cut_widths.append(running)
        if not cut_widths:
            continue

        peak_cuts = sorted(
            range(len(cut_widths)),
            key=lambda cut: (cut_widths[cut], -abs((2 * cut + 1) - n_vars)),
            reverse=True,
        )[:peak_cap]
        if not peak_cuts or cut_widths[peak_cuts[0]] <= 0:
            continue

        for var in range(n_vars):
            span_lo = interval_lo[var]
            span_hi = interval_hi[var]
            if span_hi <= span_lo:
                continue
            peak_hits = 0
            peak_weight = 0
            best_closeness = -n_vars
            for cut in peak_cuts:
                if span_lo <= cut < span_hi:
                    peak_hits += 1
                    peak_weight += cut_widths[cut]
                    best_closeness = max(
                        best_closeness,
                        -min(abs(positions[var] - cut), abs(positions[var] - (cut + 1))),
                    )
            if peak_hits == 0:
                continue
            scores = aggregate_scores.setdefault(var, [0, 0, 0, 0, 0, 0])
            scores[0] += peak_weight
            scores[1] += peak_hits
            scores[2] += len(adjacency[var])
            scores[3] = max(scores[3], span_hi - span_lo)
            scores[4] = max(scores[4], best_closeness)
            scores[5] += int(var in preferred)

    ranked = sorted(
        aggregate_scores,
        key=lambda var: (
            aggregate_scores[var][5],
            aggregate_scores[var][0],
            aggregate_scores[var][1],
            aggregate_scores[var][4],
            aggregate_scores[var][2],
            aggregate_scores[var][3],
            -var,
        ),
        reverse=True,
    )
    return tuple(ranked[: min(len(ranked), int(max_candidates))])

def _merge_q3_free_cutset_candidate_orders(
    *candidate_orders: Sequence[int],
    max_candidates: int,
) -> tuple[int, ...]:
    merged: list[int] = []
    seen: set[int] = set()
    for order in candidate_orders:
        for var in order:
            if var in seen:
                continue
            merged.append(int(var))
            seen.add(int(var))
            if len(merged) >= int(max_candidates):
                return tuple(merged)
    return tuple(merged)


def _separator_ranked_q3_free_cutset_vertices(
    adjacency: Sequence[set[int]],
    *,
    preferred: set[int] | None = None,
    max_candidates: int = _Q3_FREE_ONE_SHOT_CUTSET_CANDIDATE_POOL,
) -> tuple[int, ...]:
    preferred = set() if preferred is None else {int(var) for var in preferred}
    all_vertices = tuple(range(len(adjacency)))
    if not all_vertices:
        return ()

    ranked: list[int] = []
    seen: set[int] = set()
    component_heap: list[tuple[int, tuple[int, ...]]] = [
        (-len(component), tuple(component))
        for component in _connected_components_on_vertices(adjacency, all_vertices)
        if len(component) >= _Q2_SEPARATOR_ORDER_MIN_VARS
    ]
    heapq.heapify(component_heap)

    while component_heap and len(ranked) < int(max_candidates):
        _neg_size, component = heapq.heappop(component_heap)
        separator_info = _choose_pair_graph_layer_separator(adjacency, component)
        if separator_info is None:
            continue
        separator, components = separator_info
        ordered_separator = sorted(
            separator,
            key=lambda var: (
                int(var in preferred),
                len(adjacency[var]),
                sum(len(adjacency[neighbor]) for neighbor in adjacency[var]),
                -int(var),
            ),
            reverse=True,
        )
        for var in ordered_separator:
            if var in seen:
                continue
            ranked.append(int(var))
            seen.add(int(var))
            if len(ranked) >= int(max_candidates):
                break
        for subcomponent in components:
            if len(subcomponent) >= _Q2_SEPARATOR_ORDER_MIN_VARS:
                heapq.heappush(component_heap, (-len(subcomponent), tuple(subcomponent)))

    return tuple(ranked)

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
