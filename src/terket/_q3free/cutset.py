"""Extracted q3-free cutset planners."""

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
    '_finalize_q3_free_cutset_conditioning_plan',
    '_attach_q3_free_cutset_runtime_cache',
    '_build_q3_free_cutset_conditioning_plan_uncached',
    '_q3_free_cutset_conditioning_plan',
    '_q3_free_one_shot_cutset_conditioning_plan',
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


def _finalize_q3_free_cutset_conditioning_plan(
    plan: _Q3FreeCutsetConditioningPlan,
    *,
    prefer_one_shot_slicing: bool = False,
) -> _Q3FreeCutsetConditioningPlan:
    """Fill in generic remaining-component plans only for the chosen cutset."""
    if (
        plan.remaining_backend != "generic"
        or plan.remaining_components
        or (not plan.remaining_q2 and not plan.remaining_isolated_vars)
    ):
        return plan

    remaining_q = _phase_function_from_parts(
        len(plan.remaining_vars),
        level=plan.level,
        q0=Fraction(0),
        q1=[0] * len(plan.remaining_vars),
        q2=plan.remaining_q2,
        q3={},
    )
    isolated_vars, component_plans = _plan_q3_free_constraint_components(
        remaining_q,
        0,
        allow_tensor_contraction=True,
        prefer_reusable_decomposition=False,
        prefer_one_shot_slicing=prefer_one_shot_slicing,
    )
    component_width = max(
        (_q3_free_component_plan_width_hint(component_plan) for component_plan in component_plans),
        default=0,
    )
    component_work = max(
        1,
        sum(_q3_free_component_plan_work_hint(component_plan) for component_plan in component_plans),
    )
    branch_count = 1 << len(plan.cutset_vars)
    return _Q3FreeCutsetConditioningPlan(
        level=plan.level,
        cutset_vars=plan.cutset_vars,
        remaining_vars=plan.remaining_vars,
        remaining_backend=plan.remaining_backend,
        remaining_q2=plan.remaining_q2,
        remaining_order=plan.remaining_order,
        cutset_remaining_q2_residue=plan.cutset_remaining_q2_residue,
        cutset_cutset_left=plan.cutset_cutset_left,
        cutset_cutset_right=plan.cutset_cutset_right,
        cutset_cutset_residue=plan.cutset_cutset_residue,
        native_treewidth_plan=plan.native_treewidth_plan,
        remaining_isolated_vars=tuple(int(var) for var in isolated_vars),
        remaining_components=tuple(component_plans),
        remaining_width=max(plan.remaining_width, component_width),
        estimated_total_work=max(plan.estimated_total_work, branch_count * component_work),
    )

def _attach_q3_free_cutset_runtime_cache(
    plan: _Q3FreeCutsetConditioningPlan,
) -> _Q3FreeCutsetConditioningPlan:
    """Attach reusable branch-side runtime arrays to a cutset plan."""
    if plan.branch_bits is not None:
        return plan

    cutset_size = len(plan.cutset_vars)
    branch_count = 1 << cutset_size
    branch_masks = np.arange(branch_count, dtype=np.uint64)
    branch_bits = _branch_assignment_bits(branch_masks, cutset_size).astype(np.int64)

    if plan.cutset_cutset_residue.size:
        branch_pair_residue = np.zeros(branch_count, dtype=np.int64)
        for left, right, residue in zip(
            plan.cutset_cutset_left,
            plan.cutset_cutset_right,
            plan.cutset_cutset_residue,
        ):
            branch_pair_residue = (
                branch_pair_residue
                + int(residue) * branch_bits[:, int(left)] * branch_bits[:, int(right)]
            ) % (1 << int(plan.level))
    else:
        branch_pair_residue = np.zeros(branch_count, dtype=np.int64)

    if plan.cutset_remaining_q2_residue.size:
        branch_remaining_shift = (
            branch_bits @ np.asarray(plan.cutset_remaining_q2_residue, dtype=np.int64)
        ) % (1 << int(plan.level))
    else:
        branch_remaining_shift = np.zeros(
            (branch_count, len(plan.remaining_vars)),
            dtype=np.int64,
        )

    return _Q3FreeCutsetConditioningPlan(
        level=plan.level,
        cutset_vars=plan.cutset_vars,
        remaining_vars=plan.remaining_vars,
        remaining_backend=plan.remaining_backend,
        remaining_q2=plan.remaining_q2,
        remaining_order=plan.remaining_order,
        cutset_remaining_q2_residue=plan.cutset_remaining_q2_residue,
        cutset_cutset_left=plan.cutset_cutset_left,
        cutset_cutset_right=plan.cutset_cutset_right,
        cutset_cutset_residue=plan.cutset_cutset_residue,
        native_treewidth_plan=plan.native_treewidth_plan,
        remaining_isolated_vars=plan.remaining_isolated_vars,
        remaining_components=plan.remaining_components,
        remaining_width=plan.remaining_width,
        estimated_total_work=plan.estimated_total_work,
        branch_bits=branch_bits,
        branch_pair_residue=branch_pair_residue,
        branch_remaining_shift=branch_remaining_shift,
    )

def _build_q3_free_cutset_conditioning_plan_uncached(
    q: PhaseFunction,
    *,
    max_size: int = _Q3_FREE_CUTSET_MAX_SIZE,
    candidate_pool: int = _Q3_FREE_CUTSET_CANDIDATE_POOL,
    beam_width: int = _Q3_FREE_CUTSET_BEAM_WIDTH,
    branches_per_state: int = _Q3_FREE_CUTSET_BRANCHES_PER_STATE,
    prioritize_width: bool = False,
    target_remaining_width: int | None = None,
    candidate_override: Sequence[int] | None = None,
    remaining_order_hint: Sequence[int] | None = None,
    allow_generic_remaining: bool = False,
    prefer_one_shot_slicing: bool = False,
) -> _Q3FreeCutsetConditioningPlan | None:
    if q.n <= 1 or not q.q2:
        return None

    adjacency, edges = _q3_free_graph(q)
    depth, chords = _q3_free_spanning_data(adjacency, edges)
    tensor_hint = _q3_free_tensor_slice_hint(q)
    preferred = set(_select_feedback_vertices(q.n, chords, depth))
    preferred.update(tensor_hint)
    if candidate_override is None:
        order_guided_candidates = _order_guided_q3_free_cutset_vertices(
            adjacency,
                candidate_orders=_iter_q3_free_cheap_order_hints(q.n, q=q),
            preferred=preferred,
            max_candidates=int(candidate_pool),
        )
        generic_candidates = _candidate_q3_free_cutset_vertices(
            adjacency,
            preferred=preferred,
            max_candidates=int(candidate_pool),
        )
        candidates = _merge_q3_free_cutset_candidate_orders(
            order_guided_candidates,
            generic_candidates,
            max_candidates=int(candidate_pool),
        )
    else:
        candidates = _merge_q3_free_cutset_candidate_orders(
            candidate_override,
            max_candidates=int(candidate_pool),
        )
    if not candidates:
        return None

    best_evaluation: _Q3FreeCutsetCandidateEvaluation | None = None
    evaluation_cache: dict[tuple[int, ...], _Q3FreeCutsetCandidateEvaluation | None] = {}
    projection_cache: dict[tuple[int, ...], _Q3FreeResidualProjection | None] = {}
    remaining_universe = tuple(range(q.n))
    frontier: list[tuple[tuple[int, ...], _Q3FreeCutsetCandidateEvaluation | None]] = [((), None)]
    max_size = min(int(max_size), len(candidates))
    use_native_candidate_shortlist = (
        prefer_one_shot_slicing
        and allow_generic_remaining
        and q.n >= _Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS
        and not q.q3
    )
    giant_low_degree_surrogate = (
        use_native_candidate_shortlist
        and max((len(neighbors) for neighbors in adjacency), default=0) <= 4
    )
    if giant_low_degree_surrogate:
        max_size = min(int(max_size), 6)
        candidate_pool = min(int(candidate_pool), 16)
        beam_width = min(int(beam_width), 3)
        branches_per_state = min(int(branches_per_state), 2)

    def global_remaining_order_hint(
        evaluation: _Q3FreeCutsetCandidateEvaluation | None,
    ) -> tuple[int, ...] | None:
        if (
            evaluation is None
            or not evaluation.viable
            or evaluation.plan is None
            or evaluation.plan.remaining_backend != "treewidth"
            or not evaluation.plan.remaining_order
        ):
            return None
        return tuple(
            int(evaluation.plan.remaining_vars[idx])
            for idx in evaluation.plan.remaining_order
        )

    def cached_evaluation(
        cutset_vars: tuple[int, ...],
        *,
        local_remaining_order_hint: Sequence[int] | None = None,
        parent_evaluation: _Q3FreeCutsetCandidateEvaluation | None = None,
    ) -> _Q3FreeCutsetCandidateEvaluation | None:
        cached = evaluation_cache.get(cutset_vars)
        if (
            local_remaining_order_hint is None
            and (cached is not None or cutset_vars in evaluation_cache)
        ):
            return cached
        parent_projection = None
        if (
            parent_evaluation is not None
            and parent_evaluation.plan is not None
            and set(parent_evaluation.cutset_vars) < set(cutset_vars)
        ):
            parent_projection = projection_cache.get(parent_evaluation.cutset_vars)
        residual_projection = _build_q3_free_residual_projection(
            q,
            cutset_vars,
            remaining_universe=remaining_universe,
            parent_projection=parent_projection,
        )
        projection_cache[cutset_vars] = residual_projection
        evaluation = _evaluate_q3_free_cutset_candidate(
            q,
            cutset_vars,
            remaining_universe=remaining_universe,
            residual_projection=residual_projection,
            remaining_order_hint=local_remaining_order_hint or remaining_order_hint,
            prioritize_width=prioritize_width,
            target_remaining_width=target_remaining_width,
            allow_generic_remaining=allow_generic_remaining,
            prefer_one_shot_slicing=prefer_one_shot_slicing,
        )
        cached = evaluation_cache.get(cutset_vars)
        if cached is None or (
            evaluation is not None
            and (
                not cached.viable
                or evaluation.score < cached.score
            )
        ):
            evaluation_cache[cutset_vars] = evaluation
            cached = evaluation
        elif cutset_vars not in evaluation_cache:
            evaluation_cache[cutset_vars] = evaluation
            cached = evaluation
        return cached

    def meets_width_target(evaluation: _Q3FreeCutsetCandidateEvaluation | None) -> bool:
        return bool(
            prioritize_width
            and target_remaining_width is not None
            and evaluation is not None
            and evaluation.viable
            and evaluation.plan is not None
            and evaluation.plan.remaining_width <= int(target_remaining_width)
        )

    def greedy_result_is_good_enough(
        evaluation: _Q3FreeCutsetCandidateEvaluation | None,
    ) -> bool:
        if evaluation is None or not evaluation.viable or evaluation.plan is None:
            return False
        if evaluation.plan.remaining_backend != "treewidth":
            return False
        if _q3_free_cutset_plan_generic_penalty(evaluation.plan) != 0:
            return False
        if not giant_low_degree_surrogate:
            return False
        return True

    def maybe_update_best(
        evaluation: _Q3FreeCutsetCandidateEvaluation | None,
    ) -> bool:
        nonlocal best_evaluation
        if evaluation is None or not evaluation.viable:
            return False
        if best_evaluation is None or evaluation.score < best_evaluation.score:
            best_evaluation = evaluation
            return True
        return False

    def width_value(evaluation: _Q3FreeCutsetCandidateEvaluation | None) -> int | None:
        if evaluation is None or not evaluation.viable or evaluation.plan is None:
            return None
        return int(evaluation.plan.remaining_width)

    def local_search_improve(
        seed_evaluation: _Q3FreeCutsetCandidateEvaluation | None,
    ) -> _Q3FreeCutsetCandidateEvaluation | None:
        if (
            not use_native_candidate_shortlist
            or seed_evaluation is None
            or not seed_evaluation.viable
            or len(seed_evaluation.cutset_vars) >= max_size
        ):
            return seed_evaluation
        current = seed_evaluation
        for _pass in range(_Q3_FREE_ONE_SHOT_LOCAL_SEARCH_PASSES):
            ranked_extensions = _native_rank_q3_free_cutset_extensions(
                q,
                selected_vars=current.cutset_vars,
                candidate_vars=[candidate for candidate in candidates if candidate not in current.cutset_vars],
                remaining_order_hint=global_remaining_order_hint(current),
            )
            if not ranked_extensions:
                break
            neighbor_best = current
            top_candidates = [
                candidate for candidate, _width, _work in ranked_extensions[: _Q3_FREE_ONE_SHOT_LOCAL_SEARCH_TOPK]
            ]
            for candidate in top_candidates:
                if len(current.cutset_vars) < max_size:
                    evaluation = cached_evaluation(
                        tuple(sorted(current.cutset_vars + (candidate,))),
                        local_remaining_order_hint=global_remaining_order_hint(current),
                        parent_evaluation=current,
                    )
                    if (
                        evaluation is not None
                        and evaluation.viable
                        and evaluation.score < neighbor_best.score
                    ):
                        neighbor_best = evaluation
                for remove_var in current.cutset_vars:
                    swapped = tuple(
                        sorted(var for var in current.cutset_vars if var != remove_var) + [candidate]
                    )
                    evaluation = cached_evaluation(
                        swapped,
                        local_remaining_order_hint=global_remaining_order_hint(current),
                        parent_evaluation=current,
                    )
                    if (
                        evaluation is not None
                        and evaluation.viable
                        and evaluation.score < neighbor_best.score
                    ):
                        neighbor_best = evaluation
            if len(current.cutset_vars) > 1:
                for remove_var in current.cutset_vars:
                    dropped = tuple(var for var in current.cutset_vars if var != remove_var)
                    evaluation = cached_evaluation(
                        dropped,
                        local_remaining_order_hint=global_remaining_order_hint(current),
                    )
                    if (
                        evaluation is not None
                        and evaluation.viable
                        and evaluation.score < neighbor_best.score
                    ):
                        neighbor_best = evaluation
            if neighbor_best is current:
                break
            current = neighbor_best
            maybe_update_best(current)
            if meets_width_target(current):
                break
        return current

    if tensor_hint:
        hinted_vars = [var for var in tensor_hint if var in candidates]
        for size in range(1, min(max_size, len(hinted_vars)) + 1):
            evaluation = cached_evaluation(tuple(sorted(hinted_vars[:size])))
            maybe_update_best(evaluation)

    selected: list[int] = []
    selected_evaluation: _Q3FreeCutsetCandidateEvaluation | None = None
    greedy_remaining_order_hint = remaining_order_hint
    for _size in range(1, max_size + 1):
        best_choice: tuple[int, _Q3FreeCutsetCandidateEvaluation] | None = None
        candidate_iterable: Sequence[int] = candidates
        if use_native_candidate_shortlist:
            remaining_candidates = [candidate for candidate in candidates if candidate not in selected]
            ranked_extensions = _native_rank_q3_free_cutset_extensions(
                q,
                selected_vars=selected,
                candidate_vars=remaining_candidates,
                remaining_order_hint=greedy_remaining_order_hint,
            )
            if ranked_extensions is not None:
                shortlist_size = max(int(branches_per_state) * 2, int(beam_width), 8)
                candidate_iterable = tuple(
                    candidate for candidate, _width, _work in ranked_extensions[:shortlist_size]
                )
        for candidate in candidate_iterable:
            if candidate in selected:
                continue
            evaluation = cached_evaluation(
                tuple(sorted(selected + [candidate])),
                local_remaining_order_hint=greedy_remaining_order_hint,
                parent_evaluation=selected_evaluation,
            )
            if evaluation is None:
                continue
            if best_choice is None or evaluation.score < best_choice[1].score:
                best_choice = (candidate, evaluation)
        if best_choice is None:
            break
        selected.append(best_choice[0])
        selected_evaluation = best_choice[1]
        greedy_remaining_order_hint = global_remaining_order_hint(best_choice[1])
        best_width_before = width_value(best_evaluation)
        if maybe_update_best(best_choice[1]):
            best_width_after = width_value(best_evaluation)
            width_improved = (
                best_width_after is not None
                and (best_width_before is None or best_width_after < best_width_before)
            )
            if width_improved:
                selected_evaluation = local_search_improve(selected_evaluation)
            if giant_low_degree_surrogate and meets_width_target(best_evaluation):
                break
            if not prioritize_width or meets_width_target(selected_evaluation):
                break
        if giant_low_degree_surrogate and best_evaluation is not None and meets_width_target(best_evaluation):
            break

    if greedy_result_is_good_enough(best_evaluation):
        plan = best_evaluation.plan
        assert plan is not None
        return _attach_q3_free_cutset_runtime_cache(plan)

    if (
        best_evaluation is not None
        and q.n >= 64
        and len(best_evaluation.cutset_vars) <= 2
        and (not prioritize_width or meets_width_target(best_evaluation))
    ):
        plan = best_evaluation.plan
        assert plan is not None
        remaining_order = plan.remaining_order
        remaining_width = plan.remaining_width
        estimated_total_work = plan.estimated_total_work
        if plan.remaining_backend == "treewidth" and remaining_order:
            refined_q = _phase_function_from_parts(
                len(plan.remaining_vars),
                level=plan.level,
                q0=Fraction(0),
                q1=[0] * len(plan.remaining_vars),
                q2=plan.remaining_q2,
                q3={},
            )
            refined_order, refined_width = _finalize_q3_free_treewidth_order(
                refined_q,
                remaining_order,
            )
            remaining_order = tuple(int(var) for var in refined_order)
            remaining_width = int(refined_width)
            estimated_total_work = max(
                1,
                (1 << len(plan.cutset_vars)) * _estimate_treewidth_dp_work(refined_q, refined_order),
            )
        return _attach_q3_free_cutset_runtime_cache(_Q3FreeCutsetConditioningPlan(
            level=plan.level,
            cutset_vars=plan.cutset_vars,
            remaining_vars=plan.remaining_vars,
            remaining_backend=plan.remaining_backend,
            remaining_q2=plan.remaining_q2,
            remaining_order=remaining_order,
            cutset_remaining_q2_residue=plan.cutset_remaining_q2_residue,
            cutset_cutset_left=plan.cutset_cutset_left,
            cutset_cutset_right=plan.cutset_cutset_right,
            cutset_cutset_residue=plan.cutset_cutset_residue,
            native_treewidth_plan=(
                _build_native_q3_free_treewidth_plan(
                    n_vars=len(plan.remaining_vars),
                    level=plan.level,
                    q2=plan.remaining_q2,
                    order=remaining_order,
                )
                if plan.remaining_backend == "treewidth"
                else None
            ),
            remaining_isolated_vars=plan.remaining_isolated_vars,
            remaining_components=plan.remaining_components,
            remaining_width=remaining_width,
            estimated_total_work=estimated_total_work,
        ))

    stagnation_steps = 0
    best_width_seen = width_value(best_evaluation)
    for _size in range(1, max_size + 1):
        expansions: list[tuple[tuple[int, ...], tuple[int, ...], _Q3FreeCutsetCandidateEvaluation]] = []
        seen_cutsets: set[tuple[int, ...]] = set()
        for selected_indexes, parent_evaluation in frontier:
            next_idx = selected_indexes[-1] + 1 if selected_indexes else 0
            candidate_indexes: list[int] = list(
                range(
                    next_idx,
                    min(len(candidates), next_idx + int(branches_per_state)),
                )
            )
            if use_native_candidate_shortlist:
                prefix_candidates = [candidates[idx] for idx in selected_indexes]
                available_candidates = [candidates[idx] for idx in range(next_idx, len(candidates))]
                ranked_extensions = _native_rank_q3_free_cutset_extensions(
                    q,
                    selected_vars=prefix_candidates,
                    candidate_vars=available_candidates,
                    remaining_order_hint=global_remaining_order_hint(parent_evaluation),
                )
                if ranked_extensions is not None:
                    shortlisted = {
                        int(candidate)
                        for candidate, _width, _work in ranked_extensions[: max(int(branches_per_state), 4)]
                    }
                    candidate_indexes = [
                        idx for idx in range(next_idx, len(candidates))
                        if candidates[idx] in shortlisted
                    ]
            for candidate_idx in candidate_indexes:
                expanded_indexes = selected_indexes + (candidate_idx,)
                cutset_vars = tuple(sorted(candidates[idx] for idx in expanded_indexes))
                if cutset_vars in seen_cutsets:
                    continue
                seen_cutsets.add(cutset_vars)
                evaluation = cached_evaluation(
                    cutset_vars,
                    local_remaining_order_hint=global_remaining_order_hint(parent_evaluation),
                    parent_evaluation=parent_evaluation,
                )
                if evaluation is None:
                    continue
                expansions.append((evaluation.score, expanded_indexes, evaluation))

        if not expansions:
            break

        expansions.sort(key=lambda item: item[0])
        improved_this_round = False
        width_improved_this_round = False
        for _score, _expanded_indexes, evaluation in expansions:
            if maybe_update_best(evaluation):
                improved_this_round = True
                candidate_width = width_value(best_evaluation)
                if (
                    candidate_width is not None
                    and (best_width_seen is None or candidate_width < best_width_seen)
                ):
                    best_width_seen = candidate_width
                    width_improved_this_round = True

        if improved_this_round and (not giant_low_degree_surrogate or width_improved_this_round):
            best_evaluation = local_search_improve(best_evaluation)
            improved_this_round = True

        frontier_limit = int(beam_width)
        if giant_low_degree_surrogate and not width_improved_this_round:
            frontier_limit = 1
        frontier = [
            (expanded_indexes, evaluation)
            for _score, expanded_indexes, _evaluation in expansions[: frontier_limit]
            for evaluation in (_evaluation,)
        ]
        if improved_this_round:
            stagnation_steps = 0
            if meets_width_target(best_evaluation):
                break
        else:
            stagnation_steps += 1
            if (
                giant_low_degree_surrogate
                and best_evaluation is not None
                and not width_improved_this_round
            ):
                break
            if (
                use_native_candidate_shortlist
                and best_evaluation is not None
                and stagnation_steps >= _Q3_FREE_ONE_SHOT_STAGNATION_LIMIT
            ):
                break

    if best_evaluation is None:
        return None
    plan = best_evaluation.plan
    if plan is not None and plan.remaining_backend == "generic":
        plan = _finalize_q3_free_cutset_conditioning_plan(
            plan,
            prefer_one_shot_slicing=prefer_one_shot_slicing,
        )
    if (
        plan is not None
        and plan.remaining_backend == "treewidth"
        and plan.remaining_order
    ):
        refined_q = _phase_function_from_parts(
            len(plan.remaining_vars),
            level=plan.level,
            q0=Fraction(0),
            q1=[0] * len(plan.remaining_vars),
            q2=plan.remaining_q2,
            q3={},
        )
        refined_order, refined_width = _finalize_q3_free_treewidth_order(
            refined_q,
            plan.remaining_order,
        )
        refined_native_treewidth_plan = _build_native_q3_free_treewidth_plan(
            n_vars=len(plan.remaining_vars),
            level=plan.level,
            q2=plan.remaining_q2,
            order=refined_order,
        )
        plan = _Q3FreeCutsetConditioningPlan(
            level=plan.level,
            cutset_vars=plan.cutset_vars,
            remaining_vars=plan.remaining_vars,
            remaining_backend=plan.remaining_backend,
            remaining_q2=plan.remaining_q2,
            remaining_order=tuple(int(var) for var in refined_order),
            cutset_remaining_q2_residue=plan.cutset_remaining_q2_residue,
            cutset_cutset_left=plan.cutset_cutset_left,
            cutset_cutset_right=plan.cutset_cutset_right,
            cutset_cutset_residue=plan.cutset_cutset_residue,
            native_treewidth_plan=refined_native_treewidth_plan,
            remaining_isolated_vars=plan.remaining_isolated_vars,
            remaining_components=plan.remaining_components,
            remaining_width=int(refined_width),
            estimated_total_work=max(
                1,
                (1 << len(plan.cutset_vars)) * _estimate_treewidth_dp_work(refined_q, refined_order),
            ),
        )
    if (
        plan is not None
        and plan.remaining_backend == "treewidth"
        and plan.native_treewidth_plan is None
    ):
        return _attach_q3_free_cutset_runtime_cache(_Q3FreeCutsetConditioningPlan(
            level=plan.level,
            cutset_vars=plan.cutset_vars,
            remaining_vars=plan.remaining_vars,
            remaining_backend=plan.remaining_backend,
            remaining_q2=plan.remaining_q2,
            remaining_order=plan.remaining_order,
            cutset_remaining_q2_residue=plan.cutset_remaining_q2_residue,
            cutset_cutset_left=plan.cutset_cutset_left,
            cutset_cutset_right=plan.cutset_cutset_right,
            cutset_cutset_residue=plan.cutset_cutset_residue,
            native_treewidth_plan=_build_native_q3_free_treewidth_plan(
                n_vars=len(plan.remaining_vars),
                level=plan.level,
                q2=plan.remaining_q2,
                order=plan.remaining_order,
            ),
            remaining_isolated_vars=plan.remaining_isolated_vars,
            remaining_components=plan.remaining_components,
            remaining_width=plan.remaining_width,
            estimated_total_work=plan.estimated_total_work,
        ))
    return _attach_q3_free_cutset_runtime_cache(plan)

def _q3_free_cutset_conditioning_plan(
    q: PhaseFunction,
    *,
    max_size: int | None = None,
    candidate_pool: int | None = None,
    beam_width: int | None = None,
    branches_per_state: int | None = None,
    prioritize_width: bool = False,
    target_remaining_width: int | None = None,
    allow_generic_remaining: bool = False,
    prefer_one_shot_slicing: bool = False,
) -> _Q3FreeCutsetConditioningPlan | None:
    _cfg = _get_solver_config()
    max_size = _cfg.cutset_max_size if max_size is None else int(max_size)
    candidate_pool = _cfg.cutset_candidate_pool if candidate_pool is None else int(candidate_pool)
    beam_width = _cfg.cutset_beam_width if beam_width is None else int(beam_width)
    branches_per_state = _cfg.cutset_branches_per_state if branches_per_state is None else int(branches_per_state)
    cache_key = (
        _q_structure_key(q),
        int(max_size),
        int(candidate_pool),
        int(beam_width),
        int(branches_per_state),
        bool(prioritize_width),
        -1 if target_remaining_width is None else int(target_remaining_width),
        bool(allow_generic_remaining),
        bool(prefer_one_shot_slicing),
        bool(_quimb_import_enabled()),
        bool(_kahypar_available()),
    )
    cached = _STRUCTURE_Q3_FREE_CUTSET_PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    plan = _build_q3_free_cutset_conditioning_plan_uncached(
        q,
        max_size=max_size,
        candidate_pool=candidate_pool,
        beam_width=beam_width,
        branches_per_state=branches_per_state,
        prioritize_width=prioritize_width,
        target_remaining_width=target_remaining_width,
        allow_generic_remaining=allow_generic_remaining,
        prefer_one_shot_slicing=prefer_one_shot_slicing,
    )
    if plan is not None:
        _STRUCTURE_Q3_FREE_CUTSET_PLAN_CACHE[cache_key] = plan
    return plan

def _q3_free_one_shot_cutset_conditioning_plan(
    q: PhaseFunction,
) -> _Q3FreeCutsetConditioningPlan | None:
    """Try a stronger cutset search for giant one-shot q3-free components."""
    _cfg = _get_solver_config()
    adjacency, edges = _q3_free_graph(q)
    max_degree = max((len(neighbors) for neighbors in adjacency), default=0)
    giant_surrogate_path = (
        q.n >= _Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS
        and max_degree <= 4
    )
    depth, chords = _q3_free_spanning_data(adjacency, edges)
    if giant_surrogate_path:
        cheap_order, cheap_width = _best_cheap_q3_free_order(q)
        if cheap_width <= min(_q3_free_treewidth_width_limit(), _cfg.tensor_hint_target_width + 2):
            feedback_size = len(_select_feedback_vertices(q.n, chords, depth))
            if _q3_free_treewidth_candidate_is_viable(q, cheap_order, cheap_width, feedback_size):
                return None
    plan = None if giant_surrogate_path else _q3_free_cutset_conditioning_plan(q, prefer_one_shot_slicing=True)
    if (
        plan is not None
        and plan.remaining_width <= _cfg.tensor_hint_target_width
        and _q3_free_cutset_plan_generic_penalty(plan) == 0
    ):
        return plan
    preferred = set(_select_feedback_vertices(q.n, chords, depth))
    preferred.update(_q3_free_tensor_slice_hint(q))
    direct_plan = (
        _direct_order_guided_q3_free_cutset_plan(
            q,
            adjacency,
            preferred=preferred,
            max_size=_cfg.one_shot_cutset_max_size,
            target_remaining_width=_cfg.tensor_hint_target_width,
            allow_generic_remaining=True,
        )
        if giant_surrogate_path
        else None
    )
    if (
        direct_plan is not None
        and (
            (
                direct_plan.remaining_backend == "treewidth"
                and direct_plan.remaining_width <= _q3_free_treewidth_width_limit()
            )
            or (
                direct_plan.remaining_width <= _cfg.tensor_hint_target_width
                and _q3_free_cutset_plan_generic_penalty(direct_plan) == 0
            )
        )
    ):
        return direct_plan
    peel_order, core_vars = _q3_free_series_reduction_core(adjacency)
    separator_candidates = _separator_ranked_q3_free_cutset_vertices(
        adjacency,
        preferred=preferred,
        max_candidates=_cfg.one_shot_cutset_candidate_pool,
    )
    generic_candidates = _candidate_q3_free_cutset_vertices(
        adjacency,
        preferred=preferred,
        max_candidates=_cfg.one_shot_cutset_candidate_pool,
    )
    merged_candidates = _merge_q3_free_cutset_candidate_orders(
        separator_candidates,
        generic_candidates,
        max_candidates=_cfg.one_shot_cutset_candidate_pool,
    )
    unified_candidate_orders: list[Sequence[int]] = [merged_candidates] if merged_candidates else []
    hint_candidates: list[tuple[int, ...]] = []
    if (
        len(core_vars) >= _Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS
        and q.n - len(core_vars) >= _Q3_FREE_SERIES_CORE_RECURSE_MIN_SHRINK
    ):
        core_q = _component_restriction(q, core_vars)
        core_adjacency, core_edges = _q3_free_graph(core_q)
        core_depth, core_chords = _q3_free_spanning_data(core_adjacency, core_edges)
        core_preferred = set(_select_feedback_vertices(core_q.n, core_chords, core_depth))
        core_preferred.update(_q3_free_tensor_slice_hint(core_q))
        core_contract_candidates = _merge_q3_free_cutset_candidate_orders(
            _separator_ranked_q3_free_cutset_vertices(
                core_adjacency,
                preferred=core_preferred,
                max_candidates=_cfg.one_shot_cutset_candidate_pool,
            ),
            _order_guided_q3_free_cutset_vertices(
                core_adjacency,
                candidate_orders=_iter_q3_free_cheap_order_hints(core_q.n, q=core_q),
                preferred=core_preferred,
                max_candidates=_cfg.one_shot_cutset_candidate_pool,
            ),
            _candidate_q3_free_cutset_vertices(
                core_adjacency,
                preferred=core_preferred,
                max_candidates=_cfg.one_shot_cutset_candidate_pool,
            ),
            max_candidates=_cfg.one_shot_cutset_candidate_pool,
        )
        core_plan = None
        should_run_core_seed_search = (
            not giant_surrogate_path
            or len(core_contract_candidates) < min(4, _cfg.one_shot_cutset_candidate_pool)
        )
        if should_run_core_seed_search:
            core_plan = _build_q3_free_cutset_conditioning_plan_uncached(
                core_q,
                max_size=min(_cfg.one_shot_cutset_max_size, 4 if giant_surrogate_path else _cfg.one_shot_cutset_max_size),
                candidate_pool=min(12 if giant_surrogate_path else _cfg.one_shot_cutset_candidate_pool, core_q.n),
                beam_width=min(2 if giant_surrogate_path else _cfg.one_shot_cutset_beam_width, _cfg.one_shot_cutset_beam_width),
                branches_per_state=min(2 if giant_surrogate_path else _cfg.one_shot_cutset_branches_per_state, _cfg.one_shot_cutset_branches_per_state),
                prioritize_width=True,
                target_remaining_width=_cfg.tensor_hint_target_width,
                allow_generic_remaining=True,
                prefer_one_shot_slicing=True,
            )
        mapped_contract_candidates = tuple(
            int(core_vars[idx]) for idx in core_contract_candidates
        )
        if core_plan is not None and core_plan.cutset_vars:
            mapped_cutset = tuple(int(core_vars[idx]) for idx in core_plan.cutset_vars)
            core_remaining_order_hint = None
            if (
                core_plan.remaining_backend == "treewidth"
                and core_plan.remaining_order
            ):
                core_remaining_vars = tuple(
                    int(core_vars[idx]) for idx in core_plan.remaining_vars
                )
                core_remaining_order = tuple(
                    int(core_remaining_vars[idx]) for idx in core_plan.remaining_order
                )
                peeled_remaining = tuple(
                    int(var) for var in peel_order if var not in mapped_cutset
                )
                core_remaining_order_hint = peeled_remaining + core_remaining_order
            mapped_contract_candidates = _merge_q3_free_cutset_candidate_orders(
                mapped_cutset,
                mapped_contract_candidates,
                max_candidates=_cfg.one_shot_cutset_candidate_pool,
            )
            if mapped_contract_candidates:
                unified_candidate_orders.append(mapped_contract_candidates)
            if core_remaining_order_hint is not None:
                hint_candidates.append(core_remaining_order_hint)

    order_guided_variants: dict[tuple[int, ...], tuple[tuple[int, ...], int]] = {}
    for cheap_order in _iter_q3_free_cheap_order_hints(q.n, q=q):
        order_guided_candidates = _order_guided_q3_free_cutset_vertices(
            adjacency,
            candidate_orders=(cheap_order,),
            preferred=preferred,
            max_candidates=_cfg.one_shot_cutset_candidate_pool,
        )
        if not order_guided_candidates:
            continue
        hint_width = _cubic_order_width(q, cheap_order)
        cached_variant = order_guided_variants.get(order_guided_candidates)
        if cached_variant is None or hint_width < cached_variant[1]:
            order_guided_variants[order_guided_candidates] = (
                tuple(int(var) for var in cheap_order),
                int(hint_width),
            )
    for order_guided_candidates, (cheap_order, _hint_width) in order_guided_variants.items():
        unified_candidate_orders.append(order_guided_candidates)
        hint_candidates.append(cheap_order)

    unified_candidates = _merge_q3_free_cutset_candidate_orders(
        *unified_candidate_orders,
        max_candidates=_cfg.one_shot_cutset_candidate_pool,
    ) if unified_candidate_orders else ()
    best_hint = None
    best_hint_width = None
    seen_hints: set[tuple[int, ...]] = set()
    for hint in hint_candidates:
        hint_key = tuple(int(var) for var in hint)
        if hint_key in seen_hints or len(hint_key) != q.n:
            continue
        seen_hints.add(hint_key)
        try:
            hint_width = _cubic_order_width(q, hint_key)
        except ValueError:
            continue
        if best_hint_width is None or hint_width < best_hint_width:
            best_hint = hint_key
            best_hint_width = int(hint_width)
    unified_plan = (
        _build_q3_free_cutset_conditioning_plan_uncached(
            q,
            max_size=_cfg.one_shot_cutset_max_size,
            candidate_pool=max(len(unified_candidates), 1),
            beam_width=_cfg.one_shot_cutset_beam_width,
            branches_per_state=_cfg.one_shot_cutset_branches_per_state,
            prioritize_width=True,
            target_remaining_width=_cfg.tensor_hint_target_width,
            candidate_override=unified_candidates,
            remaining_order_hint=best_hint,
            allow_generic_remaining=True,
            prefer_one_shot_slicing=True,
        )
        if unified_candidates
        else None
    )

    def plan_score(candidate: _Q3FreeCutsetConditioningPlan | None) -> tuple[int, int, int, int, int, int]:
        if candidate is None:
            return (1, 1 << 30, 1, 1 << 30, 1 << 30, 1 << 30)
        return (
            0,
            _q3_free_cutset_plan_generic_penalty(candidate),
            int(candidate.remaining_width > _cfg.tensor_hint_target_width),
            int(candidate.remaining_width),
            int(candidate.estimated_total_work),
            len(candidate.cutset_vars),
        )

    return min((plan, direct_plan, unified_plan), key=plan_score)

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
