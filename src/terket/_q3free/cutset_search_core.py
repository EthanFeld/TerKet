"""Owned q3-free reusable cutset search and refinement orchestration."""

from __future__ import annotations

from fractions import Fraction

from .._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals

_LOCAL_NAMES = {
    "_build_q3_free_cutset_conditioning_plan_uncached",
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


def _normalize_cutset_search_limits(
    max_size: int | None,
    candidate_pool: int | None,
    beam_width: int | None,
    branches_per_state: int | None,
) -> tuple[int, int, int, int]:
    return (
        int(_Q3_FREE_CUTSET_MAX_SIZE if max_size is None else max_size),
        int(_Q3_FREE_CUTSET_CANDIDATE_POOL if candidate_pool is None else candidate_pool),
        int(_Q3_FREE_CUTSET_BEAM_WIDTH if beam_width is None else beam_width),
        int(_Q3_FREE_CUTSET_BRANCHES_PER_STATE if branches_per_state is None else branches_per_state),
    )


def _select_cutset_candidates(
    q: PhaseFunction,
    adjacency,
    depth,
    chords,
    *,
    candidate_pool: int,
    candidate_override: Sequence[int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    tensor_hint = tuple(int(var) for var in _q3_free_tensor_slice_hint(q))
    preferred = set(_select_feedback_vertices(q.n, chords, depth))
    preferred.update(tensor_hint)
    if candidate_override is not None:
        candidates = _merge_q3_free_cutset_candidate_orders(
            candidate_override,
            max_candidates=int(candidate_pool),
        )
        return candidates, tensor_hint

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
    return candidates, tensor_hint


def _candidate_width(
    evaluation: _Q3FreeCutsetCandidateEvaluation | None,
) -> int | None:
    if evaluation is None or not evaluation.viable or evaluation.plan is None:
        return None
    return int(evaluation.plan.remaining_width)


def _global_remaining_order_hint(
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


def _greedy_result_is_good_enough(
    evaluation: _Q3FreeCutsetCandidateEvaluation | None,
    *,
    giant_low_degree_surrogate: bool,
) -> bool:
    if evaluation is None or not evaluation.viable or evaluation.plan is None:
        return False
    if evaluation.plan.remaining_backend != "treewidth":
        return False
    if _q3_free_cutset_plan_generic_penalty(evaluation.plan) != 0:
        return False
    return bool(giant_low_degree_surrogate)


def _refine_treewidth_cutset_plan(
    plan: _Q3FreeCutsetConditioningPlan,
) -> _Q3FreeCutsetConditioningPlan:
    if plan.remaining_backend != "treewidth" or not plan.remaining_order:
        return plan

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
    return _Q3FreeCutsetConditioningPlan(
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
        native_treewidth_plan=_build_native_q3_free_treewidth_plan(
            n_vars=len(plan.remaining_vars),
            level=plan.level,
            q2=plan.remaining_q2,
            order=refined_order,
        ),
        remaining_isolated_vars=plan.remaining_isolated_vars,
        remaining_components=plan.remaining_components,
        remaining_width=int(refined_width),
        estimated_total_work=max(
            1,
            (1 << len(plan.cutset_vars)) * _estimate_treewidth_dp_work(refined_q, refined_order),
        ),
    )


def _ensure_native_treewidth_cutset_plan(
    plan: _Q3FreeCutsetConditioningPlan,
) -> _Q3FreeCutsetConditioningPlan:
    if plan.remaining_backend != "treewidth" or plan.native_treewidth_plan is not None:
        return plan

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
    )


class _Q3FreeCutsetSearch:
    def __init__(
        self,
        q: PhaseFunction,
        *,
        candidates: Sequence[int],
        remaining_order_hint: Sequence[int] | None,
        prioritize_width: bool,
        target_remaining_width: int | None,
        allow_generic_remaining: bool,
        prefer_one_shot_slicing: bool,
        max_size: int,
        beam_width: int,
        branches_per_state: int,
        use_native_candidate_shortlist: bool,
    ) -> None:
        self.q = q
        self.candidates = tuple(int(candidate) for candidate in candidates)
        self.remaining_order_hint = remaining_order_hint
        self.prioritize_width = bool(prioritize_width)
        self.target_remaining_width = target_remaining_width
        self.allow_generic_remaining = bool(allow_generic_remaining)
        self.prefer_one_shot_slicing = bool(prefer_one_shot_slicing)
        self.max_size = int(max_size)
        self.beam_width = int(beam_width)
        self.branches_per_state = int(branches_per_state)
        self.use_native_candidate_shortlist = bool(use_native_candidate_shortlist)
        self.remaining_universe = tuple(range(q.n))
        self.best_evaluation = None
        self.best_width_seen = None
        self.evaluation_cache = {}
        self.projection_cache = {}

    def cached_evaluation(
        self,
        cutset_vars: tuple[int, ...],
        *,
        local_remaining_order_hint: Sequence[int] | None = None,
        parent_evaluation: _Q3FreeCutsetCandidateEvaluation | None = None,
    ) -> _Q3FreeCutsetCandidateEvaluation | None:
        cached = self.evaluation_cache.get(cutset_vars)
        if local_remaining_order_hint is None and (cached is not None or cutset_vars in self.evaluation_cache):
            return cached

        parent_projection = None
        if (
            parent_evaluation is not None
            and parent_evaluation.plan is not None
            and set(parent_evaluation.cutset_vars) < set(cutset_vars)
        ):
            parent_projection = self.projection_cache.get(parent_evaluation.cutset_vars)
        residual_projection = _build_q3_free_residual_projection(
            self.q,
            cutset_vars,
            remaining_universe=self.remaining_universe,
            parent_projection=parent_projection,
        )
        self.projection_cache[cutset_vars] = residual_projection
        evaluation = _evaluate_q3_free_cutset_candidate(
            self.q,
            cutset_vars,
            remaining_universe=self.remaining_universe,
            residual_projection=residual_projection,
            remaining_order_hint=local_remaining_order_hint or self.remaining_order_hint,
            prioritize_width=self.prioritize_width,
            target_remaining_width=self.target_remaining_width,
            allow_generic_remaining=self.allow_generic_remaining,
            prefer_one_shot_slicing=self.prefer_one_shot_slicing,
        )
        if cached is None or (
            evaluation is not None
            and (not cached.viable or evaluation.score < cached.score)
        ):
            self.evaluation_cache[cutset_vars] = evaluation
            return evaluation
        if cutset_vars not in self.evaluation_cache:
            self.evaluation_cache[cutset_vars] = evaluation
        return self.evaluation_cache[cutset_vars]

    def meets_width_target(
        self,
        evaluation: _Q3FreeCutsetCandidateEvaluation | None,
    ) -> bool:
        return bool(
            self.prioritize_width
            and self.target_remaining_width is not None
            and evaluation is not None
            and evaluation.viable
            and evaluation.plan is not None
            and evaluation.plan.remaining_width <= int(self.target_remaining_width)
        )

    def maybe_update_best(
        self,
        evaluation: _Q3FreeCutsetCandidateEvaluation | None,
    ) -> bool:
        if evaluation is None or not evaluation.viable:
            return False
        if self.best_evaluation is None or evaluation.score < self.best_evaluation.score:
            self.best_evaluation = evaluation
            return True
        return False

    def _ranked_candidate_iterable(
        self,
        selected: Sequence[int],
        greedy_remaining_order_hint: Sequence[int] | None,
    ) -> Sequence[int]:
        if not self.use_native_candidate_shortlist:
            return self.candidates

        remaining_candidates = [candidate for candidate in self.candidates if candidate not in selected]
        ranked_extensions = _native_rank_q3_free_cutset_extensions(
            self.q,
            selected_vars=selected,
            candidate_vars=remaining_candidates,
            remaining_order_hint=greedy_remaining_order_hint,
        )
        if ranked_extensions is None:
            return self.candidates
        shortlist_size = max(int(self.branches_per_state) * 2, int(self.beam_width), 8)
        return tuple(
            candidate for candidate, _width, _work in ranked_extensions[:shortlist_size]
        )

    def _best_local_neighbor(
        self,
        current: _Q3FreeCutsetCandidateEvaluation,
    ) -> _Q3FreeCutsetCandidateEvaluation:
        neighbor_best = current
        ranked_extensions = _native_rank_q3_free_cutset_extensions(
            self.q,
            selected_vars=current.cutset_vars,
            candidate_vars=[candidate for candidate in self.candidates if candidate not in current.cutset_vars],
            remaining_order_hint=_global_remaining_order_hint(current),
        )
        if not ranked_extensions:
            return neighbor_best

        top_candidates = [
            candidate for candidate, _width, _work in ranked_extensions[: _Q3_FREE_ONE_SHOT_LOCAL_SEARCH_TOPK]
        ]
        for candidate in top_candidates:
            neighbor_best = self._best_added_or_swapped_neighbor(current, candidate, neighbor_best)
        if len(current.cutset_vars) <= 1:
            return neighbor_best
        for remove_var in current.cutset_vars:
            dropped = tuple(var for var in current.cutset_vars if var != remove_var)
            evaluation = self.cached_evaluation(
                dropped,
                local_remaining_order_hint=_global_remaining_order_hint(current),
            )
            if evaluation is not None and evaluation.viable and evaluation.score < neighbor_best.score:
                neighbor_best = evaluation
        return neighbor_best

    def _best_added_or_swapped_neighbor(
        self,
        current: _Q3FreeCutsetCandidateEvaluation,
        candidate: int,
        neighbor_best: _Q3FreeCutsetCandidateEvaluation,
    ) -> _Q3FreeCutsetCandidateEvaluation:
        if len(current.cutset_vars) < self.max_size:
            evaluation = self.cached_evaluation(
                tuple(sorted(current.cutset_vars + (candidate,))),
                local_remaining_order_hint=_global_remaining_order_hint(current),
                parent_evaluation=current,
            )
            if evaluation is not None and evaluation.viable and evaluation.score < neighbor_best.score:
                neighbor_best = evaluation
        for remove_var in current.cutset_vars:
            swapped = tuple(
                sorted(var for var in current.cutset_vars if var != remove_var) + [candidate]
            )
            evaluation = self.cached_evaluation(
                swapped,
                local_remaining_order_hint=_global_remaining_order_hint(current),
                parent_evaluation=current,
            )
            if evaluation is not None and evaluation.viable and evaluation.score < neighbor_best.score:
                neighbor_best = evaluation
        return neighbor_best

    def local_search_improve(
        self,
        seed_evaluation: _Q3FreeCutsetCandidateEvaluation | None,
    ) -> _Q3FreeCutsetCandidateEvaluation | None:
        if (
            not self.use_native_candidate_shortlist
            or seed_evaluation is None
            or not seed_evaluation.viable
            or len(seed_evaluation.cutset_vars) >= self.max_size
        ):
            return seed_evaluation

        current = seed_evaluation
        for _pass in range(_Q3_FREE_ONE_SHOT_LOCAL_SEARCH_PASSES):
            neighbor_best = self._best_local_neighbor(current)
            if neighbor_best is current:
                break
            current = neighbor_best
            self.maybe_update_best(current)
            if self.meets_width_target(current):
                break
        return current

    def seed_tensor_hints(self, tensor_hint: Sequence[int]) -> None:
        hinted_vars = [var for var in tensor_hint if var in self.candidates]
        for size in range(1, min(self.max_size, len(hinted_vars)) + 1):
            self.maybe_update_best(self.cached_evaluation(tuple(sorted(hinted_vars[:size]))))
        self.best_width_seen = _candidate_width(self.best_evaluation)

    def _best_greedy_extension(
        self,
        selected: Sequence[int],
        selected_evaluation: _Q3FreeCutsetCandidateEvaluation | None,
        greedy_remaining_order_hint: Sequence[int] | None,
    ) -> tuple[int, _Q3FreeCutsetCandidateEvaluation] | None:
        best_choice = None
        for candidate in self._ranked_candidate_iterable(selected, greedy_remaining_order_hint):
            if candidate in selected:
                continue
            evaluation = self.cached_evaluation(
                tuple(sorted(tuple(selected) + (candidate,))),
                local_remaining_order_hint=greedy_remaining_order_hint,
                parent_evaluation=selected_evaluation,
            )
            if evaluation is None:
                continue
            if best_choice is None or evaluation.score < best_choice[1].score:
                best_choice = (candidate, evaluation)
        return best_choice

    def greedy_search(self, *, giant_low_degree_surrogate: bool) -> None:
        selected: list[int] = []
        selected_evaluation = None
        greedy_remaining_order_hint = self.remaining_order_hint
        for _size in range(1, self.max_size + 1):
            best_choice = self._best_greedy_extension(
                selected,
                selected_evaluation,
                greedy_remaining_order_hint,
            )
            if best_choice is None:
                return
            selected.append(best_choice[0])
            selected_evaluation = best_choice[1]
            greedy_remaining_order_hint = _global_remaining_order_hint(best_choice[1])
            best_width_before = _candidate_width(self.best_evaluation)
            if not self.maybe_update_best(best_choice[1]):
                if giant_low_degree_surrogate and self.meets_width_target(self.best_evaluation):
                    return
                continue
            if _candidate_width(self.best_evaluation) not in (None, best_width_before):
                selected_evaluation = self.local_search_improve(selected_evaluation)
            if giant_low_degree_surrogate and self.meets_width_target(self.best_evaluation):
                return
            if not self.prioritize_width or self.meets_width_target(selected_evaluation):
                return

    def _beam_expansions(
        self,
        frontier: list[tuple[tuple[int, ...], _Q3FreeCutsetCandidateEvaluation | None]],
    ) -> list[tuple[tuple[int, ...], tuple[int, ...], _Q3FreeCutsetCandidateEvaluation]]:
        expansions = []
        seen_cutsets = set()
        for selected_indexes, parent_evaluation in frontier:
            next_idx = selected_indexes[-1] + 1 if selected_indexes else 0
            candidate_indexes = list(
                range(next_idx, min(len(self.candidates), next_idx + int(self.branches_per_state)))
            )
            if self.use_native_candidate_shortlist:
                candidate_indexes = self._shortlisted_candidate_indexes(
                    selected_indexes,
                    next_idx,
                    parent_evaluation,
                    candidate_indexes,
                )
            for candidate_idx in candidate_indexes:
                expanded_indexes = selected_indexes + (candidate_idx,)
                cutset_vars = tuple(sorted(self.candidates[idx] for idx in expanded_indexes))
                if cutset_vars in seen_cutsets:
                    continue
                seen_cutsets.add(cutset_vars)
                evaluation = self.cached_evaluation(
                    cutset_vars,
                    local_remaining_order_hint=_global_remaining_order_hint(parent_evaluation),
                    parent_evaluation=parent_evaluation,
                )
                if evaluation is not None:
                    expansions.append((evaluation.score, expanded_indexes, evaluation))
        return expansions

    def _shortlisted_candidate_indexes(
        self,
        selected_indexes: tuple[int, ...],
        next_idx: int,
        parent_evaluation: _Q3FreeCutsetCandidateEvaluation | None,
        fallback: list[int],
    ) -> list[int]:
        prefix_candidates = [self.candidates[idx] for idx in selected_indexes]
        available_candidates = [self.candidates[idx] for idx in range(next_idx, len(self.candidates))]
        ranked_extensions = _native_rank_q3_free_cutset_extensions(
            self.q,
            selected_vars=prefix_candidates,
            candidate_vars=available_candidates,
            remaining_order_hint=_global_remaining_order_hint(parent_evaluation),
        )
        if ranked_extensions is None:
            return fallback
        shortlisted = {
            int(candidate)
            for candidate, _width, _work in ranked_extensions[: max(int(self.branches_per_state), 4)]
        }
        return [
            idx for idx in range(next_idx, len(self.candidates))
            if self.candidates[idx] in shortlisted
        ]

    def beam_search(self, *, giant_low_degree_surrogate: bool) -> None:
        frontier = [((), None)]
        stagnation_steps = 0
        best_width_seen = _candidate_width(self.best_evaluation)
        for _size in range(1, self.max_size + 1):
            expansions = self._beam_expansions(frontier)
            if not expansions:
                return
            expansions.sort(key=lambda item: item[0])
            improved_this_round, width_improved = self._beam_round_improvements(expansions, best_width_seen)
            best_width_seen = self.best_width_seen
            if improved_this_round and (not giant_low_degree_surrogate or width_improved):
                self.best_evaluation = self.local_search_improve(self.best_evaluation)
                improved_this_round = True
            frontier = self._next_frontier(expansions, giant_low_degree_surrogate, width_improved)
            if improved_this_round:
                stagnation_steps = 0
                if self.meets_width_target(self.best_evaluation):
                    return
                continue
            stagnation_steps += 1
            if giant_low_degree_surrogate and self.best_evaluation is not None and not width_improved:
                return
            if (
                self.use_native_candidate_shortlist
                and self.best_evaluation is not None
                and stagnation_steps >= _Q3_FREE_ONE_SHOT_STAGNATION_LIMIT
            ):
                return

    def _beam_round_improvements(
        self,
        expansions: list[tuple[tuple[int, ...], tuple[int, ...], _Q3FreeCutsetCandidateEvaluation]],
        best_width_seen: int | None,
    ) -> tuple[bool, bool]:
        improved_this_round = False
        width_improved = False
        current_best_width = best_width_seen
        for _score, _expanded_indexes, evaluation in expansions:
            if not self.maybe_update_best(evaluation):
                continue
            improved_this_round = True
            candidate_width = _candidate_width(self.best_evaluation)
            if candidate_width is not None and (current_best_width is None or candidate_width < current_best_width):
                current_best_width = candidate_width
                width_improved = True
        self.best_width_seen = current_best_width
        return improved_this_round, width_improved

    def _next_frontier(
        self,
        expansions: list[tuple[tuple[int, ...], tuple[int, ...], _Q3FreeCutsetCandidateEvaluation]],
        giant_low_degree_surrogate: bool,
        width_improved: bool,
    ) -> list[tuple[tuple[int, ...], _Q3FreeCutsetCandidateEvaluation | None]]:
        frontier_limit = int(self.beam_width)
        if giant_low_degree_surrogate and not width_improved:
            frontier_limit = 1
        return [
            (expanded_indexes, evaluation)
            for _score, expanded_indexes, _evaluation in expansions[:frontier_limit]
            for evaluation in (_evaluation,)
        ]

    def maybe_large_instance_shortcut(self) -> _Q3FreeCutsetConditioningPlan | None:
        if (
            self.best_evaluation is None
            or self.q.n < 64
            or len(self.best_evaluation.cutset_vars) > 2
            or (self.prioritize_width and not self.meets_width_target(self.best_evaluation))
        ):
            return None
        plan = self.best_evaluation.plan
        if plan is None:
            return None
        return _attach_q3_free_cutset_runtime_cache(
            _ensure_native_treewidth_cutset_plan(_refine_treewidth_cutset_plan(plan))
        )

    def finalize_plan(self) -> _Q3FreeCutsetConditioningPlan | None:
        if self.best_evaluation is None:
            return None
        plan = self.best_evaluation.plan
        if plan is None:
            return None
        if plan.remaining_backend == "generic":
            plan = _finalize_q3_free_cutset_conditioning_plan(
                plan,
                prefer_one_shot_slicing=self.prefer_one_shot_slicing,
            )
        plan = _ensure_native_treewidth_cutset_plan(_refine_treewidth_cutset_plan(plan))
        return _attach_q3_free_cutset_runtime_cache(plan)


def _build_q3_free_cutset_conditioning_plan_uncached(
    q: PhaseFunction,
    *,
    max_size: int | None = None,
    candidate_pool: int | None = None,
    beam_width: int | None = None,
    branches_per_state: int | None = None,
    prioritize_width: bool = False,
    target_remaining_width: int | None = None,
    candidate_override: Sequence[int] | None = None,
    remaining_order_hint: Sequence[int] | None = None,
    allow_generic_remaining: bool = False,
    prefer_one_shot_slicing: bool = False,
) -> _Q3FreeCutsetConditioningPlan | None:
    if q.n <= 1 or not q.q2:
        return None

    max_size, candidate_pool, beam_width, branches_per_state = _normalize_cutset_search_limits(
        max_size,
        candidate_pool,
        beam_width,
        branches_per_state,
    )
    adjacency, edges = _q3_free_graph(q)
    depth, chords = _q3_free_spanning_data(adjacency, edges)
    candidates, tensor_hint = _select_cutset_candidates(
        q,
        adjacency,
        depth,
        chords,
        candidate_pool=int(candidate_pool),
        candidate_override=candidate_override,
    )
    if not candidates:
        return None

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
    max_size = min(int(max_size), len(candidates))
    if giant_low_degree_surrogate:
        max_size = min(int(max_size), 6)
        candidate_pool = min(int(candidate_pool), 16)
        beam_width = min(int(beam_width), 3)
        branches_per_state = min(int(branches_per_state), 2)

    search = _Q3FreeCutsetSearch(
        q,
        candidates=candidates[:candidate_pool],
        remaining_order_hint=remaining_order_hint,
        prioritize_width=prioritize_width,
        target_remaining_width=target_remaining_width,
        allow_generic_remaining=allow_generic_remaining,
        prefer_one_shot_slicing=prefer_one_shot_slicing,
        max_size=max_size,
        beam_width=beam_width,
        branches_per_state=branches_per_state,
        use_native_candidate_shortlist=use_native_candidate_shortlist,
    )
    search.seed_tensor_hints(tensor_hint)
    search.greedy_search(giant_low_degree_surrogate=giant_low_degree_surrogate)
    if _greedy_result_is_good_enough(
        search.best_evaluation,
        giant_low_degree_surrogate=giant_low_degree_surrogate,
    ):
        plan = search.best_evaluation.plan
        assert plan is not None
        return _attach_q3_free_cutset_runtime_cache(plan)

    large_instance_plan = search.maybe_large_instance_shortcut()
    if large_instance_plan is not None:
        return large_instance_plan
    search.beam_search(giant_low_degree_surrogate=giant_low_degree_surrogate)
    return search.finalize_plan()


_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
