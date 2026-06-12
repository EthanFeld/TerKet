"""Owned q3-free cutset cache and one-shot search orchestration.

Owns:
- reusable cutset plan cache wrapper
- one-shot cutset candidate assembly for giant q3-free components
"""

from __future__ import annotations

from .._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals

_LOCAL_NAMES = {
    "_q3_free_cutset_conditioning_plan",
    "_q3_free_one_shot_cutset_conditioning_plan",
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


def _giant_surrogate_path_enabled(q: PhaseFunction, adjacency) -> bool:
    return bool(
        q.n >= _Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS
        and max((len(neighbors) for neighbors in adjacency), default=0) <= 4
    )


def _giant_surrogate_cutset_not_needed(
    q: PhaseFunction,
    *,
    depth,
    chords,
    cfg: SolverConfig,
) -> bool:
    cheap_order, cheap_width = _best_cheap_q3_free_order(q)
    width_limit = min(_q3_free_treewidth_width_limit(), cfg.tensor_hint_target_width + 2)
    if cheap_width > width_limit:
        return False
    feedback_size = len(_select_feedback_vertices(q.n, chords, depth))
    return _q3_free_treewidth_candidate_is_viable(q, cheap_order, cheap_width, feedback_size)


def _preferred_one_shot_vertices(
    q: PhaseFunction,
    *,
    depth,
    chords,
) -> set[int]:
    preferred = set(_select_feedback_vertices(q.n, chords, depth))
    preferred.update(_q3_free_tensor_slice_hint(q))
    return preferred


def _plan_hits_tensor_target(
    plan: _Q3FreeCutsetConditioningPlan | None,
    *,
    cfg: SolverConfig,
) -> bool:
    return bool(
        plan is not None
        and plan.remaining_width <= cfg.tensor_hint_target_width
        and _q3_free_cutset_plan_generic_penalty(plan) == 0
    )


def _direct_plan_is_good_enough(
    plan: _Q3FreeCutsetConditioningPlan | None,
    *,
    cfg: SolverConfig,
) -> bool:
    if plan is None:
        return False
    if plan.remaining_backend == "treewidth":
        return plan.remaining_width <= _q3_free_treewidth_width_limit()
    return _plan_hits_tensor_target(plan, cfg=cfg)


def _base_one_shot_candidate_orders(
    adjacency,
    *,
    preferred: set[int],
    cfg: SolverConfig,
) -> list[Sequence[int]]:
    separator_candidates = _separator_ranked_q3_free_cutset_vertices(
        adjacency,
        preferred=preferred,
        max_candidates=cfg.one_shot_cutset_candidate_pool,
    )
    generic_candidates = _candidate_q3_free_cutset_vertices(
        adjacency,
        preferred=preferred,
        max_candidates=cfg.one_shot_cutset_candidate_pool,
    )
    merged_candidates = _merge_q3_free_cutset_candidate_orders(
        separator_candidates,
        generic_candidates,
        max_candidates=cfg.one_shot_cutset_candidate_pool,
    )
    return [merged_candidates] if merged_candidates else []


def _core_remaining_order_hint(
    core_plan: _Q3FreeCutsetConditioningPlan,
    *,
    core_vars: Sequence[int],
    peel_order: Sequence[int],
    mapped_cutset: Sequence[int],
) -> tuple[int, ...] | None:
    if core_plan.remaining_backend != "treewidth" or not core_plan.remaining_order:
        return None
    core_remaining_vars = tuple(int(core_vars[idx]) for idx in core_plan.remaining_vars)
    core_remaining_order = tuple(int(core_remaining_vars[idx]) for idx in core_plan.remaining_order)
    peeled_remaining = tuple(int(var) for var in peel_order if var not in mapped_cutset)
    return peeled_remaining + core_remaining_order


def _extend_core_seed_candidates(
    q: PhaseFunction,
    *,
    core_vars: Sequence[int],
    peel_order: Sequence[int],
    cfg: SolverConfig,
    giant_surrogate_path: bool,
    unified_candidate_orders: list[Sequence[int]],
    hint_candidates: list[tuple[int, ...]],
) -> None:
    if (
        len(core_vars) < _Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS
        or q.n - len(core_vars) < _Q3_FREE_SERIES_CORE_RECURSE_MIN_SHRINK
    ):
        return

    core_q = _component_restriction(q, core_vars)
    core_adjacency, core_edges = _q3_free_graph(core_q)
    core_depth, core_chords = _q3_free_spanning_data(core_adjacency, core_edges)
    core_preferred = _preferred_one_shot_vertices(core_q, depth=core_depth, chords=core_chords)
    core_contract_candidates = _merge_q3_free_cutset_candidate_orders(
        _separator_ranked_q3_free_cutset_vertices(
            core_adjacency,
            preferred=core_preferred,
            max_candidates=cfg.one_shot_cutset_candidate_pool,
        ),
        _order_guided_q3_free_cutset_vertices(
            core_adjacency,
            candidate_orders=_iter_q3_free_cheap_order_hints(core_q.n, q=core_q),
            preferred=core_preferred,
            max_candidates=cfg.one_shot_cutset_candidate_pool,
        ),
        _candidate_q3_free_cutset_vertices(
            core_adjacency,
            preferred=core_preferred,
            max_candidates=cfg.one_shot_cutset_candidate_pool,
        ),
        max_candidates=cfg.one_shot_cutset_candidate_pool,
    )
    should_run_core_seed_search = (
        not giant_surrogate_path
        or len(core_contract_candidates) < min(4, cfg.one_shot_cutset_candidate_pool)
    )
    core_plan = None
    if should_run_core_seed_search:
        core_plan = _build_q3_free_cutset_conditioning_plan_uncached(
            core_q,
            max_size=min(cfg.one_shot_cutset_max_size, 4 if giant_surrogate_path else cfg.one_shot_cutset_max_size),
            candidate_pool=min(12 if giant_surrogate_path else cfg.one_shot_cutset_candidate_pool, core_q.n),
            beam_width=min(2 if giant_surrogate_path else cfg.one_shot_cutset_beam_width, cfg.one_shot_cutset_beam_width),
            branches_per_state=min(2 if giant_surrogate_path else cfg.one_shot_cutset_branches_per_state, cfg.one_shot_cutset_branches_per_state),
            prioritize_width=True,
            target_remaining_width=cfg.tensor_hint_target_width,
            allow_generic_remaining=True,
            prefer_one_shot_slicing=True,
        )
    if core_plan is None or not core_plan.cutset_vars:
        return

    mapped_cutset = tuple(int(core_vars[idx]) for idx in core_plan.cutset_vars)
    mapped_contract_candidates = _merge_q3_free_cutset_candidate_orders(
        mapped_cutset,
        tuple(int(core_vars[idx]) for idx in core_contract_candidates),
        max_candidates=cfg.one_shot_cutset_candidate_pool,
    )
    if mapped_contract_candidates:
        unified_candidate_orders.append(mapped_contract_candidates)
    hint = _core_remaining_order_hint(
        core_plan,
        core_vars=core_vars,
        peel_order=peel_order,
        mapped_cutset=mapped_cutset,
    )
    if hint is not None:
        hint_candidates.append(hint)


def _collect_order_guided_candidate_orders(
    q: PhaseFunction,
    adjacency,
    *,
    preferred: set[int],
    cfg: SolverConfig,
) -> tuple[list[Sequence[int]], list[tuple[int, ...]]]:
    candidate_orders = []
    hint_candidates = []
    order_guided_variants = {}
    for cheap_order in _iter_q3_free_cheap_order_hints(q.n, q=q):
        order_guided_candidates = _order_guided_q3_free_cutset_vertices(
            adjacency,
            candidate_orders=(cheap_order,),
            preferred=preferred,
            max_candidates=cfg.one_shot_cutset_candidate_pool,
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
        candidate_orders.append(order_guided_candidates)
        hint_candidates.append(cheap_order)
    return candidate_orders, hint_candidates


def _best_one_shot_hint(
    q: PhaseFunction,
    hint_candidates: Sequence[Sequence[int]],
) -> tuple[int, ...] | None:
    best_hint = None
    best_hint_width = None
    seen_hints = set()
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
    return best_hint


def _one_shot_plan_score(
    candidate: _Q3FreeCutsetConditioningPlan | None,
    *,
    cfg: SolverConfig,
) -> tuple[int, int, int, int, int, int]:
    if candidate is None:
        return (1, 1 << 30, 1, 1 << 30, 1 << 30, 1 << 30)
    return (
        0,
        _q3_free_cutset_plan_generic_penalty(candidate),
        int(candidate.remaining_width > cfg.tensor_hint_target_width),
        int(candidate.remaining_width),
        int(candidate.estimated_total_work),
        len(candidate.cutset_vars),
    )


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
    cfg = _get_solver_config()
    max_size = cfg.cutset_max_size if max_size is None else int(max_size)
    candidate_pool = cfg.cutset_candidate_pool if candidate_pool is None else int(candidate_pool)
    beam_width = cfg.cutset_beam_width if beam_width is None else int(beam_width)
    branches_per_state = cfg.cutset_branches_per_state if branches_per_state is None else int(branches_per_state)
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
    cfg = _get_solver_config()
    adjacency, edges = _q3_free_graph(q)
    depth, chords = _q3_free_spanning_data(adjacency, edges)
    giant_surrogate_path = _giant_surrogate_path_enabled(q, adjacency)
    if giant_surrogate_path and _giant_surrogate_cutset_not_needed(q, depth=depth, chords=chords, cfg=cfg):
        return None

    plan = None if giant_surrogate_path else _q3_free_cutset_conditioning_plan(q, prefer_one_shot_slicing=True)
    if _plan_hits_tensor_target(plan, cfg=cfg):
        return plan

    preferred = _preferred_one_shot_vertices(q, depth=depth, chords=chords)
    direct_plan = None
    if giant_surrogate_path:
        direct_plan = _direct_order_guided_q3_free_cutset_plan(
            q,
            adjacency,
            preferred=preferred,
            max_size=cfg.one_shot_cutset_max_size,
            target_remaining_width=cfg.tensor_hint_target_width,
            allow_generic_remaining=True,
        )
    if _direct_plan_is_good_enough(direct_plan, cfg=cfg):
        return direct_plan

    peel_order, core_vars = _q3_free_series_reduction_core(adjacency)
    unified_candidate_orders = _base_one_shot_candidate_orders(
        adjacency,
        preferred=preferred,
        cfg=cfg,
    )
    hint_candidates = []
    _extend_core_seed_candidates(
        q,
        core_vars=core_vars,
        peel_order=peel_order,
        cfg=cfg,
        giant_surrogate_path=giant_surrogate_path,
        unified_candidate_orders=unified_candidate_orders,
        hint_candidates=hint_candidates,
    )
    order_guided_candidate_orders, order_guided_hints = _collect_order_guided_candidate_orders(
        q,
        adjacency,
        preferred=preferred,
        cfg=cfg,
    )
    unified_candidate_orders.extend(order_guided_candidate_orders)
    hint_candidates.extend(order_guided_hints)

    unified_candidates = ()
    if unified_candidate_orders:
        unified_candidates = _merge_q3_free_cutset_candidate_orders(
            *unified_candidate_orders,
            max_candidates=cfg.one_shot_cutset_candidate_pool,
        )
    best_hint = _best_one_shot_hint(q, hint_candidates)
    unified_plan = None
    if unified_candidates:
        unified_plan = _build_q3_free_cutset_conditioning_plan_uncached(
            q,
            max_size=cfg.one_shot_cutset_max_size,
            candidate_pool=max(len(unified_candidates), 1),
            beam_width=cfg.one_shot_cutset_beam_width,
            branches_per_state=cfg.one_shot_cutset_branches_per_state,
            prioritize_width=True,
            target_remaining_width=cfg.tensor_hint_target_width,
            candidate_override=unified_candidates,
            remaining_order_hint=best_hint,
            allow_generic_remaining=True,
            prefer_one_shot_slicing=True,
        )
    return min(
        (plan, direct_plan, unified_plan),
        key=lambda candidate: _one_shot_plan_score(candidate, cfg=cfg),
    )


_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
