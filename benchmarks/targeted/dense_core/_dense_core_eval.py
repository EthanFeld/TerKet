"""Exact and surrogate evaluation helpers for dense-core QAOA experiments."""

from __future__ import annotations

from benchmarks.targeted.dense_core._dense_core_types import *  # noqa: F403


def evaluate_cutset(case: DenseCoreCase, cutset_vars: tuple[int, ...]) -> CandidateRow | None:
    evaluation = _evaluate_q3_free_cutset_candidate(
        case.q,
        cutset_vars,
        allow_generic_remaining=True,
        prefer_one_shot_slicing=True,
        target_remaining_width=TARGET_REMAINING_WIDTH,
    )
    if evaluation is None:
        return None
    plan = evaluation.plan
    viable = bool(evaluation.viable and plan is not None)
    remaining_backend = plan.remaining_backend if plan is not None else "none"
    remaining_width = int(plan.remaining_width) if plan is not None else case.free_var_count
    total_work = int(plan.estimated_total_work) if plan is not None else 10**18
    generic_penalty = 1 if plan is not None and plan.remaining_backend == "generic" else 0
    builtin_overlap = len(set(cutset_vars) & set(case.builtin_cutset))
    return CandidateRow(
        size=case.size,
        category="",
        heuristic="",
        description="",
        budget=len(cutset_vars),
        cutset_vars=cutset_vars,
        viable=viable,
        remaining_backend=remaining_backend,
        remaining_width=remaining_width,
        estimated_total_work=total_work,
        estimated_total_work_log2=math.log2(max(1, total_work)),
        generic_penalty=generic_penalty,
        improvement_vs_min_fill=int(case.min_fill_width - remaining_width),
        builtin_cutset_overlap=int(builtin_overlap),
    )


def scan_heuristics(
    case: DenseCoreCase,
    specs: list[HeuristicSpec],
    budgets: Iterable[int],
) -> list[CandidateRow]:
    rows: list[CandidateRow] = []
    max_budget = max(int(budget) for budget in budgets)
    for spec in specs:
        ranking = spec.ranker(case, max_budget)
        seen = set()
        clean_ranking = []
        for var in ranking:
            if 0 <= int(var) < case.free_var_count and int(var) not in seen:
                clean_ranking.append(int(var))
                seen.add(int(var))
        clean_ranking.extend(var for var in range(case.free_var_count) if var not in seen)
        for budget in budgets:
            cutset_vars = tuple(clean_ranking[: int(budget)])
            evaluation = evaluate_cutset(case, cutset_vars)
            if evaluation is None:
                continue
            rows.append(
                CandidateRow(
                    size=evaluation.size,
                    category=spec.category,
                    heuristic=spec.name,
                    description=spec.description,
                    budget=evaluation.budget,
                    cutset_vars=evaluation.cutset_vars,
                    viable=evaluation.viable,
                    remaining_backend=evaluation.remaining_backend,
                    remaining_width=evaluation.remaining_width,
                    estimated_total_work=evaluation.estimated_total_work,
                    estimated_total_work_log2=evaluation.estimated_total_work_log2,
                    generic_penalty=evaluation.generic_penalty,
                    improvement_vs_min_fill=evaluation.improvement_vs_min_fill,
                    builtin_cutset_overlap=evaluation.builtin_cutset_overlap,
                )
            )
    return rows


def builtin_cutset_row(case: DenseCoreCase) -> CandidateRow | None:
    if not case.builtin_cutset:
        return None
    evaluation = evaluate_cutset(case, tuple(case.builtin_cutset))
    if evaluation is None:
        return None
    return CandidateRow(
        size=evaluation.size,
        category="baseline",
        heuristic="baseline_builtin_cutset",
        description="Current engine cutset-plan chooser.",
        budget=evaluation.budget,
        cutset_vars=evaluation.cutset_vars,
        viable=evaluation.viable,
        remaining_backend=evaluation.remaining_backend,
        remaining_width=evaluation.remaining_width,
        estimated_total_work=evaluation.estimated_total_work,
        estimated_total_work_log2=evaluation.estimated_total_work_log2,
        generic_penalty=evaluation.generic_penalty,
        improvement_vs_min_fill=evaluation.improvement_vs_min_fill,
        builtin_cutset_overlap=evaluation.builtin_cutset_overlap,
    )


def exact_cutset_total(case: DenseCoreCase, cutset_vars: tuple[int, ...]):
    evaluation = _evaluate_q3_free_cutset_candidate(
        case.q,
        cutset_vars,
        allow_generic_remaining=True,
        prefer_one_shot_slicing=True,
        target_remaining_width=TARGET_REMAINING_WIDTH,
    )
    if evaluation is None or evaluation.plan is None:
        raise RuntimeError("No exact cutset plan for requested cutset.")
    return _evaluate_q3_free_cutset_conditioning_plan_scaled(
        evaluation.plan,
        case.q.q1,
        level=case.q.level,
    )


def exact_full_total(case: DenseCoreCase):
    return _gauss_sum_q3_free_scaled(case.q)[0]


def row_to_dict(row: CandidateRow) -> dict[str, object]:
    return {
        "size": row.size,
        "category": row.category,
        "heuristic": row.heuristic,
        "description": row.description,
        "budget": row.budget,
        "cutset_vars": " ".join(str(var) for var in row.cutset_vars),
        "viable": int(row.viable),
        "remaining_backend": row.remaining_backend,
        "remaining_width": row.remaining_width,
        "estimated_total_work": row.estimated_total_work,
        "estimated_total_work_log2": row.estimated_total_work_log2,
        "generic_penalty": row.generic_penalty,
        "improvement_vs_min_fill": row.improvement_vs_min_fill,
        "builtin_cutset_overlap": row.builtin_cutset_overlap,
    }
