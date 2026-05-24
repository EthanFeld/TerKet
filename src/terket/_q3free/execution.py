"""Extracted q3-free execution planning and scoring."""

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

from ..cubic_arithmetic import CubicFunction, PhaseFunction, detect_factorization
from .._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals
from ..scaling import ScaledAmplitude, ScaledComplex
from ..spec import CircuitSpec, Gate
from ..state import BitSequence, CircuitInput, EchelonCache, ExtendedReductionMode, ReducerInfo, ReductionInfo, SolverConfig

_LOCAL_NAMES = {
    '_q3_free_execution_plan_cache_key',
    '_q3_free_reusable_execution_plan_cache_key',
    '_build_q3_free_reusable_execution_plan',
    '_materialize_q3_free_execution_plan',
    '_build_q3_free_execution_plan',
    '_evaluate_q3_free_planned_components_scaled',
    '_evaluate_q3_free_execution_plan_scaled',
    '_q3_free_execution_plan_runtime_score',
    '_q3_free_planned_components_runtime_score',
    '_q3_free_runtime_score_is_good_baseline',
    '_rewrite_q3_free_phase_to_normal_form',
    '_optimize_q3_free_phase',
    '_phase3_execution_plan_runtime_score',
    '_phase3_runtime_score_is_good_baseline',
    '_evaluate_q3_free_constraint_plan_scaled',
    '_evaluate_q3_free_constraint_plan_scaled_batch',
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


def _q3_free_execution_plan_cache_key(
    q: PhaseFunction,
    *,
    allow_tensor_contraction: bool,
    prefer_reusable_decomposition: bool,
    prefer_one_shot_slicing: bool,
) -> tuple[Any, ...]:
    return (
        _q_key(q),
        bool(allow_tensor_contraction),
        bool(prefer_reusable_decomposition),
        bool(prefer_one_shot_slicing),
        _get_solver_config(),
        bool(_quimb_import_enabled()),
        bool(_kahypar_available()),
    )

_Q3_FREE_REUSABLE_EXECUTION_PLAN_PENDING = object()

def _q3_free_reusable_execution_plan_cache_key(
    q: PhaseFunction,
    *,
    allow_tensor_contraction: bool,
    prefer_one_shot_slicing: bool,
) -> tuple[Any, ...]:
    return (
        _q_structure_key(q),
        _qubit_quadratic_tensor_obstruction_support(q),
        bool(allow_tensor_contraction),
        bool(prefer_one_shot_slicing),
        _get_solver_config(),
        bool(_quimb_import_enabled()),
        bool(_kahypar_available()),
    )

def _build_q3_free_reusable_execution_plan(
    *,
    q: PhaseFunction,
    allow_tensor_contraction: bool,
    prefer_one_shot_slicing: bool = False,
) -> _Q3FreeReusableExecutionPlan:
    isolated_vars, component_plans = _plan_q3_free_constraint_components(
        q,
        q.n,
        allow_tensor_contraction=allow_tensor_contraction,
        prefer_reusable_decomposition=True,
        prefer_one_shot_slicing=prefer_one_shot_slicing,
    )
    return _Q3FreeReusableExecutionPlan(
        level=q.level,
        isolated_vars=_compact_index_storage_array(isolated_vars, upper_bound=q.n),
        components=tuple(component_plans),
    )

def _materialize_q3_free_execution_plan(
    q: PhaseFunction,
    reusable_plan: _Q3FreeReusableExecutionPlan,
) -> _Q3FreeExecutionPlan:
    return _Q3FreeExecutionPlan(
        level=q.level,
        q0=q.q0,
        q1=_compact_residue_storage_array(q.q1, modulus=q.mod_q1),
        isolated_vars=reusable_plan.isolated_vars,
        components=reusable_plan.components,
    )

def _build_q3_free_execution_plan(
    *,
    q: PhaseFunction,
    allow_tensor_contraction: bool,
    prefer_reusable_decomposition: bool = False,
    prefer_one_shot_slicing: bool = False,
    context: _ReductionContext | None = None,
) -> _Q3FreeExecutionPlan:
    """Plan one instantiated q3-free phase once, then reuse it across solvers."""
    assert not q.q3, "q3-free execution plans require a q3-free kernel."
    cache_key = _q3_free_execution_plan_cache_key(
        q,
        allow_tensor_contraction=allow_tensor_contraction,
        prefer_reusable_decomposition=prefer_reusable_decomposition,
        prefer_one_shot_slicing=prefer_one_shot_slicing,
    )
    if context is not None:
        cached = context.q3_free_constraint_plan_cache.get(cache_key)
        if cached is not None:
            return cached
    cached = _STRUCTURE_Q3_FREE_EXECUTION_PLAN_CACHE.get(cache_key)
    if cached is not None:
        if context is not None:
            context.q3_free_constraint_plan_cache[cache_key] = cached
        return cached

    reusable_cache_key = None
    if q.n >= _Q3_FREE_REUSABLE_EXECUTION_PLAN_MIN_VARS and q.q2:
        reusable_cache_key = _q3_free_reusable_execution_plan_cache_key(
            q,
            allow_tensor_contraction=allow_tensor_contraction,
            prefer_one_shot_slicing=prefer_one_shot_slicing,
        )
        reusable_cached = _STRUCTURE_Q3_FREE_REUSABLE_EXECUTION_PLAN_CACHE.get(reusable_cache_key)
        if reusable_cached is _Q3_FREE_REUSABLE_EXECUTION_PLAN_PENDING:
            reusable_cached = _build_q3_free_reusable_execution_plan(
                q=q,
                allow_tensor_contraction=allow_tensor_contraction,
                prefer_one_shot_slicing=prefer_one_shot_slicing,
            )
            _STRUCTURE_Q3_FREE_REUSABLE_EXECUTION_PLAN_CACHE[reusable_cache_key] = reusable_cached
        elif reusable_cached is None and prefer_reusable_decomposition:
            reusable_cached = _build_q3_free_reusable_execution_plan(
                q=q,
                allow_tensor_contraction=allow_tensor_contraction,
                prefer_one_shot_slicing=prefer_one_shot_slicing,
            )
            _STRUCTURE_Q3_FREE_REUSABLE_EXECUTION_PLAN_CACHE[reusable_cache_key] = reusable_cached
        elif reusable_cached is None:
            _STRUCTURE_Q3_FREE_REUSABLE_EXECUTION_PLAN_CACHE[reusable_cache_key] = (
                _Q3_FREE_REUSABLE_EXECUTION_PLAN_PENDING
            )
        if isinstance(reusable_cached, _Q3FreeReusableExecutionPlan):
            plan = _materialize_q3_free_execution_plan(q, reusable_cached)
            _STRUCTURE_Q3_FREE_EXECUTION_PLAN_CACHE[cache_key] = plan
            if context is not None:
                context.q3_free_constraint_plan_cache[cache_key] = plan
            return plan

    isolated_vars, component_plans = _plan_q3_free_constraint_components(
        q,
        q.n,
        allow_tensor_contraction=allow_tensor_contraction,
        prefer_reusable_decomposition=prefer_reusable_decomposition,
        prefer_one_shot_slicing=prefer_one_shot_slicing,
    )
    plan = _Q3FreeExecutionPlan(
        level=q.level,
        q0=q.q0,
        q1=_compact_residue_storage_array(q.q1, modulus=q.mod_q1),
        isolated_vars=_compact_index_storage_array(isolated_vars, upper_bound=q.n),
        components=tuple(component_plans),
    )
    _STRUCTURE_Q3_FREE_EXECUTION_PLAN_CACHE[cache_key] = plan
    if context is not None:
        context.q3_free_constraint_plan_cache[cache_key] = plan
    return plan

def _evaluate_q3_free_planned_components_scaled(
    *,
    q0: Fraction,
    q1: Sequence[int],
    isolated_vars: Sequence[int],
    components: Sequence[_Q3FreeConstraintComponentPlan],
    level: int,
    output_scale_half_pow2: int = 0,
) -> ScaledComplex:
    """Execute already-planned q3-free backends with no further optimization."""
    total = _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q0)))

    if len(isolated_vars):
        total = _mul_scaled_complex(
            total,
            _product_q1_sum_scaled([q1[var] for var in isolated_vars], level=level),
        )

    for component_plan in components:
        q1_local = [q1[var] for var in component_plan.variables]
        component_total = _evaluate_q3_free_component_plan_scaled(
            component_plan,
            q1_local,
            level=level,
        )
        total = _mul_scaled_complex(total, component_total)

    return _scale_scaled_complex(total, output_scale_half_pow2)

def _evaluate_q3_free_execution_plan_scaled(
    plan: _Q3FreeExecutionPlan,
    *,
    output_scale_half_pow2: int = 0,
) -> ScaledComplex:
    """Execute a fully instantiated q3-free execution plan."""
    return _evaluate_q3_free_planned_components_scaled(
        q0=plan.q0,
        q1=plan.q1,
        isolated_vars=plan.isolated_vars,
        components=plan.components,
        level=plan.level,
        output_scale_half_pow2=output_scale_half_pow2,
    )

def _q3_free_execution_plan_runtime_score(
    plan: _Q3FreeExecutionPlan,
) -> tuple[int, int, int, int, int]:
    """Approximate runtime score for a fully planned q3-free execution plan."""
    total_work = 0
    max_width = 0
    generic_penalty = 0
    direct_treewidth_penalty = 0

    for component_plan in plan.components:
        total_work += _q3_free_component_plan_work_hint(component_plan)
        max_width = max(max_width, _q3_free_component_plan_width_hint(component_plan))
        if component_plan.backend == "generic" and not component_plan.prefer_cutset_backend:
            generic_penalty += 1
        if (
            component_plan.backend == "treewidth"
            and component_plan.cutset_plan is None
        ):
            direct_treewidth_penalty += 1

    return (
        int(total_work),
        int(max_width),
        int(generic_penalty),
        int(direct_treewidth_penalty),
        len(plan.components),
    )

def _q3_free_planned_components_runtime_score(
    isolated_vars: Sequence[int],
    components: Sequence[_Q3FreeConstraintComponentPlan],
) -> tuple[int, int, int, int, int]:
    """Approximate runtime score for already-planned q3-free components."""
    total_work = 1 if len(isolated_vars) else 0
    max_width = 0
    generic_penalty = 0
    direct_treewidth_penalty = 0

    for component_plan in components:
        total_work += _q3_free_component_plan_work_hint(component_plan)
        max_width = max(max_width, _q3_free_component_plan_width_hint(component_plan))
        if component_plan.backend == "generic" and not component_plan.prefer_cutset_backend:
            generic_penalty += 1
        if (
            component_plan.backend == "treewidth"
            and component_plan.cutset_plan is None
        ):
            direct_treewidth_penalty += 1

    return (
        int(total_work),
        int(max_width),
        int(generic_penalty),
        int(direct_treewidth_penalty),
        len(components),
    )

def _q3_free_runtime_score_is_good_baseline(
    runtime_score: tuple[int, int, int, int, int],
    *,
    prefer_one_shot_slicing: bool,
) -> bool:
    """Return whether a baseline q3-free plan is already good enough to trust."""
    if not prefer_one_shot_slicing:
        return False
    _total_work, _max_width, generic_penalty, direct_treewidth_penalty, _components = runtime_score
    return generic_penalty == 0 and direct_treewidth_penalty == 0

def _rewrite_q3_free_phase_to_normal_form(
    q: PhaseFunction,
    *,
    allow_tensor_contraction: bool,
    prefer_reusable_decomposition: bool = False,
    prefer_one_shot_slicing: bool = False,
    baseline_runtime_score: tuple[int, int, int, int, int] | None = None,
    context: _ReductionContext | None = None,
) -> tuple[PhaseFunction | None, int, bool, _Q3FreeExecutionPlan | None, tuple[int, int, int, int, int] | None]:
    """Apply exact no-branch q3-free rewrites when they improve planned runtime."""
    assert not q.q3, "q3-free normal-form rewrite expects a q3-free kernel."
    if q.level > 3:
        reduced_q, scale_half_pow2, changed = _apply_safe_q3_free_parity_substitutions(q)
        if reduced_q is None:
            return None, scale_half_pow2, True, None, None
        if not changed:
            return q, 0, False, None, baseline_runtime_score
    else:
        reduced_q, scale_half_pow2, _exact_info, _blocked_quadratics = _apply_exact_eliminations(q, context=context)

        if reduced_q is None:
            return None, scale_half_pow2, True, None, None
        if reduced_q is q and scale_half_pow2 == 0:
            return q, 0, False, None, baseline_runtime_score

    candidate_plan: _Q3FreeExecutionPlan | None
    if reduced_q.n == 0:
        candidate_plan = None
        candidate_runtime_score = (0, 0, 0, 0, 0)
    else:
        candidate_plan = _build_q3_free_execution_plan(
            q=reduced_q,
            allow_tensor_contraction=allow_tensor_contraction,
            prefer_reusable_decomposition=prefer_reusable_decomposition,
            prefer_one_shot_slicing=prefer_one_shot_slicing,
            context=context,
        )
        candidate_runtime_score = _q3_free_execution_plan_runtime_score(candidate_plan)
    if baseline_runtime_score is None:
        if q.n == 0:
            baseline_runtime_score = (0, 0, 0, 0, 0)
        else:
            baseline_plan = _build_q3_free_execution_plan(
                q=q,
                allow_tensor_contraction=allow_tensor_contraction,
                prefer_reusable_decomposition=prefer_reusable_decomposition,
                prefer_one_shot_slicing=prefer_one_shot_slicing,
                context=context,
            )
            baseline_runtime_score = _q3_free_execution_plan_runtime_score(baseline_plan)

    if candidate_runtime_score < baseline_runtime_score:
        return reduced_q, scale_half_pow2, True, candidate_plan, candidate_runtime_score
    return q, 0, False, None, baseline_runtime_score

def _optimize_q3_free_phase(
    q: PhaseFunction,
    *,
    allow_tensor_contraction: bool = True,
    prefer_reusable_decomposition: bool = False,
    prefer_one_shot_slicing: bool = False,
    baseline_runtime_score: tuple[int, int, int, int, int] | None = None,
    context: _ReductionContext | None = None,
) -> tuple[PhaseFunction, bool]:
    """Apply q3-free optimization only when it improves planned runtime."""
    engine = importlib.import_module("terket._engine_impl")
    if _FORCE_ENGINE_BINDINGS_REFRESH:
        _sync_from_engine(engine)
    assert not q.q3, "q3-free optimization expects a q3-free phase function."
    if q.n > _Q3_FREE_OPTIONAL_REWRITE_MAX_VARS:
        return q, False
    if (
        baseline_runtime_score is not None
        and _q3_free_runtime_score_is_good_baseline(
            baseline_runtime_score,
            prefer_one_shot_slicing=prefer_one_shot_slicing,
        )
    ):
        return q, False
    optimized_q, changed = engine._optimize_phase_function_structure(q, context=context)
    if not changed:
        return q, False

    if baseline_runtime_score is None:
        if q.n == 0:
            baseline_runtime_score = (0, 0, 0, 0, 0)
        else:
            baseline_plan = engine._build_q3_free_execution_plan(
                q=q,
                allow_tensor_contraction=allow_tensor_contraction,
                prefer_reusable_decomposition=prefer_reusable_decomposition,
                prefer_one_shot_slicing=prefer_one_shot_slicing,
                context=context,
            )
            baseline_runtime_score = engine._q3_free_execution_plan_runtime_score(baseline_plan)
    if optimized_q.n == 0:
        candidate_runtime_score = (0, 0, 0, 0, 0)
    else:
        candidate_plan = engine._build_q3_free_execution_plan(
            q=optimized_q,
            allow_tensor_contraction=allow_tensor_contraction,
            prefer_reusable_decomposition=prefer_reusable_decomposition,
            prefer_one_shot_slicing=prefer_one_shot_slicing,
            context=context,
        )
        candidate_runtime_score = engine._q3_free_execution_plan_runtime_score(candidate_plan)
    if candidate_runtime_score < baseline_runtime_score:
        return optimized_q, True
    return q, False

def _phase3_execution_plan_runtime_score(
    q: PhaseFunction,
    *,
    allow_tensor_contraction: bool,
) -> tuple[int, int, int, int, int]:
    cover, order, width, structural_obstruction, backend = _phase3_plan(
        q,
        allow_tensor_contraction=allow_tensor_contraction,
    )
    fully_peeled = backend == "treewidth_dp_peeled"
    if not fully_peeled:
        core_vars, peel_order = _q3_hypergraph_2core(q)
        fully_peeled = bool(peel_order) and not core_vars
    selected_backend, runtime_score, _separator = _choose_phase3_backend(
        q,
        cover,
        order,
        width,
        structural_obstruction,
        allow_tensor_contraction=allow_tensor_contraction,
        fully_peeled=fully_peeled,
        extended_reductions="auto",
    )
    if backend is not None and backend != selected_backend:
        return _phase3_backend_runtime_score(
            q,
            cover,
            order,
            width,
            structural_obstruction,
            backend,
            fully_peeled=fully_peeled,
        )
    return runtime_score

def _phase3_runtime_score_is_good_baseline(
    runtime_score: tuple[int, int, int, int, int],
) -> bool:
    backend_rank, work, width, _cover_size, structural_obstruction = runtime_score
    return (
        backend_rank == 0
        and structural_obstruction == 0
        and width <= _Q3_TREEWIDTH_DP_PEELED_MAX_WIDTH
        and work <= _Q3_TREEWIDTH_DP_PEELED_MAX_WORK
    )

def _evaluate_q3_free_constraint_plan_scaled(
    plan: _Q3FreeConstraintPlan,
    output_bits: BitSequence,
    *,
    allow_tensor_contraction: bool = True,
) -> ScaledComplex:
    engine = importlib.import_module("terket._engine_impl")
    if _FORCE_ENGINE_BINDINGS_REFRESH:
        _sync_from_engine(engine)
    rhs_bits = _q3_free_constraint_rhs(plan, output_bits)
    if rhs_bits is None:
        return _ZERO_SCALED

    q1 = list(plan.base_q1)
    for lambda_idx, rhs in enumerate(rhs_bits):
        if rhs:
            q1[plan.lambda_offset + lambda_idx] = plan.rhs_linear_coeff

    instantiated_q = _phase_function_from_parts(
        len(q1),
        level=plan.level,
        q0=plan.q0,
        q1=q1,
        q2=plan.base_q2,
        q3={},
    )
    baseline_runtime_score = engine._q3_free_planned_components_runtime_score(
        plan.isolated_vars,
        plan.components,
    )
    rewritten_q, rewrite_scale_half_pow2, rewrite_changed, rewritten_plan, runtime_score = _rewrite_q3_free_phase_to_normal_form(
        instantiated_q,
        allow_tensor_contraction=allow_tensor_contraction,
        prefer_one_shot_slicing=True,
        baseline_runtime_score=baseline_runtime_score,
    )
    if rewritten_q is None:
        return _ZERO_SCALED
    if rewrite_changed:
        optimized_q, changed = engine._optimize_q3_free_phase(
            rewritten_q,
            allow_tensor_contraction=allow_tensor_contraction,
            prefer_one_shot_slicing=True,
            baseline_runtime_score=runtime_score,
        )
        if changed:
            execution_plan = engine._build_q3_free_execution_plan(
                q=optimized_q,
                allow_tensor_contraction=allow_tensor_contraction,
                prefer_one_shot_slicing=True,
            )
        else:
            execution_plan = (
                rewritten_plan
                if rewritten_plan is not None
                else engine._build_q3_free_execution_plan(
                    q=rewritten_q,
                    allow_tensor_contraction=allow_tensor_contraction,
                    prefer_one_shot_slicing=True,
                )
            )
        return engine._evaluate_q3_free_execution_plan_scaled(
            execution_plan,
            output_scale_half_pow2=(rewrite_scale_half_pow2 - 2 * plan.rank),
        )

    optimized_q, changed = engine._optimize_q3_free_phase(
        instantiated_q,
        allow_tensor_contraction=allow_tensor_contraction,
        prefer_one_shot_slicing=True,
        baseline_runtime_score=baseline_runtime_score,
    )
    if changed:
        execution_plan = engine._build_q3_free_execution_plan(
            q=optimized_q,
            allow_tensor_contraction=allow_tensor_contraction,
            prefer_one_shot_slicing=True,
        )
        return engine._evaluate_q3_free_execution_plan_scaled(
            execution_plan,
            output_scale_half_pow2=-2 * plan.rank,
        )

    return _evaluate_q3_free_planned_components_scaled(
        q0=plan.q0,
        q1=q1,
        isolated_vars=plan.isolated_vars,
        components=plan.components,
        level=plan.level,
        output_scale_half_pow2=-2 * plan.rank,
    )

def _evaluate_q3_free_constraint_plan_scaled_batch(
    plan: _Q3FreeConstraintPlan,
    output_bits_batch: Sequence[BitSequence],
) -> list[ScaledComplex]:
    """Evaluate a q3-free constraint plan for many output assignments."""
    _refresh_engine_bindings()
    if not output_bits_batch:
        return []
    if any(len(output_bits) != plan.cache.n for output_bits in output_bits_batch):
        raise ValueError(f"Expected every output to have length {plan.cache.n}.")

    rhs_rows: list[tuple[int, ...]] = []
    supported_indices: list[int] = []
    results: list[ScaledComplex] = [_ZERO_SCALED] * len(output_bits_batch)

    for idx, output_bits in enumerate(output_bits_batch):
        rhs_bits = _q3_free_constraint_rhs(plan, output_bits)
        if rhs_bits is None:
            continue
        supported_indices.append(idx)
        rhs_rows.append(rhs_bits)

    if not supported_indices:
        return results

    q1_batch = np.broadcast_to(
        np.asarray(plan.base_q1, dtype=np.int64),
        (len(supported_indices), len(plan.base_q1)),
    ).copy()
    if plan.rank:
        rhs_matrix = np.asarray(rhs_rows, dtype=np.bool_)
        q1_batch[:, plan.lambda_offset : plan.lambda_offset + plan.rank] = (
            rhs_matrix.astype(np.int64) * int(plan.rhs_linear_coeff)
        )

    totals = [
        _scale_scaled_complex(
            _make_scaled_complex(cmath.exp(2j * cmath.pi * float(plan.q0))),
            -2 * plan.rank,
        )
        for _ in supported_indices
    ]

    if len(plan.isolated_vars):
        isolated = np.asarray(plan.isolated_vars, dtype=np.int64)
        isolated_q1 = q1_batch[:, isolated]
        for idx, coeffs in enumerate(isolated_q1):
            totals[idx] = _mul_scaled_complex(
                totals[idx],
                _product_q1_sum_scaled(coeffs.tolist(), level=plan.level),
            )

    for component_plan in plan.components:
        q1_local_batch = q1_batch[:, component_plan.variables]
        component_totals = _evaluate_q3_free_component_plan_scaled_batch(
            component_plan,
            q1_local_batch,
            level=plan.level,
        )
        for idx, component_total in enumerate(component_totals):
            totals[idx] = _mul_scaled_complex(totals[idx], component_total)

    for output_idx, total in zip(supported_indices, totals):
        results[output_idx] = total
    return results

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
