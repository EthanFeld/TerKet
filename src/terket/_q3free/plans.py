"""q3-free plan builders, heuristics, and reusable constraint planning."""

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

from .batch import _compact_index_storage_array, _compact_residue_storage_array
from ..cubic_arithmetic import CubicFunction, PhaseFunction, detect_factorization
from .._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals
from ..scaling import ScaledAmplitude, ScaledComplex
from ..spec import CircuitSpec, Gate
from ..state import BitSequence, CircuitInput, EchelonCache, ExtendedReductionMode, ReducerInfo, ReductionInfo, SolverConfig

_LOCAL_NAMES = {
    '_q3_free_constraint_plan_key',
    '_q3_free_edge_density',
    '_q3_free_prefers_locality_preserving_cutset',
    '_q3_free_prefers_reusable_cutset',
    '_q3_free_prefers_one_shot_cutset',
    '_q3_free_prefers_dense_one_shot_direct',
    '_sum_q3_free_via_one_shot_cutset_scaled',
    '_plan_q3_free_constraint_components',
    '_build_q3_free_constraint_plan',
    '_q3_free_constraint_rhs',
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


def _q3_free_constraint_plan_key(
    state: SchurState,
    cache: EchelonCache,
    *,
    allow_tensor_contraction: bool = True,
    prefer_reusable_decomposition: bool = False,
) -> tuple[Any, ...]:
    return (
        _q_key(state.q),
        tuple(state.eps),
        tuple(state.eps0),
        cache.echelon_rows,
        cache.pivot_col,
        cache.row_ops,
        bool(allow_tensor_contraction),
        bool(prefer_reusable_decomposition),
    )

def _q3_free_edge_density(q: PhaseFunction) -> float:
    if q.n <= 1:
        return 0.0
    return (2.0 * len(q.q2)) / (q.n * (q.n - 1))

def _q3_free_prefers_locality_preserving_cutset(
    q: PhaseFunction,
    *,
    feedback_size: int,
    max_degree: int,
    edge_density: float,
    allow_tensor_contraction: bool,
) -> bool:
    """Return whether dense q3-free routing should stay on TN-guided exact backends."""
    if not allow_tensor_contraction or q.q3 or not q.q2:
        return False

    factor_density = len(_build_factor_scopes(q)) / max(1, q.n)
    if factor_density < _Q3_TENSOR_CONTRACTION_MIN_FACTOR_DENSITY:
        return False

    if q.n >= _get_solver_config().tensor_hint_min_vars:
        return bool(_q3_free_tensor_slice_hint(q))

    if q.n <= _Q3_TENSOR_CONTRACTION_MAX_VARS or q.n > _Q3_HYBRID_CONTRACTION_MAX_VARS:
        return False
    if (
        max_degree < _Q3_FREE_DENSE_PLAN_MIN_DEGREE
        and edge_density < _Q3_FREE_DENSE_PLAN_MIN_DENSITY
    ):
        return False
    return feedback_size > _SCHUR_COMPLEMENT_CROSSOVER_FVS or max_degree >= _Q3_FREE_DENSE_PLAN_MIN_DEGREE

def _q3_free_prefers_reusable_cutset(
    q: PhaseFunction,
    *,
    treewidth_order: Sequence[int] | None,
    cutset_plan: _Q3FreeCutsetConditioningPlan | None,
    lambda_count: int,
) -> bool:
    """Return whether reusable q3-free workloads should prefer a cutset plan."""
    if (
        treewidth_order is None
        or cutset_plan is None
        or q.q3
        or not q.q2
        or lambda_count < _Q3_FREE_REUSABLE_CUTSET_MIN_LAMBDA_VARS
    ):
        return False

    direct_width = _treewidth_order_width(q, treewidth_order)
    if direct_width < _Q3_FREE_REUSABLE_CUTSET_MIN_TREEWIDTH:
        return False

    direct_work = max(1, _estimate_treewidth_dp_work(q, treewidth_order))
    cutset_work = max(1, cutset_plan.estimated_total_work)
    reuse_multiplier = 1 << min(_Q3_FREE_REUSABLE_CUTSET_MAX_LOG2_REUSE, lambda_count)
    width_gain = direct_width - cutset_plan.remaining_width
    if width_gain >= 2 and cutset_work <= direct_work * reuse_multiplier:
        return True
    if width_gain >= 1 and cutset_work * 2 <= direct_work * reuse_multiplier:
        return True
    return False

def _q3_free_prefers_one_shot_cutset(
    q: PhaseFunction,
    *,
    treewidth_order: Sequence[int] | None,
    cutset_plan: _Q3FreeCutsetConditioningPlan | None,
    allow_tensor_contraction: bool,
) -> bool:
    """Return whether one-shot exact amplitudes should switch to cutset slicing."""
    if treewidth_order is None or cutset_plan is None or q.q3 or not q.q2:
        return False

    direct_width = _treewidth_order_width(q, treewidth_order)
    if direct_width < _Q3_FREE_ONE_SHOT_CUTSET_MIN_TREEWIDTH:
        return False

    if cutset_plan.remaining_width > _Q3_FREE_CUTSET_TENSOR_HINT_TARGET_WIDTH:
        return False
    if _q3_free_cutset_plan_generic_penalty(cutset_plan) > 0:
        return False

    width_gain = direct_width - cutset_plan.remaining_width
    if width_gain < 2:
        return False

    direct_work = max(1, _estimate_treewidth_dp_work(q, treewidth_order))
    cutset_work = max(1, cutset_plan.estimated_total_work)
    if cutset_work <= direct_work:
        return True
    if (
        allow_tensor_contraction
        and q.n >= _Q3_FREE_CUTSET_TENSOR_HINT_MIN_VARS
        and _q3_free_tensor_slice_hint(q)
        and cutset_work * 2 <= direct_work
    ):
        return True
    return False

def _q3_free_prefers_dense_one_shot_direct(
    q: PhaseFunction,
    *,
    direct_width: int,
) -> bool:
    """Return whether a giant dense q2 core should bypass the generic mediator path."""
    if q.q3 or not q.q2 or _is_half_phase_q2(q):
        return False
    if q.n < _Q3_FREE_ONE_SHOT_DIRECT_MIN_VARS:
        return False
    if direct_width < _Q3_FREE_ONE_SHOT_DIRECT_MIN_WIDTH:
        return False
    return len(q.q2) >= int(math.ceil(_Q3_FREE_ONE_SHOT_DIRECT_MIN_Q2_PER_VAR * q.n))

def _sum_q3_free_via_one_shot_cutset_scaled(q: PhaseFunction) -> ScaledComplex | None:
    """Directly route giant dense q2 kernels through the one-shot cutset planner."""
    if q.q3 or not q.q2 or _is_half_phase_q2(q):
        return None
    if q.n < _Q3_FREE_ONE_SHOT_DIRECT_MIN_VARS:
        return None
    if len(q.q2) < int(math.ceil(_Q3_FREE_ONE_SHOT_DIRECT_MIN_Q2_PER_VAR * q.n)):
        return None
    _order, direct_width = _min_fill_cubic_order(q)
    if not _q3_free_prefers_dense_one_shot_direct(q, direct_width=direct_width):
        return None

    plan = _q3_free_one_shot_cutset_conditioning_plan(q)
    if plan is None or plan.remaining_backend == "generic":
        return None
    if plan.remaining_width > _Q3_FREE_ONE_SHOT_DIRECT_MAX_REMAINING_WIDTH:
        return None
    if plan.remaining_width >= direct_width:
        return None

    return _evaluate_q3_free_cutset_conditioning_plan_scaled(
        plan,
        q.q1,
        level=q.level,
    )

def _plan_q3_free_constraint_components(
    base_q: PhaseFunction,
    lambda_offset: int,
    *,
    order_hint: Sequence[int] | None = None,
    allow_tensor_contraction: bool = True,
    prefer_reusable_decomposition: bool = False,
    prefer_one_shot_slicing: bool = False,
) -> tuple[tuple[int, ...], tuple[_Q3FreeConstraintComponentPlan, ...]]:
    """Plan reusable component backends for an augmented q3-free constraint kernel."""
    component_sets = detect_factorization(base_q)
    covered = set().union(*component_sets) if component_sets else set()
    isolated_vars = tuple(sorted(set(range(base_q.n)) - covered))
    component_plans: list[_Q3FreeConstraintComponentPlan] = []

    hint_positions = None if order_hint is None else {var: idx for idx, var in enumerate(order_hint)}

    for component in component_sets:
        variables = tuple(sorted(component))
        stored_variables = _compact_index_storage_array(variables, upper_bound=base_q.n)
        component_q = _component_restriction(base_q, variables)
        lambda_count = sum(1 for var in variables if var >= lambda_offset)
        adjacency, edges = _q3_free_graph(component_q)
        max_degree = max((len(neighbors) for neighbors in adjacency), default=0)
        edge_density = _q3_free_edge_density(component_q)
        local_order_hint = None
        if hint_positions is not None:
            hinted_variables = sorted(
                variables,
                key=lambda var: (hint_positions.get(var, len(hint_positions)), var),
            )
            local_remap = {var: idx for idx, var in enumerate(variables)}
            local_order_hint = [local_remap[var] for var in hinted_variables]
        dense_component = (
            max_degree >= _Q3_FREE_DENSE_PLAN_MIN_DEGREE
            and edge_density >= _Q3_FREE_DENSE_PLAN_MIN_DENSITY
        )
        binary_phase_plan = None
        skip_dense_schur = False
        if _is_half_phase_q2(component_q):
            fixed_nonbinary_support = _component_fixed_nonbinary_unary_support_size(
                component_q,
                variables,
                lambda_offset=lambda_offset,
            )
            if fixed_nonbinary_support <= _Q3_FREE_HALF_PHASE_UNARY_EXPANSION_MAX_SUPPORT:
                binary_phase_plan = _build_binary_phase_quadratic_plan(component_q)

        if lambda_count == 0 and not prefer_one_shot_slicing and not prefer_reusable_decomposition:
            if dense_component:
                mediator_plan = _build_half_phase_mediator_plan(component_q)
                generic_mediator_plan = (
                    _build_generic_q2_mediator_plan(component_q)
                    if mediator_plan is None
                    else None
                )
                cluster_plan = _build_q1_cluster_plan(component_q)
                dense_schur_ok = _supports_exact_dense_schur(component_q)
                component_plans.append(
                    _Q3FreeConstraintComponentPlan(
                        variables=stored_variables,
                        level=component_q.level,
                        q2=component_q.q2,
                        backend="generic",
                        dense_q2=_dense_q2_matrix(component_q),
                        binary_phase_plan=binary_phase_plan,
                        mediator_plan=mediator_plan,
                        generic_mediator_plan=generic_mediator_plan,
                        cluster_plan=cluster_plan,
                        skip_dense_schur=(
                            skip_dense_schur
                            or not dense_schur_ok
                        ),
                        direct_schur_ok=(
                            binary_phase_plan is None
                            and mediator_plan is None
                            and generic_mediator_plan is None
                            and cluster_plan is None
                            and dense_schur_ok
                        ),
                        quadratic_tensor_q2=_is_half_phase_q2(component_q),
                        lambda_offset=lambda_offset,
                    )
                )
                continue
            component_plans.append(
                _Q3FreeConstraintComponentPlan(
                    variables=stored_variables,
                    level=component_q.level,
                    q2=component_q.q2,
                    backend="constant",
                    precomputed_total=_sum_q3_free_component_scaled(
                        component_q,
                        allow_tensor_contraction=allow_tensor_contraction,
                    ),
                    mediator_plan=_build_half_phase_mediator_plan(component_q),
                    quadratic_tensor_q2=_is_half_phase_q2(component_q),
                    lambda_offset=lambda_offset,
                )
            )
            continue

        # Dense conditioned kernels are a poor match for the sparse spanning /
        # treewidth planner. Route them straight to the dense generic backend,
        # which can evaluate via schur complement without paying that planning
        # cost up front.
        if (
            max_degree >= _Q3_FREE_DENSE_PLAN_MIN_DEGREE
            and edge_density >= _Q3_FREE_DENSE_PLAN_MIN_DENSITY
        ):
            mediator_plan = _build_half_phase_mediator_plan(component_q)
            generic_mediator_plan = (
                _build_generic_q2_mediator_plan(component_q)
                if mediator_plan is None
                else None
            )
            cluster_plan = _build_q1_cluster_plan(component_q)
            dense_schur_ok = _supports_exact_dense_schur(component_q)
            direct_schur_ok = (
                binary_phase_plan is None
                and mediator_plan is None
                and generic_mediator_plan is None
                and cluster_plan is None
                and dense_schur_ok
            )
            depth, chords = _q3_free_spanning_data(adjacency, edges)
            feedback_vars = _select_feedback_vertices(component_q.n, chords, depth)
            treewidth_order = (
                _q3_free_treewidth_order(
                    component_q,
                    len(feedback_vars),
                    order_hint=local_order_hint,
                    max_degree=max_degree,
                )
                if prefer_reusable_decomposition or prefer_one_shot_slicing
                else None
            )
            if treewidth_order is None and _schur_native is not None:
                treewidth_order = _q3_free_treewidth_order(
                    component_q,
                    len(feedback_vars),
                    order_hint=local_order_hint,
                    max_degree=max_degree,
                )
            if treewidth_order is not None:
                native_component_plan, _native_order, _native_width = (
                    _q3_free_native_treewidth_component_plan(
                        component_q,
                        stored_variables,
                        treewidth_order,
                        lambda_offset=lambda_offset,
                        prefer_reusable_decomposition=prefer_reusable_decomposition,
                    )
                )
                if native_component_plan is not None:
                    component_plans.append(native_component_plan)
                    continue
            prefer_cutset = _q3_free_prefers_locality_preserving_cutset(
                component_q,
                feedback_size=len(feedback_vars),
                max_degree=max_degree,
                edge_density=edge_density,
                allow_tensor_contraction=allow_tensor_contraction,
            )
            cutset_plan = (
                (
                    _q3_free_one_shot_cutset_conditioning_plan(component_q)
                    if prefer_one_shot_slicing
                    else _q3_free_cutset_conditioning_plan(component_q)
                )
                if prefer_cutset or prefer_reusable_decomposition or prefer_one_shot_slicing
                else None
            )
            prefer_reusable_cutset = (
                prefer_reusable_decomposition
                and _q3_free_prefers_reusable_cutset(
                    component_q,
                    treewidth_order=treewidth_order,
                    cutset_plan=cutset_plan,
                    lambda_count=lambda_count,
                )
            )
            prefer_one_shot_cutset = (
                prefer_one_shot_slicing
                and _q3_free_prefers_one_shot_cutset(
                    component_q,
                    treewidth_order=treewidth_order,
                    cutset_plan=cutset_plan,
                    allow_tensor_contraction=allow_tensor_contraction,
                )
            )
            prefer_cutset_backend = (
                cutset_plan is not None
                and (
                    prefer_cutset
                    or prefer_reusable_cutset
                    or prefer_one_shot_cutset
                    or prefer_one_shot_slicing
                )
            )
            if mediator_plan is not None or generic_mediator_plan is not None:
                component_plans.append(
                    _Q3FreeConstraintComponentPlan(
                        variables=stored_variables,
                        level=component_q.level,
                        q2=component_q.q2,
                        backend="generic",
                        dense_q2=_dense_q2_matrix(component_q),
                        binary_phase_plan=binary_phase_plan,
                        mediator_plan=mediator_plan,
                        generic_mediator_plan=generic_mediator_plan,
                        cluster_plan=cluster_plan,
                        cutset_plan=cutset_plan,
                        skip_dense_schur=(
                            skip_dense_schur
                            or not dense_schur_ok
                            or prefer_cutset_backend
                        ),
                        direct_schur_ok=direct_schur_ok and not prefer_cutset_backend,
                        quadratic_tensor_q2=_is_half_phase_q2(component_q),
                        lambda_offset=lambda_offset,
                        prefer_reusable_decomposition=prefer_reusable_decomposition,
                        prefer_cutset_backend=prefer_cutset_backend,
                    )
                )
                continue
            component_plans.append(
                _Q3FreeConstraintComponentPlan(
                    variables=stored_variables,
                    level=component_q.level,
                    q2=component_q.q2,
                    backend="generic",
                    dense_q2=_dense_q2_matrix(component_q),
                    binary_phase_plan=binary_phase_plan,
                    mediator_plan=None,
                    generic_mediator_plan=None,
                    cluster_plan=cluster_plan,
                    cutset_plan=cutset_plan,
                    skip_dense_schur=(
                        skip_dense_schur
                        or not dense_schur_ok
                        or prefer_cutset_backend
                    ),
                    direct_schur_ok=direct_schur_ok and not prefer_cutset_backend,
                    quadratic_tensor_q2=_is_half_phase_q2(component_q),
                    lambda_offset=lambda_offset,
                    prefer_reusable_decomposition=prefer_reusable_decomposition,
                    prefer_cutset_backend=prefer_cutset_backend,
                )
            )
            continue

        depth, chords = _q3_free_spanning_data(adjacency, edges)
        if not chords:
            component_plans.append(
                _Q3FreeConstraintComponentPlan(
                    variables=stored_variables,
                    level=component_q.level,
                    q2=component_q.q2,
                    backend="forest",
                    adjacency=tuple(
                        tuple(sorted(neighbors.items()))
                        for neighbors in adjacency
                    ),
                    mediator_plan=_build_half_phase_mediator_plan(component_q),
                    quadratic_tensor_q2=_is_half_phase_q2(component_q),
                    lambda_offset=lambda_offset,
                    prefer_reusable_decomposition=prefer_reusable_decomposition,
                )
            )
            continue

        feedback_vars = _select_feedback_vertices(component_q.n, chords, depth)
        cluster_plan = (
            _build_q1_cluster_plan(component_q)
            if (
                max_degree <= 4
                and component_q.n >= _Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS
            )
            else None
        )
        treewidth_order = _q3_free_treewidth_order(
            component_q,
            len(feedback_vars),
            order_hint=local_order_hint,
            max_degree=max_degree,
        )
        if treewidth_order is not None:
            native_component_plan, treewidth_order, _direct_width_hint = (
                _q3_free_native_treewidth_component_plan(
                    component_q,
                    stored_variables,
                    treewidth_order,
                    lambda_offset=lambda_offset,
                    prefer_reusable_decomposition=prefer_reusable_decomposition,
                )
            )
            if native_component_plan is not None:
                component_plans.append(native_component_plan)
                continue
            direct_width = _treewidth_order_width(component_q, treewidth_order)
            if (
                cluster_plan is not None
                and cluster_plan.width <= _q3_free_treewidth_width_limit()
                and (
                    cluster_plan.width + 2 < direct_width
                    or (
                        prefer_one_shot_slicing
                        and direct_width >= _Q3_FREE_ONE_SHOT_CUTSET_ACTIVATION_WIDTH
                    )
                )
            ):
                component_plans.append(
                    _Q3FreeConstraintComponentPlan(
                        variables=stored_variables,
                        level=component_q.level,
                        q2=component_q.q2,
                        backend="generic",
                        binary_phase_plan=binary_phase_plan,
                        cluster_plan=cluster_plan,
                        skip_dense_schur=True,
                        direct_schur_ok=False,
                        quadratic_tensor_q2=_is_half_phase_q2(component_q),
                        lambda_offset=lambda_offset,
                        prefer_reusable_decomposition=prefer_reusable_decomposition,
                    )
                )
                continue
            prefer_one_shot_cutset_candidate = (
                prefer_one_shot_slicing
                and direct_width >= _Q3_FREE_ONE_SHOT_CUTSET_ACTIVATION_WIDTH
            )
            cutset_plan = (
                (
                    _q3_free_one_shot_cutset_conditioning_plan(component_q)
                    if prefer_one_shot_cutset_candidate
                    else _q3_free_cutset_conditioning_plan(component_q)
                )
                if prefer_reusable_decomposition or prefer_one_shot_cutset_candidate
                else None
            )
            prefer_reusable_cutset = (
                prefer_reusable_decomposition
                and _q3_free_prefers_reusable_cutset(
                    component_q,
                    treewidth_order=treewidth_order,
                    cutset_plan=cutset_plan,
                    lambda_count=lambda_count,
                )
            )
            prefer_one_shot_cutset = (
                prefer_one_shot_cutset_candidate
                and _q3_free_prefers_one_shot_cutset(
                    component_q,
                    treewidth_order=treewidth_order,
                    cutset_plan=cutset_plan,
                    allow_tensor_contraction=allow_tensor_contraction,
                )
            )
            if (prefer_reusable_cutset or prefer_one_shot_cutset) and cutset_plan is not None:
                component_plans.append(
                    _Q3FreeConstraintComponentPlan(
                        variables=stored_variables,
                        level=component_q.level,
                        q2=component_q.q2,
                        backend="generic",
                        binary_phase_plan=binary_phase_plan,
                        cutset_plan=cutset_plan,
                        skip_dense_schur=True,
                        direct_schur_ok=False,
                        quadratic_tensor_q2=_is_half_phase_q2(component_q),
                        lambda_offset=lambda_offset,
                        prefer_reusable_decomposition=True,
                        prefer_cutset_backend=True,
                    )
                )
                continue
            component_plans.append(
                _Q3FreeConstraintComponentPlan(
                    variables=stored_variables,
                    level=component_q.level,
                    q2=component_q.q2,
                    backend="treewidth",
                    order=_compact_index_storage_array(treewidth_order, upper_bound=len(variables)),
                    native_treewidth_plan=_build_native_q3_free_treewidth_plan(
                        n_vars=component_q.n,
                        level=component_q.level,
                        q2=component_q.q2,
                        order=treewidth_order,
                    ),
                    mediator_plan=_build_half_phase_mediator_plan(component_q),
                    quadratic_tensor_q2=_is_half_phase_q2(component_q),
                    lambda_offset=lambda_offset,
                    prefer_reusable_decomposition=prefer_reusable_decomposition,
                )
            )
            continue

        mediator_plan = _build_half_phase_mediator_plan(component_q)
        generic_mediator_plan = (
            _build_generic_q2_mediator_plan(component_q)
            if mediator_plan is None
            else None
        )
        cluster_plan = _build_q1_cluster_plan(component_q)
        dense_schur_ok = _supports_exact_dense_schur(component_q)
        prefer_cutset = _q3_free_prefers_locality_preserving_cutset(
            component_q,
            feedback_size=len(feedback_vars),
            max_degree=max_degree,
            edge_density=edge_density,
            allow_tensor_contraction=allow_tensor_contraction,
        )
        cutset_plan = _q3_free_cutset_conditioning_plan(component_q)
        prefer_cutset_backend = (
            cutset_plan is not None
            and (
                prefer_cutset
                or prefer_one_shot_slicing
                or (
                    prefer_reusable_decomposition
                    and lambda_count >= _Q3_FREE_REUSABLE_CUTSET_MIN_LAMBDA_VARS
                )
            )
        )
        component_plans.append(
            _Q3FreeConstraintComponentPlan(
                variables=stored_variables,
                level=component_q.level,
                q2=component_q.q2,
                backend="generic",
                dense_q2=_dense_q2_matrix(component_q),
                binary_phase_plan=binary_phase_plan,
                mediator_plan=mediator_plan,
                generic_mediator_plan=generic_mediator_plan,
                cluster_plan=cluster_plan,
                cutset_plan=cutset_plan,
                skip_dense_schur=(
                    skip_dense_schur
                    or not dense_schur_ok
                    or prefer_cutset_backend
                ),
                direct_schur_ok=(
                    not prefer_cutset_backend
                    and len(feedback_vars) > _SCHUR_COMPLEMENT_CROSSOVER_FVS
                    and binary_phase_plan is None
                    and mediator_plan is None
                    and generic_mediator_plan is None
                    and cluster_plan is None
                    and dense_schur_ok
                ),
                quadratic_tensor_q2=_is_half_phase_q2(component_q),
                lambda_offset=lambda_offset,
                prefer_reusable_decomposition=prefer_reusable_decomposition,
                prefer_cutset_backend=prefer_cutset_backend,
            )
        )

    return isolated_vars, tuple(component_plans)

def _build_q3_free_constraint_plan(
    state: SchurState,
    cache: EchelonCache,
    order_hint: Sequence[int] | None = None,
    *,
    allow_tensor_contraction: bool = True,
    prefer_reusable_decomposition: bool = False,
) -> _Q3FreeConstraintPlan:
    """Build a reusable exact constraint-sum plan for a q3-free Schur state.

    Above dyadic level 3, affine output restriction can require quartic or
    higher ANF terms. Instead of substituting output parities into the phase,
    introduce one dual variable per independent output constraint and enforce
    the affine system through a character sum. The augmented phase remains
    q3-free, so it can be evaluated exactly by the existing q3-free reducer
    without any unsafe degree truncation. The same plan is also useful at level
    3 because it avoids repeating the generic exact-elimination pipeline for
    every closely related marginal query.
    """
    assert not state.q.q3, "q3-free constraint plans require a q3-free kernel."

    lambda_offset = state.q.n
    row_indices = [row_idx for row_idx, pivot in enumerate(cache.pivot_col) if pivot >= 0]
    rank = len(row_indices)
    augmented_q2 = dict(state.q.q2)
    bilinear_half_phase = state.q.mod_q2 // 2

    for lambda_idx, row_idx in enumerate(row_indices):
        dual_var = lambda_offset + lambda_idx
        for var in _iter_mask_bits(cache.echelon_rows[row_idx]):
            key = (var, dual_var) if var < dual_var else (dual_var, var)
            value = (augmented_q2.get(key, 0) + bilinear_half_phase) % state.q.mod_q2
            if value:
                augmented_q2[key] = value
            elif key in augmented_q2:
                del augmented_q2[key]

    base_q = _phase_function_from_parts(
        state.q.n + rank,
        level=state.q.level,
        q0=state.q.q0,
        q1=list(state.q.q1) + ([0] * rank),
        q2=augmented_q2,
        q3={},
    )
    isolated_vars, component_plans = _plan_q3_free_constraint_components(
        base_q,
        lambda_offset,
        order_hint=order_hint,
        allow_tensor_contraction=allow_tensor_contraction,
        prefer_reusable_decomposition=prefer_reusable_decomposition,
    )

    return _Q3FreeConstraintPlan(
        cache=cache,
        eps0=tuple(state.eps0),
        level=state.q.level,
        q0=state.q.q0,
        base_q1=_compact_residue_storage_array(
            tuple(state.q.q1) + ((0,) * rank),
            modulus=state.q.mod_q1,
        ),
        base_q2=dict(augmented_q2),
        lambda_offset=lambda_offset,
        rank=rank,
        n_free_after_constraints=cache.n_free,
        rhs_linear_coeff=state.q.mod_q1 // 2,
        isolated_vars=_compact_index_storage_array(isolated_vars, upper_bound=base_q.n),
        components=tuple(component_plans),
    )

def _q3_free_constraint_rhs(plan: _Q3FreeConstraintPlan, output_bits: BitSequence) -> tuple[int, ...] | None:
    if len(output_bits) != plan.cache.n:
        raise ValueError(f"Expected {plan.cache.n} output bits, received {len(output_bits)}.")

    target_mask = 0
    for idx, bit in enumerate(output_bits):
        if (int(bit) ^ plan.eps0[idx]) & 1:
            target_mask |= 1 << idx

    rhs_bits = []
    for row_idx, pivot in enumerate(plan.cache.pivot_col):
        rhs = _parity(target_mask & plan.cache.row_ops[row_idx])
        if pivot < 0 and rhs:
            return None
        if pivot >= 0:
            rhs_bits.append(rhs)
    return tuple(rhs_bits)

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
