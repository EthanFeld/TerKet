"""Raw-output q3-free constraint planning and restriction helpers."""

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
    '_build_q3_free_raw_constraint_plan',
    '_restrict_q3_free_component_plan',
    '_restrict_q3_free_raw_constraint_plan',
    '_evaluate_q3_free_raw_constraint_plan_scaled',
    '_evaluate_q3_free_raw_constraint_plan_scaled_batch',
    '_component_restriction',
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


def _build_q3_free_raw_constraint_plan(
    state: SchurState,
    *,
    order_hint: Sequence[int] | None = None,
    allow_tensor_contraction: bool = True,
    prefer_reusable_decomposition: bool = False,
    prefer_one_shot_slicing: bool = False,
) -> _Q3FreeRawConstraintPlan:
    """Build a reusable exact q3-free constraint plan on the raw output rows."""
    assert not state.q.q3, "Raw q3-free constraint plans require a q3-free kernel."

    lambda_offset = state.q.n
    augmented_q2 = dict(state.q.q2)
    bilinear_half_phase = state.q.mod_q2 // 2

    for lambda_idx, row_mask in enumerate(state.eps):
        dual_var = lambda_offset + lambda_idx
        for var in _iter_mask_bits(row_mask):
            key = (var, dual_var) if var < dual_var else (dual_var, var)
            value = (augmented_q2.get(key, 0) + bilinear_half_phase) % state.q.mod_q2
            if value:
                augmented_q2[key] = value
            elif key in augmented_q2:
                del augmented_q2[key]

    base_q = _phase_function_from_parts(
        state.q.n + state.n,
        level=state.q.level,
        q0=state.q.q0,
        q1=list(state.q.q1) + ([0] * state.n),
        q2=augmented_q2,
        q3={},
    )
    isolated_vars, component_plans = _plan_q3_free_constraint_components(
        base_q,
        lambda_offset,
        order_hint=order_hint,
        allow_tensor_contraction=allow_tensor_contraction,
        prefer_reusable_decomposition=prefer_reusable_decomposition,
        prefer_one_shot_slicing=prefer_one_shot_slicing,
    )
    return _Q3FreeRawConstraintPlan(
        eps0=tuple(state.eps0),
        level=state.q.level,
        q0=state.q.q0,
        base_q1=_compact_residue_storage_array(
            tuple(state.q.q1) + ((0,) * state.n),
            modulus=state.q.mod_q1,
        ),
        base_q2=dict(augmented_q2),
        lambda_offset=lambda_offset,
        constraint_count=state.n,
        rhs_linear_coeff=state.q.mod_q1 // 2,
        isolated_vars=_compact_index_storage_array(isolated_vars, upper_bound=base_q.n),
        components=component_plans,
    )

def _restrict_q3_free_component_plan(
    component_plan: _Q3FreeConstraintComponentPlan,
    keep_positions: Sequence[int],
) -> _Q3FreeConstraintComponentPlan | None:
    """Restrict a reusable component plan to a subset of its local variables."""
    keep_positions = tuple(int(pos) for pos in keep_positions)
    if not keep_positions:
        return None
    if len(keep_positions) == len(component_plan.variables):
        return component_plan

    keep_set = set(keep_positions)
    remap = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_positions)}
    variables = tuple(component_plan.variables[idx] for idx in keep_positions)
    q2 = {
        (remap[i], remap[j]): coeff
        for (i, j), coeff in component_plan.q2.items()
        if i in keep_set and j in keep_set
    }

    adjacency = ()
    if component_plan.adjacency:
        adjacency = tuple(
            tuple(
                (remap[neighbor], coeff)
                for neighbor, coeff in component_plan.adjacency[idx]
                if neighbor in keep_set
            )
            for idx in keep_positions
        )

    order = ()
    if len(component_plan.order):
        order = tuple(remap[idx] for idx in component_plan.order if idx in keep_set)

    dense_q2 = None
    if component_plan.dense_q2 is not None:
        dense_q2 = component_plan.dense_q2[np.ix_(keep_positions, keep_positions)].copy()

    binary_phase_plan = None
    restricted_q = _phase_function_from_parts(
        len(variables),
        level=component_plan.level,
        q0=Fraction(0),
        q1=[0] * len(variables),
        q2=q2,
        q3={},
    )
    if component_plan.binary_phase_plan is not None:
        binary_phase_plan = _build_binary_phase_quadratic_plan(restricted_q)
        mediator_plan = _build_half_phase_mediator_plan(restricted_q)
        generic_mediator_plan = _build_generic_q2_mediator_plan(restricted_q) if mediator_plan is None else None
        cluster_plan = _build_q1_cluster_plan(restricted_q)
    else:
        mediator_plan = None
        generic_mediator_plan = (
            _build_generic_q2_mediator_plan(restricted_q)
            if component_plan.generic_mediator_plan is not None
            else None
        )
        cluster_plan = _build_q1_cluster_plan(restricted_q) if component_plan.cluster_plan is not None else None
    cutset_plan = None
    prefer_cutset_backend = False
    native_treewidth_plan = None

    backend = component_plan.backend
    direct_schur_ok = component_plan.direct_schur_ok
    dense_schur_ok = _supports_exact_dense_schur(restricted_q)
    if backend == "generic" and len(variables) > 1 and q2:
        adjacency_maps, edges = _q3_free_graph(restricted_q)
        depth, chords = _q3_free_spanning_data(adjacency_maps, edges)
        if not chords:
            backend = "forest"
            adjacency = tuple(
                tuple(sorted(neighbors.items()))
                for neighbors in adjacency_maps
            )
            order = ()
            dense_q2 = None
            direct_schur_ok = False
        else:
            feedback_vars = _select_feedback_vertices(len(variables), chords, depth)
            max_degree = max((len(neighbors) for neighbors in adjacency_maps), default=0)
            treewidth_order = _q3_free_treewidth_order(
                restricted_q,
                len(feedback_vars),
                max_degree=max_degree,
            )
            lambda_count = (
                sum(1 for var in variables if var >= component_plan.lambda_offset)
                if component_plan.lambda_offset >= 0
                else 0
            )
            if treewidth_order is not None:
                native_treewidth_plan = _build_native_q3_free_treewidth_plan(
                    n_vars=len(variables),
                    level=component_plan.level,
                    q2=q2,
                    order=treewidth_order,
                )
                if native_treewidth_plan is not None:
                    backend = "treewidth"
                    order = tuple(treewidth_order)
                    direct_schur_ok = False
                    dense_q2 = None
                    cutset_plan = None
                    prefer_cutset_backend = False
                else:
                    reusable_cutset_plan = (
                        _q3_free_cutset_conditioning_plan(restricted_q)
                        if component_plan.prefer_reusable_decomposition
                        else None
                    )
                    prefer_reusable_cutset = (
                        component_plan.prefer_reusable_decomposition
                        and _q3_free_prefers_reusable_cutset(
                            restricted_q,
                            treewidth_order=treewidth_order,
                            cutset_plan=reusable_cutset_plan,
                            lambda_count=lambda_count,
                        )
                    )
                    if prefer_reusable_cutset and reusable_cutset_plan is not None:
                        backend = "generic"
                        order = ()
                        direct_schur_ok = False
                        dense_q2 = None
                        cutset_plan = reusable_cutset_plan
                        prefer_cutset_backend = True
                    else:
                        backend = "treewidth"
                        order = tuple(treewidth_order)
                        direct_schur_ok = False
                        dense_q2 = None
            else:
                cutset_plan = _q3_free_cutset_conditioning_plan(restricted_q)
                prefer_cutset_backend = (
                    cutset_plan is not None
                    and (
                        component_plan.prefer_cutset_backend
                        or (
                            component_plan.prefer_reusable_decomposition
                            and lambda_count >= _Q3_FREE_REUSABLE_CUTSET_MIN_LAMBDA_VARS
                        )
                    )
                )
                direct_schur_ok = (
                    dense_q2 is not None
                    and len(feedback_vars) > _SCHUR_COMPLEMENT_CROSSOVER_FVS
                    and binary_phase_plan is None
                    and mediator_plan is None
                    and generic_mediator_plan is None
                    and cluster_plan is None
                    and dense_schur_ok
                    and not (
                        component_plan.cutset_plan is not None
                        and (
                            prefer_cutset_backend
                            or _q3_free_prefers_locality_preserving_cutset(
                                restricted_q,
                                feedback_size=len(feedback_vars),
                                max_degree=max_degree,
                                edge_density=_q3_free_edge_density(restricted_q),
                                allow_tensor_contraction=True,
                            )
                        )
                    )
                )
    else:
        direct_schur_ok = False

    return _Q3FreeConstraintComponentPlan(
        variables=_compact_index_storage_array(variables),
        level=component_plan.level,
        q2=q2,
        backend=backend,
        adjacency=adjacency,
        order=_compact_index_storage_array(order, upper_bound=len(variables)),
        dense_q2=dense_q2,
        binary_phase_plan=binary_phase_plan,
        mediator_plan=mediator_plan,
        generic_mediator_plan=generic_mediator_plan,
        cluster_plan=cluster_plan,
        cutset_plan=cutset_plan,
        native_treewidth_plan=(
            native_treewidth_plan
            if backend == "treewidth" and native_treewidth_plan is not None
            else _build_native_q3_free_treewidth_plan(
                n_vars=len(variables),
                level=component_plan.level,
                q2=q2,
                order=order,
            )
            if backend == "treewidth"
            else None
        ),
        skip_dense_schur=(
            component_plan.skip_dense_schur
            or (backend == "generic" and not dense_schur_ok)
        ),
        direct_schur_ok=direct_schur_ok,
        quadratic_tensor_q2=_is_half_phase_q2(restricted_q),
        lambda_offset=component_plan.lambda_offset,
        prefer_reusable_decomposition=component_plan.prefer_reusable_decomposition,
        prefer_cutset_backend=prefer_cutset_backend,
    )

def _restrict_q3_free_raw_constraint_plan(
    plan: _Q3FreeRawConstraintPlan,
    active_count: int,
) -> _Q3FreeRawConstraintRestrictedPlan:
    """Restrict a raw-output q3-free plan to the first ``active_count`` outputs."""
    if not 0 <= active_count <= plan.constraint_count:
        raise ValueError(
            f"Expected active_count in [0, {plan.constraint_count}], received {active_count}."
        )

    lambda_limit = plan.lambda_offset + active_count
    isolated_vars = tuple(
        var for var in plan.isolated_vars
        if var < plan.lambda_offset or var < lambda_limit
    )
    components = []
    for component_plan in plan.components:
        keep_positions = [
            idx
            for idx, var in enumerate(component_plan.variables)
            if var < plan.lambda_offset or var < lambda_limit
        ]
        restricted = _restrict_q3_free_component_plan(component_plan, keep_positions)
        if restricted is not None:
            components.append(restricted)

    return _Q3FreeRawConstraintRestrictedPlan(
        active_count=active_count,
        isolated_vars=_compact_index_storage_array(isolated_vars, upper_bound=lambda_limit),
        components=tuple(components),
    )

def _evaluate_q3_free_raw_constraint_plan_scaled(
    plan: _Q3FreeRawConstraintPlan,
    restricted_plan: _Q3FreeRawConstraintRestrictedPlan,
    output_bits: BitSequence,
    *,
    allow_tensor_contraction: bool = True,
) -> ScaledComplex:
    """Evaluate a raw-output q3-free plan for one active output prefix."""
    if len(output_bits) != restricted_plan.active_count:
        raise ValueError(
            f"Expected {restricted_plan.active_count} output bits, received {len(output_bits)}."
        )

    q1 = list(plan.base_q1)
    for idx, bit in enumerate(output_bits):
        if (int(bit) ^ plan.eps0[idx]) & 1:
            q1[plan.lambda_offset + idx] = plan.rhs_linear_coeff

    instantiated_q = _phase_function_from_parts(
        len(q1),
        level=plan.level,
        q0=plan.q0,
        q1=q1,
        q2=plan.base_q2,
        q3={},
    )
    baseline_runtime_score = _q3_free_planned_components_runtime_score(
        restricted_plan.isolated_vars,
        restricted_plan.components,
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
        optimized_q, changed = _optimize_q3_free_phase(
            rewritten_q,
            allow_tensor_contraction=allow_tensor_contraction,
            prefer_one_shot_slicing=True,
            baseline_runtime_score=runtime_score,
        )
        if changed:
            execution_plan = _build_q3_free_execution_plan(
                q=optimized_q,
                allow_tensor_contraction=allow_tensor_contraction,
                prefer_one_shot_slicing=True,
            )
        else:
            execution_plan = (
                rewritten_plan
                if rewritten_plan is not None
                else _build_q3_free_execution_plan(
                    q=rewritten_q,
                    allow_tensor_contraction=allow_tensor_contraction,
                    prefer_one_shot_slicing=True,
                )
            )
        return _evaluate_q3_free_execution_plan_scaled(
            execution_plan,
            output_scale_half_pow2=(rewrite_scale_half_pow2 - 2 * restricted_plan.active_count),
        )

    optimized_q, changed = _optimize_q3_free_phase(
        instantiated_q,
        allow_tensor_contraction=allow_tensor_contraction,
        prefer_one_shot_slicing=True,
        baseline_runtime_score=baseline_runtime_score,
    )
    if changed:
        execution_plan = _build_q3_free_execution_plan(
            q=optimized_q,
            allow_tensor_contraction=allow_tensor_contraction,
            prefer_one_shot_slicing=True,
        )
        return _evaluate_q3_free_execution_plan_scaled(
            execution_plan,
            output_scale_half_pow2=-2 * restricted_plan.active_count,
        )

    return _evaluate_q3_free_planned_components_scaled(
        q0=plan.q0,
        q1=q1,
        isolated_vars=restricted_plan.isolated_vars,
        components=restricted_plan.components,
        level=plan.level,
        output_scale_half_pow2=-2 * restricted_plan.active_count,
    )

def _evaluate_q3_free_raw_constraint_plan_scaled_batch(
    plan: _Q3FreeRawConstraintPlan,
    restricted_plan: _Q3FreeRawConstraintRestrictedPlan,
    output_bits_batch: Sequence[BitSequence],
) -> list[ScaledComplex]:
    """Evaluate a raw-output q3-free plan for many active output prefixes."""
    if not output_bits_batch:
        return []
    if any(len(output_bits) != restricted_plan.active_count for output_bits in output_bits_batch):
        raise ValueError(
            f"Expected every output to have length {restricted_plan.active_count}."
        )

    q1_batch = np.broadcast_to(
        np.asarray(plan.base_q1, dtype=np.int64),
        (len(output_bits_batch), len(plan.base_q1)),
    ).copy()
    if restricted_plan.active_count:
        output_matrix = np.asarray(output_bits_batch, dtype=np.bool_)
        rhs_mask = np.logical_xor(
            output_matrix,
            np.asarray(plan.eps0[: restricted_plan.active_count], dtype=np.bool_),
        )
        q1_batch[:, plan.lambda_offset : plan.lambda_offset + restricted_plan.active_count] = (
            rhs_mask.astype(np.int64) * int(plan.rhs_linear_coeff)
        )

    totals = [
        _scale_scaled_complex(
            _make_scaled_complex(cmath.exp(2j * cmath.pi * float(plan.q0))),
            -2 * restricted_plan.active_count,
        )
        for _ in output_bits_batch
    ]

    if len(restricted_plan.isolated_vars):
        isolated = np.asarray(restricted_plan.isolated_vars, dtype=np.int64)
        isolated_q1 = q1_batch[:, isolated]
        for idx, coeffs in enumerate(isolated_q1):
            totals[idx] = _mul_scaled_complex(
                totals[idx],
                _product_q1_sum_scaled(coeffs.tolist(), level=plan.level),
            )

    for component_plan in restricted_plan.components:
        q1_local_batch = q1_batch[:, component_plan.variables]
        component_totals = _evaluate_q3_free_component_plan_scaled_batch(
            component_plan,
            q1_local_batch,
            level=plan.level,
        )
        for idx, component_total in enumerate(component_totals):
            totals[idx] = _mul_scaled_complex(totals[idx], component_total)

    return totals

def _component_restriction(q, component):
    comp = sorted(component)
    remap = {old: new for new, old in enumerate(comp)}
    q1 = [q.q1[idx] for idx in comp]
    q2 = {
        (remap[i], remap[j]): value
        for (i, j), value in q.q2.items()
        if i in remap and j in remap
    }
    q3 = {
        (remap[i], remap[j], remap[k]): value
        for (i, j, k), value in q.q3.items()
        if i in remap and j in remap and k in remap
    }
    return PhaseFunction(len(comp), level=q.level, q0=Fraction(0), q1=q1, q2=q2, q3=q3)

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
