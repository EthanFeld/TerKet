"""Extracted q3-free component evaluators."""

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
    '_evaluate_q3_free_component_plan_scaled',
    '_evaluate_q3_free_component_plan_scaled_batch',
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


def _evaluate_q3_free_component_plan_scaled(
    component_plan: _Q3FreeConstraintComponentPlan,
    q1_local: Sequence[int],
    *,
    level: int,
) -> ScaledComplex:
    """Evaluate one reusable q3-free component plan under a concrete q1 vector."""
    if component_plan.backend == "constant":
        component_total = component_plan.precomputed_total
        assert component_total is not None
        return component_total
    if (
        component_plan.quadratic_tensor_q2
        and _is_qubit_quadratic_tensor_q1_vector(q1_local, level=level)
    ):
        component_q = _phase_function_from_parts(
            len(component_plan.variables),
            level=level,
            q0=Fraction(0),
            q1=list(q1_local),
            q2=component_plan.q2,
            q3={},
        )
        return _sum_bl26_quadratic_tensor_component_scaled(component_q)
    if component_plan.backend == "forest":
        adjacency = [dict(neighbors) for neighbors in component_plan.adjacency]
        return _forest_transfer_sum_scaled(list(q1_local), adjacency, level=level)
    if component_plan.backend == "treewidth":
        component_q = _phase_function_from_parts(
            len(component_plan.variables),
            level=level,
            q0=Fraction(0),
            q1=list(q1_local),
            q2=component_plan.q2,
            q3={},
        )
        component_total, _ = _sum_via_treewidth_dp_scaled(component_q, list(component_plan.order))
        return component_total
    if component_plan.binary_phase_plan is not None:
        if _is_binary_phase_q1_vector(q1_local, level=level):
            return _evaluate_binary_phase_quadratic_plan_scaled_batch(
                component_plan.binary_phase_plan,
                np.asarray([q1_local], dtype=np.int64),
                level=level,
            )[0]
        expanded_total = _sum_half_phase_q2_unary_expansion_with_plan_scaled(
            q1_local,
            level=level,
            plan=component_plan.binary_phase_plan,
        )
        if expanded_total is not None:
            return expanded_total
    if component_plan.cutset_plan is not None and component_plan.prefer_cutset_backend:
        return _evaluate_q3_free_cutset_conditioning_plan_scaled(
            component_plan.cutset_plan,
            q1_local,
            level=level,
        )
    if component_plan.mediator_plan is not None:
        return _evaluate_half_phase_mediator_plan_scaled(
            component_plan.mediator_plan,
            q1_local,
        )
    if component_plan.generic_mediator_plan is not None:
        return _evaluate_generic_q2_mediator_plan_scaled(
            component_plan.generic_mediator_plan,
            q1_local,
        )
    if component_plan.cluster_plan is not None:
        return _evaluate_half_phase_cluster_plan_scaled(
            component_plan.cluster_plan,
            q1_local,
        )
    if component_plan.cutset_plan is not None:
        return _evaluate_q3_free_cutset_conditioning_plan_scaled(
            component_plan.cutset_plan,
            q1_local,
            level=level,
        )
    component_q = None
    if component_plan.dense_q2 is not None or component_plan.direct_schur_ok:
        component_q = _phase_function_from_parts(
            len(component_plan.variables),
            level=level,
            q0=Fraction(0),
            q1=list(q1_local),
            q2=component_plan.q2,
            q3={},
        )
        parity_reduced_total = _sum_half_phase_parity_component_reduction_scaled(component_q)
        if parity_reduced_total is not None:
            return parity_reduced_total
    if component_plan.direct_schur_ok:
        component_total = _schur_complement_q3_free_sum_scaled(
            component_q,
            allow_recursive_fallback=True,
        )
        if component_total is not None:
            return component_total

    component_total = None
    if component_plan.dense_q2 is not None and not component_plan.skip_dense_schur:
        component_total = _schur_complement_q3_free_sum_scaled_dense(
            level,
            list(q1_local),
            component_plan.dense_q2,
            allow_recursive_fallback=False,
        )
    if component_total is not None:
        return component_total

    if component_q is None:
        component_q = _phase_function_from_parts(
            len(component_plan.variables),
            level=level,
            q0=Fraction(0),
            q1=list(q1_local),
            q2=component_plan.q2,
            q3={},
        )
    return _sum_q3_free_component_scaled(component_q)

def _evaluate_q3_free_component_plan_scaled_batch(
    component_plan: _Q3FreeConstraintComponentPlan,
    q1_local_batch: np.ndarray,
    *,
    level: int,
) -> list[ScaledComplex]:
    if component_plan.backend == "constant":
        component_total = component_plan.precomputed_total
        assert component_total is not None
        return [component_total] * len(q1_local_batch)
    if component_plan.quadratic_tensor_q2:
        threshold = max(1, (1 << level) // 4)
        residues = np.remainder(np.asarray(q1_local_batch, dtype=np.int64), 1 << level)
        if threshold <= 1 or np.all((residues % threshold) == 0):
            if component_plan.binary_phase_plan is not None and _is_binary_phase_q1_vector(
                residues.ravel(),
                level=level,
            ):
                return _evaluate_binary_phase_quadratic_plan_scaled_batch(
                    component_plan.binary_phase_plan,
                    residues,
                    level=level,
                )
            return [
                _sum_bl26_quadratic_tensor_component_scaled(
                    _phase_function_from_parts(
                        len(component_plan.variables),
                        level=level,
                        q0=Fraction(0),
                        q1=row.tolist(),
                        q2=component_plan.q2,
                        q3={},
                    )
                )
                for row in residues
            ]
    if component_plan.backend == "forest":
        adjacency = [dict(neighbors) for neighbors in component_plan.adjacency]
        return _forest_transfer_sum_scaled_batch(
            np.asarray(q1_local_batch, dtype=np.int64),
            adjacency,
            level=level,
        )
    if component_plan.backend == "treewidth":
        return _sum_q3_free_treewidth_dp_scaled_batch(
            n_vars=len(component_plan.variables),
            level=level,
            q1_batch=np.asarray(q1_local_batch, dtype=np.int64),
            q2=component_plan.q2,
            order=component_plan.order,
            native_plan=component_plan.native_treewidth_plan,
        )
    if component_plan.binary_phase_plan is not None:
        if _is_binary_phase_q1_vector(q1_local_batch.ravel(), level=level):
            return _evaluate_binary_phase_quadratic_plan_scaled_batch(
                component_plan.binary_phase_plan,
                q1_local_batch,
                level=level,
            )
        expanded_totals = _sum_half_phase_q2_unary_expansion_with_plan_scaled_batch(
            np.asarray(q1_local_batch, dtype=np.int64),
            level=level,
            plan=component_plan.binary_phase_plan,
        )
        if expanded_totals is not None:
            return expanded_totals
    if component_plan.cutset_plan is not None and component_plan.prefer_cutset_backend:
        return _evaluate_q3_free_cutset_conditioning_plan_scaled_batch(
            component_plan.cutset_plan,
            q1_local_batch,
            level=level,
        )
    if component_plan.mediator_plan is not None:
        if len(q1_local_batch) < _Q3_FREE_MEDIATOR_BATCH_MIN_ROWS:
            return [
                _evaluate_half_phase_mediator_plan_scaled(
                    component_plan.mediator_plan,
                    q1_local.tolist(),
                )
                for q1_local in q1_local_batch
            ]
        return _evaluate_half_phase_mediator_plan_scaled_batch(
            component_plan.mediator_plan,
            np.asarray(q1_local_batch, dtype=np.int64),
        )
    if component_plan.generic_mediator_plan is not None:
        if len(q1_local_batch) < _Q3_FREE_MEDIATOR_BATCH_MIN_ROWS:
            return [
                _evaluate_generic_q2_mediator_plan_scaled(
                    component_plan.generic_mediator_plan,
                    q1_local.tolist(),
                )
                for q1_local in q1_local_batch
            ]
        return _evaluate_generic_q2_mediator_plan_scaled_batch(
            component_plan.generic_mediator_plan,
            np.asarray(q1_local_batch, dtype=np.int64),
        )
    if component_plan.cluster_plan is not None:
        return _evaluate_half_phase_cluster_plan_scaled_batch(
            component_plan.cluster_plan,
            np.asarray(q1_local_batch, dtype=np.int64),
        )
    if component_plan.cutset_plan is not None:
        return _evaluate_q3_free_cutset_conditioning_plan_scaled_batch(
            component_plan.cutset_plan,
            q1_local_batch,
            level=level,
        )
    if component_plan.dense_q2 is not None:
        mod_q1 = 1 << level
        half_q1 = mod_q1 // 2
        residues = np.remainder(q1_local_batch, mod_q1)
        if np.all((residues == 0) | (residues == half_q1)):
            candidate_q = _phase_function_from_parts(
                len(component_plan.variables),
                level=level,
                q0=Fraction(0),
                q1=[0] * len(component_plan.variables),
                q2=component_plan.q2,
                q3={},
            )
            binary_phase_plan = _build_binary_phase_quadratic_plan(candidate_q)
            if binary_phase_plan is not None:
                return _evaluate_binary_phase_quadratic_plan_scaled_batch(
                    binary_phase_plan,
                    q1_local_batch,
                    level=level,
                )
    return [
        _evaluate_q3_free_component_plan_scaled(
            component_plan,
            q1_local,
            level=level,
        )
        for q1_local in q1_local_batch
    ]

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
