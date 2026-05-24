"""Extracted q3-free native treewidth helpers."""

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
    '_build_native_q3_free_treewidth_plan',
    '_build_native_q3_free_treewidth_plan_cached',
    '_q3_free_native_treewidth_component_plan',
    '_sum_q3_free_treewidth_dp_scaled_batch',
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


def _build_native_q3_free_treewidth_plan(
    *,
    n_vars: int,
    level: int,
    q2: dict[tuple[int, int], int],
    order: Sequence[int],
) -> object | None:
    """Build a reusable native q3-free treewidth plan when available."""
    if _schur_native is None or not hasattr(_schur_native, "build_q3_free_treewidth_plan"):
        return None
    q2_key = tuple(
        sorted((int(i), int(j), int(value)) for (i, j), value in q2.items())
    )
    order_key = tuple(int(var) for var in order)
    return _build_native_q3_free_treewidth_plan_cached(
        int(n_vars),
        int(level),
        q2_key,
        order_key,
    )

@lru_cache(maxsize=2048)
def _build_native_q3_free_treewidth_plan_cached(
    n_vars: int,
    level: int,
    q2_key: tuple[tuple[int, int, int], ...],
    order: tuple[int, ...],
) -> object | None:
    try:
        q2 = {(int(i), int(j)): int(value) for i, j, value in q2_key}
        return _schur_native.build_q3_free_treewidth_plan(
            int(n_vars),
            int(level),
            q2,
            order,
        )
    except Exception:
        return None

register_lru_cache("engine.native_q3_free_treewidth_plan", _build_native_q3_free_treewidth_plan_cached)

def _q3_free_native_treewidth_component_plan(
    q: PhaseFunction,
    variables: Sequence[int],
    order: Sequence[int],
    *,
    lambda_offset: int,
    prefer_reusable_decomposition: bool,
) -> tuple[_Q3FreeConstraintComponentPlan | None, tuple[int, ...], int]:
    """Return native q3-free treewidth component plan when native can execute it."""
    finalized_order, width = _finalize_q3_free_treewidth_order(q, order)
    native_treewidth_plan = _build_native_q3_free_treewidth_plan(
        n_vars=q.n,
        level=q.level,
        q2=q.q2,
        order=finalized_order,
    )
    finalized_order = tuple(int(var) for var in finalized_order)
    if native_treewidth_plan is None:
        return None, finalized_order, int(width)
    return (
        _Q3FreeConstraintComponentPlan(
            variables=tuple(int(var) for var in variables),
            level=q.level,
            q2=q.q2,
            backend="treewidth",
            order=_compact_index_storage_array(finalized_order, upper_bound=q.n),
            native_treewidth_plan=native_treewidth_plan,
            quadratic_tensor_q2=_is_half_phase_q2(q),
            lambda_offset=lambda_offset,
            prefer_reusable_decomposition=prefer_reusable_decomposition,
        ),
        finalized_order,
        int(width),
    )

def _sum_q3_free_treewidth_dp_scaled_batch(
    *,
    n_vars: int,
    level: int,
    q1_batch: np.ndarray,
    q2: dict[tuple[int, int], int],
    order: Sequence[int],
    native_plan: object | None = None,
) -> list[ScaledComplex]:
    """Evaluate many q3-free treewidth-DP sums sharing the same q2/order."""
    q1_batch = np.asarray(q1_batch, dtype=np.int64)
    if len(q1_batch) == 0:
        return []
    q1_batch = np.ascontiguousarray(np.remainder(q1_batch, 1 << int(level)), dtype=np.int64)

    if native_plan is None:
        native_plan = _build_native_q3_free_treewidth_plan(
            n_vars=n_vars,
            level=level,
            q2=q2,
            order=order,
        )
    if (
        native_plan is not None
        and _schur_native is not None
        and hasattr(_schur_native, "sum_q3_free_treewidth_preplanned_batch_scaled_array")
    ):
        try:
            native_rows = _schur_native.sum_q3_free_treewidth_preplanned_batch_scaled_array(
                native_plan,
                q1_batch,
            )
            return [
                (complex(value), int(half_pow2_exp))
                for value, half_pow2_exp, _max_scope in native_rows
            ]
        except Exception:
            pass

    if (
        native_plan is not None
        and _schur_native is not None
        and hasattr(_schur_native, "sum_q3_free_treewidth_preplanned_batch_scaled")
    ):
        try:
            native_rows = _schur_native.sum_q3_free_treewidth_preplanned_batch_scaled(
                native_plan,
                q1_batch.tolist(),
            )
            return [
                (complex(value), int(half_pow2_exp))
                for value, half_pow2_exp, _max_scope in native_rows
            ]
        except Exception:
            pass

    if (
        _schur_native is not None
        and hasattr(_schur_native, "sum_q3_free_treewidth_batch_scaled")
    ):
        try:
            native_rows = _schur_native.sum_q3_free_treewidth_batch_scaled(
                int(n_vars),
                int(level),
                q1_batch.tolist(),
                q2,
                tuple(int(var) for var in order),
            )
            return [
                (complex(value), int(half_pow2_exp))
                for value, half_pow2_exp, _max_scope in native_rows
            ]
        except Exception:
            pass

    totals: list[ScaledComplex] = []
    for q1_local in q1_batch:
        component_q = _phase_function_from_parts(
            int(n_vars),
            level=int(level),
            q0=Fraction(0),
            q1=q1_local.tolist(),
            q2=q2,
            q3={},
        )
        total, _ = _sum_via_treewidth_dp_scaled(component_q, list(order))
        totals.append(total)
    return totals

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
