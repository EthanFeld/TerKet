"""Extracted phase-3 factor-table helpers."""

from __future__ import annotations

import bisect
import cmath
from collections import deque
from dataclasses import dataclass
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

from .._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals
from ..cubic_arithmetic import CubicFunction, PhaseFunction, detect_factorization
from ..scaling import ScaledAmplitude, ScaledComplex
from ..spec import CircuitSpec, Gate
from ..state import BitSequence, CircuitInput, EchelonCache, ExtendedReductionMode, ReducerInfo, ReductionInfo, SolverConfig

_LOCAL_NAMES = {
    '_build_cubic_factors',
    '_build_cubic_factors_scaled',
    '_freeze_complex_factor_tables',
    '_build_cached_cubic_factors',
    '_freeze_scaled_factor_tables',
    '_build_cached_phase3_treewidth_factor_plan_scaled',
    '_build_native_phase3_treewidth_plan',
    '_build_native_level3_phase3_treewidth_plan',
    '_build_native_level3_phase3_treewidth_batch_support_plan',
    '_sum_native_level3_phase3_treewidth_batch_shared_support',
    '_build_native_phase_function_treewidth_batch_support_plan',
    '_sum_native_phase_function_treewidth_batch_shared_support',
    '_maybe_get_native_level3_phase3_treewidth_plan',
    '_sum_native_level3_phase3_treewidth_preplanned',
    '_factor_table_to_tensor_data',
    '_sum_via_treewidth_dp',
    '_sum_via_treewidth_dp_scaled',
    '_sum_via_treewidth_dp_scaled_batch_shared_support',
    '_sum_via_treewidth_dp_peeled_scaled',
    '_sum_via_treewidth_dp_peeled'
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


def _build_cubic_factors(q):
    """Convert q into local complex factors over sorted variable scopes."""
    factors = {}
    scalar = cmath.exp(2j * cmath.pi * float(q.q0))
    omega = _omega_table(q.level)

    for var, coeff in enumerate(q.q1):
        coeff %= q.mod_q1
        if coeff:
            scalar *= _combine_factor(factors, (var,), [1.0 + 0j, omega[coeff]])

    for (i, j), coeff in q.q2.items():
        coeff %= q.mod_q2
        if coeff:
            shift = ((q.mod_q1 // q.mod_q2) * coeff) % q.mod_q1
            scalar *= _combine_factor(
                factors,
                (i, j),
                [
                    1.0 + 0j,
                    1.0 + 0j,
                    1.0 + 0j,
                    omega[shift],
                ],
            )

    for (i, j, k), coeff in q.q3.items():
        coeff %= q.mod_q3
        if coeff:
            table = [1.0 + 0j] * 8
            table[7] = omega[((q.mod_q1 // q.mod_q3) * coeff) % q.mod_q1]
            scalar *= _combine_factor(factors, (i, j, k), table)

    return scalar, factors


def _build_cubic_factors_scaled(q):
    """Scaled-complex companion to ``_build_cubic_factors``."""
    factors = {}
    scalar = _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q.q0)))
    omega_scaled = _omega_scaled_table(q.level)

    for var, coeff in enumerate(q.q1):
        coeff %= q.mod_q1
        if coeff:
            scalar = _mul_scaled_complex(
                scalar,
                _combine_factor_scaled(
                    factors,
                    (var,),
                    [_ONE_SCALED, omega_scaled[coeff]],
                ),
            )

    for (i, j), coeff in q.q2.items():
        coeff %= q.mod_q2
        if coeff:
            shift = ((q.mod_q1 // q.mod_q2) * coeff) % q.mod_q1
            scalar = _mul_scaled_complex(
                scalar,
                _combine_factor_scaled(
                    factors,
                    (i, j),
                    [
                        _ONE_SCALED,
                        _ONE_SCALED,
                        _ONE_SCALED,
                        omega_scaled[shift],
                    ],
                ),
            )

    for (i, j, k), coeff in q.q3.items():
        coeff %= q.mod_q3
        if coeff:
            table = [_ONE_SCALED] * 8
            table[7] = omega_scaled[((q.mod_q1 // q.mod_q3) * coeff) % q.mod_q1]
            scalar = _mul_scaled_complex(
                scalar,
                _combine_factor_scaled(factors, (i, j, k), table),
            )

    return scalar, factors


def _freeze_complex_factor_tables(
    factors: dict[tuple[int, ...], Sequence[complex]],
) -> MappingProxyType:
    return MappingProxyType({
        tuple(scope): tuple(complex(entry) for entry in table)
        for scope, table in factors.items()
    })


def _build_cached_cubic_factors(q) -> tuple[complex, MappingProxyType]:
    cache_key = _q_key(q)
    cached = _STRUCTURE_PHASE3_FACTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    scalar, factors = _build_cubic_factors(q)
    cached = (complex(scalar), _freeze_complex_factor_tables(factors))
    _STRUCTURE_PHASE3_FACTOR_CACHE[cache_key] = cached
    return cached


def _freeze_scaled_factor_tables(
    factors: dict[tuple[int, ...], Sequence[ScaledComplex]],
) -> MappingProxyType:
    return MappingProxyType({
        tuple(scope): tuple(tuple(entry) for entry in table)
        for scope, table in factors.items()
    })


def _build_cached_phase3_treewidth_factor_plan_scaled(
    q,
) -> tuple[ScaledComplex, MappingProxyType]:
    cache_key = _q_key(q)
    cached = _STRUCTURE_PHASE3_TREEWIDTH_FACTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    scalar, factors = _build_cubic_factors_scaled(q)
    cached = (scalar, _freeze_scaled_factor_tables(factors))
    _STRUCTURE_PHASE3_TREEWIDTH_FACTOR_CACHE[cache_key] = cached
    return cached


def _build_native_phase3_treewidth_plan(
    *,
    q,
    order: Sequence[int],
) -> object | None:
    if (
        _schur_native is None
        or not hasattr(_schur_native, "build_scaled_factor_treewidth_plan")
    ):
        return None
    cache_key = (_q_key(q), tuple(int(var) for var in order))
    cached = _STRUCTURE_PHASE3_TREEWIDTH_NATIVE_PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    scalar, factors = _build_cached_phase3_treewidth_factor_plan_scaled(q)
    del scalar
    try:
        native_plan = _schur_native.build_scaled_factor_treewidth_plan(
            int(q.n),
            dict(factors),
            tuple(int(var) for var in order),
        )
    except Exception:
        return None
    _STRUCTURE_PHASE3_TREEWIDTH_NATIVE_PLAN_CACHE[cache_key] = native_plan
    return native_plan


def _build_native_level3_phase3_treewidth_plan(
    *,
    q,
    order: Sequence[int],
) -> object | None:
    if (
        _schur_native is None
        or not hasattr(_schur_native, "build_level3_treewidth_plan")
    ):
        return None
    cache_key = (_q_key(q), tuple(int(var) for var in order))
    cached = _STRUCTURE_PHASE3_LEVEL3_NATIVE_PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        native_plan = _schur_native.build_level3_treewidth_plan(
            int(q.n),
            tuple(int(coeff) for coeff in q.q1),
            q.q2,
            q.q3,
            tuple(int(var) for var in order),
        )
    except Exception:
        return None
    _STRUCTURE_PHASE3_LEVEL3_NATIVE_PLAN_CACHE[cache_key] = native_plan
    return native_plan


def _build_native_level3_phase3_treewidth_batch_support_plan(
    *,
    q,
    order: Sequence[int],
) -> object | None:
    native_build = _native_symbol("build_level3_treewidth_plan")
    native_batch = _native_symbol("sum_level3_treewidth_preplanned_batch_array")
    if native_build is None or native_batch is None or not _native_level3_enabled(q):
        return None

    cache_key = (_q_cubic_treewidth_batch_support_key(q), tuple(int(var) for var in order))
    cached = _STRUCTURE_PHASE3_LEVEL3_BATCH_NATIVE_PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    q2_support, q3_support = _build_cubic_treewidth_batch_support(q)
    try:
        native_plan = native_build(
            int(q.n),
            (0,) * int(q.n),
            {edge: 1 for edge in q2_support},
            {edge: 1 for edge in q3_support},
            tuple(int(var) for var in order),
        )
    except Exception:
        return None
    _STRUCTURE_PHASE3_LEVEL3_BATCH_NATIVE_PLAN_CACHE[cache_key] = native_plan
    return native_plan


def _sum_native_level3_phase3_treewidth_batch_shared_support(
    q_batch: Sequence[PhaseFunction],
    order: Sequence[int],
) -> tuple[list[ScaledComplex], int] | None:
    native_sum = _native_symbol("sum_level3_treewidth_preplanned_batch_array")
    if native_sum is None or not q_batch or not _native_level3_enabled(q_batch[0]):
        return None

    ref_q = q_batch[0]
    native_plan = _build_native_level3_phase3_treewidth_batch_support_plan(q=ref_q, order=order)
    if native_plan is None:
        return None

    q2_support, q3_support = _build_cubic_treewidth_batch_support(ref_q)
    mod_q1 = int(ref_q.mod_q1)
    mod_q2 = int(ref_q.mod_q2)
    mod_q3 = int(ref_q.mod_q3)

    q1_batch = np.ascontiguousarray([
        [int(coeff) % mod_q1 for coeff in q.q1]
        for q in q_batch
    ], dtype=np.int64)
    q2_batch = np.ascontiguousarray([
        [int(q.q2.get(edge, 0)) % mod_q2 for edge in q2_support]
        for q in q_batch
    ], dtype=np.int64)
    q3_batch = np.ascontiguousarray([
        [int(q.q3.get(edge, 0)) % mod_q3 for edge in q3_support]
        for q in q_batch
    ], dtype=np.int64)

    try:
        core_totals, actual_width = native_sum(native_plan, q1_batch, q2_batch, q3_batch)
    except Exception:
        return None

    totals = [
        _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q.q0)) * complex(core_total))
        for q, core_total in zip(q_batch, core_totals)
    ]
    return totals, int(actual_width)


def _build_native_phase_function_treewidth_batch_support_plan(
    *,
    q,
    order: Sequence[int],
) -> object | None:
    native_build = _native_symbol("build_phase_function_treewidth_support_plan")
    native_batch = _native_symbol("sum_phase_function_treewidth_preplanned_batch_scaled_array")
    if native_build is None or native_batch is None:
        return None

    cache_key = (_q_cubic_treewidth_batch_support_key(q), int(q.level), tuple(int(var) for var in order))
    cached = _STRUCTURE_PHASE3_GENERIC_BATCH_NATIVE_PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    q2_support, q3_support = _build_cubic_treewidth_batch_support(q)
    try:
        native_plan = native_build(
            int(q.n),
            int(q.level),
            {edge: 1 for edge in q2_support},
            {edge: 1 for edge in q3_support},
            tuple(int(var) for var in order),
        )
    except Exception:
        return None
    _STRUCTURE_PHASE3_GENERIC_BATCH_NATIVE_PLAN_CACHE[cache_key] = native_plan
    return native_plan


def _sum_native_phase_function_treewidth_batch_shared_support(
    q_batch: Sequence[PhaseFunction],
    order: Sequence[int],
) -> tuple[list[ScaledComplex], int] | None:
    native_sum = _native_symbol("sum_phase_function_treewidth_preplanned_batch_scaled_array")
    if native_sum is None or not q_batch:
        return None

    ref_q = q_batch[0]
    native_plan = _build_native_phase_function_treewidth_batch_support_plan(q=ref_q, order=order)
    if native_plan is None:
        return None

    q2_support, q3_support = _build_cubic_treewidth_batch_support(ref_q)
    mod_q1 = int(ref_q.mod_q1)
    mod_q2 = int(ref_q.mod_q2)
    mod_q3 = int(ref_q.mod_q3)

    q1_batch = np.ascontiguousarray([
        [int(coeff) % mod_q1 for coeff in q.q1]
        for q in q_batch
    ], dtype=np.int64)
    q2_batch = np.ascontiguousarray([
        [int(q.q2.get(edge, 0)) % mod_q2 for edge in q2_support]
        for q in q_batch
    ], dtype=np.int64)
    q3_batch = np.ascontiguousarray([
        [int(q.q3.get(edge, 0)) % mod_q3 for edge in q3_support]
        for q in q_batch
    ], dtype=np.int64)

    try:
        core_rows, actual_width = native_sum(native_plan, q1_batch, q2_batch, q3_batch)
    except Exception:
        return None

    totals = [
        _mul_scaled_complex(
            _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q.q0))),
            (complex(core_value), int(core_half_pow2_exp)),
        )
        for q, (core_value, core_half_pow2_exp) in zip(q_batch, core_rows)
    ]
    return totals, int(actual_width)


def _maybe_get_native_level3_phase3_treewidth_plan(
    *,
    q,
    order: Sequence[int],
) -> object | None:
    cache_key = (_q_key(q), tuple(int(var) for var in order))
    cached = _STRUCTURE_PHASE3_LEVEL3_NATIVE_PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if _STRUCTURE_PHASE3_LEVEL3_NATIVE_PLAN_SEEN_CACHE.get(cache_key) is None:
        _STRUCTURE_PHASE3_LEVEL3_NATIVE_PLAN_SEEN_CACHE[cache_key] = True
        return None
    return _build_native_level3_phase3_treewidth_plan(q=q, order=order)


def _sum_native_level3_phase3_treewidth_preplanned(
    *,
    q,
    order: Sequence[int],
) -> tuple[complex, int] | None:
    native_sum = _native_symbol("sum_level3_treewidth_preplanned")
    if native_sum is None:
        return None
    native_plan = _maybe_get_native_level3_phase3_treewidth_plan(q=q, order=order)
    if native_plan is None:
        return None
    try:
        core_total, max_scope = native_sum(native_plan)
    except Exception:
        return None
    total = cmath.exp(2j * cmath.pi * float(q.q0)) * complex(core_total)
    return total, int(max_scope)


def _factor_table_to_tensor_data(scope, table):
    """Reshape a factor table into a tensor with the same bit ordering."""
    if not scope:
        return np.asarray(table, dtype=np.complex128).reshape(())
    return np.asarray(table, dtype=np.complex128).reshape((2,) * len(scope), order="F")


def _sum_via_treewidth_dp(q, order):
    """Exact cubic sum by factor elimination along a low-width order."""
    if _native_level3_enabled(q):
        planned = _sum_native_level3_phase3_treewidth_preplanned(q=q, order=order)
        if planned is not None:
            return planned
        core_total, max_scope = _schur_native.sum_treewidth_dp_level3(
            q.n,
            q.q1,
            q.q2,
            q.q3,
            order,
        )
        return cmath.exp(2j * cmath.pi * float(q.q0)) * complex(core_total), max_scope

    scalar, factors = _build_cubic_factors(q)
    max_scope = 0

    for var in order:
        bucket_scopes = [scope for scope in factors if var in scope]
        if not bucket_scopes:
            scalar *= 2.0
            max_scope = max(max_scope, 1)
            continue

        bucket = [(scope, factors.pop(scope)) for scope in bucket_scopes]
        union_scope = tuple(sorted({vertex for scope, _ in bucket for vertex in scope}))
        max_scope = max(max_scope, len(union_scope))

        var_pos = union_scope.index(var)
        new_scope = tuple(vertex for vertex in union_scope if vertex != var)
        positions = [
            tuple(union_scope.index(vertex) for vertex in scope)
            for scope, _ in bucket
        ]

        new_table = [0j] * (1 << len(new_scope))
        for reduced_assignment in range(1 << len(new_scope)):
            total = 0j
            for fixed_value in [0, 1]:
                full_assignment = (
                    (reduced_assignment & ((1 << var_pos) - 1))
                    | (fixed_value << var_pos)
                    | ((reduced_assignment >> var_pos) << (var_pos + 1))
                )
                weight = 1.0 + 0j
                for (_, table), pos in zip(bucket, positions):
                    weight *= table[_project_assignment_bits(full_assignment, pos)]
                total += weight
            new_table[reduced_assignment] = total

        scalar *= _combine_factor(factors, new_scope, new_table)

    assert not factors, "All variables should be eliminated by the DP order."
    return scalar, max_scope


def _sum_via_treewidth_dp_scaled(q, order):
    """Scaled-complex companion to ``_sum_via_treewidth_dp``."""
    if _native_level3_enabled(q):
        planned = _sum_native_level3_phase3_treewidth_preplanned(q=q, order=order)
        if planned is not None:
            return _make_scaled_complex(planned[0]), int(planned[1])
        core_total, max_scope = _schur_native.sum_treewidth_dp_level3(
            q.n,
            q.q1,
            q.q2,
            q.q3,
            order,
        )
        return _make_scaled_complex(
            cmath.exp(2j * cmath.pi * float(q.q0)) * complex(core_total),
        ), max_scope

    scalar, factors = _build_cubic_factors_scaled(q)
    return _sum_factor_tables_scaled(q.n, factors, order, scalar=scalar)


def _sum_via_treewidth_dp_scaled_batch_shared_support(
    q_batch: Sequence[PhaseFunction],
    order: Sequence[int],
) -> tuple[list[ScaledComplex], int]:
    """Batch cubic treewidth-DP for kernels sharing q2/q3 support."""
    if not q_batch:
        return [], 0

    ref_q = q_batch[0]
    support_key = _q_cubic_treewidth_batch_support_key(ref_q)
    q2_support, q3_support = _build_cubic_treewidth_batch_support(ref_q)
    for q in q_batch[1:]:
        if q.n != ref_q.n or q.level != ref_q.level:
            raise ValueError("Treewidth batch requires matching variable count and precision level.")
        if _q_cubic_treewidth_batch_support_key(q) != support_key:
            raise ValueError("Treewidth batch requires shared q2/q3 support.")

    if int(ref_q.level) == 3:
        native_batch = _sum_native_level3_phase3_treewidth_batch_shared_support(q_batch, order)
        if native_batch is not None:
            return native_batch
    native_batch = _sum_native_phase_function_treewidth_batch_shared_support(q_batch, order)
    if native_batch is not None:
        return native_batch

    batch_size = len(q_batch)
    mod_q1 = int(ref_q.mod_q1)
    q2_lift = mod_q1 // int(ref_q.mod_q2) if ref_q.mod_q2 else 0
    q3_lift = mod_q1 // int(ref_q.mod_q3) if ref_q.mod_q3 else 0
    omega_values, omega_exponents = _omega_scaled_arrays(ref_q.level)

    scalar_values = np.empty(batch_size, dtype=np.complex128)
    scalar_exponents = np.zeros(batch_size, dtype=np.int64)
    q1_batch = np.asarray([
        [int(coeff) % mod_q1 for coeff in q.q1]
        for q in q_batch
    ], dtype=np.int64)
    for row_idx, q in enumerate(q_batch):
        scalar_values[row_idx] = cmath.exp(2j * cmath.pi * float(q.q0))

    factors: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}

    active_q1 = np.flatnonzero(np.any(q1_batch != 0, axis=0))
    for var in active_q1.tolist():
        residues = q1_batch[:, int(var)]
        table_values = np.ones((batch_size, 2), dtype=np.complex128)
        table_exponents = np.zeros((batch_size, 2), dtype=np.int64)
        table_values[:, 1] = omega_values[residues]
        table_exponents[:, 1] = omega_exponents[residues]
        factors[(int(var),)] = (table_values, table_exponents)

    if q2_support:
        q2_rows = np.asarray([
            [int(q.q2.get(edge, 0)) % int(ref_q.mod_q2) for edge in q2_support]
            for q in q_batch
        ], dtype=np.int64)
        q2_residues = np.asarray((q2_rows * q2_lift) % mod_q1, dtype=np.int64)
        for edge_idx, edge in enumerate(q2_support):
            residues = q2_residues[:, edge_idx]
            table_values = np.ones((batch_size, 4), dtype=np.complex128)
            table_exponents = np.zeros((batch_size, 4), dtype=np.int64)
            table_values[:, 3] = omega_values[residues]
            table_exponents[:, 3] = omega_exponents[residues]
            factors[edge] = (table_values, table_exponents)

    if q3_support:
        q3_rows = np.asarray([
            [int(q.q3.get(edge, 0)) % int(ref_q.mod_q3) for edge in q3_support]
            for q in q_batch
        ], dtype=np.int64)
        q3_residues = np.asarray((q3_rows * q3_lift) % mod_q1, dtype=np.int64)
        for edge_idx, edge in enumerate(q3_support):
            residues = q3_residues[:, edge_idx]
            table_values = np.ones((batch_size, 8), dtype=np.complex128)
            table_exponents = np.zeros((batch_size, 8), dtype=np.int64)
            table_values[:, 7] = omega_values[residues]
            table_exponents[:, 7] = omega_exponents[residues]
            factors[edge] = (table_values, table_exponents)

    return _sum_factor_tables_scaled_batch(
        ref_q.n,
        factors,
        order,
        scalar=(scalar_values, scalar_exponents),
    )


def _sum_via_treewidth_dp_peeled_scaled(q, order):
    """Cached/native exact DP for fully peeled cubic kernels."""
    if _native_level3_enabled(q):
        planned = _sum_native_level3_phase3_treewidth_preplanned(q=q, order=order)
        if planned is not None:
            return _make_scaled_complex(planned[0]), int(planned[1])
        core_total, max_scope = _schur_native.sum_treewidth_dp_level3(
            q.n,
            q.q1,
            q.q2,
            q.q3,
            order,
        )
        return _make_scaled_complex(
            cmath.exp(2j * cmath.pi * float(q.q0)) * complex(core_total),
        ), max_scope
    scalar, factors = _build_cached_phase3_treewidth_factor_plan_scaled(q)
    native_plan = _build_native_phase3_treewidth_plan(q=q, order=order)
    if (
        native_plan is not None
        and _schur_native is not None
        and hasattr(_schur_native, "sum_scaled_factor_treewidth_preplanned")
    ):
        try:
            core_total, max_scope = _schur_native.sum_scaled_factor_treewidth_preplanned(native_plan)
            return _mul_scaled_complex(scalar, (complex(core_total[0]), int(core_total[1]))), int(max_scope)
        except Exception:
            pass
    return _sum_factor_tables_scaled(q.n, factors, order, scalar=scalar)


def _sum_via_treewidth_dp_peeled(q, order):
    total, max_scope = _sum_via_treewidth_dp_peeled_scaled(q, order)
    return _scaled_to_complex(total), max_scope

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
