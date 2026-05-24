"""Extracted q3-free cutset evaluators."""

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
    '_sum_q3_free_via_cutset_conditioning_scaled',
    '_evaluate_q3_free_cutset_conditioning_plan_scaled_batch',
    '_evaluate_q3_free_cutset_conditioning_plan_scaled',
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


def _sum_q3_free_via_cutset_conditioning_scaled(q: PhaseFunction) -> ScaledComplex | None:
    plan = _q3_free_cutset_conditioning_plan(q)
    if plan is None:
        return None

    branch_count = 1 << len(plan.cutset_vars)
    branch_masks = np.arange(branch_count, dtype=np.uint64)
    branch_bits = _branch_assignment_bits(branch_masks, len(plan.cutset_vars))
    q0_eff = np.full(branch_count, _phase_fraction_to_residue(q.q0, q.mod_q1), dtype=np.int64)

    if len(plan.cutset_vars):
        cutset_q1 = np.asarray([q.q1[var] % q.mod_q1 for var in plan.cutset_vars], dtype=np.int64)
        q0_eff = (q0_eff + branch_bits @ cutset_q1) % q.mod_q1

    if plan.cutset_cutset_residue.size:
        for left, right, residue in zip(
            plan.cutset_cutset_left,
            plan.cutset_cutset_right,
            plan.cutset_cutset_residue,
        ):
            q0_eff = (
                q0_eff
                + int(residue) * branch_bits[:, int(left)] * branch_bits[:, int(right)]
            ) % q.mod_q1

    if len(plan.remaining_vars):
        base_q1 = np.fromiter(
            (q.q1[int(var)] % q.mod_q1 for var in plan.remaining_vars),
            dtype=np.int64,
            count=len(plan.remaining_vars),
        )
        q1_batch = np.broadcast_to(base_q1, (branch_count, len(plan.remaining_vars))).copy()
        if plan.cutset_remaining_q2_residue.size:
            np.add(q1_batch, branch_bits @ plan.cutset_remaining_q2_residue, out=q1_batch)
            np.remainder(q1_batch, q.mod_q1, out=q1_batch)
    else:
        q1_batch = np.zeros((branch_count, 0), dtype=np.int64)

    if plan.remaining_backend == "product":
        branch_totals = [_product_q1_sum_scaled(row, level=q.level) for row in q1_batch]
    elif plan.remaining_backend == "generic":
        unique_batch, inverse = _fold_phase_shifted_q1_batch(q1_batch)
        unique_totals: list[ScaledComplex] = [_ONE_SCALED] * len(unique_batch)
        if len(plan.remaining_isolated_vars):
            isolated_columns = unique_batch[:, plan.remaining_isolated_vars]
            unique_totals = [
                _product_q1_sum_scaled(row, level=q.level)
                for row in isolated_columns
            ]
        for component_plan in plan.remaining_components:
            component_batch = unique_batch[:, component_plan.variables]
            component_totals = _evaluate_q3_free_component_plan_scaled_batch(
                component_plan,
                component_batch,
                level=q.level,
            )
            unique_totals = [
                _mul_scaled_complex(total, component_total)
                for total, component_total in zip(unique_totals, component_totals)
            ]
        branch_totals = [unique_totals[idx] for idx in inverse]
    else:
        if q1_batch.size == 0:
            branch_totals = [_ONE_SCALED] * branch_count
        else:
            unique_batch, inverse = _fold_phase_shifted_q1_batch(q1_batch)
            unique_totals = _sum_q3_free_treewidth_dp_scaled_batch(
                n_vars=len(plan.remaining_vars),
                level=q.level,
                q1_batch=unique_batch,
                q2=plan.remaining_q2,
                order=plan.remaining_order,
                native_plan=plan.native_treewidth_plan,
            )
            branch_totals = [unique_totals[idx] for idx in inverse]

    omega_scaled = _omega_scaled_table(q.level)
    total = _ZERO_SCALED
    for q0_residue, branch_total in zip(q0_eff, branch_totals):
        total = _add_scaled_complex(
            total,
            _mul_scaled_complex(omega_scaled[int(q0_residue) % q.mod_q1], branch_total),
        )
    return total

def _evaluate_q3_free_cutset_conditioning_plan_scaled_batch(
    plan: _Q3FreeCutsetConditioningPlan,
    q1_batch: np.ndarray,
    *,
    level: int,
) -> list[ScaledComplex]:
    """Evaluate a reusable q3-free cutset plan for many q1 assignments."""
    plan = _attach_q3_free_cutset_runtime_cache(plan)
    q1_batch = np.asarray(q1_batch, dtype=np.int64)
    if q1_batch.ndim != 2:
        raise ValueError("Expected q1_batch to have shape (batch, n_vars).")

    n_vars = len(plan.cutset_vars) + len(plan.remaining_vars)
    if q1_batch.shape[1] != n_vars:
        raise ValueError(f"Expected q1 rows of length {n_vars}, received {q1_batch.shape[1]}.")

    batch_size = q1_batch.shape[0]
    if batch_size == 0:
        return []

    mod_q1 = 1 << int(level)
    branch_bits = np.asarray(plan.branch_bits, dtype=np.int64)
    branch_count = int(branch_bits.shape[0])
    q0_eff = np.zeros((batch_size, branch_count), dtype=np.int64)
    if len(plan.cutset_vars):
        cutset_q1 = np.asarray(q1_batch[:, plan.cutset_vars], dtype=np.int64) % mod_q1
        q0_eff = cutset_q1 @ branch_bits.T
        np.remainder(q0_eff, mod_q1, out=q0_eff)

    branch_pair_residue = np.asarray(plan.branch_pair_residue, dtype=np.int64)
    if branch_pair_residue.size:
        np.add(q0_eff, branch_pair_residue[None, :], out=q0_eff)
        np.remainder(q0_eff, mod_q1, out=q0_eff)

    if len(plan.remaining_vars):
        base_remaining_q1 = np.array(q1_batch[:, plan.remaining_vars], dtype=np.int64, copy=True)
        np.remainder(base_remaining_q1, mod_q1, out=base_remaining_q1)
        branch_remaining_shift = np.asarray(plan.branch_remaining_shift, dtype=np.int64)
        if branch_remaining_shift.size:
            remaining_q1 = base_remaining_q1[:, None, :] + branch_remaining_shift[None, :, :]
            np.remainder(remaining_q1, mod_q1, out=remaining_q1)
        else:
            remaining_q1 = np.broadcast_to(
                base_remaining_q1[:, None, :],
                (batch_size, branch_count, len(plan.remaining_vars)),
            ).copy()
    else:
        remaining_q1 = np.zeros((batch_size, branch_count, 0), dtype=np.int64)

    if plan.remaining_backend == "product":
        branch_totals = [
            _product_q1_sum_scaled(row, level=level)
            for row in remaining_q1.reshape(batch_size * branch_count, -1)
        ]
    elif plan.remaining_backend == "generic":
        flat_remaining_q1 = remaining_q1.reshape(batch_size * branch_count, -1)
        unique_batch, inverse = _fold_phase_shifted_q1_batch(flat_remaining_q1)
        unique_totals: list[ScaledComplex] = [_ONE_SCALED] * len(unique_batch)
        if len(plan.remaining_isolated_vars):
            isolated_columns = unique_batch[:, plan.remaining_isolated_vars]
            unique_totals = [
                _product_q1_sum_scaled(row, level=level)
                for row in isolated_columns
            ]
        for component_plan in plan.remaining_components:
            component_batch = unique_batch[:, component_plan.variables]
            component_totals = _evaluate_q3_free_component_plan_scaled_batch(
                component_plan,
                component_batch,
                level=level,
            )
            unique_totals = [
                _mul_scaled_complex(total, component_total)
                for total, component_total in zip(unique_totals, component_totals)
            ]
        branch_totals = [unique_totals[idx] for idx in inverse]
    else:
        if remaining_q1.size == 0:
            branch_totals = [_ONE_SCALED] * (batch_size * branch_count)
        else:
            flat_remaining_q1 = remaining_q1.reshape(batch_size * branch_count, -1)
            if flat_remaining_q1.shape[0] <= 2:
                branch_totals = _sum_q3_free_treewidth_dp_scaled_batch(
                    n_vars=len(plan.remaining_vars),
                    level=level,
                    q1_batch=np.ascontiguousarray(flat_remaining_q1, dtype=np.int64),
                    q2=plan.remaining_q2,
                    order=plan.remaining_order,
                    native_plan=plan.native_treewidth_plan,
                )
            else:
                unique_batch, inverse = _fold_phase_shifted_q1_batch(flat_remaining_q1)
                unique_totals = _sum_q3_free_treewidth_dp_scaled_batch(
                    n_vars=len(plan.remaining_vars),
                    level=level,
                    q1_batch=unique_batch,
                    q2=plan.remaining_q2,
                    order=plan.remaining_order,
                    native_plan=plan.native_treewidth_plan,
                )
                branch_totals = [unique_totals[idx] for idx in inverse]

    omega_scaled = _omega_scaled_table(level)
    totals: list[ScaledComplex] = []
    for row_idx in range(batch_size):
        row_total = _ZERO_SCALED
        base_idx = row_idx * branch_count
        for branch_idx in range(branch_count):
            row_total = _add_scaled_complex(
                row_total,
                _mul_scaled_complex(
                    omega_scaled[int(q0_eff[row_idx, branch_idx]) % mod_q1],
                    branch_totals[base_idx + branch_idx],
                ),
            )
        totals.append(row_total)
    return totals

def _evaluate_q3_free_cutset_conditioning_plan_scaled(
    plan: _Q3FreeCutsetConditioningPlan,
    q1_local: Sequence[int],
    *,
    level: int,
) -> ScaledComplex:
    """Evaluate one reusable q3-free cutset plan under a concrete q1 vector."""
    return _evaluate_q3_free_cutset_conditioning_plan_scaled_batch(
        plan,
        np.asarray([q1_local], dtype=np.int64),
        level=level,
    )[0]

# ==================================================================
# Phase-3 backend planning and execution
# ==================================================================

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
