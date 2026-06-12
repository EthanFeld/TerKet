"""q3-free primitive graph, residue, and unary-expansion helpers."""

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

from ..scaling import _omega_table
from ..cubic_arithmetic import CubicFunction, PhaseFunction, detect_factorization
from .._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals
from ..scaling import ScaledAmplitude, ScaledComplex
from ..spec import CircuitSpec, Gate
from ..state import BitSequence, CircuitInput, EchelonCache, ExtendedReductionMode, ReducerInfo, ReductionInfo, SolverConfig

_LOCAL_NAMES = {
    '_q3_free_graph',
    '_is_binary_phase_quadratic',
    '_is_half_phase_q2',
    '_is_binary_phase_q1_vector',
    '_nonbinary_unary_support_size',
    '_is_qubit_quadratic_tensor_q1_vector',
    '_is_qubit_quadratic_tensor',
    '_q3_free_phase3_backend_name',
    '_component_fixed_nonbinary_unary_support_size',
    '_build_binary_phase_quadratic_plan',
    '_evaluate_binary_phase_quadratic_plan_scaled_batch',
    '_sum_binary_phase_quadratic_scaled',
    '_sum_half_phase_q2_unary_expansion_with_plan_scaled',
    '_sum_half_phase_q2_unary_expansion_with_plan_scaled_batch',
    '_sum_half_phase_q2_unary_expansion_scaled',
    '_apply_safe_q3_free_parity_substitutions',
    '_half_phase_parity_component_reduction',
    '_sum_half_phase_parity_component_reduction_scaled',
    '_build_half_phase_mediator_plan',
    '_build_generic_q2_mediator_plan'
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


def _q3_free_graph(q):
    """Return the q2 graph with edge phases represented in q1 residues."""
    adjacency = [dict() for _ in range(q.n)]
    edges = []
    for (i, j), value in q.q2.items():
        phase_shift = ((q.mod_q1 // q.mod_q2) * value) % q.mod_q1
        adjacency[i][j] = phase_shift
        adjacency[j][i] = phase_shift
        edges.append((i, j, phase_shift))
    return adjacency, edges

def _is_binary_phase_quadratic(q) -> bool:
    """Return whether ``q`` only contributes +/-1 phases on computational basis states."""
    if not _is_half_phase_q2(q):
        return False
    return _is_binary_phase_q1_vector(q.q1, level=q.level)

def _is_half_phase_q2(q) -> bool:
    """Return whether every quadratic coupling is in the half-phase class."""
    if q.q3:
        return False
    half_q2 = q.mod_q2 // 2 if q.mod_q2 else 0
    for coeff in q.q2.values():
        residue = coeff % q.mod_q2
        if residue not in (0, half_q2):
            return False
    return True

def _is_binary_phase_q1_vector(q1, *, level: int) -> bool:
    """Return whether every unary residue is binary (`0` or `half`)."""
    modulus = 1 << level
    half_q1 = modulus // 2
    for coeff in q1:
        residue = int(coeff) % modulus
        if residue not in (0, half_q1):
            return False
    return True

def _nonbinary_unary_support_size(q1, *, level: int) -> int:
    """Count unary residues outside the binary-phase class (`0` or `half`)."""
    modulus = 1 << level
    half_q1 = modulus // 2
    support_size = 0
    for coeff in q1:
        residue = int(coeff) % modulus
        if residue not in (0, half_q1):
            support_size += 1
    return support_size

def _is_qubit_quadratic_tensor_q1_vector(q1, *, level: int) -> bool:
    """Return whether ``q1`` lies in BL26's 4-element qubit quadratic class."""
    threshold = max(1, (1 << level) // 4)
    if threshold <= 1:
        return True
    for coeff in q1:
        if int(coeff) % threshold:
            return False
    return True

def _is_qubit_quadratic_tensor(q) -> bool:
    """Return whether ``q`` is a q3-free BL26 qubit quadratic tensor."""
    return not q.q3 and _qubit_quadratic_tensor_obstruction(q) == 0

def _q3_free_phase3_backend_name(q) -> str:
    """Report the exact backend family for a q3-free kernel."""
    return "quadratic_tensor" if _is_qubit_quadratic_tensor(q) else "q3_free"

def _component_fixed_nonbinary_unary_support_size(
    component_q,
    variables: Sequence[int],
    *,
    lambda_offset: int,
) -> int:
    """Count non-binary unary residues on the original, non-dual variables."""
    modulus = 1 << component_q.level
    half_q1 = modulus // 2
    support_size = 0
    for local_idx, var in enumerate(variables):
        if var >= lambda_offset:
            continue
        residue = int(component_q.q1[local_idx]) % modulus
        if residue not in (0, half_q1):
            support_size += 1
    return support_size

def _build_binary_phase_quadratic_plan(q) -> _BinaryPhaseQuadraticPlan | None:
    """Precompute the half-phase q2 elimination schedule for a q3-free kernel."""
    if not _is_half_phase_q2(q):
        return None

    adjacency = np.zeros((q.n, q.n), dtype=np.bool_)
    half_q2 = q.mod_q2 // 2 if q.mod_q2 else 0
    for (left, right), coeff in q.q2.items():
        if (coeff % q.mod_q2) == half_q2:
            adjacency[left, right] = True
            adjacency[right, left] = True

    active_count = q.n
    partner_swaps: list[int] = []
    pivot_swaps: list[int] = []
    c1_rows: list[np.ndarray] = []
    c2_rows: list[np.ndarray] = []
    c1_and_c2_rows: list[np.ndarray] = []
    half_pow2 = 0

    while active_count:
        block = adjacency[:active_count, :active_count]
        degrees = block.sum(axis=1, dtype=np.int64)
        if not np.any(degrees):
            break

        pivot_local = int(np.argmax(degrees))
        neighbors_local = np.flatnonzero(block[pivot_local])
        partner_local = int(neighbors_local[0])
        if partner_local == pivot_local:
            if neighbors_local.size < 2:
                return None
            partner_local = int(neighbors_local[1])

        if pivot_local > partner_local:
            pivot_local, partner_local = partner_local, pivot_local

        first_pivot = active_count - 2
        second_pivot = active_count - 1

        partner_swap = -1
        if partner_local != second_pivot:
            partner_swap = partner_local
            _swap_dense_q2_variables(np.zeros(q.n, dtype=np.bool_), adjacency, partner_local, second_pivot)
            if pivot_local == second_pivot:
                pivot_local = partner_local

        pivot_swap = -1
        if pivot_local != first_pivot:
            pivot_swap = pivot_local
            _swap_dense_q2_variables(np.zeros(q.n, dtype=np.bool_), adjacency, pivot_local, first_pivot)

        c1 = adjacency[first_pivot, :first_pivot].copy()
        c2 = adjacency[second_pivot, :first_pivot].copy()
        c1_rows.append(c1)
        c2_rows.append(c2)
        c1_and_c2_rows.append(np.logical_and(c1, c2))
        partner_swaps.append(partner_swap)
        pivot_swaps.append(pivot_swap)

        if first_pivot:
            update = np.logical_xor(np.outer(c1, c2), np.outer(c2, c1))
            subblock = adjacency[:first_pivot, :first_pivot]
            subblock ^= update
            np.fill_diagonal(subblock, False)
        adjacency[first_pivot:active_count, :active_count] = False
        adjacency[:active_count, first_pivot:active_count] = False
        active_count = first_pivot
        half_pow2 += 2

    return _BinaryPhaseQuadraticPlan(
        n=q.n,
        residual_active_count=active_count,
        half_pow2_exp=half_pow2,
        partner_swaps=tuple(partner_swaps),
        pivot_swaps=tuple(pivot_swaps),
        c1_rows=tuple(c1_rows),
        c2_rows=tuple(c2_rows),
        c1_and_c2_rows=tuple(c1_and_c2_rows),
    )

def _evaluate_binary_phase_quadratic_plan_scaled_batch(
    plan: _BinaryPhaseQuadraticPlan,
    q1_batch: np.ndarray,
    *,
    level: int,
) -> list[ScaledComplex]:
    """Evaluate many binary-phase q1 assignments over one fixed dense q2 plan."""
    q1_batch = np.ascontiguousarray(np.asarray(q1_batch, dtype=np.int64))
    if q1_batch.ndim != 2 or q1_batch.shape[1] != plan.n:
        raise ValueError("q1_batch must have shape (batch, plan.n).")

    half_q1 = (1 << level) // 2
    work = np.remainder(q1_batch, 1 << level) == half_q1
    sign_bits = np.zeros(work.shape[0], dtype=np.bool_)
    active_count = plan.n

    for partner_swap, pivot_swap, c1, c2, c1_and_c2 in zip(
        plan.partner_swaps,
        plan.pivot_swaps,
        plan.c1_rows,
        plan.c2_rows,
        plan.c1_and_c2_rows,
    ):
        first_pivot = active_count - 2
        second_pivot = active_count - 1
        if partner_swap >= 0:
            tmp = work[:, partner_swap].copy()
            work[:, partner_swap] = work[:, second_pivot]
            work[:, second_pivot] = tmp
        if pivot_swap >= 0:
            tmp = work[:, pivot_swap].copy()
            work[:, pivot_swap] = work[:, first_pivot]
            work[:, first_pivot] = tmp

        sign_bits ^= np.logical_and(work[:, first_pivot], work[:, second_pivot])
        if first_pivot:
            work[:, :first_pivot] ^= (
                (work[:, [first_pivot]] & c2[None, :])
                ^ (work[:, [second_pivot]] & c1[None, :])
                ^ c1_and_c2[None, :]
            )
        work[:, first_pivot:active_count] = False
        active_count = first_pivot

    zero_mask = np.any(work[:, :plan.residual_active_count], axis=1)
    result: list[ScaledComplex] = []
    final_half_pow2 = plan.half_pow2_exp + (2 * plan.residual_active_count)
    for is_zero, sign_bit in zip(zero_mask, sign_bits):
        if is_zero:
            result.append(_ZERO_SCALED)
        else:
            result.append(
                _scale_scaled_complex(
                    _make_scaled_complex(-1.0 if bool(sign_bit) else 1.0),
                    final_half_pow2,
                )
            )
    return result

def _sum_binary_phase_quadratic_scaled(q) -> ScaledComplex | None:
    """Exactly sum a q3-free kernel whose phases are only +/-1.

    In this regime the exponent is a boolean quadratic form over GF(2):
    ``sum_{i<j} a_ij x_i x_j + sum_i b_i x_i``. Summing over any edge pair
    ``(i, j)`` produces another boolean quadratic form on the remaining
    variables:

    ``sum_{x_i, x_j} (-1)^{x_i x_j + p x_i + q x_j} = 2 (-1)^{p q}``

    where ``p`` and ``q`` are affine forms in the remaining variables. This
    turns the exact sum into a sequence of dense GF(2) pivot eliminations with
    O(n^3) worst-case cost instead of the generic feedback-set branching path.
    """
    if not _is_binary_phase_quadratic(q):
        return None
    if q.n == 0:
        return _ONE_SCALED

    plan = _build_binary_phase_quadratic_plan(q)
    if plan is None:
        return None
    return _evaluate_binary_phase_quadratic_plan_scaled_batch(
        plan,
        np.asarray([q.q1], dtype=np.int64),
        level=q.level,
    )[0]

def _sum_half_phase_q2_unary_expansion_with_plan_scaled(
    q1: Sequence[int],
    *,
    level: int,
    plan: _BinaryPhaseQuadraticPlan,
) -> ScaledComplex | None:
    """Exactly sum a half-phase q2 core by expanding only the hard unary terms.

    For each binary variable ``x_i`` with unary phase ``omega^(a_i x_i)``, use

    ``omega^(a_i x_i) = alpha_i + beta_i (-1)^(x_i)``

    where ``alpha_i = (1 + omega^a_i) / 2`` and
    ``beta_i = (1 - omega^a_i) / 2``.

    This turns the exact sum into a weighted combination of binary-phase
    quadratic character sums over the same q2 core, which can be evaluated by
    ``_evaluate_binary_phase_quadratic_plan_scaled_batch``. The expansion is
    only practical when the number of non-binary unary residues is small.
    """
    if len(q1) != plan.n:
        raise ValueError(f"Expected q1 of length {plan.n}, received {len(q1)}.")

    modulus = 1 << level
    half_q1 = modulus // 2
    omega = _omega_table(level)

    fixed_half_positions: list[int] = []
    support_positions: list[int] = []
    alpha_terms: list[complex] = []
    beta_terms: list[complex] = []

    for idx, coeff in enumerate(q1):
        residue = int(coeff) % modulus
        if residue == 0:
            continue
        if residue == half_q1:
            fixed_half_positions.append(idx)
            continue
        support_positions.append(idx)
        phase = omega[residue]
        alpha_terms.append((1.0 + phase) * 0.5)
        beta_terms.append((1.0 - phase) * 0.5)

    support_size = len(support_positions)
    if support_size > _Q3_FREE_HALF_PHASE_UNARY_EXPANSION_MAX_SUPPORT:
        return None

    if support_size == 0:
        base_q1 = np.zeros((1, plan.n), dtype=np.int64)
        if fixed_half_positions:
            base_q1[0, np.asarray(fixed_half_positions, dtype=np.int64)] = half_q1
        return _evaluate_binary_phase_quadratic_plan_scaled_batch(
            plan,
            base_q1,
            level=level,
        )[0]

    base_q1 = np.zeros(plan.n, dtype=np.int64)
    if fixed_half_positions:
        base_q1[np.asarray(fixed_half_positions, dtype=np.int64)] = half_q1

    alpha_array = np.asarray(alpha_terms, dtype=np.complex128)
    beta_array = np.asarray(beta_terms, dtype=np.complex128)
    support_array = np.asarray(support_positions, dtype=np.int64)
    total = _ZERO_SCALED
    mask_count = 1 << support_size
    batch_size = min(_Q3_FREE_HALF_PHASE_UNARY_EXPANSION_BATCH_SIZE, mask_count)

    for start in range(0, mask_count, batch_size):
        stop = min(start + batch_size, mask_count)
        masks = np.arange(start, stop, dtype=np.uint64)
        q1_batch = np.broadcast_to(base_q1, (stop - start, plan.n)).copy()
        coeff_batch = np.ones(stop - start, dtype=np.complex128)

        for local_idx, position in enumerate(support_array):
            bit_is_one = ((masks >> np.uint64(local_idx)) & np.uint64(1)).astype(np.bool_)
            q1_batch[bit_is_one, position] = half_q1
            coeff_batch *= np.where(bit_is_one, beta_array[local_idx], alpha_array[local_idx])

        binary_totals = _evaluate_binary_phase_quadratic_plan_scaled_batch(
            plan,
            q1_batch,
            level=level,
        )
        for coeff, binary_total in zip(coeff_batch, binary_totals):
            if coeff == 0j or binary_total[0] == 0j:
                continue
            total = _add_scaled_complex(
                total,
                _mul_scaled_complex(_make_scaled_complex(coeff), binary_total),
            )

    return total

def _sum_half_phase_q2_unary_expansion_with_plan_scaled_batch(
    q1_batch: np.ndarray,
    *,
    level: int,
    plan: _BinaryPhaseQuadraticPlan,
) -> list[ScaledComplex] | None:
    """Batch exact hard-unary expansion over one shared half-phase q2 core."""
    batch = np.ascontiguousarray(np.asarray(q1_batch, dtype=np.int64))
    if batch.ndim != 2 or batch.shape[1] != plan.n:
        raise ValueError(f"Expected q1_batch with shape (batch, {plan.n}).")
    if len(batch) == 0:
        return []

    modulus = 1 << level
    half_q1 = modulus // 2
    residues = np.remainder(batch, modulus)
    binary_mask = (residues == 0) | (residues == half_q1)
    support_mask = (~binary_mask) & (residues != 0)
    support_sizes = np.count_nonzero(support_mask, axis=1)
    if np.any(support_sizes > _Q3_FREE_HALF_PHASE_UNARY_EXPANSION_MAX_SUPPORT):
        return None

    fixed_mask = residues == half_q1
    omega = _omega_table(level)
    results: list[ScaledComplex] = [_ZERO_SCALED] * len(batch)
    grouped_rows: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
    for row_idx in range(len(batch)):
        support_positions = tuple(np.flatnonzero(support_mask[row_idx]).tolist())
        fixed_positions = tuple(np.flatnonzero(fixed_mask[row_idx] & ~support_mask[row_idx]).tolist())
        grouped_rows.setdefault((support_positions, fixed_positions), []).append(row_idx)

    for (support_positions, fixed_positions), row_indices in grouped_rows.items():
        support_size = len(support_positions)
        row_array = np.asarray(row_indices, dtype=np.int64)
        group_size = len(row_array)

        base_rows = np.zeros((group_size, plan.n), dtype=np.int64)
        if fixed_positions:
            base_rows[:, np.asarray(fixed_positions, dtype=np.int64)] = half_q1

        if support_size == 0:
            group_totals = _evaluate_binary_phase_quadratic_plan_scaled_batch(
                plan,
                base_rows,
                level=level,
            )
            for row_idx, total in zip(row_indices, group_totals):
                results[row_idx] = total
            continue

        support_array = np.asarray(support_positions, dtype=np.int64)
        residue_group = residues[row_array][:, support_array]
        alpha_matrix = (1.0 + np.vectorize(lambda coeff: omega[int(coeff)])(residue_group)) * 0.5
        beta_matrix = (1.0 - np.vectorize(lambda coeff: omega[int(coeff)])(residue_group)) * 0.5
        total_values, total_exponents = _scaled_arrays_from_constant(_ZERO_SCALED, (group_size,))
        mask_count = 1 << support_size
        block_size = min(_Q3_FREE_HALF_PHASE_UNARY_EXPANSION_BATCH_SIZE, mask_count)

        for start in range(0, mask_count, block_size):
            stop = min(start + block_size, mask_count)
            masks = np.arange(start, stop, dtype=np.uint64)
            block_rows = stop - start
            q1_expanded = np.broadcast_to(
                base_rows[:, None, :],
                (group_size, block_rows, plan.n),
            ).copy()
            coeff_matrix = np.ones((group_size, block_rows), dtype=np.complex128)

            for local_idx, position in enumerate(support_array):
                bit_is_one = ((masks >> np.uint64(local_idx)) & np.uint64(1)).astype(np.bool_)
                if np.any(bit_is_one):
                    q1_expanded[:, bit_is_one, position] = half_q1
                coeff_matrix *= np.where(
                    bit_is_one[None, :],
                    beta_matrix[:, [local_idx]],
                    alpha_matrix[:, [local_idx]],
                )

            binary_totals = _evaluate_binary_phase_quadratic_plan_scaled_batch(
                plan,
                q1_expanded.reshape(group_size * block_rows, plan.n),
                level=level,
            )
            block_values, block_exponents = _scaled_list_to_arrays(
                binary_totals,
                (group_size, block_rows),
            )
            weighted_values, weighted_exponents = _mul_scaled_complex_arrays(
                np.broadcast_to(coeff_matrix, block_values.shape),
                np.zeros(block_values.shape, dtype=np.int64),
                block_values,
                block_exponents,
            )
            block_total_values = weighted_values[:, 0]
            block_total_exponents = weighted_exponents[:, 0]
            for column in range(1, block_rows):
                block_total_values, block_total_exponents = _add_scaled_complex_arrays(
                    block_total_values,
                    block_total_exponents,
                    weighted_values[:, column],
                    weighted_exponents[:, column],
                )
            total_values, total_exponents = _add_scaled_complex_arrays(
                total_values,
                total_exponents,
                block_total_values,
                block_total_exponents,
            )

        for row_idx, value, half_pow2_exp in zip(row_indices, total_values, total_exponents):
            results[row_idx] = (complex(value), int(half_pow2_exp))

    return results

def _sum_half_phase_q2_unary_expansion_scaled(q) -> ScaledComplex | None:
    """Exact hard-unary expansion over a half-phase q2 core, when support is small."""
    if not _is_half_phase_q2(q):
        return None
    if _nonbinary_unary_support_size(q.q1, level=q.level) > _Q3_FREE_HALF_PHASE_UNARY_EXPANSION_MAX_SUPPORT:
        return None
    plan = _build_binary_phase_quadratic_plan(q)
    if plan is None:
        return None
    return _sum_half_phase_q2_unary_expansion_with_plan_scaled(
        q.q1,
        level=q.level,
        plan=plan,
    )

def _q3_free_nonzero_q2_adjacency(q) -> list[set[int]]:
    adjacency = [set() for _ in range(q.n)]
    for (left, right), coeff in q.q2.items():
        if coeff % q.mod_q2:
            adjacency[left].add(right)
            adjacency[right].add(left)
    return adjacency

def _best_q3_free_double_parity_action(
    q,
    *,
    var: int,
    left: int,
    right: int,
    adjacency: Sequence[set[int]],
) -> tuple[tuple[int, int, int], tuple[str, int, int, int, int]]:
    fill_left_to_right = sum(
        1
        for neighbor in adjacency[left]
        if neighbor not in (right, var) and neighbor not in adjacency[right]
    )
    fill_right_to_left = sum(
        1
        for neighbor in adjacency[right]
        if neighbor not in (left, var) and neighbor not in adjacency[left]
    )
    if fill_right_to_left <= fill_left_to_right:
        keep, remove = left, right
        fill_cost = fill_right_to_left
    else:
        keep, remove = right, left
        fill_cost = fill_left_to_right
    return (fill_cost, len(adjacency[remove]), var), ("double", var, keep, remove)

def _choose_q3_free_parity_action(
    q,
    *,
    classification_data,
    threshold: int,
    include_zero_and_decoupled: bool,
) -> tuple[bool, list[int], tuple | None]:
    adjacency = _q3_free_nonzero_q2_adjacency(q)
    decoupled_constraints: list[int] = []
    best_action = None
    best_score = None
    half_q1 = q.mod_q1 // 2
    for var in range(q.n):
        entry = _classification_entry(
            q,
            var,
            classification_data=classification_data,
            threshold=threshold,
        )
        tag = entry[0]
        if include_zero_and_decoupled:
            if tag == _CLASS_CONSTRAINT_ZERO:
                return True, [], None
            if tag == _CLASS_CONSTRAINT_DECOUPLED:
                decoupled_constraints.append(var)
                continue
        if tag != _CLASS_CONSTRAINT_PARITY:
            continue
        partners = tuple(int(partner) for partner in entry[1])
        target = 1 if int(entry[2]) % q.mod_q1 == half_q1 else 0
        if len(partners) == 1:
            score = (-1, var)
            action = ("single", var, partners[0], target)
        elif len(partners) == 2:
            score, partial_action = _best_q3_free_double_parity_action(
                q,
                var=var,
                left=partners[0],
                right=partners[1],
                adjacency=adjacency,
            )
            action = partial_action + (target,)
        else:
            continue
        if best_score is None or score < best_score:
            best_score = score
            best_action = action
    return False, decoupled_constraints, best_action

def _apply_q3_free_parity_action(
    q,
    action: tuple[str, int, int, int] | tuple[str, int, int, int, int],
):
    if action[0] == "single":
        return _elim_single_partner_constraint(
            q,
            action[1],
            action[2],
            action[3],
        )
    return _elim_two_partner_constraint_q3_free(
        q,
        action[1],
        action[2],
        action[3],
        action[4],
    )

def _apply_safe_q3_free_parity_substitutions(
    q: PhaseFunction,
) -> tuple[PhaseFunction | None, int, bool]:
    """Apply only q3-free-preserving exact parity substitutions.

    This is the proved-safe high-precision subset:
    - decoupled constraints
    - zero constraints
    - single-partner parity constraints
    - two-partner parity constraints in q3-free kernels

    Higher-arity parity substitutions are deliberately skipped because the
    affine recomposition can create q3 or higher terms outside the current
    backend representation.
    """
    assert not q.q3, "safe parity substitutions expect a q3-free kernel."

    reduced_q = q
    scale_half_pow2 = 0
    changed = False

    while True:
        classification_data = _build_classification_data(reduced_q)
        threshold = max(1, reduced_q.mod_q1 // 4)
        zero_found, decoupled_constraints, best_action = _choose_q3_free_parity_action(
            reduced_q,
            classification_data=classification_data,
            threshold=threshold,
            include_zero_and_decoupled=True,
        )
        if zero_found:
            return None, 0, True

        if decoupled_constraints:
            reduced_q, half_pow2 = _elim_decoupled_constraints_batch(reduced_q, decoupled_constraints)
            scale_half_pow2 += half_pow2
            changed = True
            continue

        if best_action is None:
            break
        result = _apply_q3_free_parity_action(reduced_q, best_action)
        if result is None:
            break
        reduced_q, half_pow2 = result
        scale_half_pow2 += half_pow2
        changed = True

    return reduced_q, scale_half_pow2, changed

def _half_phase_parity_component_reduction(q) -> tuple[object, int] | ScaledComplex | None:
    """Peel q3-free-preserving low-arity parity constraints.

    A parity constraint with one partner fixes that partner.  A parity
    constraint with two partners enforces ``x_a xor x_b = target``; substituting
    one partner into the other keeps every q3-free half-phase q2 term at most
    pairwise.  Higher-arity parity constraints are deliberately left alone
    because the analogous substitution can create q3 terms.
    """
    if q.q3 or not _is_half_phase_q2(q):
        return None

    reduced_q = q
    scale_half_pow2 = 0
    eliminated = 0

    while True:
        classification_data = _build_classification_data(reduced_q)
        threshold = max(1, reduced_q.mod_q1 // 4)
        _zero_found, _decoupled_constraints, best_action = _choose_q3_free_parity_action(
            reduced_q,
            classification_data=classification_data,
            threshold=threshold,
            include_zero_and_decoupled=False,
        )

        if best_action is None:
            break
        result = _apply_q3_free_parity_action(reduced_q, best_action)
        if result is None:
            break
        reduced_q, half_pow2 = result
        scale_half_pow2 += half_pow2
        eliminated += 1

    if not eliminated:
        return None
    return reduced_q, scale_half_pow2

def _sum_half_phase_parity_component_reduction_scaled(q) -> ScaledComplex | None:
    """Exact q3-free sum after linear parity-component collapse, when useful."""
    reduction = _half_phase_parity_component_reduction(q)
    if reduction is None:
        return None
    if reduction == _ZERO_SCALED:
        return _ZERO_SCALED
    reduced_q, scale_half_pow2 = reduction
    constant = _make_scaled_complex(cmath.exp(2j * cmath.pi * float(reduced_q.q0)))
    if reduced_q.q0:
        reduced_q = _phase_function_from_parts(
            reduced_q.n,
            level=reduced_q.level,
            q0=Fraction(0),
            q1=reduced_q.q1,
            q2=reduced_q.q2,
            q3=reduced_q.q3,
        )
    total = _mul_scaled_complex(constant, _sum_q3_free_component_scaled(reduced_q))
    return _scale_scaled_complex(total, scale_half_pow2)

def _build_half_phase_mediator_plan(q) -> _HalfPhaseMediatorPlan | None:
    """Plan an exact mediator-elimination pass for half-phase q2 kernels.

    This pass targets the IBM-style pattern where every non-BL unary variable is
    an independent degree-1/2 mediator attached to a lower-treewidth core.
    Eliminating such mediators produces exact unary/pair factors on the core,
    which can then be closed by the generic factor-graph treewidth DP.
    """
    if not _is_half_phase_q2(q):
        return None
    if q.n > _Q3_FREE_OPTIONAL_REWRITE_MAX_VARS:
        return None

    threshold = max(1, q.mod_q1 // 4)
    adjacency = [set() for _ in range(q.n)]
    for (i, j), coeff in q.q2.items():
        if coeff % q.mod_q2:
            adjacency[i].add(j)
            adjacency[j].add(i)

    candidates = [
        var
        for var, coeff in enumerate(q.q1)
        if (coeff % threshold) != 0 and len(adjacency[var]) <= 2
    ]
    if not candidates:
        return None

    candidate_set = set(candidates)
    if any(neighbor in candidate_set for var in candidates for neighbor in adjacency[var]):
        return None

    core_vars = tuple(var for var in range(q.n) if var not in candidate_set)
    core_remap = {var: idx for idx, var in enumerate(core_vars)}

    core_q2 = {
        (core_remap[i], core_remap[j]): coeff
        for (i, j), coeff in q.q2.items()
        if i in core_remap and j in core_remap
    }
    dummy_q2 = {edge: 1 for edge in core_q2}
    mediator_specs: list[_HalfPhaseMediatorSpec] = []

    for var in candidates:
        neighbor_vars = tuple(sorted(core_remap[neighbor] for neighbor in adjacency[var]))
        if len(neighbor_vars) > 2:
            return None
        if len(neighbor_vars) == 2:
            edge = (neighbor_vars[0], neighbor_vars[1])
            dummy_q2.setdefault(edge, 1)
        mediator_specs.append(
            _HalfPhaseMediatorSpec(
                mediator_var=var,
                neighbor_vars=neighbor_vars,
            )
        )

    dummy_core = _phase_function_from_parts(
        len(core_vars),
        level=q.level,
        q0=Fraction(0),
        q1=[0] * len(core_vars),
        q2=dummy_q2,
        q3={},
    )
    dummy_adjacency = [set() for _ in range(len(core_vars))]
    for left, right in dummy_q2:
        dummy_adjacency[left].add(right)
        dummy_adjacency[right].add(left)
    width_limit = _q3_free_treewidth_width_limit()
    degeneracy_lower_bound = _pair_graph_degeneracy(dummy_adjacency)
    if degeneracy_lower_bound > width_limit or degeneracy_lower_bound >= len(candidates):
        return None
    order, width = _min_fill_cubic_order(dummy_core)
    separator_order = _pair_graph_separator_order(dummy_core)
    if separator_order is not None:
        candidate_order, candidate_width = separator_order
        if candidate_width < width:
            order, width = candidate_order, candidate_width
    if width > width_limit or width >= len(candidates):
        return None

    return _HalfPhaseMediatorPlan(
        level=q.level,
        core_vars=core_vars,
        core_q2=core_q2,
        order=tuple(order),
        width=width,
        mediators=tuple(mediator_specs),
    )

def _build_generic_q2_mediator_plan(q) -> _GenericQ2MediatorPlan | None:
    """Plan exact elimination of independent degree<=2 q2 mediators.

    Unlike the half-phase mediator path, this keeps the full q2 residue on each
    mediator edge and collapses the eliminated variables into exact 1- or
    2-qubit boundary factors on the remaining core. The plan is deliberately
    conservative: it only selects an independent set of low-degree mediators so
    the induced factors attach directly to the core without overlap.
    """
    if q.q3 or not q.q2 or _is_half_phase_q2(q):
        return None

    adjacency = [set() for _ in range(q.n)]
    for (left, right), coeff in q.q2.items():
        if coeff % q.mod_q2:
            adjacency[left].add(right)
            adjacency[right].add(left)

    if not any(adjacency):
        return None

    mod_q1 = 1 << q.level
    half_q1 = mod_q1 // 2
    half_q2 = q.mod_q2 // 2 if q.mod_q2 else 0
    mod_q2 = max(1, 1 << (q.level - 1))
    q2_lift = mod_q1 // mod_q2 if mod_q2 else 0

    def candidate_score(var: int) -> tuple[int, int, int]:
        unary_residue = int(q.q1[var]) % mod_q1
        hard_unary = unary_residue not in (0, half_q1)
        hard_edge = any(
            (q.q2.get((min(var, neighbor), max(var, neighbor)), 0) % q.mod_q2) not in (0, half_q2)
            for neighbor in adjacency[var]
        )
        return (int(hard_unary or hard_edge), len(adjacency[var]), -var)

    candidates = sorted(
        (var for var in range(q.n) if 0 < len(adjacency[var]) <= 2),
        key=candidate_score,
        reverse=True,
    )
    if not candidates:
        return None

    selected: list[int] = []
    blocked: set[int] = set()
    for var in candidates:
        if var in blocked:
            continue
        selected.append(var)
        blocked.add(var)
        blocked.update(adjacency[var])

    if not selected:
        return None

    selected_set = set(selected)
    core_vars = tuple(var for var in range(q.n) if var not in selected_set)
    core_remap = {var: idx for idx, var in enumerate(core_vars)}
    core_q2 = {
        (core_remap[i], core_remap[j]): coeff
        for (i, j), coeff in q.q2.items()
        if i in core_remap and j in core_remap
    }

    mediator_specs: list[_GenericQ2MediatorSpec] = []
    factor_scopes: list[tuple[int, ...]] = [edge for edge in core_q2]
    for mediator_var in selected:
        ordered_neighbors = tuple(sorted(adjacency[mediator_var]))
        neighbor_vars = tuple(core_remap[neighbor] for neighbor in ordered_neighbors)
        neighbor_couplings = tuple(
            int(q.q2.get((min(mediator_var, neighbor), max(mediator_var, neighbor)), 0))
            for neighbor in ordered_neighbors
        )
        if len(neighbor_vars) != len(neighbor_couplings):
            return None
        if neighbor_vars:
            factor_scopes.append(neighbor_vars)
        assignment_residue_shifts = tuple(
            sum(
                (q2_lift * int(coeff))
                for neighbor_idx, coeff in enumerate(neighbor_couplings)
                if (assignment >> neighbor_idx) & 1
            ) % mod_q1
            for assignment in range(1 << len(neighbor_vars))
        )
        mediator_specs.append(
            _GenericQ2MediatorSpec(
                mediator_var=mediator_var,
                neighbor_vars=neighbor_vars,
                neighbor_couplings=neighbor_couplings,
                assignment_residue_shifts=assignment_residue_shifts,
            )
        )

    width_limit = _q3_free_treewidth_width_limit()
    degeneracy_lower_bound = _factor_scope_degeneracy(len(core_vars), factor_scopes)
    if degeneracy_lower_bound > width_limit:
        return None

    order, width = _factor_scope_order(len(core_vars), factor_scopes)
    if width > width_limit:
        return None

    return _GenericQ2MediatorPlan(
        level=q.level,
        core_vars=core_vars,
        core_q2=core_q2,
        order=tuple(order),
        width=width,
        mediators=tuple(mediator_specs),
    )

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
