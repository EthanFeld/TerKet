"""Extracted direct q3-free exact backends."""

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
    '_sum_q3_free_direct_scaled',
    '_sum_factorized_components_scaled',
    '_bruteforce_q3_free_sum',
    '_q3_free_spanning_data',
    '_select_feedback_vertices',
    '_forest_transfer_sum',
    '_forest_postorder_components',
    '_forest_transfer_sum_scaled',
    '_forest_transfer_sum_scaled_batch',
    '_dense_q2_matrix',
    '_quadratic_residue_threshold',
    '_quadratic_pair_correction',
    '_phase_from_dense_q2',
    '_swap_dense_q2_variables',
    '_swap_dense_matrix_variables',
    '_schur_complement_q3_free_sum_scaled_dense',
    '_schur_complement_q3_free_sum_scaled',
    '_schur_complement_q3_free_sum',
    '_qubit_quadratic_tensor_obstruction_support',
    '_qubit_quadratic_tensor_obstruction',
    '_supports_exact_dense_schur',
    '_sum_bl26_quadratic_tensor_component_scaled',
    '_sum_bl26_quadratic_tensor_component',
    '_sum_q3_free_via_gauss_reduction_scaled',
    '_sum_q3_free_via_gauss_reduction',
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


def _sum_q3_free_direct_scaled(q, context=None):
    """Solve an already q3-free kernel directly without re-entering cubic reduction."""
    assert not q.q3, "Direct q3-free helper expects a q3-free phase function."
    if context is not None and not context.preserve_scale:
        total_complex, phase_info = _gauss_sum_q3_free(
            q,
            allow_tensor_contraction=context.allow_tensor_contraction,
        )
        total = _make_scaled_complex(total_complex)
    else:
        total, phase_info = _gauss_sum_q3_free_scaled(
            q,
            allow_tensor_contraction=(True if context is None else context.allow_tensor_contraction),
        )
    structural_obstruction = 0
    gauss_obstruction = _gauss_obstruction(q, structural_obstruction)
    return total, {
        'quad': 0,
        'constraint': 0,
        'branched': 0,
        'remaining': 0,
        'structural_obstruction': structural_obstruction,
        'gauss_obstruction': gauss_obstruction,
        'cost_r': 0,
        'phase_states': phase_info.get('phase_states', 0),
        'phase_splits': phase_info.get('phase_splits', 0),
        'phase3_backend': _q3_free_phase3_backend_name(q),
    }

def _sum_factorized_components_scaled(q, components, context=None):
    """Reduce disconnected components independently and multiply the results."""
    if not components:
        return _mul_scaled_complex(
            _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q.q0))),
            _scale_scaled_complex(_ONE_SCALED, 2 * q.n),
        ), {
            'quad': 0,
            'constraint': 0,
            'branched': 0,
            'remaining': 0,
            'structural_obstruction': 0,
            'gauss_obstruction': 0,
            'cost_r': 0,
            'phase_states': 0,
            'phase_splits': 0,
            'phase3_backend': None,
        }

    covered = set().union(*components)
    total = _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q.q0)))
    if len(covered) < q.n:
        total = _scale_scaled_complex(total, 2 * (q.n - len(covered)))

    total_quad = total_constraint = 0
    max_branched = 0
    max_remaining = 0
    max_structural = 0
    max_gauss = 0
    max_cost_r = 0
    phase_states = phase_splits = 0
    phase3_backend = None
    dominant_cost_r = -1

    for component in components:
        restricted = _component_restriction(q, component)
        if not restricted.q3:
            component_total, component_info = _sum_q3_free_direct_scaled(restricted, context=context)
        else:
            component_total, component_info = _reduce_and_sum_scaled(restricted, context=context)
        total = _mul_scaled_complex(total, component_total)
        total_quad += component_info['quad']
        total_constraint += component_info['constraint']
        max_branched = max(max_branched, component_info['branched'])
        max_remaining = max(max_remaining, component_info['remaining'])
        max_structural = max(
            max_structural,
            component_info.get('structural_obstruction', component_info['remaining']),
        )
        max_gauss = max(
            max_gauss,
            component_info.get(
                'gauss_obstruction',
                component_info.get('structural_obstruction', component_info['remaining']),
            ),
        )
        component_cost_r = component_info.get('cost_r', component_info['remaining'])
        max_cost_r = max(max_cost_r, component_cost_r)
        phase_states += component_info.get('phase_states', 0)
        phase_splits += component_info.get('phase_splits', 0)
        component_backend = component_info.get('phase3_backend')
        if component_backend is None:
            continue
        if component_cost_r > dominant_cost_r:
            phase3_backend = component_backend
            dominant_cost_r = component_cost_r
        elif component_cost_r == dominant_cost_r and phase3_backend not in {None, component_backend}:
            phase3_backend = "mixed"

    return total, {
        'quad': total_quad,
        'constraint': total_constraint,
        'branched': max_branched,
        'remaining': max_remaining,
        'structural_obstruction': max_structural,
        'gauss_obstruction': max_gauss,
        'cost_r': max_cost_r,
        'phase_states': phase_states,
        'phase_splits': phase_splits,
        'phase3_backend': phase3_backend,
    }

def _bruteforce_q3_free_sum(q):
    return sum(
        cmath.exp(2j * cmath.pi * float(q.evaluate([(mask >> bit) & 1 for bit in range(q.n)])))
        for mask in range(2**q.n)
    )

def _q3_free_spanning_data(adjacency, edges):
    """Build a spanning forest and record the non-tree edges."""
    n = len(adjacency)
    depth = [0] * n
    visited = [False] * n
    tree_edges = set()

    for root in range(n):
        if visited[root]:
            continue
        visited[root] = True
        stack = [root]
        while stack:
            node = stack.pop()
            for nbr in sorted(adjacency[node], reverse=True):
                if visited[nbr]:
                    continue
                visited[nbr] = True
                depth[nbr] = depth[node] + 1
                tree_edges.add((min(node, nbr), max(node, nbr)))
                stack.append(nbr)

    chords = [(i, j, phase) for i, j, phase in edges if (i, j) not in tree_edges]
    return depth, chords

def _select_feedback_vertices(n, chords, depth):
    """
    Pick a small set of vertices that covers every non-tree edge.

    Fixing these variables removes every cycle edge, leaving a forest that can
    be summed in linear time by transfer messages.
    """
    if not chords:
        return []

    incident: list[list[int]] = [[] for _ in range(n)]
    for idx, (i, j, _) in enumerate(chords):
        incident[i].append(idx)
        incident[j].append(idx)

    uncovered = [True] * len(chords)
    uncovered_count = len(chords)
    uncovered_incident = [len(edges) for edges in incident]
    chosen: list[int] = []
    heap: list[tuple[int, int, int]] = []
    for var in range(n):
        if uncovered_incident[var]:
            heapq.heappush(heap, (-uncovered_incident[var], -depth[var], var))

    while uncovered_count:
        best = -1
        while heap:
            neg_count, neg_depth, var = heapq.heappop(heap)
            current_count = uncovered_incident[var]
            if current_count <= 0:
                continue
            if neg_count != -current_count or neg_depth != -depth[var]:
                heapq.heappush(heap, (-current_count, -depth[var], var))
                continue
            best = var
            break
        if best < 0:
            raise RuntimeError("Failed to cover q3-free cycle edges.")
        chosen.append(best)
        for edge_idx in incident[best]:
            if not uncovered[edge_idx]:
                continue
            uncovered[edge_idx] = False
            uncovered_count -= 1
            left, right, _phase = chords[edge_idx]
            uncovered_incident[left] -= 1
            uncovered_incident[right] -= 1
            if uncovered_incident[left] > 0:
                heapq.heappush(heap, (-uncovered_incident[left], -depth[left], left))
            if uncovered_incident[right] > 0:
                heapq.heappush(heap, (-uncovered_incident[right], -depth[right], right))
    return sorted(chosen)

def _forest_transfer_sum(q1, adjacency, level: int = 3):
    """
    Evaluate a q2-forest by leaf-to-root transfer.

    Each subtree message is the residue-8 transfer function

        F_v(r) = A_v + omega^r B_v

    compressed into the pair ``(A_v, B_v)``. Querying the parent-imposed
    residue shift r then costs O(1), so a full forest sums in linear time.
    """
    n = len(q1)
    if n == 0:
        return 1.0 + 0j
    if not any(adjacency):
        return _product_q1_sum(q1, level=level)

    omega = _omega_table(level)
    modulus = 1 << level
    total = 1.0 + 0j
    visited = [False] * n
    base = [0j] * n
    excited = [0j] * n

    for root in range(n):
        if visited[root]:
            continue

        postorder = []
        stack = [(root, -1, False)]
        while stack:
            node, parent, expanded = stack.pop()
            if expanded:
                postorder.append((node, parent))
                continue
            if visited[node]:
                continue
            visited[node] = True
            stack.append((node, parent, True))
            for nbr in sorted(adjacency[node], reverse=True):
                if nbr == parent:
                    continue
                if visited[nbr]:
                    raise RuntimeError("Feedback elimination left a cycle in the q3-free forest.")
                stack.append((nbr, node, False))

        for node, parent in postorder:
            off_term = 1.0 + 0j
            on_term = omega[q1[node] % modulus]
            for child, shift in adjacency[node].items():
                if child == parent:
                    continue
                off_term *= base[child] + excited[child]
                on_term *= base[child] + omega[shift % modulus] * excited[child]
            base[node] = off_term
            excited[node] = on_term

        total *= base[root] + excited[root]

    return total

def _forest_postorder_components(adjacency) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
    """Return rooted postorders for each connected component of a forest."""
    visited = [False] * len(adjacency)
    components: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for root in range(len(adjacency)):
        if visited[root]:
            continue

        postorder: list[tuple[int, int]] = []
        stack = [(root, -1, False)]
        while stack:
            node, parent, expanded = stack.pop()
            if expanded:
                postorder.append((node, parent))
                continue
            if visited[node]:
                continue
            visited[node] = True
            stack.append((node, parent, True))
            neighbors = adjacency[node]
            iterable = neighbors.keys() if isinstance(neighbors, dict) else (neighbor for neighbor, _shift in neighbors)
            for nbr in sorted(iterable, reverse=True):
                if nbr == parent:
                    continue
                if visited[nbr]:
                    raise RuntimeError("Feedback elimination left a cycle in the q3-free forest.")
                stack.append((nbr, node, False))
        components.append((root, tuple(postorder)))
    return tuple(components)

def _forest_transfer_sum_scaled(q1, adjacency, level: int = 3):
    """Scaled-complex companion to ``_forest_transfer_sum`` for tiny amplitudes."""
    n = len(q1)
    if n == 0:
        return _ONE_SCALED
    if not any(adjacency):
        return _product_q1_sum_scaled(q1, level=level)

    omega_scaled = _omega_scaled_table(level)
    modulus = 1 << level
    total = _ONE_SCALED
    base = [_ZERO_SCALED] * n
    excited = [_ZERO_SCALED] * n

    for root, postorder in _forest_postorder_components(adjacency):
        for node, parent in postorder:
            off_term = _ONE_SCALED
            on_term = omega_scaled[q1[node] % modulus]
            for child, shift in adjacency[node].items():
                if child == parent:
                    continue
                off_term = _mul_scaled_complex(off_term, _add_scaled_complex(base[child], excited[child]))
                on_term = _mul_scaled_complex(
                    on_term,
                    _add_scaled_complex(
                        base[child],
                        _mul_scaled_complex(
                            omega_scaled[shift % modulus],
                            excited[child],
                        ),
                    ),
                )
            base[node] = off_term
            excited[node] = on_term

        total = _mul_scaled_complex(total, _add_scaled_complex(base[root], excited[root]))

    return total

def _forest_transfer_sum_scaled_batch(
    q1_batch: np.ndarray,
    adjacency,
    *,
    level: int = 3,
) -> list[ScaledComplex]:
    """Batch scaled transfer over one shared q2-forest."""
    if len(q1_batch) == 0:
        return []

    batch = np.asarray(q1_batch, dtype=np.int64)
    n = batch.shape[1]
    if n == 0:
        return [_ONE_SCALED] * len(batch)
    if not any(adjacency):
        return [
            _product_q1_sum_scaled(row.tolist(), level=level)
            for row in batch
        ]

    omega_scaled = _omega_scaled_table(level)
    omega_values, omega_exponents = _scaled_table_to_arrays(omega_scaled)
    modulus = 1 << level
    total_values, total_exponents = _scaled_arrays_from_constant(_ONE_SCALED, (len(batch),))
    base_values, base_exponents = _scaled_arrays_from_constant(_ZERO_SCALED, (len(batch), n))
    excited_values, excited_exponents = _scaled_arrays_from_constant(_ZERO_SCALED, (len(batch), n))

    for root, postorder in _forest_postorder_components(adjacency):
        for node, parent in postorder:
            off_values, off_exponents = _scaled_arrays_from_constant(_ONE_SCALED, (len(batch),))
            residues = np.remainder(batch[:, node], modulus)
            on_values = omega_values[residues]
            on_exponents = omega_exponents[residues]
            for child, shift in adjacency[node].items():
                if child == parent:
                    continue
                child_total_values, child_total_exponents = _add_scaled_complex_arrays(
                    base_values[:, child],
                    base_exponents[:, child],
                    excited_values[:, child],
                    excited_exponents[:, child],
                )
                off_values, off_exponents = _mul_scaled_complex_arrays(
                    off_values,
                    off_exponents,
                    child_total_values,
                    child_total_exponents,
                )

                shifted_excited_values, shifted_excited_exponents = _mul_scaled_complex_arrays(
                    omega_values[np.full(len(batch), shift % modulus, dtype=np.int64)],
                    omega_exponents[np.full(len(batch), shift % modulus, dtype=np.int64)],
                    excited_values[:, child],
                    excited_exponents[:, child],
                )
                on_child_values, on_child_exponents = _add_scaled_complex_arrays(
                    base_values[:, child],
                    base_exponents[:, child],
                    shifted_excited_values,
                    shifted_excited_exponents,
                )
                on_values, on_exponents = _mul_scaled_complex_arrays(
                    on_values,
                    on_exponents,
                    on_child_values,
                    on_child_exponents,
                )
            base_values[:, node] = off_values
            base_exponents[:, node] = off_exponents
            excited_values[:, node] = on_values
            excited_exponents[:, node] = on_exponents

        component_values, component_exponents = _add_scaled_complex_arrays(
            base_values[:, root],
            base_exponents[:, root],
            excited_values[:, root],
            excited_exponents[:, root],
        )
        total_values, total_exponents = _mul_scaled_complex_arrays(
            total_values,
            total_exponents,
            component_values,
            component_exponents,
        )

    return [
        (complex(value), int(half_pow2_exp))
        for value, half_pow2_exp in zip(total_values, total_exponents)
    ]

def _dense_q2_matrix(q):
    """Materialize the q2 coefficients of a q3-free kernel as a symmetric matrix."""
    matrix = np.zeros((q.n, q.n), dtype=np.int64)
    for (i, j), coeff in q.q2.items():
        value = int(coeff % q.mod_q2)
        if not value:
            continue
        matrix[i, j] = value
        matrix[j, i] = value
    return matrix

def _quadratic_residue_threshold(q) -> int:
    """Return the quarter-turn residue threshold used by exact quadratic pivots."""
    return max(1, q.mod_q1 // 4)

def _quadratic_pair_correction(q, left_coeff: int, right_coeff: int) -> int:
    """Return the exact q2-space correction induced by one quadratic pivot.

    For q3-free BL26 kernels, every admissible incident q2 coefficient is either
    ``0`` or ``threshold = mod_q1 // 4`` in q2-residue space. Summing out a
    quadratic pivot therefore introduces an XOR-parity coupling on the pivot's
    neighborhood, which in q2-residue space is represented by ``threshold`` for
    every pair of active neighbors. Expressed generically in the same residue
    space as ``q.q2``, that correction is ``left * right / threshold``.
    """
    threshold = _quadratic_residue_threshold(q)
    if not left_coeff or not right_coeff:
        return 0
    return int((int(left_coeff) * int(right_coeff) // threshold) % q.mod_q2)

def _phase_from_dense_q2(level: int, q1, q2_matrix: np.ndarray, active) -> PhaseFunction:
    """Convert dense q1/q2 data on ``active`` variables back into a q3-free phase."""
    active = tuple(int(idx) for idx in active)
    if not active:
        return _phase_function_from_parts(0, level=level, q0=Fraction(0), q1=[], q2={}, q3={})

    submatrix = q2_matrix[np.ix_(active, active)]
    upper_rows, upper_cols = np.nonzero(np.triu(submatrix, 1))
    q2 = {
        (int(row), int(col)): int(submatrix[row, col])
        for row, col in zip(upper_rows.tolist(), upper_cols.tolist())
        if int(submatrix[row, col])
    }
    return _phase_function_from_parts(
        len(active),
        level=level,
        q0=Fraction(0),
        q1=[int(q1[idx]) for idx in active],
        q2=q2,
        q3={},
    )

def _swap_dense_q2_variables(q1: np.ndarray, q2_matrix: np.ndarray, left: int, right: int) -> None:
    """Swap two variable positions inside the dense q1/q2 representation."""
    if left == right:
        return
    q1[left], q1[right] = q1[right], q1[left]
    q2_matrix[[left, right], :] = q2_matrix[[right, left], :]
    q2_matrix[:, [left, right]] = q2_matrix[:, [right, left]]

def _swap_dense_matrix_variables(matrix: np.ndarray, left: int, right: int) -> None:
    """Swap two variable positions inside a dense square matrix."""
    if left == right:
        return
    matrix[[left, right], :] = matrix[[right, left], :]
    matrix[:, [left, right]] = matrix[:, [right, left]]

def _schur_complement_q3_free_sum_scaled_dense(
    level: int,
    q1,
    q2_matrix: np.ndarray,
    *,
    q0: Fraction = Fraction(0),
    allow_recursive_fallback: bool = True,
    return_residual_on_fallback: bool = False,
):
    """Dense-array implementation of the BL26 q3-free Schur fallback."""
    if len(q1) == 0:
        return _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q0)))

    mod_q1 = 1 << level
    mod_q2 = 1 << max(level - 1, 0)
    threshold = max(1, mod_q1 // 4)
    threshold_shift = threshold.bit_length() - 1 if threshold > 1 else 0
    q1 = np.remainder(np.asarray(q1, dtype=np.int64), mod_q1)
    q2_matrix = np.asarray(q2_matrix, dtype=np.int64).copy()
    adjacency = q2_matrix != 0
    odd_adjacency = (q2_matrix & 1) != 0
    degrees = adjacency.sum(axis=1, dtype=np.int64)
    odd_counts = odd_adjacency.sum(axis=1, dtype=np.int64)
    active_count = len(q1)
    scale_half_pow2 = 0

    while active_count:
        active_degrees = degrees[:active_count]
        coeffs = q1[:active_count]

        if not np.any(active_degrees):
            residual = _product_q1_sum_scaled(coeffs.tolist(), level=level)
            constant = _scale_scaled_complex(
                _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q0))),
                scale_half_pow2,
            )
            return _mul_scaled_complex(constant, residual)

        if threshold > 1:
            divisible = (coeffs & (threshold - 1)) == 0
            reduced = (coeffs >> threshold_shift) & 3
        else:
            divisible = np.ones(active_count, dtype=np.bool_)
            reduced = coeffs & 3
        odd_coupling = odd_counts[:active_count] != 0
        quadratic_mask = divisible & ~odd_coupling & ((reduced == 1) | (reduced == 3))

        zero_mask = divisible & (active_degrees == 0) & (reduced == 2)
        if np.any(zero_mask):
            return _ZERO_SCALED

        decoupled_mask = divisible & (active_degrees == 0) & (reduced == 0)
        if np.any(decoupled_mask):
            pivot_idx = active_count - 1
            if decoupled_mask[pivot_idx]:
                local_idx = pivot_idx
            else:
                local_idx = int(np.flatnonzero(decoupled_mask)[-1])
            _swap_dense_q2_variables(q1, q2_matrix, local_idx, pivot_idx)
            _swap_dense_matrix_variables(adjacency, local_idx, pivot_idx)
            _swap_dense_matrix_variables(odd_adjacency, local_idx, pivot_idx)
            degrees[local_idx], degrees[pivot_idx] = degrees[pivot_idx], degrees[local_idx]
            odd_counts[local_idx], odd_counts[pivot_idx] = odd_counts[pivot_idx], odd_counts[local_idx]
            adjacency[pivot_idx, :active_count] = False
            adjacency[:active_count, pivot_idx] = False
            odd_adjacency[pivot_idx, :active_count] = False
            odd_adjacency[:active_count, pivot_idx] = False
            degrees[pivot_idx] = 0
            odd_counts[pivot_idx] = 0
            active_count -= 1
            scale_half_pow2 += 2
            continue

        if not np.any(quadratic_mask):
            if not allow_recursive_fallback:
                if return_residual_on_fallback:
                    residual_phase = _phase_from_dense_q2(level, q1, q2_matrix, range(active_count))
                    residual_phase.q0 = q0
                    return residual_phase, scale_half_pow2
                return None
            residual_phase = _phase_from_dense_q2(level, q1, q2_matrix, range(active_count))
            residual_total = _sum_q3_free_component_scaled(
                residual_phase,
                allow_schur_complement=False,
            )
            constant = _scale_scaled_complex(
                _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q0))),
                scale_half_pow2,
            )
            return _mul_scaled_complex(constant, residual_total)

        pivot_idx = active_count - 1
        if quadratic_mask[pivot_idx]:
            local_idx = pivot_idx
        else:
            local_idx = int(np.flatnonzero(quadratic_mask)[-1])
        reduced_value = int(reduced[local_idx])
        _swap_dense_q2_variables(q1, q2_matrix, local_idx, pivot_idx)
        _swap_dense_matrix_variables(adjacency, local_idx, pivot_idx)
        _swap_dense_matrix_variables(odd_adjacency, local_idx, pivot_idx)
        degrees[local_idx], degrees[pivot_idx] = degrees[pivot_idx], degrees[local_idx]
        odd_counts[local_idx], odd_counts[pivot_idx] = odd_counts[pivot_idx], odd_counts[local_idx]
        row = q2_matrix[pivot_idx, :pivot_idx].copy()
        nz = np.flatnonzero(row)
        odd_nz = np.flatnonzero(row & 1)

        q0 = (q0 + (Fraction(1, 8) if reduced_value == 1 else Fraction(7, 8))) % 1
        scale_half_pow2 += 1
        if pivot_idx:
            sign = -1 if reduced_value == 1 else 1
            q1[:pivot_idx] = np.remainder(q1[:pivot_idx] + sign * row, mod_q1)
            if nz.size > 1:
                nz_values = row[nz]
                block_index = np.ix_(nz, nz)
                old_adj = adjacency[block_index].copy()
                old_odd = odd_adjacency[block_index].copy()
                correction = np.remainder(
                    np.multiply.outer(nz_values, nz_values) // threshold,
                    mod_q2,
                )
                q2_matrix[block_index] = np.remainder(q2_matrix[block_index] + correction, mod_q2)
                diag = np.arange(nz.size)
                q2_matrix[nz[diag], nz[diag]] = 0
                new_adj = q2_matrix[block_index] != 0
                new_odd = (q2_matrix[block_index] & 1) != 0
                adjacency[block_index] = new_adj
                odd_adjacency[block_index] = new_odd
                degrees[nz] += (
                    new_adj.sum(axis=1, dtype=np.int64) - old_adj.sum(axis=1, dtype=np.int64)
                )
                odd_counts[nz] += (
                    new_odd.sum(axis=1, dtype=np.int64) - old_odd.sum(axis=1, dtype=np.int64)
                )
            if nz.size:
                degrees[nz] -= 1
            if odd_nz.size:
                odd_counts[odd_nz] -= 1
        q2_matrix[pivot_idx, :active_count] = 0
        q2_matrix[:active_count, pivot_idx] = 0
        adjacency[pivot_idx, :active_count] = False
        adjacency[:active_count, pivot_idx] = False
        odd_adjacency[pivot_idx, :active_count] = False
        odd_adjacency[:active_count, pivot_idx] = False
        degrees[pivot_idx] = 0
        odd_counts[pivot_idx] = 0
        active_count -= 1

    return _scale_scaled_complex(
        _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q0))),
        scale_half_pow2,
    )

def _schur_complement_q3_free_sum_scaled(
    q,
    *,
    allow_recursive_fallback: bool = True,
):
    """Dense BL26-style q3-free fallback using vectorized Schur updates.

    This fallback is only exact while it can keep applying the same one-variable
    quadratic Gauss eliminations as ``_elim_quadratic``. The difference is that
    once a dense q2 component has crossed into the regime where graph-based
    feedback branching is unattractive, the coefficient updates are carried out
    on dense NumPy arrays rather than through repeated Python dict surgery.

    If the dense Schur pass reaches a residual kernel that no longer exposes a
    valid quadratic pivot, the remaining exact work is handed back to the
    existing q3-free solver.
    """
    assert not q.q3, "BL26 dense fallback only applies to q3-free kernels."
    if not _supports_exact_dense_schur(q):
        return None
    if q.n == 0:
        return _ONE_SCALED
    if not q.q2:
        return _mul_scaled_complex(
                _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q.q0))),
                _product_q1_sum_scaled(q.q1, level=q.level),
            )

    return _schur_complement_q3_free_sum_scaled_dense(
        q.level,
        q.q1,
        _dense_q2_matrix(q),
        q0=q.q0,
        allow_recursive_fallback=allow_recursive_fallback,
    )

def _schur_complement_q3_free_sum(
    q,
    *,
    allow_recursive_fallback: bool = True,
):
    """Unscaled wrapper for the dense BL26 q3-free fallback."""
    scaled = _schur_complement_q3_free_sum_scaled(
        q,
        allow_recursive_fallback=allow_recursive_fallback,
    )
    if scaled is None:
        return None
    return _scaled_to_complex(scaled)

def _qubit_quadratic_tensor_obstruction_support(q) -> tuple[int, ...]:
    """Return variables that keep ``q`` outside BL26's qubit quadratic class.

    For qubit quadratic tensors, BL26's coefficient groups permit only q1
    coefficients from a 4-element class and q2 coefficients from a 2-element
    class. In TerKet's dyadic integer encoding at level ``L``, both conditions
    reduce to requiring every q1/q2 coefficient to be a multiple of
    ``2^(L-2)``. Any surviving coefficient outside that class requires a
    higher-order tensor description even when ``q.q3`` is empty.
    """
    threshold = max(1, q.mod_q1 // 4)
    if threshold <= 1:
        return ()

    support: set[int] = set()
    for idx, coeff in enumerate(q.q1):
        if coeff % threshold:
            support.add(idx)
    for (left, right), coeff in q.q2.items():
        if coeff % threshold:
            support.add(left)
            support.add(right)
    return tuple(sorted(support))

def _qubit_quadratic_tensor_obstruction(q) -> int:
    """Return the size of the residual non-quadratic qubit support of ``q``."""
    return len(_qubit_quadratic_tensor_obstruction_support(q))

def _supports_exact_dense_schur(q) -> bool:
    """Return whether the dense Schur q3-free backend is exact for ``q``.

    The dense backend is exact on q3-free kernels whose q2 support lies
    entirely in the half-phase class. Unary coefficients may be arbitrary; the
    dense eliminator only pivots on quarter-turn q1 residues and hands any
    remaining hard-unary work back to the generic exact q3-free solver.
    """
    return int(q.level) >= 3 and _is_half_phase_q2(q)

def _sum_bl26_quadratic_tensor_component_scaled(q):
    """Exactly contract one BL26 qubit quadratic-tensor component."""
    assert not q.q3, "Quadratic-tensor contraction requires a q3-free kernel."
    if _qubit_quadratic_tensor_obstruction(q):
        raise ValueError("Quadratic-tensor contraction requires zero gauss obstruction.")

    if q.n == 0:
        return _ONE_SCALED
    if not q.q2:
        return _product_q1_sum_scaled(q.q1, level=q.level)

    binary_total = _sum_binary_phase_quadratic_scaled(q)
    if binary_total is not None:
        return binary_total

    dense_result = _schur_complement_q3_free_sum_scaled_dense(
        q.level,
        q.q1,
        _dense_q2_matrix(q),
        q0=q.q0,
        allow_recursive_fallback=False,
        return_residual_on_fallback=True,
    )
    if isinstance(dense_result[0], complex):
        return dense_result

    residual_phase, scale_half_pow2 = dense_result
    if _qubit_quadratic_tensor_obstruction(residual_phase):
        raise RuntimeError(
            "BL26 quadratic-tensor contraction failed on a zero-obstruction q3-free kernel."
        )
    residual_total = _sum_bl26_quadratic_tensor_component_scaled(residual_phase)
    return _scale_scaled_complex(residual_total, scale_half_pow2)

def _sum_bl26_quadratic_tensor_component(q):
    """Unscaled wrapper for ``_sum_bl26_quadratic_tensor_component_scaled``."""
    return _scaled_to_complex(_sum_bl26_quadratic_tensor_component_scaled(q))

def _sum_q3_free_via_gauss_reduction_scaled(q):
    """Try exact q3-free backends that explicitly target gauss obstruction."""
    if q.q3:
        return None
    if _qubit_quadratic_tensor_obstruction(q) == 0:
        return _sum_bl26_quadratic_tensor_component_scaled(q)

    half_phase_expansion_total = _sum_half_phase_q2_unary_expansion_scaled(q)
    if half_phase_expansion_total is not None:
        return half_phase_expansion_total

    mediator_plan = _build_half_phase_mediator_plan(q)
    if mediator_plan is not None:
        return _evaluate_half_phase_mediator_plan_scaled(
            mediator_plan,
            q.q1,
        )

    cluster_plan = _build_q1_cluster_plan(q)
    if cluster_plan is not None:
        return _evaluate_half_phase_cluster_plan_scaled(
            cluster_plan,
            q.q1,
        )

    one_shot_cutset_total = _sum_q3_free_via_one_shot_cutset_scaled(q)
    if one_shot_cutset_total is not None:
        return one_shot_cutset_total

    generic_mediator_plan = _build_generic_q2_mediator_plan(q)
    if generic_mediator_plan is not None:
        return _evaluate_generic_q2_mediator_plan_scaled(
            generic_mediator_plan,
            q.q1,
        )

    bad_q2_cover = _minimum_bad_q2_vertex_cover(q)
    if _bad_q2_cover_dispatch_allowed(q, bad_q2_cover):
        bad_q2_cover_total = _sum_q3_free_via_bad_q2_cover_scaled(q, cover=bad_q2_cover)
        if bad_q2_cover_total is not None:
            return bad_q2_cover_total

    parity_reduced_total = _sum_half_phase_parity_component_reduction_scaled(q)
    if parity_reduced_total is not None:
        return parity_reduced_total

    return _sum_q3_free_via_nonquadratic_support_scaled(q)

def _sum_q3_free_via_gauss_reduction(q):
    """Unscaled wrapper for ``_sum_q3_free_via_gauss_reduction_scaled``."""
    scaled = _sum_q3_free_via_gauss_reduction_scaled(q)
    if scaled is None:
        return None
    return _scaled_to_complex(scaled)

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
