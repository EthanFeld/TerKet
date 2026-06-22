"""Native batch bridge for q3-free residue-forest samples."""

from __future__ import annotations

import cmath

import numpy as np

from ..scaling import (
    ScaledComplex,
    _ZERO_SCALED,
    _add_scaled_complex,
    _make_scaled_complex,
    _mul_scaled_complex,
    _normalize_scaled_complex,
    _omega_scaled_table,
    _scale_scaled_complex,
)


def _forest_leaf_order(adjacency: list[dict[int, int]]) -> list[int]:
    degrees = [len(neighbors) for neighbors in adjacency]
    active = [True] * len(adjacency)
    pending = [idx for idx, degree in enumerate(degrees) if degree <= 1]
    order: list[int] = []
    while pending:
        node = pending.pop()
        if not active[node]:
            continue
        active[node] = False
        order.append(node)
        for neighbor in adjacency[node]:
            if active[neighbor]:
                degrees[neighbor] -= 1
                if degrees[neighbor] <= 1:
                    pending.append(neighbor)
    order.extend(idx for idx, is_active in enumerate(active) if is_active)
    return order


def _forest_parent_data(adjacency: list[dict[int, int]]) -> tuple[list[int], list[int], list[int]]:
    parent = [-2] * len(adjacency)
    parent_phase = [0] * len(adjacency)
    postorder: list[int] = []
    for root in range(len(adjacency)):
        if parent[root] != -2:
            continue
        parent[root] = -1
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                postorder.append(node)
                continue
            stack.append((node, True))
            for neighbor, phase in adjacency[node].items():
                if neighbor == parent[node] or parent[neighbor] != -2:
                    continue
                parent[neighbor] = node
                parent_phase[neighbor] = int(phase)
                stack.append((neighbor, False))
    return parent, parent_phase, postorder


def _sum_arbitrary_residue_forest_native_batch(
    q1_batch: np.ndarray,
    adjacency: list[dict[int, int]],
    *,
    level: int,
) -> list[ScaledComplex] | None:
    try:
        from ..native import _schur_native
    except Exception:
        return None
    if _schur_native is None or not hasattr(_schur_native, "sum_residue_forest_batch_scaled_array"):
        return None
    parent, parent_phase, postorder = _forest_parent_data(adjacency)
    try:
        rows = _schur_native.sum_residue_forest_batch_scaled_array(
            int(level),
            np.ascontiguousarray(q1_batch, dtype=np.int64),
            parent,
            parent_phase,
            postorder,
        )
    except Exception:
        return None
    return [(complex(value), int(half_pow2_exp)) for value, half_pow2_exp, _scope in rows]


def _native_free_q2(free_adjacency: list[dict[int, int]], target_level: int) -> dict[tuple[int, int], int] | None:
    q2_lift = 2 if int(target_level) > 1 else 1
    q2: dict[tuple[int, int], int] = {}
    for left, neighbors in enumerate(free_adjacency):
        for right, phase in neighbors.items():
            if left >= right:
                continue
            if int(phase) % q2_lift:
                return None
            q2[(left, right)] = int(phase) // q2_lift
    return q2


def _q1_batch_and_constants(
    *,
    target_level: int,
    fixed_bit_rows: np.ndarray,
    base_q1: list[int],
    fixed_linear: list[tuple[int, int]],
    fixed_to_free: list[tuple[int, int, int]],
    fixed_to_fixed: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(fixed_bit_rows, dtype=np.int64)
    modulus = 1 << int(target_level)
    q1_batch = np.tile(np.asarray(base_q1, dtype=np.int64), (int(len(rows)), 1))
    const_residues = np.zeros(int(len(rows)), dtype=np.int64)
    for bit_idx, residue in fixed_linear:
        const_residues = (const_residues + rows[:, bit_idx] * int(residue)) % modulus
    for bit_idx, free_var, residue in fixed_to_free:
        q1_batch[:, free_var] = (
            q1_batch[:, free_var] + rows[:, bit_idx] * int(residue)
        ) % modulus
    for left_bit, right_bit, residue in fixed_to_fixed:
        active = rows[:, left_bit] & rows[:, right_bit]
        const_residues = (const_residues + active * int(residue)) % modulus
    return q1_batch, const_residues


def _stable_unique_q1_batch(q1_batch: np.ndarray, *, level: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Deduplicate q1 rows only when a cheap fingerprint detects collisions."""
    if len(q1_batch) < 2:
        return q1_batch, None
    columns = np.arange(1, q1_batch.shape[1] + 1, dtype=np.uint64)
    weights = columns * np.uint64(0x9E3779B97F4A7C15)
    fingerprints = np.bitwise_xor.reduce(q1_batch.view(np.uint64) * weights, axis=1)
    if len(np.unique(fingerprints)) == len(q1_batch):
        return q1_batch, None

    compact_dtype = (
        np.uint8 if level <= 8 else np.uint16 if level <= 16 else np.uint32 if level <= 32 else np.uint64
    )
    compact = q1_batch.astype(compact_dtype, copy=False)
    unique, first, inverse = np.unique(
        compact,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    if len(unique) == len(q1_batch):
        return q1_batch, None
    stable_order = np.argsort(first)
    sorted_to_stable = np.empty(len(stable_order), dtype=np.int64)
    sorted_to_stable[stable_order] = np.arange(len(stable_order), dtype=np.int64)
    return unique[stable_order].astype(np.int64), sorted_to_stable[inverse]


def _sum_native_rows(
    free_totals: list[ScaledComplex],
    const_residues: np.ndarray,
    *,
    target_level: int,
    q1_inverse: np.ndarray | None = None,
) -> ScaledComplex:
    modulus = 1 << int(target_level)
    omega_scaled = _omega_scaled_table(target_level)
    total = _ZERO_SCALED
    if q1_inverse is None:
        q1_inverse = np.arange(len(free_totals), dtype=np.int64)
    for const_residue, free_idx in zip(const_residues.tolist(), q1_inverse.tolist()):
        free_total = free_totals[int(free_idx)]
        total = _add_scaled_complex(
            total,
            _mul_scaled_complex(omega_scaled[int(const_residue) % modulus], free_total),
        )
    return _normalize_scaled_complex(total[0] / float(len(const_residues)), total[1])


def _sum_q3_free_residue_forest_native_batch_scaled(
    q,
    *,
    target_level: int,
    feedback_count: int,
    fixed_bit_rows: np.ndarray,
    base_q1: list[int],
    free_adjacency: list[dict[int, int]],
    fixed_linear: list[tuple[int, int]],
    fixed_to_free: list[tuple[int, int, int]],
    fixed_to_fixed: list[tuple[int, int, int]],
) -> ScaledComplex | None:
    if not base_q1:
        return None
    q1_batch, const_residues = _q1_batch_and_constants(
        target_level=target_level,
        fixed_bit_rows=fixed_bit_rows,
        base_q1=base_q1,
        fixed_linear=fixed_linear,
        fixed_to_free=fixed_to_free,
        fixed_to_fixed=fixed_to_fixed,
    )
    unique_q1, q1_inverse = _stable_unique_q1_batch(q1_batch, level=int(target_level))
    free_totals = _sum_arbitrary_residue_forest_native_batch(
        unique_q1,
        free_adjacency,
        level=int(target_level),
    )
    if free_totals is None:
        try:
            from .native import _sum_q3_free_treewidth_dp_scaled_batch
        except Exception:
            return None
        q2 = _native_free_q2(free_adjacency, target_level)
        if q2 is None:
            return None
        try:
            free_totals = _sum_q3_free_treewidth_dp_scaled_batch(
                n_vars=len(base_q1),
                level=int(target_level),
                q1_batch=unique_q1,
                q2=q2,
                order=_forest_leaf_order(free_adjacency),
            )
        except Exception:
            return None
    mean = _sum_native_rows(
        free_totals,
        const_residues,
        target_level=target_level,
        q1_inverse=q1_inverse,
    )
    scalar = _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q.q0)))
    return _mul_scaled_complex(scalar, _scale_scaled_complex(mean, 2 * int(feedback_count)))
