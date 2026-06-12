"""Owned q3-free reusable batch dataclasses and compact branch helpers.

Owns:
- compact storage helpers for reusable q3-free plans
- cover-conditioned branch-template construction
- exact batched branch evaluation for q3-cover execution

Key invariants:
- residue arrays stay packed to smallest unsigned dtype that can hold values
- q3-free reusable batching lives here even when Phase-3 calls into it
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..scaling import _omega_table


_Q3_COVER_BRANCH_CHUNK_MAX = 128
_Q3_COVER_ASSIGNMENT_CHUNK_LOG2 = 13


def _as_int64_array(values) -> np.ndarray:
    if not values:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(values, dtype=np.int64)


def _compact_unsigned_storage_dtype(max_value: int):
    max_value = max(0, int(max_value))
    if max_value <= np.iinfo(np.uint8).max:
        return np.uint8
    if max_value <= np.iinfo(np.uint16).max:
        return np.uint16
    if max_value <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.int64


def _compact_index_storage_array(values, *, upper_bound: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if upper_bound is None:
        max_value = int(array.max(initial=0)) if array.size else 0
    else:
        max_value = max(0, int(upper_bound) - 1)
    return np.asarray(array, dtype=_compact_unsigned_storage_dtype(max_value))


def _compact_residue_storage_array(values, *, modulus: int) -> np.ndarray:
    array = np.asarray(values)
    return np.asarray(array, dtype=_compact_unsigned_storage_dtype(int(modulus) - 1))


def _phase_fraction_to_residue(value: Fraction, modulus: int) -> int:
    scaled = Fraction(value) * modulus
    if scaled.denominator != 1:
        raise ValueError(f"Phase constant {value!r} is not representable modulo {modulus}.")
    return int(scaled.numerator % modulus)


@dataclass(frozen=True, slots=True)
class Q3FreeBranchTemplate:
    """Shared residue updates for exact q3-cover branch batching."""

    cover_vars: tuple[int, ...]
    remaining_vars: tuple[int, ...]
    n_cover: int
    n_remaining: int
    mod_q1: int
    level: int
    base_q0_residue: int
    base_q1_residue: np.ndarray
    pair_left: np.ndarray
    pair_right: np.ndarray
    base_q2_residue: np.ndarray
    cover_q1_residue: np.ndarray
    cover_remaining_q2_residue: np.ndarray
    cover_cover_left: np.ndarray
    cover_cover_right: np.ndarray
    cover_cover_residue: np.ndarray
    cubic_pair_cover: np.ndarray
    cubic_pair_index: np.ndarray
    cubic_pair_residue: np.ndarray
    cubic_linear_cover_left: np.ndarray
    cubic_linear_cover_right: np.ndarray
    cubic_linear_var: np.ndarray
    cubic_linear_residue: np.ndarray
    cubic_constant_left: np.ndarray
    cubic_constant_middle: np.ndarray
    cubic_constant_right: np.ndarray
    cubic_constant_residue: np.ndarray


def _build_q3_free_branch_template(q, cover) -> Q3FreeBranchTemplate:
    """Precompute cover-conditioned residue updates for exact branch batching."""

    cover_vars = tuple(int(var) for var in cover)
    cover_map = {var: idx for idx, var in enumerate(cover_vars)}
    remaining_vars = tuple(var for var in range(q.n) if var not in cover_map)
    remaining_map = {var: idx for idx, var in enumerate(remaining_vars)}
    mod_q1 = q.mod_q1

    pair_keys: list[tuple[int, int]] = []
    pair_map: dict[tuple[int, int], int] = {}

    def ensure_pair(left_var: int, right_var: int) -> int:
        key = tuple(sorted((remaining_map[left_var], remaining_map[right_var])))
        existing = pair_map.get(key)
        if existing is not None:
            return existing
        idx = len(pair_keys)
        pair_keys.append(key)
        pair_map[key] = idx
        return idx

    for (left, right), coeff in q.q2.items():
        if left in remaining_map and right in remaining_map and coeff % q.mod_q2:
            ensure_pair(left, right)
    for triple, coeff in q.q3.items():
        if not coeff % q.mod_q3:
            continue
        remaining = tuple(var for var in triple if var in remaining_map)
        if len(remaining) == 2:
            ensure_pair(remaining[0], remaining[1])

    pair_left = _compact_index_storage_array([left for left, _ in pair_keys], upper_bound=len(remaining_vars))
    pair_right = _compact_index_storage_array([right for _, right in pair_keys], upper_bound=len(remaining_vars))
    base_q2_residue = np.zeros(len(pair_keys), dtype=np.int64)
    q2_lift = q.mod_q1 // q.mod_q2 if q.mod_q2 else 0
    q3_lift = q.mod_q1 // q.mod_q3 if q.mod_q3 else 0

    for (left, right), coeff in q.q2.items():
        if left in remaining_map and right in remaining_map:
            pair_idx = ensure_pair(left, right)
            base_q2_residue[pair_idx] = (base_q2_residue[pair_idx] + q2_lift * coeff) % mod_q1

    cover_q1_residue = np.array([q.q1[var] % mod_q1 for var in cover_vars], dtype=np.int64)
    base_q1_residue = np.array([q.q1[var] % mod_q1 for var in remaining_vars], dtype=np.int64)
    cover_remaining_q2_residue = np.zeros((len(cover_vars), len(remaining_vars)), dtype=np.int64)
    cover_cover_left: list[int] = []
    cover_cover_right: list[int] = []
    cover_cover_residue: list[int] = []

    for (left, right), coeff in q.q2.items():
        residue = (q2_lift * coeff) % mod_q1
        if not residue:
            continue
        if left in cover_map and right in remaining_map:
            cover_remaining_q2_residue[cover_map[left], remaining_map[right]] = (
                cover_remaining_q2_residue[cover_map[left], remaining_map[right]] + residue
            ) % mod_q1
        elif right in cover_map and left in remaining_map:
            cover_remaining_q2_residue[cover_map[right], remaining_map[left]] = (
                cover_remaining_q2_residue[cover_map[right], remaining_map[left]] + residue
            ) % mod_q1
        elif left in cover_map and right in cover_map:
            cover_cover_left.append(cover_map[left])
            cover_cover_right.append(cover_map[right])
            cover_cover_residue.append(residue)

    cubic_pair_cover: list[int] = []
    cubic_pair_index: list[int] = []
    cubic_pair_residue: list[int] = []
    cubic_linear_cover_left: list[int] = []
    cubic_linear_cover_right: list[int] = []
    cubic_linear_var: list[int] = []
    cubic_linear_residue: list[int] = []
    cubic_constant_left: list[int] = []
    cubic_constant_middle: list[int] = []
    cubic_constant_right: list[int] = []
    cubic_constant_residue: list[int] = []

    for triple, coeff in q.q3.items():
        residue = (q3_lift * coeff) % mod_q1
        if not residue:
            continue
        cover_entries = [cover_map[var] for var in triple if var in cover_map]
        remaining_entries = [remaining_map[var] for var in triple if var in remaining_map]
        if len(cover_entries) == 1 and len(remaining_entries) == 2:
            pair_idx = pair_map[tuple(sorted(remaining_entries))]
            cubic_pair_cover.append(cover_entries[0])
            cubic_pair_index.append(pair_idx)
            cubic_pair_residue.append(residue)
        elif len(cover_entries) == 2 and len(remaining_entries) == 1:
            cubic_linear_cover_left.append(cover_entries[0])
            cubic_linear_cover_right.append(cover_entries[1])
            cubic_linear_var.append(remaining_entries[0])
            cubic_linear_residue.append(residue)
        elif len(cover_entries) == 3:
            cubic_constant_left.append(cover_entries[0])
            cubic_constant_middle.append(cover_entries[1])
            cubic_constant_right.append(cover_entries[2])
            cubic_constant_residue.append(residue)
        else:
            raise ValueError("q3-cover template encountered an unfixed cubic term.")

    return Q3FreeBranchTemplate(
        cover_vars=cover_vars,
        remaining_vars=remaining_vars,
        n_cover=len(cover_vars),
        n_remaining=len(remaining_vars),
        mod_q1=mod_q1,
        level=q.level,
        base_q0_residue=_phase_fraction_to_residue(q.q0, mod_q1),
        base_q1_residue=_compact_residue_storage_array(base_q1_residue, modulus=mod_q1),
        pair_left=pair_left,
        pair_right=pair_right,
        base_q2_residue=_compact_residue_storage_array(base_q2_residue, modulus=mod_q1),
        cover_q1_residue=_compact_residue_storage_array(cover_q1_residue, modulus=mod_q1),
        cover_remaining_q2_residue=_compact_residue_storage_array(cover_remaining_q2_residue, modulus=mod_q1),
        cover_cover_left=_compact_index_storage_array(cover_cover_left, upper_bound=len(cover_vars)),
        cover_cover_right=_compact_index_storage_array(cover_cover_right, upper_bound=len(cover_vars)),
        cover_cover_residue=_compact_residue_storage_array(cover_cover_residue, modulus=mod_q1),
        cubic_pair_cover=_compact_index_storage_array(cubic_pair_cover, upper_bound=len(cover_vars)),
        cubic_pair_index=_compact_index_storage_array(cubic_pair_index, upper_bound=len(pair_keys)),
        cubic_pair_residue=_compact_residue_storage_array(cubic_pair_residue, modulus=mod_q1),
        cubic_linear_cover_left=_compact_index_storage_array(cubic_linear_cover_left, upper_bound=len(cover_vars)),
        cubic_linear_cover_right=_compact_index_storage_array(cubic_linear_cover_right, upper_bound=len(cover_vars)),
        cubic_linear_var=_compact_index_storage_array(cubic_linear_var, upper_bound=len(remaining_vars)),
        cubic_linear_residue=_compact_residue_storage_array(cubic_linear_residue, modulus=mod_q1),
        cubic_constant_left=_compact_index_storage_array(cubic_constant_left, upper_bound=len(cover_vars)),
        cubic_constant_middle=_compact_index_storage_array(cubic_constant_middle, upper_bound=len(cover_vars)),
        cubic_constant_right=_compact_index_storage_array(cubic_constant_right, upper_bound=len(cover_vars)),
        cubic_constant_residue=_compact_residue_storage_array(cubic_constant_residue, modulus=mod_q1),
    )


def _branch_assignment_bits(branch_masks: np.ndarray, n_cover: int):
    if n_cover == 0:
        return np.zeros((len(branch_masks), 0), dtype=np.int64)
    masks = np.asarray(branch_masks, dtype=np.uint64).reshape(-1, 1)
    shifts = np.arange(n_cover, dtype=np.uint64).reshape(1, -1)
    return ((masks >> shifts) & 1).astype(np.int64)


def _q3_cover_branch_chunk_size(template: Q3FreeBranchTemplate, budget_bytes: int | None) -> tuple[int, int]:
    assignment_chunk = 1 << min(template.n_remaining, _Q3_COVER_ASSIGNMENT_CHUNK_LOG2)
    branch_chunk = min(_Q3_COVER_BRANCH_CHUNK_MAX, 1 << template.n_cover)
    if budget_bytes is None:
        return branch_chunk, assignment_chunk

    while branch_chunk > 1:
        estimated = assignment_chunk * branch_chunk * 32
        if estimated <= budget_bytes:
            break
        branch_chunk //= 2
    return max(1, branch_chunk), assignment_chunk


def _evaluate_q3_free_branch_template_batch(
    template: Q3FreeBranchTemplate,
    branch_masks: np.ndarray,
    *,
    assignment_chunk_size: int | None = None,
) -> np.ndarray:
    if branch_masks.size == 0:
        return np.zeros(0, dtype=np.complex128)

    if assignment_chunk_size is None:
        assignment_chunk_size = 1 << min(template.n_remaining, _Q3_COVER_ASSIGNMENT_CHUNK_LOG2)

    branch_bits = _branch_assignment_bits(branch_masks, template.n_cover)
    mod_q1 = int(template.mod_q1)
    omega = np.asarray(_omega_table(template.level), dtype=np.complex128)

    q0_eff = np.full(branch_bits.shape[0], int(template.base_q0_residue), dtype=np.int64)
    q1_eff = np.broadcast_to(
        np.asarray(template.base_q1_residue, dtype=np.int64),
        (branch_bits.shape[0], template.n_remaining),
    ).copy()
    q2_eff = np.broadcast_to(
        np.asarray(template.base_q2_residue, dtype=np.int64),
        (branch_bits.shape[0], template.pair_left.size),
    ).copy()

    if template.cover_q1_residue.size:
        q0_eff = (q0_eff + branch_bits @ np.asarray(template.cover_q1_residue, dtype=np.int64)) % mod_q1
    if template.cover_remaining_q2_residue.size:
        q1_eff = (q1_eff + branch_bits @ np.asarray(template.cover_remaining_q2_residue, dtype=np.int64)) % mod_q1
    if template.cover_cover_residue.size:
        for left, right, residue in zip(
            template.cover_cover_left,
            template.cover_cover_right,
            template.cover_cover_residue,
        ):
            q0_eff = (q0_eff + int(residue) * branch_bits[:, int(left)] * branch_bits[:, int(right)]) % mod_q1
    if template.cubic_pair_residue.size:
        for cover_idx, pair_idx, residue in zip(
            template.cubic_pair_cover,
            template.cubic_pair_index,
            template.cubic_pair_residue,
        ):
            q2_eff[:, int(pair_idx)] = (
                q2_eff[:, int(pair_idx)] + int(residue) * branch_bits[:, int(cover_idx)]
            ) % mod_q1
    if template.cubic_linear_residue.size:
        for left, right, var_idx, residue in zip(
            template.cubic_linear_cover_left,
            template.cubic_linear_cover_right,
            template.cubic_linear_var,
            template.cubic_linear_residue,
        ):
            q1_eff[:, int(var_idx)] = (
                q1_eff[:, int(var_idx)]
                + int(residue) * branch_bits[:, int(left)] * branch_bits[:, int(right)]
            ) % mod_q1
    if template.cubic_constant_residue.size:
        for left, middle, right, residue in zip(
            template.cubic_constant_left,
            template.cubic_constant_middle,
            template.cubic_constant_right,
            template.cubic_constant_residue,
        ):
            q0_eff = (
                q0_eff
                + int(residue)
                * branch_bits[:, int(left)]
                * branch_bits[:, int(middle)]
                * branch_bits[:, int(right)]
            ) % mod_q1

    totals = np.zeros(branch_bits.shape[0], dtype=np.complex128)
    n_assignments = 1 << template.n_remaining

    for start in range(0, n_assignments, assignment_chunk_size):
        stop = min(n_assignments, start + assignment_chunk_size)
        assignments = np.arange(start, stop, dtype=np.uint64).reshape(-1, 1)
        if template.n_remaining:
            shifts = np.arange(template.n_remaining, dtype=np.uint64).reshape(1, -1)
            x = ((assignments >> shifts) & 1).astype(np.int64)
            residues = x @ q1_eff.T
            if template.pair_left.size:
                pair_terms = x[:, np.asarray(template.pair_left, dtype=np.int64)] * x[
                    :, np.asarray(template.pair_right, dtype=np.int64)
                ]
                residues = residues + pair_terms @ q2_eff.T
        else:
            residues = np.zeros((stop - start, branch_bits.shape[0]), dtype=np.int64)
        residues = (residues + q0_eff.reshape(1, -1)) % mod_q1
        totals = totals + omega[residues].sum(axis=0)

    return np.asarray(totals, dtype=np.complex128)


__all__ = [
    "Q3FreeBranchTemplate",
    "_Q3_COVER_ASSIGNMENT_CHUNK_LOG2",
    "_Q3_COVER_BRANCH_CHUNK_MAX",
    "_as_int64_array",
    "_branch_assignment_bits",
    "_build_q3_free_branch_template",
    "_compact_index_storage_array",
    "_compact_residue_storage_array",
    "_compact_unsigned_storage_dtype",
    "_evaluate_q3_free_branch_template_batch",
    "_phase_fraction_to_residue",
    "_q3_cover_branch_chunk_size",
]
