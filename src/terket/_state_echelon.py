"""State output-constraint and echelon-solver helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ._state_config import BitSequence


def _iter_mask_bits(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _mask_from_vector(bits: Sequence[int]) -> int:
    mask = 0
    for idx, bit in enumerate(bits):
        if int(bit) & 1:
            mask |= 1 << idx
    return mask


def _mask_bit(mask: int, idx: int) -> int:
    return (mask >> idx) & 1


def _parity(mask: int) -> int:
    return mask.bit_count() & 1


def _row_reduce_output_constraints(n_rows: int, rows: list[int]) -> tuple[list[int], list[int], list[int], int]:
    """Return reduced output rows plus row-operation witnesses."""

    row_ops = [1 << idx for idx in range(n_rows)]
    pivot_col = [-1] * n_rows
    used_mask = 0

    for row_idx in range(n_rows):
        available = rows[row_idx] & ~used_mask
        if not available:
            continue
        pivot_bit = available & -available
        pivot = pivot_bit.bit_length() - 1
        pivot_col[row_idx] = pivot
        used_mask |= pivot_bit
        for other_idx in range(n_rows):
            if other_idx != row_idx and (rows[other_idx] & pivot_bit):
                rows[other_idx] ^= rows[row_idx]
                row_ops[other_idx] ^= row_ops[row_idx]

    return rows, row_ops, pivot_col, used_mask


@dataclass(frozen=True, slots=True)
class EchelonCache:
    """Reusable row-echelon form of output constraint matrix."""

    n: int
    m: int
    echelon_rows: tuple[int, ...]
    pivot_col: tuple[int, ...]
    used_mask: int
    row_ops: tuple[int, ...]
    free_vars: tuple[int, ...]
    gamma_masks: tuple[int, ...]
    n_free: int


def _prepare_affine_constraint_cache(n_constraints: int, n_vars: int, row_masks: Sequence[int]) -> EchelonCache:
    rows = [int(mask) for mask in row_masks]
    rows, row_ops, pivot_col, used_mask = _row_reduce_output_constraints(n_constraints, rows)

    free = tuple(var for var in range(n_vars) if not (used_mask >> var) & 1)
    n_free = len(free)
    gamma = [0] * n_vars
    for free_idx, free_var in enumerate(free):
        gamma[free_var] = 1 << free_idx
        for row_idx, pivot in enumerate(pivot_col):
            if pivot >= 0 and (rows[row_idx] >> free_var) & 1:
                gamma[pivot] ^= 1 << free_idx

    return EchelonCache(
        n=n_constraints,
        m=n_vars,
        echelon_rows=tuple(rows),
        pivot_col=tuple(pivot_col),
        used_mask=used_mask,
        row_ops=tuple(row_ops),
        free_vars=free,
        gamma_masks=tuple(gamma),
        n_free=n_free,
    )


def _solve_echelon_rhs(cache: EchelonCache, rhs_mask: int) -> int | None:
    shift_mask = 0
    for row_idx, pivot in enumerate(cache.pivot_col):
        rhs = _parity(rhs_mask & cache.row_ops[row_idx])
        if pivot < 0:
            if rhs:
                return None
            continue
        if rhs:
            shift_mask |= 1 << pivot
    return shift_mask


def _solve_output_from_echelon(
    eps0: Sequence[int],
    cache: EchelonCache,
    output_bits: BitSequence,
    *,
    native_solver: Callable[[Sequence[int], EchelonCache, BitSequence], int | None] | None = None,
) -> tuple[int, tuple[int, ...], tuple[int, ...], int] | None:
    if len(output_bits) != cache.n:
        raise ValueError(f"Expected {cache.n} output bits, received {len(output_bits)}.")

    if native_solver is not None:
        native_shift_mask = native_solver(eps0, cache, output_bits)
        if native_shift_mask is not None:
            return native_shift_mask, cache.free_vars, cache.gamma_masks, cache.n_free

    target_mask = 0
    for idx, bit in enumerate(output_bits):
        if (int(bit) ^ int(eps0[idx])) & 1:
            target_mask |= 1 << idx

    shift_mask = _solve_echelon_rhs(cache, target_mask)
    if shift_mask is None:
        return None
    return shift_mask, cache.free_vars, cache.gamma_masks, cache.n_free


__all__ = [
    "EchelonCache",
    "_iter_mask_bits",
    "_mask_bit",
    "_mask_from_vector",
    "_parity",
    "_prepare_affine_constraint_cache",
    "_row_reduce_output_constraints",
    "_solve_echelon_rhs",
    "_solve_output_from_echelon",
]
