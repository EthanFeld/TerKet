"""Feedback-row samplers for approximate q3-free summation."""

from __future__ import annotations

import numpy as np


def _reduce_gf2_path_basis(generator: np.ndarray) -> np.ndarray:
    """Reduce generator-row weights using invertible GF(2) row operations."""
    reduced = generator.copy()
    changed = True
    while changed:
        changed = False
        weights = reduced.sum(axis=1)
        for row_idx in range(len(reduced)):
            best = reduced[row_idx]
            best_weight = int(weights[row_idx])
            for other_idx in range(len(reduced)):
                if row_idx == other_idx:
                    continue
                candidate = reduced[row_idx] ^ reduced[other_idx]
                candidate_weight = int(candidate.sum())
                if candidate_weight < best_weight:
                    best = candidate
                    best_weight = candidate_weight
            if best_weight < int(weights[row_idx]):
                reduced[row_idx] = best
                changed = True
    return reduced


def _unified_path_sample_rows(
    feedback_count: int,
    sample_budget: int,
    rng: np.random.Generator,
    *,
    priority_columns: np.ndarray | None = None,
) -> np.ndarray:
    """Generate distinct randomized affine Gray paths through feedback space."""
    max_rows = 1 << int(feedback_count) if int(feedback_count) < 63 else int(sample_budget)
    row_count = min(max(1, int(sample_budget)), max_rows)
    if row_count == 1:
        return rng.integers(0, 2, size=(1, feedback_count), dtype=np.uint8)

    path_bits = (row_count - 1).bit_length()
    indices = np.arange(row_count, dtype=np.uint64)
    gray = indices ^ (indices >> np.uint64(1))
    bit_positions = np.arange(path_bits, dtype=np.uint64)
    path = ((gray[:, None] >> bit_positions[None, :]) & np.uint64(1)).astype(np.uint8)

    columns = (
        rng.permutation(feedback_count)
        if priority_columns is None
        else np.asarray(priority_columns, dtype=np.int64)
    )
    generator = np.zeros((path_bits, feedback_count), dtype=np.uint8)
    generator[:, columns[:path_bits]] = np.eye(path_bits, dtype=np.uint8)
    extra_columns = columns[path_bits:]
    if len(extra_columns):
        variation = rng.integers(0, 2, size=(path_bits, len(extra_columns)), dtype=np.uint8)
        constant = np.flatnonzero(~variation.any(axis=0))
        if len(constant):
            variation[rng.integers(0, path_bits, size=len(constant)), constant] = 1
        generator[:, extra_columns] = variation
    if row_count & (row_count - 1) == 0:
        generator = _reduce_gf2_path_basis(generator)
    rows = (path @ generator) & 1
    rows ^= rng.integers(0, 2, size=(1, feedback_count), dtype=np.uint8)
    return rows


def _feedback_sample_rows(
    feedback_count: int,
    sample_budget: int,
    rng: np.random.Generator,
    *,
    mode: str,
    stratified_vars: int,
    priority_columns: np.ndarray | None = None,
) -> np.ndarray:
    feedback_count = int(feedback_count)
    sample_budget = max(1, int(sample_budget))
    if feedback_count == 0:
        return np.zeros((1, 0), dtype=np.uint8)
    normalized = str(mode).strip().lower()
    if normalized in {"unified", "path", "path_variations", "nondegenerate"}:
        return _unified_path_sample_rows(
            feedback_count,
            sample_budget,
            rng,
            priority_columns=priority_columns,
        )
    if normalized in {"unified_random", "path_random"}:
        return _unified_path_sample_rows(feedback_count, sample_budget, rng)
    if normalized in {"unified_dual", "renormalized"}:
        ranked_count = (sample_budget + 1) // 2
        random_count = sample_budget - ranked_count
        ranked = _unified_path_sample_rows(
            feedback_count,
            ranked_count,
            rng,
            priority_columns=priority_columns,
        )
        if random_count == 0:
            return ranked
        random = _unified_path_sample_rows(feedback_count, random_count, rng)
        return np.concatenate((ranked, random), axis=0)
    if normalized in {"uniform", "random"}:
        return rng.integers(0, 2, size=(sample_budget, feedback_count), dtype=np.uint8)
    if normalized in {"antithetic", "paired"}:
        half = (sample_budget + 1) // 2
        base = rng.integers(0, 2, size=(half, feedback_count), dtype=np.uint8)
        rows = np.empty((2 * half, feedback_count), dtype=np.uint8)
        rows[0::2] = base
        rows[1::2] = 1 - base
        return rows[:sample_budget]
    if normalized in {"balanced", "latin", "lhs"}:
        rows = np.empty((sample_budget, feedback_count), dtype=np.uint8)
        half = sample_budget // 2
        for col_idx in range(feedback_count):
            col = np.zeros(sample_budget, dtype=np.uint8)
            col[:half] = 1
            if sample_budget % 2 and rng.random() < 0.5:
                col[-1] = 1
            rng.shuffle(col)
            rows[:, col_idx] = col
        return rows
    if normalized in {"stratified", "prefix_stratified", "prefix"}:
        prefix = min(max(0, int(stratified_vars)), feedback_count)
        if prefix == 0:
            return rng.integers(0, 2, size=(sample_budget, feedback_count), dtype=np.uint8)
        while prefix > 0 and (1 << prefix) > sample_budget:
            prefix -= 1
        block = 1 << prefix
        outer = max(1, sample_budget // block)
        rows = np.empty((outer * block, feedback_count), dtype=np.uint8)
        for outer_idx in range(outer):
            tail = rng.integers(0, 2, size=feedback_count - prefix, dtype=np.uint8)
            for mask in range(block):
                row = outer_idx * block + mask
                for bit_idx in range(prefix):
                    rows[row, bit_idx] = (mask >> bit_idx) & 1
                rows[row, prefix:] = tail
        return rows
    raise ValueError(
        "approx_tensor_residue_sample_mode must be one of "
        "'uniform', 'antithetic', 'balanced', 'stratified', 'unified_random', "
        "'unified_dual', or 'unified'; "
        f"received {mode!r}."
    )


def _feedback_bond_order(
    feedback_count: int,
    modulus: int,
    fixed_linear: list[tuple[int, int]],
    fixed_to_free: list[tuple[int, int, int]],
    fixed_to_fixed: list[tuple[int, int, int]],
) -> np.ndarray:
    """Rank feedback bonds by local tensor singular-value strength."""
    scores = np.zeros(int(feedback_count), dtype=np.float64)

    def strength(residue: int) -> float:
        return 2.0 * abs(np.sin(np.pi * (int(residue) % modulus) / float(modulus)))

    for bit_idx, residue in fixed_linear:
        scores[bit_idx] += strength(residue)
    for bit_idx, _free_var, residue in fixed_to_free:
        scores[bit_idx] += strength(residue)
    for left_bit, right_bit, residue in fixed_to_fixed:
        edge_strength = strength(residue)
        scores[left_bit] += edge_strength
        scores[right_bit] += edge_strength
    return np.lexsort((np.arange(feedback_count, dtype=np.int64), -scores))
