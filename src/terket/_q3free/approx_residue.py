"""Positive-residue approximate q3-free summation helpers."""

from __future__ import annotations

import cmath

import numpy as np

from ..scaling import (
    ScaledComplex,
    _make_scaled_complex,
    _scale_scaled_complex,
)
from ..state import SolverConfig, _get_solver_config
from .exact import _q3_free_spanning_data, _select_feedback_vertices
from .primitives import _q3_free_graph
from .approx_residue_native import _sum_q3_free_residue_forest_native_batch_scaled
from .approx_sampling import _feedback_bond_order, _feedback_sample_rows

__all__ = [
    "_sum_q3_free_residue_forest_scaled",
]


def _coarse_residue(value: int, *, source_level: int, target_level: int) -> int:
    value = int(value) % (1 << int(source_level))
    shift = int(source_level) - int(target_level)
    if shift <= 0:
        return value % (1 << int(target_level))
    return ((value + (1 << (shift - 1))) >> shift) % (1 << int(target_level))


def _residue_characteristic_phases(level: int) -> np.ndarray:
    modulus = 1 << int(level)
    residues = np.arange(modulus, dtype=np.float64)
    return np.exp((2j * np.pi / float(modulus)) * residues)


def _forest_residue_characteristic(
    q1: list[int],
    adjacency: list[dict[int, int]],
    *,
    level: int,
) -> np.ndarray:
    """Return Fourier representation of a nonnegative forest residue distribution."""
    n = len(q1)
    modulus = 1 << int(level)
    if n == 0:
        return np.ones(modulus, dtype=np.complex128)

    phase = _residue_characteristic_phases(level)
    powers = np.arange(modulus, dtype=np.int64)
    visited = [False] * n
    base = [np.ones(modulus, dtype=np.complex128) for _ in range(n)]
    excited = [np.ones(modulus, dtype=np.complex128) for _ in range(n)]
    total = np.ones(modulus, dtype=np.complex128)

    for root in range(n):
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
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor != parent:
                    stack.append((neighbor, node, False))

        for node, parent in postorder:
            off_term = np.ones(modulus, dtype=np.complex128)
            on_term = phase[(powers * (int(q1[node]) % modulus)) % modulus].copy()
            for child, shift in adjacency[node].items():
                if child == parent:
                    continue
                shifted_child = phase[(powers * (int(shift) % modulus)) % modulus] * excited[child]
                off_term *= 0.5 * (base[child] + excited[child])
                on_term *= 0.5 * (base[child] + shifted_child)
            base[node] = off_term
            excited[node] = on_term

        total *= 0.5 * (base[root] + excited[root])

    return total


def _sum_q3_free_residue_forest_scaled(
    q,
    *,
    config: SolverConfig | None = None,
) -> ScaledComplex | None:
    """Sample feedback vars, exactly sum remaining forest residue distribution."""
    if q.q3:
        return None
    cfg = _get_solver_config() if config is None else config
    if q.n > int(cfg.approx_tensor_max_vars):
        return None

    target_level = max(1, min(int(q.level), int(cfg.approx_tensor_residue_level)))
    modulus = 1 << target_level
    adjacency, edges = _q3_free_graph(q)
    depth, chords = _q3_free_spanning_data(adjacency, edges)
    feedback_vars = tuple(_select_feedback_vertices(q.n, chords, depth))
    sample_budget = max(1, int(cfg.approx_tensor_residue_forest_samples))
    enumerate_feedback = len(feedback_vars) <= 62 and (1 << len(feedback_vars)) <= sample_budget
    fixed_pos = {var: idx for idx, var in enumerate(feedback_vars)}
    free_vars = [var for var in range(q.n) if var not in fixed_pos]
    free_index = {var: idx for idx, var in enumerate(free_vars)}
    base_q1 = [
        _coarse_residue(q.q1[var], source_level=q.level, target_level=target_level)
        for var in free_vars
    ]
    free_adjacency = [dict() for _ in free_vars]
    fixed_linear = [
        (idx, _coarse_residue(q.q1[var], source_level=q.level, target_level=target_level))
        for var, idx in fixed_pos.items()
    ]
    fixed_to_free: list[tuple[int, int, int]] = []
    fixed_to_fixed: list[tuple[int, int, int]] = []
    for left, right, phase in edges:
        coarse_phase = _coarse_residue(phase, source_level=q.level, target_level=target_level)
        bit_left = fixed_pos.get(left)
        bit_right = fixed_pos.get(right)
        if bit_left is not None and bit_right is not None:
            fixed_to_fixed.append((bit_left, bit_right, coarse_phase))
        elif bit_left is not None:
            fixed_to_free.append((bit_left, free_index[right], coarse_phase))
        elif bit_right is not None:
            fixed_to_free.append((bit_right, free_index[left], coarse_phase))
        else:
            mapped_left = free_index[left]
            mapped_right = free_index[right]
            free_adjacency[mapped_left][mapped_right] = coarse_phase
            free_adjacency[mapped_right][mapped_left] = coarse_phase

    rng = np.random.default_rng(int(cfg.approx_tensor_residue_seed))
    if enumerate_feedback:
        fixed_bit_rows = np.asarray(
            [
                [(mask >> bit) & 1 for bit in range(len(feedback_vars))]
                for mask in range(1 << len(feedback_vars))
            ],
            dtype=np.uint8,
        )
    else:
        bond_order = _feedback_bond_order(
            len(feedback_vars),
            modulus,
            fixed_linear,
            fixed_to_free,
            fixed_to_fixed,
        )
        fixed_bit_rows = _feedback_sample_rows(
            len(feedback_vars),
            sample_budget,
            rng,
            mode=str(cfg.approx_tensor_residue_sample_mode),
            stratified_vars=int(cfg.approx_tensor_residue_stratified_vars),
            priority_columns=bond_order,
        )

    native_total = _sum_q3_free_residue_forest_native_batch_scaled(
        q,
        target_level=target_level,
        feedback_count=len(feedback_vars),
        fixed_bit_rows=fixed_bit_rows,
        base_q1=base_q1,
        free_adjacency=free_adjacency,
        fixed_linear=fixed_linear,
        fixed_to_free=fixed_to_free,
        fixed_to_fixed=fixed_to_fixed,
    )
    if native_total is not None:
        return native_total

    mean_phase = 0.0 + 0.0j
    phase = _residue_characteristic_phases(target_level)
    for fixed_bits in fixed_bit_rows:
        q1_shifted = list(base_q1)
        const_residue = 0
        for bit_idx, residue in fixed_linear:
            if fixed_bits[bit_idx]:
                const_residue = (const_residue + residue) % modulus
        for bit_idx, free_var, residue in fixed_to_free:
            if fixed_bits[bit_idx]:
                q1_shifted[free_var] = (q1_shifted[free_var] + residue) % modulus
        for left_bit, right_bit, residue in fixed_to_fixed:
            if fixed_bits[left_bit] and fixed_bits[right_bit]:
                const_residue = (const_residue + residue) % modulus
        characteristic = _forest_residue_characteristic(
            q1_shifted,
            free_adjacency,
            level=target_level,
        )
        mean_phase += phase[const_residue] * characteristic[1]

    mean_phase /= float(len(fixed_bit_rows))
    scalar = cmath.exp(2j * cmath.pi * float(q.q0))
    return _scale_scaled_complex(_make_scaled_complex(scalar * mean_phase), 2 * int(q.n))
