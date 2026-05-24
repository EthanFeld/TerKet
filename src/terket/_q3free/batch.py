"""Owned q3-free reusable batch dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
