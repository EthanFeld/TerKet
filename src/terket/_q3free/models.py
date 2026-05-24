"""Owned q3-free plan dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np

from ..cubic_arithmetic import PhaseFunction
from ..scaling import ScaledComplex
from ..state import EchelonCache


@dataclass(frozen=True, slots=True)
class _BinaryPhaseQuadraticPlan:
    """Precomputed elimination schedule for a binary +/-1 quadratic phase."""

    n: int
    residual_active_count: int
    half_pow2_exp: int
    partner_swaps: tuple[int, ...]
    pivot_swaps: tuple[int, ...]
    c1_rows: tuple[np.ndarray, ...]
    c2_rows: tuple[np.ndarray, ...]
    c1_and_c2_rows: tuple[np.ndarray, ...]


@dataclass(frozen=True, slots=True)
class _Q3FreeCutsetConditioningPlan:
    """Reusable exact cutset-conditioned treewidth plan for a q3-free kernel."""

    level: int
    cutset_vars: tuple[int, ...]
    remaining_vars: tuple[int, ...]
    remaining_backend: Literal["product", "treewidth", "generic"]
    remaining_q2: dict[tuple[int, int], int]
    remaining_order: tuple[int, ...]
    cutset_remaining_q2_residue: np.ndarray
    cutset_cutset_left: np.ndarray
    cutset_cutset_right: np.ndarray
    cutset_cutset_residue: np.ndarray
    native_treewidth_plan: object | None = None
    remaining_isolated_vars: tuple[int, ...] = ()
    remaining_components: tuple["_Q3FreeConstraintComponentPlan", ...] = ()
    remaining_width: int = 0
    estimated_total_work: int = 0
    branch_bits: np.ndarray | None = None
    branch_pair_residue: np.ndarray | None = None
    branch_remaining_shift: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class _Q3FreeResidualProjection:
    """Projected q3-free remainder induced by a candidate cutset."""

    remaining_vars: tuple[int, ...]
    remaining_q: PhaseFunction


@dataclass(frozen=True, slots=True)
class _Q3FreeCutsetCandidateEvaluation:
    """Search-time summary for a candidate q3-free cutset."""

    cutset_vars: tuple[int, ...]
    plan: _Q3FreeCutsetConditioningPlan | None
    viable: bool
    score: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Q3FreeConstraintComponentPlan:
    """Preplanned q3-free component topology for repeated constrained sums."""

    variables: tuple[int, ...]
    level: int
    q2: dict[tuple[int, int], int]
    backend: Literal["constant", "forest", "treewidth", "generic"]
    adjacency: tuple[tuple[tuple[int, int], ...], ...] = ()
    order: tuple[int, ...] = ()
    dense_q2: np.ndarray | None = None
    precomputed_total: ScaledComplex | None = None
    binary_phase_plan: _BinaryPhaseQuadraticPlan | None = None
    mediator_plan: _HalfPhaseMediatorPlan | None = None
    generic_mediator_plan: _GenericQ2MediatorPlan | None = None
    cluster_plan: _HalfPhaseClusterPlan | None = None
    cutset_plan: _Q3FreeCutsetConditioningPlan | None = None
    native_treewidth_plan: object | None = None
    skip_dense_schur: bool = False
    direct_schur_ok: bool = False
    quadratic_tensor_q2: bool = False
    lambda_offset: int = -1
    prefer_reusable_decomposition: bool = False
    prefer_cutset_backend: bool = False


@dataclass(frozen=True, slots=True)
class _Q3FreeConstraintPlan:
    """Exact constrained-sum plan that avoids affine parity substitution."""

    cache: EchelonCache
    eps0: tuple[int, ...]
    level: int
    q0: Fraction
    base_q1: tuple[int, ...]
    base_q2: dict[tuple[int, int], int]
    lambda_offset: int
    rank: int
    n_free_after_constraints: int
    rhs_linear_coeff: int
    isolated_vars: tuple[int, ...]
    components: tuple[_Q3FreeConstraintComponentPlan, ...]


@dataclass(frozen=True, slots=True)
class _Q3FreeRawConstraintPlan:
    """Exact constrained-sum plan over the raw output rows of a q3-free state."""

    eps0: tuple[int, ...]
    level: int
    q0: Fraction
    base_q1: tuple[int, ...]
    base_q2: dict[tuple[int, int], int]
    lambda_offset: int
    constraint_count: int
    rhs_linear_coeff: int
    isolated_vars: tuple[int, ...]
    components: tuple[_Q3FreeConstraintComponentPlan, ...]


@dataclass(frozen=True, slots=True)
class _Q3FreeRawConstraintRestrictedPlan:
    """Prefix-restricted view of a raw-output q3-free constraint plan."""

    active_count: int
    isolated_vars: tuple[int, ...]
    components: tuple[_Q3FreeConstraintComponentPlan, ...]


@dataclass(frozen=True, slots=True)
class _Q3FreeExecutionPlan:
    """Fully instantiated q3-free execution plan."""

    level: int
    q0: Fraction
    q1: tuple[int, ...]
    isolated_vars: tuple[int, ...]
    components: tuple[_Q3FreeConstraintComponentPlan, ...]


@dataclass(frozen=True, slots=True)
class _Q3FreeReusableExecutionPlan:
    """Q3-free execution topology reusable across q1/q0 shifts of one structure."""

    level: int
    isolated_vars: tuple[int, ...]
    components: tuple[_Q3FreeConstraintComponentPlan, ...]
