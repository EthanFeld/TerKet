"""State/config types and solver preference helpers."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, Protocol, Sequence, TypedDict

from .spec import CircuitSpec, Gate


BitSequence = Sequence[int]
ExtendedReductionMode = Literal["auto", "always", "never"]


class SupportsQiskitCircuit(Protocol):
    """Structural type for Qiskit-style circuit inputs."""

    num_qubits: int
    data: Any


CircuitInput = CircuitSpec | str | SupportsQiskitCircuit


class ReducerInfo(TypedDict):
    """Reducer metadata returned by ``reduce_and_sum()``."""

    quad: int
    constraint: int
    branched: int
    remaining: int
    structural_obstruction: int
    gauss_obstruction: int
    cost_r: int
    phase_states: int
    phase_splits: int
    phase3_backend: str | None
    approx_q3_free_method: NotRequired[str]
    approx_q3_free_reliable: NotRequired[bool]
    approx_q3_free_rejection_reason: NotRequired[str]
    approx_q3_free_repeats: NotRequired[int]
    approx_q3_free_level: NotRequired[int]
    approx_q3_free_samples: NotRequired[int]
    approx_q3_free_log2_abs: NotRequired[float]
    approx_q3_free_error_log2_abs: NotRequired[float]
    approx_q3_free_rel_stderr: NotRequired[float]
    approx_q3_free_log2_spread: NotRequired[float]
    approx_q3_free_bound_violation_log2: NotRequired[float]
    approx_q3_free_mps_bond: NotRequired[int]
    approx_q3_free_mps_order: NotRequired[str]
    approx_q3_free_mps_route_swaps: NotRequired[int]
    approx_q3_free_mps_width: NotRequired[int]
    approx_q3_free_mps_peak_active: NotRequired[int]
    approx_q3_free_mps_peak_bond: NotRequired[int]
    approx_q3_free_mps_discarded_rss: NotRequired[float]
    approx_q3_free_mps_max_discarded: NotRequired[float]


class ReductionInfo(TypedDict):
    """Public amplitude-query metadata returned by the high-level API."""

    initial_free: int
    quad_eliminated: int
    constraint_eliminated: int
    branched: int
    remaining_free: int
    branches: int
    cost_model_r: int
    cubic_obstruction: int
    has_cubic_obstruction: bool
    gauss_obstruction: int
    has_gauss_obstruction: bool
    phase_states: int
    phase_splits: int
    phase3_backend: str | None
    is_zero: bool
    approx_q3_free_method: NotRequired[str]
    approx_q3_free_reliable: NotRequired[bool]
    approx_q3_free_rejection_reason: NotRequired[str]
    approx_q3_free_repeats: NotRequired[int]
    approx_q3_free_level: NotRequired[int]
    approx_q3_free_samples: NotRequired[int]
    approx_q3_free_log2_abs: NotRequired[float]
    approx_q3_free_error_log2_abs: NotRequired[float]
    approx_q3_free_rel_stderr: NotRequired[float]
    approx_q3_free_log2_spread: NotRequired[float]
    approx_q3_free_bound_violation_log2: NotRequired[float]
    approx_q3_free_mps_bond: NotRequired[int]
    approx_q3_free_mps_order: NotRequired[str]
    approx_q3_free_mps_route_swaps: NotRequired[int]
    approx_q3_free_mps_width: NotRequired[int]
    approx_q3_free_mps_peak_active: NotRequired[int]
    approx_q3_free_mps_peak_bond: NotRequired[int]
    approx_q3_free_mps_discarded_rss: NotRequired[float]
    approx_q3_free_mps_max_discarded: NotRequired[float]


_DEFAULT_CUTSET_MAX_SIZE = 6
_DEFAULT_CUTSET_CANDIDATE_POOL = 24
_DEFAULT_CUTSET_BEAM_WIDTH = 4
_DEFAULT_CUTSET_BRANCHES_PER_STATE = 3
_DEFAULT_ONE_SHOT_CUTSET_MAX_SIZE = 10
_DEFAULT_ONE_SHOT_CUTSET_CANDIDATE_POOL = 40
_DEFAULT_ONE_SHOT_CUTSET_BEAM_WIDTH = 6
_DEFAULT_ONE_SHOT_CUTSET_BRANCHES_PER_STATE = 4
_DEFAULT_TENSOR_HINT_TARGET_WIDTH = 14
_DEFAULT_TENSOR_HINT_MAX_REPEATS = 4
_DEFAULT_TENSOR_HINT_MAX_TIME = 2.0
_DEFAULT_TENSOR_HINT_MIN_VARS = 128
_DEFAULT_TENSOR_HINT_MAX_VARS = 384
_DEFAULT_APPROX_Q3_FREE_TENSOR = False
_DEFAULT_APPROX_TENSOR_MAX_BOND = 64
_DEFAULT_APPROX_TENSOR_CUTOFF = 1e-10
_DEFAULT_APPROX_TENSOR_OPTIMIZE = "greedy"
_DEFAULT_APPROX_TENSOR_MAX_VARS = 100_000
_DEFAULT_APPROX_TENSOR_MAX_DEGREE = 16
_DEFAULT_APPROX_TENSOR_METHOD = "residue_forest"
_DEFAULT_APPROX_TENSOR_BP_MAX_ITERS = 30
_DEFAULT_APPROX_TENSOR_BP_TOL = 1e-8
_DEFAULT_APPROX_TENSOR_BP_DAMPING = 0.25
_DEFAULT_APPROX_TENSOR_RESIDUE_SAMPLES = 4096
_DEFAULT_APPROX_TENSOR_RESIDUE_BATCH = 256
_DEFAULT_APPROX_TENSOR_RESIDUE_SEED = 0
_DEFAULT_APPROX_TENSOR_RESIDUE_LEVEL = 16
_DEFAULT_APPROX_TENSOR_RESIDUE_FOREST_SAMPLES = 32
_DEFAULT_APPROX_TENSOR_RESIDUE_SAMPLE_MODE = "unified"
_DEFAULT_APPROX_TENSOR_RESIDUE_STRATIFIED_VARS = 0
_DEFAULT_APPROX_TENSOR_RELIABILITY_REPEATS = 3
_DEFAULT_APPROX_TENSOR_RELIABILITY_SEED_STRIDE = 104729
_DEFAULT_APPROX_TENSOR_RELIABILITY_MAX_LOG2_SPREAD = 8.0
_DEFAULT_APPROX_TENSOR_RELIABILITY_MAX_REL_STDERR = 1.0
_DEFAULT_APPROX_TENSOR_RELIABILITY_MIN_LOG2_ABS_FOR_REL = -40.0
_DEFAULT_APPROX_TENSOR_RELIABILITY_REJECT = True
_DEFAULT_APPROX_TENSOR_AMPLITUDE_BOUND_SLACK_LOG2 = 1e-9
_DEFAULT_APPROX_TENSOR_RAISE_ON_UNRELIABLE = True
_DEFAULT_APPROX_TENSOR_MPS_FALLBACK = True
_DEFAULT_APPROX_TENSOR_MPS_MAX_BOND = 16
_DEFAULT_APPROX_TENSOR_MPS_MAX_REL_CHANGE = 0.25
_DEFAULT_APPROX_TENSOR_MPS_MAX_DISCARDED = 0.15


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """User-tunable solver preference knobs for TerKet's phase-sum backends."""

    cutset_max_size: int = _DEFAULT_CUTSET_MAX_SIZE
    cutset_candidate_pool: int = _DEFAULT_CUTSET_CANDIDATE_POOL
    cutset_beam_width: int = _DEFAULT_CUTSET_BEAM_WIDTH
    cutset_branches_per_state: int = _DEFAULT_CUTSET_BRANCHES_PER_STATE
    one_shot_cutset_max_size: int = _DEFAULT_ONE_SHOT_CUTSET_MAX_SIZE
    one_shot_cutset_candidate_pool: int = _DEFAULT_ONE_SHOT_CUTSET_CANDIDATE_POOL
    one_shot_cutset_beam_width: int = _DEFAULT_ONE_SHOT_CUTSET_BEAM_WIDTH
    one_shot_cutset_branches_per_state: int = _DEFAULT_ONE_SHOT_CUTSET_BRANCHES_PER_STATE
    tensor_hint_target_width: int = _DEFAULT_TENSOR_HINT_TARGET_WIDTH
    tensor_hint_max_repeats: int = _DEFAULT_TENSOR_HINT_MAX_REPEATS
    tensor_hint_max_time: float = _DEFAULT_TENSOR_HINT_MAX_TIME
    tensor_hint_min_vars: int = _DEFAULT_TENSOR_HINT_MIN_VARS
    tensor_hint_max_vars: int = _DEFAULT_TENSOR_HINT_MAX_VARS
    approx_q3_free_tensor: bool = _DEFAULT_APPROX_Q3_FREE_TENSOR
    approx_tensor_max_bond: int = _DEFAULT_APPROX_TENSOR_MAX_BOND
    approx_tensor_cutoff: float = _DEFAULT_APPROX_TENSOR_CUTOFF
    approx_tensor_optimize: str = _DEFAULT_APPROX_TENSOR_OPTIMIZE
    approx_tensor_max_vars: int = _DEFAULT_APPROX_TENSOR_MAX_VARS
    approx_tensor_max_degree: int = _DEFAULT_APPROX_TENSOR_MAX_DEGREE
    approx_tensor_method: str = _DEFAULT_APPROX_TENSOR_METHOD
    approx_tensor_bp_max_iters: int = _DEFAULT_APPROX_TENSOR_BP_MAX_ITERS
    approx_tensor_bp_tol: float = _DEFAULT_APPROX_TENSOR_BP_TOL
    approx_tensor_bp_damping: float = _DEFAULT_APPROX_TENSOR_BP_DAMPING
    approx_tensor_residue_samples: int = _DEFAULT_APPROX_TENSOR_RESIDUE_SAMPLES
    approx_tensor_residue_batch: int = _DEFAULT_APPROX_TENSOR_RESIDUE_BATCH
    approx_tensor_residue_seed: int = _DEFAULT_APPROX_TENSOR_RESIDUE_SEED
    approx_tensor_residue_level: int = _DEFAULT_APPROX_TENSOR_RESIDUE_LEVEL
    approx_tensor_residue_forest_samples: int = _DEFAULT_APPROX_TENSOR_RESIDUE_FOREST_SAMPLES
    approx_tensor_residue_sample_mode: str = _DEFAULT_APPROX_TENSOR_RESIDUE_SAMPLE_MODE
    approx_tensor_residue_stratified_vars: int = _DEFAULT_APPROX_TENSOR_RESIDUE_STRATIFIED_VARS
    approx_tensor_reliability_repeats: int = _DEFAULT_APPROX_TENSOR_RELIABILITY_REPEATS
    approx_tensor_reliability_seed_stride: int = _DEFAULT_APPROX_TENSOR_RELIABILITY_SEED_STRIDE
    approx_tensor_reliability_max_log2_spread: float = _DEFAULT_APPROX_TENSOR_RELIABILITY_MAX_LOG2_SPREAD
    approx_tensor_reliability_max_rel_stderr: float = _DEFAULT_APPROX_TENSOR_RELIABILITY_MAX_REL_STDERR
    approx_tensor_reliability_min_log2_abs_for_rel: float = (
        _DEFAULT_APPROX_TENSOR_RELIABILITY_MIN_LOG2_ABS_FOR_REL
    )
    approx_tensor_reliability_reject: bool = _DEFAULT_APPROX_TENSOR_RELIABILITY_REJECT
    approx_tensor_amplitude_bound_slack_log2: float = _DEFAULT_APPROX_TENSOR_AMPLITUDE_BOUND_SLACK_LOG2
    approx_tensor_raise_on_unreliable: bool = _DEFAULT_APPROX_TENSOR_RAISE_ON_UNRELIABLE
    approx_tensor_mps_fallback: bool = _DEFAULT_APPROX_TENSOR_MPS_FALLBACK
    approx_tensor_mps_max_bond: int = _DEFAULT_APPROX_TENSOR_MPS_MAX_BOND
    approx_tensor_mps_max_rel_change: float = _DEFAULT_APPROX_TENSOR_MPS_MAX_REL_CHANGE
    approx_tensor_mps_max_discarded: float = _DEFAULT_APPROX_TENSOR_MPS_MAX_DISCARDED


_DEFAULT_SOLVER_CONFIG = SolverConfig()
_SOLVER_CONFIG_VAR: contextvars.ContextVar[SolverConfig] = contextvars.ContextVar(
    "_terket_solver_config",
    default=_DEFAULT_SOLVER_CONFIG,
)


def _get_solver_config() -> SolverConfig:
    return _SOLVER_CONFIG_VAR.get()


def _set_solver_config(solver_config: SolverConfig | None):
    if solver_config is None:
        return None
    return _SOLVER_CONFIG_VAR.set(solver_config)


def _reset_solver_config(token) -> None:
    if token is not None:
        _SOLVER_CONFIG_VAR.reset(token)


def _normalize_extended_reductions(mode: ExtendedReductionMode | str | None) -> ExtendedReductionMode:
    if mode is None:
        return "auto"
    normalized = str(mode).strip().lower()
    if normalized not in {"auto", "always", "never"}:
        raise ValueError(
            f"extended_reductions must be one of 'auto', 'always', or 'never'; received {mode!r}."
        )
    return normalized


def _should_apply_extended_gate_rewrite(
    mode: ExtendedReductionMode | str | None,
    gates: Sequence[Gate],
    *,
    auto_max_gates: int,
) -> bool:
    normalized = _normalize_extended_reductions(mode)
    if normalized == "always":
        return True
    if normalized == "never":
        return False
    return len(gates) <= int(auto_max_gates)


__all__ = [
    "BitSequence",
    "CircuitInput",
    "ExtendedReductionMode",
    "ReducerInfo",
    "ReductionInfo",
    "SolverConfig",
    "SupportsQiskitCircuit",
    "_get_solver_config",
    "_normalize_extended_reductions",
    "_reset_solver_config",
    "_set_solver_config",
    "_should_apply_extended_gate_rewrite",
]
