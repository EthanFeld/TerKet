"""State/config types and solver preference helpers."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence, TypedDict

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
_DEFAULT_BP_HEURISTIC_MAX_LOG2_ABS_SPREAD = 2.0
_DEFAULT_BP_HEURISTIC_MAX_PHASE_SPREAD = 0.5
_DEFAULT_BP_HEURISTIC_BOUND_LOG2_TOL = 1e-6


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
    allow_approximate: bool = False
    bp_heuristic_max_log2_abs_spread: float = _DEFAULT_BP_HEURISTIC_MAX_LOG2_ABS_SPREAD
    bp_heuristic_max_phase_spread: float = _DEFAULT_BP_HEURISTIC_MAX_PHASE_SPREAD
    bp_heuristic_bound_log2_tol: float = _DEFAULT_BP_HEURISTIC_BOUND_LOG2_TOL


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
