"""State/config/echelon primitives for strong simulator."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
import importlib
from typing import Any, Callable, Literal, Protocol, Sequence, TypedDict

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
    """Reusable row-echelon form of the output constraint matrix."""

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


_STATE_RUNTIME_EXPORTS = {
    "SchurState": "._state_runtime",
    "build_state": "._amplitude_api",
    "_apply_gate_sequence_to_state": "._amplitude_api",
    "_apply_gate_sequence_to_state_linear": "._amplitude_api",
    "_batch_query_state": "._amplitude_api",
    "_fork_state_for_extension": "._reduction_runtime",
}


def __getattr__(name: str):
    module_name = _STATE_RUNTIME_EXPORTS.get(name)
    if module_name is not None:
        module = importlib.import_module(module_name, __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BitSequence",
    "CircuitInput",
    "EchelonCache",
    "ExtendedReductionMode",
    "ReducerInfo",
    "ReductionInfo",
    "SchurState",
    "SolverConfig",
    "SupportsQiskitCircuit",
    "_get_solver_config",
    "_reset_solver_config",
    "_set_solver_config",
    "_apply_gate_sequence_to_state",
    "_apply_gate_sequence_to_state_linear",
    "_batch_query_state",
    "_fork_state_for_extension",
    "_iter_mask_bits",
    "_mask_bit",
    "_mask_from_vector",
    "_normalize_extended_reductions",
    "_parity",
    "_prepare_affine_constraint_cache",
    "_row_reduce_output_constraints",
    "_should_apply_extended_gate_rewrite",
    "_solve_echelon_rhs",
    "_solve_output_from_echelon",
    "build_state",
]
