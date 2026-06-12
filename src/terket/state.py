"""Focused state facade for config, echelon helpers, and runtime exports.

This module keeps stable imports for callers while delegating concrete ownership
to:

- ``terket._state_config`` for solver config and public metadata types
- ``terket._state_echelon`` for output-constraint helpers
- runtime owner modules for heavy state-construction behavior
"""

from __future__ import annotations

import importlib
from ._state_config import (
    BitSequence,
    CircuitInput,
    ExtendedReductionMode,
    ReducerInfo,
    ReductionInfo,
    SolverConfig,
    SupportsQiskitCircuit,
    _get_solver_config,
    _normalize_extended_reductions,
    _reset_solver_config,
    _set_solver_config,
    _should_apply_extended_gate_rewrite,
)
from ._state_echelon import (
    EchelonCache,
    _iter_mask_bits,
    _mask_bit,
    _mask_from_vector,
    _parity,
    _prepare_affine_constraint_cache,
    _row_reduce_output_constraints,
    _solve_echelon_rhs,
    _solve_output_from_echelon,
)


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
