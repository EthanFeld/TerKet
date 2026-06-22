"""Exact PauliExpBox and Pauli expectation facade."""

from __future__ import annotations

from ._pauli_api import (
    _build_pauli_expectation_base_state,
    _prepare_pauli_expectation_request,
    _select_pauli_direct_replay_template,
    compute_circuit_pauli_expectations,
)
from ._pauli_support import (
    _pauli_expbox_dyadic_snap_level,
    apply_pauli_expbox_to_state,
)
from ._reduction_runtime import _pauli_string_gates, _validate_pauli_observables

__all__ = [
    "_build_pauli_expectation_base_state",
    "_pauli_expbox_dyadic_snap_level",
    "_pauli_string_gates",
    "_prepare_pauli_expectation_request",
    "_select_pauli_direct_replay_template",
    "_validate_pauli_observables",
    "apply_pauli_expbox_to_state",
    "compute_circuit_pauli_expectations",
]
