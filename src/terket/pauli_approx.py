"""Opt-in approximate Pauli expectation facade."""

from __future__ import annotations

from ._pauli_api import (
    _compute_native_mps_approx_pauli_expectations,
    _compute_pauli_beam_approx_fast_path,
    compute_circuit_pauli_expectations_approx,
)
from ._pauli_approx_runtime import (
    _NativeApproxMPS,
    _native_mps_apply_gate,
    _native_mps_apply_pauli_expbox,
    _native_mps_approx_mirror_fidelity,
    _native_mps_approx_pauli_expectations,
    _native_mps_approx_state,
    _native_mps_one_qubit_matrix,
    _native_mps_rx_matrix,
    _native_mps_rzz_matrix,
)
from ._pauli_support import (
    _approx_pauli_expectation_info,
    _native_mps_approx_bond,
    _pauli_beam_approx_pauli_expectations,
    _pauli_beam_approx_terms,
    _pauli_beam_needs_large_default,
    _pauli_beam_prune,
    _pauli_beam_reverse_ops,
)

__all__ = [
    "_NativeApproxMPS",
    "_approx_pauli_expectation_info",
    "_compute_native_mps_approx_pauli_expectations",
    "_compute_pauli_beam_approx_fast_path",
    "_native_mps_apply_gate",
    "_native_mps_apply_pauli_expbox",
    "_native_mps_approx_bond",
    "_native_mps_approx_mirror_fidelity",
    "_native_mps_approx_pauli_expectations",
    "_native_mps_approx_state",
    "_native_mps_one_qubit_matrix",
    "_native_mps_rx_matrix",
    "_native_mps_rzz_matrix",
    "_pauli_beam_approx_pauli_expectations",
    "_pauli_beam_approx_terms",
    "_pauli_beam_needs_large_default",
    "_pauli_beam_prune",
    "_pauli_beam_reverse_ops",
    "compute_circuit_pauli_expectations_approx",
]
