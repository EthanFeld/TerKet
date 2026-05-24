"""Opt-in approximate backends."""

from __future__ import annotations

from ._arbitrary_bp import (
    _arbitrary_bp_backend,
    _mark_invalid_arbitrary_bp_info,
    _raise_if_invalid_arbitrary_bp_amplitude,
    _sum_factor_graph_bethe_scaled,
    _sum_factor_graph_with_sparse_parity_bethe_scaled,
    _sum_pairwise_factor_graph_bethe_scaled,
)
from ._arbitrary_runtime import (
    _arbitrary_bp_heuristic_candidate,
    _factor_graph_is_forest,
    _sum_arbitrary_bp_heuristic_ensemble_scaled,
    _sum_with_arbitrary_phases_approx_scaled,
    solve_arbitrary_approx,
)
from ._pauli_api import compute_circuit_pauli_expectations_approx
from ._pauli_approx_runtime import (
    _native_mps_approx_mirror_fidelity,
    _native_mps_approx_pauli_expectations,
    _native_mps_approx_state,
)
from ._pauli_support import (
    _native_mps_approx_bond,
    _pauli_beam_approx_pauli_expectations,
    _pauli_beam_approx_terms,
)

__all__ = [
    "_arbitrary_bp_backend",
    "_arbitrary_bp_heuristic_candidate",
    "_factor_graph_is_forest",
    "_mark_invalid_arbitrary_bp_info",
    "_native_mps_approx_bond",
    "_native_mps_approx_mirror_fidelity",
    "_native_mps_approx_pauli_expectations",
    "_native_mps_approx_state",
    "_pauli_beam_approx_pauli_expectations",
    "_pauli_beam_approx_terms",
    "_raise_if_invalid_arbitrary_bp_amplitude",
    "_sum_arbitrary_bp_heuristic_ensemble_scaled",
    "_sum_factor_graph_bethe_scaled",
    "_sum_factor_graph_with_sparse_parity_bethe_scaled",
    "_sum_pairwise_factor_graph_bethe_scaled",
    "_sum_with_arbitrary_phases_approx_scaled",
    "compute_circuit_pauli_expectations_approx",
    "solve_arbitrary_approx",
]
