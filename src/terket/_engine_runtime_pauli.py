"""Engine-runtime splice module for Pauli expectation exports."""

from __future__ import annotations

from ._engine_runtime_core import *

_NATIVE_MPS_APPROX_DEFAULT_BOND = 1
_PAULI_BEAM_APPROX_DEFAULT_TERMS = 16
_PAULI_BEAM_APPROX_LARGE_TERMS = 4096
_bind_extracted_forwarders(
    "_pauli_support",
    "_pauli_expbox_dyadic_snap_level",
    "_native_mps_approx_bond",
    "_approx_pauli_expectation_info",
    "_pauli_beam_approx_terms",
    "_pauli_beam_needs_large_default",
    "_pauli_masks_from_string",
    "_pauli_masks_from_sparse",
)

_PAULI_PRODUCT_PHASE = {
    (0, 0): 1.0 + 0.0j,
    (0, 1): 1.0 + 0.0j,
    (0, 2): 1.0 + 0.0j,
    (0, 3): 1.0 + 0.0j,
    (1, 0): 1.0 + 0.0j,
    (2, 0): 1.0 + 0.0j,
    (3, 0): 1.0 + 0.0j,
    (1, 1): 1.0 + 0.0j,
    (2, 2): 1.0 + 0.0j,
    (3, 3): 1.0 + 0.0j,
    (1, 2): 1.0j,
    (2, 1): -1.0j,
    (2, 3): 1.0j,
    (3, 2): -1.0j,
    (3, 1): 1.0j,
    (1, 3): -1.0j,
}

_bind_extracted_forwarders(
    "_pauli_support",
    "_pauli_code",
    "_pauli_product_phase",
    "_pauli_product_phase_left_parts",
    "_pauli_beam_prune",
)

_PauliBeamOp = tuple[str, object]
_bind_extracted_forwarders(
    "_pauli_support",
    "_pauli_beam_reverse_ops",
    "_pauli_beam_approx_pauli_expectations",
)
_bind_extracted_forwarders("_pauli_approx_runtime", "_native_mps_one_qubit_matrix")

_NATIVE_MPS_CNOT = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
    dtype=np.complex128,
).reshape(2, 2, 2, 2)
_NATIVE_MPS_CZ = np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128).reshape(2, 2, 2, 2)
_NATIVE_MPS_SWAP = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.complex128,
).reshape(2, 2, 2, 2)

_bind_extracted_forwarders(
    "_pauli_approx_runtime",
    "_native_mps_rx_matrix",
    "_native_mps_rzz_matrix",
    "_native_mps_apply_pauli_expbox",
    "_native_mps_approx_state",
    "_native_mps_apply_gate",
    "_native_mps_approx_mirror_fidelity",
    "_native_mps_approx_pauli_expectations",
)
_bind_extracted_forwarders(
    "_pauli_api",
    "_prepare_pauli_expectation_request",
    "_pauli_expectation_result",
    "_compute_pauli_beam_approx_fast_path",
    "_compute_native_mps_approx_pauli_expectations",
    "compute_circuit_pauli_expectations_approx",
    "_build_pauli_expectation_base_state",
    "_select_pauli_direct_replay_template",
    "compute_circuit_pauli_expectations",
    "analyze_amplitudes",
    "analyze_circuit",
    "compute_amplitude",
    "compute_amplitude_scaled",
)

__all__ = [name for name in globals() if not name.startswith("__")]
