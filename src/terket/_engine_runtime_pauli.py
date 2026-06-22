"""Engine-runtime splice module for Pauli expectation exports."""

from __future__ import annotations

from ._engine_runtime_core import *

_bind_extracted_forwarders(
    "_pauli_support",
    "_pauli_expbox_dyadic_snap_level",
)

_bind_extracted_forwarders(
    "_pauli_api",
    "_prepare_pauli_expectation_request",
    "_pauli_expectation_result",
    "_build_pauli_expectation_base_state",
    "_select_pauli_direct_replay_template",
    "compute_circuit_pauli_expectations",
    "analyze_amplitudes",
    "analyze_circuit",
    "compute_amplitude",
    "compute_amplitude_scaled",
)

__all__ = [name for name in globals() if not name.startswith("__")]
