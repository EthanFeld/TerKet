from __future__ import annotations

from ._engine_runtime_core import *
from ._engine_runtime_state import _engine_cache

_bind_extracted_forwarders("_amplitude_api", "affine_compose", "reduce_and_sum")
_bind_synced_local_impl_forwarders("_amplitude_api", "build_state")

_SUBCIRCUIT_MACRO_MIN_TOTAL_GATES = 128
_SUBCIRCUIT_MACRO_WINDOW_LENGTHS = (32, 24, 16, 12, 8, 6, 5, 4)
_SUBCIRCUIT_MACRO_SEEN_CACHE_SIZE = 1 << 14
_SUBCIRCUIT_MACRO_COMPILED_CACHE_SIZE = 1 << 11
_SUBCIRCUIT_MACRO_PLAN_CACHE_SIZE = 64
_SUBCIRCUIT_MACRO_PLAN_STEP = "__subcircuit_macro__"

_SubcircuitMacro = Callable[[Any, tuple[int, ...]], None]
_SUBCIRCUIT_MACRO_PLAN_PENDING = object()
_NO_SUBCIRCUIT_MACRO_PLAN = object()
_SUBCIRCUIT_MACRO_PLAN_CACHE = _engine_cache("subcircuit_macro.plan", _SUBCIRCUIT_MACRO_PLAN_CACHE_SIZE)

_bind_extracted_forwarders(
    "_amplitude_api",
    "_apply_gate_sequence_to_state_linear",
    "_subcircuit_qubit_slot",
    "_canonicalize_subcircuit_window",
    "_compiled_subcircuit_macro_source_line",
    "_compile_subcircuit_macro",
    "_build_subcircuit_macro_replay_plan",
    "_apply_gate_sequence_to_state",
    "_batch_query_state",
    "compute_circuit_amplitude",
    "compute_circuit_amplitude_scaled",
    "compute_amplitudes",
)

__all__ = [name for name in globals() if not name.startswith("__")]
