"""Extracted Pauli expectation API helpers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from typing import Literal, Sequence, overload

from ._engine_runtime_core import _configure_extracted_module
from .cubic_arithmetic import PhaseFunction
from .scaling import ScaledAmplitude, ScaledComplex
from .spec import CircuitSpec, Gate
from .state import BitSequence, CircuitInput, EchelonCache, ExtendedReductionMode, ReductionInfo, SolverConfig

_LOCAL_NAMES = {
    '_prepare_pauli_expectation_request',
    '_pauli_expectation_result',
    '_compute_pauli_beam_approx_fast_path',
    '_compute_native_mps_approx_pauli_expectations',
    'compute_circuit_pauli_expectations_approx',
    '_build_pauli_expectation_base_state',
    '_select_pauli_direct_replay_template',
    'compute_circuit_pauli_expectations',
    'analyze_amplitudes',
    'analyze_circuit',
    'compute_amplitude',
    'compute_amplitude_scaled'
}
_LOCAL_IMPLS = {}
_configure_extracted_module(globals(), local_names=_LOCAL_NAMES, local_impls=_LOCAL_IMPLS)


def _refresh_engine_bindings() -> None:
    _sync_from_engine(importlib.import_module("terket._engine_impl"))


@dataclass(frozen=True, slots=True)
class _PauliExpectationRequest:
    spec: CircuitSpec
    input_bits: tuple[int, ...]
    observables: tuple[str, ...]
    observable_gate_sets: tuple[tuple[Gate, ...], ...]
    inverse_gates: tuple[Gate, ...]

def _prepare_pauli_expectation_request(
    circuit: CircuitInput,
    observables: Sequence[str],
    input_bits: BitSequence | None,
) -> _PauliExpectationRequest:
    from .spec import normalize_circuit

    spec = normalize_circuit(circuit)
    prepared_input = tuple(int(bit) & 1 for bit in (input_bits if input_bits is not None else (0,) * spec.n_qubits))
    if len(prepared_input) != spec.n_qubits:
        raise ValueError(f"Expected {spec.n_qubits} input bits, received {len(prepared_input)}.")
    normalized_observables = _validate_pauli_observables(observables, spec.n_qubits)
    return _PauliExpectationRequest(
        spec=spec,
        input_bits=prepared_input,
        observables=normalized_observables,
        observable_gate_sets=tuple(_pauli_string_gates(observable) for observable in normalized_observables),
        inverse_gates=_invert_native_gates(spec.gates),
    )


def _pauli_expectation_result(
    value: complex,
    info: ReductionInfo,
    *,
    as_complex: bool,
) -> tuple[ScaledAmplitude | complex, ReductionInfo]:
    amp = value if as_complex else ScaledAmplitude.from_tuple(_make_scaled_complex(value))
    return amp, dict(info)


def _scaled_pauli_expectation_result(
    value: ScaledComplex,
    info: ReductionInfo,
    *,
    as_complex: bool,
) -> tuple[ScaledAmplitude | complex, ReductionInfo]:
    amp = _scaled_to_complex(value) if as_complex else ScaledAmplitude.from_tuple(value)
    return amp, info


def _zero_pauli_expectation_result(*, as_complex: bool) -> tuple[ScaledAmplitude | complex, ReductionInfo]:
    return _pauli_expectation_result(0j, _info(0, 0, 0, 0, 0, zero=True), as_complex=as_complex)


def _compute_pauli_beam_approx_fast_path(
    request: _PauliExpectationRequest,
    *,
    as_complex: bool,
    max_terms: int | None = None,
) -> list[tuple[ScaledAmplitude | complex, ReductionInfo]] | None:
    native_values = _pauli_beam_approx_pauli_expectations(
        request.spec,
        request.input_bits,
        request.observables,
        max_terms=max_terms,
    )
    if native_values is None:
        return None
    info = _approx_pauli_expectation_info(request.spec, "pauli_beam_approx")
    if max_terms is not None:
        info["pauli_beam_max_terms"] = max(1, int(max_terms))  # type: ignore[typeddict-unknown-key]
    return [_pauli_expectation_result(value, info, as_complex=as_complex) for value in native_values]


def _compute_native_mps_approx_pauli_expectations(
    request: _PauliExpectationRequest,
    *,
    as_complex: bool,
    max_bond: int | None = None,
) -> list[tuple[ScaledAmplitude | complex, ReductionInfo]] | None:
    native_values = _native_mps_approx_pauli_expectations(
        request.spec,
        request.input_bits,
        request.observables,
        max_bond=max_bond,
    )
    if native_values is None:
        return None
    info = _approx_pauli_expectation_info(request.spec, "native_mps_approx")
    if max_bond is not None:
        info["mps_max_bond"] = max(1, int(max_bond))  # type: ignore[typeddict-unknown-key]
    return [_pauli_expectation_result(value, info, as_complex=as_complex) for value in native_values]


def compute_circuit_pauli_expectations_approx(
    circuit: CircuitInput,
    observables: Sequence[str],
    *,
    input_bits: BitSequence | None = None,
    as_complex: bool = False,
    backend: str = "pauli_beam",
    max_terms: int | None = None,
    max_bond: int | None = None,
) -> list[tuple[ScaledAmplitude | complex, ReductionInfo]] | None:
    """Opt-in Pauli observable approximation entrypoint."""
    _refresh_engine_bindings()
    request = _prepare_pauli_expectation_request(circuit, observables, input_bits)
    if backend in {"pauli_beam", "pauli_beam_approx"}:
        return _compute_pauli_beam_approx_fast_path(request, as_complex=as_complex, max_terms=max_terms)
    if backend in {"native_mps", "native_mps_approx"}:
        return _compute_native_mps_approx_pauli_expectations(request, as_complex=as_complex, max_bond=max_bond)
    raise ValueError(f"Unsupported approximate Pauli backend {backend!r}.")


def _build_pauli_expectation_base_state(
    request: _PauliExpectationRequest,
    *,
    extended_reductions: ExtendedReductionMode | str,
) -> SchurState:
    return build_state(
        request.spec.n_qubits,
        request.spec.gates,
        request.input_bits,
        global_phase_radians=0.0,
        extended_reductions=extended_reductions,
    )


def _select_pauli_direct_replay_template(
    base_state: SchurState,
    request: _PauliExpectationRequest,
):
    direct_template = _build_direct_post_replay_template(
        base_state,
        request.inverse_gates,
        len(request.observables),
    )
    if direct_template is None:
        return None

    validation_observable = _build_direct_post_replay_validation_observable(request.observables)
    if validation_observable is None:
        return direct_template
    validation_state = _build_post_replay_state(
        base_state,
        _pauli_string_gates(validation_observable),
        request.inverse_gates,
    )
    validation_payload = _construct_direct_post_replay_payload(
        base_state,
        validation_observable,
        direct_template,
    )
    if not _direct_post_replay_payload_matches_state(
        validation_payload,
        validation_state,
        direct_template,
    ):
        return None
    return direct_template


def compute_circuit_pauli_expectations(
    circuit: CircuitInput,
    observables: Sequence[str],
    *,
    input_bits: BitSequence | None = None,
    as_complex: bool = False,
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
    solver_config: SolverConfig | None = None,
) -> list[tuple[ScaledAmplitude | complex, ReductionInfo]]:
    """
    Compute ``<x|U^† P U|x>`` for many Pauli strings while reusing reductions.

    Observables are interpreted in the normalized circuit's qubit order.
    """
    _refresh_engine_bindings()
    request = _prepare_pauli_expectation_request(circuit, observables, input_bits)
    spec = request.spec
    prepared_input = request.input_bits
    normalized_observables = request.observables

    _token = _set_solver_config(solver_config)
    try:
        allow_approximate = bool(_get_solver_config().allow_approximate)
        if allow_approximate and any(str(gate[0]) == "pauli_expbox" for gate in request.spec.gates):
            approximate = _compute_pauli_beam_approx_fast_path(request, as_complex=as_complex)
            if approximate is not None:
                return approximate
    finally:
        _reset_solver_config(_token)

    observable_gate_sets = request.observable_gate_sets
    inverse_gates = request.inverse_gates
    # Global phase cancels in U P U†, so keep the reusable U prefix phase-neutral.
    base_state = _build_pauli_expectation_base_state(request, extended_reductions=extended_reductions)

    _token = _set_solver_config(solver_config)
    try:
        allow_approximate = bool(_get_solver_config().allow_approximate)
        context = _ReductionContext(
            preserve_scale=not as_complex,
            allow_tensor_contraction=allow_tensor_contraction,
            extended_reductions=extended_reductions,
        )
        results: list[tuple[ScaledAmplitude | complex, ReductionInfo] | None] = [None] * len(normalized_observables)
        pending: list[tuple[int, complex, int, int, PhaseFunction]] = []
        bp_identity_cache: tuple[ScaledComplex, ReductionInfo] | None = None
        native_mps_approx_cache: dict[int, complex] | None | bool = False
        suffix_query_cache: dict[
            tuple[int, tuple[int, ...], tuple[int, ...]],
            tuple[EchelonCache, tuple[int, tuple[int, ...], tuple[int, ...], int] | None],
        ] = {}
        direct_template = _select_pauli_direct_replay_template(base_state, request)
        direct_solved_cache: dict[
            tuple[int, ...],
            tuple[int, tuple[int, ...], tuple[int, ...], int] | None,
        ] = {}

        def native_mps_approx_result(
            obs_idx: int,
            *,
            fallback_reason: str | None = None,
        ) -> tuple[ScaledAmplitude | complex, ReductionInfo] | None:
            nonlocal native_mps_approx_cache
            if native_mps_approx_cache is False:
                native_values = _native_mps_approx_pauli_expectations(
                    spec,
                    prepared_input,
                    normalized_observables,
                )
                native_mps_approx_cache = (
                    {idx: value for idx, value in enumerate(native_values)}
                    if native_values is not None
                    else None
                )
            if not isinstance(native_mps_approx_cache, dict) or obs_idx not in native_mps_approx_cache:
                return None
            candidate = native_mps_approx_cache[obs_idx]
            info = _approx_pauli_expectation_info(spec, "native_mps_approx")
            if fallback_reason is not None:
                info["fallback_reason"] = fallback_reason  # type: ignore[typeddict-unknown-key]
            return _pauli_expectation_result(candidate, info, as_complex=as_complex)

        for obs_idx, (observable, observable_gates) in enumerate(zip(normalized_observables, observable_gate_sets)):
            if direct_template is not None:
                eps0, scalar, scalar_half_pow2, q = _construct_direct_post_replay_payload(
                    base_state,
                    observable,
                    direct_template,
                )
                solved_key = tuple(int(bit) & 1 for bit in eps0)
                if solved_key in direct_solved_cache:
                    solved = direct_solved_cache[solved_key]
                else:
                    solved = _solve_output_from_echelon(
                        eps0,
                        direct_template.echelon_cache,
                        prepared_input,
                    )
                    direct_solved_cache[solved_key] = solved

                if solved is None:
                    results[obs_idx] = _zero_pauli_expectation_result(as_complex=as_complex)
                    continue

                shift_mask, _, gamma, initial_free = solved
                q_free = _aff_compose_cached(q, shift_mask, gamma, initial_free, context=context)
                pending.append((obs_idx, scalar, scalar_half_pow2, initial_free, q_free))
                continue

            state = _build_post_replay_state(base_state, observable_gates, inverse_gates)

            if state._arbitrary_phases:
                use_native_mps = allow_approximate and all(char in "IZ" for char in normalized_observables[obs_idx])
                if use_native_mps:
                    mps_result = native_mps_approx_result(obs_idx)
                    if mps_result is not None:
                        results[obs_idx] = mps_result
                        continue

                try:
                    raw_amp, info = state._amplitude_internal(
                        prepared_input,
                        preserve_scale=True,
                        allow_tensor_contraction=allow_tensor_contraction,
                        extended_reductions=extended_reductions,
                        allow_unbounded_bp_result=True,
                    )
                except RuntimeError:
                    mps_result = native_mps_approx_result(obs_idx, fallback_reason="arbitrary_bp_unavailable") if allow_approximate else None
                    if mps_result is None:
                        raise
                    results[obs_idx] = mps_result
                    continue
                assert isinstance(raw_amp, ScaledAmplitude)
                if _arbitrary_bp_backend(info.get("phase3_backend")):
                    amp_complex = None if raw_amp.log2_abs() > 1000.0 else _scaled_to_complex(raw_amp.as_tuple())
                    if bp_identity_cache is None:
                        identity_state = _build_post_replay_state(base_state, (), inverse_gates)
                        identity_amp_raw, identity_info = identity_state._amplitude_internal(
                            prepared_input,
                            preserve_scale=True,
                            allow_tensor_contraction=allow_tensor_contraction,
                            extended_reductions=extended_reductions,
                            allow_unbounded_bp_result=True,
                        )
                        assert isinstance(identity_amp_raw, ScaledAmplitude)
                        bp_identity_cache = (identity_amp_raw.as_tuple(), identity_info)
                    candidate = amp_complex
                    identity_scaled, _identity_info = bp_identity_cache
                    normalized = _scaled_complex_ratio_to_plain(raw_amp.as_tuple(), identity_scaled)
                    if normalized is not None:
                        candidate = normalized
                        info["phase3_backend"] = "arbitrary_bethe_bp_normalized"
                    if (
                        candidate is None
                        or not (math.isfinite(candidate.real) and math.isfinite(candidate.imag))
                        or abs(candidate.real) > 1.0 + 1e-6
                        or abs(candidate.imag) > 1e-6
                    ):
                        mps_result = native_mps_approx_result(obs_idx, fallback_reason="arbitrary_bp_invalid") if allow_approximate else None
                        if mps_result is None:
                            info["phase3_backend"] = "arbitrary_bethe_bp_invalid"
                            info["bp_invalid_reason"] = "observable_estimate_out_of_bounds"  # type: ignore[typeddict-unknown-key]
                            raise RuntimeError(
                                "Unreliable arbitrary-angle BP observable estimate: normalized Pauli expectation "
                                "is non-finite or outside [-1, 1]. Use an exact path or a fidelity-validated "
                                "approximate backend."
                            )
                        results[obs_idx] = mps_result
                        continue
                    amp = candidate if as_complex else ScaledAmplitude.from_tuple(_make_scaled_complex(candidate))
                else:
                    amp = _scaled_to_complex(raw_amp.as_tuple()) if as_complex else raw_amp
                results[obs_idx] = (amp, info)
                continue

            if state.m == 0:
                ok = all(state.eps0[idx] == prepared_input[idx] for idx in range(state.n))
                if ok:
                    scaled = _normalize_scaled_complex(
                        state.scalar * cmath.exp(2j * cmath.pi * float(state.q.q0)),
                        state.scalar_half_pow2,
                    )
                    results[obs_idx] = _scaled_pauli_expectation_result(
                        scaled,
                        _info(0, 0, 0, 0, 0, zero=False),
                        as_complex=as_complex,
                    )
                else:
                    results[obs_idx] = _zero_pauli_expectation_result(as_complex=as_complex)
                continue

            query_signature = (int(state.m), tuple(int(bit) for bit in state.eps0), tuple(int(mask) for mask in state.eps))
            cached_query = suffix_query_cache.get(query_signature)
            if cached_query is None:
                cache = state._prepare_echelon()
                solved = state._solve_for_output(cache, prepared_input)
                suffix_query_cache[query_signature] = (cache, solved)
            else:
                cache, solved = cached_query
            if solved is None:
                results[obs_idx] = _zero_pauli_expectation_result(as_complex=as_complex)
                continue

            shift_mask, _, gamma, initial_free = solved
            q_free = _aff_compose_cached(state.q, shift_mask, gamma, initial_free, context=context)
            pending.append((obs_idx, complex(state.scalar), int(state.scalar_half_pow2), initial_free, q_free))

        if pending:
            reduced_rows = _reduce_and_sum_scaled_batch(
                [q_free for _obs_idx, _scalar, _scalar_half_pow2, _initial_free, q_free in pending],
                context=context,
            )
            for (obs_idx, scalar, scalar_half_pow2, initial_free, _q_free), (reduced_total, elim_info) in zip(pending, reduced_rows):
                info = _info(
                    initial_free,
                    elim_info['quad'],
                    elim_info['constraint'],
                    elim_info['branched'],
                    elim_info['remaining'],
                    structural_obstruction=elim_info.get('structural_obstruction', elim_info['remaining']),
                    gauss_obstruction=elim_info.get(
                        'gauss_obstruction',
                        elim_info.get('structural_obstruction', elim_info['remaining']),
                    ),
                    phase_states=elim_info.get('phase_states', 0),
                    phase_splits=elim_info.get('phase_splits', 0),
                    cost_model_r=elim_info.get('cost_r', elim_info['remaining']),
                    phase3_backend=elim_info.get('phase3_backend'),
                )
                scaled_amp = _normalize_scaled_complex(
                    complex(scalar) * reduced_total[0],
                    reduced_total[1] + int(scalar_half_pow2),
                )
                results[obs_idx] = _scaled_pauli_expectation_result(scaled_amp, info, as_complex=as_complex)

        assert all(result is not None for result in results)
        return [result for result in results if result is not None]
    finally:
        _reset_solver_config(_token)


def analyze_amplitudes(
    circuit: CircuitInput,
    input_bits: BitSequence,
    output_list: Sequence[BitSequence],
    *,
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
    solver_config: SolverConfig | None = None,
) -> list[ReductionInfo]:
    """Return reduction metadata for multiple outputs without materializing amplitudes."""
    _refresh_engine_bindings()
    from .spec import _circuit_global_phase_radians, normalize_circuit

    spec = normalize_circuit(circuit)
    state = build_state(
        spec.n_qubits,
        spec.gates,
        input_bits,
        global_phase_radians=_circuit_global_phase_radians(spec),
        extended_reductions=extended_reductions,
    )
    _token = _set_solver_config(solver_config)
    try:
        return _batch_query_state(
            state,
            output_list,
            preserve_scale=True,
            allow_tensor_contraction=allow_tensor_contraction,
            extended_reductions=extended_reductions,
            analyze_only=True,
        )
    finally:
        _reset_solver_config(_token)


def analyze_circuit(
    circuit: CircuitInput,
    input_bits: BitSequence,
    output_bits: BitSequence,
    *,
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
    solver_config: SolverConfig | None = None,
) -> ReductionInfo:
    """Return reduction metadata for one amplitude query."""
    return analyze_amplitudes(
        circuit,
        input_bits,
        [output_bits],
        allow_tensor_contraction=allow_tensor_contraction,
        extended_reductions=extended_reductions,
        solver_config=solver_config,
    )[0]


@overload
def compute_amplitude(
    n: int,
    gates: Sequence[Gate],
    input_bits: BitSequence,
    output_bits: BitSequence,
    *,
    as_complex: Literal[False] = False,
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
) -> tuple[ScaledAmplitude, ReductionInfo]:
    ...


@overload
def compute_amplitude(
    n: int,
    gates: Sequence[Gate],
    input_bits: BitSequence,
    output_bits: BitSequence,
    *,
    as_complex: Literal[True],
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
) -> tuple[complex, ReductionInfo]:
    ...


def compute_amplitude(
    n: int,
    gates: Sequence[Gate],
    input_bits: BitSequence,
    output_bits: BitSequence,
    *,
    as_complex: bool = False,
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
) -> tuple[ScaledAmplitude | complex, ReductionInfo]:
    """
    Compute an amplitude from a normalized gate list.

    By default this returns ``ScaledAmplitude``. Pass ``as_complex=True`` to
    request a native ``complex`` instead. Set
    ``allow_tensor_contraction=False`` to disable tensor-guided q3-free
    planning hints. Approximate arbitrary-angle BP fallback is disabled unless
    ``SolverConfig.allow_approximate`` is true.
    """
    _refresh_engine_bindings()
    state = build_state(n, gates, input_bits, extended_reductions=extended_reductions)
    return state.amplitude(
        list(output_bits),
        as_complex=as_complex,
        allow_tensor_contraction=allow_tensor_contraction,
        extended_reductions=extended_reductions,
    )


def compute_amplitude_scaled(
    n: int,
    gates: Sequence[Gate],
    input_bits: BitSequence,
    output_bits: BitSequence,
    *,
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
) -> tuple[ScaledAmplitude, ReductionInfo]:
    """Compute an amplitude as ``ScaledAmplitude`` plus the usual reduction info."""
    return compute_amplitude(
        n,
        gates,
        input_bits,
        output_bits,
        as_complex=False,
        allow_tensor_contraction=allow_tensor_contraction,
        extended_reductions=extended_reductions,
    )

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
