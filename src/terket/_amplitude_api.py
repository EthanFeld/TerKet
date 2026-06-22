"""Extracted public amplitude and batch-query helpers."""

from __future__ import annotations

import bisect
import cmath
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import importlib
import heapq
from itertools import combinations
import math
import os
import struct
import sys
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Sequence, overload

import numpy as np

from .cubic_arithmetic import CubicFunction, PhaseFunction, detect_factorization
from ._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals
from ._approx_reliability import _copy_approx_reducer_info, _validate_approx_amplitude_reliability
from .interop.rewrite import _rewrite_gate_sequence
from .scaling import ScaledAmplitude, ScaledComplex
from .spec import CircuitSpec, Gate
from .state import BitSequence, CircuitInput, EchelonCache, ExtendedReductionMode, ReducerInfo, ReductionInfo, SolverConfig

_LOCAL_NAMES = {
    'affine_compose',
    'reduce_and_sum',
    'build_state',
    '_apply_gate_sequence_to_state_linear',
    '_subcircuit_qubit_slot',
    '_canonicalize_subcircuit_window',
    '_compiled_subcircuit_macro_source_line',
    '_compile_subcircuit_macro',
    '_build_subcircuit_macro_replay_plan',
    '_apply_gate_sequence_to_state',
    '_batch_query_state',
    'compute_circuit_amplitude',
    'compute_circuit_amplitude_scaled',
    'compute_amplitudes'
}


_MISSING = object()
_ENGINE_LOCAL_BASELINES = {}
_LOCAL_IMPLS = {}
_FORCE_ENGINE_BINDINGS_REFRESH = "pytest" in sys.modules


def _sync_from_engine(engine) -> None:
    _sync_extracted_globals(
        globals(),
        engine,
        local_names=_LOCAL_NAMES,
        local_impls=_LOCAL_IMPLS,
        baselines=_ENGINE_LOCAL_BASELINES,
        missing=_MISSING,
        respect_mock_wraps=True,
    )


_MISSING, _ENGINE_LOCAL_BASELINES = _bootstrap_extracted_globals(
    globals(),
    local_names=_LOCAL_NAMES,
    local_impls=_LOCAL_IMPLS,
    respect_mock_wraps=True,
)


def _refresh_engine_bindings() -> None:
    if not _FORCE_ENGINE_BINDINGS_REFRESH:
        return
    _sync_from_engine(importlib.import_module("terket._engine_impl"))


def affine_compose(
    q: PhaseFunction,
    shift: int | BitSequence,
    gamma: Sequence[int] | AffineRows,
    k: int,
) -> PhaseFunction:
    """Public affine restriction helper for TerKet phase functions."""
    return _aff_compose(q, shift, gamma, k)


def reduce_and_sum(
    q: PhaseFunction,
    *,
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
) -> tuple[complex, ReducerInfo]:
    """Public exact reducer for cubic phase functions over Z2."""
    context = _ReductionContext(
        allow_tensor_contraction=allow_tensor_contraction,
        extended_reductions=extended_reductions,
    )
    return _reduce_and_sum(q, context=context)


def build_state(
    n: int,
    gates: Sequence[Gate],
    input_bits: BitSequence | None = None,
    *,
    global_phase_radians: float = 0.0,
    extended_reductions: ExtendedReductionMode | str = "auto",
) -> SchurState:
    """Construct a ``SchurState`` from a gate list and optional input state."""
    _refresh_engine_bindings()
    state = SchurState(n)
    if input_bits is not None:
        state.eps0 = list(input_bits)
    if global_phase_radians:
        state.scalar *= cmath.exp(1j * float(global_phase_radians))
    gate_sequence = gates if isinstance(gates, tuple) else tuple(gates)
    if _should_apply_extended_gate_rewrite(extended_reductions, gate_sequence):
        cached_gate_sequence = _REWRITTEN_GATE_SEQUENCE_CACHE.get(gate_sequence)
        if cached_gate_sequence is None:
            cached_gate_sequence = _rewrite_gate_sequence(gate_sequence)
            _REWRITTEN_GATE_SEQUENCE_CACHE[gate_sequence] = cached_gate_sequence
        gate_sequence = cached_gate_sequence
    defer_early_elim = any(str(gate[0]) == "pauli_expbox" for gate in gate_sequence)
    if defer_early_elim:
        state._defer_early_elim = True
    try:
        _apply_gate_sequence_to_state(state, gate_sequence)
    finally:
        state._defer_early_elim = False
    state._flush_pending_dead_variables()
    return state


_SUBCIRCUIT_MACRO_MIN_TOTAL_GATES = 128
_SUBCIRCUIT_MACRO_WINDOW_LENGTHS = (32, 24, 16, 12, 8, 6, 5, 4)
_SUBCIRCUIT_MACRO_SEEN_CACHE_SIZE = 1 << 14
_SUBCIRCUIT_MACRO_COMPILED_CACHE_SIZE = 1 << 11
_SUBCIRCUIT_MACRO_PLAN_CACHE_SIZE = 64
_REWRITTEN_GATE_SEQUENCE_CACHE_SIZE = 256
_STATE_BUILD_CACHE_SIZE = 64
_SUBCIRCUIT_MACRO_PLAN_STEP = "__subcircuit_macro__"

_SubcircuitMacro = Callable[[SchurState, tuple[int, ...]], None]
_SUBCIRCUIT_MACRO_PLAN_PENDING = object()
_NO_SUBCIRCUIT_MACRO_PLAN = object()
_SUBCIRCUIT_MACRO_PLAN_CACHE = _engine_cache("subcircuit_macro.plan", _SUBCIRCUIT_MACRO_PLAN_CACHE_SIZE)
_REWRITTEN_GATE_SEQUENCE_CACHE = _engine_cache(
    "amplitude_api.rewritten_gate_sequence",
    _REWRITTEN_GATE_SEQUENCE_CACHE_SIZE,
)
_STATE_BUILD_CACHE = _engine_cache(
    "amplitude_api.state_build",
    _STATE_BUILD_CACHE_SIZE,
)


def _build_state_cache_key(
    n: int,
    gates: Sequence[Gate],
    input_bits: BitSequence | None,
    *,
    global_phase_radians: float,
    extended_reductions: ExtendedReductionMode | str,
) -> tuple[Any, ...]:
    gate_sequence = gates if isinstance(gates, tuple) else tuple(gates)
    normalized_input = None if input_bits is None else tuple(int(bit) for bit in input_bits)
    return (
        int(n),
        gate_sequence,
        normalized_input,
        float(global_phase_radians),
        _normalize_extended_reductions(extended_reductions),
    )


def _build_cached_state(
    n: int,
    gates: Sequence[Gate],
    input_bits: BitSequence | None = None,
    *,
    global_phase_radians: float = 0.0,
    extended_reductions: ExtendedReductionMode | str = "auto",
) -> SchurState:
    cache_key = _build_state_cache_key(
        n,
        gates,
        input_bits,
        global_phase_radians=global_phase_radians,
        extended_reductions=extended_reductions,
    )
    cached = _STATE_BUILD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    state = build_state(
        n,
        gates,
        input_bits,
        global_phase_radians=global_phase_radians,
        extended_reductions=extended_reductions,
    )
    _STATE_BUILD_CACHE[cache_key] = state
    return state


def _apply_gate_sequence_to_state_linear(
    state: SchurState,
    gates: Sequence[Gate],
) -> None:
    if not gates:
        return
    if _FORCE_ENGINE_BINDINGS_REFRESH:
        engine = importlib.import_module("terket._engine_impl")
        importlib.import_module("terket._state_runtime")._sync_from_engine(engine)
        importlib.import_module("terket._pauli_support")._sync_from_engine(engine)
    gate_ops = {name: getattr(state, name) for name in {gate[0] for gate in gates}}
    for gate in gates:
        gate_ops[gate[0]](*gate[1:])


def _subcircuit_qubit_slot(
    actual_qubit: int,
    slot_for_qubit: dict[int, int],
    qubits: list[int],
) -> int:
    slot = slot_for_qubit.get(actual_qubit)
    if slot is not None:
        return slot
    slot = len(qubits)
    slot_for_qubit[actual_qubit] = slot
    qubits.append(actual_qubit)
    return slot


def _canonicalize_subcircuit_window(
    gates: Sequence[Gate],
    start: int,
    length: int,
) -> tuple[tuple[Gate, ...], tuple[int, ...]]:
    template: list[Gate] = []
    qubits: list[int] = []
    slot_for_qubit: dict[int, int] = {}
    stop = start + length
    for idx in range(start, stop):
        gate = gates[idx]
        name = str(gate[0])
        if name == "rz_dyadic":
            qubit = _subcircuit_qubit_slot(int(gate[1]), slot_for_qubit, qubits)
            template.append((name, qubit, int(gate[2]), int(gate[3])))
            continue
        if name == "rz_arbitrary":
            qubit = _subcircuit_qubit_slot(int(gate[1]), slot_for_qubit, qubits)
            template.append((name, qubit, float(gate[2])))
            continue
        if name == "rzz_dyadic":
            q0 = _subcircuit_qubit_slot(int(gate[1]), slot_for_qubit, qubits)
            q1 = _subcircuit_qubit_slot(int(gate[2]), slot_for_qubit, qubits)
            template.append((name, q0, q1, int(gate[3]), int(gate[4])))
            continue
        if len(gate) == 2:
            qubit = _subcircuit_qubit_slot(int(gate[1]), slot_for_qubit, qubits)
            template.append((name, qubit))
            continue
        q0 = _subcircuit_qubit_slot(int(gate[1]), slot_for_qubit, qubits)
        q1 = _subcircuit_qubit_slot(int(gate[2]), slot_for_qubit, qubits)
        template.append((name, q0, q1))
    return tuple(template), tuple(qubits)


def _compiled_subcircuit_macro_source_line(
    gate: Gate,
    op_alias: str,
) -> str:
    name = str(gate[0])
    if name == "rz_dyadic":
        return f"    {op_alias}(q{int(gate[1])}, {int(gate[2])}, {int(gate[3])})"
    if name == "rz_arbitrary":
        return f"    {op_alias}(q{int(gate[1])}, {float(gate[2])!r})"
    if name == "rzz_dyadic":
        return f"    {op_alias}(q{int(gate[1])}, q{int(gate[2])}, {int(gate[3])}, {int(gate[4])})"
    args = ", ".join(f"q{int(qubit)}" for qubit in gate[1:])
    return f"    {op_alias}({args})"


def _compile_subcircuit_macro(template: tuple[Gate, ...]) -> _SubcircuitMacro:
    qubit_count = 0
    for gate in template:
        name = str(gate[0])
        if name == "rz_dyadic" or name == "rz_arbitrary":
            qubit_count = max(qubit_count, int(gate[1]) + 1)
        elif name == "rzz_dyadic":
            qubit_count = max(qubit_count, int(gate[1]) + 1, int(gate[2]) + 1)
        else:
            qubit_count = max(qubit_count, *(int(qubit) + 1 for qubit in gate[1:]))

    gate_names = tuple(dict.fromkeys(str(gate[0]) for gate in template))
    op_aliases = {name: f"_op_{idx}" for idx, name in enumerate(gate_names)}
    lines = ["def _macro(state, qubits):"]
    for qubit in range(qubit_count):
        lines.append(f"    q{qubit} = qubits[{qubit}]")
    for name in gate_names:
        lines.append(f"    {op_aliases[name]} = state.{name}")
    for gate in template:
        lines.append(_compiled_subcircuit_macro_source_line(gate, op_aliases[str(gate[0])]))
    namespace: dict[str, Any] = {}
    exec("\n".join(lines), {}, namespace)
    return namespace["_macro"]


def _build_subcircuit_macro_replay_plan(
    gates: Sequence[Gate],
) -> tuple[tuple[Any, ...], ...] | None:
    seen_templates = _BoundedMemoCache(_SUBCIRCUIT_MACRO_SEEN_CACHE_SIZE)
    compiled_templates = _BoundedMemoCache(_SUBCIRCUIT_MACRO_COMPILED_CACHE_SIZE)
    plan: list[tuple[Any, ...]] = []
    macro_count = 0
    idx = 0
    gate_count = len(gates)
    while idx < gate_count:
        macro_applied = False
        for window_length in _SUBCIRCUIT_MACRO_WINDOW_LENGTHS:
            if idx + window_length > gate_count:
                continue
            template, qubits = _canonicalize_subcircuit_window(gates, idx, window_length)
            macro = compiled_templates.get(template)
            if macro is None and seen_templates.get(template):
                macro = _compile_subcircuit_macro(template)
                compiled_templates[template] = macro
            if macro is not None:
                plan.append((_SUBCIRCUIT_MACRO_PLAN_STEP, macro, qubits))
                idx += window_length
                macro_count += 1
                macro_applied = True
                break
            seen_templates[template] = True
        if macro_applied:
            continue
        plan.append(gates[idx])
        idx += 1
    return tuple(plan) if macro_count else None


def _apply_gate_sequence_to_state(
    state: SchurState,
    gates: Sequence[Gate],
) -> None:
    if not gates:
        return
    if len(gates) < _SUBCIRCUIT_MACRO_MIN_TOTAL_GATES:
        _apply_gate_sequence_to_state_linear(state, gates)
        return
    if any(str(gate[0]) == "pauli_expbox" for gate in gates):
        _apply_gate_sequence_to_state_linear(state, gates)
        return

    gate_sequence = gates if isinstance(gates, tuple) else tuple(gates)
    cached_plan = _SUBCIRCUIT_MACRO_PLAN_CACHE.get(gate_sequence)
    if cached_plan is None:
        _SUBCIRCUIT_MACRO_PLAN_CACHE[gate_sequence] = _SUBCIRCUIT_MACRO_PLAN_PENDING
        _apply_gate_sequence_to_state_linear(state, gate_sequence)
        return
    if cached_plan is _SUBCIRCUIT_MACRO_PLAN_PENDING:
        cached_plan = _build_subcircuit_macro_replay_plan(gate_sequence)
        _SUBCIRCUIT_MACRO_PLAN_CACHE[gate_sequence] = (
            _NO_SUBCIRCUIT_MACRO_PLAN if cached_plan is None else cached_plan
        )
    elif cached_plan is _NO_SUBCIRCUIT_MACRO_PLAN:
        _apply_gate_sequence_to_state_linear(state, gate_sequence)
        return

    if cached_plan is None:
        _apply_gate_sequence_to_state_linear(state, gate_sequence)
        return

    gate_ops = {name: getattr(state, name) for name in {gate[0] for gate in gate_sequence}}
    for step in cached_plan:
        if step[0] == _SUBCIRCUIT_MACRO_PLAN_STEP:
            step[1](state, step[2])
            continue
        gate_ops[step[0]](*step[1:])


def _batch_query_state(
    state: SchurState,
    output_list: Sequence[BitSequence],
    *,
    preserve_scale: bool = False,
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
    analyze_only: bool = False,
    context: _ReductionContext | None = None,
    constraint_cache: EchelonCache | None = None,
):
    if state._arbitrary_phases:
        results = []
        for output_bits in output_list:
            amplitude, info = state._amplitude_internal(
                output_bits,
                preserve_scale=True if analyze_only else preserve_scale,
                allow_tensor_contraction=allow_tensor_contraction,
                extended_reductions=extended_reductions,
            )
            results.append(info if analyze_only else (amplitude, info))
        return results

    if len(output_list) > 1 and not state.q.q3 and state.m:
        for output_bits in output_list:
            if len(output_bits) != state.n:
                raise ValueError(f"Expected {state.n} output bits, received {len(output_bits)}.")

        cache = state._prepare_constraint_echelon()
        plan = _build_q3_free_raw_constraint_plan(
            state,
            allow_tensor_contraction=allow_tensor_contraction,
            prefer_reusable_decomposition=True,
        )
        restricted_plan = _restrict_q3_free_raw_constraint_plan(plan, state.n)
        reduced_totals = _evaluate_q3_free_raw_constraint_plan_scaled_batch(
            plan,
            restricted_plan,
            output_list,
        )
        gauss_obstruction = _gauss_obstruction(state.q, 0)
        phase3_backend = _q3_free_phase3_backend_name(state.q)
        results = []
        for reduced_total in reduced_totals:
            scaled_amp = _normalize_scaled_complex(
                complex(state.scalar) * reduced_total[0],
                reduced_total[1] + state.scalar_half_pow2,
            )
            info = _info(
                cache.n_free,
                0,
                0,
                0,
                0,
                structural_obstruction=0,
                gauss_obstruction=gauss_obstruction,
                phase_states=0,
                phase_splits=0,
                zero=scaled_amp[0] == 0j,
                cost_model_r=0,
                phase3_backend=phase3_backend,
            )
            if analyze_only:
                results.append(info)
            else:
                amp = ScaledAmplitude.from_tuple(scaled_amp) if preserve_scale else _scaled_to_complex(scaled_amp)
                results.append((amp, info))
        return results

    cache = None if not state.m else constraint_cache
    if state.m and cache is None:
        cache = state._prepare_echelon()
    if context is None:
        context = _ReductionContext(
            preserve_scale=preserve_scale,
            allow_tensor_contraction=allow_tensor_contraction,
            extended_reductions=extended_reductions,
        )
    results = []
    native_solved = None
    pending_queries: list[tuple[int, int, PhaseFunction]] = []
    if state.m and cache is not None:
        native_solved = _native_solve_for_output_batch(state.eps0, cache, output_list)

    for output_idx, output_bits in enumerate(output_list):
        if len(output_bits) != state.n:
            raise ValueError(f"Expected {state.n} output bits, received {len(output_bits)}.")

        if state.m == 0:
            ok = all(state.eps0[idx] == output_bits[idx] for idx in range(state.n))
            info = _info(0, 0, 0, 0, 0, zero=not ok)
            if analyze_only:
                results.append(info)
                continue
            scaled = (
                _normalize_scaled_complex(
                    state.scalar * cmath.exp(2j * cmath.pi * float(state.q.q0)),
                    state.scalar_half_pow2,
                )
                if ok
                else _make_scaled_complex(0j)
            )
            amp = ScaledAmplitude.from_tuple(scaled) if preserve_scale else _scaled_to_complex(scaled)
            results.append((amp, info))
            continue

        assert cache is not None
        if native_solved is not None:
            native_shift_mask = native_solved[output_idx]
            solved = (
                None
                if native_shift_mask is None
                else (native_shift_mask, cache.free_vars, cache.gamma_masks, cache.n_free)
            )
        else:
            solved = state._solve_for_output(cache, output_bits)
        if solved is None:
            info = _info(0, 0, 0, 0, 0, zero=True)
            if analyze_only:
                results.append(info)
            else:
                zero_scaled = _make_scaled_complex(0j)
                amp = ScaledAmplitude.from_tuple(zero_scaled) if preserve_scale else 0j
                results.append((amp, info))
            continue

        shift_mask, _, gamma, initial_free = solved
        q_free = _aff_compose_cached(state.q, shift_mask, gamma, initial_free, context=context)
        pending_queries.append((len(results), initial_free, q_free))
        results.append(None)

    if pending_queries:
        reduced_rows = _reduce_and_sum_scaled_batch(
            [q_free for _result_idx, _initial_free, q_free in pending_queries],
            context=context,
        )
        for (result_idx, initial_free, _q_free), (reduced_total, elim_info) in zip(pending_queries, reduced_rows):
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
            _copy_approx_reducer_info(info, elim_info)
            from ._q3free.approx_guard import _get_q3_free_approx_diagnostics

            approx_info = _get_q3_free_approx_diagnostics()
            if approx_info is not None:
                _copy_approx_reducer_info(info, approx_info)
            if analyze_only:
                results[result_idx] = info
                continue

            scaled_amp = _normalize_scaled_complex(
                complex(state.scalar) * reduced_total[0],
                reduced_total[1] + state.scalar_half_pow2,
            )
            _validate_approx_amplitude_reliability(scaled_amp, info)
            amp = ScaledAmplitude.from_tuple(scaled_amp) if preserve_scale else _scaled_to_complex(scaled_amp)
            results[result_idx] = (amp, info)

    assert all(result is not None for result in results)
    return [result for result in results if result is not None]


@overload
def compute_circuit_amplitude(
    circuit: CircuitInput,
    input_bits: BitSequence,
    output_bits: BitSequence,
    *,
    as_complex: Literal[False] = False,
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
    solver_config: SolverConfig | None = None,
) -> tuple[ScaledAmplitude, ReductionInfo]:
    ...


@overload
def compute_circuit_amplitude(
    circuit: CircuitInput,
    input_bits: BitSequence,
    output_bits: BitSequence,
    *,
    as_complex: Literal[True],
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
    solver_config: SolverConfig | None = None,
) -> tuple[complex, ReductionInfo]:
    ...


def compute_circuit_amplitude(
    circuit: CircuitInput,
    input_bits: BitSequence,
    output_bits: BitSequence,
    *,
    as_complex: bool = False,
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
    solver_config: SolverConfig | None = None,
) -> tuple[ScaledAmplitude | complex, ReductionInfo]:
    """Compute one amplitude from a gate list, QASM string, or Qiskit circuit."""
    _refresh_engine_bindings()
    from .spec import _circuit_global_phase_radians, normalize_circuit

    spec = normalize_circuit(circuit)
    state = _build_cached_state(
        spec.n_qubits,
        spec.gates,
        input_bits,
        global_phase_radians=_circuit_global_phase_radians(spec),
        extended_reductions=extended_reductions,
    )
    return state.amplitude(
        list(output_bits),
        as_complex=as_complex,
        allow_tensor_contraction=allow_tensor_contraction,
        extended_reductions=extended_reductions,
        solver_config=solver_config,
    )


def compute_circuit_amplitude_scaled(
    circuit: CircuitInput,
    input_bits: BitSequence,
    output_bits: BitSequence,
    *,
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
    solver_config: SolverConfig | None = None,
) -> tuple[ScaledAmplitude, ReductionInfo]:
    """Compute an amplitude in scaled form without collapsing tiny values to zero."""
    _refresh_engine_bindings()
    return compute_circuit_amplitude(
        circuit,
        input_bits,
        output_bits,
        as_complex=False,
        allow_tensor_contraction=allow_tensor_contraction,
        extended_reductions=extended_reductions,
        solver_config=solver_config,
    )


def compute_amplitudes(
    circuit: CircuitInput,
    input_bits: BitSequence,
    output_list: Sequence[BitSequence],
    *,
    as_complex: bool = False,
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
    solver_config: SolverConfig | None = None,
) -> list[tuple[ScaledAmplitude | complex, ReductionInfo]]:
    """Compute amplitudes for many outputs while reusing Schur-state work."""
    _refresh_engine_bindings()
    from .spec import _circuit_global_phase_radians, normalize_circuit

    spec = normalize_circuit(circuit)
    state = _build_cached_state(
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
            preserve_scale=not as_complex,
            allow_tensor_contraction=allow_tensor_contraction,
            extended_reductions=extended_reductions,
            analyze_only=False,
        )
    finally:
        _reset_solver_config(_token)


_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
