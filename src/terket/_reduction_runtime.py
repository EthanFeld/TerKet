"""Extracted helpers for _reduction_runtime.py."""

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
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Sequence, overload

import numpy as np

from .cubic_arithmetic import CubicFunction, PhaseFunction, detect_factorization
from ._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals
from ._reduction_high_precision import (
    _apply_safe_high_precision_eliminations,
    _elim_decoupled_constraints_batch,
)
from .scaling import ScaledAmplitude, ScaledComplex
from .spec import CircuitSpec, Gate
from .state import BitSequence, CircuitInput, EchelonCache, ExtendedReductionMode, ReducerInfo, ReductionInfo, SolverConfig

_LOCAL_NAMES = {
    '_pre_exact_phase3_treewidth_escape',
    '_reduce_and_sum_scaled',
    '_reduce_and_sum',
    '_reduce_and_sum_scaled_batch',
    '_invert_native_gate',
    '_invert_native_gates',
    '_fork_state_for_extension',
    '_pauli_string_gates',
    '_validate_pauli_observables',
    '_elim_decoupled_constraints_batch',
    '_apply_safe_high_precision_eliminations',
    '_apply_exact_eliminations',
    '_product_q1_sum',
    '_product_q1_sum_scaled'
}


_MISSING = object()
_ENGINE_LOCAL_BASELINES = {}
_LOCAL_IMPLS = {}


def _sync_from_engine(engine) -> None:
    _sync_extracted_globals(
        globals(),
        engine,
        local_names=_LOCAL_NAMES,
        local_impls=_LOCAL_IMPLS,
        baselines=_ENGINE_LOCAL_BASELINES,
        missing=_MISSING,
    )


_INITIAL_ENGINE = importlib.import_module("terket._engine_impl")
_MISSING, _ENGINE_LOCAL_BASELINES = _bootstrap_extracted_globals(
    globals(),
    local_names=_LOCAL_NAMES,
    local_impls=_LOCAL_IMPLS,
)
del _INITIAL_ENGINE


def _pre_exact_phase3_treewidth_escape(
    q,
    *,
    allow_tensor_contraction: bool,
):
    """Return a direct Phase-3 treewidth plan worth taking before exact elim."""
    if not q.q3 or q.n < _PRE_EXACT_PHASE3_TREEWIDTH_ESCAPE_MIN_VARS:
        return None

    q3_terms = sum(1 for coeff in q.q3.values() if coeff % q.mod_q3)
    if q3_terms == 0 or q3_terms > _PRE_EXACT_PHASE3_TREEWIDTH_ESCAPE_MAX_Q3_TERMS:
        return None

    active_q3 = _active_q3_variables(q)
    if active_q3 and len(active_q3) * 4 > q.n:
        return None

    cover, order, width, structural_obstruction, fully_peeled = _phase3_support_plan(q)
    treewidth_work = max(1, int(_estimate_treewidth_dp_work(q, order)))
    backend = _select_direct_phase3_backend(
        q,
        cover,
        order,
        width,
        allow_tensor_contraction=allow_tensor_contraction,
        fully_peeled=fully_peeled,
        treewidth_work=treewidth_work,
    )
    if backend not in {"treewidth_dp", "treewidth_dp_peeled"}:
        return None
    if (
        len(cover) > _PRE_EXACT_PHASE3_TREEWIDTH_ESCAPE_MAX_COVER
        or structural_obstruction > _PRE_EXACT_PHASE3_TREEWIDTH_ESCAPE_MAX_COVER
    ):
        return None

    if treewidth_work > _PRE_EXACT_PHASE3_TREEWIDTH_ESCAPE_MAX_WORK:
        return None

    return cover, order, width, structural_obstruction, backend

def _make_reducer_info(
    *,
    quad: int = 0,
    constraint: int = 0,
    branched: int = 0,
    remaining: int = 0,
    structural_obstruction: int | None = None,
    gauss_obstruction: int | None = None,
    cost_r: int | None = None,
    phase_states: int = 0,
    phase_splits: int = 0,
    phase3_backend=None,
) -> ReducerInfo:
    remaining = int(remaining)
    if structural_obstruction is None:
        structural_obstruction = remaining
    structural_obstruction = int(structural_obstruction)
    if gauss_obstruction is None:
        gauss_obstruction = structural_obstruction
    if cost_r is None:
        cost_r = remaining
    return {
        'quad': int(quad),
        'constraint': int(constraint),
        'branched': int(branched),
        'remaining': remaining,
        'structural_obstruction': structural_obstruction,
        'gauss_obstruction': int(gauss_obstruction),
        'cost_r': int(cost_r),
        'phase_states': int(phase_states),
        'phase_splits': int(phase_splits),
        'phase3_backend': phase3_backend,
    }

def _offset_reducer_info(
    info: ReducerInfo,
    *,
    quad: int = 0,
    constraint: int = 0,
    branched: int = 0,
) -> ReducerInfo:
    remaining = info['remaining']
    structural_obstruction = info.get('structural_obstruction', remaining)
    return _make_reducer_info(
        quad=quad + info['quad'],
        constraint=constraint + info['constraint'],
        branched=branched + info['branched'],
        remaining=remaining,
        structural_obstruction=structural_obstruction,
        gauss_obstruction=info.get('gauss_obstruction', structural_obstruction),
        cost_r=info.get('cost_r', remaining),
        phase_states=info.get('phase_states', 0),
        phase_splits=info.get('phase_splits', 0),
        phase3_backend=info.get('phase3_backend'),
    )

def _sum_q3_free_exact_scaled(
    q,
    *,
    context: _ReductionContext,
    quad: int = 0,
    constraint: int = 0,
) -> tuple[ScaledComplex, ReducerInfo]:
    if not context.preserve_scale:
        total_complex, phase_info = _gauss_sum_q3_free(
            q,
            allow_tensor_contraction=context.allow_tensor_contraction,
        )
        total = _make_scaled_complex(total_complex)
    else:
        total, phase_info = _gauss_sum_q3_free_scaled(
            q,
            allow_tensor_contraction=context.allow_tensor_contraction,
        )
    return total, _make_reducer_info(
        quad=quad,
        constraint=constraint,
        gauss_obstruction=_gauss_obstruction(q, 0),
        phase_states=phase_info['phase_states'],
        phase_splits=phase_info['phase_splits'],
        phase3_backend=_q3_free_phase3_backend_name(q),
    )

def _reduce_and_sum_scaled(q, context=None):
    """Reduce, recurse, and exactly sum the remaining kernel."""
    if context is None:
        context = _ReductionContext()
    extended_reductions = context.extended_reductions

    cache_key = _q_key(q)
    cached = context.reduce_cache.get(cache_key)
    if cached is not None:
        return cached[0], dict(cached[1])

    allow_tensor_contraction = context.allow_tensor_contraction

    # Level-1 phases do not satisfy the Clifford+T single-variable
    # elimination identities. Generic factor elimination remains exact.
    if q.level == 1:
        total, info = _sum_irreducible_cubic_core(
            q,
            context=context,
            allow_tensor_contraction=allow_tensor_contraction,
        )
        context.reduce_cache[cache_key] = (total, dict(info))
        return total, info

    # Above Clifford+T precision, general parity/Gauss substitutions can leave
    # cubic form. First apply the safe low-arity subset, then use generic exact
    # backends for the residual.
    if q.level > 3:
        reduced_q, scale_half_pow2, exact_info = _apply_safe_high_precision_eliminations(q)
        if reduced_q is None:
            total = _make_scaled_complex(0j)
            info = _make_reducer_info(constraint=exact_info['constraint'])
            context.reduce_cache[cache_key] = (total, dict(info))
            return total, info
        if reduced_q.q3:
            total, info = _sum_irreducible_cubic_core(
                reduced_q,
                context=context,
                allow_tensor_contraction=allow_tensor_contraction,
            )
        else:
            total, info = _sum_q3_free_exact_scaled(reduced_q, context=context)
        result = _scale_scaled_complex(total, scale_half_pow2)
        info = _offset_reducer_info(info, constraint=exact_info['constraint'])
        context.reduce_cache[cache_key] = (result, dict(info))
        return result, info

    pre_exact_phase3 = _pre_exact_phase3_treewidth_escape(
        q,
        allow_tensor_contraction=allow_tensor_contraction,
    )
    if pre_exact_phase3 is not None:
        (
            phase3_cover,
            phase3_order,
            phase3_width,
            phase3_structural_obstruction,
            direct_phase3_backend,
        ) = pre_exact_phase3
        phase3_total, phase3_info = _sum_irreducible_cubic_core(
            q,
            context=context,
            cover=phase3_cover,
            order=phase3_order,
            width=phase3_width,
            structural_obstruction=phase3_structural_obstruction,
            backend=direct_phase3_backend,
            allow_tensor_contraction=allow_tensor_contraction,
        )
        context.reduce_cache[cache_key] = (phase3_total, dict(phase3_info))
        return phase3_total, phase3_info

    q, scale_half_pow2, exact_info, blocked_quadratics = _apply_exact_eliminations(q, context=context)
    nq = exact_info['quad']
    nc = exact_info['constraint']
    nb = 0
    if q is None:
        zero = _make_scaled_complex(0j)
        info = _make_reducer_info(quad=nq, constraint=nc)
        context.reduce_cache[cache_key] = (zero, dict(info))
        return zero, info

    if q.n == 0:
        total = _scale_scaled_complex(
            _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q.q0))),
            scale_half_pow2,
        )
        info = _make_reducer_info(quad=nq, constraint=nc)
        context.reduce_cache[cache_key] = (total, dict(info))
        return total, info

    enable_extended_q3_reductions = _should_apply_extended_q3_reductions(q, extended_reductions)
    baseline_phase3_runtime_score = None
    if q.q3 and enable_extended_q3_reductions:
        baseline_phase3_runtime_score = _phase3_execution_plan_runtime_score(
            q,
            allow_tensor_contraction=allow_tensor_contraction,
        )
        optimized_q = q
        optimized = False
        if not _phase3_runtime_score_is_good_baseline(baseline_phase3_runtime_score):
            candidate_q, candidate_changed = _optimize_phase_function_structure(q, context=context)
            if candidate_changed:
                candidate_runtime_score = _phase3_execution_plan_runtime_score(
                    candidate_q,
                    allow_tensor_contraction=allow_tensor_contraction,
                )
                if candidate_runtime_score < baseline_phase3_runtime_score:
                    optimized_q = candidate_q
                    optimized = True
        if optimized:
            optimized_total, optimized_info = _reduce_and_sum_scaled(optimized_q, context=context)
            result = _scale_scaled_complex(optimized_total, scale_half_pow2)
            info = _offset_reducer_info(optimized_info, quad=nq, constraint=nc)
            context.reduce_cache[cache_key] = (result, dict(info))
            return result, info

    if q.q3 and enable_extended_q3_reductions:
        components = detect_factorization(q)
        if len(components) > 1:
            factorized_total, factorized_info = _sum_factorized_components_scaled(q, components, context=context)
            result = _scale_scaled_complex(factorized_total, scale_half_pow2)
            info = _offset_reducer_info(factorized_info, quad=nq, constraint=nc)
            context.reduce_cache[cache_key] = (result, dict(info))
            return result, info

    if not q.q3:
        # Every q3-free kernel is summed exactly here; the earlier "quadratic"
        # tag only describes the single-variable elimination rule.
        total, info = _sum_q3_free_exact_scaled(
            q,
            context=context,
            quad=nq,
            constraint=nc,
        )
        result = _scale_scaled_complex(total, scale_half_pow2)
        context.reduce_cache[cache_key] = (result, dict(info))
        return result, info

    phase3_cover = None
    phase3_order = None
    phase3_width = None
    phase3_structural_obstruction = None
    direct_phase3_backend = None
    if q.n >= _PHASE2_TREEWIDTH_ESCAPE_MIN_VARS or q.n <= _Q3_TENSOR_CONTRACTION_MAX_VARS:
        phase3_plan = _phase3_plan(
            q,
            allow_tensor_contraction=allow_tensor_contraction,
        )
        if len(phase3_plan) == 4:
            phase3_cover, phase3_order, phase3_width, direct_phase3_backend = phase3_plan
            phase3_structural_obstruction = q.n
        else:
            (
                phase3_cover,
                phase3_order,
                phase3_width,
                phase3_structural_obstruction,
                direct_phase3_backend,
            ) = phase3_plan
    prefer_direct_phase3 = direct_phase3_backend is not None

    if q.n > 0 and not prefer_direct_phase3:
        classification_data = _build_classification_data(q)
        threshold = max(1, q.mod_q1 // 4)
        best_var, best_unlocks = -1, -1
        for var in range(q.n):
            if _classification_entry(
                q,
                var,
                classification_data=classification_data,
                threshold=threshold,
            )[0] != _CLASS_CUBIC:
                continue
            if q.q1[var] % 2 == 0:
                continue
            unlocks = 0
            for j in range(q.n):
                if j == var:
                    continue
                if q.q1[j] % 2 != 0:
                    continue
                key = (min(var,j), max(var,j))
                if q.q2.get(key, 0) % 2 != 0:
                    unlocks += 1
            if unlocks > best_unlocks:
                best_var, best_unlocks = var, unlocks

        if best_var >= 0 and best_unlocks > 0:
            total = _make_scaled_complex(0j)
            max_remaining = 0
            max_structural = 0
            max_gauss = 0
            max_cost_r = 0
            max_branched = 0
            phase_states = phase_splits = 0
            phase3_backend = None
            phase3_backend_cost_r = -1
            for fval in [0, 1]:
                q_branch = _fix_variable(q, best_var, fval, context=context)
                branch_result, branch_info = _reduce_and_sum_scaled(q_branch, context=context)
                total = _add_scaled_complex(total, branch_result)
                nq += branch_info['quad']
                nc += branch_info['constraint']
                max_branched = max(max_branched, branch_info['branched'])
                max_remaining = max(max_remaining, branch_info['remaining'])
                max_structural = max(
                    max_structural,
                    branch_info.get('structural_obstruction', branch_info['remaining']),
                )
                max_gauss = max(
                    max_gauss,
                    branch_info.get(
                        'gauss_obstruction',
                        branch_info.get('structural_obstruction', branch_info['remaining']),
                    ),
                )
                branch_cost_r = branch_info.get('cost_r', branch_info['remaining'])
                max_cost_r = max(max_cost_r, branch_cost_r)
                phase_states += branch_info.get('phase_states', 0)
                phase_splits += branch_info.get('phase_splits', 0)
                branch_phase3_backend = branch_info.get('phase3_backend')
                if branch_phase3_backend is not None:
                    if branch_cost_r > phase3_backend_cost_r:
                        phase3_backend = branch_phase3_backend
                        phase3_backend_cost_r = branch_cost_r
                    elif (
                        branch_cost_r == phase3_backend_cost_r
                        and phase3_backend is not None
                        and branch_phase3_backend != phase3_backend
                    ):
                        phase3_backend = "mixed"
            result = _scale_scaled_complex(total, scale_half_pow2)
            info = {
                'quad': nq,
                'constraint': nc,
                'branched': 1 + max_branched,
                'remaining': max_remaining,
                'structural_obstruction': max_structural,
                'gauss_obstruction': max_gauss,
                'cost_r': max_cost_r,
                'phase_states': phase_states,
                'phase_splits': phase_splits,
                'phase3_backend': phase3_backend,
            }
            context.reduce_cache[cache_key] = (result, dict(info))
            return result, info

    if blocked_quadratics and not prefer_direct_phase3:
        split_result, split_info = _elim_quadratic_via_split(q, blocked_quadratics[0], context=context)
        result = _scale_scaled_complex(split_result, scale_half_pow2)
        info = _offset_reducer_info(split_info, quad=nq, constraint=nc, branched=nb)
        context.reduce_cache[cache_key] = (result, dict(info))
        return result, info

    phase3_total, phase3_info = _sum_irreducible_cubic_core(
        q,
        context=context,
        cover=phase3_cover,
        order=phase3_order,
        width=phase3_width,
        structural_obstruction=phase3_structural_obstruction,
        backend=direct_phase3_backend,
        allow_tensor_contraction=allow_tensor_contraction,
    )
    result = _scale_scaled_complex(phase3_total, scale_half_pow2)
    info = _offset_reducer_info(phase3_info, quad=nq, constraint=nc, branched=nb)
    context.reduce_cache[cache_key] = (result, dict(info))
    return result, info

def _reduce_and_sum(q, context=None):
    result, info = _reduce_and_sum_scaled(q, context=context)
    return _scaled_to_complex(result), info

def _reduce_and_sum_scaled_batch(
    q_batch: Sequence[PhaseFunction],
    *,
    context: _ReductionContext | None = None,
) -> list[tuple[ScaledComplex, ReducerInfo]]:
    """Batch companion to ``_reduce_and_sum_scaled`` for repeated exact queries."""
    if context is None:
        context = _ReductionContext()
    if not q_batch:
        return []

    results: list[tuple[ScaledComplex, ReducerInfo] | None] = [None] * len(q_batch)
    direct_groups: dict[
        tuple[str, tuple[int, ...], int, int, tuple[int, int, bytes]],
        list[tuple[int, PhaseFunction]],
    ] = {}
    support_groups: dict[
        tuple[int, int, bytes],
        list[tuple[int, PhaseFunction]],
    ] = {}
    fallback: list[tuple[int, PhaseFunction]] = []

    for batch_idx, q in enumerate(q_batch):
        cache_key = _q_key(q)
        cached = context.reduce_cache.get(cache_key)
        if cached is not None:
            results[batch_idx] = (cached[0], dict(cached[1]))
            continue

        if not q.q3 or q.n < _PRE_EXACT_PHASE3_TREEWIDTH_ESCAPE_MIN_VARS:
            fallback.append((batch_idx, q))
            continue
        support_groups.setdefault(_q_cubic_treewidth_batch_support_key(q), []).append((batch_idx, q))

    for support_key, group in support_groups.items():
        if len(group) == 1:
            fallback.extend(group)
            continue
        ref_q = group[0][1]
        native_level3_batch_ok = (
            _native_symbol("sum_level3_treewidth_preplanned_batch_array") is not None
            and _native_level3_enabled(ref_q)
        )
        native_generic_batch_ok = (
            _native_symbol("build_phase_function_treewidth_support_plan") is not None
            and _native_symbol("sum_phase_function_treewidth_preplanned_batch_scaled_array") is not None
        )
        native_batch_ok = native_level3_batch_ok or native_generic_batch_ok

        pre_exact_phase3 = None
        if native_batch_ok:
            q3_terms = sum(1 for coeff in ref_q.q3.values() if coeff % ref_q.mod_q3)
            active_q3 = _active_q3_variables(ref_q)
            if (
                q3_terms > 0
                and q3_terms <= _PRE_EXACT_PHASE3_TREEWIDTH_ESCAPE_MAX_Q3_TERMS
                and (not active_q3 or len(active_q3) * 4 <= ref_q.n)
            ):
                cover, order, width, structural_obstruction, fully_peeled = _phase3_batch_support_plan_fast(ref_q)
                width_limit = (
                    _Q3_TREEWIDTH_DP_PEELED_MAX_WIDTH
                    if fully_peeled
                    else _Q3_TREEWIDTH_DP_MAX_WIDTH
                )
                if (
                    width <= width_limit
                    and len(cover) <= _PRE_EXACT_PHASE3_TREEWIDTH_ESCAPE_MAX_COVER
                    and structural_obstruction <= _PRE_EXACT_PHASE3_TREEWIDTH_ESCAPE_MAX_COVER
                ):
                    backend = "treewidth_dp_peeled" if fully_peeled else "treewidth_dp"
                    pre_exact_phase3 = (
                        cover,
                        order,
                        width,
                        structural_obstruction,
                        backend,
                    )
        if pre_exact_phase3 is None:
            pre_exact_phase3 = _pre_exact_phase3_treewidth_escape(
                ref_q,
                allow_tensor_contraction=context.allow_tensor_contraction,
            )
            if pre_exact_phase3 is None or pre_exact_phase3[4] not in {"treewidth_dp", "treewidth_dp_peeled"}:
                fallback.extend(group)
                continue

        cover, order, width, structural_obstruction, backend = pre_exact_phase3
        del cover
        group_key = (
            str(backend),
            tuple(int(var) for var in order),
            int(width),
            int(structural_obstruction),
            support_key,
        )
        direct_groups[group_key] = list(group)

    for (backend, order, _width, structural_obstruction, _support_key), group in direct_groups.items():
        if len(group) == 1:
            fallback.extend(group)
            continue
        native_level3_batch_ok = (
            _native_symbol("sum_level3_treewidth_preplanned_batch_array") is not None
            and _native_level3_enabled(group[0][1])
        )
        native_generic_batch_ok = (
            _native_symbol("build_phase_function_treewidth_support_plan") is not None
            and _native_symbol("sum_phase_function_treewidth_preplanned_batch_scaled_array") is not None
        )
        if (
            _width > _PYTHON_TREEWIDTH_BATCH_MAX_WIDTH
            and not native_level3_batch_ok
            and not native_generic_batch_ok
        ):
            fallback.extend(group)
            continue

        q_group = [q for _batch_idx, q in group]
        try:
            totals, actual_width = _sum_via_treewidth_dp_scaled_batch_shared_support(
                q_group,
                list(order),
            )
        except ValueError:
            fallback.extend(group)
            continue

        for (batch_idx, q), total in zip(group, totals):
            info: ReducerInfo = {
                'quad': 0,
                'constraint': 0,
                'branched': 0,
                'remaining': int(actual_width),
                'structural_obstruction': int(structural_obstruction),
                'gauss_obstruction': _gauss_obstruction(q, int(structural_obstruction)),
                'cost_r': int(actual_width),
                'phase_states': 0,
                'phase_splits': 0,
                'phase3_backend': backend,
            }
            context.reduce_cache[_q_key(q)] = (total, dict(info))
            results[batch_idx] = (total, info)

    for batch_idx, q in fallback:
        results[batch_idx] = _reduce_and_sum_scaled(q, context=context)

    assert all(result is not None for result in results)
    return [result for result in results if result is not None]

def _invert_native_gate(gate: Gate) -> Gate:
    name = gate[0]
    if name in {"h", "x", "z", "cnot", "cz"}:
        return gate
    if name == "sx":
        return ("sxdg", gate[1])
    if name == "sxdg":
        return ("sx", gate[1])
    if name == "s":
        return ("sdg", gate[1])
    if name == "sdg":
        return ("s", gate[1])
    if name == "t":
        return ("tdg", gate[1])
    if name == "tdg":
        return ("t", gate[1])
    if name == "rz_dyadic":
        return ("rz_dyadic", int(gate[1]), -int(gate[2]), int(gate[3]))
    if name == "rz_arbitrary":
        return ("rz_arbitrary", int(gate[1]), -float(gate[2]))
    if name == "rzz_dyadic":
        return ("rzz_dyadic", int(gate[1]), int(gate[2]), -int(gate[3]), int(gate[4]))
    if name == "pauli_expbox":
        return ("pauli_expbox", tuple(gate[1]), tuple(int(qubit) for qubit in gate[2]), -float(gate[3]))
    if name == "rz_pi_16":
        return ("rz_pi_16_dg", int(gate[1]))
    if name == "rz_pi_16_dg":
        return ("rz_pi_16", int(gate[1]))
    if name == "rz_pi_32":
        return ("rz_pi_32_dg", int(gate[1]))
    if name == "rz_pi_32_dg":
        return ("rz_pi_32", int(gate[1]))
    raise ValueError(f"Unsupported inverse gate: {gate!r}")

def _invert_native_gates(gates: Sequence[Gate]) -> tuple[Gate, ...]:
    return tuple(_invert_native_gate(gate) for gate in reversed(gates))

def _fork_state_for_extension(state: SchurState) -> SchurState:
    """Clone a built state while sharing the phase polynomial until first write."""
    state._flush_pending_dead_variables()
    if getattr(state.q, "_schur_mutable", True):
        state.q._schur_mutable = False

    clone = SchurState(state.n)
    clone.m = state.m
    clone.eps = list(state.eps)
    clone.eps0 = list(state.eps0)
    clone.q = state.q
    clone.scalar = complex(state.scalar)
    clone.scalar_half_pow2 = int(state.scalar_half_pow2)
    clone.output_refcount = list(state.output_refcount)
    clone._arbitrary_phases = list(state._arbitrary_phases)
    clone._pending_dead = set(state._pending_dead)
    clone._defer_early_elim = bool(state._defer_early_elim)
    clone._cached_classification_data = state._cached_classification_data
    clone._cached_classification_q = state._cached_classification_q
    return clone

def _pauli_string_gates(pauli: str) -> tuple[Gate, ...]:
    gates: list[Gate] = []
    for qubit, pauli_char in enumerate(pauli):
        if pauli_char == "I":
            continue
        if pauli_char == "X":
            gates.append(("x", int(qubit)))
            continue
        if pauli_char == "Y":
            gates.extend((("sdg", int(qubit)), ("x", int(qubit)), ("s", int(qubit))))
            continue
        if pauli_char == "Z":
            gates.append(("z", int(qubit)))
            continue
        raise ValueError(f"Observable must use only I/X/Y/Z characters, received {pauli!r}.")
    return tuple(gates)

def _validate_pauli_observables(observables: Sequence[str], n_qubits: int) -> tuple[str, ...]:
    normalized: list[str] = []
    for observable in observables:
        if len(observable) != n_qubits:
            raise ValueError(
                f"Expected Pauli observable of length {n_qubits}, received length {len(observable)}."
            )
        if any(ch not in "IXYZ" for ch in observable):
            raise ValueError(f"Observable must use only I/X/Y/Z characters, received {observable!r}.")
        normalized.append(str(observable))
    return tuple(normalized)

def _elim_decoupled_constraints_batch(q, variables):
    if not variables:
        return q, 0

    removed = set(variables)
    remap = {}
    idx = 0
    for j in range(q.n):
        if j in removed:
            continue
        remap[j] = idx
        idx += 1

    reduced = _phase_function_from_parts_mutable(
        q.n - len(removed),
        level=q.level,
        q0=q.q0,
        q1=[q.q1[j] for j in range(q.n) if j not in removed],
        q2={
            (remap[i], remap[j]): value
            for (i, j), value in q.q2.items()
            if i not in removed and j not in removed
        },
        q3={
            (remap[i], remap[j], remap[k]): value
            for (i, j, k), value in q.q3.items()
            if i not in removed and j not in removed and k not in removed
        },
    )
    # Each removed decoupled constraint contributes an exact factor of 2.
    # Track this in half-powers of two to avoid overflowing float(2**m) on
    # very wide but still q3-free instances such as large Toffoli chains.
    return reduced, 2 * len(removed)

def _apply_exact_eliminations(q, context=None):
    """Apply single-term quadratic and constraint eliminations until saturation."""
    scale_half_pow2 = 0
    nq = nc = 0
    blocked_quadratics = []
    changed = True
    while changed:
        changed = False
        blocked_quadratics = []
        decoupled_constraints = []
        classification_data = _build_classification_data(q)
        prefer_cheap_actions = q.n >= _EXACT_ELIM_CHEAP_ACTION_MIN_VARS
        can_batch_sparse_quadratics = int(q.level) == 3 and not q.q3
        sparse_quadratics = []
        chosen_action = None
        chosen_quadratic = None
        chosen_parity = None
        first_blocked_quadratic = None
        for var in range(q.n):
            entry = _classification_entry(
                q,
                var,
                classification_data=classification_data,
            )
            tag = entry[0]
            if tag >= _CLASS_CONSTRAINT_DECOUPLED:
                if tag == _CLASS_CONSTRAINT_ZERO:
                    return None, 0, {'quad': nq, 'constraint': nc}, []
                if tag == _CLASS_CONSTRAINT_DECOUPLED:
                    decoupled_constraints.append(var)
                    continue
                if chosen_action is None:
                    chosen_action = (tag, var, entry)
                    if not prefer_cheap_actions:
                        break
                if (
                    chosen_parity is None
                    or len(entry[1]) < len(chosen_parity[2][1])
                ):
                    chosen_parity = (tag, var, entry)
                continue
            if tag == _CLASS_QUADRATIC:
                if entry[2]:
                    if first_blocked_quadratic is None:
                        first_blocked_quadratic = var
                    continue
                if chosen_action is None:
                    chosen_action = (tag, var, entry)
                if chosen_quadratic is None:
                    chosen_quadratic = (tag, var, entry)
                if can_batch_sparse_quadratics:
                    sparse_quadratics.append(var)
                elif not prefer_cheap_actions:
                    break

        if decoupled_constraints:
            q, half_pow2 = _elim_decoupled_constraints_batch(q, decoupled_constraints)
            scale_half_pow2 += half_pow2
            nc += len(decoupled_constraints)
            changed = True
            continue

        if len(sparse_quadratics) >= 8:
            q, half_pow2, removed = _elim_sparse_dead_quadratics_batch(
                q,
                sparse_quadratics,
                classification_data=classification_data,
            )
            if removed:
                scale_half_pow2 += half_pow2
                nq += len(removed)
                changed = True
                continue

        if prefer_cheap_actions:
            chosen_action = chosen_quadratic if chosen_quadratic is not None else chosen_parity
        if chosen_action is None:
            if first_blocked_quadratic is not None:
                blocked_quadratics.append(first_blocked_quadratic)
            continue

        tag, var, entry = chosen_action
        if tag == _CLASS_QUADRATIC:
            q, half_pow2 = _elim_quadratic(q, var, classification_data=classification_data)
            scale_half_pow2 += half_pow2
            nq += 1
            changed = True
            continue
        if tag == _CLASS_CONSTRAINT_PARITY:
            partners = entry[1]
            target = 1 if entry[2] == (q.mod_q1 // 2) else 0
            if len(partners) == 1:
                result = _elim_single_partner_constraint(q, var, partners[0], target)
            elif len(partners) == 2:
                result = _elim_two_partner_constraint(q, var, partners[1], partners[0], target)
            else:
                result = _elim_constraint(
                    q,
                    var,
                    {'type': 'parity', 'partners': partners, 'q1': entry[2]},
                    context=context,
                )
            if result is None:
                return None, 0, {'quad': nq, 'constraint': nc}, []
            q, half_pow2 = result
            scale_half_pow2 += half_pow2
            nc += 1
            changed = True
    if q is not None:
        q._schur_mutable = False
    return q, scale_half_pow2, {'quad': nq, 'constraint': nc}, blocked_quadratics

def _product_q1_sum(q1, level: int = 3):
    omega = _omega_table(level)
    modulus = 1 << level
    total = 1.0 + 0j
    for coeff in q1:
        total *= 1 + omega[coeff % modulus]
    return total

def _product_q1_sum_scaled(q1, level: int = 3):
    omega_plus_one = _omega_plus_one_scaled_table(level)
    modulus = 1 << level
    total = _ONE_SCALED
    for coeff in q1:
        total = _mul_scaled_complex(total, omega_plus_one[int(coeff) % modulus])
    return total

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
