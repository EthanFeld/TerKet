"""Extracted phase-3 execution helpers."""

from __future__ import annotations

import bisect
import cmath
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import heapq
from itertools import combinations
import math
import os
import struct
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Sequence, overload

import numpy as np

from .._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals
from ..cubic_arithmetic import CubicFunction, PhaseFunction, detect_factorization
from ..scaling import ScaledAmplitude, ScaledComplex
from ..spec import CircuitSpec, Gate
from ..state import BitSequence, CircuitInput, EchelonCache, ExtendedReductionMode, ReducerInfo, ReductionInfo, SolverConfig

_LOCAL_NAMES = {
    '_sum_via_q3_separator',
    '_sum_via_q3_treewidth_cutset',
    '_as_int64_array',
    '_compact_unsigned_storage_dtype',
    '_compact_index_storage_array',
    '_compact_residue_storage_array',
    '_phase_fraction_to_residue',
    '_build_q3_free_branch_template',
    '_branch_assignment_bits',
    '_q3_cover_branch_chunk_size',
    '_evaluate_q3_free_branch_template_batch',
    '_sum_via_q3_cover',
    '_sum_irreducible_cubic_core'
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
        respect_mock_wraps=True,
    )


_MISSING, _ENGINE_LOCAL_BASELINES = _bootstrap_extracted_globals(
    globals(),
    local_names=_LOCAL_NAMES,
    local_impls=_LOCAL_IMPLS,
    respect_mock_wraps=True,
)


def _sum_via_q3_separator(q, separator, context=None, *, structural_obstruction=None):
    """Branch on a small separator, then reduce each disconnected branch factor."""
    total = _make_scaled_complex(0j)
    total_quad = total_constraint = 0
    max_branched = 0
    max_cost_r = 0
    phase_states = phase_splits = 0

    for mask in range(1 << len(separator)):
        fixed_values = [(mask >> idx) & 1 for idx in range(len(separator))]
        branch_q = _fix_variables(q, separator, fixed_values, context=context)
        if not branch_q.q3:
            branch_total, branch_info = _sum_q3_free_direct_scaled(branch_q, context=context)
        else:
            components = detect_factorization(branch_q)
            if len(components) > 1:
                branch_total, branch_info = _sum_factorized_components_scaled(branch_q, components, context=context)
            else:
                branch_total, branch_info = _reduce_and_sum_scaled(branch_q, context=context)

        total = _add_scaled_complex(total, branch_total)
        total_quad += branch_info['quad']
        total_constraint += branch_info['constraint']
        max_branched = max(max_branched, branch_info['branched'])
        branch_cost_r = len(separator) + branch_info.get('cost_r', branch_info['remaining'])
        max_cost_r = max(max_cost_r, branch_cost_r)
        phase_states += branch_info.get('phase_states', 0)
        phase_splits += branch_info.get('phase_splits', 0)

    cubic_obstruction = len(separator) if structural_obstruction is None else structural_obstruction
    return total, {
        'quad': total_quad,
        'constraint': total_constraint,
        'branched': len(separator) + max_branched,
        'remaining': max_cost_r,
        'structural_obstruction': cubic_obstruction,
        'gauss_obstruction': _gauss_obstruction(q, cubic_obstruction),
        'cost_r': max_cost_r,
        'phase_states': phase_states,
        'phase_splits': phase_splits,
        'phase3_backend': 'q3_separator',
    }


def _sum_via_q3_treewidth_cutset(q, cutset, context=None, *, structural_obstruction=None):
    """Branch on a cutset so each residual avoids high-width Phase-3 DP."""
    total = _make_scaled_complex(0j)
    total_quad = total_constraint = 0
    max_branched = 0
    max_remaining = 0
    max_structural = 0
    max_gauss = 0
    max_cost_r = 0
    phase_states = phase_splits = 0
    phase3_backend = None
    phase3_backend_cost_r = -1

    cutset = tuple(int(var) for var in cutset)
    for mask in range(1 << len(cutset)):
        fixed_values = [(mask >> idx) & 1 for idx in range(len(cutset))]
        branch_q = _fix_variables(q, cutset, fixed_values, context=context)
        branch_total, branch_info = _reduce_and_sum_scaled(branch_q, context=context)

        total = _add_scaled_complex(total, branch_total)
        total_quad += branch_info['quad']
        total_constraint += branch_info['constraint']
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
        branch_cost_r = len(cutset) + branch_info.get('cost_r', branch_info['remaining'])
        max_cost_r = max(max_cost_r, branch_cost_r)
        phase_states += branch_info.get('phase_states', 0)
        phase_splits += branch_info.get('phase_splits', 0)
        branch_backend = branch_info.get('phase3_backend')
        if branch_backend is not None:
            if branch_cost_r > phase3_backend_cost_r:
                phase3_backend = branch_backend
                phase3_backend_cost_r = branch_cost_r
            elif branch_cost_r == phase3_backend_cost_r and branch_backend != phase3_backend:
                phase3_backend = "mixed"

    cubic_obstruction = len(cutset) if structural_obstruction is None else structural_obstruction
    return total, {
        'quad': total_quad,
        'constraint': total_constraint,
        'branched': len(cutset) + max_branched,
        'remaining': max(max_remaining, max_cost_r),
        'structural_obstruction': max(cubic_obstruction, max_structural),
        'gauss_obstruction': max(_gauss_obstruction(q, cubic_obstruction), max_gauss),
        'cost_r': max_cost_r,
        'phase_states': phase_states,
        'phase_splits': phase_splits,
        'phase3_backend': 'q3_treewidth_cutset' if phase3_backend is not None else 'q3_treewidth_cutset',
    }
def _as_int64_array(values) -> np.ndarray:
    if not values:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(values, dtype=np.int64)


def _compact_unsigned_storage_dtype(max_value: int):
    max_value = max(0, int(max_value))
    if max_value <= np.iinfo(np.uint8).max:
        return np.uint8
    if max_value <= np.iinfo(np.uint16).max:
        return np.uint16
    if max_value <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.int64


def _compact_index_storage_array(values, *, upper_bound: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if upper_bound is None:
        max_value = int(array.max(initial=0)) if array.size else 0
    else:
        max_value = max(0, int(upper_bound) - 1)
    return np.asarray(array, dtype=_compact_unsigned_storage_dtype(max_value))


def _compact_residue_storage_array(values, *, modulus: int) -> np.ndarray:
    array = np.asarray(values)
    return np.asarray(array, dtype=_compact_unsigned_storage_dtype(int(modulus) - 1))


def _phase_fraction_to_residue(value: Fraction, modulus: int) -> int:
    scaled = Fraction(value) * modulus
    if scaled.denominator != 1:
        raise ValueError(f"Phase constant {value!r} is not representable modulo {modulus}.")
    return int(scaled.numerator % modulus)


def _build_q3_free_branch_template(q, cover) -> Q3FreeBranchTemplate:
    """Precompute the cover-conditioned residue updates for exact branch batching."""
    cover_vars = tuple(int(var) for var in cover)
    cover_map = {var: idx for idx, var in enumerate(cover_vars)}
    remaining_vars = tuple(var for var in range(q.n) if var not in cover_map)
    remaining_map = {var: idx for idx, var in enumerate(remaining_vars)}
    mod_q1 = q.mod_q1

    pair_keys: list[tuple[int, int]] = []
    pair_map: dict[tuple[int, int], int] = {}

    def ensure_pair(left_var: int, right_var: int) -> int:
        key = tuple(sorted((remaining_map[left_var], remaining_map[right_var])))
        existing = pair_map.get(key)
        if existing is not None:
            return existing
        idx = len(pair_keys)
        pair_keys.append(key)
        pair_map[key] = idx
        return idx

    for (left, right), coeff in q.q2.items():
        if left in remaining_map and right in remaining_map and coeff % q.mod_q2:
            ensure_pair(left, right)
    for triple, coeff in q.q3.items():
        if not coeff % q.mod_q3:
            continue
        remaining = tuple(var for var in triple if var in remaining_map)
        if len(remaining) == 2:
            ensure_pair(remaining[0], remaining[1])

    pair_left = _compact_index_storage_array([left for left, _ in pair_keys], upper_bound=len(remaining_vars))
    pair_right = _compact_index_storage_array([right for _, right in pair_keys], upper_bound=len(remaining_vars))
    base_q2_residue = np.zeros(len(pair_keys), dtype=np.int64)
    q2_lift = q.mod_q1 // q.mod_q2 if q.mod_q2 else 0
    q3_lift = q.mod_q1 // q.mod_q3 if q.mod_q3 else 0

    for (left, right), coeff in q.q2.items():
        if left in remaining_map and right in remaining_map:
            pair_idx = ensure_pair(left, right)
            base_q2_residue[pair_idx] = (base_q2_residue[pair_idx] + q2_lift * coeff) % mod_q1

    cover_q1_residue = np.array([q.q1[var] % mod_q1 for var in cover_vars], dtype=np.int64)
    base_q1_residue = np.array([q.q1[var] % mod_q1 for var in remaining_vars], dtype=np.int64)
    cover_remaining_q2_residue = np.zeros((len(cover_vars), len(remaining_vars)), dtype=np.int64)
    cover_cover_left: list[int] = []
    cover_cover_right: list[int] = []
    cover_cover_residue: list[int] = []

    for (left, right), coeff in q.q2.items():
        residue = (q2_lift * coeff) % mod_q1
        if not residue:
            continue
        if left in cover_map and right in remaining_map:
            cover_remaining_q2_residue[cover_map[left], remaining_map[right]] = (
                cover_remaining_q2_residue[cover_map[left], remaining_map[right]] + residue
            ) % mod_q1
        elif right in cover_map and left in remaining_map:
            cover_remaining_q2_residue[cover_map[right], remaining_map[left]] = (
                cover_remaining_q2_residue[cover_map[right], remaining_map[left]] + residue
            ) % mod_q1
        elif left in cover_map and right in cover_map:
            cover_cover_left.append(cover_map[left])
            cover_cover_right.append(cover_map[right])
            cover_cover_residue.append(residue)

    cubic_pair_cover: list[int] = []
    cubic_pair_index: list[int] = []
    cubic_pair_residue: list[int] = []
    cubic_linear_cover_left: list[int] = []
    cubic_linear_cover_right: list[int] = []
    cubic_linear_var: list[int] = []
    cubic_linear_residue: list[int] = []
    cubic_constant_left: list[int] = []
    cubic_constant_middle: list[int] = []
    cubic_constant_right: list[int] = []
    cubic_constant_residue: list[int] = []

    for triple, coeff in q.q3.items():
        residue = (q3_lift * coeff) % mod_q1
        if not residue:
            continue
        cover_entries = [cover_map[var] for var in triple if var in cover_map]
        remaining_entries = [remaining_map[var] for var in triple if var in remaining_map]
        if len(cover_entries) == 1 and len(remaining_entries) == 2:
            pair_idx = pair_map[tuple(sorted(remaining_entries))]
            cubic_pair_cover.append(cover_entries[0])
            cubic_pair_index.append(pair_idx)
            cubic_pair_residue.append(residue)
        elif len(cover_entries) == 2 and len(remaining_entries) == 1:
            cubic_linear_cover_left.append(cover_entries[0])
            cubic_linear_cover_right.append(cover_entries[1])
            cubic_linear_var.append(remaining_entries[0])
            cubic_linear_residue.append(residue)
        elif len(cover_entries) == 3:
            cubic_constant_left.append(cover_entries[0])
            cubic_constant_middle.append(cover_entries[1])
            cubic_constant_right.append(cover_entries[2])
            cubic_constant_residue.append(residue)
        else:
            raise ValueError("q3-cover template encountered an unfixed cubic term.")

    return Q3FreeBranchTemplate(
        cover_vars=cover_vars,
        remaining_vars=remaining_vars,
        n_cover=len(cover_vars),
        n_remaining=len(remaining_vars),
        mod_q1=mod_q1,
        level=q.level,
        base_q0_residue=_phase_fraction_to_residue(q.q0, mod_q1),
        base_q1_residue=_compact_residue_storage_array(base_q1_residue, modulus=mod_q1),
        pair_left=pair_left,
        pair_right=pair_right,
        base_q2_residue=_compact_residue_storage_array(base_q2_residue, modulus=mod_q1),
        cover_q1_residue=_compact_residue_storage_array(cover_q1_residue, modulus=mod_q1),
        cover_remaining_q2_residue=_compact_residue_storage_array(cover_remaining_q2_residue, modulus=mod_q1),
        cover_cover_left=_compact_index_storage_array(cover_cover_left, upper_bound=len(cover_vars)),
        cover_cover_right=_compact_index_storage_array(cover_cover_right, upper_bound=len(cover_vars)),
        cover_cover_residue=_compact_residue_storage_array(cover_cover_residue, modulus=mod_q1),
        cubic_pair_cover=_compact_index_storage_array(cubic_pair_cover, upper_bound=len(cover_vars)),
        cubic_pair_index=_compact_index_storage_array(cubic_pair_index, upper_bound=len(pair_keys)),
        cubic_pair_residue=_compact_residue_storage_array(cubic_pair_residue, modulus=mod_q1),
        cubic_linear_cover_left=_compact_index_storage_array(cubic_linear_cover_left, upper_bound=len(cover_vars)),
        cubic_linear_cover_right=_compact_index_storage_array(cubic_linear_cover_right, upper_bound=len(cover_vars)),
        cubic_linear_var=_compact_index_storage_array(cubic_linear_var, upper_bound=len(remaining_vars)),
        cubic_linear_residue=_compact_residue_storage_array(cubic_linear_residue, modulus=mod_q1),
        cubic_constant_left=_compact_index_storage_array(cubic_constant_left, upper_bound=len(cover_vars)),
        cubic_constant_middle=_compact_index_storage_array(cubic_constant_middle, upper_bound=len(cover_vars)),
        cubic_constant_right=_compact_index_storage_array(cubic_constant_right, upper_bound=len(cover_vars)),
        cubic_constant_residue=_compact_residue_storage_array(cubic_constant_residue, modulus=mod_q1),
    )


def _branch_assignment_bits(branch_masks: np.ndarray, n_cover: int):
    if n_cover == 0:
        return np.zeros((len(branch_masks), 0), dtype=np.int64)
    masks = np.asarray(branch_masks, dtype=np.uint64).reshape(-1, 1)
    shifts = np.arange(n_cover, dtype=np.uint64).reshape(1, -1)
    return ((masks >> shifts) & 1).astype(np.int64)


def _q3_cover_branch_chunk_size(template: Q3FreeBranchTemplate, budget_bytes: int | None) -> tuple[int, int]:
    assignment_chunk = 1 << min(template.n_remaining, _Q3_COVER_ASSIGNMENT_CHUNK_LOG2)
    branch_chunk = min(_Q3_COVER_BRANCH_CHUNK_MAX, 1 << template.n_cover)
    if budget_bytes is None:
        return branch_chunk, assignment_chunk

    while branch_chunk > 1:
        estimated = assignment_chunk * branch_chunk * 32
        if estimated <= budget_bytes:
            break
        branch_chunk //= 2
    return max(1, branch_chunk), assignment_chunk


def _evaluate_q3_free_branch_template_batch(
    template: Q3FreeBranchTemplate,
    branch_masks: np.ndarray,
    *,
    assignment_chunk_size: int | None = None,
) -> np.ndarray:
    if branch_masks.size == 0:
        return np.zeros(0, dtype=np.complex128)

    if assignment_chunk_size is None:
        assignment_chunk_size = 1 << min(template.n_remaining, _Q3_COVER_ASSIGNMENT_CHUNK_LOG2)

    branch_bits = _branch_assignment_bits(branch_masks, template.n_cover)
    mod_q1 = int(template.mod_q1)
    omega = np.asarray(_omega_table(template.level), dtype=np.complex128)

    q0_eff = np.full(branch_bits.shape[0], int(template.base_q0_residue), dtype=np.int64)
    q1_eff = np.broadcast_to(
        np.asarray(template.base_q1_residue, dtype=np.int64),
        (branch_bits.shape[0], template.n_remaining),
    ).copy()
    q2_eff = np.broadcast_to(
        np.asarray(template.base_q2_residue, dtype=np.int64),
        (branch_bits.shape[0], template.pair_left.size),
    ).copy()

    if template.cover_q1_residue.size:
        q0_eff = (q0_eff + branch_bits @ np.asarray(template.cover_q1_residue, dtype=np.int64)) % mod_q1
    if template.cover_remaining_q2_residue.size:
        q1_eff = (q1_eff + branch_bits @ np.asarray(template.cover_remaining_q2_residue, dtype=np.int64)) % mod_q1
    if template.cover_cover_residue.size:
        for left, right, residue in zip(
            template.cover_cover_left,
            template.cover_cover_right,
            template.cover_cover_residue,
        ):
            q0_eff = (q0_eff + int(residue) * branch_bits[:, int(left)] * branch_bits[:, int(right)]) % mod_q1
    if template.cubic_pair_residue.size:
        for cover_idx, pair_idx, residue in zip(
            template.cubic_pair_cover,
            template.cubic_pair_index,
            template.cubic_pair_residue,
        ):
            q2_eff[:, int(pair_idx)] = (
                q2_eff[:, int(pair_idx)] + int(residue) * branch_bits[:, int(cover_idx)]
            ) % mod_q1
    if template.cubic_linear_residue.size:
        for left, right, var_idx, residue in zip(
            template.cubic_linear_cover_left,
            template.cubic_linear_cover_right,
            template.cubic_linear_var,
            template.cubic_linear_residue,
        ):
            q1_eff[:, int(var_idx)] = (
                q1_eff[:, int(var_idx)]
                + int(residue) * branch_bits[:, int(left)] * branch_bits[:, int(right)]
            ) % mod_q1
    if template.cubic_constant_residue.size:
        for left, middle, right, residue in zip(
            template.cubic_constant_left,
            template.cubic_constant_middle,
            template.cubic_constant_right,
            template.cubic_constant_residue,
        ):
            q0_eff = (
                q0_eff
                + int(residue)
                * branch_bits[:, int(left)]
                * branch_bits[:, int(middle)]
                * branch_bits[:, int(right)]
            ) % mod_q1

    totals = np.zeros(branch_bits.shape[0], dtype=np.complex128)
    n_assignments = 1 << template.n_remaining

    for start in range(0, n_assignments, assignment_chunk_size):
        stop = min(n_assignments, start + assignment_chunk_size)
        assignments = np.arange(start, stop, dtype=np.uint64).reshape(-1, 1)
        if template.n_remaining:
            shifts = np.arange(template.n_remaining, dtype=np.uint64).reshape(1, -1)
            x = ((assignments >> shifts) & 1).astype(np.int64)
            residues = x @ q1_eff.T
            if template.pair_left.size:
                pair_terms = x[:, np.asarray(template.pair_left, dtype=np.int64)] * x[
                    :, np.asarray(template.pair_right, dtype=np.int64)
                ]
                residues = residues + pair_terms @ q2_eff.T
        else:
            residues = np.zeros((stop - start, branch_bits.shape[0]), dtype=np.int64)
        residues = (residues + q0_eff.reshape(1, -1)) % mod_q1
        totals = totals + omega[residues].sum(axis=0)

    return np.asarray(totals, dtype=np.complex128)


def _sum_via_q3_cover(q, context=None, *, structural_obstruction=None):
    """Branch on a q3-hypergraph cover, then use q3-free exact sums."""
    cover = _minimum_q3_vertex_cover(q)
    if not cover:
        if context is not None and not context.preserve_scale:
            total_complex, phase_info = _gauss_sum_q3_free(
                q,
                allow_tensor_contraction=context.allow_tensor_contraction,
            )
            total = _make_scaled_complex(total_complex)
        else:
            total, phase_info = _gauss_sum_q3_free_scaled(
                q,
                allow_tensor_contraction=(
                    True if context is None else context.allow_tensor_contraction
                ),
            )
        cubic_obstruction = 0 if structural_obstruction is None else structural_obstruction
        return total, {
            'quad': 0,
            'constraint': 0,
            'branched': 0,
            'remaining': 0,
            'structural_obstruction': cubic_obstruction,
            'gauss_obstruction': _gauss_obstruction(q, cubic_obstruction),
            'cost_r': 0,
            'phase_states': phase_info.get('phase_states', 0),
            'phase_splits': phase_info.get('phase_splits', 0),
            'phase3_backend': _q3_free_phase3_backend_name(q),
        }

    template_cache_key = (_q_key(q), tuple(int(var) for var in cover))
    template = _STRUCTURE_Q3_COVER_TEMPLATE_CACHE.get(template_cache_key)
    if template is None:
        template = _build_q3_free_branch_template(q, cover)
        _STRUCTURE_Q3_COVER_TEMPLATE_CACHE[template_cache_key] = template

    total = _make_scaled_complex(0j)
    nq = nc = 0
    max_branched = 0
    max_remaining = 0
    max_gauss = 0
    max_cost_r = 0
    phase_states = phase_splits = 0
    phase3_backend = None
    dominant_cost_r = -1
    n_branches = 1 << len(cover)
    branch_chunk_size, assignment_chunk_size = _q3_cover_branch_chunk_size(template, budget_bytes=None)
    complex_total = 0.0 + 0.0j
    for start in range(0, n_branches, branch_chunk_size):
        stop = min(n_branches, start + branch_chunk_size)
        branch_masks = np.arange(start, stop, dtype=np.uint64)
        branch_totals = _evaluate_q3_free_branch_template_batch(
            template,
            branch_masks,
            assignment_chunk_size=assignment_chunk_size,
        )
        complex_total += complex(np.asarray(branch_totals, dtype=np.complex128).sum())
    total = _make_scaled_complex(complex_total)
    phase3_backend = 'q3_cover'

    cubic_obstruction = len(cover) if structural_obstruction is None else structural_obstruction
    return total, {
        'quad': nq,
        'constraint': nc,
        'branched': len(cover) + max_branched,
        'remaining': max(len(cover), max_remaining),
        'structural_obstruction': cubic_obstruction,
        'gauss_obstruction': max(_gauss_obstruction(q, cubic_obstruction), max_gauss),
        'cost_r': max(len(cover), max_cost_r),
        'phase_states': phase_states,
        'phase_splits': phase_splits,
        'phase3_backend': phase3_backend if phase3_backend is not None else 'q3_cover',
    }


def _sum_irreducible_cubic_core(
    q,
    context=None,
    cover=None,
    order=None,
    width=None,
    structural_obstruction=None,
    backend=None,
    allow_tensor_contraction=True,
):
    """Phase-3 solver for a genuinely cubic residual kernel."""
    cover_missing = cover is None
    order_missing = order is None
    width_missing = width is None
    structural_missing = structural_obstruction is None
    if cover_missing or order_missing or width_missing or structural_missing:
        planned_cover, planned_order, planned_width, planned_structural_obstruction, planned_backend = _phase3_plan(
            q,
            allow_tensor_contraction=allow_tensor_contraction,
        )
        if cover_missing:
            cover = planned_cover
        if order_missing:
            order = planned_order
        if width_missing:
            width = planned_width
        if structural_missing:
            structural_obstruction = len(cover) if not cover_missing else planned_structural_obstruction
        if backend is None and cover_missing and order_missing and width_missing and structural_missing:
            backend = planned_backend

    assert cover is not None
    assert order is not None
    assert width is not None
    assert structural_obstruction is not None

    def run_cubic_contraction():
        plan = plan_contraction(q, order=order)
        total = execute_plan_cpu(plan)
        backend_name = 'cubic_contraction'
        return _make_scaled_complex(total), {
            'quad': 0,
            'constraint': 0,
            'branched': 0,
            'remaining': plan.max_scope_size,
            'structural_obstruction': structural_obstruction,
            'gauss_obstruction': _gauss_obstruction(q, structural_obstruction),
            'cost_r': plan.max_scope_size,
            'phase_states': 0,
            'phase_splits': 0,
            'phase3_backend': backend_name,
        }

    def run_treewidth(selected_backend: str):
        if selected_backend == "treewidth_dp_peeled":
            total, actual_width = _sum_via_treewidth_dp_peeled_scaled(q, order)
        else:
            total, actual_width = _sum_via_treewidth_dp_scaled(q, order)
        return total, {
            'quad': 0,
            'constraint': 0,
            'branched': 0,
            'remaining': actual_width,
            'structural_obstruction': structural_obstruction,
            'gauss_obstruction': _gauss_obstruction(q, structural_obstruction),
            'cost_r': actual_width,
            'phase_states': 0,
            'phase_splits': 0,
            'phase3_backend': selected_backend,
        }

    if backend in {"treewidth_dp", "treewidth_dp_peeled"}:
        return run_treewidth(backend)
    if backend in {"cubic_contraction", "cubic_contraction_cpu"}:
        return run_cubic_contraction()
    if backend == "q3_cover":
        return _sum_via_q3_cover(q, context=context, structural_obstruction=structural_obstruction)
    if backend == "q3_separator":
        separator = None
        extended_reductions = "auto" if context is None else context.extended_reductions
        if _should_apply_extended_q3_reductions(q, extended_reductions):
            separator = _find_small_q3_separator(q)
        if separator is not None and len(separator) < len(cover):
            return _sum_via_q3_separator(
                q,
                separator,
                context=context,
                structural_obstruction=structural_obstruction,
            )
        return _sum_via_q3_cover(q, context=context, structural_obstruction=structural_obstruction)
    if backend == "q3_treewidth_cutset":
        core_vars, peel_order = _q3_hypergraph_2core(q)
        fully_peeled = bool(peel_order) and not core_vars
        cutset_plan = _find_q3_treewidth_cutset(
            q,
            order=order,
            width=width,
            fully_peeled=fully_peeled,
        )
        if cutset_plan is not None and len(cutset_plan[0]) < len(cover):
            return _sum_via_q3_treewidth_cutset(
                q,
                cutset_plan[0],
                context=context,
                structural_obstruction=structural_obstruction,
            )
        return _sum_via_q3_cover(q, context=context, structural_obstruction=structural_obstruction)
    if backend is not None:
        raise ValueError(f"Unknown Phase-3 backend {backend!r}.")

    core_vars, peel_order = _q3_hypergraph_2core(q)
    fully_peeled = bool(peel_order) and not core_vars
    extended_reductions = "auto" if context is None else context.extended_reductions
    selected_backend, _runtime_score, selected_separator = _choose_phase3_backend(
        q,
        cover,
        order,
        width,
        structural_obstruction,
        allow_tensor_contraction=allow_tensor_contraction,
        fully_peeled=fully_peeled,
        extended_reductions=extended_reductions,
    )

    if selected_backend in {"treewidth_dp", "treewidth_dp_peeled"}:
        return run_treewidth(selected_backend)

    if selected_backend == "cubic_contraction":
        return run_cubic_contraction()

    if selected_backend == "q3_separator" and selected_separator is not None:
        return _sum_via_q3_separator(
            q,
            selected_separator,
            context=context,
            structural_obstruction=structural_obstruction,
        )

    if selected_backend == "q3_treewidth_cutset":
        cutset_plan = _find_q3_treewidth_cutset(
            q,
            order=order,
            width=width,
            fully_peeled=fully_peeled,
        )
        if cutset_plan is not None:
            return _sum_via_q3_treewidth_cutset(
                q,
                cutset_plan[0],
                context=context,
                structural_obstruction=structural_obstruction,
            )

    if selected_backend == "q3_cover":
        return _sum_via_q3_cover(q, context=context, structural_obstruction=structural_obstruction)

    if structural_obstruction > 0 and _prefer_cubic_contraction_phase3(
        q,
        cover,
        order,
        width,
        fully_peeled=fully_peeled,
    ):
        return run_cubic_contraction()

    return _sum_via_q3_cover(q, context=context, structural_obstruction=structural_obstruction)

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
