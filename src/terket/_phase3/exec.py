"""Phase-3 execution helpers for genuinely cubic residual kernels.

Owns:
- Phase-3 backend execution flow for separator, cutset, cover, and DP paths
- cubic residual backend selection fallback behavior

Key invariants:
- q3-free batching/storage helpers live under ``terket._q3free.batch``
- this module owns Phase-3 execution, not generic q3-free reusable plan storage
"""

from __future__ import annotations

import bisect
import cmath
from collections import deque
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
from .._q3free.batch import (
    _as_int64_array,
    _branch_assignment_bits,
    _build_q3_free_branch_template,
    _compact_index_storage_array,
    _compact_residue_storage_array,
    _compact_unsigned_storage_dtype,
    _evaluate_q3_free_branch_template_batch,
    _phase_fraction_to_residue,
    _q3_cover_branch_chunk_size,
)
from ..cubic_arithmetic import CubicFunction, PhaseFunction, detect_factorization
from ..scaling import ScaledAmplitude, ScaledComplex
from ..spec import CircuitSpec, Gate
from ..state import BitSequence, CircuitInput, EchelonCache, ExtendedReductionMode, ReducerInfo, ReductionInfo, SolverConfig

_LOCAL_NAMES = {
    '_sum_via_q3_separator',
    '_sum_via_q3_treewidth_cutset',
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
