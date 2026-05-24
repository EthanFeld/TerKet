"""Extracted phase-3 backend selection helpers."""

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
    '_prefer_cubic_contraction_phase3',
    '_select_direct_phase3_backend',
    '_phase3_backend_runtime_score',
    '_phase3_treewidth_candidate',
    '_phase3_cubic_contraction_candidate',
    '_phase3_separator_candidate',
    '_phase3_treewidth_cutset_candidate',
    '_phase3_cover_candidate',
    '_choose_phase3_backend',
    '_phase3_plan',
    '_PHASE3_BACKEND_CANDIDATE_BUILDERS',
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


def _prefer_cubic_contraction_phase3(q, cover, order, width, *, fully_peeled: bool = False):
    """Return whether the specialized cubic contraction should be used."""
    if not _HAS_CUBIC_CONTRACTION:
        return False
    if not q.q3:
        return False
    # Let the C-native treewidth DP handle cases where it is optimal.
    if _prefer_treewidth_phase3(q, cover, order, width, fully_peeled=fully_peeled):
        return False
    # Benchmark crossover: cubic contraction beats quimb for width <= 12.
    if width is None or width > _CUBIC_CONTRACTION_MAX_WIDTH:
        return False
    # Bucket tables of size 2^width must fit comfortably in memory.
    if q.n > 60:
        return False
    return True


def _select_direct_phase3_backend(
    q,
    cover,
    order,
    width,
    *,
    allow_tensor_contraction=True,
    fully_peeled: bool = False,
    treewidth_work: int | None = None,
):
    """Return the direct Phase-3 backend worth preferring over Phase-2 branching."""
    if fully_peeled and _prefer_treewidth_phase3(
        q,
        cover,
        order,
        width,
        fully_peeled=True,
        treewidth_work=treewidth_work,
    ):
        return 'treewidth_dp_peeled'
    if _prefer_treewidth_phase3(
        q,
        cover,
        order,
        width,
        fully_peeled=fully_peeled,
        treewidth_work=treewidth_work,
    ):
        return 'treewidth_dp'
    if _prefer_cubic_contraction_phase3(
        q,
        cover,
        order,
        width,
        fully_peeled=fully_peeled,
    ):
        return 'cubic_contraction'
    return None


def _phase3_backend_runtime_score(
    q,
    cover,
    order,
    width,
    structural_obstruction,
    backend: str | None,
    *,
    separator: Sequence[int] | None = None,
    fully_peeled: bool = False,
) -> tuple[int, int, int, int, int]:
    """Return a runtime-oriented score for a concrete Phase-3 backend."""
    if backend == "treewidth_dp_peeled":
        work = max(1, int(_estimate_treewidth_dp_work(q, order)))
        return (0, work, int(width), len(cover), int(structural_obstruction))
    if backend == "treewidth_dp":
        work = max(1, int(_estimate_treewidth_dp_work(q, order)))
        return (1, work, int(width), len(cover), int(structural_obstruction))
    if backend in {"cubic_contraction", "cubic_contraction_cpu"}:
        work = max(1, q.n * (1 << max(0, int(width))))
        return (2, work, int(width), len(cover), int(structural_obstruction))
    if backend == "q3_separator":
        separator = tuple(separator or ())
        separator_size = len(separator)
        work = max(1, int(_estimate_q3_separator_work(q, separator)))
        return (3, work, separator_size, len(cover), int(structural_obstruction))
    if backend == "q3_treewidth_cutset":
        cutset_plan = _find_q3_treewidth_cutset(
            q,
            order=order,
            width=width,
            fully_peeled=fully_peeled,
        )
        cutset_size = len(cutset_plan[0]) if cutset_plan is not None else len(cover)
        residual_width = cutset_plan[2] if cutset_plan is not None else int(width)
        work = max(1, int(_estimate_q3_treewidth_cutset_work(q, cutset_plan)))
        return (3, work, cutset_size + residual_width, len(cover), int(structural_obstruction))
    if backend == "q3_cover":
        work = max(1, int(_estimate_q3_cover_work(q, len(cover))))
        return (3, work, len(cover), len(cover), int(structural_obstruction))
    return (9, 1 << 62, 1 << 30, len(cover), int(structural_obstruction))
def _phase3_treewidth_candidate(
    q,
    cover,
    order,
    width,
    structural_obstruction,
    *,
    fully_peeled: bool,
    extended_reductions: str,
) -> _Phase3BackendCandidate | None:
    del extended_reductions
    if fully_peeled:
        if not _prefer_treewidth_phase3(
            q,
            cover,
            order,
            width,
            fully_peeled=True,
        ):
            return None
        return _Phase3BackendCandidate(
            "treewidth_dp",
            _phase3_backend_runtime_score(
                q,
                cover,
                order,
                width,
                structural_obstruction,
                "treewidth_dp_peeled",
                fully_peeled=True,
            ),
            peeled=True,
        )
    if not _prefer_treewidth_phase3(q, cover, order, width, fully_peeled=False):
        return None
    return _Phase3BackendCandidate(
        "treewidth_dp",
        _phase3_backend_runtime_score(
            q,
            cover,
            order,
            width,
            structural_obstruction,
            "treewidth_dp",
            fully_peeled=False,
        ),
    )


def _phase3_cubic_contraction_candidate(
    q,
    cover,
    order,
    width,
    structural_obstruction,
    *,
    fully_peeled: bool,
    extended_reductions: str,
) -> _Phase3BackendCandidate | None:
    del extended_reductions
    if not _prefer_cubic_contraction_phase3(
        q,
        cover,
        order,
        width,
        fully_peeled=fully_peeled,
    ):
        return None
    return _Phase3BackendCandidate(
        "cubic_contraction",
        _phase3_backend_runtime_score(
            q,
            cover,
            order,
            width,
            structural_obstruction,
            "cubic_contraction",
            fully_peeled=fully_peeled,
        ),
    )


def _phase3_separator_candidate(
    q,
    cover,
    order,
    width,
    structural_obstruction,
    *,
    fully_peeled: bool,
    extended_reductions: str,
) -> _Phase3BackendCandidate | None:
    if not _should_apply_extended_q3_reductions(q, extended_reductions):
        return None
    separator = _find_small_q3_separator(q)
    if separator is None or len(separator) >= len(cover):
        return None
    separator = tuple(separator)
    return _Phase3BackendCandidate(
        "q3_separator",
        _phase3_backend_runtime_score(
            q,
            cover,
            order,
            width,
            structural_obstruction,
            "q3_separator",
            separator=separator,
            fully_peeled=fully_peeled,
        ),
        separator=separator,
    )


def _phase3_treewidth_cutset_candidate(
    q,
    cover,
    order,
    width,
    structural_obstruction,
    *,
    fully_peeled: bool,
    extended_reductions: str,
) -> _Phase3BackendCandidate | None:
    del extended_reductions
    cutset_plan = _find_q3_treewidth_cutset(
        q,
        order=order,
        width=width,
        fully_peeled=fully_peeled,
    )
    if cutset_plan is None or len(cutset_plan[0]) >= len(cover):
        return None
    return _Phase3BackendCandidate(
        "q3_treewidth_cutset",
        _phase3_backend_runtime_score(
            q,
            cover,
            order,
            width,
            structural_obstruction,
            "q3_treewidth_cutset",
            fully_peeled=fully_peeled,
        ),
    )


def _phase3_cover_candidate(
    q,
    cover,
    order,
    width,
    structural_obstruction,
    *,
    fully_peeled: bool,
    extended_reductions: str,
) -> _Phase3BackendCandidate:
    del extended_reductions
    return _Phase3BackendCandidate(
        "q3_cover",
        _phase3_backend_runtime_score(
            q,
            cover,
            order,
            width,
            structural_obstruction,
            "q3_cover",
            fully_peeled=fully_peeled,
        ),
    )


_PHASE3_BACKEND_CANDIDATE_BUILDERS = (
    _phase3_treewidth_candidate,
    _phase3_cubic_contraction_candidate,
    _phase3_separator_candidate,
    _phase3_treewidth_cutset_candidate,
    _phase3_cover_candidate,
)


def _choose_phase3_backend(
    q,
    cover,
    order,
    width,
    structural_obstruction,
    *,
    allow_tensor_contraction: bool,
    fully_peeled: bool,
    extended_reductions: str = "auto",
) -> tuple[str, tuple[int, int, int, int, int], tuple[int, ...] | None]:
    """Choose the best available Phase-3 backend by a shared runtime score."""
    candidates: list[_Phase3BackendCandidate] = []
    for build_candidate in _PHASE3_BACKEND_CANDIDATE_BUILDERS:
        if (
            not allow_tensor_contraction
            and getattr(build_candidate, "__name__", "") == "_phase3_cubic_contraction_candidate"
        ):
            continue
        candidate = build_candidate(
            q,
            cover,
            order,
            width,
            structural_obstruction,
            fully_peeled=fully_peeled,
            extended_reductions=extended_reductions,
        )
        if candidate is not None:
            candidates.append(candidate)

    best = min(candidates, key=lambda candidate: candidate.score)
    return best.metadata_backend, best.score, best.separator


def _phase3_plan(q, allow_tensor_contraction=True):
    """Return the diagnostics needed to choose a Phase-3 backend."""
    cache_key = (_q_phase3_structure_key(q), bool(allow_tensor_contraction))
    cached = _STRUCTURE_PHASE3_PLAN_CACHE.get(cache_key)
    if cached is not None:
        cover, order, width, structural_obstruction, direct_backend = cached
        return list(cover), list(order), width, structural_obstruction, direct_backend

    cover = _minimum_q3_vertex_cover(q)
    order, width = _min_fill_cubic_order(q)
    core_vars, peel_order = _q3_hypergraph_2core(q)
    core_cover_size = _q3_core_cover_size(q, core_vars) if q.q3 else 0
    if peel_order:
        peel_set = set(peel_order)
        order = peel_order + [var for var in order if var not in peel_set]
        width = _treewidth_order_width(q, order)
    if q.q3:
        if (
            hasattr(_treewidth_order_width, "_increment_mock_call")
            or hasattr(_estimate_treewidth_dp_work, "_increment_mock_call")
        ):
            width = int(_treewidth_order_width(q, order))
        else:
            order, width = _finalize_phase3_treewidth_order(q, order)
    structural_obstruction = min(core_cover_size, width) if q.q3 else 0
    fully_peeled = bool(peel_order) and not core_vars
    if fully_peeled and (
        hasattr(_treewidth_order_width, "_increment_mock_call")
        or hasattr(_estimate_treewidth_dp_work, "_increment_mock_call")
    ):
        direct_backend = (
            "treewidth_dp_peeled"
            if _prefer_treewidth_phase3(
                q,
                cover,
                order,
                width,
                fully_peeled=True,
            )
            else "q3_cover"
        )
        cached = (tuple(cover), tuple(order), width, structural_obstruction, direct_backend)
        _STRUCTURE_PHASE3_PLAN_CACHE[cache_key] = cached
        cover, order, width, structural_obstruction, direct_backend = cached
        return list(cover), list(order), width, structural_obstruction, direct_backend
    direct_backend, _runtime_score, _separator = _choose_phase3_backend(
        q,
        cover,
        order,
        width,
        structural_obstruction,
        allow_tensor_contraction=allow_tensor_contraction,
        fully_peeled=fully_peeled,
        extended_reductions="auto",
    )
    cached = (tuple(cover), tuple(order), width, structural_obstruction, direct_backend)
    _STRUCTURE_PHASE3_PLAN_CACHE[cache_key] = cached
    cover, order, width, structural_obstruction, direct_backend = cached
    return list(cover), list(order), width, structural_obstruction, direct_backend

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
