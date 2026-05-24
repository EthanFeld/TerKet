"""
Shared runtime helpers and public engine plumbing for TerKet.
"""

from __future__ import annotations

import bisect
import cmath
from collections import deque
from fractions import Fraction
from functools import lru_cache
import hashlib
import heapq
import importlib
from itertools import combinations
import math
import os
import struct
import sys
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Sequence, overload

import numpy as np

from .cache import _BoundedMemoCache, make_bounded_cache, register_lru_cache
from .cubic_arithmetic import CubicFunction, PhaseFunction, detect_factorization
from .native import (
    _env_flag_enabled,
    _get_quimb_tensor_module,
    _import_quimb_tensor_module,
    _kahypar_available,
    _load_schur_native_module,
    _native_aff_compose_enabled as _native_aff_compose_enabled_for,
    _native_level3_enabled as _native_level3_enabled_for,
    _native_symbol as _native_symbol_from,
    _quimb_import_enabled,
    _quimb_import_reason,
)
from .scaling import (
    ScaledAmplitude,
    ScaledComplex,
    _ONE_SCALED,
    _ZERO_SCALED,
    _add_scaled_complex,
    _add_scaled_complex_arrays,
    _complex_logsum,
    _make_scaled_complex,
    _mul_scaled_complex,
    _mul_scaled_complex_arrays,
    _normalize_scaled_complex,
    _normalize_scaled_complex_arrays,
    _omega_plus_one_scaled_table,
    _omega_scaled_arrays,
    _omega_scaled_table,
    _omega_table,
    _renormalize_scaled_complex_if_needed,
    _scale_complex_array_by_half_pow2,
    _scale_complex_by_half_pow2,
    _scale_scaled_complex,
    _scaled_arrays_from_constant,
    _scaled_complex_log,
    _scaled_complex_ratio_to_plain,
    _scaled_from_complex_log,
    _scaled_list_to_arrays,
    _scaled_log2_abs,
    _scaled_phase,
    _scaled_probability_log2,
    _scaled_table_to_arrays,
    _scaled_to_complex,
    _scaled_to_plain_complex,
)
from .spec import CircuitSpec, Gate
from .state import (
    BitSequence,
    CircuitInput,
    EchelonCache,
    ExtendedReductionMode,
    ReducerInfo,
    ReductionInfo,
    SolverConfig,
    SupportsQiskitCircuit,
    _get_solver_config,
    _iter_mask_bits,
    _mask_bit,
    _mask_from_vector,
    _normalize_extended_reductions,
    _parity,
    _prepare_affine_constraint_cache,
    _row_reduce_output_constraints,
    _reset_solver_config,
    _set_solver_config,
    _solve_echelon_rhs,
    _solve_output_from_echelon as _state_solve_output_from_echelon,
)
from ._phase3.models import _Phase3BackendCandidate
from ._q3free.batch import Q3FreeBranchTemplate
from ._q3free.models import (
    _BinaryPhaseQuadraticPlan,
    _Q3FreeConstraintComponentPlan,
    _Q3FreeConstraintPlan,
    _Q3FreeCutsetCandidateEvaluation,
    _Q3FreeCutsetConditioningPlan,
    _Q3FreeExecutionPlan,
    _Q3FreeRawConstraintPlan,
    _Q3FreeRawConstraintRestrictedPlan,
    _Q3FreeResidualProjection,
    _Q3FreeReusableExecutionPlan,
)


_schur_native = _load_schur_native_module()
_FORCE_EXTRACTED_SYNC = "pytest" in sys.modules
_EXTRACTED_MODULE_CACHE: dict[str, Any] = {}


def _sync_extracted_globals(
    namespace: dict[str, Any],
    engine: Any,
    *,
    local_names: set[str] | frozenset[str],
    local_impls: Mapping[str, Any],
    baselines: Mapping[str, Any],
    missing: object,
    respect_mock_wraps: bool = False,
) -> None:
    for name, value in vars(engine).items():
        if name.startswith("__"):
            continue
        if name in local_names:
            baseline = baselines.get(name, missing)
            if respect_mock_wraps and getattr(value, "_mock_wraps", None) is baseline:
                namespace[name] = baseline
            elif value is not baseline:
                namespace[name] = value
            elif name in local_impls:
                namespace[name] = local_impls[name]
            continue
        namespace[name] = value


def _bootstrap_extracted_globals(
    namespace: dict[str, Any],
    *,
    local_names: set[str] | frozenset[str],
    local_impls: Mapping[str, Any],
    engine_module_name: str = "terket._engine_impl",
    respect_mock_wraps: bool = False,
) -> tuple[object, dict[str, Any]]:
    missing = object()
    initial_engine = importlib.import_module(engine_module_name)
    baselines = {name: getattr(initial_engine, name, missing) for name in local_names}
    _sync_extracted_globals(
        namespace,
        initial_engine,
        local_names=local_names,
        local_impls=local_impls,
        baselines=baselines,
        missing=missing,
        respect_mock_wraps=respect_mock_wraps,
    )
    return missing, baselines


def _configure_extracted_module(
    namespace: dict[str, Any],
    *,
    local_names: set[str] | frozenset[str],
    local_impls: Mapping[str, Any],
    engine_module_name: str = "terket._engine_impl",
    respect_mock_wraps: bool = False,
) -> None:
    missing, baselines = _bootstrap_extracted_globals(
        namespace,
        local_names=local_names,
        local_impls=local_impls,
        engine_module_name=engine_module_name,
        respect_mock_wraps=respect_mock_wraps,
    )

    def _sync_from_engine(engine) -> None:
        current_local_impls = namespace.get("_LOCAL_IMPLS", local_impls)
        _sync_extracted_globals(
            namespace,
            engine,
            local_names=local_names,
            local_impls=current_local_impls,
            baselines=baselines,
            missing=missing,
            respect_mock_wraps=respect_mock_wraps,
        )

    namespace["_MISSING"] = missing
    namespace["_ENGINE_LOCAL_BASELINES"] = baselines
    namespace["_sync_from_engine"] = _sync_from_engine


def _load_extracted_module(module_name: str):
    module = _EXTRACTED_MODULE_CACHE.get(module_name)
    if module is None:
        module = importlib.import_module(f".{module_name}", __package__)
        _EXTRACTED_MODULE_CACHE[module_name] = module
    return module


def _engine_module() -> Any:
    return sys.modules.get(f"{__package__}._engine_impl", sys.modules[__name__])


def _native_module() -> Any:
    return getattr(_engine_module(), "_schur_native", _schur_native)


def _prepare_extracted_module(module: Any, engine: Any) -> None:
    local_names = getattr(module, "_LOCAL_NAMES", ())
    baselines = getattr(module, "_ENGINE_LOCAL_BASELINES", None)
    if isinstance(baselines, dict):
        missing = getattr(module, "_MISSING", None)
        for name in local_names:
            baseline = baselines.get(name, missing)
            if baseline is missing and hasattr(engine, name):
                baselines[name] = getattr(engine, name)

    sync = getattr(module, "_sync_from_engine", None)
    if sync is None:
        return

    prepared = getattr(module, "_ENGINE_RUNTIME_PREPARED", False)
    if not prepared or _FORCE_EXTRACTED_SYNC:
        sync(engine)
        try:
            module._ENGINE_RUNTIME_PREPARED = True
        except Exception:
            pass

def _native_level3_enabled(q: PhaseFunction | None = None) -> bool:
    return _native_level3_enabled_for(q, native_module=_native_module())

def _native_aff_compose_enabled() -> bool:
    return _native_aff_compose_enabled_for(native_module=_native_module())

def _native_symbol(name: str):
    return _native_symbol_from(name, native_module=_native_module())

def _call_extracted(module_name: str, attr: str, *args, **kwargs):
    engine = _engine_module()
    module = _load_extracted_module(module_name)
    _prepare_extracted_module(module, engine)

    target = getattr(module, attr)
    local_impls = getattr(module, "_LOCAL_IMPLS", None)
    if isinstance(local_impls, dict) and attr in local_impls:
        baselines = getattr(module, "_ENGINE_LOCAL_BASELINES", None)
        baseline = baselines.get(attr) if isinstance(baselines, dict) else None
        local_impl = local_impls[attr]
        engine_value = getattr(engine, attr, None)
        wrapped = getattr(engine_value, "_mock_wraps", None)

        if baseline is not None and wrapped is baseline:
            if hasattr(engine_value, "_mock_check_sig"):
                engine_value._mock_check_sig(*args, **kwargs)
            if hasattr(engine_value, "_increment_mock_call"):
                engine_value._increment_mock_call(*args, **kwargs)
            try:
                engine_value._mock_wraps = local_impl
                return engine_value._execute_mock_call(*args, **kwargs)
            finally:
                engine_value._mock_wraps = wrapped

        if baseline is not None and engine_value is not None and engine_value is not baseline:
            target = engine_value
        elif baseline is not None and target is baseline:
            target = local_impl
        elif engine_value is target:
            if (
                getattr(engine_value, "__module__", None) == engine.__name__
                and getattr(engine_value, "__name__", None) == attr
            ):
                target = local_impl

    return target(*args, **kwargs)


def _call_extracted_local(module_name: str, attr: str, *args, **kwargs):
    engine = _engine_module()
    module = _load_extracted_module(module_name)
    _prepare_extracted_module(module, engine)

    local_impls = getattr(module, "_LOCAL_IMPLS", None)
    if isinstance(local_impls, dict) and attr in local_impls:
        return local_impls[attr](*args, **kwargs)
    return getattr(module, attr)(*args, **kwargs)


def _make_extracted_forwarder(module_name: str, attr: str, *, call_local: bool = False):
    caller = _call_extracted_local if call_local else _call_extracted

    def _forwarder(*args, **kwargs):
        return caller(module_name, attr, *args, **kwargs)

    _forwarder.__name__ = attr
    return _forwarder


def _bind_extracted_forwarders(
    namespace_or_module_name: dict[str, Any] | str,
    *attrs: str,
    call_local: bool = False,
) -> None:
    if isinstance(namespace_or_module_name, dict):
        namespace = namespace_or_module_name
        module_name, attrs = attrs[0], attrs[1:]
    else:
        namespace = sys._getframe(1).f_globals
        module_name = namespace_or_module_name

    for attr in attrs:
        namespace[attr] = _make_extracted_forwarder(
            module_name,
            attr,
            call_local=call_local,
        )


def _make_synced_local_impl_forwarder(
    runtime_module_name: str,
    module_name: str,
    attr: str,
):
    def _forwarder(*args, **kwargs):
        runtime_module = sys.modules[runtime_module_name]
        module = _load_extracted_module(module_name)
        _prepare_extracted_module(module, runtime_module)
        return module._LOCAL_IMPLS[attr](*args, **kwargs)

    _forwarder.__name__ = attr
    return _forwarder


def _bind_synced_local_impl_forwarders(
    namespace_or_module_name: dict[str, Any] | str,
    *attrs: str,
) -> None:
    if isinstance(namespace_or_module_name, dict):
        namespace = namespace_or_module_name
        module_name, attrs = attrs[0], attrs[1:]
    else:
        namespace = sys._getframe(1).f_globals
        module_name = namespace_or_module_name

    runtime_module_name = namespace["__name__"]
    for attr in attrs:
        namespace[attr] = _make_synced_local_impl_forwarder(
            runtime_module_name,
            module_name,
            attr,
        )

# Small q3-free kernels are faster to enumerate directly than to build transfer
# tables or dynamic-programming state for them.
_Q3_FREE_BRUTE_FORCE_CUTOFF = 8
# Treewidth-DP tables are pure Python lists of boxed complex values, so the
# practical memory ceiling is higher than 2^w * 16 bytes. Width 18 keeps the
# general Phase-3 DP region modest.
_Q3_TREEWIDTH_DP_MAX_WIDTH = 18
# Fully peeled cubic kernels are much friendlier than arbitrary residual
# Phase-3 instances, so allow a slightly wider treewidth regime there and gate
# it by the actual DP work estimate instead of width alone.
_Q3_TREEWIDTH_DP_PEELED_MAX_WIDTH = 24
_Q3_TREEWIDTH_DP_PEELED_MAX_WORK = 30_000_000_000
_Q3_TREEWIDTH_CUTSET_MAX_SIZE = 8
_Q3_TREEWIDTH_CUTSET_MAX_CANDIDATES = 12
# Some large sparse cubic kernels become much worse after exact eliminations:
# the eliminator removes many q1/q2-only variables, but densifies the tiny q3
# support into a small hard core that then falls back to q3-cover recursion.
# If the original kernel already has a cheap direct treewidth plan, take it.
_PRE_EXACT_PHASE3_TREEWIDTH_ESCAPE_MIN_VARS = 256
_PRE_EXACT_PHASE3_TREEWIDTH_ESCAPE_MAX_Q3_TERMS = 256
_PRE_EXACT_PHASE3_TREEWIDTH_ESCAPE_MAX_COVER = 8
_PRE_EXACT_PHASE3_TREEWIDTH_ESCAPE_MAX_WORK = 50_000_000
_PYTHON_TREEWIDTH_BATCH_MAX_WIDTH = 8
# The exact q3-free summation path otherwise falls back to a feedback-variable
# transfer solver whose memo table can explode on wide but still tractable q2
# graphs. Allow one or two extra width units there before giving up on DP.
_Q3_FREE_SUM_TREEWIDTH_MAX_WIDTH = 20
# When the native scaled factor-table eliminator is available, modestly wider
# q3-free treewidth instances become practical even above Clifford+T precision.
_Q3_FREE_SUM_TREEWIDTH_NATIVE_MAX_WIDTH = 31
_Q3_FREE_SUM_TREEWIDTH_NATIVE_MAX_WORK = 30_000_000_000
# If only a small set of variables sits outside BL26's qubit quadratic
# coefficient class, branch on that set and push each branch through the exact
# quadratic/constraint eliminators instead of feeding a large generic q3-free
# component to the feedback-variable solver.
_Q3_FREE_NONQUADRATIC_BRANCH_MAX_SUPPORT = 10
# Exact unary-character expansion over a half-phase q2 core is still
# exponential in the number of non-binary unary phases, but it can reuse the
# polynomial-time binary quadratic core exactly and is much cheaper than the
# generic fallback when that hard-unary support is small.
_Q3_FREE_HALF_PHASE_UNARY_EXPANSION_MAX_SUPPORT = 18
_Q3_FREE_HALF_PHASE_UNARY_EXPANSION_BATCH_SIZE = 4096
_Q3_FREE_MEDIATOR_BATCH_MIN_ROWS = 128
_Q3_FREE_BAD_Q2_COVER_MAX_SIZE = 12
_Q3_FREE_BAD_Q2_COVER_DISPATCH_MIN_VARS = 24
_Q3_FREE_BAD_Q2_COVER_DISPATCH_MIN_DENSITY = 0.15
_Q3_FREE_BAD_Q2_COVER_DISPATCH_MIN_SUPPORT_FACTOR = 3
_Q3_FREE_HALF_PHASE_CLUSTER_MAX_CLUSTER_SIZE = 8
_Q3_FREE_HALF_PHASE_CLUSTER_MAX_BOUNDARY = 6
_Q3_FREE_SMALL_BOUNDARY_REGION_MIN_SIZE = 4
_Q3_FREE_SMALL_BOUNDARY_REGION_MAX_SIZE = 24
_Q3_FREE_SMALL_BOUNDARY_REGION_MAX_BOUNDARY = 4
_Q3_FREE_SMALL_BOUNDARY_REGION_MAX_REGIONS = 16
_Q3_FREE_ORDER_GUIDED_CUTSET_MAX_PEAKS = 12
_Q3_FREE_ORDER_HINT_MAX_WIDTH = 12
_Q3_FREE_CHEAP_ORDER_HINT_MIN_VARS = 128
_Q3_FREE_SERIES_CORE_RECURSE_MIN_VARS = 128
_Q3_FREE_SERIES_CORE_RECURSE_MIN_SHRINK = 64
_Q3_FREE_CUTSET_MAX_SIZE = 6
_Q3_FREE_CUTSET_CANDIDATE_POOL = 24
_Q3_FREE_CUTSET_BEAM_WIDTH = 4
_Q3_FREE_CUTSET_BRANCHES_PER_STATE = 3
_Q3_FREE_CUTSET_TENSOR_HINT_MIN_VARS = 128
_Q3_FREE_CUTSET_TENSOR_HINT_MAX_VARS = 384
_Q3_FREE_CUTSET_TENSOR_HINT_MAX_REPEATS = 4
_Q3_FREE_CUTSET_TENSOR_HINT_MAX_TIME = 2.0
_Q3_FREE_CUTSET_TENSOR_HINT_TARGET_WIDTH = 14
_Q3_FREE_REUSABLE_CUTSET_MIN_LAMBDA_VARS = 2
_Q3_FREE_REUSABLE_CUTSET_MIN_TREEWIDTH = 12
_Q3_FREE_REUSABLE_CUTSET_MAX_LOG2_REUSE = 4
_Q3_FREE_REUSABLE_EXECUTION_PLAN_MIN_VARS = 24
_Q3_FREE_ONE_SHOT_CUTSET_MIN_TREEWIDTH = 18
_Q3_FREE_ONE_SHOT_CUTSET_ACTIVATION_WIDTH = 30
_Q3_FREE_ONE_SHOT_CUTSET_MAX_SIZE = 10
_Q3_FREE_ONE_SHOT_CUTSET_CANDIDATE_POOL = 40
_Q3_FREE_ONE_SHOT_CUTSET_BEAM_WIDTH = 6
_Q3_FREE_ONE_SHOT_CUTSET_BRANCHES_PER_STATE = 4
_Q3_FREE_ONE_SHOT_STAGNATION_LIMIT = 2
_Q3_FREE_ONE_SHOT_LOCAL_SEARCH_PASSES = 2
_Q3_FREE_ONE_SHOT_LOCAL_SEARCH_TOPK = 8
_Q3_FREE_ONE_SHOT_DIRECT_MIN_VARS = 88
_Q3_FREE_ONE_SHOT_DIRECT_MIN_WIDTH = 24
_Q3_FREE_ONE_SHOT_DIRECT_MAX_REMAINING_WIDTH = 16
_Q3_FREE_ONE_SHOT_DIRECT_MIN_Q2_PER_VAR = 3.0
_Q2_SEPARATOR_ORDER_MIN_VARS = 48
_Q2_SEPARATOR_ORDER_BASE_CASE = 24
_Q2_SEPARATOR_ORDER_MAX_LAYER_SPAN = 2
_Q2_SEPARATOR_ORDER_MAX_SEPARATOR = 96
_Q2_SEPARATOR_ORDER_MAX_BALANCE = 0.85
_SCHUR_COMPLEMENT_CROSSOVER_FVS = 12
_Q3_FREE_DENSE_PLAN_MIN_DEGREE = 24
_Q3_FREE_DENSE_PLAN_MIN_DENSITY = 0.20
# Small dense residual kernels are the only ones where quimb contraction
# planning is consistently cheaper than branching or pure-Python DP.
_Q3_TENSOR_CONTRACTION_MAX_VARS = 24
_MAX_ARBITRARY_PHASE_FACTOR_SCOPE = 24
_MAX_ARBITRARY_PATH_SUM_PY_WIDTH = 24
_MAX_ARBITRARY_PATH_SUM_NATIVE_WIDTH = 26
_MAX_ARBITRARY_PATH_SUM_WORK = 30_000_000_000
_MAX_ARBITRARY_PATH_SUM_TABLE_ENTRIES = 1 << 25
_MAX_ARBITRARY_PATH_SUM_CUTSET_SIZE = 16
_MAX_ARBITRARY_PATH_SUM_CUTSET_CANDIDATES = 16
_ARBITRARY_BP_MAX_ITERS = 50
_ARBITRARY_FACTOR_BP_MAX_ITERS = 25
_ARBITRARY_FACTOR_BP_LARGE_EDGE_THRESHOLD = 250_000
_ARBITRARY_FACTOR_BP_LARGE_MAX_ITERS = 8
_ARBITRARY_BP_DAMPING = 0.5
_ARBITRARY_BP_TOL = 1e-8
_ARBITRARY_BP_DIRECT_PROB_LOG2_TOL = 1e-6
_ARBITRARY_BP_HEURISTIC_SCHEDULES = (
    (50, 0.5),
    (50, 0.8),
    (100, 0.5),
)
_ARBITRARY_BP_HEURISTIC_MAX_LOG2_ABS_SPREAD = 2.0
_ARBITRARY_BP_HEURISTIC_MAX_PHASE_SPREAD = 0.5
_ARBITRARY_BP_HEURISTIC_BOUND_LOG2_TOL = 1e-6
_Q3_HYBRID_CONTRACTION_MAX_VARS = 60
_Q3_HYBRID_CONTRACTION_MAX_WIDTH = 25
_PAULI_EXPBOX_FINAL_DEAD_FLUSH_MAX_CANDIDATES = 1024
# Below this width, the Python treewidth DP typically beats contraction-planner
# overhead on the same reduced cubic core.
_Q3_TENSOR_CONTRACTION_TREEWIDTH_CROSSOVER = 5
# Require at least moderately dense factor graphs before tensor contraction
# starts paying for its extra setup work.
_Q3_TENSOR_CONTRACTION_MIN_FACTOR_DENSITY = 2.0
# Exact branch-and-bound q3-cover search remains cheap around twenty branch
# variables on the benchmark families targeted by this package.
_Q3_VERTEX_COVER_EXACT_SIZE_CUTOFF = 20
# Dense q3 hypergraphs can defeat the exact cover search before size alone
# does, so cap the edge count separately.
_Q3_VERTEX_COVER_EXACT_EDGE_CUTOFF = 256
# The exact cover fallback keeps a failed-state table. Random dense cores can
# otherwise spend gigabytes proving a marginally smaller cover does not exist.
_Q3_VERTEX_COVER_EXACT_FAILED_STATE_CUTOFF = 200_000
# Exact XOR basis simplification is only worth probing on moderate cubic cores.
_Q3_BASIS_SIMPLIFY_MAX_VARS = 40
_Q3_BASIS_SIMPLIFY_MAX_ACTIVE_VARS = 12
_Q3_BASIS_SIMPLIFY_MAX_PASSES = 4
# Bounded phase-function structural optimization searches over exact XOR basis
# changes after conversion to a PhaseFunction. The goal is solver-facing:
# reduce the live cubic core first, then avoid q2 dense-core formation.
_PHASE_STRUCTURE_OPT_MAX_VARS = 48
_PHASE_STRUCTURE_OPT_MAX_ACTIVE_VARS = 10
_PHASE_STRUCTURE_OPT_BEAM_WIDTH = 4
_PHASE_STRUCTURE_OPT_MAX_PASSES = 3
_PHASE_STRUCTURE_OPT_TWO_SOURCE_LIMIT = 3
_PHASE_STRUCTURE_LOCAL_REGION_MAX_VARS = 24
_PHASE_STRUCTURE_LOCAL_REGION_RADIUS = 2
_PHASE_STRUCTURE_LOCAL_MAX_CENTERS = 6
_PHASE_STRUCTURE_LOCAL_MAX_PASSES = 3
_PHASE_STRUCTURE_LOCAL_CANDIDATE_POOL = 6
# Cubic residuals reach this optimizer only after exact eliminations stall, so
# spend a little more search budget there than on q3-free rewrites.
_PHASE_STRUCTURE_CUBIC_OPT_MAX_VARS = 72
_PHASE_STRUCTURE_CUBIC_OPT_MAX_ACTIVE_VARS = 14
_PHASE_STRUCTURE_CUBIC_OPT_BEAM_WIDTH = 6
_PHASE_STRUCTURE_CUBIC_OPT_MAX_PASSES = 4
_PHASE_STRUCTURE_CUBIC_LOCAL_REGION_MAX_VARS = 32
_PHASE_STRUCTURE_CUBIC_LOCAL_MAX_CENTERS = 10
_PHASE_STRUCTURE_CUBIC_LOCAL_MAX_PASSES = 4
_PHASE_STRUCTURE_CUBIC_LOCAL_CANDIDATE_POOL = 10
# Very large q3-free kernels can spend minutes in optional structure-scoring
# and mediator-order planning without changing the eventual exact backend.
# Past this size, skip those optional rewrites and let the core q3-free planner
# work directly on the sparse graph.
_Q3_FREE_OPTIONAL_REWRITE_MAX_VARS = 1024
# Residual cubic treewidth planning uses a tighter local refinement than the
# q3-free planner because backend choice depends directly on width/work here.
_PHASE3_TREEWIDTH_REFINE_MAX_WIDTH = 28
_PHASE3_TREEWIDTH_REFINE_MAX_VARS = 256
_PHASE3_TREEWIDTH_REFINE_MAX_PASSES = 3
_PHASE3_TREEWIDTH_REFINE_MAX_HOTSPOTS = 12
_PHASE3_TREEWIDTH_REFINE_MOVE_RADIUS = 3
# Very large exact-elimination chains benefit from prioritizing the cheapest
# local eliminations first; smaller kernels often do better with the original
# first-hit ordering that exposes strong parity structure early.
_EXACT_ELIM_CHEAP_ACTION_MIN_VARS = 5000
# Branching on a tiny projected separator can beat monolithic q3 cover search.
_Q3_SEPARATOR_MAX_SIZE = 2
_Q3_SEPARATOR_MAX_CANDIDATES = 12
# Auto mode keeps the pre-Schur rewrite off large gate sequences where the
# rewrite walk costs more than the local cancellations it tends to expose.
_EXTENDED_REWRITE_AUTO_MAX_GATES = 128
_EXTENDED_REWRITE_TRIGGER_GATES = frozenset(
    {
        "t",
        "tdg",
        "s",
        "sdg",
        "z",
        "sx",
        "sxdg",
        "rz_pi_16",
        "rz_pi_16_dg",
        "rz_pi_32",
        "rz_pi_32_dg",
        "rz_dyadic",
        "rzz_dyadic",
        "rz_arbitrary",
        "pauli_expbox",
    }
)
# Auto mode enables the new q3 reductions only once the residual cubic
# obstruction is genuinely large.
_EXTENDED_Q3_AUTO_MIN_OBSTRUCTION = 8
# On large residuals, spending more time on Phase-2 branching heuristics rarely
# beats committing to a Phase-3 plan early.
_PHASE2_TREEWIDTH_ESCAPE_MIN_VARS = 64
# The optional native affine composer packs q3 indices into 21-bit lanes inside
# a uint64_t; larger variable indices fall back to the pure-Python path.
_NATIVE_AFF_COMPOSE_Q3_INDEX_LIMIT = 1 << 21
# Large Pauli-expectation batches can avoid replaying a long inverse suffix gate
# by gate once the suffix row trajectory is fixed and calibration matches.
_DIRECT_POST_REPLAY_MIN_SUFFIX_GATES = 256
_DIRECT_POST_REPLAY_MIN_OBSERVABLES = 8
# Keep the direct two-partner parity rewrite on small kernels where Python-side
# bookkeeping is cheap; large kernels are faster through native aff_compose.
_DIRECT_TWO_PARTNER_CONSTRAINT_MAX_VARS = 128
_CUPY_MODULE = None
_CUPY_IMPORT_ERROR = None

try:
    from .cubic_contraction import plan_contraction, execute_plan_cpu
    _HAS_CUBIC_CONTRACTION = True
except ImportError:
    _HAS_CUBIC_CONTRACTION = False

@lru_cache(maxsize=1 << 16)
def _support_from_mask(*args, **kwargs):
    return _call_extracted("_state_runtime", "_support_from_mask", *args, **kwargs)

register_lru_cache("engine.native_support_from_mask", _support_from_mask)

def _should_apply_extended_gate_rewrite(
    mode: ExtendedReductionMode | str | None,
    gates: Sequence[Gate],
) -> bool:
    normalized = _normalize_extended_reductions(mode)
    if normalized == "always":
        return True
    if normalized == "never":
        return False
    if len(gates) > _EXTENDED_REWRITE_AUTO_MAX_GATES:
        return False
    return any(str(gate[0]) in _EXTENDED_REWRITE_TRIGGER_GATES for gate in gates)

def _should_apply_extended_q3_reductions(
    q: PhaseFunction,
    mode: ExtendedReductionMode | str | None,
) -> bool:
    normalized = _normalize_extended_reductions(mode)
    if normalized == "always":
        return bool(q.q3)
    if normalized == "never" or not q.q3:
        return False
    core_vars, _ = _q3_hypergraph_2core(q)
    return _q3_core_cover_size(q, core_vars) >= _EXTENDED_Q3_AUTO_MIN_OBSTRUCTION

# ==================================================================
# Helpers, scaling, and reducer metadata
# ==================================================================

__all__ = [name for name in globals() if not name.startswith("__")]
