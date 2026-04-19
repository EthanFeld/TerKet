"""
Qubit Clifford+T amplitude reduction engine for TerKet.

The reducer builds cubic phase functions over binary free variables, applies
exact affine and quadratic eliminations, sums every q3-free kernel exactly, and
uses Phase-3 fallback methods only when a genuine cubic core remains.

The public metadata distinguishes:

- `cubic_obstruction`: residual genuine q3 obstruction after exact reductions
- `gauss_obstruction`: broader obstruction to a BL26-style quadratic-tensor
  contraction after exact reductions. This includes both genuine q3 structure
  and any surviving qubit q1/q2 coefficients outside the qubit quadratic
  coefficient groups.
- `cost_model_r`: exponent paid by the chosen Phase-3 backend

Module layout:

1. Helpers, scaling primitives, and reducer metadata
2. `SchurState` circuit construction and output constraint solving
3. Recursive exact reduction plus the q3-free solver
4. Phase-3 backend planning and execution
5. Variable classification and elimination rules
6. Public API wrappers
"""

from __future__ import annotations

import bisect
import cmath
from collections import OrderedDict
import contextvars
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import heapq
import importlib.machinery
import importlib.util
from itertools import combinations
import math
import os
from pathlib import Path
import platform
import struct
import sys
from types import MappingProxyType
from typing import Any, Literal, Protocol, Sequence, TypedDict, overload

import numpy as np

from .circuits import CircuitSpec, Gate, _rewrite_gate_sequence
from .cubic_arithmetic import CubicFunction, PhaseFunction, detect_factorization

def _load_schur_native_module():
    package_dir = Path(__file__).resolve().parent
    module_name = "terket._schur_native"
    suffixes = tuple(getattr(importlib.machinery, "EXTENSION_SUFFIXES", ()))
    in_tree_candidates: list[Path] = []
    for suffix in suffixes:
        in_tree_candidates.extend(sorted(package_dir.glob(f"_schur_native*{suffix}")))
    in_tree_path = max(in_tree_candidates, key=lambda path: path.stat().st_mtime_ns) if in_tree_candidates else None
    build_candidates = sorted(package_dir.parent.glob("build/lib.*/terket/_schur_native*"))

    candidate_paths: list[Path] = []
    if build_candidates:
        freshest_build = max(build_candidates, key=lambda path: path.stat().st_mtime_ns)
        if in_tree_path is None or freshest_build.stat().st_mtime_ns > in_tree_path.stat().st_mtime_ns:
            candidate_paths.append(freshest_build)
    if in_tree_path is not None:
        candidate_paths.append(in_tree_path)

    for candidate in candidate_paths:
        try:
            spec = importlib.util.spec_from_file_location(module_name, candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        except ImportError:
            sys.modules.pop(module_name, None)
            continue
        except OSError:
            sys.modules.pop(module_name, None)
            continue
    return None


_schur_native = _load_schur_native_module()


BitSequence = Sequence[int]
AffineRows = Sequence[Sequence[int]]
ScaledComplex = tuple[complex, int]
ExtendedReductionMode = Literal["auto", "always", "never"]


class SupportsQiskitCircuit(Protocol):
    """Structural type for Qiskit-style circuit inputs."""

    num_qubits: int
    data: Any


CircuitInput = CircuitSpec | str | SupportsQiskitCircuit


class ReducerInfo(TypedDict):
    """Reducer metadata returned by ``reduce_and_sum()``."""

    quad: int
    constraint: int
    branched: int
    remaining: int
    structural_obstruction: int
    gauss_obstruction: int
    cost_r: int
    phase_states: int
    phase_splits: int
    phase3_backend: str | None


class ReductionInfo(TypedDict):
    """Public amplitude-query metadata returned by the high-level API."""

    initial_free: int
    quad_eliminated: int
    constraint_eliminated: int
    branched: int
    remaining_free: int
    branches: int
    cost_model_r: int
    cubic_obstruction: int
    has_cubic_obstruction: bool
    gauss_obstruction: int
    has_gauss_obstruction: bool
    phase_states: int
    phase_splits: int
    phase3_backend: str | None
    is_zero: bool


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
_Q3_FREE_ORDER_HINT_MAX_WIDTH = 12
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
_Q3_FREE_ONE_SHOT_CUTSET_MIN_TREEWIDTH = 18
_Q3_FREE_ONE_SHOT_CUTSET_ACTIVATION_WIDTH = 30
_Q3_FREE_ONE_SHOT_CUTSET_MAX_SIZE = 10
_Q3_FREE_ONE_SHOT_CUTSET_CANDIDATE_POOL = 40
_Q3_FREE_ONE_SHOT_CUTSET_BEAM_WIDTH = 6
_Q3_FREE_ONE_SHOT_CUTSET_BRANCHES_PER_STATE = 4
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
_Q3_TENSOR_CONTRACTION_OPTIMIZE = "greedy"
_Q3_HYBRID_CONTRACTION_MAX_VARS = 60
_Q3_HYBRID_CONTRACTION_MAX_WIDTH = 25
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
# Branching on a tiny projected separator can beat monolithic q3 cover search.
_Q3_SEPARATOR_MAX_SIZE = 2
_Q3_SEPARATOR_MAX_CANDIDATES = 12
# Auto mode keeps the pre-Schur rewrite off large gate sequences where the
# rewrite walk costs more than the local cancellations it tends to expose.
_EXTENDED_REWRITE_AUTO_MAX_GATES = 128
# Auto mode enables the new q3 reductions only once the residual cubic
# obstruction is genuinely large.
_EXTENDED_Q3_AUTO_MIN_OBSTRUCTION = 8
# On large residuals, spending more time on Phase-2 branching heuristics rarely
# beats committing to a Phase-3 plan early.
_PHASE2_TREEWIDTH_ESCAPE_MIN_VARS = 64
_SQRT2 = math.sqrt(2.0)
_INV_SQRT2 = 1.0 / _SQRT2


@dataclass(frozen=True)
class SolverConfig:
    """User-tunable solver preference knobs for TerKet's exact phase-sum backends.

    All parameters are preferences only — changing them never affects correctness,
    only the trade-off between runtime and search quality.

    Parameters
    ----------
    cutset_max_size:
        Maximum number of variables conditioned on in the regular cutset path
        (default 6, i.e. 2**6 = 64 branches).  Increase for harder circuits.
    cutset_candidate_pool:
        Number of candidate cutset variables evaluated during greedy search.
    cutset_beam_width:
        Beam width for the beam-search expansion of the cutset.
    cutset_branches_per_state:
        Candidate expansions tried per beam-search state.
    one_shot_cutset_max_size:
        Maximum cutset size for the one-shot (single-amplitude) path, which
        runs a stronger search and is activated for high-treewidth components
        (default 10, i.e. up to 2**10 = 1024 branches).
    one_shot_cutset_candidate_pool:
        Candidate pool size for the one-shot cutset search.
    one_shot_cutset_beam_width:
        Beam width for the one-shot cutset beam search.
    one_shot_cutset_branches_per_state:
        Candidate expansions per state in the one-shot beam search.
    tensor_hint_target_width:
        Target treewidth that kahypar tries to achieve via slicing (default 14).
        Raising this allows the optimizer to find fewer slice variables at the
        cost of harder per-slice evaluation.
    tensor_hint_max_repeats:
        Number of kahypar optimization repeats (default 4).
    tensor_hint_max_time:
        Wall-clock budget in seconds for each kahypar optimization run (default 2.0).
    tensor_hint_min_vars:
        Minimum component size before the tensor-hint path is attempted (default 128).
    tensor_hint_max_vars:
        Maximum component size for the tensor-hint path (default 384).
    """

    cutset_max_size: int = _Q3_FREE_CUTSET_MAX_SIZE
    cutset_candidate_pool: int = _Q3_FREE_CUTSET_CANDIDATE_POOL
    cutset_beam_width: int = _Q3_FREE_CUTSET_BEAM_WIDTH
    cutset_branches_per_state: int = _Q3_FREE_CUTSET_BRANCHES_PER_STATE
    one_shot_cutset_max_size: int = _Q3_FREE_ONE_SHOT_CUTSET_MAX_SIZE
    one_shot_cutset_candidate_pool: int = _Q3_FREE_ONE_SHOT_CUTSET_CANDIDATE_POOL
    one_shot_cutset_beam_width: int = _Q3_FREE_ONE_SHOT_CUTSET_BEAM_WIDTH
    one_shot_cutset_branches_per_state: int = _Q3_FREE_ONE_SHOT_CUTSET_BRANCHES_PER_STATE
    tensor_hint_target_width: int = _Q3_FREE_CUTSET_TENSOR_HINT_TARGET_WIDTH
    tensor_hint_max_repeats: int = _Q3_FREE_CUTSET_TENSOR_HINT_MAX_REPEATS
    tensor_hint_max_time: float = _Q3_FREE_CUTSET_TENSOR_HINT_MAX_TIME
    tensor_hint_min_vars: int = _Q3_FREE_CUTSET_TENSOR_HINT_MIN_VARS
    tensor_hint_max_vars: int = _Q3_FREE_CUTSET_TENSOR_HINT_MAX_VARS


_DEFAULT_SOLVER_CONFIG = SolverConfig()
_SOLVER_CONFIG_VAR: contextvars.ContextVar[SolverConfig] = contextvars.ContextVar(
    "_terket_solver_config",
    default=_DEFAULT_SOLVER_CONFIG,
)


def _get_solver_config() -> SolverConfig:
    return _SOLVER_CONFIG_VAR.get()
_SCALED_RENORMALIZE_MIN = math.ldexp(1.0, -256)
_SCALED_RENORMALIZE_MAX = math.ldexp(1.0, 256)
# The optional native affine composer packs q3 indices into 21-bit lanes inside
# a uint64_t; larger variable indices fall back to the pure-Python path.
_NATIVE_AFF_COMPOSE_Q3_INDEX_LIMIT = 1 << 21
_QUIMB_TENSOR_MODULE = None
_QUIMB_TENSOR_IMPORT_ERROR = None
_CUPY_MODULE = None
_CUPY_IMPORT_ERROR = None

try:
    from .cubic_contraction import plan_contraction, execute_plan_cpu
    _HAS_CUBIC_CONTRACTION = True
except ImportError:
    _HAS_CUBIC_CONTRACTION = False

def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _quimb_import_enabled() -> bool:
    """Return whether optional quimb imports are allowed in this process."""
    return not _env_flag_enabled("TERKET_DISABLE_QUIMB")


def _quimb_import_reason() -> str:
    """Explain why quimb-backed paths are unavailable."""
    if _env_flag_enabled("TERKET_DISABLE_QUIMB"):
        return "quimb support is disabled by TERKET_DISABLE_QUIMB."
    if _QUIMB_TENSOR_IMPORT_ERROR is not None:
        return f"quimb import failed: {_QUIMB_TENSOR_IMPORT_ERROR}"
    return "quimb is not installed."


def _gpu_import_enabled() -> bool:
    """Return whether optional GPU paths are enabled in this process."""
    if _env_flag_enabled("TERKET_DISABLE_GPU"):
        return False
    if "TERKET_ENABLE_GPU" in os.environ:
        return _env_flag_enabled("TERKET_ENABLE_GPU")
    return True


def _import_quimb_tensor_module():
    """Import ``quimb.tensor`` while avoiding Python's slow Windows WMI probe."""
    if sys.platform != "win32" or not hasattr(platform, "_wmi_query"):
        import quimb.tensor as qtn

        return qtn

    original_wmi_query = platform._wmi_query
    original_uname_cache = getattr(platform, "_uname_cache", None)

    def _disabled_wmi_query(*args, **kwargs):
        raise OSError("disabled during quimb import")

    try:
        platform._wmi_query = _disabled_wmi_query
        if hasattr(platform, "_uname_cache"):
            platform._uname_cache = None
        import quimb.tensor as qtn

        return qtn
    finally:
        platform._wmi_query = original_wmi_query
        if hasattr(platform, "_uname_cache"):
            platform._uname_cache = original_uname_cache


def _iter_mask_bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _mask_from_vector(bits):
    mask = 0
    for idx, bit in enumerate(bits):
        if int(bit) & 1:
            mask |= 1 << idx
    return mask


def _mask_bit(mask, idx):
    return (mask >> idx) & 1


def _parity(mask):
    return mask.bit_count() & 1


def _row_reduce_output_constraints(n_rows: int, rows: list[int]) -> tuple[list[int], list[int], list[int], int]:
    """Return the reduced output rows plus the row-operation witnesses."""
    row_ops = [1 << idx for idx in range(n_rows)]
    pivot_col = [-1] * n_rows
    used_mask = 0

    for row_idx in range(n_rows):
        available = rows[row_idx] & ~used_mask
        if not available:
            continue
        pivot_bit = available & -available
        pivot = pivot_bit.bit_length() - 1
        pivot_col[row_idx] = pivot
        used_mask |= pivot_bit
        for other_idx in range(n_rows):
            if other_idx != row_idx and (rows[other_idx] & pivot_bit):
                rows[other_idx] ^= rows[row_idx]
                row_ops[other_idx] ^= row_ops[row_idx]

    return rows, row_ops, pivot_col, used_mask


def _native_level3_enabled(q: PhaseFunction | None = None) -> bool:
    return _schur_native is not None and (q is None or getattr(q, "level", 3) == 3)


def _native_aff_compose_enabled() -> bool:
    return _schur_native is not None


def _native_symbol(name: str):
    """Return an optional native helper without assuming a full ABI match."""
    if _schur_native is None:
        return None
    return getattr(_schur_native, name, None)


@lru_cache(maxsize=1)
def _kahypar_available() -> bool:
    return importlib.util.find_spec("kahypar") is not None


@lru_cache(maxsize=1 << 16)
def _support_from_mask(mask):
    if _native_level3_enabled():
        return _schur_native.support_from_mask(mask)
    return tuple(_iter_mask_bits(mask))


def _get_quimb_tensor_module():
    """Return ``quimb.tensor`` when available, otherwise ``None``."""
    global _QUIMB_TENSOR_IMPORT_ERROR, _QUIMB_TENSOR_MODULE
    if not _quimb_import_enabled():
        return None
    if _QUIMB_TENSOR_MODULE is False:
        return None
    if _QUIMB_TENSOR_MODULE is None:
        try:
            qtn = _import_quimb_tensor_module()
        except Exception as exc:
            _QUIMB_TENSOR_IMPORT_ERROR = exc
            _QUIMB_TENSOR_MODULE = False
        else:
            _QUIMB_TENSOR_IMPORT_ERROR = None
            _QUIMB_TENSOR_MODULE = qtn
    return None if _QUIMB_TENSOR_MODULE is False else _QUIMB_TENSOR_MODULE


def _cupy_import_reason() -> str:
    """Explain why optional CuPy-backed paths are unavailable."""
    if not _gpu_import_enabled():
        return "GPU support is disabled by TERKET_DISABLE_GPU."
    if _CUPY_IMPORT_ERROR is not None:
        return f"cupy import failed: {_CUPY_IMPORT_ERROR}"
    return "cupy is not installed."


def _get_cupy_module():
    """Return ``cupy`` when GPU acceleration is enabled and available."""
    global _CUPY_IMPORT_ERROR, _CUPY_MODULE
    if not _gpu_import_enabled():
        return None
    if _CUPY_MODULE is False:
        return None
    if _CUPY_MODULE is None:
        try:
            import cupy as cp
        except Exception as exc:
            _CUPY_IMPORT_ERROR = exc
            _CUPY_MODULE = False
        else:
            _CUPY_IMPORT_ERROR = None
            _CUPY_MODULE = cp
    return None if _CUPY_MODULE is False else _CUPY_MODULE


def _cupy_available() -> bool:
    return _get_cupy_module() is not None


def _gpu_memory_limit_bytes() -> int | None:
    raw_limit = os.environ.get("TERKET_GPU_MEMORY_LIMIT_MB")
    if raw_limit is None or not raw_limit.strip():
        return None
    try:
        limit_mb = float(raw_limit)
    except ValueError:
        return None
    if limit_mb <= 0:
        return None
    return int(limit_mb * 1024 * 1024)


def _gpu_memory_budget_bytes(cp=None) -> int | None:
    if cp is None:
        cp = _get_cupy_module()
    if cp is None:
        return _gpu_memory_limit_bytes()

    limit = _gpu_memory_limit_bytes()
    try:
        free_bytes, _ = cp.cuda.Device().mem_info
    except Exception:
        return limit
    return free_bytes if limit is None else min(int(free_bytes), limit)


def _normalize_extended_reductions(mode: ExtendedReductionMode | str | None) -> ExtendedReductionMode:
    if mode is None:
        return "auto"
    normalized = str(mode).strip().lower()
    if normalized not in {"auto", "always", "never"}:
        raise ValueError(
            f"extended_reductions must be one of 'auto', 'always', or 'never'; received {mode!r}."
        )
    return normalized


def _should_apply_extended_gate_rewrite(
    mode: ExtendedReductionMode | str | None,
    gates: Sequence[Gate],
) -> bool:
    normalized = _normalize_extended_reductions(mode)
    if normalized == "always":
        return True
    if normalized == "never":
        return False
    return len(gates) <= _EXTENDED_REWRITE_AUTO_MAX_GATES


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

class _ReductionContext:
    """Per-query memo tables for affine substitutions and reduced branch states."""

    def __init__(
        self,
        preserve_scale: bool = False,
        allow_tensor_contraction: bool = True,
        extended_reductions: ExtendedReductionMode | str = "auto",
    ) -> None:
        self.affine_compose_cache: dict[tuple[Any, ...], PhaseFunction] = {}
        self.fix_variables_cache: dict[tuple[Any, ...], PhaseFunction] = {}
        self.reduce_cache: dict[tuple[Any, ...], tuple[ScaledComplex, ReducerInfo]] = {}
        self.q3_free_constraint_plan_cache: dict[tuple[Any, ...], Any] = {}
        self.preserve_scale = preserve_scale
        self.allow_tensor_contraction = allow_tensor_contraction
        self.extended_reductions = _normalize_extended_reductions(extended_reductions)


class _BoundedMemoCache(OrderedDict):
    """Small LRU cache keyed by compact digests rather than full phase structures."""

    def __init__(self, max_entries: int):
        super().__init__()
        self.max_entries = max_entries

    def get(self, key, default=None):
        try:
            value = super().__getitem__(key)
        except KeyError:
            return default
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if self.max_entries <= 0:
            return
        if key in self:
            super().__delitem__(key)
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.max_entries:
            self.popitem(last=False)


@dataclass(frozen=True)
class EchelonCache:
    """Reusable row-echelon form of the output constraint matrix."""

    n: int
    m: int
    echelon_rows: tuple[int, ...]
    pivot_col: tuple[int, ...]
    used_mask: int
    row_ops: tuple[int, ...]
    free_vars: tuple[int, ...]
    gamma_masks: tuple[int, ...]
    n_free: int


@dataclass(frozen=True)
class _ArbitraryPhaseTerm:
    """Deferred exact phase on an affine Boolean output of the current state."""

    row_mask: int
    offset: int
    angle: float


@dataclass(frozen=True)
class _ArbitraryPhaseBranchPlan:
    """Independent branch basis for deferred arbitrary affine phases."""

    basis_masks: tuple[int, ...]
    term_dependency_masks: tuple[int, ...]
    term_offsets: tuple[int, ...]
    term_angles: tuple[float, ...]


@dataclass(frozen=True)
class _BinaryPhaseQuadraticPlan:
    """Precomputed elimination schedule for a binary +/-1 quadratic phase."""

    n: int
    residual_active_count: int
    half_pow2_exp: int
    partner_swaps: tuple[int, ...]
    pivot_swaps: tuple[int, ...]
    c1_rows: tuple[np.ndarray, ...]
    c2_rows: tuple[np.ndarray, ...]
    c1_and_c2_rows: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class _HalfPhaseMediatorSpec:
    """One eliminated mediator variable and its remaining-core neighborhood."""

    mediator_var: int
    neighbor_vars: tuple[int, ...]


@dataclass(frozen=True)
class _HalfPhaseMediatorPlan:
    """Exact reduction of independent hard-unary mediators to pair factors."""

    level: int
    core_vars: tuple[int, ...]
    core_q2: dict[tuple[int, int], int]
    order: tuple[int, ...]
    width: int
    mediators: tuple[_HalfPhaseMediatorSpec, ...]


@dataclass(frozen=True)
class _GenericQ2MediatorSpec:
    """One exact low-degree q2 mediator collapsed to a small boundary factor."""

    mediator_var: int
    neighbor_vars: tuple[int, ...]
    neighbor_couplings: tuple[int, ...]
    assignment_residue_shifts: tuple[int, ...] = ()


@dataclass(frozen=True)
class _GenericQ2MediatorPlan:
    """Exact reduction of independent low-degree q2 mediators onto a core."""

    level: int
    core_vars: tuple[int, ...]
    core_q2: dict[tuple[int, int], int]
    order: tuple[int, ...]
    width: int
    mediators: tuple[_GenericQ2MediatorSpec, ...]


@dataclass(frozen=True)
class _HalfPhaseClusterSpec:
    """One exact hard-support cluster collapsed to a boundary factor."""

    cluster_vars: tuple[int, ...]
    boundary_vars: tuple[int, ...]
    internal_q2: dict[tuple[int, int], int]
    boundary_couplings: tuple[tuple[int, int, int], ...]
    boundary_shift_table: np.ndarray | None = None
    cluster_order: tuple[int, ...] = ()
    native_treewidth_plan: object | None = None


@dataclass(frozen=True)
class _HalfPhaseClusterPlan:
    """Exact elimination of small hard-support clusters onto a core factor graph."""

    level: int
    core_vars: tuple[int, ...]
    core_q2: dict[tuple[int, int], int]
    order: tuple[int, ...]
    width: int
    clusters: tuple[_HalfPhaseClusterSpec, ...]


@dataclass(frozen=True)
class _Q3FreeCutsetConditioningPlan:
    """Reusable exact cutset-conditioned treewidth plan for a q3-free kernel."""

    level: int
    cutset_vars: tuple[int, ...]
    remaining_vars: tuple[int, ...]
    remaining_backend: Literal["product", "treewidth", "generic"]
    remaining_q2: dict[tuple[int, int], int]
    remaining_order: tuple[int, ...]
    cutset_remaining_q2_residue: np.ndarray
    cutset_cutset_left: np.ndarray
    cutset_cutset_right: np.ndarray
    cutset_cutset_residue: np.ndarray
    native_treewidth_plan: object | None = None
    remaining_isolated_vars: tuple[int, ...] = ()
    remaining_components: tuple["_Q3FreeConstraintComponentPlan", ...] = ()
    remaining_width: int = 0
    estimated_total_work: int = 0
    branch_bits: np.ndarray | None = None
    branch_pair_residue: np.ndarray | None = None
    branch_remaining_shift: np.ndarray | None = None


@dataclass(frozen=True)
class _Q3FreeCutsetCandidateEvaluation:
    """Search-time summary for a candidate q3-free cutset."""

    cutset_vars: tuple[int, ...]
    plan: _Q3FreeCutsetConditioningPlan | None
    viable: bool
    score: tuple[int, ...]


@dataclass(frozen=True)
class _Q3FreeConstraintComponentPlan:
    """Preplanned q3-free component topology for repeated constrained sums."""

    variables: tuple[int, ...]
    level: int
    q2: dict[tuple[int, int], int]
    backend: Literal["constant", "forest", "treewidth", "generic"]
    adjacency: tuple[tuple[tuple[int, int], ...], ...] = ()
    order: tuple[int, ...] = ()
    dense_q2: np.ndarray | None = None
    precomputed_total: ScaledComplex | None = None
    binary_phase_plan: _BinaryPhaseQuadraticPlan | None = None
    mediator_plan: _HalfPhaseMediatorPlan | None = None
    generic_mediator_plan: _GenericQ2MediatorPlan | None = None
    cluster_plan: _HalfPhaseClusterPlan | None = None
    cutset_plan: _Q3FreeCutsetConditioningPlan | None = None
    native_treewidth_plan: object | None = None
    skip_dense_schur: bool = False
    direct_schur_ok: bool = False
    quadratic_tensor_q2: bool = False
    lambda_offset: int = -1
    prefer_reusable_decomposition: bool = False
    prefer_cutset_backend: bool = False


@dataclass(frozen=True)
class _Q3FreeConstraintPlan:
    """Exact constrained-sum plan that avoids affine parity substitution."""

    cache: EchelonCache
    eps0: tuple[int, ...]
    level: int
    q0: Fraction
    base_q1: tuple[int, ...]
    base_q2: dict[tuple[int, int], int]
    lambda_offset: int
    rank: int
    n_free_after_constraints: int
    rhs_linear_coeff: int
    isolated_vars: tuple[int, ...]
    components: tuple[_Q3FreeConstraintComponentPlan, ...]


@dataclass(frozen=True)
class _Q3FreeRawConstraintPlan:
    """Exact constrained-sum plan over the raw output rows of a q3-free state."""

    eps0: tuple[int, ...]
    level: int
    q0: Fraction
    base_q1: tuple[int, ...]
    base_q2: dict[tuple[int, int], int]
    lambda_offset: int
    constraint_count: int
    rhs_linear_coeff: int
    isolated_vars: tuple[int, ...]
    components: tuple[_Q3FreeConstraintComponentPlan, ...]


@dataclass(frozen=True)
class _Q3FreeRawConstraintRestrictedPlan:
    """Prefix-restricted view of a raw-output q3-free constraint plan."""

    active_count: int
    isolated_vars: tuple[int, ...]
    components: tuple[_Q3FreeConstraintComponentPlan, ...]


@dataclass(frozen=True)
class _Q3FreeExecutionPlan:
    """Fully instantiated q3-free execution plan.

    This is the optimizer/backend boundary for q3-free exact sums:
    optimization may rewrite the phase function, planning chooses reusable
    backend component plans, and execution only consumes this object.
    """

    level: int
    q0: Fraction
    q1: tuple[int, ...]
    isolated_vars: tuple[int, ...]
    components: tuple[_Q3FreeConstraintComponentPlan, ...]


@dataclass(frozen=True)
class ScaledAmplitude:
    """Amplitude represented as ``mantissa * 2 ** (half_pow2_exp / 2)``."""

    mantissa: complex
    half_pow2_exp: int = 0

    def __post_init__(self) -> None:
        value, half_pow2_exp = _normalize_scaled_complex(self.mantissa, self.half_pow2_exp)
        object.__setattr__(self, "mantissa", value)
        object.__setattr__(self, "half_pow2_exp", half_pow2_exp)

    @classmethod
    def from_tuple(cls, scaled: ScaledComplex) -> ScaledAmplitude:
        value, half_pow2_exp = scaled
        return cls(value, half_pow2_exp)

    def as_tuple(self) -> ScaledComplex:
        return self.mantissa, self.half_pow2_exp

    def to_complex(self) -> complex:
        return _scaled_to_complex(self.as_tuple())

    def log2_abs(self) -> float:
        if self.mantissa == 0j:
            return -math.inf
        return math.log2(abs(self.mantissa)) + self.half_pow2_exp / 2.0


def _apply_affine_bit_in_place(q, row_mask, offset, alpha):
    """Add alpha/mod_q1 * (offset xor parity(row_mask*f)) minus the constant term."""
    alpha %= q.mod_q1
    if not alpha or not row_mask:
        return

    support = _support_from_mask(row_mask)
    support_len = len(support)
    linear = alpha if not offset else (-alpha) % q.mod_q1
    pair = (-alpha) % q.mod_q2 if not offset else alpha % q.mod_q2
    cubic = alpha % q.mod_q3 if not offset else (-alpha) % q.mod_q3

    if linear:
        for idx in support:
            q.q1[idx] = (q.q1[idx] + linear) % q.mod_q1

    if pair:
        if support_len == 2:
            key = (support[0], support[1])
            value = (q.q2.get(key, 0) + pair) % q.mod_q2
            if value:
                q.q2[key] = value
            elif key in q.q2:
                del q.q2[key]
        elif support_len == 3:
            idx0, idx1, idx2 = support
            for key in ((idx0, idx1), (idx0, idx2), (idx1, idx2)):
                value = (q.q2.get(key, 0) + pair) % q.mod_q2
                if value:
                    q.q2[key] = value
                elif key in q.q2:
                    del q.q2[key]
        else:
            for idx0, idx1 in combinations(support, 2):
                key = (idx0, idx1)
                value = (q.q2.get(key, 0) + pair) % q.mod_q2
                if value:
                    q.q2[key] = value
                elif key in q.q2:
                    del q.q2[key]

    if cubic and support_len >= 3:
        if support_len == 3:
            key = (support[0], support[1], support[2])
            value = (q.q3.get(key, 0) + cubic) % q.mod_q3
            if value:
                q.q3[key] = value
            elif key in q.q3:
                del q.q3[key]
        else:
            for idx0, idx1, idx2 in combinations(support, 3):
                key = (idx0, idx1, idx2)
                value = (q.q3.get(key, 0) + cubic) % q.mod_q3
                if value:
                    q.q3[key] = value
                elif key in q.q3:
                    del q.q3[key]


def _apply_diag_phase_in_place(q, row_mask, shift, alpha):
    """Apply alpha/mod_q1 * g where g is an affine output bit."""
    if shift:
        q.q0 = (q.q0 + Fraction(alpha, q.mod_q1)) % 1
    _apply_affine_bit_in_place(q, row_mask, shift, alpha)


def _apply_bilinear_phase_in_place(q, row_mask0, shift0, row_mask1, shift1, coeff):
    """Apply coeff/mod_q2 * g0 * g1 for affine output bits g0 and g1."""
    coeff %= q.mod_q2
    if not coeff:
        return
    if shift0 and shift1:
        q.q0 = (q.q0 + Fraction(coeff, q.mod_q2)) % 1
    _apply_affine_bit_in_place(q, row_mask0, shift0, coeff)
    _apply_affine_bit_in_place(q, row_mask1, shift1, coeff)
    _apply_affine_bit_in_place(q, row_mask0 ^ row_mask1, shift0 ^ shift1, (-coeff) % q.mod_q1)


def _prepare_affine_constraint_cache(n_constraints: int, n_vars: int, row_masks: Sequence[int]) -> EchelonCache:
    rows = [int(mask) for mask in row_masks]
    rows, row_ops, pivot_col, used_mask = _row_reduce_output_constraints(n_constraints, rows)

    free = tuple(var for var in range(n_vars) if not (used_mask >> var) & 1)
    n_free = len(free)
    gamma = [0] * n_vars
    for free_idx, free_var in enumerate(free):
        gamma[free_var] = 1 << free_idx
        for row_idx, pivot in enumerate(pivot_col):
            if pivot >= 0 and (rows[row_idx] >> free_var) & 1:
                gamma[pivot] ^= 1 << free_idx

    return EchelonCache(
        n=n_constraints,
        m=n_vars,
        echelon_rows=tuple(rows),
        pivot_col=tuple(pivot_col),
        used_mask=used_mask,
        row_ops=tuple(row_ops),
        free_vars=free,
        gamma_masks=tuple(gamma),
        n_free=n_free,
    )


def _solve_echelon_rhs(cache: EchelonCache, rhs_mask: int) -> int | None:
    shift_mask = 0
    for row_idx, pivot in enumerate(cache.pivot_col):
        rhs = _parity(rhs_mask & cache.row_ops[row_idx])
        if pivot < 0:
            if rhs:
                return None
            continue
        if rhs:
            shift_mask |= 1 << pivot
    return shift_mask


def _coalesce_arbitrary_phase_terms(
    terms: Sequence[_ArbitraryPhaseTerm],
) -> tuple[_ArbitraryPhaseTerm, ...]:
    merged: dict[tuple[int, int], float] = {}
    for term in terms:
        key = (int(term.row_mask), int(term.offset) & 1)
        merged[key] = merged.get(key, 0.0) + float(term.angle)

    coalesced: list[_ArbitraryPhaseTerm] = []
    for (row_mask, offset), angle in merged.items():
        if math.isclose(math.remainder(angle, 2.0 * math.pi), 0.0, rel_tol=0.0, abs_tol=1e-15):
            continue
        coalesced.append(_ArbitraryPhaseTerm(row_mask, offset, angle))

    coalesced.sort(key=lambda term: (term.row_mask, term.offset, term.angle))
    return tuple(coalesced)


def _build_arbitrary_phase_branch_plan(
    terms: Sequence[_ArbitraryPhaseTerm],
) -> _ArbitraryPhaseBranchPlan:
    if not terms:
        return _ArbitraryPhaseBranchPlan((), (), (), ())

    basis_by_pivot: dict[int, int] = {}
    for term in terms:
        reduced = int(term.row_mask)
        for pivot in sorted(basis_by_pivot, reverse=True):
            if (reduced >> pivot) & 1:
                reduced ^= basis_by_pivot[pivot]
        if reduced:
            basis_by_pivot[reduced.bit_length() - 1] = reduced

    ordered_pivots = tuple(sorted(basis_by_pivot, reverse=True))
    basis_masks = tuple(basis_by_pivot[pivot] for pivot in ordered_pivots)
    dependency_masks: list[int] = []
    offsets: list[int] = []
    angles: list[float] = []

    for term in terms:
        reduced = int(term.row_mask)
        dependency_mask = 0
        for basis_idx, basis_mask in enumerate(basis_masks):
            pivot = basis_mask.bit_length() - 1
            if (reduced >> pivot) & 1:
                reduced ^= basis_mask
                dependency_mask ^= 1 << basis_idx
        if reduced:  # pragma: no cover - internal consistency guard
            raise RuntimeError("Failed to express arbitrary-phase mask in its branch basis.")
        dependency_masks.append(dependency_mask)
        offsets.append(int(term.offset) & 1)
        angles.append(float(term.angle))

    return _ArbitraryPhaseBranchPlan(
        basis_masks=basis_masks,
        term_dependency_masks=tuple(dependency_masks),
        term_offsets=tuple(offsets),
        term_angles=tuple(angles),
    )


# ==================================================================
# Schur-state construction and output constraint solving
# ==================================================================

class SchurState:
    """3rd-order tensor data (E, eps, q) over G=Z2^n [BL26 Def. 33]."""

    def __init__(self, n: int) -> None:
        self.n, self.m = n, 0
        self.eps: list[int] = [0] * n
        self.eps0: list[int] = [0] * n
        self.q: PhaseFunction = CubicFunction(0)
        self.scalar: complex = 1.0 + 0j
        self.scalar_half_pow2: int = 0
        self.output_refcount: list[int] = []
        self._arbitrary_phases: list[_ArbitraryPhaseTerm] = []
        self._pending_dead: set[int] = set()
        self._cached_classification_data = None
        self._cached_classification_q = None

    def _invalidate_classification_cache(self) -> None:
        self._cached_classification_data = None
        self._cached_classification_q = None

    def _current_classification_data(self):
        if self._cached_classification_q is self.q and self._cached_classification_data is not None:
            return self._cached_classification_data
        classification_data = _build_classification_data(self.q)
        self._cached_classification_data = classification_data
        self._cached_classification_q = self.q
        return classification_data

    def _add_var(self):
        if not getattr(self.q, "_schur_mutable", True):
            self.q = _copy_cubic_function(self.q)
        self._invalidate_classification_cache()
        idx = self.m
        self.m += 1
        self.output_refcount.append(0)
        self.q.n = self.m
        self.q.q1.append(0)
        return idx

    def _update_reference_mask(self, old_mask, new_mask):
        changed = old_mask ^ new_mask
        newly_dead = []
        while changed:
            bit = changed & -changed
            idx = bit.bit_length() - 1
            if new_mask & bit:
                self.output_refcount[idx] += 1
            else:
                self.output_refcount[idx] -= 1
                if self.output_refcount[idx] == 0:
                    newly_dead.append(idx)
            changed ^= bit
        return newly_dead

    def _set_row_mask(self, qubit, new_mask):
        old_mask = self.eps[qubit]
        if old_mask == new_mask:
            return []
        newly_dead = self._update_reference_mask(old_mask, new_mask)
        self.eps[qubit] = new_mask
        return newly_dead

    def _rebuild_output_refcount(self):
        self.output_refcount = [0] * self.m
        for row_mask in self.eps:
            for idx in _iter_mask_bits(row_mask):
                self.output_refcount[idx] += 1
        for term in self._arbitrary_phases:
            for idx in _iter_mask_bits(term.row_mask):
                self.output_refcount[idx] += 1

    def _remap_mask_after_removal(self, mask, removed):
        if not mask or not removed:
            return mask
        new_mask = 0
        for idx in _iter_mask_bits(mask):
            shift = bisect.bisect_left(removed, idx)
            new_mask |= 1 << (idx - shift)
        return new_mask

    def _apply_elimination_result(self, new_q, half_pow2, removed):
        self.q = new_q
        self.m = new_q.n
        self.scalar_half_pow2 += half_pow2
        self._invalidate_classification_cache()
        removed = sorted(removed)
        self.eps = [self._remap_mask_after_removal(mask, removed) for mask in self.eps]
        remapped_terms: list[_ArbitraryPhaseTerm] = []
        for term in self._arbitrary_phases:
            remapped_mask = self._remap_mask_after_removal(term.row_mask, removed)
            if remapped_mask:
                remapped_terms.append(_ArbitraryPhaseTerm(remapped_mask, term.offset, term.angle))
            elif term.offset:
                self.scalar *= cmath.exp(1j * term.angle)
        self._arbitrary_phases = remapped_terms
        self._rebuild_output_refcount()

    def _queue_dead_candidates(self, candidates):
        if not candidates:
            return
        self._pending_dead.update(candidates)
        if len(self._pending_dead) >= _build_early_elim_batch_size(self.q.level):
            self._flush_pending_dead_variables()

    def _flush_pending_dead_variables(self):
        if not self._pending_dead:
            return
        candidates = tuple(sorted(self._pending_dead))
        self._pending_dead.clear()
        self._early_eliminate_dead_variables(candidates)

    def _early_eliminate_dead_variables(self, candidates=None):
        if self.scalar == 0j or self.m == 0:
            self._pending_dead.clear()
            return

        if candidates is None:
            dead = {idx for idx, count in enumerate(self.output_refcount) if count == 0}
        else:
            dead = {idx for idx in candidates if idx < self.m and self.output_refcount[idx] == 0}
        if not dead:
            return

        changed = True
        while changed and dead:
            changed = False
            classification_data = self._current_classification_data()
            ordered_dead = tuple(sorted(dead))
            classification_entries = {
                var: _classification_entry(self.q, var, classification_data=classification_data)
                for var in ordered_dead
            }

            decoupled = [
                var
                for var in ordered_dead
                if classification_entries[var][0] == _CLASS_CONSTRAINT_DECOUPLED
            ]
            if decoupled:
                new_q, half_pow2 = _elim_decoupled_constraints_batch(self.q, decoupled)
                self._apply_elimination_result(new_q, half_pow2, decoupled)
                dead = {idx for idx, count in enumerate(self.output_refcount) if count == 0}
                changed = True
                continue

            for var in ordered_dead:
                entry = classification_entries[var]
                tag = entry[0]
                if tag in {_CLASS_CONSTRAINT_DECOUPLED, _CLASS_CONSTRAINT_ZERO, _CLASS_CONSTRAINT_PARITY}:
                    if tag == _CLASS_CONSTRAINT_ZERO:
                        self.scalar = 0j
                        self.scalar_half_pow2 = 0
                        self.q = PhaseFunction(0, level=self.q.level)
                        self.m = 0
                        self.eps = [0] * self.n
                        self.eps0 = [0] * self.n
                        self.output_refcount = []
                        self._arbitrary_phases = []
                        self._pending_dead.clear()
                        return
                    continue
                if tag != _CLASS_QUADRATIC:
                    continue
                # Above Clifford+T precision the same affine substitutions used
                # by `_elim_quadratic(...)` can leave the q3-free kernel outside
                # the current exact PhaseFunction representation. The runtime
                # reducer already bypasses that elimination regime, so keep the
                # build-time dead-variable pass consistent and defer the exact
                # q3-free work to the final solver.
                if self.q.level > 3:
                    continue
                if bool(entry[2]):
                    continue
                if _should_defer_build_quadratic_elimination(
                    self.q,
                    var,
                    classification_data=classification_data,
                ):
                    continue
                new_q, half_pow2 = _elim_quadratic(
                    self.q,
                    var,
                    classification_data=classification_data,
                )
                self._apply_elimination_result(new_q, half_pow2, [var])
                dead = {idx for idx, count in enumerate(self.output_refcount) if count == 0}
                changed = True
                break

    # -- Gates --

    def _ensure_mutable_phase_function(self) -> None:
        if not getattr(self.q, "_schur_mutable", True):
            self.q = _copy_cubic_function(self.q)
            self._invalidate_classification_cache()

    def _ensure_phase_precision(self, precision_level: int) -> None:
        if precision_level > self.q.level:
            self._ensure_mutable_phase_function()
            self.q.promote_in_place(precision_level)
            self._invalidate_classification_cache()

    def _lift_linear_coeff(self, coeff: int, precision_level: int) -> int:
        self._ensure_phase_precision(precision_level)
        return (int(coeff) * (1 << (self.q.level - precision_level))) % self.q.mod_q1

    def _lift_quadratic_coeff(self, coeff: int, precision_level: int) -> int:
        self._ensure_phase_precision(precision_level)
        return (int(coeff) * (1 << (self.q.level - precision_level))) % self.q.mod_q2

    def cnot(self, c: int, t: int) -> None:                   # [BL26 Sec. VA]
        newly_dead = self._set_row_mask(t, self.eps[t] ^ self.eps[c])
        self.eps0[t] = (self.eps0[t]+self.eps0[c])%2
        self._queue_dead_candidates(newly_dead)

    def _diag(self, qubit: int, q1v: int, precision_level: int = 3) -> None:  # [BL26 Eq.290]
        """Phase (q1v/mod_q1)*g_qubit where g_qubit = eps_row*e xor eps0_qubit.

        Key: over Z2, (q1v/8)*(e xor s) != (q1v/8)*e + (q1v/8)*s.
        Must use affine composition.
        """
        q1v = self._lift_linear_coeff(q1v, precision_level)
        row_mask, sh = self.eps[qubit], self.eps0[qubit]
        if not row_mask:
            # No internal variables: just a constant phase
            self.scalar *= cmath.exp(2j * cmath.pi * q1v * sh / self.q.mod_q1)
        else:
            self._ensure_mutable_phase_function()
            _apply_diag_phase_in_place(self.q, row_mask, sh, q1v)
            self._invalidate_classification_cache()

    def t(self, q: int) -> None:
        self._diag(q, 1, precision_level=3)

    def tdg(self, q: int) -> None:
        self._diag(q, -1, precision_level=3)

    def s(self, q: int) -> None:
        self._diag(q, 2, precision_level=3)

    def sdg(self, q: int) -> None:
        self._diag(q, -2, precision_level=3)

    def z(self, q: int) -> None:
        self._diag(q, 4, precision_level=3)

    def rz_arbitrary(self, qubit: int, angle: float) -> None:
        """Apply diag(1, exp(i * angle)) exactly without dyadic approximation."""
        angle_value = float(angle)
        if not math.isfinite(angle_value):
            raise ValueError(f"rz_arbitrary angle must be finite, received {angle!r}.")
        if math.isclose(math.remainder(angle_value, 2.0 * math.pi), 0.0, rel_tol=0.0, abs_tol=1e-15):
            return
        row_mask = self.eps[qubit]
        offset = self.eps0[qubit] & 1
        if not row_mask:
            if offset:
                self.scalar *= cmath.exp(1j * angle_value)
            return
        self._arbitrary_phases.append(_ArbitraryPhaseTerm(row_mask, offset, angle_value))
        self._update_reference_mask(0, row_mask)

    def rz_dyadic(self, qubit: int, coeff: int, precision_level: int) -> None:
        """Apply diag(1, exp(2*pi*i*coeff / 2^precision_level))."""
        self._diag(qubit, coeff, precision_level=precision_level)

    def sx(self, qubit: int) -> None:
        """Apply ``sqrt(X)`` with one fresh path variable instead of ``HSH``.

        The matrix entries are all ``2**-1/2`` times eighth roots of unity:

            SX[a, g] = 2**-1/2 * w**(1 + 6 a + 6 g + 4 a g)

        where ``g`` is the incoming affine output bit and ``a`` is the fresh
        replacement bit. Expanding this directly preserves exactness while
        avoiding the second Hadamard variable introduced by the ``H; S; H``
        synthesis. On Sycamore-like RCS workloads this keeps the one-shot
        q3-free remainder materially narrower upstream of the dense-core cliff.
        """
        old_mask = self.eps[qubit]
        old_shift = self.eps0[qubit]
        new_var = self._add_var()

        self.q.q0 = (
            self.q.q0
            + Fraction(self._lift_linear_coeff(1, 3), self.q.mod_q1)
        ) % 1
        self.q.q1[new_var] = (
            self.q.q1[new_var] + self._lift_linear_coeff(6, 3)
        ) % self.q.mod_q1

        if old_mask or old_shift:
            _apply_diag_phase_in_place(
                self.q,
                old_mask,
                old_shift,
                self._lift_linear_coeff(6, 3),
            )
            _apply_bilinear_phase_in_place(
                self.q,
                old_mask,
                old_shift,
                1 << new_var,
                0,
                self._lift_quadratic_coeff(2, 3),
            )

        self._invalidate_classification_cache()
        newly_dead = self._set_row_mask(qubit, 1 << new_var)
        self.eps0[qubit] = 0
        self.scalar_half_pow2 -= 1
        self._queue_dead_candidates(newly_dead)

    def sxdg(self, qubit: int) -> None:
        """Apply ``sqrt(X)†`` with one fresh path variable instead of ``HSdgH``.

        The matrix entries are all ``2**-1/2`` times eighth roots of unity:

            SXdg[a, g] = 2**-1/2 * w**(7 + 2 a + 2 g + 4 a g)

        where ``g`` is the incoming affine output bit and ``a`` is the fresh
        replacement bit. This is the complex conjugate of SX — the bilinear
        coupling coefficient c3=4 is identical, but the linear coefficients
        are negated mod 8 (c0: 1→7, c1: 6→2, c2: 6→2). The implementation
        avoids the two Hadamard variables introduced by the ``H; Sdg; H``
        synthesis, matching the native SX path's one-variable budget.
        """
        old_mask = self.eps[qubit]
        old_shift = self.eps0[qubit]
        new_var = self._add_var()

        self.q.q0 = (
            self.q.q0
            + Fraction(self._lift_linear_coeff(7, 3), self.q.mod_q1)
        ) % 1
        self.q.q1[new_var] = (
            self.q.q1[new_var] + self._lift_linear_coeff(2, 3)
        ) % self.q.mod_q1

        if old_mask or old_shift:
            _apply_diag_phase_in_place(
                self.q,
                old_mask,
                old_shift,
                self._lift_linear_coeff(2, 3),
            )
            _apply_bilinear_phase_in_place(
                self.q,
                old_mask,
                old_shift,
                1 << new_var,
                0,
                self._lift_quadratic_coeff(2, 3),
            )

        self._invalidate_classification_cache()
        newly_dead = self._set_row_mask(qubit, 1 << new_var)
        self.eps0[qubit] = 0
        self.scalar_half_pow2 -= 1
        self._queue_dead_candidates(newly_dead)

    def rz_pi_2k(self, qubit: int, k: int, dagger: bool = False) -> None:
        coeff = -1 if dagger else 1
        self.rz_dyadic(qubit, coeff, precision_level=k + 1)

    def rz_pi_16(self, qubit: int) -> None:
        self.rz_pi_2k(qubit, 4, dagger=False)

    def rz_pi_16_dg(self, qubit: int) -> None:
        self.rz_pi_2k(qubit, 4, dagger=True)

    def rz_pi_32(self, qubit: int) -> None:
        self.rz_pi_2k(qubit, 5, dagger=False)

    def rz_pi_32_dg(self, qubit: int) -> None:
        self.rz_pi_2k(qubit, 5, dagger=True)

    def x(self, qubit: int) -> None:
        """Apply a Pauli X gate by toggling the affine output offset."""
        self.eps0[qubit] = (self.eps0[qubit] + 1) % 2

    def cz(self, q0: int, q1: int) -> None:                  # [BL26 Eq.309]
        """CZ: phase g0*g1/2. Uses affine composition for Z2 correctness."""
        r0,r1 = self.eps[q0],self.eps[q1]
        s0,s1 = self.eps0[q0],self.eps0[q1]
        if not r0 and not r1:
            self.scalar *= cmath.exp(2j*cmath.pi*s0*s1/2)
        else:
            self._ensure_mutable_phase_function()
            _apply_bilinear_phase_in_place(
                self.q,
                r0,
                s0,
                r1,
                s1,
                self._lift_quadratic_coeff(2, 3),
            )
            self._invalidate_classification_cache()

    def rzz_dyadic(self, q0: int, q1: int, coeff: int, precision_level: int) -> None:
        """
        Apply the exact dyadic parity phase ``diag(1, w, w, 1)`` on ``(q0, q1)``.

        This matches the benchmark decomposition ``cnot(q0,q1); rz_dyadic(q1); cnot(q0,q1)``
        but avoids introducing the two explicit CNOT layers.
        """
        modulus = 1 << int(precision_level)
        coeff = int(coeff) % modulus
        if coeff == 0:
            return
        self._diag(q0, coeff, precision_level=precision_level)
        self._diag(q1, coeff, precision_level=precision_level)
        r0, r1 = self.eps[q0], self.eps[q1]
        s0, s1 = self.eps0[q0], self.eps0[q1]
        if not r0 and not r1:
            self.scalar *= cmath.exp(
                2j
                * cmath.pi
                * Fraction(self._lift_quadratic_coeff(-coeff, precision_level), self.q.mod_q2)
                * s0
                * s1
            )
        else:
            self._ensure_mutable_phase_function()
            _apply_bilinear_phase_in_place(
                self.q,
                r0,
                s0,
                r1,
                s1,
                self._lift_quadratic_coeff(-coeff, precision_level),
            )
            self._invalidate_classification_cache()

    def h(self, qubit: int) -> None:                         # [BL26 Eq.284]
        k=qubit
        old_mask = self.eps[k]
        a=self._add_var()
        for j in _iter_mask_bits(old_mask):
            key=(min(j,a),max(j,a))
            value = (self.q.q2.get(key, 0) + self._lift_quadratic_coeff(2, 3)) % self.q.mod_q2
            if value:
                self.q.q2[key] = value
            elif key in self.q.q2:
                del self.q.q2[key]
        if self.eps0[k]:
            self.q.q1[a] = (self.q.q1[a] + self._lift_linear_coeff(4, 3)) % self.q.mod_q1
        newly_dead = self._set_row_mask(k, 1 << a)
        self.eps0[k]=0
        self.scalar_half_pow2 -= 1
        self._queue_dead_candidates(newly_dead)

    def _prepare_echelon(self) -> EchelonCache:
        """Row-reduce the output constraint matrix once for batch reuse."""
        self._flush_pending_dead_variables()
        rows = self.eps[:]
        rows, row_ops, pivot_col, used_mask = _row_reduce_output_constraints(self.n, rows)

        free = tuple(var for var in range(self.m) if not (used_mask >> var) & 1)
        n_free = len(free)
        gamma = [0] * self.m
        for free_idx, free_var in enumerate(free):
            gamma[free_var] = 1 << free_idx
            for row_idx, pivot in enumerate(pivot_col):
                if pivot >= 0 and (rows[row_idx] >> free_var) & 1:
                    gamma[pivot] ^= 1 << free_idx

        return EchelonCache(
            n=self.n,
            m=self.m,
            echelon_rows=tuple(rows),
            pivot_col=tuple(pivot_col),
            used_mask=used_mask,
            row_ops=tuple(row_ops),
            free_vars=free,
            gamma_masks=tuple(gamma),
            n_free=n_free,
        )

    def _prepare_constraint_echelon(self) -> EchelonCache:
        """
        Row-reduce the output constraints without constructing a solution basis.

        The q3-free probability path only needs the reduced rows, pivot map,
        row-operation witnesses, and the free-variable count. Skipping the
        explicit ``free_vars`` / ``gamma_masks`` construction avoids repeated
        O(n * m) work when evaluating a long chain of exact prefix marginals.
        """
        self._flush_pending_dead_variables()
        rows = self.eps[:]
        rows, row_ops, pivot_col, used_mask = _row_reduce_output_constraints(self.n, rows)
        return EchelonCache(
            n=self.n,
            m=self.m,
            echelon_rows=tuple(rows),
            pivot_col=tuple(pivot_col),
            used_mask=used_mask,
            row_ops=tuple(row_ops),
            free_vars=(),
            gamma_masks=(),
            n_free=self.m - used_mask.bit_count(),
        )

    def _solve_for_output(
        self,
        cache: EchelonCache,
        output_bits: BitSequence,
    ) -> tuple[int, tuple[int, ...], tuple[int, ...], int] | None:
        """Solve the output constraints for one output string."""
        if len(output_bits) != cache.n:
            raise ValueError(f"Expected {cache.n} output bits, received {len(output_bits)}.")

        native_shift_mask = _native_solve_for_output(self.eps0, cache, output_bits)
        if native_shift_mask is not None:
            return native_shift_mask, cache.free_vars, cache.gamma_masks, cache.n_free

        target_mask = 0
        for idx, bit in enumerate(output_bits):
            if (int(bit) ^ self.eps0[idx]) & 1:
                target_mask |= 1 << idx

        shift_mask = 0
        for row_idx, pivot in enumerate(cache.pivot_col):
            rhs = _parity(target_mask & cache.row_ops[row_idx])
            if pivot < 0 and rhs:
                return None
            if pivot >= 0 and rhs:
                shift_mask |= 1 << pivot

        return shift_mask, cache.free_vars, cache.gamma_masks, cache.n_free

    def _transform_arbitrary_phases(
        self,
        shift_mask: int,
        gamma_masks: Sequence[int],
    ) -> tuple[complex, tuple[_ArbitraryPhaseTerm, ...]]:
        scalar = 1.0 + 0.0j
        transformed: list[_ArbitraryPhaseTerm] = []
        for term in self._arbitrary_phases:
            offset = (int(term.offset) ^ _parity(int(term.row_mask) & shift_mask)) & 1
            row_mask = 0
            for idx in _iter_mask_bits(int(term.row_mask)):
                row_mask ^= int(gamma_masks[idx])
            if row_mask == 0:
                if offset:
                    scalar *= cmath.exp(1j * float(term.angle))
                continue
            transformed.append(_ArbitraryPhaseTerm(row_mask, offset, float(term.angle)))
        return scalar, _coalesce_arbitrary_phase_terms(transformed)

    # -- Amplitude -------------------------------------------------

    def _amplitude_internal(
        self,
        output_bits: BitSequence,
        preserve_scale: bool = False,
        allow_tensor_contraction: bool = True,
        extended_reductions: ExtendedReductionMode | str = "auto",
    ) -> tuple[ScaledAmplitude | complex, ReductionInfo]:
        if len(output_bits) != self.n:
            raise ValueError(f"Expected {self.n} output bits, received {len(output_bits)}.")
        if self.m == 0:
            ok = all(self.eps0[idx] == output_bits[idx] for idx in range(self.n))
            scaled = (
                _normalize_scaled_complex(
                    self.scalar * cmath.exp(2j * cmath.pi * float(self.q.q0)),
                    self.scalar_half_pow2,
                )
                if ok
                else _make_scaled_complex(0j)
            )
            amp = ScaledAmplitude.from_tuple(scaled) if preserve_scale else _scaled_to_complex(scaled)
            return amp, _info(0, 0, 0, 0, 0, zero=not ok)

        # For q3-free states without arbitrary-angle side terms, evaluate the
        # output constraints directly on the raw rows instead of materializing
        # an affine-parity substitution. This keeps the one-shot high-precision
        # path exact without paying the raw-plan slicing search cost.
        if not self._arbitrary_phases and not self.q.q3:
            cache = self._prepare_constraint_echelon()
            plan = _build_q3_free_raw_constraint_plan(
                self,
                allow_tensor_contraction=allow_tensor_contraction,
                prefer_one_shot_slicing=True,
            )
            restricted_plan = _restrict_q3_free_raw_constraint_plan(plan, self.n)
            result = _evaluate_q3_free_raw_constraint_plan_scaled(
                plan,
                restricted_plan,
                output_bits,
                allow_tensor_contraction=allow_tensor_contraction,
            )
            scaled_amp = _normalize_scaled_complex(
                complex(self.scalar) * result[0],
                result[1] + self.scalar_half_pow2,
            )
            amp = ScaledAmplitude.from_tuple(scaled_amp) if preserve_scale else _scaled_to_complex(scaled_amp)
            return amp, _info(
                cache.n_free,
                0,
                0,
                0,
                0,
                structural_obstruction=0,
                gauss_obstruction=_gauss_obstruction(self.q, 0),
                phase_states=0,
                phase_splits=0,
                zero=scaled_amp[0] == 0j,
                cost_model_r=0,
                phase3_backend=_q3_free_phase3_backend_name(self.q),
            )

        cache = self._prepare_echelon()
        solved = self._solve_for_output(cache, output_bits)
        if solved is None:
            zero_scaled = _make_scaled_complex(0j)
            amp = ScaledAmplitude.from_tuple(zero_scaled) if preserve_scale else 0j
            return amp, _info(0, 0, 0, 0, 0, zero=True)

        context = _ReductionContext(
            preserve_scale=preserve_scale,
            allow_tensor_contraction=allow_tensor_contraction,
            extended_reductions=extended_reductions,
        )
        shift_mask, _, gamma, k = solved
        q_free = _aff_compose_cached(self.q, shift_mask, gamma, k, context=context)
        arbitrary_scalar, arbitrary_terms = self._transform_arbitrary_phases(shift_mask, gamma)

        if arbitrary_terms:
            branch_plan = _build_arbitrary_phase_branch_plan(arbitrary_terms)
            branch_cache = _prepare_affine_constraint_cache(
                len(branch_plan.basis_masks),
                k,
                branch_plan.basis_masks,
            )
            result = _make_scaled_complex(0j)
            quad_eliminated = 0
            constraint_eliminated = 0
            max_remaining = 0
            max_structural = 0
            max_gauss = 0
            max_cost_r = 0
            max_branched = 0
            phase_states = 0
            phase_splits = 0
            phase3_backend: str | None = None
            phase3_backend_cost_r = -1

            for assignment_mask in range(1 << len(branch_plan.basis_masks)):
                branch_phase = 1.0 + 0.0j

                for dependency_mask, offset, angle in zip(
                    branch_plan.term_dependency_masks,
                    branch_plan.term_offsets,
                    branch_plan.term_angles,
                ):
                    if _parity(dependency_mask & assignment_mask) ^ offset:
                        branch_phase *= cmath.exp(1j * angle)

                extra_shift_mask = _solve_echelon_rhs(branch_cache, assignment_mask)
                if extra_shift_mask is None:
                    continue

                q_branch = _aff_compose_cached(
                    q_free,
                    extra_shift_mask,
                    branch_cache.gamma_masks,
                    branch_cache.n_free,
                    context=context,
                )
                branch_result, branch_info = _reduce_and_sum_scaled(q_branch, context=context)
                result = _add_scaled_complex(
                    result,
                    _mul_scaled_complex(_make_scaled_complex(branch_phase), branch_result),
                )

                quad_eliminated += branch_info['quad']
                constraint_eliminated += branch_info['constraint']
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
                    elif branch_cost_r == phase3_backend_cost_r and phase3_backend is None:
                        phase3_backend = branch_phase3_backend

            elim_info = {
                'quad': quad_eliminated,
                'constraint': constraint_eliminated,
                'branched': len(branch_plan.basis_masks) + max_branched,
                'remaining': max_remaining,
                'structural_obstruction': max_structural,
                'gauss_obstruction': max_gauss,
                'cost_r': max_cost_r,
                'phase_states': phase_states,
                'phase_splits': phase_splits,
                'phase3_backend': phase3_backend,
            }
        else:
            result, elim_info = _reduce_and_sum_scaled(q_free, context=context)

        scaled_amp = _normalize_scaled_complex(
            complex(self.scalar) * arbitrary_scalar * result[0],
            result[1] + self.scalar_half_pow2,
        )
        amp = ScaledAmplitude.from_tuple(scaled_amp) if preserve_scale else _scaled_to_complex(scaled_amp)
        return amp, _info(
            k,
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
            zero=scaled_amp[0] == 0j,
            cost_model_r=elim_info.get('cost_r', elim_info['remaining']),
            phase3_backend=elim_info.get('phase3_backend'),
        )

    @overload
    def amplitude(
        self,
        output_bits: BitSequence,
        *,
        as_complex: Literal[False] = False,
        allow_tensor_contraction: bool = True,
        extended_reductions: ExtendedReductionMode | str = "auto",
        solver_config: "SolverConfig | None" = None,
    ) -> tuple[ScaledAmplitude, ReductionInfo]:
        ...

    @overload
    def amplitude(
        self,
        output_bits: BitSequence,
        *,
        as_complex: Literal[True],
        allow_tensor_contraction: bool = True,
        extended_reductions: ExtendedReductionMode | str = "auto",
        solver_config: "SolverConfig | None" = None,
    ) -> tuple[complex, ReductionInfo]:
        ...

    def amplitude(
        self,
        output_bits: BitSequence,
        *,
        as_complex: bool = False,
        allow_tensor_contraction: bool = True,
        extended_reductions: ExtendedReductionMode | str = "auto",
        solver_config: "SolverConfig | None" = None,
    ) -> tuple[ScaledAmplitude | complex, ReductionInfo]:
        _token = _SOLVER_CONFIG_VAR.set(solver_config) if solver_config is not None else None
        try:
            return self._amplitude_internal(
                output_bits,
                preserve_scale=not as_complex,
                allow_tensor_contraction=allow_tensor_contraction,
                extended_reductions=extended_reductions,
            )
        finally:
            if _token is not None:
                _SOLVER_CONFIG_VAR.reset(_token)

    def amplitude_scaled(
        self,
        output_bits: BitSequence,
        *,
        allow_tensor_contraction: bool = True,
        extended_reductions: ExtendedReductionMode | str = "auto",
        solver_config: "SolverConfig | None" = None,
    ) -> tuple[ScaledAmplitude, ReductionInfo]:
        return self.amplitude(
            output_bits,
            as_complex=False,
            allow_tensor_contraction=allow_tensor_contraction,
            extended_reductions=extended_reductions,
            solver_config=solver_config,
        )

    def amplitudes(
        self,
        output_list: Sequence[BitSequence],
        *,
        as_complex: bool = False,
        allow_tensor_contraction: bool = True,
        extended_reductions: ExtendedReductionMode | str = "auto",
        solver_config: "SolverConfig | None" = None,
    ) -> list[tuple[ScaledAmplitude | complex, ReductionInfo]]:
        _token = _SOLVER_CONFIG_VAR.set(solver_config) if solver_config is not None else None
        try:
            return _batch_query_state(
                self,
                output_list,
                preserve_scale=not as_complex,
                allow_tensor_contraction=allow_tensor_contraction,
                extended_reductions=extended_reductions,
                analyze_only=False,
            )
        finally:
            if _token is not None:
                _SOLVER_CONFIG_VAR.reset(_token)

    def amplitudes_scaled(
        self,
        output_list: Sequence[BitSequence],
        *,
        allow_tensor_contraction: bool = True,
        extended_reductions: ExtendedReductionMode | str = "auto",
        solver_config: "SolverConfig | None" = None,
    ) -> list[tuple[ScaledAmplitude, ReductionInfo]]:
        return self.amplitudes(
            output_list,
            as_complex=False,
            allow_tensor_contraction=allow_tensor_contraction,
            extended_reductions=extended_reductions,
            solver_config=solver_config,
        )


# ==================================================================
# Core: recursive reduction and summation
# ==================================================================

def _reduce_and_sum_scaled(q, context=None):
    """
    Reduce q by eliminating variables, then sum over remainder.
    Returns (complex sum, info dict).

    Algorithm:
      1. Eliminate all quadratic/constraint variables [Props. 9, 11]
      2. If stuck, BRANCH on the odd-q1 variable that unblocks the most
         even-q1 partners, then recurse into each branch
      3. Base case: sum over remaining genuinely obstructed variables
    """
    if context is None:
        context = _ReductionContext()
    extended_reductions = context.extended_reductions

    cache_key = _q_key(q)
    cached = context.reduce_cache.get(cache_key)
    if cached is not None:
        return cached[0], dict(cached[1])

    # Above Clifford+T precision, affine parity substitutions can require
    # quartic or higher terms that the current PhaseFunction backend does not
    # represent. q3-free kernels are already solvable exactly, so bypass the
    # exact-elimination stage that would otherwise recompose parity constraints
    # through `_aff_compose(...)`.
    if q.level > 3 and not q.q3:
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
        structural_obstruction = 0
        gauss_obstruction = _gauss_obstruction(q, structural_obstruction)
        info = {
            'quad': 0,
            'constraint': 0,
            'branched': 0,
            'remaining': 0,
            'structural_obstruction': structural_obstruction,
            'gauss_obstruction': gauss_obstruction,
            'cost_r': 0,
            'phase_states': phase_info['phase_states'],
            'phase_splits': phase_info['phase_splits'],
            'phase3_backend': _q3_free_phase3_backend_name(q),
        }
        context.reduce_cache[cache_key] = (total, dict(info))
        return total, info

    q, scale_half_pow2, exact_info, blocked_quadratics = _apply_exact_eliminations(q, context=context)
    nq = exact_info['quad']
    nc = exact_info['constraint']
    nb = 0
    if q is None:
        zero = _make_scaled_complex(0j)
        info = {
            'quad': nq,
            'constraint': nc,
            'branched': 0,
            'remaining': 0,
            'structural_obstruction': 0,
            'gauss_obstruction': 0,
            'cost_r': 0,
            'phase_states': 0,
            'phase_splits': 0,
            'phase3_backend': None,
        }
        context.reduce_cache[cache_key] = (zero, dict(info))
        return zero, info

    if q.n == 0:
        total = _scale_scaled_complex(
            _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q.q0))),
            scale_half_pow2,
        )
        info = {
            'quad': nq,
            'constraint': nc,
            'branched': 0,
            'remaining': 0,
            'structural_obstruction': 0,
            'gauss_obstruction': 0,
            'cost_r': 0,
            'phase_states': 0,
            'phase_splits': 0,
            'phase3_backend': None,
        }
        context.reduce_cache[cache_key] = (total, dict(info))
        return total, info

    enable_extended_q3_reductions = _should_apply_extended_q3_reductions(q, extended_reductions)

    allow_tensor_contraction = True if context is None else context.allow_tensor_contraction
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
            info = {
                'quad': nq + optimized_info['quad'],
                'constraint': nc + optimized_info['constraint'],
                'branched': optimized_info['branched'],
                'remaining': optimized_info['remaining'],
                'structural_obstruction': optimized_info.get(
                    'structural_obstruction',
                    optimized_info['remaining'],
                ),
                'gauss_obstruction': optimized_info.get(
                    'gauss_obstruction',
                    optimized_info.get('structural_obstruction', optimized_info['remaining']),
                ),
                'cost_r': optimized_info.get('cost_r', optimized_info['remaining']),
                'phase_states': optimized_info.get('phase_states', 0),
                'phase_splits': optimized_info.get('phase_splits', 0),
                'phase3_backend': optimized_info.get('phase3_backend'),
            }
            context.reduce_cache[cache_key] = (result, dict(info))
            return result, info

    if q.q3 and enable_extended_q3_reductions:
        components = detect_factorization(q)
        if len(components) > 1:
            factorized_total, factorized_info = _sum_factorized_components_scaled(q, components, context=context)
            result = _scale_scaled_complex(factorized_total, scale_half_pow2)
            info = {
                'quad': nq + factorized_info['quad'],
                'constraint': nc + factorized_info['constraint'],
                'branched': factorized_info['branched'],
                'remaining': factorized_info['remaining'],
                'structural_obstruction': factorized_info.get(
                    'structural_obstruction',
                    factorized_info['remaining'],
                ),
                'gauss_obstruction': factorized_info.get(
                    'gauss_obstruction',
                    factorized_info.get('structural_obstruction', factorized_info['remaining']),
                ),
                'cost_r': factorized_info.get('cost_r', factorized_info['remaining']),
                'phase_states': factorized_info.get('phase_states', 0),
                'phase_splits': factorized_info.get('phase_splits', 0),
                'phase3_backend': factorized_info.get('phase3_backend'),
            }
            context.reduce_cache[cache_key] = (result, dict(info))
            return result, info

    if not q.q3:
        # Over this engine's binary free variables, every q3-free kernel is
        # summed exactly here, including odd q1 / odd q2 structure. The
        # single-variable "quadratic" classification below is narrower: it is
        # only the Prop. 9 elimination rule used before reaching this reducer.
        # Scaled mode remains the robust default. Callers that explicitly
        # requested plain ``complex`` arithmetic can take the faster unscaled
        # q3-free solver and normalize back into a scaled tuple afterwards.
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
        result = _scale_scaled_complex(total, scale_half_pow2)
        structural_obstruction = 0
        gauss_obstruction = _gauss_obstruction(q, structural_obstruction)
        info = {
            'quad': nq,
            'constraint': nc,
            'branched': 0,
            'remaining': 0,
            'structural_obstruction': structural_obstruction,
            'gauss_obstruction': gauss_obstruction,
            'cost_r': 0,
            'phase_states': phase_info['phase_states'],
            'phase_splits': phase_info['phase_splits'],
            'phase3_backend': _q3_free_phase3_backend_name(q),
        }
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

    # Phase 2: if a genuine cubic core remains, try conditional branching.
    if q.n > 0 and not prefer_direct_phase3:
        classification_lookup = _classification_lookup(q)
        # Find best branch variable: odd-q1 var that unlocks most even partners
        best_var, best_unlocks = -1, -1
        for var in range(q.n):
            if classification_lookup[var][q.q1[var] % q.mod_q1][0] != _CLASS_CUBIC:
                continue
            if q.q1[var] % 2 == 0:
                continue  # only branch on odd-q1 vars
            # Count how many even-q1 vars are currently blocked by odd coupling to var
            unlocks = 0
            for j in range(q.n):
                if j == var:
                    continue
                if q.q1[j] % 2 != 0:
                    continue  # j must be even-q1
                key = (min(var,j), max(var,j))
                if q.q2.get(key, 0) % 2 != 0:  # odd coupling to var
                    unlocks += 1
            if unlocks > best_unlocks:
                best_var, best_unlocks = var, unlocks

        if best_var >= 0 and best_unlocks > 0:
            # Branch on best_var
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
                # Fix best_var = fval, get restricted function
                q_branch = _fix_variable(q, best_var, fval, context=context)
                # Recurse
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
        info = {
            'quad': nq + split_info['quad'],
            'constraint': nc + split_info['constraint'],
            'branched': nb + split_info['branched'],
            'remaining': split_info['remaining'],
            'structural_obstruction': split_info.get('structural_obstruction', split_info['remaining']),
            'gauss_obstruction': split_info.get(
                'gauss_obstruction',
                split_info.get('structural_obstruction', split_info['remaining']),
            ),
            'cost_r': split_info.get('cost_r', split_info['remaining']),
            'phase_states': split_info.get('phase_states', 0),
            'phase_splits': split_info.get('phase_splits', 0),
            'phase3_backend': split_info.get('phase3_backend'),
        }
        context.reduce_cache[cache_key] = (result, dict(info))
        return result, info

    # Phase 3: no profitable branching. Choose among a low-treewidth
    # pure-Python DP, an optional quimb tensor contraction on the reduced core,
    # and q3-cover branching into q3-free leaves.
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
    info = {
        'quad': nq + phase3_info['quad'],
        'constraint': nc + phase3_info['constraint'],
        'branched': nb + phase3_info['branched'],
        'remaining': phase3_info['remaining'],
        'structural_obstruction': phase3_info.get('structural_obstruction', phase3_info['remaining']),
        'gauss_obstruction': phase3_info.get(
            'gauss_obstruction',
            phase3_info.get('structural_obstruction', phase3_info['remaining']),
        ),
        'cost_r': phase3_info.get('cost_r', phase3_info['remaining']),
        'phase_states': phase3_info.get('phase_states', 0),
        'phase_splits': phase3_info.get('phase_splits', 0),
        'phase3_backend': phase3_info.get('phase3_backend'),
    }
    context.reduce_cache[cache_key] = (result, dict(info))
    return result, info


def _reduce_and_sum(q, context=None):
    result, info = _reduce_and_sum_scaled(q, context=context)
    return _scaled_to_complex(result), info


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

    reduced = _phase_function_from_parts(
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
        classification_lookup = _classification_lookup(q)
        chosen_action = None
        first_blocked_quadratic = None
        for var in range(q.n):
            entry = classification_lookup[var][q.q1[var] % q.mod_q1]
            tag = entry[0]
            if tag >= _CLASS_CONSTRAINT_DECOUPLED:
                if tag == _CLASS_CONSTRAINT_ZERO:
                    return None, 0, {'quad': nq, 'constraint': nc}, []
                if tag == _CLASS_CONSTRAINT_DECOUPLED:
                    decoupled_constraints.append(var)
                    continue
                if chosen_action is None:
                    chosen_action = (tag, var, entry)
                continue
            if tag == _CLASS_QUADRATIC:
                if entry[2]:
                    if first_blocked_quadratic is None:
                        first_blocked_quadratic = var
                    continue
                if chosen_action is None:
                    chosen_action = (tag, var, entry)

        if decoupled_constraints:
            q, half_pow2 = _elim_decoupled_constraints_batch(q, decoupled_constraints)
            scale_half_pow2 += half_pow2
            nc += len(decoupled_constraints)
            changed = True
            continue

        if chosen_action is None:
            if first_blocked_quadratic is not None:
                blocked_quadratics.append(first_blocked_quadratic)
            continue

        tag, var, entry = chosen_action
        if tag == _CLASS_QUADRATIC:
            q, half_pow2 = _elim_quadratic(q, var)
            scale_half_pow2 += half_pow2
            nq += 1
            changed = True
            continue
        if tag == _CLASS_CONSTRAINT_PARITY:
            partners = entry[1]
            target = 1 if entry[2] == (q.mod_q1 // 2) else 0
            if len(partners) == 1:
                result = _elim_single_partner_constraint(q, var, partners[0], target)
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
    return q, scale_half_pow2, {'quad': nq, 'constraint': nc}, blocked_quadratics


@lru_cache(maxsize=16)
def _omega_table(level: int) -> tuple[complex, ...]:
    modulus = 1 << level
    return tuple(cmath.exp(2j * cmath.pi * residue / modulus) for residue in range(modulus))


@lru_cache(maxsize=16)
def _omega_scaled_table(level: int) -> tuple[ScaledComplex, ...]:
    return tuple(_make_scaled_complex(value) for value in _omega_table(level))


@lru_cache(maxsize=16)
def _omega_scaled_arrays(level: int) -> tuple[np.ndarray, np.ndarray]:
    return _scaled_table_to_arrays(_omega_scaled_table(level))


def _product_q1_sum(q1, level: int = 3):
    omega = _omega_table(level)
    modulus = 1 << level
    total = 1.0 + 0j
    for coeff in q1:
        total *= 1 + omega[coeff % modulus]
    return total


def _product_q1_sum_scaled(q1, level: int = 3):
    omega = _omega_table(level)
    modulus = 1 << level
    total = _ONE_SCALED
    for coeff in q1:
        total = _mul_scaled_complex(total, _make_scaled_complex(1 + omega[coeff % modulus]))
    return total


# ==================================================================
# q3-free exact summation and scaled-number helpers
# ==================================================================

def _q3_free_graph(q):
    """Return the q2 graph with edge phases represented in q1 residues."""
    adjacency = [dict() for _ in range(q.n)]
    edges = []
    for (i, j), value in q.q2.items():
        phase_shift = ((q.mod_q1 // q.mod_q2) * value) % q.mod_q1
        adjacency[i][j] = phase_shift
        adjacency[j][i] = phase_shift
        edges.append((i, j, phase_shift))
    return adjacency, edges


def _make_scaled_complex(value):
    return _normalize_scaled_complex(complex(value), 0)


def _normalize_scaled_complex(value, half_pow2_exp):
    if value == 0j:
        return 0j, 0

    magnitude = max(abs(value.real), abs(value.imag))
    _, shift = math.frexp(magnitude)
    if shift:
        value = complex(
            math.ldexp(value.real, -shift),
            math.ldexp(value.imag, -shift),
        )
        half_pow2_exp += 2 * shift
    return value, half_pow2_exp


def _renormalize_scaled_complex_if_needed(value, half_pow2_exp):
    if value == 0j:
        return 0j, 0

    magnitude = max(abs(value.real), abs(value.imag))
    if _SCALED_RENORMALIZE_MIN <= magnitude < _SCALED_RENORMALIZE_MAX:
        return value, half_pow2_exp
    return _normalize_scaled_complex(value, half_pow2_exp)


def _scale_scaled_complex(scaled, half_pow2_exp):
    value, base_half_pow2_exp = scaled
    return _normalize_scaled_complex(value, base_half_pow2_exp + half_pow2_exp)


def _scale_complex_array_by_half_pow2(values: np.ndarray, half_pow2_exp: np.ndarray) -> np.ndarray:
    """Vectorized companion to ``_scale_complex_by_half_pow2``."""
    scaled = np.asarray(values, dtype=np.complex128).copy()
    exponents = np.asarray(half_pow2_exp, dtype=np.int64).copy()
    if scaled.size == 0:
        return scaled

    odd_mask = (exponents & 1) != 0
    positive_odd = odd_mask & (exponents > 0)
    negative_odd = odd_mask & (exponents < 0)
    if np.any(positive_odd):
        scaled[positive_odd] *= _SQRT2
        exponents[positive_odd] -= 1
    if np.any(negative_odd):
        scaled[negative_odd] *= _INV_SQRT2
        exponents[negative_odd] += 1

    shift = exponents // 2
    return np.ldexp(scaled.real, shift) + 1j * np.ldexp(scaled.imag, shift)


def _normalize_scaled_complex_arrays(
    values: np.ndarray,
    half_pow2_exp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize arrays of scaled-complex values elementwise."""
    values_array = np.asarray(values, dtype=np.complex128).copy()
    exponents = np.asarray(half_pow2_exp, dtype=np.int64).copy()
    if values_array.size == 0:
        return values_array, exponents

    zero_mask = values_array == 0j
    if np.any(~zero_mask):
        magnitudes = np.maximum(np.abs(values_array.real), np.abs(values_array.imag))
        _, shifts = np.frexp(magnitudes[~zero_mask])
        shift_array = np.zeros_like(exponents)
        shift_array[~zero_mask] = shifts.astype(np.int64, copy=False)
        nonzero_shift = (~zero_mask) & (shift_array != 0)
        if np.any(nonzero_shift):
            values_array.real[nonzero_shift] = np.ldexp(
                values_array.real[nonzero_shift],
                -shift_array[nonzero_shift],
            )
            values_array.imag[nonzero_shift] = np.ldexp(
                values_array.imag[nonzero_shift],
                -shift_array[nonzero_shift],
            )
            exponents[nonzero_shift] += 2 * shift_array[nonzero_shift]
    exponents[zero_mask] = 0
    return values_array, exponents


def _scaled_arrays_from_constant(
    scaled: ScaledComplex,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast one scaled value into array form."""
    value, half_pow2_exp = scaled
    return (
        np.full(shape, complex(value), dtype=np.complex128),
        np.full(shape, int(half_pow2_exp), dtype=np.int64),
    )


def _scaled_table_to_arrays(table: Sequence[ScaledComplex]) -> tuple[np.ndarray, np.ndarray]:
    """Pack a scaled table into parallel value/exponent arrays."""
    return (
        np.asarray([complex(value) for value, _exp in table], dtype=np.complex128),
        np.asarray([int(exp) for _value, exp in table], dtype=np.int64),
    )


def _scaled_list_to_arrays(
    scaled_list: Sequence[ScaledComplex],
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape a list of scaled values into parallel arrays."""
    values = np.asarray([complex(value) for value, _exp in scaled_list], dtype=np.complex128)
    exponents = np.asarray([int(exp) for _value, exp in scaled_list], dtype=np.int64)
    return values.reshape(shape), exponents.reshape(shape)


def _mul_scaled_complex_arrays(
    left_values: np.ndarray,
    left_half_pow2_exp: np.ndarray,
    right_values: np.ndarray,
    right_half_pow2_exp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise batch multiplication of scaled-complex arrays."""
    return _normalize_scaled_complex_arrays(
        np.asarray(left_values, dtype=np.complex128) * np.asarray(right_values, dtype=np.complex128),
        np.asarray(left_half_pow2_exp, dtype=np.int64) + np.asarray(right_half_pow2_exp, dtype=np.int64),
    )


def _add_scaled_complex_arrays(
    left_values: np.ndarray,
    left_half_pow2_exp: np.ndarray,
    right_values: np.ndarray,
    right_half_pow2_exp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise batch addition of scaled-complex arrays."""
    left_values = np.asarray(left_values, dtype=np.complex128)
    left_exponents = np.asarray(left_half_pow2_exp, dtype=np.int64)
    right_values = np.asarray(right_values, dtype=np.complex128)
    right_exponents = np.asarray(right_half_pow2_exp, dtype=np.int64)

    result_values = left_values.copy()
    result_exponents = left_exponents.copy()

    left_zero = left_values == 0j
    right_zero = right_values == 0j
    take_right = left_zero & ~right_zero
    if np.any(take_right):
        result_values[take_right] = right_values[take_right]
        result_exponents[take_right] = right_exponents[take_right]

    active = ~(left_zero | right_zero)
    if not np.any(active):
        result_exponents[result_values == 0j] = 0
        return result_values, result_exponents

    base_values = left_values[active].copy()
    base_exponents = left_exponents[active].copy()
    other_values = right_values[active].copy()
    other_exponents = right_exponents[active].copy()

    swap_mask = base_exponents < other_exponents
    if np.any(swap_mask):
        base_values_swap = base_values[swap_mask].copy()
        base_exponents_swap = base_exponents[swap_mask].copy()
        base_values[swap_mask] = other_values[swap_mask]
        base_exponents[swap_mask] = other_exponents[swap_mask]
        other_values[swap_mask] = base_values_swap
        other_exponents[swap_mask] = base_exponents_swap

    aligned_other = _scale_complex_array_by_half_pow2(
        other_values,
        other_exponents - base_exponents,
    )
    summed_values, summed_exponents = _normalize_scaled_complex_arrays(
        base_values + aligned_other,
        base_exponents,
    )
    result_values[active] = summed_values
    result_exponents[active] = summed_exponents
    result_exponents[result_values == 0j] = 0
    return result_values, result_exponents


def _mul_scaled_complex(left, right):
    left_value, left_half_pow2_exp = left
    right_value, right_half_pow2_exp = right
    return _renormalize_scaled_complex_if_needed(
        left_value * right_value,
        left_half_pow2_exp + right_half_pow2_exp,
    )


def _is_binary_phase_quadratic(q) -> bool:
    """Return whether ``q`` only contributes +/-1 phases on computational basis states."""
    if not _is_half_phase_q2(q):
        return False
    return _is_binary_phase_q1_vector(q.q1, level=q.level)


def _is_half_phase_q2(q) -> bool:
    """Return whether every quadratic coupling is in the half-phase class."""
    if q.q3:
        return False
    half_q2 = q.mod_q2 // 2 if q.mod_q2 else 0
    for coeff in q.q2.values():
        residue = coeff % q.mod_q2
        if residue not in (0, half_q2):
            return False
    return True


def _is_binary_phase_q1_vector(q1, *, level: int) -> bool:
    """Return whether every unary residue is binary (`0` or `half`)."""
    modulus = 1 << level
    half_q1 = modulus // 2
    for coeff in q1:
        residue = int(coeff) % modulus
        if residue not in (0, half_q1):
            return False
    return True


def _nonbinary_unary_support_size(q1, *, level: int) -> int:
    """Count unary residues outside the binary-phase class (`0` or `half`)."""
    modulus = 1 << level
    half_q1 = modulus // 2
    support_size = 0
    for coeff in q1:
        residue = int(coeff) % modulus
        if residue not in (0, half_q1):
            support_size += 1
    return support_size


def _is_qubit_quadratic_tensor_q1_vector(q1, *, level: int) -> bool:
    """Return whether ``q1`` lies in BL26's 4-element qubit quadratic class."""
    threshold = max(1, (1 << level) // 4)
    if threshold <= 1:
        return True
    for coeff in q1:
        if int(coeff) % threshold:
            return False
    return True


def _is_qubit_quadratic_tensor(q) -> bool:
    """Return whether ``q`` is a q3-free BL26 qubit quadratic tensor."""
    return not q.q3 and _qubit_quadratic_tensor_obstruction(q) == 0


def _q3_free_phase3_backend_name(q) -> str:
    """Report the exact backend family for a q3-free kernel."""
    return "quadratic_tensor" if _is_qubit_quadratic_tensor(q) else "q3_free"


def _component_fixed_nonbinary_unary_support_size(
    component_q,
    variables: Sequence[int],
    *,
    lambda_offset: int,
) -> int:
    """Count non-binary unary residues on the original, non-dual variables."""
    modulus = 1 << component_q.level
    half_q1 = modulus // 2
    support_size = 0
    for local_idx, var in enumerate(variables):
        if var >= lambda_offset:
            continue
        residue = int(component_q.q1[local_idx]) % modulus
        if residue not in (0, half_q1):
            support_size += 1
    return support_size


def _build_binary_phase_quadratic_plan(q) -> _BinaryPhaseQuadraticPlan | None:
    """Precompute the half-phase q2 elimination schedule for a q3-free kernel."""
    if not _is_half_phase_q2(q):
        return None

    adjacency = np.zeros((q.n, q.n), dtype=np.bool_)
    half_q2 = q.mod_q2 // 2 if q.mod_q2 else 0
    for (left, right), coeff in q.q2.items():
        if (coeff % q.mod_q2) == half_q2:
            adjacency[left, right] = True
            adjacency[right, left] = True

    active_count = q.n
    partner_swaps: list[int] = []
    pivot_swaps: list[int] = []
    c1_rows: list[np.ndarray] = []
    c2_rows: list[np.ndarray] = []
    c1_and_c2_rows: list[np.ndarray] = []
    half_pow2 = 0

    while active_count:
        block = adjacency[:active_count, :active_count]
        degrees = block.sum(axis=1, dtype=np.int64)
        if not np.any(degrees):
            break

        pivot_local = int(np.argmax(degrees))
        neighbors_local = np.flatnonzero(block[pivot_local])
        partner_local = int(neighbors_local[0])
        if partner_local == pivot_local:
            if neighbors_local.size < 2:
                return None
            partner_local = int(neighbors_local[1])

        if pivot_local > partner_local:
            pivot_local, partner_local = partner_local, pivot_local

        first_pivot = active_count - 2
        second_pivot = active_count - 1

        partner_swap = -1
        if partner_local != second_pivot:
            partner_swap = partner_local
            _swap_dense_q2_variables(np.zeros(q.n, dtype=np.bool_), adjacency, partner_local, second_pivot)
            if pivot_local == second_pivot:
                pivot_local = partner_local

        pivot_swap = -1
        if pivot_local != first_pivot:
            pivot_swap = pivot_local
            _swap_dense_q2_variables(np.zeros(q.n, dtype=np.bool_), adjacency, pivot_local, first_pivot)

        c1 = adjacency[first_pivot, :first_pivot].copy()
        c2 = adjacency[second_pivot, :first_pivot].copy()
        c1_rows.append(c1)
        c2_rows.append(c2)
        c1_and_c2_rows.append(np.logical_and(c1, c2))
        partner_swaps.append(partner_swap)
        pivot_swaps.append(pivot_swap)

        if first_pivot:
            update = np.logical_xor(np.outer(c1, c2), np.outer(c2, c1))
            subblock = adjacency[:first_pivot, :first_pivot]
            subblock ^= update
            np.fill_diagonal(subblock, False)
        adjacency[first_pivot:active_count, :active_count] = False
        adjacency[:active_count, first_pivot:active_count] = False
        active_count = first_pivot
        half_pow2 += 2

    return _BinaryPhaseQuadraticPlan(
        n=q.n,
        residual_active_count=active_count,
        half_pow2_exp=half_pow2,
        partner_swaps=tuple(partner_swaps),
        pivot_swaps=tuple(pivot_swaps),
        c1_rows=tuple(c1_rows),
        c2_rows=tuple(c2_rows),
        c1_and_c2_rows=tuple(c1_and_c2_rows),
    )


def _evaluate_binary_phase_quadratic_plan_scaled_batch(
    plan: _BinaryPhaseQuadraticPlan,
    q1_batch: np.ndarray,
    *,
    level: int,
) -> list[ScaledComplex]:
    """Evaluate many binary-phase q1 assignments over one fixed dense q2 plan."""
    q1_batch = np.ascontiguousarray(np.asarray(q1_batch, dtype=np.int64))
    if q1_batch.ndim != 2 or q1_batch.shape[1] != plan.n:
        raise ValueError("q1_batch must have shape (batch, plan.n).")

    half_q1 = (1 << level) // 2
    work = np.remainder(q1_batch, 1 << level) == half_q1
    sign_bits = np.zeros(work.shape[0], dtype=np.bool_)
    active_count = plan.n

    for partner_swap, pivot_swap, c1, c2, c1_and_c2 in zip(
        plan.partner_swaps,
        plan.pivot_swaps,
        plan.c1_rows,
        plan.c2_rows,
        plan.c1_and_c2_rows,
    ):
        first_pivot = active_count - 2
        second_pivot = active_count - 1
        if partner_swap >= 0:
            tmp = work[:, partner_swap].copy()
            work[:, partner_swap] = work[:, second_pivot]
            work[:, second_pivot] = tmp
        if pivot_swap >= 0:
            tmp = work[:, pivot_swap].copy()
            work[:, pivot_swap] = work[:, first_pivot]
            work[:, first_pivot] = tmp

        sign_bits ^= np.logical_and(work[:, first_pivot], work[:, second_pivot])
        if first_pivot:
            work[:, :first_pivot] ^= (
                (work[:, [first_pivot]] & c2[None, :])
                ^ (work[:, [second_pivot]] & c1[None, :])
                ^ c1_and_c2[None, :]
            )
        work[:, first_pivot:active_count] = False
        active_count = first_pivot

    zero_mask = np.any(work[:, :plan.residual_active_count], axis=1)
    result: list[ScaledComplex] = []
    final_half_pow2 = plan.half_pow2_exp + (2 * plan.residual_active_count)
    for is_zero, sign_bit in zip(zero_mask, sign_bits):
        if is_zero:
            result.append(_ZERO_SCALED)
        else:
            result.append(
                _scale_scaled_complex(
                    _make_scaled_complex(-1.0 if bool(sign_bit) else 1.0),
                    final_half_pow2,
                )
            )
    return result


def _sum_binary_phase_quadratic_scaled(q) -> ScaledComplex | None:
    """Exactly sum a q3-free kernel whose phases are only +/-1.

    In this regime the exponent is a boolean quadratic form over GF(2):
    ``sum_{i<j} a_ij x_i x_j + sum_i b_i x_i``. Summing over any edge pair
    ``(i, j)`` produces another boolean quadratic form on the remaining
    variables:

    ``sum_{x_i, x_j} (-1)^{x_i x_j + p x_i + q x_j} = 2 (-1)^{p q}``

    where ``p`` and ``q`` are affine forms in the remaining variables. This
    turns the exact sum into a sequence of dense GF(2) pivot eliminations with
    O(n^3) worst-case cost instead of the generic feedback-set branching path.
    """
    if not _is_binary_phase_quadratic(q):
        return None
    if q.n == 0:
        return _ONE_SCALED

    plan = _build_binary_phase_quadratic_plan(q)
    if plan is None:
        return None
    return _evaluate_binary_phase_quadratic_plan_scaled_batch(
        plan,
        np.asarray([q.q1], dtype=np.int64),
        level=q.level,
    )[0]


def _sum_half_phase_q2_unary_expansion_with_plan_scaled(
    q1: Sequence[int],
    *,
    level: int,
    plan: _BinaryPhaseQuadraticPlan,
) -> ScaledComplex | None:
    """Exactly sum a half-phase q2 core by expanding only the hard unary terms.

    For each binary variable ``x_i`` with unary phase ``omega^(a_i x_i)``, use

    ``omega^(a_i x_i) = alpha_i + beta_i (-1)^(x_i)``

    where ``alpha_i = (1 + omega^a_i) / 2`` and
    ``beta_i = (1 - omega^a_i) / 2``.

    This turns the exact sum into a weighted combination of binary-phase
    quadratic character sums over the same q2 core, which can be evaluated by
    ``_evaluate_binary_phase_quadratic_plan_scaled_batch``. The expansion is
    only practical when the number of non-binary unary residues is small.
    """
    if len(q1) != plan.n:
        raise ValueError(f"Expected q1 of length {plan.n}, received {len(q1)}.")

    modulus = 1 << level
    half_q1 = modulus // 2
    omega = _omega_table(level)

    fixed_half_positions: list[int] = []
    support_positions: list[int] = []
    alpha_terms: list[complex] = []
    beta_terms: list[complex] = []

    for idx, coeff in enumerate(q1):
        residue = int(coeff) % modulus
        if residue == 0:
            continue
        if residue == half_q1:
            fixed_half_positions.append(idx)
            continue
        support_positions.append(idx)
        phase = omega[residue]
        alpha_terms.append((1.0 + phase) * 0.5)
        beta_terms.append((1.0 - phase) * 0.5)

    support_size = len(support_positions)
    if support_size > _Q3_FREE_HALF_PHASE_UNARY_EXPANSION_MAX_SUPPORT:
        return None

    if support_size == 0:
        base_q1 = np.zeros((1, plan.n), dtype=np.int64)
        if fixed_half_positions:
            base_q1[0, np.asarray(fixed_half_positions, dtype=np.int64)] = half_q1
        return _evaluate_binary_phase_quadratic_plan_scaled_batch(
            plan,
            base_q1,
            level=level,
        )[0]

    base_q1 = np.zeros(plan.n, dtype=np.int64)
    if fixed_half_positions:
        base_q1[np.asarray(fixed_half_positions, dtype=np.int64)] = half_q1

    alpha_array = np.asarray(alpha_terms, dtype=np.complex128)
    beta_array = np.asarray(beta_terms, dtype=np.complex128)
    support_array = np.asarray(support_positions, dtype=np.int64)
    total = _ZERO_SCALED
    mask_count = 1 << support_size
    batch_size = min(_Q3_FREE_HALF_PHASE_UNARY_EXPANSION_BATCH_SIZE, mask_count)

    for start in range(0, mask_count, batch_size):
        stop = min(start + batch_size, mask_count)
        masks = np.arange(start, stop, dtype=np.uint64)
        q1_batch = np.broadcast_to(base_q1, (stop - start, plan.n)).copy()
        coeff_batch = np.ones(stop - start, dtype=np.complex128)

        for local_idx, position in enumerate(support_array):
            bit_is_one = ((masks >> np.uint64(local_idx)) & np.uint64(1)).astype(np.bool_)
            q1_batch[bit_is_one, position] = half_q1
            coeff_batch *= np.where(bit_is_one, beta_array[local_idx], alpha_array[local_idx])

        binary_totals = _evaluate_binary_phase_quadratic_plan_scaled_batch(
            plan,
            q1_batch,
            level=level,
        )
        for coeff, binary_total in zip(coeff_batch, binary_totals):
            if coeff == 0j or binary_total[0] == 0j:
                continue
            total = _add_scaled_complex(
                total,
                _mul_scaled_complex(_make_scaled_complex(coeff), binary_total),
            )

    return total


def _sum_half_phase_q2_unary_expansion_with_plan_scaled_batch(
    q1_batch: np.ndarray,
    *,
    level: int,
    plan: _BinaryPhaseQuadraticPlan,
) -> list[ScaledComplex] | None:
    """Batch exact hard-unary expansion over one shared half-phase q2 core."""
    batch = np.ascontiguousarray(np.asarray(q1_batch, dtype=np.int64))
    if batch.ndim != 2 or batch.shape[1] != plan.n:
        raise ValueError(f"Expected q1_batch with shape (batch, {plan.n}).")
    if len(batch) == 0:
        return []

    modulus = 1 << level
    half_q1 = modulus // 2
    residues = np.remainder(batch, modulus)
    binary_mask = (residues == 0) | (residues == half_q1)
    support_mask = (~binary_mask) & (residues != 0)
    support_sizes = np.count_nonzero(support_mask, axis=1)
    if np.any(support_sizes > _Q3_FREE_HALF_PHASE_UNARY_EXPANSION_MAX_SUPPORT):
        return None

    fixed_mask = residues == half_q1
    omega = _omega_table(level)
    results: list[ScaledComplex] = [_ZERO_SCALED] * len(batch)
    grouped_rows: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
    for row_idx in range(len(batch)):
        support_positions = tuple(np.flatnonzero(support_mask[row_idx]).tolist())
        fixed_positions = tuple(np.flatnonzero(fixed_mask[row_idx] & ~support_mask[row_idx]).tolist())
        grouped_rows.setdefault((support_positions, fixed_positions), []).append(row_idx)

    for (support_positions, fixed_positions), row_indices in grouped_rows.items():
        support_size = len(support_positions)
        row_array = np.asarray(row_indices, dtype=np.int64)
        group_size = len(row_array)

        base_rows = np.zeros((group_size, plan.n), dtype=np.int64)
        if fixed_positions:
            base_rows[:, np.asarray(fixed_positions, dtype=np.int64)] = half_q1

        if support_size == 0:
            group_totals = _evaluate_binary_phase_quadratic_plan_scaled_batch(
                plan,
                base_rows,
                level=level,
            )
            for row_idx, total in zip(row_indices, group_totals):
                results[row_idx] = total
            continue

        support_array = np.asarray(support_positions, dtype=np.int64)
        residue_group = residues[row_array][:, support_array]
        alpha_matrix = (1.0 + np.vectorize(lambda coeff: omega[int(coeff)])(residue_group)) * 0.5
        beta_matrix = (1.0 - np.vectorize(lambda coeff: omega[int(coeff)])(residue_group)) * 0.5
        total_values, total_exponents = _scaled_arrays_from_constant(_ZERO_SCALED, (group_size,))
        mask_count = 1 << support_size
        block_size = min(_Q3_FREE_HALF_PHASE_UNARY_EXPANSION_BATCH_SIZE, mask_count)

        for start in range(0, mask_count, block_size):
            stop = min(start + block_size, mask_count)
            masks = np.arange(start, stop, dtype=np.uint64)
            block_rows = stop - start
            q1_expanded = np.broadcast_to(
                base_rows[:, None, :],
                (group_size, block_rows, plan.n),
            ).copy()
            coeff_matrix = np.ones((group_size, block_rows), dtype=np.complex128)

            for local_idx, position in enumerate(support_array):
                bit_is_one = ((masks >> np.uint64(local_idx)) & np.uint64(1)).astype(np.bool_)
                if np.any(bit_is_one):
                    q1_expanded[:, bit_is_one, position] = half_q1
                coeff_matrix *= np.where(
                    bit_is_one[None, :],
                    beta_matrix[:, [local_idx]],
                    alpha_matrix[:, [local_idx]],
                )

            binary_totals = _evaluate_binary_phase_quadratic_plan_scaled_batch(
                plan,
                q1_expanded.reshape(group_size * block_rows, plan.n),
                level=level,
            )
            block_values, block_exponents = _scaled_list_to_arrays(
                binary_totals,
                (group_size, block_rows),
            )
            weighted_values, weighted_exponents = _mul_scaled_complex_arrays(
                np.broadcast_to(coeff_matrix, block_values.shape),
                np.zeros(block_values.shape, dtype=np.int64),
                block_values,
                block_exponents,
            )
            block_total_values = weighted_values[:, 0]
            block_total_exponents = weighted_exponents[:, 0]
            for column in range(1, block_rows):
                block_total_values, block_total_exponents = _add_scaled_complex_arrays(
                    block_total_values,
                    block_total_exponents,
                    weighted_values[:, column],
                    weighted_exponents[:, column],
                )
            total_values, total_exponents = _add_scaled_complex_arrays(
                total_values,
                total_exponents,
                block_total_values,
                block_total_exponents,
            )

        for row_idx, value, half_pow2_exp in zip(row_indices, total_values, total_exponents):
            results[row_idx] = (complex(value), int(half_pow2_exp))

    return results


def _sum_half_phase_q2_unary_expansion_scaled(q) -> ScaledComplex | None:
    """Exact hard-unary expansion over a half-phase q2 core, when support is small."""
    if not _is_half_phase_q2(q):
        return None
    if _nonbinary_unary_support_size(q.q1, level=q.level) > _Q3_FREE_HALF_PHASE_UNARY_EXPANSION_MAX_SUPPORT:
        return None
    plan = _build_binary_phase_quadratic_plan(q)
    if plan is None:
        return None
    return _sum_half_phase_q2_unary_expansion_with_plan_scaled(
        q.q1,
        level=q.level,
        plan=plan,
    )


def _half_phase_parity_component_reduction(q) -> tuple[object, int] | ScaledComplex | None:
    """Peel q3-free-preserving low-arity parity constraints.

    A parity constraint with one partner fixes that partner.  A parity
    constraint with two partners enforces ``x_a xor x_b = target``; substituting
    one partner into the other keeps every q3-free half-phase q2 term at most
    pairwise.  Higher-arity parity constraints are deliberately left alone
    because the analogous substitution can create q3 terms.
    """
    if q.q3 or not _is_half_phase_q2(q):
        return None

    reduced_q = q
    scale_half_pow2 = 0
    eliminated = 0

    while True:
        classification_lookup = _classification_lookup(reduced_q)
        adjacency = [set() for _ in range(reduced_q.n)]
        for (left, right), coeff in reduced_q.q2.items():
            if coeff % reduced_q.mod_q2:
                adjacency[left].add(right)
                adjacency[right].add(left)

        best_action = None
        best_score = None
        for var in range(reduced_q.n):
            entry = classification_lookup[var][reduced_q.q1[var] % reduced_q.mod_q1]
            if entry[0] != _CLASS_CONSTRAINT_PARITY:
                continue
            partners = tuple(int(partner) for partner in entry[1])
            target = 1 if int(entry[2]) % reduced_q.mod_q1 == (reduced_q.mod_q1 // 2) else 0
            if len(partners) == 1:
                score = (-1, var)
                action = ("single", var, partners[0], target)
            elif len(partners) == 2:
                left, right = partners
                fill_left_to_right = len(
                    [
                        neighbor
                        for neighbor in adjacency[left]
                        if neighbor not in (right, var) and neighbor not in adjacency[right]
                    ]
                )
                fill_right_to_left = len(
                    [
                        neighbor
                        for neighbor in adjacency[right]
                        if neighbor not in (left, var) and neighbor not in adjacency[left]
                    ]
                )
                if fill_right_to_left <= fill_left_to_right:
                    keep, remove = left, right
                    fill_cost = fill_right_to_left
                else:
                    keep, remove = right, left
                    fill_cost = fill_left_to_right
                score = (fill_cost, len(adjacency[remove]), var)
                action = ("double", var, keep, remove, target)
            else:
                continue
            if best_score is None or score < best_score:
                best_score = score
                best_action = action

        if best_action is None:
            break
        if best_action[0] == "single":
            result = _elim_single_partner_constraint(
                reduced_q,
                best_action[1],
                best_action[2],
                best_action[3],
            )
        else:
            result = _elim_two_partner_constraint_q3_free(
                reduced_q,
                best_action[1],
                best_action[2],
                best_action[3],
                best_action[4],
            )
        if result is None:
            break
        reduced_q, half_pow2 = result
        scale_half_pow2 += half_pow2
        eliminated += 1

    if not eliminated:
        return None
    return reduced_q, scale_half_pow2


def _sum_half_phase_parity_component_reduction_scaled(q) -> ScaledComplex | None:
    """Exact q3-free sum after linear parity-component collapse, when useful."""
    reduction = _half_phase_parity_component_reduction(q)
    if reduction is None:
        return None
    if reduction == _ZERO_SCALED:
        return _ZERO_SCALED
    reduced_q, scale_half_pow2 = reduction
    constant = _make_scaled_complex(cmath.exp(2j * cmath.pi * float(reduced_q.q0)))
    if reduced_q.q0:
        reduced_q = _phase_function_from_parts(
            reduced_q.n,
            level=reduced_q.level,
            q0=Fraction(0),
            q1=reduced_q.q1,
            q2=reduced_q.q2,
            q3=reduced_q.q3,
        )
    total = _mul_scaled_complex(constant, _sum_q3_free_component_scaled(reduced_q))
    return _scale_scaled_complex(total, scale_half_pow2)


def _build_half_phase_mediator_plan(q) -> _HalfPhaseMediatorPlan | None:
    """Plan an exact mediator-elimination pass for half-phase q2 kernels.

    This pass targets the IBM-style pattern where every non-BL unary variable is
    an independent degree-1/2 mediator attached to a lower-treewidth core.
    Eliminating such mediators produces exact unary/pair factors on the core,
    which can then be closed by the generic factor-graph treewidth DP.
    """
    if not _is_half_phase_q2(q):
        return None

    threshold = max(1, q.mod_q1 // 4)
    adjacency = [set() for _ in range(q.n)]
    for (i, j), coeff in q.q2.items():
        if coeff % q.mod_q2:
            adjacency[i].add(j)
            adjacency[j].add(i)

    candidates = [
        var
        for var, coeff in enumerate(q.q1)
        if (coeff % threshold) != 0 and len(adjacency[var]) <= 2
    ]
    if not candidates:
        return None

    candidate_set = set(candidates)
    if any(neighbor in candidate_set for var in candidates for neighbor in adjacency[var]):
        return None

    core_vars = tuple(var for var in range(q.n) if var not in candidate_set)
    core_remap = {var: idx for idx, var in enumerate(core_vars)}

    core_q2 = {
        (core_remap[i], core_remap[j]): coeff
        for (i, j), coeff in q.q2.items()
        if i in core_remap and j in core_remap
    }
    dummy_q2 = {edge: 1 for edge in core_q2}
    mediator_specs: list[_HalfPhaseMediatorSpec] = []

    for var in candidates:
        neighbor_vars = tuple(sorted(core_remap[neighbor] for neighbor in adjacency[var]))
        if len(neighbor_vars) > 2:
            return None
        if len(neighbor_vars) == 2:
            edge = (neighbor_vars[0], neighbor_vars[1])
            dummy_q2.setdefault(edge, 1)
        mediator_specs.append(
            _HalfPhaseMediatorSpec(
                mediator_var=var,
                neighbor_vars=neighbor_vars,
            )
        )

    dummy_core = _phase_function_from_parts(
        len(core_vars),
        level=q.level,
        q0=Fraction(0),
        q1=[0] * len(core_vars),
        q2=dummy_q2,
        q3={},
    )
    dummy_adjacency = [set() for _ in range(len(core_vars))]
    for left, right in dummy_q2:
        dummy_adjacency[left].add(right)
        dummy_adjacency[right].add(left)
    width_limit = _q3_free_treewidth_width_limit()
    degeneracy_lower_bound = _pair_graph_degeneracy(dummy_adjacency)
    if degeneracy_lower_bound > width_limit or degeneracy_lower_bound >= len(candidates):
        return None
    order, width = _min_fill_cubic_order(dummy_core)
    separator_order = _pair_graph_separator_order(dummy_core)
    if separator_order is not None:
        candidate_order, candidate_width = separator_order
        if candidate_width < width:
            order, width = candidate_order, candidate_width
    if width > width_limit or width >= len(candidates):
        return None

    return _HalfPhaseMediatorPlan(
        level=q.level,
        core_vars=core_vars,
        core_q2=core_q2,
        order=tuple(order),
        width=width,
        mediators=tuple(mediator_specs),
    )


def _build_generic_q2_mediator_plan(q) -> _GenericQ2MediatorPlan | None:
    """Plan exact elimination of independent degree<=2 q2 mediators.

    Unlike the half-phase mediator path, this keeps the full q2 residue on each
    mediator edge and collapses the eliminated variables into exact 1- or
    2-qubit boundary factors on the remaining core. The plan is deliberately
    conservative: it only selects an independent set of low-degree mediators so
    the induced factors attach directly to the core without overlap.
    """
    if q.q3 or not q.q2 or _is_half_phase_q2(q):
        return None

    adjacency = [set() for _ in range(q.n)]
    for (left, right), coeff in q.q2.items():
        if coeff % q.mod_q2:
            adjacency[left].add(right)
            adjacency[right].add(left)

    if not any(adjacency):
        return None

    mod_q1 = 1 << q.level
    half_q1 = mod_q1 // 2
    half_q2 = q.mod_q2 // 2 if q.mod_q2 else 0
    mod_q2 = max(1, 1 << (q.level - 1))
    q2_lift = mod_q1 // mod_q2 if mod_q2 else 0

    def candidate_score(var: int) -> tuple[int, int, int]:
        unary_residue = int(q.q1[var]) % mod_q1
        hard_unary = unary_residue not in (0, half_q1)
        hard_edge = any(
            (q.q2.get((min(var, neighbor), max(var, neighbor)), 0) % q.mod_q2) not in (0, half_q2)
            for neighbor in adjacency[var]
        )
        return (int(hard_unary or hard_edge), len(adjacency[var]), -var)

    candidates = sorted(
        (var for var in range(q.n) if 0 < len(adjacency[var]) <= 2),
        key=candidate_score,
        reverse=True,
    )
    if not candidates:
        return None

    selected: list[int] = []
    blocked: set[int] = set()
    for var in candidates:
        if var in blocked:
            continue
        selected.append(var)
        blocked.add(var)
        blocked.update(adjacency[var])

    if not selected:
        return None

    selected_set = set(selected)
    core_vars = tuple(var for var in range(q.n) if var not in selected_set)
    core_remap = {var: idx for idx, var in enumerate(core_vars)}
    core_q2 = {
        (core_remap[i], core_remap[j]): coeff
        for (i, j), coeff in q.q2.items()
        if i in core_remap and j in core_remap
    }

    mediator_specs: list[_GenericQ2MediatorSpec] = []
    factor_scopes: list[tuple[int, ...]] = [edge for edge in core_q2]
    for mediator_var in selected:
        ordered_neighbors = tuple(sorted(adjacency[mediator_var]))
        neighbor_vars = tuple(core_remap[neighbor] for neighbor in ordered_neighbors)
        neighbor_couplings = tuple(
            int(q.q2.get((min(mediator_var, neighbor), max(mediator_var, neighbor)), 0))
            for neighbor in ordered_neighbors
        )
        if len(neighbor_vars) != len(neighbor_couplings):
            return None
        if neighbor_vars:
            factor_scopes.append(neighbor_vars)
        assignment_residue_shifts = tuple(
            sum(
                (q2_lift * int(coeff))
                for neighbor_idx, coeff in enumerate(neighbor_couplings)
                if (assignment >> neighbor_idx) & 1
            ) % mod_q1
            for assignment in range(1 << len(neighbor_vars))
        )
        mediator_specs.append(
            _GenericQ2MediatorSpec(
                mediator_var=mediator_var,
                neighbor_vars=neighbor_vars,
                neighbor_couplings=neighbor_couplings,
                assignment_residue_shifts=assignment_residue_shifts,
            )
        )

    width_limit = _q3_free_treewidth_width_limit()
    degeneracy_lower_bound = _factor_scope_degeneracy(len(core_vars), factor_scopes)
    if degeneracy_lower_bound > width_limit:
        return None

    order, width = _factor_scope_order(len(core_vars), factor_scopes)
    if width > width_limit:
        return None

    return _GenericQ2MediatorPlan(
        level=q.level,
        core_vars=core_vars,
        core_q2=core_q2,
        order=tuple(order),
        width=width,
        mediators=tuple(mediator_specs),
    )


def _factor_scope_order(
    n_vars: int,
    factor_scopes: Sequence[Sequence[int]],
) -> tuple[list[int], int]:
    if n_vars == 0:
        return [], 0

    dummy_q2: dict[tuple[int, int], int] = {}
    for scope in factor_scopes:
        ordered_scope = tuple(sorted({int(var) for var in scope}))
        for left, right in combinations(ordered_scope, 2):
            dummy_q2.setdefault((left, right), 1)

    dummy_q = _phase_function_from_parts(
        n_vars,
        level=3,
        q0=Fraction(0),
        q1=[0] * n_vars,
        q2=dummy_q2,
        q3={},
    )
    order, width = _min_fill_cubic_order(dummy_q)
    separator_order = _pair_graph_separator_order(dummy_q)
    if separator_order is not None:
        candidate_order, candidate_width = separator_order
        if candidate_width < width:
            order, width = candidate_order, candidate_width
    return list(order), int(width)


def _factor_scope_degeneracy(n_vars: int, factor_scopes: Sequence[Sequence[int]]) -> int:
    """Return the degeneracy lower bound of the factor-induced pair graph."""
    adjacency = [set() for _ in range(n_vars)]
    for scope in factor_scopes:
        ordered_scope = tuple(sorted({int(var) for var in scope}))
        for left, right in combinations(ordered_scope, 2):
            adjacency[left].add(right)
            adjacency[right].add(left)
    return _pair_graph_degeneracy(adjacency)


def _build_cluster_boundary_shift_table(
    *,
    cluster_size: int,
    boundary_size: int,
    boundary_couplings: Sequence[tuple[int, int, int]],
    q2_lift: int,
    mod_q1: int,
) -> np.ndarray:
    shift_table = np.zeros((1 << boundary_size, cluster_size), dtype=np.int64)
    for cluster_idx, boundary_idx, coeff in boundary_couplings:
        active_assignments = ((np.arange(1 << boundary_size, dtype=np.int64) >> int(boundary_idx)) & 1).astype(np.bool_)
        if not np.any(active_assignments):
            continue
        shift_table[active_assignments, int(cluster_idx)] = (
            shift_table[active_assignments, int(cluster_idx)] + (q2_lift * int(coeff))
        ) % mod_q1
    return shift_table


def _build_half_phase_cluster_plan(q) -> _HalfPhaseClusterPlan | None:
    """Plan exact elimination of small hard-support clusters onto the remaining core."""
    if not _is_half_phase_q2(q):
        return None

    support = _qubit_quadratic_tensor_obstruction_support(q)
    if not support:
        return None

    adjacency = [set() for _ in range(q.n)]
    for (left, right), coeff in q.q2.items():
        if coeff % q.mod_q2:
            adjacency[left].add(right)
            adjacency[right].add(left)

    support_components = _connected_components_on_vertices(adjacency, support)
    selected_clusters: list[tuple[tuple[int, ...], tuple[int, ...], dict[tuple[int, int], int], tuple[tuple[int, int, int], ...]]] = []
    selected_cluster_vars: set[int] = set()

    for component in support_components:
        cluster_vars = tuple(sorted(int(var) for var in component))
        if (
            not cluster_vars
            or len(cluster_vars) > _Q3_FREE_HALF_PHASE_CLUSTER_MAX_CLUSTER_SIZE
        ):
            continue
        boundary_vars = tuple(
            sorted(
                {
                    int(neighbor)
                    for var in cluster_vars
                    for neighbor in adjacency[var]
                    if neighbor not in component
                }
            )
        )
        if (
            not boundary_vars
            or len(boundary_vars) > _Q3_FREE_HALF_PHASE_CLUSTER_MAX_BOUNDARY
        ):
            continue

        cluster_set = set(cluster_vars)
        boundary_set = set(boundary_vars)
        cluster_remap = {var: idx for idx, var in enumerate(cluster_vars)}
        boundary_remap = {var: idx for idx, var in enumerate(boundary_vars)}
        internal_q2 = {
            (cluster_remap[i], cluster_remap[j]): coeff
            for (i, j), coeff in q.q2.items()
            if i in cluster_set and j in cluster_set
        }
        boundary_couplings: list[tuple[int, int, int]] = []
        for (left, right), coeff in q.q2.items():
            if coeff % q.mod_q2 == 0:
                continue
            if left in cluster_set and right in boundary_set:
                boundary_couplings.append((cluster_remap[left], boundary_remap[right], int(coeff)))
            elif right in cluster_set and left in boundary_set:
                boundary_couplings.append((cluster_remap[right], boundary_remap[left], int(coeff)))

        if not boundary_couplings:
            continue

        selected_clusters.append(
            (
                cluster_vars,
                boundary_vars,
                internal_q2,
                tuple(boundary_couplings),
            )
        )
        selected_cluster_vars.update(cluster_vars)

    if not selected_clusters:
        return None

    core_vars = tuple(var for var in range(q.n) if var not in selected_cluster_vars)
    core_remap = {var: idx for idx, var in enumerate(core_vars)}
    core_q2 = {
        (core_remap[i], core_remap[j]): coeff
        for (i, j), coeff in q.q2.items()
        if i in core_remap and j in core_remap
    }
    mod_q1 = 1 << q.level
    mod_q2 = max(1, 1 << (q.level - 1))
    q2_lift = mod_q1 // mod_q2 if mod_q2 else 0

    factor_scopes: list[tuple[int, ...]] = [edge for edge in core_q2]
    cluster_specs: list[_HalfPhaseClusterSpec] = []
    for cluster_vars, boundary_vars, internal_q2, boundary_couplings in selected_clusters:
        if not all(var in core_remap for var in boundary_vars):
            return None
        boundary_core = tuple(core_remap[var] for var in boundary_vars)
        factor_scopes.append(boundary_core)
        cluster_order, _cluster_width = _factor_scope_order(
            len(cluster_vars),
            list(internal_q2),
        )
        native_treewidth_plan = _build_native_q3_free_treewidth_plan(
            n_vars=len(cluster_vars),
            level=q.level,
            q2=internal_q2,
            order=cluster_order,
        )
        cluster_specs.append(
            _HalfPhaseClusterSpec(
                cluster_vars=cluster_vars,
                boundary_vars=boundary_core,
                internal_q2=internal_q2,
                boundary_couplings=boundary_couplings,
                boundary_shift_table=_build_cluster_boundary_shift_table(
                    cluster_size=len(cluster_vars),
                    boundary_size=len(boundary_vars),
                    boundary_couplings=boundary_couplings,
                    q2_lift=q2_lift,
                    mod_q1=mod_q1,
                ),
                cluster_order=tuple(cluster_order),
                native_treewidth_plan=native_treewidth_plan,
            )
        )

    width_limit = _q3_free_treewidth_width_limit()
    degeneracy_lower_bound = _factor_scope_degeneracy(len(core_vars), factor_scopes)
    if degeneracy_lower_bound > width_limit:
        return None

    order, width = _factor_scope_order(len(core_vars), factor_scopes)
    if width > width_limit:
        return None

    return _HalfPhaseClusterPlan(
        level=q.level,
        core_vars=core_vars,
        core_q2=core_q2,
        order=tuple(order),
        width=width,
        clusters=tuple(cluster_specs),
    )


def _build_generic_q1_cluster_plan(q) -> _HalfPhaseClusterPlan | None:
    """Plan exact elimination of small bad-q1 clusters under arbitrary q2."""
    if q.q3 or not q.q2 or _is_half_phase_q2(q):
        return None

    threshold = max(1, q.mod_q1 // 4)
    support = tuple(
        int(var)
        for var, coeff in enumerate(q.q1)
        if int(coeff) % threshold
    )
    if not support:
        return None

    adjacency = [set() for _ in range(q.n)]
    for (left, right), coeff in q.q2.items():
        if coeff % q.mod_q2:
            adjacency[left].add(right)
            adjacency[right].add(left)

    support_components = _connected_components_on_vertices(adjacency, support)
    selected_clusters: list[tuple[tuple[int, ...], tuple[int, ...], dict[tuple[int, int], int], tuple[tuple[int, int, int], ...]]] = []
    selected_cluster_vars: set[int] = set()

    for component in support_components:
        cluster_vars = tuple(sorted(int(var) for var in component))
        if (
            not cluster_vars
            or len(cluster_vars) > _Q3_FREE_HALF_PHASE_CLUSTER_MAX_CLUSTER_SIZE
        ):
            continue
        boundary_vars = tuple(
            sorted(
                {
                    int(neighbor)
                    for var in cluster_vars
                    for neighbor in adjacency[var]
                    if neighbor not in component
                }
            )
        )
        if (
            not boundary_vars
            or len(boundary_vars) > _Q3_FREE_HALF_PHASE_CLUSTER_MAX_BOUNDARY
        ):
            continue

        cluster_set = set(cluster_vars)
        boundary_set = set(boundary_vars)
        cluster_remap = {var: idx for idx, var in enumerate(cluster_vars)}
        boundary_remap = {var: idx for idx, var in enumerate(boundary_vars)}
        internal_q2 = {
            (cluster_remap[i], cluster_remap[j]): coeff
            for (i, j), coeff in q.q2.items()
            if i in cluster_set and j in cluster_set
        }
        boundary_couplings: list[tuple[int, int, int]] = []
        for (left, right), coeff in q.q2.items():
            if coeff % q.mod_q2 == 0:
                continue
            if left in cluster_set and right in boundary_set:
                boundary_couplings.append((cluster_remap[left], boundary_remap[right], int(coeff)))
            elif right in cluster_set and left in boundary_set:
                boundary_couplings.append((cluster_remap[right], boundary_remap[left], int(coeff)))

        if not boundary_couplings:
            continue

        selected_clusters.append(
            (
                cluster_vars,
                boundary_vars,
                internal_q2,
                tuple(boundary_couplings),
            )
        )
        selected_cluster_vars.update(cluster_vars)

    if not selected_clusters:
        return None

    core_vars = tuple(var for var in range(q.n) if var not in selected_cluster_vars)
    core_remap = {var: idx for idx, var in enumerate(core_vars)}
    core_q2 = {
        (core_remap[i], core_remap[j]): coeff
        for (i, j), coeff in q.q2.items()
        if i in core_remap and j in core_remap
    }
    mod_q1 = 1 << q.level
    mod_q2 = max(1, 1 << (q.level - 1))
    q2_lift = mod_q1 // mod_q2 if mod_q2 else 0

    factor_scopes: list[tuple[int, ...]] = [edge for edge in core_q2]
    cluster_specs: list[_HalfPhaseClusterSpec] = []
    for cluster_vars, boundary_vars, internal_q2, boundary_couplings in selected_clusters:
        if not all(var in core_remap for var in boundary_vars):
            return None
        boundary_core = tuple(core_remap[var] for var in boundary_vars)
        factor_scopes.append(boundary_core)
        cluster_order, _cluster_width = _factor_scope_order(
            len(cluster_vars),
            list(internal_q2),
        )
        native_treewidth_plan = _build_native_q3_free_treewidth_plan(
            n_vars=len(cluster_vars),
            level=q.level,
            q2=internal_q2,
            order=cluster_order,
        )
        cluster_specs.append(
            _HalfPhaseClusterSpec(
                cluster_vars=cluster_vars,
                boundary_vars=boundary_core,
                internal_q2=internal_q2,
                boundary_couplings=boundary_couplings,
                boundary_shift_table=_build_cluster_boundary_shift_table(
                    cluster_size=len(cluster_vars),
                    boundary_size=len(boundary_vars),
                    boundary_couplings=boundary_couplings,
                    q2_lift=q2_lift,
                    mod_q1=mod_q1,
                ),
                cluster_order=tuple(cluster_order),
                native_treewidth_plan=native_treewidth_plan,
            )
        )

    width_limit = _q3_free_treewidth_width_limit()
    degeneracy_lower_bound = _factor_scope_degeneracy(len(core_vars), factor_scopes)
    if degeneracy_lower_bound > width_limit:
        return None

    order, width = _factor_scope_order(len(core_vars), factor_scopes)
    if width > width_limit:
        return None

    return _HalfPhaseClusterPlan(
        level=q.level,
        core_vars=core_vars,
        core_q2=core_q2,
        order=tuple(order),
        width=width,
        clusters=tuple(cluster_specs),
    )


def _build_q1_cluster_plan(q) -> _HalfPhaseClusterPlan | None:
    """Build the best exact hard-q1 cluster plan available for ``q``."""
    cluster_plan = _build_half_phase_cluster_plan(q)
    if cluster_plan is not None:
        return cluster_plan
    return _build_generic_q1_cluster_plan(q)


def _fold_phase_shifted_q1_batch(
    q1_batch: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Deduplicate identical phase-shifted q1 rows while preserving encounter order."""
    batch = np.ascontiguousarray(np.asarray(q1_batch, dtype=np.int64))
    if batch.ndim != 2:
        raise ValueError("Expected q1_batch to have shape (batch, n_vars).")
    if len(batch) == 0:
        return np.zeros((0, batch.shape[1]), dtype=np.int64), []

    row_map: dict[tuple[int, ...], int] = {}
    unique_rows: list[np.ndarray] = []
    inverse: list[int] = []
    for row in batch:
        key = tuple(int(value) for value in row.tolist())
        existing = row_map.get(key)
        if existing is None:
            existing = len(unique_rows)
            row_map[key] = existing
            unique_rows.append(row.copy())
        inverse.append(existing)

    unique_batch = (
        np.vstack(unique_rows)
        if unique_rows
        else np.zeros((0, batch.shape[1]), dtype=np.int64)
    )
    return unique_batch, inverse


def _evaluate_half_phase_cluster_plan_scaled(
    cluster_plan: _HalfPhaseClusterPlan,
    q1_local: Sequence[int],
) -> ScaledComplex:
    """Evaluate one exact hard-support-cluster plan under a concrete q1 vector."""
    max_index = max(
        max(cluster_plan.core_vars, default=-1),
        max(
            (var for spec in cluster_plan.clusters for var in spec.cluster_vars),
            default=-1,
        ),
    )
    if len(q1_local) <= max_index:
        raise ValueError(
            f"Expected q1_local to cover cluster-plan indices through {max_index}, "
            f"received length {len(q1_local)}."
        )

    mod_q1 = 1 << cluster_plan.level
    mod_q2 = max(1, 1 << (cluster_plan.level - 1))
    q2_lift = mod_q1 // mod_q2 if mod_q2 else 0
    omega_scaled = _omega_scaled_table(cluster_plan.level)

    factors: dict[tuple[int, ...], list[ScaledComplex]] = {}
    scalar = _ONE_SCALED

    for core_idx, var in enumerate(cluster_plan.core_vars):
        residue = int(q1_local[var]) % mod_q1
        if residue:
            scalar = _mul_scaled_complex(
                scalar,
                _combine_factor_scaled(
                    factors,
                    (core_idx,),
                    [_ONE_SCALED, omega_scaled[residue]],
                ),
            )

    for (left, right), coeff in cluster_plan.core_q2.items():
        residue = (q2_lift * int(coeff)) % mod_q1
        if residue:
            scalar = _mul_scaled_complex(
                scalar,
                _combine_factor_scaled(
                    factors,
                    (left, right),
                    [
                        _ONE_SCALED,
                        _ONE_SCALED,
                        _ONE_SCALED,
                        omega_scaled[residue],
                    ],
                ),
            )

    for spec in cluster_plan.clusters:
        base_cluster_q1 = np.asarray(
            [int(q1_local[var]) % mod_q1 for var in spec.cluster_vars],
            dtype=np.int64,
        )
        boundary_count = 1 << len(spec.boundary_vars)
        expanded_batch = np.broadcast_to(
            base_cluster_q1[None, :],
            (boundary_count, len(spec.cluster_vars)),
        ).copy()
        if spec.boundary_shift_table is not None and spec.boundary_shift_table.size:
            expanded_batch = (expanded_batch + spec.boundary_shift_table) % mod_q1
        folded_batch, folded_inverse = _fold_phase_shifted_q1_batch(expanded_batch)
        folded_totals = _sum_q3_free_treewidth_dp_scaled_batch(
            n_vars=len(spec.cluster_vars),
            level=cluster_plan.level,
            q1_batch=folded_batch,
            q2=spec.internal_q2,
            order=spec.cluster_order,
            native_plan=spec.native_treewidth_plan,
        )
        table = [folded_totals[idx] for idx in folded_inverse]
        scalar = _mul_scaled_complex(
            scalar,
            _combine_factor_scaled(factors, spec.boundary_vars, table),
        )

    total, _ = _sum_factor_tables_scaled(
        len(cluster_plan.core_vars),
        factors,
        cluster_plan.order,
        scalar=scalar,
    )
    return total


def _build_core_factor_batch(
    *,
    level: int,
    core_vars: Sequence[int],
    core_q2: dict[tuple[int, int], int],
    q1_local_batch: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]]]:
    """Build batched core unary/q2 factors shared by mediator and cluster paths."""
    batch = np.ascontiguousarray(np.asarray(q1_local_batch, dtype=np.int64))
    batch_size = len(batch)
    factors: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
    scalar_values, scalar_exponents = _scaled_arrays_from_constant(_ONE_SCALED, (batch_size,))
    mod_q1 = 1 << level
    mod_q2 = max(1, 1 << (level - 1))
    q2_lift = mod_q1 // mod_q2 if mod_q2 else 0
    omega_values, omega_exponents = _omega_scaled_arrays(level)

    for core_idx, var in enumerate(core_vars):
        residues = np.remainder(batch[:, var], mod_q1)
        if not np.any(residues):
            continue
        table_values, table_exponents = _scaled_arrays_from_constant(_ONE_SCALED, (batch_size, 2))
        table_values[:, 1] = omega_values[residues]
        table_exponents[:, 1] = omega_exponents[residues]
        factor_values, factor_exponents = _combine_factor_scaled_batch(
            factors,
            (core_idx,),
            table_values,
            table_exponents,
            batch_size=batch_size,
        )
        scalar_values, scalar_exponents = _mul_scaled_complex_arrays(
            scalar_values,
            scalar_exponents,
            factor_values,
            factor_exponents,
        )

    for (left, right), coeff in core_q2.items():
        residue = (q2_lift * int(coeff)) % mod_q1
        if not residue:
            continue
        phase_value = (omega_values[residue], int(omega_exponents[residue]))
        table_values, table_exponents = _scaled_arrays_from_constant(_ONE_SCALED, (batch_size, 4))
        table_values[:, 3] = phase_value[0]
        table_exponents[:, 3] = phase_value[1]
        factor_values, factor_exponents = _combine_factor_scaled_batch(
            factors,
            (left, right),
            table_values,
            table_exponents,
            batch_size=batch_size,
        )
        scalar_values, scalar_exponents = _mul_scaled_complex_arrays(
            scalar_values,
            scalar_exponents,
            factor_values,
            factor_exponents,
        )

    return (scalar_values, scalar_exponents), factors


def _evaluate_half_phase_mediator_plan_scaled_batch(
    mediator_plan: _HalfPhaseMediatorPlan,
    q1_local_batch: np.ndarray,
) -> list[ScaledComplex]:
    """Batch exact evaluation of one half-phase mediator plan."""
    batch = np.ascontiguousarray(np.asarray(q1_local_batch, dtype=np.int64))
    if len(batch) == 0:
        return []

    max_index = max(
        max(mediator_plan.core_vars, default=-1),
        max((spec.mediator_var for spec in mediator_plan.mediators), default=-1),
    )
    if batch.shape[1] <= max_index:
        raise ValueError(
            f"Expected q1_local_batch to cover mediator-plan indices through {max_index}, "
            f"received width {batch.shape[1]}."
        )

    (scalar_values, scalar_exponents), factors = _build_core_factor_batch(
        level=mediator_plan.level,
        core_vars=mediator_plan.core_vars,
        core_q2=mediator_plan.core_q2,
        q1_local_batch=batch,
    )
    omega_values, omega_exponents = _omega_scaled_arrays(mediator_plan.level)
    mod_q1 = 1 << mediator_plan.level
    one_values, one_exponents = _scaled_arrays_from_constant(_ONE_SCALED, (len(batch),))

    for spec in mediator_plan.mediators:
        residues = np.remainder(batch[:, spec.mediator_var], mod_q1)
        phase_values = omega_values[residues]
        phase_exponents = omega_exponents[residues]
        even_values, even_exponents = _add_scaled_complex_arrays(
            one_values,
            one_exponents,
            phase_values,
            phase_exponents,
        )
        odd_values, odd_exponents = _add_scaled_complex_arrays(
            one_values,
            one_exponents,
            -phase_values,
            phase_exponents,
        )
        if len(spec.neighbor_vars) == 0:
            scalar_values, scalar_exponents = _mul_scaled_complex_arrays(
                scalar_values,
                scalar_exponents,
                even_values,
                even_exponents,
            )
            continue
        if len(spec.neighbor_vars) == 1:
            table_values = np.stack((even_values, odd_values), axis=1)
            table_exponents = np.stack((even_exponents, odd_exponents), axis=1)
        else:
            table_values = np.stack((even_values, odd_values, odd_values, even_values), axis=1)
            table_exponents = np.stack((even_exponents, odd_exponents, odd_exponents, even_exponents), axis=1)
        factor_values, factor_exponents = _combine_factor_scaled_batch(
            factors,
            spec.neighbor_vars,
            table_values,
            table_exponents,
            batch_size=len(batch),
        )
        scalar_values, scalar_exponents = _mul_scaled_complex_arrays(
            scalar_values,
            scalar_exponents,
            factor_values,
            factor_exponents,
        )

    totals, _ = _sum_factor_tables_scaled_batch(
        len(mediator_plan.core_vars),
        factors,
        mediator_plan.order,
        scalar=(scalar_values, scalar_exponents),
    )
    return totals


def _evaluate_generic_q2_mediator_plan_scaled_batch(
    mediator_plan: _GenericQ2MediatorPlan,
    q1_local_batch: np.ndarray,
) -> list[ScaledComplex]:
    """Batch exact evaluation of one arbitrary-q2 mediator plan."""
    batch = np.ascontiguousarray(np.asarray(q1_local_batch, dtype=np.int64))
    if len(batch) == 0:
        return []

    max_index = max(
        max(mediator_plan.core_vars, default=-1),
        max((spec.mediator_var for spec in mediator_plan.mediators), default=-1),
    )
    if batch.shape[1] <= max_index:
        raise ValueError(
            f"Expected q1_local_batch to cover generic-mediator indices through {max_index}, "
            f"received width {batch.shape[1]}."
        )

    (scalar_values, scalar_exponents), factors = _build_core_factor_batch(
        level=mediator_plan.level,
        core_vars=mediator_plan.core_vars,
        core_q2=mediator_plan.core_q2,
        q1_local_batch=batch,
    )
    omega_values, omega_exponents = _omega_scaled_arrays(mediator_plan.level)
    mod_q1 = 1 << mediator_plan.level
    one_values, one_exponents = _scaled_arrays_from_constant(_ONE_SCALED, (len(batch),))

    for spec in mediator_plan.mediators:
        assignment_count = 1 << len(spec.neighbor_vars)
        table_values = np.empty((len(batch), assignment_count), dtype=np.complex128)
        table_exponents = np.empty((len(batch), assignment_count), dtype=np.int64)
        base_residues = np.remainder(batch[:, spec.mediator_var], mod_q1)
        for assignment in range(assignment_count):
            shift = spec.assignment_residue_shifts[assignment] if spec.assignment_residue_shifts else 0
            residues = (base_residues + int(shift)) % mod_q1
            one_plus_values, one_plus_exponents = _add_scaled_complex_arrays(
                one_values,
                one_exponents,
                omega_values[residues],
                omega_exponents[residues],
            )
            table_values[:, assignment] = one_plus_values
            table_exponents[:, assignment] = one_plus_exponents
        factor_values, factor_exponents = _combine_factor_scaled_batch(
            factors,
            spec.neighbor_vars,
            table_values,
            table_exponents,
            batch_size=len(batch),
        )
        scalar_values, scalar_exponents = _mul_scaled_complex_arrays(
            scalar_values,
            scalar_exponents,
            factor_values,
            factor_exponents,
        )

    totals, _ = _sum_factor_tables_scaled_batch(
        len(mediator_plan.core_vars),
        factors,
        mediator_plan.order,
        scalar=(scalar_values, scalar_exponents),
    )
    return totals


def _evaluate_half_phase_cluster_plan_scaled_batch(
    cluster_plan: _HalfPhaseClusterPlan,
    q1_local_batch: np.ndarray,
) -> list[ScaledComplex]:
    """Batch exact evaluation of one half-phase cluster plan."""
    batch = np.ascontiguousarray(np.asarray(q1_local_batch, dtype=np.int64))
    if len(batch) == 0:
        return []

    max_index = max(
        max(cluster_plan.core_vars, default=-1),
        max((var for spec in cluster_plan.clusters for var in spec.cluster_vars), default=-1),
    )
    if batch.shape[1] <= max_index:
        raise ValueError(
            f"Expected q1_local_batch to cover cluster-plan indices through {max_index}, "
            f"received width {batch.shape[1]}."
        )

    (scalar_values, scalar_exponents), factors = _build_core_factor_batch(
        level=cluster_plan.level,
        core_vars=cluster_plan.core_vars,
        core_q2=cluster_plan.core_q2,
        q1_local_batch=batch,
    )
    mod_q1 = 1 << cluster_plan.level

    for spec in cluster_plan.clusters:
        cluster_vars = np.asarray(spec.cluster_vars, dtype=np.int64)
        cluster_q1_batch = np.remainder(batch[:, cluster_vars], mod_q1)
        boundary_count = 1 << len(spec.boundary_vars)
        expanded_batch = np.broadcast_to(
            cluster_q1_batch[:, None, :],
            (len(batch), boundary_count, len(spec.cluster_vars)),
        ).copy()
        if spec.boundary_shift_table is not None and spec.boundary_shift_table.size:
            expanded_batch = (expanded_batch + spec.boundary_shift_table[None, :, :]) % mod_q1
        folded_batch, folded_inverse = _fold_phase_shifted_q1_batch(
            expanded_batch.reshape(len(batch) * boundary_count, len(spec.cluster_vars))
        )
        folded_totals = _sum_q3_free_treewidth_dp_scaled_batch(
            n_vars=len(spec.cluster_vars),
            level=cluster_plan.level,
            q1_batch=folded_batch,
            q2=spec.internal_q2,
            order=spec.cluster_order,
            native_plan=spec.native_treewidth_plan,
        )
        table_values, table_exponents = _scaled_list_to_arrays(
            [folded_totals[idx] for idx in folded_inverse],
            (len(batch), boundary_count),
        )
        factor_values, factor_exponents = _combine_factor_scaled_batch(
            factors,
            spec.boundary_vars,
            table_values,
            table_exponents,
            batch_size=len(batch),
        )
        scalar_values, scalar_exponents = _mul_scaled_complex_arrays(
            scalar_values,
            scalar_exponents,
            factor_values,
            factor_exponents,
        )

    totals, _ = _sum_factor_tables_scaled_batch(
        len(cluster_plan.core_vars),
        factors,
        cluster_plan.order,
        scalar=(scalar_values, scalar_exponents),
    )
    return totals


def _add_scaled_complex(left, right):
    left_value, left_half_pow2_exp = left
    right_value, right_half_pow2_exp = right

    if left_value == 0j:
        return right
    if right_value == 0j:
        return left
    if left_half_pow2_exp < right_half_pow2_exp:
        left_value, right_value = right_value, left_value
        left_half_pow2_exp, right_half_pow2_exp = right_half_pow2_exp, left_half_pow2_exp

    aligned_right = _scale_complex_by_half_pow2(
        right_value,
        right_half_pow2_exp - left_half_pow2_exp,
    )
    return _renormalize_scaled_complex_if_needed(left_value + aligned_right, left_half_pow2_exp)


def _scaled_to_complex(scaled, extra_scalar=1.0 + 0j, extra_half_pow2=0):
    value, half_pow2_exp = scaled
    return _scale_complex_by_half_pow2(
        complex(extra_scalar) * value,
        half_pow2_exp + extra_half_pow2,
    )


def _scale_complex_by_half_pow2(value, half_pow2_exp):
    """Scale a complex value by 2 ** (half_pow2_exp / 2) without huge floats."""
    if value == 0j or half_pow2_exp == 0:
        return complex(value)

    scaled = complex(value)
    if half_pow2_exp > 0 and half_pow2_exp % 2:
        scaled *= _SQRT2
        half_pow2_exp -= 1
    elif half_pow2_exp < 0 and half_pow2_exp % 2:
        scaled *= _INV_SQRT2
        half_pow2_exp += 1

    shift = half_pow2_exp // 2
    return complex(
        math.ldexp(scaled.real, shift),
        math.ldexp(scaled.imag, shift),
    )


_ONE_SCALED = _make_scaled_complex(1.0 + 0j)
_ZERO_SCALED = _make_scaled_complex(0j)

_CLASS_CUBIC = 0
_CLASS_QUADRATIC = 1
_CLASS_CONSTRAINT_DECOUPLED = 2
_CLASS_CONSTRAINT_ZERO = 3
_CLASS_CONSTRAINT_PARITY = 4
_BUILD_EARLY_ELIM_BATCH = 16
_BUILD_EARLY_ELIM_BATCH_HIGH_PRECISION = 256
_LEVEL3_BUILD_ELIM_DEFER_MIN_DEGREE = 5
_STRUCTURE_CLASSIFICATION_CACHE_MAX = 1 << 12
_STRUCTURE_PHASE3_CACHE_MAX = 1 << 11

_STRUCTURE_CLASSIFICATION_DATA_CACHE = _BoundedMemoCache(_STRUCTURE_CLASSIFICATION_CACHE_MAX)
_STRUCTURE_CLASSIFICATION_LOOKUP_CACHE = _BoundedMemoCache(_STRUCTURE_CLASSIFICATION_CACHE_MAX)
_STRUCTURE_MIN_FILL_CACHE = _BoundedMemoCache(_STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_Q3_COVER_CACHE = _BoundedMemoCache(_STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_BAD_Q2_COVER_CACHE = _BoundedMemoCache(_STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_PLAN_CACHE = _BoundedMemoCache(_STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_TREEWIDTH_FACTOR_CACHE = _BoundedMemoCache(_STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_TREEWIDTH_NATIVE_PLAN_CACHE = _BoundedMemoCache(_STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_LEVEL3_NATIVE_PLAN_CACHE = _BoundedMemoCache(_STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_LEVEL3_NATIVE_PLAN_SEEN_CACHE = _BoundedMemoCache(_STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_FACTOR_CACHE = _BoundedMemoCache(_STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_Q3_SEPARATOR_CACHE = _BoundedMemoCache(_STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_Q3_COVER_TEMPLATE_CACHE = _BoundedMemoCache(_STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_FIX_VARIABLE_TEMPLATE_CACHE = _BoundedMemoCache(_STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_Q3_2CORE_CACHE = _BoundedMemoCache(_STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_Q3_FREE_CUTSET_PLAN_CACHE = _BoundedMemoCache(_STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_Q3_FREE_TENSOR_HINT_CACHE = _BoundedMemoCache(_STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_Q3_FREE_EXECUTION_PLAN_CACHE = _BoundedMemoCache(_STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_Q3_FREE_REFINED_ORDER_CACHE = _BoundedMemoCache(_STRUCTURE_PHASE3_CACHE_MAX)


def _build_early_elim_batch_size(level: int) -> int:
    return _BUILD_EARLY_ELIM_BATCH if int(level) <= 3 else _BUILD_EARLY_ELIM_BATCH_HIGH_PRECISION


def _project_quadratic_elimination_q2_nnz_delta(
    q,
    k: int,
    *,
    classification_data=None,
) -> tuple[int, int]:
    """
    Return ``(degree, q2_nnz_delta)`` for eliminating quadratic variable ``k``.

    The delta counts how many q2 nonzeros would be added or removed relative to
    the current q2 graph. Negative values mean the elimination sparsifies q2;
    positive values mean it densifies q2. This is purely a local structural
    projection and does not mutate ``q``.
    """
    if int(q.level) == 3:
        if classification_data is None:
            classification_data = _build_classification_data(q)
        _cubic_incidence, odd_bilinear, parity_partners = classification_data
        if odd_bilinear[k]:
            return 0, 0

        neighbors = tuple(int(var) for var in parity_partners[k])
        degree = len(neighbors)
        if degree <= 1:
            return degree, -degree

        missing_edges = 0
        parity_edges = 0
        for left_pos in range(degree):
            left = neighbors[left_pos]
            for right_pos in range(left_pos + 1, degree):
                right = neighbors[right_pos]
                edge_key = (left, right) if left < right else (right, left)
                old_value = q.q2.get(edge_key, 0) % q.mod_q2
                if old_value == 0:
                    missing_edges += 1
                elif old_value == (q.mod_q2 // 2):
                    parity_edges += 1
        return degree, missing_edges - parity_edges - degree

    neighbors: list[int] = []
    couplings: list[int] = []
    for j in range(q.n):
        if j == k:
            continue
        value = q.q2.get((min(k, j), max(k, j)), 0) % q.mod_q2
        if value:
            neighbors.append(j)
            couplings.append(value)

    degree = len(neighbors)
    if degree <= 1:
        return degree, -degree

    old_nonzero = 0
    new_nonzero = 0
    for left_pos in range(degree):
        left = neighbors[left_pos]
        left_coupling = couplings[left_pos]
        for right_pos in range(left_pos + 1, degree):
            right = neighbors[right_pos]
            right_coupling = couplings[right_pos]
            edge_key = (left, right) if left < right else (right, left)
            old_value = q.q2.get(edge_key, 0) % q.mod_q2
            if old_value:
                old_nonzero += 1
            correction = (left_coupling * right_coupling // 2) % q.mod_q2
            new_value = (old_value + correction) % q.mod_q2
            if new_value:
                new_nonzero += 1

    return degree, new_nonzero - old_nonzero - degree


def _should_defer_build_quadratic_elimination(
    q,
    k: int,
    *,
    classification_data=None,
) -> bool:
    """
    Keep build-time level-3 elimination from creating dense q2 fill-in.

    This is intentionally conservative for correctness: it only skips an exact
    build-time elimination and leaves the unreduced state to the later exact
    solver. It never enables new eliminations or changes the higher-precision
    cutoff.
    """
    if int(q.level) != 3:
        return False
    degree, q2_nnz_delta = _project_quadratic_elimination_q2_nnz_delta(
        q,
        k,
        classification_data=classification_data,
    )
    return degree >= _LEVEL3_BUILD_ELIM_DEFER_MIN_DEGREE and q2_nnz_delta > 0


@lru_cache(maxsize=64)
def _fraction_from_residue(level: int, residue: int) -> Fraction:
    modulus = 1 << level
    return Fraction(residue % modulus, modulus)


_PACK_QKEY_HEADER = struct.Struct("<iiqq")
_PACK_QKEY_Q1 = struct.Struct("<q")
_PACK_QKEY_Q2 = struct.Struct("<iiq")
_PACK_QKEY_Q3 = struct.Struct("<iiiq")
_PACK_QSTRUCT_HEADER = struct.Struct("<ii")


def _q_key_digest(q) -> bytes:
    hasher = hashlib.blake2b(digest_size=20)
    hasher.update(_PACK_QKEY_HEADER.pack(q.n, q.level, q.q0.numerator, q.q0.denominator))
    for coeff in q.q1:
        hasher.update(_PACK_QKEY_Q1.pack(coeff))
    for (i, j), coeff in q.q2.items():
        hasher.update(_PACK_QKEY_Q2.pack(i, j, coeff))
    for (i, j, k), coeff in q.q3.items():
        hasher.update(_PACK_QKEY_Q3.pack(i, j, k, coeff))
    return hasher.digest()


def _q_key(q):
    cached = getattr(q, "_schur_q_key", None)
    if cached is not None:
        return cached
    key = (q.n, q.level, _q_key_digest(q))
    # Reduction intermediates are treated as immutable once they enter the
    # cache pipeline. Keep only a compact digest on the object rather than
    # materializing full q1/q2/q3 tuples for every intermediate.
    q._schur_q_key = key
    return key


def _cache_phase_structure_key(q, attr_name: str, key):
    if not getattr(q, "_schur_mutable", True):
        setattr(q, attr_name, key)
    return key


def _q_structure_key(q):
    cached = getattr(q, "_schur_q_structure_key", None)
    if cached is not None:
        return cached
    hasher = hashlib.blake2b(digest_size=20)
    hasher.update(_PACK_QSTRUCT_HEADER.pack(q.n, q.level))
    for (i, j), coeff in q.q2.items():
        hasher.update(_PACK_QKEY_Q2.pack(i, j, coeff))
    for (i, j, k), coeff in q.q3.items():
        hasher.update(_PACK_QKEY_Q3.pack(i, j, k, coeff))
    return _cache_phase_structure_key(q, "_schur_q_structure_key", (q.n, q.level, hasher.digest()))


def _q_phase3_structure_key(q):
    cached = getattr(q, "_schur_q_phase3_structure_key", None)
    if cached is not None:
        return cached

    # Phase-3 planning depends on which factor scopes survive, not on their
    # exact coefficients. Reuse one planner result across coefficient-only
    # variations produced by weak-sampling affine restrictions.
    hasher = hashlib.blake2b(digest_size=20)
    hasher.update(_PACK_QSTRUCT_HEADER.pack(q.n, q.level))
    for idx, coeff in enumerate(q.q1):
        if coeff % q.mod_q1:
            hasher.update(_PACK_QKEY_Q1.pack(idx))
    for (i, j), coeff in q.q2.items():
        if coeff % q.mod_q2:
            hasher.update(_PACK_QKEY_Q2.pack(i, j, 1))
    for (i, j, k), coeff in q.q3.items():
        if coeff % q.mod_q3:
            hasher.update(_PACK_QKEY_Q3.pack(i, j, k, 1))
    return _cache_phase_structure_key(
        q,
        "_schur_q_phase3_structure_key",
        (q.n, q.level, hasher.digest()),
    )


def _q_classification_structure_key(q):
    cached = getattr(q, "_schur_q_classification_structure_key", None)
    if cached is not None:
        return cached

    if _schur_native is not None and q.level == 3:
        return _cache_phase_structure_key(
            q,
            "_schur_q_classification_structure_key",
            _schur_native.classification_structure_key(q.n, q.level, q.q2, q.q3),
        )

    hasher = hashlib.blake2b(digest_size=20)
    hasher.update(_PACK_QSTRUCT_HEADER.pack(q.n, q.level))
    parity_residue = q.mod_q2 // 2 if q.mod_q2 > 1 else 0
    for (i, j), coeff in q.q2.items():
        residue = coeff % q.mod_q2
        flags = 0
        if residue % 2:
            flags |= 1
        if parity_residue and residue == parity_residue:
            flags |= 2
        if flags:
            hasher.update(_PACK_QKEY_Q2.pack(i, j, flags))
    for (i, j, k), coeff in q.q3.items():
        if coeff % q.mod_q3:
            hasher.update(_PACK_QKEY_Q3.pack(i, j, k, 1))
    return _cache_phase_structure_key(
        q,
        "_schur_q_classification_structure_key",
        (q.n, q.level, hasher.digest()),
    )


def _q_q3_support_key(q):
    cached = getattr(q, "_schur_q3_support_key", None)
    if cached is not None:
        return cached

    hasher = hashlib.blake2b(digest_size=20)
    hasher.update(_PACK_QSTRUCT_HEADER.pack(q.n, q.level))
    for (i, j, k), coeff in q.q3.items():
        if coeff % q.mod_q3:
            hasher.update(_PACK_QKEY_Q3.pack(i, j, k, 1))
    return _cache_phase_structure_key(q, "_schur_q3_support_key", (q.n, q.level, hasher.digest()))

def _phase_function_from_parts(n, *, level, q0, q1, q2, q3):
    phase = PhaseFunction.__new__(PhaseFunction)
    phase.n = n
    phase.level = level
    phase.mod_q1 = 1 << phase.level
    phase.mod_q2 = max(1, 1 << (phase.level - 1))
    phase.mod_q3 = max(1, 1 << (phase.level - 2))
    if not isinstance(q0, Fraction):
        q0 = Fraction(q0)
    phase.q0 = q0
    phase.q1 = q1
    phase.q2 = q2
    phase.q3 = q3
    phase._schur_mutable = False
    return phase


def _copy_cubic_function(q):
    phase = _phase_function_from_parts(
        q.n,
        level=q.level,
        q0=q.q0,
        q1=list(q.q1),
        q2=dict(q.q2),
        q3=dict(q.q3),
    )
    phase._schur_mutable = True
    return phase


def _evaluate_q_from_mask(q, mask):
    if _native_level3_enabled(q):
        residue = _schur_native.evaluate_q_mask_terms(q.q1, q.q2, q.q3, mask)
        return (q.q0 + Fraction(residue, q.mod_q1)) % 1

    value = q.q0
    for idx, coeff in enumerate(q.q1):
        if coeff and _mask_bit(mask, idx):
            value += Fraction(coeff, q.mod_q1)
    for (i, j), coeff in q.q2.items():
        if _mask_bit(mask, i) and _mask_bit(mask, j):
            value += Fraction(coeff, q.mod_q2)
    for (i, j, k), coeff in q.q3.items():
        if _mask_bit(mask, i) and _mask_bit(mask, j) and _mask_bit(mask, k):
            value += Fraction(coeff, q.mod_q3)
    return value % 1


def _row_masks_from_gamma(gamma):
    if gamma and isinstance(gamma[0], int):
        return tuple(gamma)
    row_masks = []
    for row in gamma:
        mask = 0
        for idx, bit in enumerate(row):
            if bit:
                mask |= 1 << idx
        row_masks.append(mask)
    return tuple(row_masks)


def _can_use_native_output_solver(cache: EchelonCache) -> bool:
    return _schur_native is not None and cache.n <= 64 and cache.m <= 64


def _native_solve_for_output(
    eps0: Sequence[int],
    cache: EchelonCache,
    output_bits: BitSequence,
) -> int | None:
    if not _can_use_native_output_solver(cache):
        return None
    return _schur_native.solve_output_shift_mask_u64(
        tuple(int(bit) & 1 for bit in eps0),
        cache.pivot_col,
        cache.row_ops,
        tuple(int(bit) & 1 for bit in output_bits),
        cache.m,
    )


def _native_solve_for_output_batch(
    eps0: Sequence[int],
    cache: EchelonCache,
    output_list: Sequence[BitSequence],
) -> tuple[int | None, ...] | None:
    if not _can_use_native_output_solver(cache) or not output_list:
        return None
    return _schur_native.solve_output_shift_masks_u64(
        tuple(int(bit) & 1 for bit in eps0),
        cache.pivot_col,
        cache.row_ops,
        [tuple(int(bit) & 1 for bit in output_bits) for output_bits in output_list],
        cache.m,
    )


def _aff_compose_cached(q, shift, gamma, k, context=None):
    if context is None:
        return _aff_compose(q, shift, gamma, k)

    shift_mask = shift if isinstance(shift, int) else _mask_from_vector(shift)
    key = (
        _q_key(q),
        shift_mask,
        _row_masks_from_gamma(gamma),
        k,
    )
    cached = context.affine_compose_cache.get(key)
    if cached is not None:
        return cached

    composed = _aff_compose(q, shift, gamma, k)
    context.affine_compose_cache[key] = composed
    return composed


def _q3_free_constraint_plan_key(
    state: SchurState,
    cache: EchelonCache,
    *,
    allow_tensor_contraction: bool = True,
    prefer_reusable_decomposition: bool = False,
) -> tuple[Any, ...]:
    return (
        _q_key(state.q),
        tuple(state.eps),
        tuple(state.eps0),
        cache.echelon_rows,
        cache.pivot_col,
        cache.row_ops,
        bool(allow_tensor_contraction),
        bool(prefer_reusable_decomposition),
    )


def _q3_free_edge_density(q: PhaseFunction) -> float:
    if q.n <= 1:
        return 0.0
    return (2.0 * len(q.q2)) / (q.n * (q.n - 1))


def _q3_free_prefers_locality_preserving_cutset(
    q: PhaseFunction,
    *,
    feedback_size: int,
    max_degree: int,
    edge_density: float,
    allow_tensor_contraction: bool,
) -> bool:
    """Return whether dense q3-free routing should stay on TN-guided exact backends."""
    if not allow_tensor_contraction or q.q3 or not q.q2:
        return False

    factor_density = len(_build_factor_scopes(q)) / max(1, q.n)
    if factor_density < _Q3_TENSOR_CONTRACTION_MIN_FACTOR_DENSITY:
        return False

    if q.n >= _get_solver_config().tensor_hint_min_vars:
        return bool(_q3_free_tensor_slice_hint(q))

    if q.n <= _Q3_TENSOR_CONTRACTION_MAX_VARS or q.n > _Q3_HYBRID_CONTRACTION_MAX_VARS:
        return False
    if (
        max_degree < _Q3_FREE_DENSE_PLAN_MIN_DEGREE
        and edge_density < _Q3_FREE_DENSE_PLAN_MIN_DENSITY
    ):
        return False
    return feedback_size > _SCHUR_COMPLEMENT_CROSSOVER_FVS or max_degree >= _Q3_FREE_DENSE_PLAN_MIN_DEGREE


def _q3_free_prefers_reusable_cutset(
    q: PhaseFunction,
    *,
    treewidth_order: Sequence[int] | None,
    cutset_plan: _Q3FreeCutsetConditioningPlan | None,
    lambda_count: int,
) -> bool:
    """Return whether reusable q3-free workloads should prefer a cutset plan."""
    if (
        treewidth_order is None
        or cutset_plan is None
        or q.q3
        or not q.q2
        or lambda_count < _Q3_FREE_REUSABLE_CUTSET_MIN_LAMBDA_VARS
    ):
        return False

    direct_width = _treewidth_order_width(q, treewidth_order)
    if direct_width < _Q3_FREE_REUSABLE_CUTSET_MIN_TREEWIDTH:
        return False

    direct_work = max(1, _estimate_treewidth_dp_work(q, treewidth_order))
    cutset_work = max(1, cutset_plan.estimated_total_work)
    reuse_multiplier = 1 << min(_Q3_FREE_REUSABLE_CUTSET_MAX_LOG2_REUSE, lambda_count)
    width_gain = direct_width - cutset_plan.remaining_width
    if width_gain >= 2 and cutset_work <= direct_work * reuse_multiplier:
        return True
    if width_gain >= 1 and cutset_work * 2 <= direct_work * reuse_multiplier:
        return True
    return False


def _q3_free_prefers_one_shot_cutset(
    q: PhaseFunction,
    *,
    treewidth_order: Sequence[int] | None,
    cutset_plan: _Q3FreeCutsetConditioningPlan | None,
    allow_tensor_contraction: bool,
) -> bool:
    """Return whether one-shot exact amplitudes should switch to cutset slicing."""
    if treewidth_order is None or cutset_plan is None or q.q3 or not q.q2:
        return False

    direct_width = _treewidth_order_width(q, treewidth_order)
    if direct_width < _Q3_FREE_ONE_SHOT_CUTSET_MIN_TREEWIDTH:
        return False

    if cutset_plan.remaining_width > _Q3_FREE_CUTSET_TENSOR_HINT_TARGET_WIDTH:
        return False
    if _q3_free_cutset_plan_generic_penalty(cutset_plan) > 0:
        return False

    width_gain = direct_width - cutset_plan.remaining_width
    if width_gain < 2:
        return False

    direct_work = max(1, _estimate_treewidth_dp_work(q, treewidth_order))
    cutset_work = max(1, cutset_plan.estimated_total_work)
    if cutset_work <= direct_work:
        return True
    if (
        allow_tensor_contraction
        and q.n >= _Q3_FREE_CUTSET_TENSOR_HINT_MIN_VARS
        and _q3_free_tensor_slice_hint(q)
        and cutset_work * 2 <= direct_work
    ):
        return True
    return False


def _plan_q3_free_constraint_components(
    base_q: PhaseFunction,
    lambda_offset: int,
    *,
    order_hint: Sequence[int] | None = None,
    allow_tensor_contraction: bool = True,
    prefer_reusable_decomposition: bool = False,
    prefer_one_shot_slicing: bool = False,
) -> tuple[tuple[int, ...], tuple[_Q3FreeConstraintComponentPlan, ...]]:
    """Plan reusable component backends for an augmented q3-free constraint kernel."""
    component_sets = detect_factorization(base_q)
    covered = set().union(*component_sets) if component_sets else set()
    isolated_vars = tuple(sorted(set(range(base_q.n)) - covered))
    component_plans: list[_Q3FreeConstraintComponentPlan] = []

    hint_positions = None if order_hint is None else {var: idx for idx, var in enumerate(order_hint)}

    for component in component_sets:
        variables = tuple(sorted(component))
        component_q = _component_restriction(base_q, variables)
        lambda_count = sum(1 for var in variables if var >= lambda_offset)
        adjacency, edges = _q3_free_graph(component_q)
        max_degree = max((len(neighbors) for neighbors in adjacency), default=0)
        edge_density = _q3_free_edge_density(component_q)
        local_order_hint = None
        if hint_positions is not None:
            hinted_variables = sorted(
                variables,
                key=lambda var: (hint_positions.get(var, len(hint_positions)), var),
            )
            local_remap = {var: idx for idx, var in enumerate(variables)}
            local_order_hint = [local_remap[var] for var in hinted_variables]
        dense_component = (
            max_degree >= _Q3_FREE_DENSE_PLAN_MIN_DEGREE
            and edge_density >= _Q3_FREE_DENSE_PLAN_MIN_DENSITY
        )
        binary_phase_plan = None
        skip_dense_schur = False
        if _is_half_phase_q2(component_q):
            fixed_nonbinary_support = _component_fixed_nonbinary_unary_support_size(
                component_q,
                variables,
                lambda_offset=lambda_offset,
            )
            if fixed_nonbinary_support <= _Q3_FREE_HALF_PHASE_UNARY_EXPANSION_MAX_SUPPORT:
                binary_phase_plan = _build_binary_phase_quadratic_plan(component_q)

        if lambda_count == 0 and not prefer_one_shot_slicing and not prefer_reusable_decomposition:
            if dense_component:
                mediator_plan = _build_half_phase_mediator_plan(component_q)
                generic_mediator_plan = (
                    _build_generic_q2_mediator_plan(component_q)
                    if mediator_plan is None
                    else None
                )
                cluster_plan = _build_q1_cluster_plan(component_q)
                dense_schur_ok = _supports_exact_dense_schur(component_q)
                component_plans.append(
                    _Q3FreeConstraintComponentPlan(
                        variables=variables,
                        level=component_q.level,
                        q2=component_q.q2,
                        backend="generic",
                        dense_q2=_dense_q2_matrix(component_q),
                        binary_phase_plan=binary_phase_plan,
                        mediator_plan=mediator_plan,
                        generic_mediator_plan=generic_mediator_plan,
                        cluster_plan=cluster_plan,
                        skip_dense_schur=(
                            skip_dense_schur
                            or not dense_schur_ok
                        ),
                        direct_schur_ok=(
                            binary_phase_plan is None
                            and mediator_plan is None
                            and generic_mediator_plan is None
                            and cluster_plan is None
                            and dense_schur_ok
                        ),
                        quadratic_tensor_q2=_is_half_phase_q2(component_q),
                        lambda_offset=lambda_offset,
                    )
                )
                continue
            component_plans.append(
                _Q3FreeConstraintComponentPlan(
                    variables=variables,
                    level=component_q.level,
                    q2=component_q.q2,
                    backend="constant",
                    precomputed_total=_sum_q3_free_component_scaled(
                        component_q,
                        allow_tensor_contraction=allow_tensor_contraction,
                    ),
                    mediator_plan=_build_half_phase_mediator_plan(component_q),
                    quadratic_tensor_q2=_is_half_phase_q2(component_q),
                    lambda_offset=lambda_offset,
                )
            )
            continue

        # Dense conditioned kernels are a poor match for the sparse spanning /
        # treewidth planner. Route them straight to the dense generic backend,
        # which can evaluate via schur complement without paying that planning
        # cost up front.
        if (
            max_degree >= _Q3_FREE_DENSE_PLAN_MIN_DEGREE
            and edge_density >= _Q3_FREE_DENSE_PLAN_MIN_DENSITY
        ):
            mediator_plan = _build_half_phase_mediator_plan(component_q)
            generic_mediator_plan = (
                _build_generic_q2_mediator_plan(component_q)
                if mediator_plan is None
                else None
            )
            cluster_plan = _build_q1_cluster_plan(component_q)
            dense_schur_ok = _supports_exact_dense_schur(component_q)
            direct_schur_ok = (
                binary_phase_plan is None
                and mediator_plan is None
                and generic_mediator_plan is None
                and cluster_plan is None
                and dense_schur_ok
            )
            depth, chords = _q3_free_spanning_data(adjacency, edges)
            feedback_vars = _select_feedback_vertices(component_q.n, chords, depth)
            treewidth_order = (
                _q3_free_treewidth_order(
                    component_q,
                    len(feedback_vars),
                    order_hint=local_order_hint,
                    max_degree=max_degree,
                )
                if prefer_reusable_decomposition or prefer_one_shot_slicing
                else None
            )
            prefer_cutset = _q3_free_prefers_locality_preserving_cutset(
                component_q,
                feedback_size=len(feedback_vars),
                max_degree=max_degree,
                edge_density=edge_density,
                allow_tensor_contraction=allow_tensor_contraction,
            )
            cutset_plan = (
                (
                    _q3_free_one_shot_cutset_conditioning_plan(component_q)
                    if prefer_one_shot_slicing
                    else _q3_free_cutset_conditioning_plan(component_q)
                )
                if prefer_cutset or prefer_reusable_decomposition or prefer_one_shot_slicing
                else None
            )
            prefer_reusable_cutset = (
                prefer_reusable_decomposition
                and _q3_free_prefers_reusable_cutset(
                    component_q,
                    treewidth_order=treewidth_order,
                    cutset_plan=cutset_plan,
                    lambda_count=lambda_count,
                )
            )
            prefer_one_shot_cutset = (
                prefer_one_shot_slicing
                and _q3_free_prefers_one_shot_cutset(
                    component_q,
                    treewidth_order=treewidth_order,
                    cutset_plan=cutset_plan,
                    allow_tensor_contraction=allow_tensor_contraction,
                )
            )
            prefer_cutset_backend = (
                cutset_plan is not None
                and (
                    prefer_cutset
                    or prefer_reusable_cutset
                    or prefer_one_shot_cutset
                    or prefer_one_shot_slicing
                )
            )
            if mediator_plan is not None or generic_mediator_plan is not None:
                component_plans.append(
                    _Q3FreeConstraintComponentPlan(
                        variables=variables,
                        level=component_q.level,
                        q2=component_q.q2,
                        backend="generic",
                        dense_q2=_dense_q2_matrix(component_q),
                        binary_phase_plan=binary_phase_plan,
                        mediator_plan=mediator_plan,
                        generic_mediator_plan=generic_mediator_plan,
                        cluster_plan=cluster_plan,
                        cutset_plan=cutset_plan,
                        skip_dense_schur=(
                            skip_dense_schur
                            or not dense_schur_ok
                            or prefer_cutset_backend
                        ),
                        direct_schur_ok=direct_schur_ok and not prefer_cutset_backend,
                        quadratic_tensor_q2=_is_half_phase_q2(component_q),
                        lambda_offset=lambda_offset,
                        prefer_reusable_decomposition=prefer_reusable_decomposition,
                        prefer_cutset_backend=prefer_cutset_backend,
                    )
                )
                continue
            component_plans.append(
                _Q3FreeConstraintComponentPlan(
                    variables=variables,
                    level=component_q.level,
                    q2=component_q.q2,
                    backend="generic",
                    dense_q2=_dense_q2_matrix(component_q),
                    binary_phase_plan=binary_phase_plan,
                    mediator_plan=None,
                    generic_mediator_plan=None,
                    cluster_plan=cluster_plan,
                    cutset_plan=cutset_plan,
                    skip_dense_schur=(
                        skip_dense_schur
                        or not dense_schur_ok
                        or prefer_cutset_backend
                    ),
                    direct_schur_ok=direct_schur_ok and not prefer_cutset_backend,
                    quadratic_tensor_q2=_is_half_phase_q2(component_q),
                    lambda_offset=lambda_offset,
                    prefer_reusable_decomposition=prefer_reusable_decomposition,
                    prefer_cutset_backend=prefer_cutset_backend,
                )
            )
            continue

        depth, chords = _q3_free_spanning_data(adjacency, edges)
        if not chords:
            component_plans.append(
                _Q3FreeConstraintComponentPlan(
                    variables=variables,
                    level=component_q.level,
                    q2=component_q.q2,
                    backend="forest",
                    adjacency=tuple(
                        tuple(sorted(neighbors.items()))
                        for neighbors in adjacency
                    ),
                    mediator_plan=_build_half_phase_mediator_plan(component_q),
                    quadratic_tensor_q2=_is_half_phase_q2(component_q),
                    lambda_offset=lambda_offset,
                    prefer_reusable_decomposition=prefer_reusable_decomposition,
                )
            )
            continue

        feedback_vars = _select_feedback_vertices(component_q.n, chords, depth)
        treewidth_order = _q3_free_treewidth_order(
            component_q,
            len(feedback_vars),
            order_hint=local_order_hint,
            max_degree=max_degree,
        )
        if treewidth_order is not None:
            treewidth_order, _direct_width_hint = _finalize_q3_free_treewidth_order(
                component_q,
                treewidth_order,
            )
            direct_width = _treewidth_order_width(component_q, treewidth_order)
            prefer_one_shot_cutset_candidate = (
                prefer_one_shot_slicing
                and direct_width >= _Q3_FREE_ONE_SHOT_CUTSET_ACTIVATION_WIDTH
            )
            cutset_plan = (
                (
                    _q3_free_one_shot_cutset_conditioning_plan(component_q)
                    if prefer_one_shot_cutset_candidate
                    else _q3_free_cutset_conditioning_plan(component_q)
                )
                if prefer_reusable_decomposition or prefer_one_shot_cutset_candidate
                else None
            )
            prefer_reusable_cutset = (
                prefer_reusable_decomposition
                and _q3_free_prefers_reusable_cutset(
                    component_q,
                    treewidth_order=treewidth_order,
                    cutset_plan=cutset_plan,
                    lambda_count=lambda_count,
                )
            )
            prefer_one_shot_cutset = (
                prefer_one_shot_cutset_candidate
                and _q3_free_prefers_one_shot_cutset(
                    component_q,
                    treewidth_order=treewidth_order,
                    cutset_plan=cutset_plan,
                    allow_tensor_contraction=allow_tensor_contraction,
                )
            )
            if (prefer_reusable_cutset or prefer_one_shot_cutset) and cutset_plan is not None:
                component_plans.append(
                    _Q3FreeConstraintComponentPlan(
                        variables=variables,
                        level=component_q.level,
                        q2=component_q.q2,
                        backend="generic",
                        binary_phase_plan=binary_phase_plan,
                        cutset_plan=cutset_plan,
                        skip_dense_schur=True,
                        direct_schur_ok=False,
                        quadratic_tensor_q2=_is_half_phase_q2(component_q),
                        lambda_offset=lambda_offset,
                        prefer_reusable_decomposition=True,
                        prefer_cutset_backend=True,
                    )
                )
                continue
            component_plans.append(
                _Q3FreeConstraintComponentPlan(
                    variables=variables,
                    level=component_q.level,
                    q2=component_q.q2,
                    backend="treewidth",
                    order=tuple(treewidth_order),
                    native_treewidth_plan=_build_native_q3_free_treewidth_plan(
                        n_vars=component_q.n,
                        level=component_q.level,
                        q2=component_q.q2,
                        order=treewidth_order,
                    ),
                    mediator_plan=_build_half_phase_mediator_plan(component_q),
                    quadratic_tensor_q2=_is_half_phase_q2(component_q),
                    lambda_offset=lambda_offset,
                    prefer_reusable_decomposition=prefer_reusable_decomposition,
                )
            )
            continue

        mediator_plan = _build_half_phase_mediator_plan(component_q)
        generic_mediator_plan = (
            _build_generic_q2_mediator_plan(component_q)
            if mediator_plan is None
            else None
        )
        cluster_plan = _build_q1_cluster_plan(component_q)
        dense_schur_ok = _supports_exact_dense_schur(component_q)
        prefer_cutset = _q3_free_prefers_locality_preserving_cutset(
            component_q,
            feedback_size=len(feedback_vars),
            max_degree=max_degree,
            edge_density=edge_density,
            allow_tensor_contraction=allow_tensor_contraction,
        )
        cutset_plan = _q3_free_cutset_conditioning_plan(component_q)
        prefer_cutset_backend = (
            cutset_plan is not None
            and (
                prefer_cutset
                or prefer_one_shot_slicing
                or (
                    prefer_reusable_decomposition
                    and lambda_count >= _Q3_FREE_REUSABLE_CUTSET_MIN_LAMBDA_VARS
                )
            )
        )
        component_plans.append(
            _Q3FreeConstraintComponentPlan(
                variables=variables,
                level=component_q.level,
                q2=component_q.q2,
                backend="generic",
                dense_q2=_dense_q2_matrix(component_q),
                binary_phase_plan=binary_phase_plan,
                mediator_plan=mediator_plan,
                generic_mediator_plan=generic_mediator_plan,
                cluster_plan=cluster_plan,
                cutset_plan=cutset_plan,
                skip_dense_schur=(
                    skip_dense_schur
                    or not dense_schur_ok
                    or prefer_cutset_backend
                ),
                direct_schur_ok=(
                    not prefer_cutset_backend
                    and len(feedback_vars) > _SCHUR_COMPLEMENT_CROSSOVER_FVS
                    and binary_phase_plan is None
                    and mediator_plan is None
                    and generic_mediator_plan is None
                    and cluster_plan is None
                    and dense_schur_ok
                ),
                quadratic_tensor_q2=_is_half_phase_q2(component_q),
                lambda_offset=lambda_offset,
                prefer_reusable_decomposition=prefer_reusable_decomposition,
                prefer_cutset_backend=prefer_cutset_backend,
            )
        )

    return isolated_vars, tuple(component_plans)


def _build_q3_free_constraint_plan(
    state: SchurState,
    cache: EchelonCache,
    order_hint: Sequence[int] | None = None,
    *,
    allow_tensor_contraction: bool = True,
    prefer_reusable_decomposition: bool = False,
) -> _Q3FreeConstraintPlan:
    """Build a reusable exact constraint-sum plan for a q3-free Schur state.

    Above dyadic level 3, affine output restriction can require quartic or
    higher ANF terms. Instead of substituting output parities into the phase,
    introduce one dual variable per independent output constraint and enforce
    the affine system through a character sum. The augmented phase remains
    q3-free, so it can be evaluated exactly by the existing q3-free reducer
    without any unsafe degree truncation. The same plan is also useful at level
    3 because it avoids repeating the generic exact-elimination pipeline for
    every closely related marginal query.
    """
    assert not state.q.q3, "q3-free constraint plans require a q3-free kernel."

    lambda_offset = state.q.n
    row_indices = [row_idx for row_idx, pivot in enumerate(cache.pivot_col) if pivot >= 0]
    rank = len(row_indices)
    augmented_q2 = dict(state.q.q2)
    bilinear_half_phase = state.q.mod_q2 // 2

    for lambda_idx, row_idx in enumerate(row_indices):
        dual_var = lambda_offset + lambda_idx
        for var in _iter_mask_bits(cache.echelon_rows[row_idx]):
            key = (var, dual_var) if var < dual_var else (dual_var, var)
            value = (augmented_q2.get(key, 0) + bilinear_half_phase) % state.q.mod_q2
            if value:
                augmented_q2[key] = value
            elif key in augmented_q2:
                del augmented_q2[key]

    base_q = _phase_function_from_parts(
        state.q.n + rank,
        level=state.q.level,
        q0=state.q.q0,
        q1=list(state.q.q1) + ([0] * rank),
        q2=augmented_q2,
        q3={},
    )
    isolated_vars, component_plans = _plan_q3_free_constraint_components(
        base_q,
        lambda_offset,
        order_hint=order_hint,
        allow_tensor_contraction=allow_tensor_contraction,
        prefer_reusable_decomposition=prefer_reusable_decomposition,
    )

    return _Q3FreeConstraintPlan(
        cache=cache,
        eps0=tuple(state.eps0),
        level=state.q.level,
        q0=state.q.q0,
        base_q1=tuple(state.q.q1) + ((0,) * rank),
        base_q2=dict(augmented_q2),
        lambda_offset=lambda_offset,
        rank=rank,
        n_free_after_constraints=cache.n_free,
        rhs_linear_coeff=state.q.mod_q1 // 2,
        isolated_vars=isolated_vars,
        components=tuple(component_plans),
    )


def _q3_free_constraint_rhs(plan: _Q3FreeConstraintPlan, output_bits: BitSequence) -> tuple[int, ...] | None:
    if len(output_bits) != plan.cache.n:
        raise ValueError(f"Expected {plan.cache.n} output bits, received {len(output_bits)}.")

    target_mask = 0
    for idx, bit in enumerate(output_bits):
        if (int(bit) ^ plan.eps0[idx]) & 1:
            target_mask |= 1 << idx

    rhs_bits = []
    for row_idx, pivot in enumerate(plan.cache.pivot_col):
        rhs = _parity(target_mask & plan.cache.row_ops[row_idx])
        if pivot < 0 and rhs:
            return None
        if pivot >= 0:
            rhs_bits.append(rhs)
    return tuple(rhs_bits)


def _evaluate_q3_free_component_plan_scaled(
    component_plan: _Q3FreeConstraintComponentPlan,
    q1_local: Sequence[int],
    *,
    level: int,
) -> ScaledComplex:
    """Evaluate one reusable q3-free component plan under a concrete q1 vector."""
    if component_plan.backend == "constant":
        component_total = component_plan.precomputed_total
        assert component_total is not None
        return component_total
    if (
        component_plan.quadratic_tensor_q2
        and _is_qubit_quadratic_tensor_q1_vector(q1_local, level=level)
    ):
        component_q = _phase_function_from_parts(
            len(component_plan.variables),
            level=level,
            q0=Fraction(0),
            q1=list(q1_local),
            q2=component_plan.q2,
            q3={},
        )
        return _sum_bl26_quadratic_tensor_component_scaled(component_q)
    if component_plan.backend == "forest":
        adjacency = [dict(neighbors) for neighbors in component_plan.adjacency]
        return _forest_transfer_sum_scaled(list(q1_local), adjacency, level=level)
    if component_plan.backend == "treewidth":
        component_q = _phase_function_from_parts(
            len(component_plan.variables),
            level=level,
            q0=Fraction(0),
            q1=list(q1_local),
            q2=component_plan.q2,
            q3={},
        )
        component_total, _ = _sum_via_treewidth_dp_scaled(component_q, list(component_plan.order))
        return component_total
    if component_plan.binary_phase_plan is not None:
        if _is_binary_phase_q1_vector(q1_local, level=level):
            return _evaluate_binary_phase_quadratic_plan_scaled_batch(
                component_plan.binary_phase_plan,
                np.asarray([q1_local], dtype=np.int64),
                level=level,
            )[0]
        expanded_total = _sum_half_phase_q2_unary_expansion_with_plan_scaled(
            q1_local,
            level=level,
            plan=component_plan.binary_phase_plan,
        )
        if expanded_total is not None:
            return expanded_total
    if component_plan.cutset_plan is not None and component_plan.prefer_cutset_backend:
        return _evaluate_q3_free_cutset_conditioning_plan_scaled(
            component_plan.cutset_plan,
            q1_local,
            level=level,
        )
    if component_plan.mediator_plan is not None:
        return _evaluate_half_phase_mediator_plan_scaled(
            component_plan.mediator_plan,
            q1_local,
        )
    if component_plan.generic_mediator_plan is not None:
        return _evaluate_generic_q2_mediator_plan_scaled(
            component_plan.generic_mediator_plan,
            q1_local,
        )
    if component_plan.cluster_plan is not None:
        return _evaluate_half_phase_cluster_plan_scaled(
            component_plan.cluster_plan,
            q1_local,
        )
    if component_plan.cutset_plan is not None:
        return _evaluate_q3_free_cutset_conditioning_plan_scaled(
            component_plan.cutset_plan,
            q1_local,
            level=level,
        )
    component_q = None
    if component_plan.dense_q2 is not None or component_plan.direct_schur_ok:
        component_q = _phase_function_from_parts(
            len(component_plan.variables),
            level=level,
            q0=Fraction(0),
            q1=list(q1_local),
            q2=component_plan.q2,
            q3={},
        )
        parity_reduced_total = _sum_half_phase_parity_component_reduction_scaled(component_q)
        if parity_reduced_total is not None:
            return parity_reduced_total
    if component_plan.direct_schur_ok:
        component_total = _schur_complement_q3_free_sum_scaled(
            component_q,
            allow_recursive_fallback=True,
        )
        if component_total is not None:
            return component_total

    component_total = None
    if component_plan.dense_q2 is not None and not component_plan.skip_dense_schur:
        component_total = _schur_complement_q3_free_sum_scaled_dense(
            level,
            list(q1_local),
            component_plan.dense_q2,
            allow_recursive_fallback=False,
        )
    if component_total is not None:
        return component_total

    if component_q is None:
        component_q = _phase_function_from_parts(
            len(component_plan.variables),
            level=level,
            q0=Fraction(0),
            q1=list(q1_local),
            q2=component_plan.q2,
            q3={},
        )
    return _sum_q3_free_component_scaled(component_q)


def _evaluate_q3_free_component_plan_scaled_batch(
    component_plan: _Q3FreeConstraintComponentPlan,
    q1_local_batch: np.ndarray,
    *,
    level: int,
) -> list[ScaledComplex]:
    if component_plan.backend == "constant":
        component_total = component_plan.precomputed_total
        assert component_total is not None
        return [component_total] * len(q1_local_batch)
    if component_plan.quadratic_tensor_q2:
        threshold = max(1, (1 << level) // 4)
        residues = np.remainder(np.asarray(q1_local_batch, dtype=np.int64), 1 << level)
        if threshold <= 1 or np.all((residues % threshold) == 0):
            if component_plan.binary_phase_plan is not None and _is_binary_phase_q1_vector(
                residues.ravel(),
                level=level,
            ):
                return _evaluate_binary_phase_quadratic_plan_scaled_batch(
                    component_plan.binary_phase_plan,
                    residues,
                    level=level,
                )
            return [
                _sum_bl26_quadratic_tensor_component_scaled(
                    _phase_function_from_parts(
                        len(component_plan.variables),
                        level=level,
                        q0=Fraction(0),
                        q1=row.tolist(),
                        q2=component_plan.q2,
                        q3={},
                    )
                )
                for row in residues
            ]
    if component_plan.backend == "forest":
        adjacency = [dict(neighbors) for neighbors in component_plan.adjacency]
        return _forest_transfer_sum_scaled_batch(
            np.asarray(q1_local_batch, dtype=np.int64),
            adjacency,
            level=level,
        )
    if component_plan.backend == "treewidth":
        return _sum_q3_free_treewidth_dp_scaled_batch(
            n_vars=len(component_plan.variables),
            level=level,
            q1_batch=np.asarray(q1_local_batch, dtype=np.int64),
            q2=component_plan.q2,
            order=component_plan.order,
            native_plan=component_plan.native_treewidth_plan,
        )
    if component_plan.binary_phase_plan is not None:
        if _is_binary_phase_q1_vector(q1_local_batch.ravel(), level=level):
            return _evaluate_binary_phase_quadratic_plan_scaled_batch(
                component_plan.binary_phase_plan,
                q1_local_batch,
                level=level,
            )
        expanded_totals = _sum_half_phase_q2_unary_expansion_with_plan_scaled_batch(
            np.asarray(q1_local_batch, dtype=np.int64),
            level=level,
            plan=component_plan.binary_phase_plan,
        )
        if expanded_totals is not None:
            return expanded_totals
    if component_plan.cutset_plan is not None and component_plan.prefer_cutset_backend:
        return _evaluate_q3_free_cutset_conditioning_plan_scaled_batch(
            component_plan.cutset_plan,
            q1_local_batch,
            level=level,
        )
    if component_plan.mediator_plan is not None:
        if len(q1_local_batch) < _Q3_FREE_MEDIATOR_BATCH_MIN_ROWS:
            return [
                _evaluate_half_phase_mediator_plan_scaled(
                    component_plan.mediator_plan,
                    q1_local.tolist(),
                )
                for q1_local in q1_local_batch
            ]
        return _evaluate_half_phase_mediator_plan_scaled_batch(
            component_plan.mediator_plan,
            np.asarray(q1_local_batch, dtype=np.int64),
        )
    if component_plan.generic_mediator_plan is not None:
        if len(q1_local_batch) < _Q3_FREE_MEDIATOR_BATCH_MIN_ROWS:
            return [
                _evaluate_generic_q2_mediator_plan_scaled(
                    component_plan.generic_mediator_plan,
                    q1_local.tolist(),
                )
                for q1_local in q1_local_batch
            ]
        return _evaluate_generic_q2_mediator_plan_scaled_batch(
            component_plan.generic_mediator_plan,
            np.asarray(q1_local_batch, dtype=np.int64),
        )
    if component_plan.cluster_plan is not None:
        return _evaluate_half_phase_cluster_plan_scaled_batch(
            component_plan.cluster_plan,
            np.asarray(q1_local_batch, dtype=np.int64),
        )
    if component_plan.cutset_plan is not None:
        return _evaluate_q3_free_cutset_conditioning_plan_scaled_batch(
            component_plan.cutset_plan,
            q1_local_batch,
            level=level,
        )
    if component_plan.dense_q2 is not None:
        mod_q1 = 1 << level
        half_q1 = mod_q1 // 2
        residues = np.remainder(q1_local_batch, mod_q1)
        if np.all((residues == 0) | (residues == half_q1)):
            candidate_q = _phase_function_from_parts(
                len(component_plan.variables),
                level=level,
                q0=Fraction(0),
                q1=[0] * len(component_plan.variables),
                q2=component_plan.q2,
                q3={},
            )
            binary_phase_plan = _build_binary_phase_quadratic_plan(candidate_q)
            if binary_phase_plan is not None:
                return _evaluate_binary_phase_quadratic_plan_scaled_batch(
                    binary_phase_plan,
                    q1_local_batch,
                    level=level,
                )
    return [
        _evaluate_q3_free_component_plan_scaled(
            component_plan,
            q1_local,
            level=level,
        )
        for q1_local in q1_local_batch
    ]


def _q3_free_execution_plan_cache_key(
    q: PhaseFunction,
    *,
    allow_tensor_contraction: bool,
    prefer_reusable_decomposition: bool,
    prefer_one_shot_slicing: bool,
) -> tuple[Any, ...]:
    return (
        _q_key(q),
        bool(allow_tensor_contraction),
        bool(prefer_reusable_decomposition),
        bool(prefer_one_shot_slicing),
        _get_solver_config(),
        bool(_quimb_import_enabled()),
        bool(_kahypar_available()),
    )


def _build_q3_free_execution_plan(
    *,
    q: PhaseFunction,
    allow_tensor_contraction: bool,
    prefer_reusable_decomposition: bool = False,
    prefer_one_shot_slicing: bool = False,
    context: _ReductionContext | None = None,
) -> _Q3FreeExecutionPlan:
    """Plan one instantiated q3-free phase once, then reuse it across solvers."""
    assert not q.q3, "q3-free execution plans require a q3-free kernel."
    cache_key = _q3_free_execution_plan_cache_key(
        q,
        allow_tensor_contraction=allow_tensor_contraction,
        prefer_reusable_decomposition=prefer_reusable_decomposition,
        prefer_one_shot_slicing=prefer_one_shot_slicing,
    )
    if context is not None:
        cached = context.q3_free_constraint_plan_cache.get(cache_key)
        if cached is not None:
            return cached
    cached = _STRUCTURE_Q3_FREE_EXECUTION_PLAN_CACHE.get(cache_key)
    if cached is not None:
        if context is not None:
            context.q3_free_constraint_plan_cache[cache_key] = cached
        return cached

    isolated_vars, component_plans = _plan_q3_free_constraint_components(
        q,
        q.n,
        allow_tensor_contraction=allow_tensor_contraction,
        prefer_reusable_decomposition=prefer_reusable_decomposition,
        prefer_one_shot_slicing=prefer_one_shot_slicing,
    )
    plan = _Q3FreeExecutionPlan(
        level=q.level,
        q0=q.q0,
        q1=tuple(q.q1),
        isolated_vars=isolated_vars,
        components=tuple(component_plans),
    )
    _STRUCTURE_Q3_FREE_EXECUTION_PLAN_CACHE[cache_key] = plan
    if context is not None:
        context.q3_free_constraint_plan_cache[cache_key] = plan
    return plan


def _evaluate_q3_free_planned_components_scaled(
    *,
    q0: Fraction,
    q1: Sequence[int],
    isolated_vars: Sequence[int],
    components: Sequence[_Q3FreeConstraintComponentPlan],
    level: int,
    output_scale_half_pow2: int = 0,
) -> ScaledComplex:
    """Execute already-planned q3-free backends with no further optimization."""
    total = _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q0)))

    if isolated_vars:
        total = _mul_scaled_complex(
            total,
            _product_q1_sum_scaled([q1[var] for var in isolated_vars], level=level),
        )

    for component_plan in components:
        q1_local = [q1[var] for var in component_plan.variables]
        component_total = _evaluate_q3_free_component_plan_scaled(
            component_plan,
            q1_local,
            level=level,
        )
        total = _mul_scaled_complex(total, component_total)

    return _scale_scaled_complex(total, output_scale_half_pow2)


def _evaluate_q3_free_execution_plan_scaled(
    plan: _Q3FreeExecutionPlan,
    *,
    output_scale_half_pow2: int = 0,
) -> ScaledComplex:
    """Execute a fully instantiated q3-free execution plan."""
    return _evaluate_q3_free_planned_components_scaled(
        q0=plan.q0,
        q1=plan.q1,
        isolated_vars=plan.isolated_vars,
        components=plan.components,
        level=plan.level,
        output_scale_half_pow2=output_scale_half_pow2,
    )


def _q3_free_execution_plan_runtime_score(
    plan: _Q3FreeExecutionPlan,
) -> tuple[int, int, int, int, int]:
    """Approximate runtime score for a fully planned q3-free execution plan.

    Lower is better. Prioritize total backend work first, then the worst
    remaining width, then backend-type penalties that prefer cutset-backed
    decompositions over plain direct-treewidth generic solves when work is
    comparable.
    """
    total_work = 0
    max_width = 0
    generic_penalty = 0
    direct_treewidth_penalty = 0

    for component_plan in plan.components:
        total_work += _q3_free_component_plan_work_hint(component_plan)
        max_width = max(max_width, _q3_free_component_plan_width_hint(component_plan))
        if component_plan.backend == "generic" and not component_plan.prefer_cutset_backend:
            generic_penalty += 1
        if (
            component_plan.backend == "treewidth"
            and component_plan.cutset_plan is None
        ):
            direct_treewidth_penalty += 1

    return (
        int(total_work),
        int(max_width),
        int(generic_penalty),
        int(direct_treewidth_penalty),
        len(plan.components),
    )


def _q3_free_planned_components_runtime_score(
    isolated_vars: Sequence[int],
    components: Sequence[_Q3FreeConstraintComponentPlan],
) -> tuple[int, int, int, int, int]:
    """Approximate runtime score for already-planned q3-free components."""
    total_work = 1 if isolated_vars else 0
    max_width = 0
    generic_penalty = 0
    direct_treewidth_penalty = 0

    for component_plan in components:
        total_work += _q3_free_component_plan_work_hint(component_plan)
        max_width = max(max_width, _q3_free_component_plan_width_hint(component_plan))
        if component_plan.backend == "generic" and not component_plan.prefer_cutset_backend:
            generic_penalty += 1
        if (
            component_plan.backend == "treewidth"
            and component_plan.cutset_plan is None
        ):
            direct_treewidth_penalty += 1

    return (
        int(total_work),
        int(max_width),
        int(generic_penalty),
        int(direct_treewidth_penalty),
        len(tuple(components)),
    )


def _q3_free_runtime_score_is_good_baseline(
    runtime_score: tuple[int, int, int, int, int],
    *,
    prefer_one_shot_slicing: bool,
) -> bool:
    """Return whether a baseline q3-free plan is already good enough to trust.

    For one-shot slicing, if the current reusable plan has no generic-tail
    penalty and no plain direct-treewidth penalty, then it is already on a
    cutset-backed or otherwise favorable route. In that case a structural
    optimizer pass is unlikely to repay the cost of building and scoring a
    candidate execution plan.
    """
    if not prefer_one_shot_slicing:
        return False
    _total_work, _max_width, generic_penalty, direct_treewidth_penalty, _components = runtime_score
    return generic_penalty == 0 and direct_treewidth_penalty == 0


def _optimize_q3_free_phase(
    q: PhaseFunction,
    *,
    allow_tensor_contraction: bool = True,
    prefer_reusable_decomposition: bool = False,
    prefer_one_shot_slicing: bool = False,
    baseline_runtime_score: tuple[int, int, int, int, int] | None = None,
    context: _ReductionContext | None = None,
) -> tuple[PhaseFunction, bool]:
    """Apply q3-free optimization only when it improves planned runtime."""
    assert not q.q3, "q3-free optimization expects a q3-free phase function."
    if (
        baseline_runtime_score is not None
        and _q3_free_runtime_score_is_good_baseline(
            baseline_runtime_score,
            prefer_one_shot_slicing=prefer_one_shot_slicing,
        )
    ):
        return q, False
    optimized_q, changed = _optimize_phase_function_structure(q, context=context)
    if not changed:
        return q, False

    candidate_plan = _build_q3_free_execution_plan(
        q=optimized_q,
        allow_tensor_contraction=allow_tensor_contraction,
        prefer_reusable_decomposition=prefer_reusable_decomposition,
        prefer_one_shot_slicing=prefer_one_shot_slicing,
        context=context,
    )
    if baseline_runtime_score is None:
        baseline_plan = _build_q3_free_execution_plan(
            q=q,
            allow_tensor_contraction=allow_tensor_contraction,
            prefer_reusable_decomposition=prefer_reusable_decomposition,
            prefer_one_shot_slicing=prefer_one_shot_slicing,
            context=context,
        )
        baseline_runtime_score = _q3_free_execution_plan_runtime_score(baseline_plan)
    if _q3_free_execution_plan_runtime_score(candidate_plan) < baseline_runtime_score:
        return optimized_q, True
    return q, False


def _phase3_execution_plan_runtime_score(
    q: PhaseFunction,
    *,
    allow_tensor_contraction: bool,
) -> tuple[int, int, int, int, int]:
    cover, order, width, structural_obstruction, backend = _phase3_plan(
        q,
        allow_tensor_contraction=allow_tensor_contraction,
    )
    fully_peeled = backend == "treewidth_dp_peeled"
    if not fully_peeled:
        core_vars, peel_order = _q3_hypergraph_2core(q)
        fully_peeled = bool(peel_order) and not core_vars
    selected_backend, runtime_score, _separator = _choose_phase3_backend(
        q,
        cover,
        order,
        width,
        structural_obstruction,
        allow_tensor_contraction=allow_tensor_contraction,
        fully_peeled=fully_peeled,
        extended_reductions="auto",
    )
    if backend is not None and backend != selected_backend:
        return _phase3_backend_runtime_score(
            q,
            cover,
            order,
            width,
            structural_obstruction,
            backend,
            fully_peeled=fully_peeled,
        )
    return runtime_score


def _phase3_runtime_score_is_good_baseline(
    runtime_score: tuple[int, int, int, int, int],
) -> bool:
    backend_rank, work, width, _cover_size, structural_obstruction = runtime_score
    return (
        backend_rank == 0
        and structural_obstruction == 0
        and width <= _Q3_TREEWIDTH_DP_PEELED_MAX_WIDTH
        and work <= _Q3_TREEWIDTH_DP_PEELED_MAX_WORK
    )


def _evaluate_q3_free_constraint_plan_scaled(
    plan: _Q3FreeConstraintPlan,
    output_bits: BitSequence,
    *,
    allow_tensor_contraction: bool = True,
) -> ScaledComplex:
    rhs_bits = _q3_free_constraint_rhs(plan, output_bits)
    if rhs_bits is None:
        return _ZERO_SCALED

    q1 = list(plan.base_q1)
    for lambda_idx, rhs in enumerate(rhs_bits):
        if rhs:
            q1[plan.lambda_offset + lambda_idx] = plan.rhs_linear_coeff

    instantiated_q = _phase_function_from_parts(
        len(q1),
        level=plan.level,
        q0=plan.q0,
        q1=q1,
        q2=plan.base_q2,
        q3={},
    )
    baseline_runtime_score = _q3_free_planned_components_runtime_score(
        plan.isolated_vars,
        plan.components,
    )
    optimized_q, changed = _optimize_q3_free_phase(
        instantiated_q,
        allow_tensor_contraction=allow_tensor_contraction,
        prefer_one_shot_slicing=True,
        baseline_runtime_score=baseline_runtime_score,
    )
    if changed:
        execution_plan = _build_q3_free_execution_plan(
            q=optimized_q,
            allow_tensor_contraction=allow_tensor_contraction,
            prefer_one_shot_slicing=True,
        )
        return _evaluate_q3_free_execution_plan_scaled(
            execution_plan,
            output_scale_half_pow2=-2 * plan.rank,
        )

    return _evaluate_q3_free_planned_components_scaled(
        q0=plan.q0,
        q1=q1,
        isolated_vars=plan.isolated_vars,
        components=plan.components,
        level=plan.level,
        output_scale_half_pow2=-2 * plan.rank,
    )


def _evaluate_q3_free_constraint_plan_scaled_batch(
    plan: _Q3FreeConstraintPlan,
    output_bits_batch: Sequence[BitSequence],
) -> list[ScaledComplex]:
    """Evaluate a q3-free constraint plan for many output assignments."""
    if not output_bits_batch:
        return []
    if any(len(output_bits) != plan.cache.n for output_bits in output_bits_batch):
        raise ValueError(f"Expected every output to have length {plan.cache.n}.")

    rhs_rows: list[tuple[int, ...]] = []
    supported_indices: list[int] = []
    results: list[ScaledComplex] = [_ZERO_SCALED] * len(output_bits_batch)

    for idx, output_bits in enumerate(output_bits_batch):
        rhs_bits = _q3_free_constraint_rhs(plan, output_bits)
        if rhs_bits is None:
            continue
        supported_indices.append(idx)
        rhs_rows.append(rhs_bits)

    if not supported_indices:
        return results

    q1_batch = np.broadcast_to(
        np.asarray(plan.base_q1, dtype=np.int64),
        (len(supported_indices), len(plan.base_q1)),
    ).copy()
    if plan.rank:
        rhs_matrix = np.asarray(rhs_rows, dtype=np.bool_)
        q1_batch[:, plan.lambda_offset : plan.lambda_offset + plan.rank] = (
            rhs_matrix.astype(np.int64) * int(plan.rhs_linear_coeff)
        )

    totals = [
        _scale_scaled_complex(
            _make_scaled_complex(cmath.exp(2j * cmath.pi * float(plan.q0))),
            -2 * plan.rank,
        )
        for _ in supported_indices
    ]

    if plan.isolated_vars:
        isolated = np.asarray(plan.isolated_vars, dtype=np.int64)
        isolated_q1 = q1_batch[:, isolated]
        for idx, coeffs in enumerate(isolated_q1):
            totals[idx] = _mul_scaled_complex(
                totals[idx],
                _product_q1_sum_scaled(coeffs.tolist(), level=plan.level),
            )

    for component_plan in plan.components:
        q1_local_batch = q1_batch[:, component_plan.variables]
        component_totals = _evaluate_q3_free_component_plan_scaled_batch(
            component_plan,
            q1_local_batch,
            level=plan.level,
        )
        for idx, component_total in enumerate(component_totals):
            totals[idx] = _mul_scaled_complex(totals[idx], component_total)

    for output_idx, total in zip(supported_indices, totals):
        results[output_idx] = total
    return results


def _build_q3_free_raw_constraint_plan(
    state: SchurState,
    *,
    order_hint: Sequence[int] | None = None,
    allow_tensor_contraction: bool = True,
    prefer_reusable_decomposition: bool = False,
    prefer_one_shot_slicing: bool = False,
) -> _Q3FreeRawConstraintPlan:
    """Build a reusable exact q3-free constraint plan on the raw output rows.

    Unlike ``_build_q3_free_constraint_plan``, this keeps one dual variable per
    original output row instead of switching to a row-echelon basis. That makes
    it suitable for one-shot marginal backends, where later queries activate a
    growing prefix of output constraints while leaving the suffix unconstrained.
    """
    assert not state.q.q3, "Raw q3-free constraint plans require a q3-free kernel."

    lambda_offset = state.q.n
    augmented_q2 = dict(state.q.q2)
    bilinear_half_phase = state.q.mod_q2 // 2

    for lambda_idx, row_mask in enumerate(state.eps):
        dual_var = lambda_offset + lambda_idx
        for var in _iter_mask_bits(row_mask):
            key = (var, dual_var) if var < dual_var else (dual_var, var)
            value = (augmented_q2.get(key, 0) + bilinear_half_phase) % state.q.mod_q2
            if value:
                augmented_q2[key] = value
            elif key in augmented_q2:
                del augmented_q2[key]

    base_q = _phase_function_from_parts(
        state.q.n + state.n,
        level=state.q.level,
        q0=state.q.q0,
        q1=list(state.q.q1) + ([0] * state.n),
        q2=augmented_q2,
        q3={},
    )
    isolated_vars, component_plans = _plan_q3_free_constraint_components(
        base_q,
        lambda_offset,
        order_hint=order_hint,
        allow_tensor_contraction=allow_tensor_contraction,
        prefer_reusable_decomposition=prefer_reusable_decomposition,
        prefer_one_shot_slicing=prefer_one_shot_slicing,
    )
    return _Q3FreeRawConstraintPlan(
        eps0=tuple(state.eps0),
        level=state.q.level,
        q0=state.q.q0,
        base_q1=tuple(state.q.q1) + ((0,) * state.n),
        base_q2=dict(augmented_q2),
        lambda_offset=lambda_offset,
        constraint_count=state.n,
        rhs_linear_coeff=state.q.mod_q1 // 2,
        isolated_vars=isolated_vars,
        components=component_plans,
    )


def _restrict_q3_free_component_plan(
    component_plan: _Q3FreeConstraintComponentPlan,
    keep_positions: Sequence[int],
) -> _Q3FreeConstraintComponentPlan | None:
    """Restrict a reusable component plan to a subset of its local variables."""
    keep_positions = tuple(int(pos) for pos in keep_positions)
    if not keep_positions:
        return None
    if len(keep_positions) == len(component_plan.variables):
        return component_plan

    keep_set = set(keep_positions)
    remap = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_positions)}
    variables = tuple(component_plan.variables[idx] for idx in keep_positions)
    q2 = {
        (remap[i], remap[j]): coeff
        for (i, j), coeff in component_plan.q2.items()
        if i in keep_set and j in keep_set
    }

    adjacency = ()
    if component_plan.adjacency:
        adjacency = tuple(
            tuple(
                (remap[neighbor], coeff)
                for neighbor, coeff in component_plan.adjacency[idx]
                if neighbor in keep_set
            )
            for idx in keep_positions
        )

    order = ()
    if component_plan.order:
        order = tuple(remap[idx] for idx in component_plan.order if idx in keep_set)

    dense_q2 = None
    if component_plan.dense_q2 is not None:
        dense_q2 = component_plan.dense_q2[np.ix_(keep_positions, keep_positions)].copy()

    binary_phase_plan = None
    restricted_q = _phase_function_from_parts(
        len(variables),
        level=component_plan.level,
        q0=Fraction(0),
        q1=[0] * len(variables),
        q2=q2,
        q3={},
    )
    if component_plan.binary_phase_plan is not None:
        binary_phase_plan = _build_binary_phase_quadratic_plan(restricted_q)
        mediator_plan = _build_half_phase_mediator_plan(restricted_q)
        generic_mediator_plan = _build_generic_q2_mediator_plan(restricted_q) if mediator_plan is None else None
        cluster_plan = _build_q1_cluster_plan(restricted_q)
    else:
        mediator_plan = None
        generic_mediator_plan = (
            _build_generic_q2_mediator_plan(restricted_q)
            if component_plan.generic_mediator_plan is not None
            else None
        )
        cluster_plan = _build_q1_cluster_plan(restricted_q) if component_plan.cluster_plan is not None else None
    cutset_plan = None
    prefer_cutset_backend = False

    backend = component_plan.backend
    direct_schur_ok = component_plan.direct_schur_ok
    dense_schur_ok = _supports_exact_dense_schur(restricted_q)
    if backend == "generic" and len(variables) > 1 and q2:
        adjacency_maps, edges = _q3_free_graph(restricted_q)
        depth, chords = _q3_free_spanning_data(adjacency_maps, edges)
        if not chords:
            backend = "forest"
            adjacency = tuple(
                tuple(sorted(neighbors.items()))
                for neighbors in adjacency_maps
            )
            order = ()
            dense_q2 = None
            direct_schur_ok = False
        else:
            feedback_vars = _select_feedback_vertices(len(variables), chords, depth)
            max_degree = max((len(neighbors) for neighbors in adjacency_maps), default=0)
            treewidth_order = _q3_free_treewidth_order(
                restricted_q,
                len(feedback_vars),
                max_degree=max_degree,
            )
            lambda_count = (
                sum(1 for var in variables if var >= component_plan.lambda_offset)
                if component_plan.lambda_offset >= 0
                else 0
            )
            if treewidth_order is not None:
                reusable_cutset_plan = (
                    _q3_free_cutset_conditioning_plan(restricted_q)
                    if component_plan.prefer_reusable_decomposition
                    else None
                )
                prefer_reusable_cutset = (
                    component_plan.prefer_reusable_decomposition
                    and _q3_free_prefers_reusable_cutset(
                        restricted_q,
                        treewidth_order=treewidth_order,
                        cutset_plan=reusable_cutset_plan,
                        lambda_count=lambda_count,
                    )
                )
                if prefer_reusable_cutset and reusable_cutset_plan is not None:
                    backend = "generic"
                    order = ()
                    direct_schur_ok = False
                    dense_q2 = None
                    cutset_plan = reusable_cutset_plan
                    prefer_cutset_backend = True
                else:
                    backend = "treewidth"
                    order = tuple(treewidth_order)
                    direct_schur_ok = False
                    dense_q2 = None
            else:
                cutset_plan = _q3_free_cutset_conditioning_plan(restricted_q)
                prefer_cutset_backend = (
                    cutset_plan is not None
                    and (
                        component_plan.prefer_cutset_backend
                        or (
                            component_plan.prefer_reusable_decomposition
                            and lambda_count >= _Q3_FREE_REUSABLE_CUTSET_MIN_LAMBDA_VARS
                        )
                    )
                )
                direct_schur_ok = (
                    dense_q2 is not None
                    and len(feedback_vars) > _SCHUR_COMPLEMENT_CROSSOVER_FVS
                    and binary_phase_plan is None
                    and mediator_plan is None
                    and generic_mediator_plan is None
                    and cluster_plan is None
                    and dense_schur_ok
                    and not (
                        component_plan.cutset_plan is not None
                        and (
                            prefer_cutset_backend
                            or _q3_free_prefers_locality_preserving_cutset(
                                restricted_q,
                                feedback_size=len(feedback_vars),
                                max_degree=max_degree,
                                edge_density=_q3_free_edge_density(restricted_q),
                                allow_tensor_contraction=True,
                            )
                        )
                    )
                )
    else:
        direct_schur_ok = False

    return _Q3FreeConstraintComponentPlan(
        variables=variables,
        level=component_plan.level,
        q2=q2,
        backend=backend,
        adjacency=adjacency,
        order=order,
        dense_q2=dense_q2,
        binary_phase_plan=binary_phase_plan,
        mediator_plan=mediator_plan,
        generic_mediator_plan=generic_mediator_plan,
        cluster_plan=cluster_plan,
        cutset_plan=cutset_plan,
        native_treewidth_plan=(
            _build_native_q3_free_treewidth_plan(
                n_vars=len(variables),
                level=component_plan.level,
                q2=q2,
                order=order,
            )
            if backend == "treewidth"
            else None
        ),
        skip_dense_schur=(
            component_plan.skip_dense_schur
            or (backend == "generic" and not dense_schur_ok)
        ),
        direct_schur_ok=direct_schur_ok,
        quadratic_tensor_q2=_is_half_phase_q2(restricted_q),
        lambda_offset=component_plan.lambda_offset,
        prefer_reusable_decomposition=component_plan.prefer_reusable_decomposition,
        prefer_cutset_backend=prefer_cutset_backend,
    )


def _restrict_q3_free_raw_constraint_plan(
    plan: _Q3FreeRawConstraintPlan,
    active_count: int,
) -> _Q3FreeRawConstraintRestrictedPlan:
    """Restrict a raw-output q3-free plan to the first ``active_count`` outputs."""
    if not 0 <= active_count <= plan.constraint_count:
        raise ValueError(
            f"Expected active_count in [0, {plan.constraint_count}], received {active_count}."
        )

    lambda_limit = plan.lambda_offset + active_count
    isolated_vars = tuple(
        var for var in plan.isolated_vars
        if var < plan.lambda_offset or var < lambda_limit
    )
    components = []
    for component_plan in plan.components:
        keep_positions = [
            idx
            for idx, var in enumerate(component_plan.variables)
            if var < plan.lambda_offset or var < lambda_limit
        ]
        restricted = _restrict_q3_free_component_plan(component_plan, keep_positions)
        if restricted is not None:
            components.append(restricted)

    return _Q3FreeRawConstraintRestrictedPlan(
        active_count=active_count,
        isolated_vars=isolated_vars,
        components=tuple(components),
    )


def _evaluate_q3_free_raw_constraint_plan_scaled(
    plan: _Q3FreeRawConstraintPlan,
    restricted_plan: _Q3FreeRawConstraintRestrictedPlan,
    output_bits: BitSequence,
    *,
    allow_tensor_contraction: bool = True,
) -> ScaledComplex:
    """Evaluate a raw-output q3-free plan for one active output prefix."""
    if len(output_bits) != restricted_plan.active_count:
        raise ValueError(
            f"Expected {restricted_plan.active_count} output bits, received {len(output_bits)}."
        )

    q1 = list(plan.base_q1)
    for idx, bit in enumerate(output_bits):
        if (int(bit) ^ plan.eps0[idx]) & 1:
            q1[plan.lambda_offset + idx] = plan.rhs_linear_coeff

    instantiated_q = _phase_function_from_parts(
        len(q1),
        level=plan.level,
        q0=plan.q0,
        q1=q1,
        q2=plan.base_q2,
        q3={},
    )
    baseline_runtime_score = _q3_free_planned_components_runtime_score(
        restricted_plan.isolated_vars,
        restricted_plan.components,
    )
    optimized_q, changed = _optimize_q3_free_phase(
        instantiated_q,
        allow_tensor_contraction=allow_tensor_contraction,
        prefer_one_shot_slicing=True,
        baseline_runtime_score=baseline_runtime_score,
    )
    if changed:
        execution_plan = _build_q3_free_execution_plan(
            q=optimized_q,
            allow_tensor_contraction=allow_tensor_contraction,
            prefer_one_shot_slicing=True,
        )
        return _evaluate_q3_free_execution_plan_scaled(
            execution_plan,
            output_scale_half_pow2=-2 * restricted_plan.active_count,
        )

    return _evaluate_q3_free_planned_components_scaled(
        q0=plan.q0,
        q1=q1,
        isolated_vars=restricted_plan.isolated_vars,
        components=restricted_plan.components,
        level=plan.level,
        output_scale_half_pow2=-2 * restricted_plan.active_count,
    )


def _evaluate_q3_free_raw_constraint_plan_scaled_batch(
    plan: _Q3FreeRawConstraintPlan,
    restricted_plan: _Q3FreeRawConstraintRestrictedPlan,
    output_bits_batch: Sequence[BitSequence],
) -> list[ScaledComplex]:
    """Evaluate a raw-output q3-free plan for many active output prefixes."""
    if not output_bits_batch:
        return []
    if any(len(output_bits) != restricted_plan.active_count for output_bits in output_bits_batch):
        raise ValueError(
            f"Expected every output to have length {restricted_plan.active_count}."
        )

    q1_batch = np.broadcast_to(
        np.asarray(plan.base_q1, dtype=np.int64),
        (len(output_bits_batch), len(plan.base_q1)),
    ).copy()
    if restricted_plan.active_count:
        output_matrix = np.asarray(output_bits_batch, dtype=np.bool_)
        rhs_mask = np.logical_xor(
            output_matrix,
            np.asarray(plan.eps0[: restricted_plan.active_count], dtype=np.bool_),
        )
        q1_batch[:, plan.lambda_offset : plan.lambda_offset + restricted_plan.active_count] = (
            rhs_mask.astype(np.int64) * int(plan.rhs_linear_coeff)
        )

    totals = [
        _scale_scaled_complex(
            _make_scaled_complex(cmath.exp(2j * cmath.pi * float(plan.q0))),
            -2 * restricted_plan.active_count,
        )
        for _ in output_bits_batch
    ]

    if restricted_plan.isolated_vars:
        isolated = np.asarray(restricted_plan.isolated_vars, dtype=np.int64)
        isolated_q1 = q1_batch[:, isolated]
        for idx, coeffs in enumerate(isolated_q1):
            totals[idx] = _mul_scaled_complex(
                totals[idx],
                _product_q1_sum_scaled(coeffs.tolist(), level=plan.level),
            )

    for component_plan in restricted_plan.components:
        q1_local_batch = q1_batch[:, component_plan.variables]
        component_totals = _evaluate_q3_free_component_plan_scaled_batch(
            component_plan,
            q1_local_batch,
            level=plan.level,
        )
        for idx, component_total in enumerate(component_totals):
            totals[idx] = _mul_scaled_complex(totals[idx], component_total)

    return totals


def _component_restriction(q, component):
    comp = sorted(component)
    remap = {old: new for new, old in enumerate(comp)}
    q1 = [q.q1[idx] for idx in comp]
    q2 = {
        (remap[i], remap[j]): value
        for (i, j), value in q.q2.items()
        if i in remap and j in remap
    }
    q3 = {
        (remap[i], remap[j], remap[k]): value
        for (i, j, k), value in q.q3.items()
        if i in remap and j in remap and k in remap
    }
    return PhaseFunction(len(comp), level=q.level, q0=Fraction(0), q1=q1, q2=q2, q3=q3)


def _sum_q3_free_direct_scaled(q, context=None):
    """Solve an already q3-free kernel directly without re-entering cubic reduction."""
    assert not q.q3, "Direct q3-free helper expects a q3-free phase function."
    if context is not None and not context.preserve_scale:
        total_complex, phase_info = _gauss_sum_q3_free(
            q,
            allow_tensor_contraction=context.allow_tensor_contraction,
        )
        total = _make_scaled_complex(total_complex)
    else:
        total, phase_info = _gauss_sum_q3_free_scaled(
            q,
            allow_tensor_contraction=(True if context is None else context.allow_tensor_contraction),
        )
    structural_obstruction = 0
    gauss_obstruction = _gauss_obstruction(q, structural_obstruction)
    return total, {
        'quad': 0,
        'constraint': 0,
        'branched': 0,
        'remaining': 0,
        'structural_obstruction': structural_obstruction,
        'gauss_obstruction': gauss_obstruction,
        'cost_r': 0,
        'phase_states': phase_info.get('phase_states', 0),
        'phase_splits': phase_info.get('phase_splits', 0),
        'phase3_backend': _q3_free_phase3_backend_name(q),
    }


def _sum_factorized_components_scaled(q, components, context=None):
    """Reduce disconnected components independently and multiply the results."""
    if not components:
        return _mul_scaled_complex(
            _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q.q0))),
            _scale_scaled_complex(_ONE_SCALED, 2 * q.n),
        ), {
            'quad': 0,
            'constraint': 0,
            'branched': 0,
            'remaining': 0,
            'structural_obstruction': 0,
            'gauss_obstruction': 0,
            'cost_r': 0,
            'phase_states': 0,
            'phase_splits': 0,
            'phase3_backend': None,
        }

    covered = set().union(*components)
    total = _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q.q0)))
    if len(covered) < q.n:
        total = _scale_scaled_complex(total, 2 * (q.n - len(covered)))

    total_quad = total_constraint = 0
    max_branched = 0
    max_remaining = 0
    max_structural = 0
    max_gauss = 0
    max_cost_r = 0
    phase_states = phase_splits = 0
    phase3_backend = None
    dominant_cost_r = -1

    for component in components:
        restricted = _component_restriction(q, component)
        if not restricted.q3:
            component_total, component_info = _sum_q3_free_direct_scaled(restricted, context=context)
        else:
            component_total, component_info = _reduce_and_sum_scaled(restricted, context=context)
        total = _mul_scaled_complex(total, component_total)
        total_quad += component_info['quad']
        total_constraint += component_info['constraint']
        max_branched = max(max_branched, component_info['branched'])
        max_remaining = max(max_remaining, component_info['remaining'])
        max_structural = max(
            max_structural,
            component_info.get('structural_obstruction', component_info['remaining']),
        )
        max_gauss = max(
            max_gauss,
            component_info.get(
                'gauss_obstruction',
                component_info.get('structural_obstruction', component_info['remaining']),
            ),
        )
        component_cost_r = component_info.get('cost_r', component_info['remaining'])
        max_cost_r = max(max_cost_r, component_cost_r)
        phase_states += component_info.get('phase_states', 0)
        phase_splits += component_info.get('phase_splits', 0)
        component_backend = component_info.get('phase3_backend')
        if component_backend is None:
            continue
        if component_cost_r > dominant_cost_r:
            phase3_backend = component_backend
            dominant_cost_r = component_cost_r
        elif component_cost_r == dominant_cost_r and phase3_backend not in {None, component_backend}:
            phase3_backend = "mixed"

    return total, {
        'quad': total_quad,
        'constraint': total_constraint,
        'branched': max_branched,
        'remaining': max_remaining,
        'structural_obstruction': max_structural,
        'gauss_obstruction': max_gauss,
        'cost_r': max_cost_r,
        'phase_states': phase_states,
        'phase_splits': phase_splits,
        'phase3_backend': phase3_backend,
    }


def _bruteforce_q3_free_sum(q):
    return sum(
        cmath.exp(2j * cmath.pi * float(q.evaluate([(mask >> bit) & 1 for bit in range(q.n)])))
        for mask in range(2**q.n)
    )


def _q3_free_spanning_data(adjacency, edges):
    """Build a spanning forest and record the non-tree edges."""
    n = len(adjacency)
    depth = [0] * n
    visited = [False] * n
    tree_edges = set()

    for root in range(n):
        if visited[root]:
            continue
        visited[root] = True
        stack = [root]
        while stack:
            node = stack.pop()
            for nbr in sorted(adjacency[node], reverse=True):
                if visited[nbr]:
                    continue
                visited[nbr] = True
                depth[nbr] = depth[node] + 1
                tree_edges.add((min(node, nbr), max(node, nbr)))
                stack.append(nbr)

    chords = [(i, j, phase) for i, j, phase in edges if (i, j) not in tree_edges]
    return depth, chords


def _select_feedback_vertices(n, chords, depth):
    """
    Pick a small set of vertices that covers every non-tree edge.

    Fixing these variables removes every cycle edge, leaving a forest that can
    be summed in linear time by transfer messages.
    """
    if not chords:
        return []

    incident = [set() for _ in range(n)]
    for idx, (i, j, _) in enumerate(chords):
        incident[i].add(idx)
        incident[j].add(idx)

    covered = set()
    chosen = []
    candidates = [var for var in range(n) if incident[var]]
    while len(covered) < len(chords):
        best = max(
            candidates,
            key=lambda var: (len(incident[var] - covered), depth[var], -var),
        )
        newly_covered = incident[best] - covered
        if not newly_covered:
            raise RuntimeError("Failed to cover q3-free cycle edges.")
        chosen.append(best)
        covered.update(newly_covered)
    return sorted(set(chosen))


def _forest_transfer_sum(q1, adjacency, level: int = 3):
    """
    Evaluate a q2-forest by leaf-to-root transfer.

    Each subtree message is the residue-8 transfer function

        F_v(r) = A_v + omega^r B_v

    compressed into the pair ``(A_v, B_v)``. Querying the parent-imposed
    residue shift r then costs O(1), so a full forest sums in linear time.
    """
    n = len(q1)
    if n == 0:
        return 1.0 + 0j
    if not any(adjacency):
        return _product_q1_sum(q1, level=level)

    omega = _omega_table(level)
    modulus = 1 << level
    total = 1.0 + 0j
    visited = [False] * n
    base = [0j] * n
    excited = [0j] * n

    for root in range(n):
        if visited[root]:
            continue

        postorder = []
        stack = [(root, -1, False)]
        while stack:
            node, parent, expanded = stack.pop()
            if expanded:
                postorder.append((node, parent))
                continue
            if visited[node]:
                continue
            visited[node] = True
            stack.append((node, parent, True))
            for nbr in sorted(adjacency[node], reverse=True):
                if nbr == parent:
                    continue
                if visited[nbr]:
                    raise RuntimeError("Feedback elimination left a cycle in the q3-free forest.")
                stack.append((nbr, node, False))

        for node, parent in postorder:
            off_term = 1.0 + 0j
            on_term = omega[q1[node] % modulus]
            for child, shift in adjacency[node].items():
                if child == parent:
                    continue
                off_term *= base[child] + excited[child]
                on_term *= base[child] + omega[shift % modulus] * excited[child]
            base[node] = off_term
            excited[node] = on_term

        total *= base[root] + excited[root]

    return total


def _forest_postorder_components(adjacency) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
    """Return rooted postorders for each connected component of a forest."""
    visited = [False] * len(adjacency)
    components: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for root in range(len(adjacency)):
        if visited[root]:
            continue

        postorder: list[tuple[int, int]] = []
        stack = [(root, -1, False)]
        while stack:
            node, parent, expanded = stack.pop()
            if expanded:
                postorder.append((node, parent))
                continue
            if visited[node]:
                continue
            visited[node] = True
            stack.append((node, parent, True))
            neighbors = adjacency[node]
            iterable = neighbors.keys() if isinstance(neighbors, dict) else (neighbor for neighbor, _shift in neighbors)
            for nbr in sorted(iterable, reverse=True):
                if nbr == parent:
                    continue
                if visited[nbr]:
                    raise RuntimeError("Feedback elimination left a cycle in the q3-free forest.")
                stack.append((nbr, node, False))
        components.append((root, tuple(postorder)))
    return tuple(components)


def _forest_transfer_sum_scaled(q1, adjacency, level: int = 3):
    """Scaled-complex companion to ``_forest_transfer_sum`` for tiny amplitudes."""
    n = len(q1)
    if n == 0:
        return _ONE_SCALED
    if not any(adjacency):
        return _product_q1_sum_scaled(q1, level=level)

    omega_scaled = _omega_scaled_table(level)
    modulus = 1 << level
    total = _ONE_SCALED
    base = [_ZERO_SCALED] * n
    excited = [_ZERO_SCALED] * n

    for root, postorder in _forest_postorder_components(adjacency):
        for node, parent in postorder:
            off_term = _ONE_SCALED
            on_term = omega_scaled[q1[node] % modulus]
            for child, shift in adjacency[node].items():
                if child == parent:
                    continue
                off_term = _mul_scaled_complex(off_term, _add_scaled_complex(base[child], excited[child]))
                on_term = _mul_scaled_complex(
                    on_term,
                    _add_scaled_complex(
                        base[child],
                        _mul_scaled_complex(
                            omega_scaled[shift % modulus],
                            excited[child],
                        ),
                    ),
                )
            base[node] = off_term
            excited[node] = on_term

        total = _mul_scaled_complex(total, _add_scaled_complex(base[root], excited[root]))

    return total


def _forest_transfer_sum_scaled_batch(
    q1_batch: np.ndarray,
    adjacency,
    *,
    level: int = 3,
) -> list[ScaledComplex]:
    """Batch scaled transfer over one shared q2-forest."""
    if len(q1_batch) == 0:
        return []

    batch = np.asarray(q1_batch, dtype=np.int64)
    n = batch.shape[1]
    if n == 0:
        return [_ONE_SCALED] * len(batch)
    if not any(adjacency):
        return [
            _product_q1_sum_scaled(row.tolist(), level=level)
            for row in batch
        ]

    omega_scaled = _omega_scaled_table(level)
    omega_values, omega_exponents = _scaled_table_to_arrays(omega_scaled)
    modulus = 1 << level
    total_values, total_exponents = _scaled_arrays_from_constant(_ONE_SCALED, (len(batch),))
    base_values, base_exponents = _scaled_arrays_from_constant(_ZERO_SCALED, (len(batch), n))
    excited_values, excited_exponents = _scaled_arrays_from_constant(_ZERO_SCALED, (len(batch), n))

    for root, postorder in _forest_postorder_components(adjacency):
        for node, parent in postorder:
            off_values, off_exponents = _scaled_arrays_from_constant(_ONE_SCALED, (len(batch),))
            residues = np.remainder(batch[:, node], modulus)
            on_values = omega_values[residues]
            on_exponents = omega_exponents[residues]
            for child, shift in adjacency[node].items():
                if child == parent:
                    continue
                child_total_values, child_total_exponents = _add_scaled_complex_arrays(
                    base_values[:, child],
                    base_exponents[:, child],
                    excited_values[:, child],
                    excited_exponents[:, child],
                )
                off_values, off_exponents = _mul_scaled_complex_arrays(
                    off_values,
                    off_exponents,
                    child_total_values,
                    child_total_exponents,
                )

                shifted_excited_values, shifted_excited_exponents = _mul_scaled_complex_arrays(
                    omega_values[np.full(len(batch), shift % modulus, dtype=np.int64)],
                    omega_exponents[np.full(len(batch), shift % modulus, dtype=np.int64)],
                    excited_values[:, child],
                    excited_exponents[:, child],
                )
                on_child_values, on_child_exponents = _add_scaled_complex_arrays(
                    base_values[:, child],
                    base_exponents[:, child],
                    shifted_excited_values,
                    shifted_excited_exponents,
                )
                on_values, on_exponents = _mul_scaled_complex_arrays(
                    on_values,
                    on_exponents,
                    on_child_values,
                    on_child_exponents,
                )
            base_values[:, node] = off_values
            base_exponents[:, node] = off_exponents
            excited_values[:, node] = on_values
            excited_exponents[:, node] = on_exponents

        component_values, component_exponents = _add_scaled_complex_arrays(
            base_values[:, root],
            base_exponents[:, root],
            excited_values[:, root],
            excited_exponents[:, root],
        )
        total_values, total_exponents = _mul_scaled_complex_arrays(
            total_values,
            total_exponents,
            component_values,
            component_exponents,
        )

    return [
        (complex(value), int(half_pow2_exp))
        for value, half_pow2_exp in zip(total_values, total_exponents)
    ]


def _dense_q2_matrix(q):
    """Materialize the q2 coefficients of a q3-free kernel as a symmetric matrix."""
    matrix = np.zeros((q.n, q.n), dtype=np.int64)
    for (i, j), coeff in q.q2.items():
        value = int(coeff % q.mod_q2)
        if not value:
            continue
        matrix[i, j] = value
        matrix[j, i] = value
    return matrix


def _quadratic_residue_threshold(q) -> int:
    """Return the quarter-turn residue threshold used by exact quadratic pivots."""
    return max(1, q.mod_q1 // 4)


def _quadratic_pair_correction(q, left_coeff: int, right_coeff: int) -> int:
    """Return the exact q2-space correction induced by one quadratic pivot.

    For q3-free BL26 kernels, every admissible incident q2 coefficient is either
    ``0`` or ``threshold = mod_q1 // 4`` in q2-residue space. Summing out a
    quadratic pivot therefore introduces an XOR-parity coupling on the pivot's
    neighborhood, which in q2-residue space is represented by ``threshold`` for
    every pair of active neighbors. Expressed generically in the same residue
    space as ``q.q2``, that correction is ``left * right / threshold``.
    """
    threshold = _quadratic_residue_threshold(q)
    if not left_coeff or not right_coeff:
        return 0
    return int((int(left_coeff) * int(right_coeff) // threshold) % q.mod_q2)


def _phase_from_dense_q2(level: int, q1, q2_matrix: np.ndarray, active) -> PhaseFunction:
    """Convert dense q1/q2 data on ``active`` variables back into a q3-free phase."""
    active = tuple(int(idx) for idx in active)
    if not active:
        return _phase_function_from_parts(0, level=level, q0=Fraction(0), q1=[], q2={}, q3={})

    submatrix = q2_matrix[np.ix_(active, active)]
    upper_rows, upper_cols = np.nonzero(np.triu(submatrix, 1))
    q2 = {
        (int(row), int(col)): int(submatrix[row, col])
        for row, col in zip(upper_rows.tolist(), upper_cols.tolist())
        if int(submatrix[row, col])
    }
    return _phase_function_from_parts(
        len(active),
        level=level,
        q0=Fraction(0),
        q1=[int(q1[idx]) for idx in active],
        q2=q2,
        q3={},
    )


def _swap_dense_q2_variables(q1: np.ndarray, q2_matrix: np.ndarray, left: int, right: int) -> None:
    """Swap two variable positions inside the dense q1/q2 representation."""
    if left == right:
        return
    q1[left], q1[right] = q1[right], q1[left]
    q2_matrix[[left, right], :] = q2_matrix[[right, left], :]
    q2_matrix[:, [left, right]] = q2_matrix[:, [right, left]]


def _swap_dense_matrix_variables(matrix: np.ndarray, left: int, right: int) -> None:
    """Swap two variable positions inside a dense square matrix."""
    if left == right:
        return
    matrix[[left, right], :] = matrix[[right, left], :]
    matrix[:, [left, right]] = matrix[:, [right, left]]


def _schur_complement_q3_free_sum_scaled_dense(
    level: int,
    q1,
    q2_matrix: np.ndarray,
    *,
    q0: Fraction = Fraction(0),
    allow_recursive_fallback: bool = True,
    return_residual_on_fallback: bool = False,
):
    """Dense-array implementation of the BL26 q3-free Schur fallback."""
    if len(q1) == 0:
        return _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q0)))

    mod_q1 = 1 << level
    mod_q2 = 1 << max(level - 1, 0)
    threshold = max(1, mod_q1 // 4)
    threshold_shift = threshold.bit_length() - 1 if threshold > 1 else 0
    q1 = np.remainder(np.asarray(q1, dtype=np.int64), mod_q1)
    q2_matrix = np.asarray(q2_matrix, dtype=np.int64).copy()
    adjacency = q2_matrix != 0
    odd_adjacency = (q2_matrix & 1) != 0
    degrees = adjacency.sum(axis=1, dtype=np.int64)
    odd_counts = odd_adjacency.sum(axis=1, dtype=np.int64)
    active_count = len(q1)
    scale_half_pow2 = 0

    while active_count:
        active_degrees = degrees[:active_count]
        coeffs = q1[:active_count]

        if not np.any(active_degrees):
            residual = _product_q1_sum_scaled(coeffs.tolist(), level=level)
            constant = _scale_scaled_complex(
                _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q0))),
                scale_half_pow2,
            )
            return _mul_scaled_complex(constant, residual)

        if threshold > 1:
            divisible = (coeffs & (threshold - 1)) == 0
            reduced = (coeffs >> threshold_shift) & 3
        else:
            divisible = np.ones(active_count, dtype=np.bool_)
            reduced = coeffs & 3
        odd_coupling = odd_counts[:active_count] != 0
        quadratic_mask = divisible & ~odd_coupling & ((reduced == 1) | (reduced == 3))

        zero_mask = divisible & (active_degrees == 0) & (reduced == 2)
        if np.any(zero_mask):
            return _ZERO_SCALED

        decoupled_mask = divisible & (active_degrees == 0) & (reduced == 0)
        if np.any(decoupled_mask):
            pivot_idx = active_count - 1
            if decoupled_mask[pivot_idx]:
                local_idx = pivot_idx
            else:
                local_idx = int(np.flatnonzero(decoupled_mask)[-1])
            _swap_dense_q2_variables(q1, q2_matrix, local_idx, pivot_idx)
            _swap_dense_matrix_variables(adjacency, local_idx, pivot_idx)
            _swap_dense_matrix_variables(odd_adjacency, local_idx, pivot_idx)
            degrees[local_idx], degrees[pivot_idx] = degrees[pivot_idx], degrees[local_idx]
            odd_counts[local_idx], odd_counts[pivot_idx] = odd_counts[pivot_idx], odd_counts[local_idx]
            adjacency[pivot_idx, :active_count] = False
            adjacency[:active_count, pivot_idx] = False
            odd_adjacency[pivot_idx, :active_count] = False
            odd_adjacency[:active_count, pivot_idx] = False
            degrees[pivot_idx] = 0
            odd_counts[pivot_idx] = 0
            active_count -= 1
            scale_half_pow2 += 2
            continue

        if not np.any(quadratic_mask):
            if not allow_recursive_fallback:
                if return_residual_on_fallback:
                    residual_phase = _phase_from_dense_q2(level, q1, q2_matrix, range(active_count))
                    residual_phase.q0 = q0
                    return residual_phase, scale_half_pow2
                return None
            residual_phase = _phase_from_dense_q2(level, q1, q2_matrix, range(active_count))
            residual_total = _sum_q3_free_component_scaled(
                residual_phase,
                allow_schur_complement=False,
            )
            constant = _scale_scaled_complex(
                _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q0))),
                scale_half_pow2,
            )
            return _mul_scaled_complex(constant, residual_total)

        pivot_idx = active_count - 1
        if quadratic_mask[pivot_idx]:
            local_idx = pivot_idx
        else:
            local_idx = int(np.flatnonzero(quadratic_mask)[-1])
        reduced_value = int(reduced[local_idx])
        _swap_dense_q2_variables(q1, q2_matrix, local_idx, pivot_idx)
        _swap_dense_matrix_variables(adjacency, local_idx, pivot_idx)
        _swap_dense_matrix_variables(odd_adjacency, local_idx, pivot_idx)
        degrees[local_idx], degrees[pivot_idx] = degrees[pivot_idx], degrees[local_idx]
        odd_counts[local_idx], odd_counts[pivot_idx] = odd_counts[pivot_idx], odd_counts[local_idx]
        row = q2_matrix[pivot_idx, :pivot_idx].copy()
        nz = np.flatnonzero(row)
        odd_nz = np.flatnonzero(row & 1)

        q0 = (q0 + (Fraction(1, 8) if reduced_value == 1 else Fraction(7, 8))) % 1
        scale_half_pow2 += 1
        if pivot_idx:
            sign = -1 if reduced_value == 1 else 1
            q1[:pivot_idx] = np.remainder(q1[:pivot_idx] + sign * row, mod_q1)
            if nz.size > 1:
                nz_values = row[nz]
                block_index = np.ix_(nz, nz)
                old_adj = adjacency[block_index].copy()
                old_odd = odd_adjacency[block_index].copy()
                correction = np.remainder(
                    np.multiply.outer(nz_values, nz_values) // threshold,
                    mod_q2,
                )
                q2_matrix[block_index] = np.remainder(q2_matrix[block_index] + correction, mod_q2)
                diag = np.arange(nz.size)
                q2_matrix[nz[diag], nz[diag]] = 0
                new_adj = q2_matrix[block_index] != 0
                new_odd = (q2_matrix[block_index] & 1) != 0
                adjacency[block_index] = new_adj
                odd_adjacency[block_index] = new_odd
                degrees[nz] += (
                    new_adj.sum(axis=1, dtype=np.int64) - old_adj.sum(axis=1, dtype=np.int64)
                )
                odd_counts[nz] += (
                    new_odd.sum(axis=1, dtype=np.int64) - old_odd.sum(axis=1, dtype=np.int64)
                )
            if nz.size:
                degrees[nz] -= 1
            if odd_nz.size:
                odd_counts[odd_nz] -= 1
        q2_matrix[pivot_idx, :active_count] = 0
        q2_matrix[:active_count, pivot_idx] = 0
        adjacency[pivot_idx, :active_count] = False
        adjacency[:active_count, pivot_idx] = False
        odd_adjacency[pivot_idx, :active_count] = False
        odd_adjacency[:active_count, pivot_idx] = False
        degrees[pivot_idx] = 0
        odd_counts[pivot_idx] = 0
        active_count -= 1

    return _scale_scaled_complex(
        _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q0))),
        scale_half_pow2,
    )


def _schur_complement_q3_free_sum_scaled(
    q,
    *,
    allow_recursive_fallback: bool = True,
):
    """Dense BL26-style q3-free fallback using vectorized Schur updates.

    This fallback is only exact while it can keep applying the same one-variable
    quadratic Gauss eliminations as ``_elim_quadratic``. The difference is that
    once a dense q2 component has crossed into the regime where graph-based
    feedback branching is unattractive, the coefficient updates are carried out
    on dense NumPy arrays rather than through repeated Python dict surgery.

    If the dense Schur pass reaches a residual kernel that no longer exposes a
    valid quadratic pivot, the remaining exact work is handed back to the
    existing q3-free solver.
    """
    assert not q.q3, "BL26 dense fallback only applies to q3-free kernels."
    if not _supports_exact_dense_schur(q):
        return None
    if q.n == 0:
        return _ONE_SCALED
    if not q.q2:
        return _mul_scaled_complex(
                _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q.q0))),
                _product_q1_sum_scaled(q.q1, level=q.level),
            )

    return _schur_complement_q3_free_sum_scaled_dense(
        q.level,
        q.q1,
        _dense_q2_matrix(q),
        q0=q.q0,
        allow_recursive_fallback=allow_recursive_fallback,
    )


def _schur_complement_q3_free_sum(
    q,
    *,
    allow_recursive_fallback: bool = True,
):
    """Unscaled wrapper for the dense BL26 q3-free fallback."""
    scaled = _schur_complement_q3_free_sum_scaled(
        q,
        allow_recursive_fallback=allow_recursive_fallback,
    )
    if scaled is None:
        return None
    return _scaled_to_complex(scaled)


def _qubit_quadratic_tensor_obstruction_support(q) -> tuple[int, ...]:
    """Return variables that keep ``q`` outside BL26's qubit quadratic class.

    For qubit quadratic tensors, BL26's coefficient groups permit only q1
    coefficients from a 4-element class and q2 coefficients from a 2-element
    class. In TerKet's dyadic integer encoding at level ``L``, both conditions
    reduce to requiring every q1/q2 coefficient to be a multiple of
    ``2^(L-2)``. Any surviving coefficient outside that class requires a
    higher-order tensor description even when ``q.q3`` is empty.
    """
    threshold = max(1, q.mod_q1 // 4)
    if threshold <= 1:
        return ()

    support: set[int] = set()
    for idx, coeff in enumerate(q.q1):
        if coeff % threshold:
            support.add(idx)
    for (left, right), coeff in q.q2.items():
        if coeff % threshold:
            support.add(left)
            support.add(right)
    return tuple(sorted(support))


def _qubit_quadratic_tensor_obstruction(q) -> int:
    """Return the size of the residual non-quadratic qubit support of ``q``."""
    return len(_qubit_quadratic_tensor_obstruction_support(q))


def _supports_exact_dense_schur(q) -> bool:
    """Return whether the dense Schur q3-free backend is exact for ``q``.

    The dense backend is exact on q3-free kernels whose q2 support lies
    entirely in the half-phase class. Unary coefficients may be arbitrary; the
    dense eliminator only pivots on quarter-turn q1 residues and hands any
    remaining hard-unary work back to the generic exact q3-free solver.
    """
    return int(q.level) >= 3 and _is_half_phase_q2(q)


def _sum_bl26_quadratic_tensor_component_scaled(q):
    """Exactly contract one BL26 qubit quadratic-tensor component."""
    assert not q.q3, "Quadratic-tensor contraction requires a q3-free kernel."
    if _qubit_quadratic_tensor_obstruction(q):
        raise ValueError("Quadratic-tensor contraction requires zero gauss obstruction.")

    if q.n == 0:
        return _ONE_SCALED
    if not q.q2:
        return _product_q1_sum_scaled(q.q1, level=q.level)

    binary_total = _sum_binary_phase_quadratic_scaled(q)
    if binary_total is not None:
        return binary_total

    dense_result = _schur_complement_q3_free_sum_scaled_dense(
        q.level,
        q.q1,
        _dense_q2_matrix(q),
        q0=q.q0,
        allow_recursive_fallback=False,
        return_residual_on_fallback=True,
    )
    if isinstance(dense_result[0], complex):
        return dense_result

    residual_phase, scale_half_pow2 = dense_result
    if _qubit_quadratic_tensor_obstruction(residual_phase):
        raise RuntimeError(
            "BL26 quadratic-tensor contraction failed on a zero-obstruction q3-free kernel."
        )
    residual_total = _sum_bl26_quadratic_tensor_component_scaled(residual_phase)
    return _scale_scaled_complex(residual_total, scale_half_pow2)


def _sum_bl26_quadratic_tensor_component(q):
    """Unscaled wrapper for ``_sum_bl26_quadratic_tensor_component_scaled``."""
    return _scaled_to_complex(_sum_bl26_quadratic_tensor_component_scaled(q))


def _sum_q3_free_via_gauss_reduction_scaled(q):
    """Try exact q3-free backends that explicitly target gauss obstruction."""
    if q.q3:
        return None
    if _qubit_quadratic_tensor_obstruction(q) == 0:
        return _sum_bl26_quadratic_tensor_component_scaled(q)

    half_phase_expansion_total = _sum_half_phase_q2_unary_expansion_scaled(q)
    if half_phase_expansion_total is not None:
        return half_phase_expansion_total

    mediator_plan = _build_half_phase_mediator_plan(q)
    if mediator_plan is not None:
        return _evaluate_half_phase_mediator_plan_scaled(
            mediator_plan,
            q.q1,
        )

    generic_mediator_plan = _build_generic_q2_mediator_plan(q)
    if generic_mediator_plan is not None:
        return _evaluate_generic_q2_mediator_plan_scaled(
            generic_mediator_plan,
            q.q1,
        )

    bad_q2_cover = _minimum_bad_q2_vertex_cover(q)
    if _bad_q2_cover_dispatch_allowed(q, bad_q2_cover):
        bad_q2_cover_total = _sum_q3_free_via_bad_q2_cover_scaled(q, cover=bad_q2_cover)
        if bad_q2_cover_total is not None:
            return bad_q2_cover_total

    cluster_plan = _build_q1_cluster_plan(q)
    if cluster_plan is not None:
        return _evaluate_half_phase_cluster_plan_scaled(
            cluster_plan,
            q.q1,
        )

    parity_reduced_total = _sum_half_phase_parity_component_reduction_scaled(q)
    if parity_reduced_total is not None:
        return parity_reduced_total

    return _sum_q3_free_via_nonquadratic_support_scaled(q)


def _sum_q3_free_via_gauss_reduction(q):
    """Unscaled wrapper for ``_sum_q3_free_via_gauss_reduction_scaled``."""
    scaled = _sum_q3_free_via_gauss_reduction_scaled(q)
    if scaled is None:
        return None
    return _scaled_to_complex(scaled)


def _non_half_phase_q2_edge_masks(q) -> list[int]:
    """Return the graph edges whose q2 residue blocks the half-phase backend."""
    if q.q3 or not q.q2:
        return []
    half_q2 = q.mod_q2 // 2 if q.mod_q2 else 0
    edge_masks: list[int] = []
    for (left, right), coeff in q.q2.items():
        residue = int(coeff) % q.mod_q2
        if residue not in (0, half_q2):
            edge_masks.append((1 << left) | (1 << right))
    return edge_masks


def _minimum_bad_q2_vertex_cover_uncached(q) -> list[int]:
    """Exact minimum cover of the non-half-phase q2 edges, heuristic otherwise."""
    edge_masks = _non_half_phase_q2_edge_masks(q)
    if not edge_masks:
        return []
    return _minimum_vertex_cover_from_edge_masks(
        q.n,
        edge_masks,
        exact_size_cutoff=_Q3_VERTEX_COVER_EXACT_SIZE_CUTOFF,
        exact_edge_cutoff=_Q3_VERTEX_COVER_EXACT_EDGE_CUTOFF,
    )


def _minimum_bad_q2_vertex_cover(q) -> list[int]:
    cache_key = _q_structure_key(q)
    cached = _STRUCTURE_BAD_Q2_COVER_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)
    cover = tuple(_minimum_bad_q2_vertex_cover_uncached(q))
    _STRUCTURE_BAD_Q2_COVER_CACHE[cache_key] = cover
    return list(cover)


def _bad_q2_cover_dispatch_allowed(q, cover: Sequence[int] | None = None) -> bool:
    """Return whether the bad-q2 cover branch is structurally promising."""
    if q.q3 or _is_half_phase_q2(q):
        return False
    if q.n < _Q3_FREE_BAD_Q2_COVER_DISPATCH_MIN_VARS:
        return False
    if cover is None:
        cover = _minimum_bad_q2_vertex_cover(q)
    if not cover or len(cover) > _Q3_FREE_BAD_Q2_COVER_MAX_SIZE:
        return False

    max_edges = q.n * (q.n - 1) // 2
    if max_edges <= 0:
        return False
    if len(q.q2) < int(math.ceil(_Q3_FREE_BAD_Q2_COVER_DISPATCH_MIN_DENSITY * max_edges)):
        return False

    bad_support = set()
    for edge_mask in _non_half_phase_q2_edge_masks(q):
        vertices = edge_mask
        while vertices:
            vertex_bit = vertices & -vertices
            bad_support.add(vertex_bit.bit_length() - 1)
            vertices ^= vertex_bit
    return len(bad_support) >= _Q3_FREE_BAD_Q2_COVER_DISPATCH_MIN_SUPPORT_FACTOR * len(cover)


def _sum_q3_free_via_bad_q2_cover_scaled(q, *, cover: Sequence[int] | None = None):
    """Fix a small cover of bad q2 edges, then solve each half-phase branch exactly."""
    if q.q3 or _is_half_phase_q2(q):
        return None

    if cover is None:
        cover = _minimum_bad_q2_vertex_cover(q)
    if not cover or len(cover) > _Q3_FREE_BAD_Q2_COVER_MAX_SIZE:
        return None

    total = _ZERO_SCALED
    for fixed_mask in range(1 << len(cover)):
        fixed_values = [(fixed_mask >> idx) & 1 for idx in range(len(cover))]
        branch_q = _fix_variables(q, cover, fixed_values)
        if not _is_half_phase_q2(branch_q):
            return None
        total = _add_scaled_complex(total, _sum_q3_free_component_scaled(branch_q))
    return total


def _gauss_obstruction(q, structural_obstruction: int = 0) -> int:
    """Return the BL26-style obstruction combining q3 and nonquadratic q1/q2."""
    return max(structural_obstruction, _qubit_quadratic_tensor_obstruction(q))


def _sum_q3_free_via_nonquadratic_support_scaled(q):
    """Branch on a small non-quadratic support, then reduce each branch exactly.

    This is an exact fallback for q3-free kernels that are almost in BL26's
    qubit quadratic-tensor class: after fixing every variable touched by a
    non-quadratic q1/q2 coefficient, the remaining branch should lie entirely in
    the quadratic-tensor class and can be handed to the existing exact q3-free
    summation path instead of the generic feedback-variable solver.
    """
    support = _qubit_quadratic_tensor_obstruction_support(q)
    if not support or len(support) > _Q3_FREE_NONQUADRATIC_BRANCH_MAX_SUPPORT:
        return None

    total = _ZERO_SCALED
    for fixed_mask in range(1 << len(support)):
        fixed_values = [(fixed_mask >> idx) & 1 for idx in range(len(support))]
        branch_q = _fix_variables(q, support, fixed_values)
        if branch_q.q3:
            return None
        if _qubit_quadratic_tensor_obstruction(branch_q):
            return None
        branch_total, _phase_info = _gauss_sum_q3_free_scaled(branch_q)
        total = _add_scaled_complex(total, branch_total)
    return total


def _sum_q3_free_via_nonquadratic_support(q):
    scaled = _sum_q3_free_via_nonquadratic_support_scaled(q)
    if scaled is None:
        return None
    return _scaled_to_complex(scaled)


def _cubic_order_width(q, order):
    if len(order) != q.n:
        raise ValueError(f"Expected elimination order of length {q.n}, received {len(order)}.")
    native_cubic_order_width = _native_symbol("cubic_order_width")
    if native_cubic_order_width is not None:
        return native_cubic_order_width(q.n, q.q2, q.q3, order)

    adjacency_masks = [0] * q.n
    for i, j in q.q2:
        bit_i = 1 << i
        bit_j = 1 << j
        adjacency_masks[i] |= bit_j
        adjacency_masks[j] |= bit_i
    for i, j, k in q.q3:
        bit_i = 1 << i
        bit_j = 1 << j
        bit_k = 1 << k
        adjacency_masks[i] |= bit_j | bit_k
        adjacency_masks[j] |= bit_i | bit_k
        adjacency_masks[k] |= bit_i | bit_j

    remaining_mask = (1 << q.n) - 1
    max_scope = 0
    for var in order:
        var_bit = 1 << var
        if not (remaining_mask & var_bit):
            raise ValueError("Elimination order must contain each variable exactly once.")
        neighbors_mask = adjacency_masks[var] & remaining_mask
        max_scope = max(max_scope, neighbors_mask.bit_count() + 1)

        neighbor_bits = tuple(_iter_mask_bits(neighbors_mask))
        remove_var_mask = ~var_bit
        for left in neighbor_bits:
            adjacency_masks[left] = (
                adjacency_masks[left]
                | (neighbors_mask & ~(1 << left))
            ) & remove_var_mask
        adjacency_masks[var] = 0
        remaining_mask &= remove_var_mask
    return max_scope


def _q3_free_treewidth_order(q, feedback_size, order_hint=None, max_degree=None):
    """
    Return a favorable elimination order for a q3-free component, if any.

    The existing q3-free reducer is exponential in the chosen feedback-set
    size. For strip-like families such as deeper QAOA rings, the q2 graph can
    still have small treewidth even when the feedback set grows linearly.
    """
    if feedback_size <= 1:
        return None

    width_limit = _q3_free_treewidth_width_limit()

    if order_hint is not None:
        hint_order = list(order_hint)
        try:
            width = _cubic_order_width(q, hint_order)
        except ValueError:
            width = width_limit + 1
        if (
            width <= min(_Q3_FREE_ORDER_HINT_MAX_WIDTH, width_limit)
            and _q3_free_treewidth_candidate_is_viable(q, hint_order, width, feedback_size)
        ):
            return hint_order

    adjacency = None
    degeneracy_lower_bound = None
    if q.q2:
        adjacency = [set() for _ in range(q.n)]
        for left, right in q.q2:
            adjacency[left].add(right)
            adjacency[right].add(left)
        degeneracy_lower_bound = _pair_graph_degeneracy(adjacency)
        if (
            degeneracy_lower_bound > width_limit
            or degeneracy_lower_bound >= feedback_size
        ):
            return None

    if max_degree is not None and max_degree <= 4:
        # On sparse strip-like q3-free graphs, min-degree usually lands within
        # one bucket of min-fill at a fraction of the planning cost. This keeps
        # deeper ring-QAOA style amplitudes from spending most of their time on
        # order search when the eventual treewidth DP is already cheap.
        order, width = _min_degree_cubic_order_uncached(q)
        if _q3_free_treewidth_candidate_is_viable(q, order, width, feedback_size):
            return order

    order, width = _min_fill_cubic_order(q)
    if _q3_free_treewidth_candidate_is_viable(q, order, width, feedback_size):
        return order
    separator_order = _pair_graph_separator_order(q)
    if separator_order is not None:
        order, width = separator_order
        if _q3_free_treewidth_candidate_is_viable(q, order, width, feedback_size):
            return order
    return None


def _sum_q3_free_component(
    q,
    *,
    allow_schur_complement: bool = True,
    allow_tensor_contraction: bool = True,
):
    """Sum a connected q3-free component by exact backends on its q2 graph."""
    return _scaled_to_complex(
        _sum_q3_free_component_scaled(
            q,
            allow_schur_complement=allow_schur_complement,
            allow_tensor_contraction=allow_tensor_contraction,
        )
    )


def _sum_q3_free_component_scaled(
    q,
    *,
    allow_schur_complement: bool = True,
    allow_tensor_contraction: bool = True,
):
    """Scaled-complex companion to ``_sum_q3_free_component``."""
    if q.n == 0:
        return _ONE_SCALED
    if q.n <= _Q3_FREE_BRUTE_FORCE_CUTOFF:
        return _make_scaled_complex(_bruteforce_q3_free_sum(q))
    if not q.q2:
        return _product_q1_sum_scaled(q.q1, level=q.level)
    gauss_reduced_total = _sum_q3_free_via_gauss_reduction_scaled(q)
    if gauss_reduced_total is not None:
        return gauss_reduced_total
    binary_total = _sum_binary_phase_quadratic_scaled(q)
    if binary_total is not None:
        return binary_total

    adjacency, edges = _q3_free_graph(q)
    depth, chords = _q3_free_spanning_data(adjacency, edges)
    if not chords:
        return _forest_transfer_sum_scaled(q.q1, adjacency, level=q.level)

    feedback_vars = _select_feedback_vertices(q.n, chords, depth)
    max_degree = max((len(neighbors) for neighbors in adjacency), default=0)
    treewidth_order = _q3_free_treewidth_order(
        q,
        len(feedback_vars),
        max_degree=max_degree,
    )
    if treewidth_order is not None:
        total, _ = _sum_via_treewidth_dp_scaled(q, treewidth_order)
        return total
    prefer_cutset = _q3_free_prefers_locality_preserving_cutset(
        q,
        feedback_size=len(feedback_vars),
        max_degree=max_degree,
        edge_density=_q3_free_edge_density(q),
        allow_tensor_contraction=allow_tensor_contraction,
    )
    if prefer_cutset:
        cutset_conditioned_total = _sum_q3_free_via_cutset_conditioning_scaled(q)
        if cutset_conditioned_total is not None:
            return cutset_conditioned_total
    if (
        allow_schur_complement
        and len(feedback_vars) > _SCHUR_COMPLEMENT_CROSSOVER_FVS
        and _supports_exact_dense_schur(q)
    ):
        schur_total = _schur_complement_q3_free_sum_scaled(q)
        if schur_total is not None:
            return schur_total
    cutset_conditioned_total = None if prefer_cutset else _sum_q3_free_via_cutset_conditioning_scaled(q)
    if cutset_conditioned_total is not None:
        return cutset_conditioned_total

    fixed_pos = {var: idx for idx, var in enumerate(feedback_vars)}
    free_vars = [var for var in range(q.n) if var not in fixed_pos]
    free_index = {var: idx for idx, var in enumerate(free_vars)}
    free_adjacency = [dict() for _ in range(len(free_vars))]
    base_q1 = [q.q1[var] for var in free_vars]

    fixed_linear = [
        (1 << bit, q.q1[var])
        for var, bit in fixed_pos.items()
        if q.q1[var]
    ]
    fixed_to_free = []
    fixed_to_fixed = []

    for i, j, phase in edges:
        bit_i = fixed_pos.get(i)
        bit_j = fixed_pos.get(j)
        if bit_i is not None and bit_j is not None:
            fixed_to_fixed.append((((1 << bit_i) | (1 << bit_j)), phase))
            continue
        if bit_i is not None:
            fixed_to_free.append((1 << bit_i, free_index[j], phase))
            continue
        if bit_j is not None:
            fixed_to_free.append((1 << bit_j, free_index[i], phase))
            continue
        a = free_index[i]
        b = free_index[j]
        free_adjacency[a][b] = phase
        free_adjacency[b][a] = phase

    forest_memo = {}
    total = _ZERO_SCALED
    omega_scaled = _omega_scaled_table(q.level)
    for mask in range(1 << len(feedback_vars)):
        q1_shifted = base_q1[:]
        const_phase = 0

        for bitmask, coeff in fixed_linear:
            if mask & bitmask:
                const_phase = (const_phase + coeff) % q.mod_q1
        for bitmask, idx, phase in fixed_to_free:
            if mask & bitmask:
                q1_shifted[idx] = (q1_shifted[idx] + phase) % q.mod_q1
        for bitmask, phase in fixed_to_fixed:
            if (mask & bitmask) == bitmask:
                const_phase = (const_phase + phase) % q.mod_q1

        key = tuple(q1_shifted)
        forest_total = forest_memo.get(key)
        if forest_total is None:
            forest_total = _forest_transfer_sum_scaled(q1_shifted, free_adjacency, level=q.level)
            forest_memo[key] = forest_total
        total = _add_scaled_complex(
            total,
            _mul_scaled_complex(omega_scaled[const_phase % q.mod_q1], forest_total),
        )

    return total


def _gauss_sum_q3_free(q, *, allow_tensor_contraction: bool = True):
    scaled_total, phase_info = _gauss_sum_q3_free_scaled(
        q,
        allow_tensor_contraction=allow_tensor_contraction,
    )
    return _scaled_to_complex(scaled_total), phase_info


def _gauss_sum_q3_free_scaled(q, *, allow_tensor_contraction: bool = True):
    """Scaled-complex companion to ``_gauss_sum_q3_free``."""
    assert not q.q3, "This function requires a q3-free kernel."

    optimized_q, _changed = _optimize_q3_free_phase(
        q,
        allow_tensor_contraction=allow_tensor_contraction,
    )
    execution_plan = _build_q3_free_execution_plan(
        q=optimized_q,
        allow_tensor_contraction=allow_tensor_contraction,
    )
    return _evaluate_q3_free_execution_plan_scaled(execution_plan), {
        'phase_states': 0,
        'phase_splits': 0,
    }


def _fix_variables(q, fixed_vars, fixed_values, context=None):
    """Fix multiple variables at once and restrict to the remaining free ones."""
    assert len(fixed_vars) == len(fixed_values)
    if not fixed_vars:
        return q

    fixed_pairs = tuple(sorted((var, value % 2) for var, value in zip(fixed_vars, fixed_values)))
    fixed = dict(fixed_pairs)
    assert len(fixed) == len(fixed_pairs)

    if context is not None:
        cache_key = (_q_key(q), fixed_pairs)
        cached = context.fix_variables_cache.get(cache_key)
        if cached is not None:
            return cached

    nf = q.n
    fixed_var_tuple = tuple(int(var) for var, _value in fixed_pairs)
    template_cache_key = (_q_key(q), fixed_var_tuple)
    template = _STRUCTURE_FIX_VARIABLE_TEMPLATE_CACHE.get(template_cache_key)
    if template is None:
        nn = nf - len(fixed)
        gamma = [0] * nf
        free_idx = 0
        for j in range(nf):
            if j in fixed:
                continue
            gamma[j] = 1 << free_idx
            free_idx += 1
        template = (nn, tuple(gamma))
        _STRUCTURE_FIX_VARIABLE_TEMPLATE_CACHE[template_cache_key] = template
    else:
        nn, gamma = template

    shift_mask = 0
    for j, value in fixed_pairs:
        if value:
            shift_mask |= 1 << j
    reduced = _aff_compose_cached(q, shift_mask, list(gamma), nn, context=context)
    if context is not None:
        context.fix_variables_cache[cache_key] = reduced
        return reduced
    return reduced


def _fix_variable(q, k, val, context=None):
    """
    Fix variable k to value val in {0,1}.
    Returns CubicFunction on (q.n - 1) variables.
    """
    return _fix_variables(q, [k], [val], context=context)


def _interaction_graph(q):
    """Primal interaction graph induced by q2 edges and q3 hyperedges."""
    adjacency = [set() for _ in range(q.n)]
    for i, j in q.q2:
        adjacency[i].add(j)
        adjacency[j].add(i)
    for i, j, k in q.q3:
        adjacency[i].update([j, k])
        adjacency[j].update([i, k])
        adjacency[k].update([i, j])
    return adjacency


def _connected_components_on_vertices(
    adjacency: Sequence[set[int]],
    vertices: Sequence[int],
) -> list[tuple[int, ...]]:
    remaining = set(int(vertex) for vertex in vertices)
    components: list[tuple[int, ...]] = []

    while remaining:
        root = remaining.pop()
        stack = [root]
        component = [root]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(tuple(sorted(component)))

    return components


def _pair_graph_degeneracy(adjacency: Sequence[Sequence[int] | dict[int, int] | set[int]]) -> int:
    """Return the exact degeneracy of an undirected pair graph."""
    n = len(adjacency)
    if n == 0:
        return 0

    degrees = [len(neighbors) for neighbors in adjacency]
    heap = [(degree, idx) for idx, degree in enumerate(degrees)]
    heapq.heapify(heap)
    active = [True] * n
    degeneracy = 0

    while heap:
        degree, idx = heapq.heappop(heap)
        if not active[idx] or degree != degrees[idx]:
            continue
        active[idx] = False
        degeneracy = max(degeneracy, degree)
        for neighbor in adjacency[idx]:
            if active[neighbor]:
                degrees[neighbor] -= 1
                heapq.heappush(heap, (degrees[neighbor], neighbor))
    return degeneracy


def _bfs_layers_on_vertices(
    adjacency: Sequence[set[int]],
    start: int,
    vertices: set[int],
) -> tuple[tuple[tuple[int, ...], ...], dict[int, int]]:
    visited = {int(start)}
    current_layer = [int(start)]
    layers: list[tuple[int, ...]] = []
    distances = {int(start): 0}
    depth = 0

    while current_layer:
        layers.append(tuple(sorted(current_layer)))
        next_layer: list[int] = []
        for current in current_layer:
            for neighbor in adjacency[current]:
                if neighbor not in vertices or neighbor in visited:
                    continue
                visited.add(neighbor)
                distances[neighbor] = depth + 1
                next_layer.append(neighbor)
        current_layer = next_layer
        depth += 1

    return tuple(layers), distances


def _farthest_vertex_on_vertices(
    adjacency: Sequence[set[int]],
    start: int,
    vertices: set[int],
) -> int:
    layers, _ = _bfs_layers_on_vertices(adjacency, start, vertices)
    if not layers:
        return int(start)
    return max(layers[-1])


def _min_fill_order_on_subgraph(
    adjacency: Sequence[set[int]],
    vertices: Sequence[int],
) -> list[int]:
    ordered_vertices = tuple(sorted(int(vertex) for vertex in vertices))
    if not ordered_vertices:
        return []
    remap = {vertex: idx for idx, vertex in enumerate(ordered_vertices)}
    q2 = {
        (remap[left], remap[right]): 1
        for left in ordered_vertices
        for right in adjacency[left]
        if left < right and right in remap
    }
    dummy_q = _phase_function_from_parts(
        len(ordered_vertices),
        level=3,
        q0=Fraction(0),
        q1=[0] * len(ordered_vertices),
        q2=q2,
        q3={},
    )
    order, _ = _min_fill_cubic_order(dummy_q)
    return [ordered_vertices[idx] for idx in order]


def _choose_pair_graph_layer_separator(
    adjacency: Sequence[set[int]],
    vertices: Sequence[int],
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]] | None:
    ordered_vertices = tuple(sorted(int(vertex) for vertex in vertices))
    if len(ordered_vertices) < _Q2_SEPARATOR_ORDER_MIN_VARS:
        return None

    vertex_set = set(ordered_vertices)
    seed = min(ordered_vertices, key=lambda vertex: (len(adjacency[vertex]), vertex))
    left = _farthest_vertex_on_vertices(adjacency, seed, vertex_set)
    right = _farthest_vertex_on_vertices(adjacency, left, vertex_set)
    layers, _ = _bfs_layers_on_vertices(adjacency, left, vertex_set)
    if len(layers) < 3:
        return None

    prefix_sizes = [0]
    for layer in layers:
        prefix_sizes.append(prefix_sizes[-1] + len(layer))

    best: tuple[tuple[float, int, int, int], tuple[int, ...], tuple[tuple[int, ...], ...]] | None = None
    total_size = len(ordered_vertices)
    max_separator_size = min(_Q2_SEPARATOR_ORDER_MAX_SEPARATOR, max(8, total_size // 3))

    for span in range(1, _Q2_SEPARATOR_ORDER_MAX_LAYER_SPAN + 1):
        for start_idx in range(1, len(layers) - span):
            stop_idx = start_idx + span
            separator = tuple(sorted(vertex for layer in layers[start_idx:stop_idx] for vertex in layer))
            if not separator or len(separator) > max_separator_size:
                continue
            separator_set = set(separator)
            remaining = tuple(vertex for vertex in ordered_vertices if vertex not in separator_set)
            components = _connected_components_on_vertices(adjacency, remaining)
            if len(components) < 2:
                continue
            largest = max(len(component) for component in components)
            if largest >= total_size:
                continue
            balance = largest / total_size
            if balance > _Q2_SEPARATOR_ORDER_MAX_BALANCE:
                continue
            left_size = prefix_sizes[start_idx]
            right_size = total_size - prefix_sizes[stop_idx]
            score = (
                balance,
                len(separator),
                abs(left_size - right_size),
                max(len(component) for component in components),
            )
            candidate = (score, separator, tuple(sorted(components, key=lambda component: (len(component), component))))
            if best is None or candidate[0] < best[0]:
                best = candidate

    if best is None:
        return None
    return best[1], best[2]


def _nested_dissection_pair_order_from_adjacency(
    adjacency: Sequence[set[int]],
    vertices: Sequence[int],
) -> list[int]:
    ordered_vertices = tuple(sorted(int(vertex) for vertex in vertices))
    if len(ordered_vertices) <= _Q2_SEPARATOR_ORDER_BASE_CASE:
        return _min_fill_order_on_subgraph(adjacency, ordered_vertices)

    separator_info = _choose_pair_graph_layer_separator(adjacency, ordered_vertices)
    if separator_info is None:
        return _min_fill_order_on_subgraph(adjacency, ordered_vertices)

    separator, components = separator_info
    order: list[int] = []
    for component in components:
        order.extend(_nested_dissection_pair_order_from_adjacency(adjacency, component))
    order.extend(separator)
    return order


def _pair_graph_separator_order(q) -> tuple[list[int], int] | None:
    if q.q3 or len(q.q2) == 0 or q.n < _Q2_SEPARATOR_ORDER_MIN_VARS:
        return None

    adjacency = [set() for _ in range(q.n)]
    for left, right in q.q2:
        adjacency[left].add(right)
        adjacency[right].add(left)

    components = _connected_components_on_vertices(adjacency, range(q.n))
    order: list[int] = []
    for component in components:
        order.extend(_nested_dissection_pair_order_from_adjacency(adjacency, component))
    if len(order) != q.n or len(set(order)) != q.n:
        return None
    width = _cubic_order_width(q, order)
    return order, width


def _min_fill_cubic_order_uncached(q):
    """
    Heuristic elimination order for low-treewidth cubic DP.

    Returns the order and the maximum factor scope size encountered by the
    corresponding variable elimination schedule.
    """
    if q.n == 0:
        return [], 0
    native_min_fill_cubic_order = _native_symbol("min_fill_cubic_order")
    if native_min_fill_cubic_order is not None:
        return native_min_fill_cubic_order(q.n, q.q2, q.q3)

    adjacency_masks = [0] * q.n
    for i, j in q.q2:
        bit_i = 1 << i
        bit_j = 1 << j
        adjacency_masks[i] |= bit_j
        adjacency_masks[j] |= bit_i
    for i, j, k in q.q3:
        bit_i = 1 << i
        bit_j = 1 << j
        bit_k = 1 << k
        adjacency_masks[i] |= bit_j | bit_k
        adjacency_masks[j] |= bit_i | bit_k
        adjacency_masks[k] |= bit_i | bit_j

    remaining_mask = (1 << q.n) - 1
    order = []
    max_scope = 1

    while remaining_mask:
        best_var = -1
        best_score = None
        best_neighbors_mask = 0
        for var in _iter_mask_bits(remaining_mask):
            neighbors_mask = adjacency_masks[var] & remaining_mask
            remaining_neighbors = neighbors_mask
            fill = 0
            while remaining_neighbors:
                left_bit = remaining_neighbors & -remaining_neighbors
                left = left_bit.bit_length() - 1
                remaining_neighbors ^= left_bit
                fill += (remaining_neighbors & ~adjacency_masks[left]).bit_count()
            score = (fill, neighbors_mask.bit_count(), var)
            if best_score is None or score < best_score:
                best_var = var
                best_score = score
                best_neighbors_mask = neighbors_mask

        order.append(best_var)
        max_scope = max(max_scope, best_neighbors_mask.bit_count() + 1)

        neighbor_bits = tuple(_iter_mask_bits(best_neighbors_mask))
        remove_var_mask = ~(1 << best_var)
        for left in neighbor_bits:
            adjacency_masks[left] = (
                adjacency_masks[left]
                | (best_neighbors_mask & ~(1 << left))
            ) & remove_var_mask
        adjacency_masks[best_var] = 0
        remaining_mask &= remove_var_mask

    return order, max_scope


def _min_degree_cubic_order_uncached(q):
    """Cheap elimination order based only on the current graph degree."""
    if q.n == 0:
        return [], 0
    native_min_degree_cubic_order = _native_symbol("min_degree_cubic_order")
    if native_min_degree_cubic_order is not None:
        return native_min_degree_cubic_order(q.n, q.q2, q.q3)

    adjacency_masks = [0] * q.n
    for i, j in q.q2:
        bit_i = 1 << i
        bit_j = 1 << j
        adjacency_masks[i] |= bit_j
        adjacency_masks[j] |= bit_i
    for i, j, k in q.q3:
        bit_i = 1 << i
        bit_j = 1 << j
        bit_k = 1 << k
        adjacency_masks[i] |= bit_j | bit_k
        adjacency_masks[j] |= bit_i | bit_k
        adjacency_masks[k] |= bit_i | bit_j

    remaining_mask = (1 << q.n) - 1
    order = []
    max_scope = 1

    while remaining_mask:
        best_var = -1
        best_degree = None
        best_neighbors_mask = 0
        for var in _iter_mask_bits(remaining_mask):
            neighbors_mask = adjacency_masks[var] & remaining_mask
            degree = neighbors_mask.bit_count()
            if best_degree is None or degree < best_degree or (degree == best_degree and var < best_var):
                best_var = var
                best_degree = degree
                best_neighbors_mask = neighbors_mask

        order.append(best_var)
        max_scope = max(max_scope, best_neighbors_mask.bit_count() + 1)

        neighbor_bits = tuple(_iter_mask_bits(best_neighbors_mask))
        remove_var_mask = ~(1 << best_var)
        for left in neighbor_bits:
            adjacency_masks[left] = (
                adjacency_masks[left]
                | (best_neighbors_mask & ~(1 << left))
            ) & remove_var_mask
        adjacency_masks[best_var] = 0
        remaining_mask &= remove_var_mask

    return order, max_scope


def _min_fill_cubic_order(q):
    cache_key = _q_structure_key(q)
    cached = _STRUCTURE_MIN_FILL_CACHE.get(cache_key)
    if cached is not None:
        order, width = cached
        return list(order), width
    order, width = _min_fill_cubic_order_uncached(q)
    cached = (tuple(order), width)
    _STRUCTURE_MIN_FILL_CACHE[cache_key] = cached
    order, width = cached
    return list(order), width


def _factor_table_multiply(left, right):
    return [left[idx] * right[idx] for idx in range(len(left))]


def _factor_table_multiply_scaled(left, right):
    return [_mul_scaled_complex(left[idx], right[idx]) for idx in range(len(left))]


def _project_assignment_bits(assignment, positions):
    idx = 0
    for out_pos, in_pos in enumerate(positions):
        idx |= ((assignment >> in_pos) & 1) << out_pos
    return idx


def _combine_factor(factors, scope, table):
    if len(scope) == 0:
        return table[0]
    existing = factors.get(scope)
    if existing is None:
        factors[scope] = table
    else:
        factors[scope] = _factor_table_multiply(existing, table)
    return 1.0 + 0j


def _combine_factor_scaled(factors, scope, table):
    if len(scope) == 0:
        return table[0]
    existing = factors.get(scope)
    if existing is None:
        factors[scope] = table
    else:
        factors[scope] = _factor_table_multiply_scaled(existing, table)
    return _ONE_SCALED


def _factor_table_multiply_scaled_batch(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise multiplication for batched scaled factor tables."""
    left_values, left_exponents = left
    right_values, right_exponents = right
    return _mul_scaled_complex_arrays(
        left_values,
        left_exponents,
        right_values,
        right_exponents,
    )


def _combine_factor_scaled_batch(
    factors: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]],
    scope: tuple[int, ...],
    table_values: np.ndarray,
    table_exponents: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched companion to ``_combine_factor_scaled``."""
    if len(scope) == 0:
        return table_values[:, 0], table_exponents[:, 0]

    existing = factors.get(scope)
    if existing is None:
        factors[scope] = (table_values, table_exponents)
    else:
        factors[scope] = _factor_table_multiply_scaled_batch(
            existing,
            (table_values, table_exponents),
        )
    return _scaled_arrays_from_constant(_ONE_SCALED, (batch_size,))


def _sum_factor_tables_scaled_batch(
    n_vars: int,
    factors: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]],
    order: Sequence[int],
    *,
    scalar: tuple[np.ndarray, np.ndarray],
) -> tuple[list[ScaledComplex], int]:
    """Batch exact scaled bucket elimination over shared factor scopes."""
    scalar_values, scalar_exponents = scalar
    scalar_values = np.asarray(scalar_values, dtype=np.complex128).copy()
    scalar_exponents = np.asarray(scalar_exponents, dtype=np.int64).copy()
    batch_size = len(scalar_values)
    factors = {
        scope: (
            np.asarray(values, dtype=np.complex128).copy(),
            np.asarray(exponents, dtype=np.int64).copy(),
        )
        for scope, (values, exponents) in factors.items()
    }
    max_scope = 0

    for var in order:
        bucket_scopes = [scope for scope in factors if var in scope]
        if not bucket_scopes:
            scalar_exponents = scalar_exponents + 2
            max_scope = max(max_scope, 1)
            continue

        bucket = [(scope, factors.pop(scope)) for scope in bucket_scopes]
        union_scope = tuple(sorted({vertex for scope, _ in bucket for vertex in scope}))
        max_scope = max(max_scope, len(union_scope))

        var_pos = union_scope.index(var)
        new_scope = tuple(vertex for vertex in union_scope if vertex != var)
        positions = [
            tuple(union_scope.index(vertex) for vertex in scope)
            for scope, _ in bucket
        ]

        table_size = 1 << len(new_scope)
        new_values = np.empty((batch_size, table_size), dtype=np.complex128)
        new_exponents = np.empty((batch_size, table_size), dtype=np.int64)
        for reduced_assignment in range(table_size):
            total_values, total_exponents = _scaled_arrays_from_constant(_ZERO_SCALED, (batch_size,))
            for fixed_value in [0, 1]:
                full_assignment = (
                    (reduced_assignment & ((1 << var_pos) - 1))
                    | (fixed_value << var_pos)
                    | ((reduced_assignment >> var_pos) << (var_pos + 1))
                )
                weight_values, weight_exponents = _scaled_arrays_from_constant(_ONE_SCALED, (batch_size,))
                for (_scope, (table_values, table_exponents)), pos in zip(bucket, positions):
                    assignment_index = _project_assignment_bits(full_assignment, pos)
                    weight_values, weight_exponents = _mul_scaled_complex_arrays(
                        weight_values,
                        weight_exponents,
                        table_values[:, assignment_index],
                        table_exponents[:, assignment_index],
                    )
                total_values, total_exponents = _add_scaled_complex_arrays(
                    total_values,
                    total_exponents,
                    weight_values,
                    weight_exponents,
                )
            new_values[:, reduced_assignment] = total_values
            new_exponents[:, reduced_assignment] = total_exponents

        factor_values, factor_exponents = _combine_factor_scaled_batch(
            factors,
            new_scope,
            new_values,
            new_exponents,
            batch_size=batch_size,
        )
        scalar_values, scalar_exponents = _mul_scaled_complex_arrays(
            scalar_values,
            scalar_exponents,
            factor_values,
            factor_exponents,
        )

    assert not factors, "All variables should be eliminated by the supplied order."
    return [
        (complex(value), int(half_pow2_exp))
        for value, half_pow2_exp in zip(scalar_values, scalar_exponents)
    ], max_scope


def _sum_factor_tables_scaled(
    n_vars: int,
    factors,
    order: Sequence[int],
    *,
    scalar: ScaledComplex = _ONE_SCALED,
):
    """Exact scaled bucket elimination over generic binary factor tables."""
    if _schur_native is not None:
        try:
            value, half_pow2_exp, max_scope = _schur_native.sum_factor_tables_scaled(
                n_vars,
                factors,
                tuple(int(var) for var in order),
                scalar,
            )
            return (complex(value), int(half_pow2_exp)), int(max_scope)
        except Exception:
            pass

    factors = dict(factors)
    scalar = scalar
    max_scope = 0

    for var in order:
        bucket_scopes = [scope for scope in factors if var in scope]
        if not bucket_scopes:
            scalar = _scale_scaled_complex(scalar, 2)
            max_scope = max(max_scope, 1)
            continue

        bucket = [(scope, factors.pop(scope)) for scope in bucket_scopes]
        union_scope = tuple(sorted({vertex for scope, _ in bucket for vertex in scope}))
        max_scope = max(max_scope, len(union_scope))

        var_pos = union_scope.index(var)
        new_scope = tuple(vertex for vertex in union_scope if vertex != var)
        positions = [
            tuple(union_scope.index(vertex) for vertex in scope)
            for scope, _ in bucket
        ]

        new_table = [_ZERO_SCALED] * (1 << len(new_scope))
        for reduced_assignment in range(1 << len(new_scope)):
            total = _ZERO_SCALED
            for fixed_value in [0, 1]:
                full_assignment = (
                    (reduced_assignment & ((1 << var_pos) - 1))
                    | (fixed_value << var_pos)
                    | ((reduced_assignment >> var_pos) << (var_pos + 1))
                )
                weight = _ONE_SCALED
                for (_, table), pos in zip(bucket, positions):
                    weight = _mul_scaled_complex(
                        weight,
                        table[_project_assignment_bits(full_assignment, pos)],
                    )
                total = _add_scaled_complex(total, weight)
            new_table[reduced_assignment] = total

        scalar = _mul_scaled_complex(
            scalar,
            _combine_factor_scaled(factors, new_scope, new_table),
        )

    assert not factors, "All variables should be eliminated by the supplied order."
    return scalar, max_scope


def _build_native_q3_free_treewidth_plan(
    *,
    n_vars: int,
    level: int,
    q2: dict[tuple[int, int], int],
    order: Sequence[int],
) -> object | None:
    """Build a reusable native q3-free treewidth plan when available."""
    if _schur_native is None or not hasattr(_schur_native, "build_q3_free_treewidth_plan"):
        return None
    try:
        return _schur_native.build_q3_free_treewidth_plan(
            int(n_vars),
            int(level),
            q2,
            tuple(int(var) for var in order),
        )
    except Exception:
        return None


def _sum_q3_free_treewidth_dp_scaled_batch(
    *,
    n_vars: int,
    level: int,
    q1_batch: np.ndarray,
    q2: dict[tuple[int, int], int],
    order: Sequence[int],
    native_plan: object | None = None,
) -> list[ScaledComplex]:
    """Evaluate many q3-free treewidth-DP sums sharing the same q2/order."""
    q1_batch = np.asarray(q1_batch, dtype=np.int64)
    if len(q1_batch) == 0:
        return []
    q1_batch = np.ascontiguousarray(np.remainder(q1_batch, 1 << int(level)), dtype=np.int64)

    if native_plan is None:
        native_plan = _build_native_q3_free_treewidth_plan(
            n_vars=n_vars,
            level=level,
            q2=q2,
            order=order,
        )
    if (
        native_plan is not None
        and _schur_native is not None
        and hasattr(_schur_native, "sum_q3_free_treewidth_preplanned_batch_scaled_array")
    ):
        try:
            native_rows = _schur_native.sum_q3_free_treewidth_preplanned_batch_scaled_array(
                native_plan,
                q1_batch,
            )
            return [
                (complex(value), int(half_pow2_exp))
                for value, half_pow2_exp, _max_scope in native_rows
            ]
        except Exception:
            pass

    if (
        native_plan is not None
        and _schur_native is not None
        and hasattr(_schur_native, "sum_q3_free_treewidth_preplanned_batch_scaled")
    ):
        try:
            native_rows = _schur_native.sum_q3_free_treewidth_preplanned_batch_scaled(
                native_plan,
                q1_batch.tolist(),
            )
            return [
                (complex(value), int(half_pow2_exp))
                for value, half_pow2_exp, _max_scope in native_rows
            ]
        except Exception:
            pass

    if (
        _schur_native is not None
        and hasattr(_schur_native, "sum_q3_free_treewidth_batch_scaled")
    ):
        try:
            native_rows = _schur_native.sum_q3_free_treewidth_batch_scaled(
                int(n_vars),
                int(level),
                q1_batch.tolist(),
                q2,
                tuple(int(var) for var in order),
            )
            return [
                (complex(value), int(half_pow2_exp))
                for value, half_pow2_exp, _max_scope in native_rows
            ]
        except Exception:
            pass

    totals: list[ScaledComplex] = []
    for q1_local in q1_batch:
        component_q = _phase_function_from_parts(
            int(n_vars),
            level=int(level),
            q0=Fraction(0),
            q1=q1_local.tolist(),
            q2=q2,
            q3={},
        )
        total, _ = _sum_via_treewidth_dp_scaled(component_q, list(order))
        totals.append(total)
    return totals


def _q3_free_component_plan_width_hint(component_plan: _Q3FreeConstraintComponentPlan) -> int:
    if component_plan.cutset_plan is not None:
        return int(component_plan.cutset_plan.remaining_width)
    if component_plan.backend == "constant":
        return 0
    if component_plan.backend == "forest":
        return 1 if component_plan.variables else 0
    if component_plan.backend == "treewidth" and component_plan.order is not None:
        dummy_q = _phase_function_from_parts(
            len(component_plan.variables),
            level=component_plan.level,
            q0=Fraction(0),
            q1=[0] * len(component_plan.variables),
            q2=component_plan.q2,
            q3={},
        )
        return int(_treewidth_order_width(dummy_q, component_plan.order))
    dummy_q = _phase_function_from_parts(
        len(component_plan.variables),
        level=component_plan.level,
        q0=Fraction(0),
        q1=[0] * len(component_plan.variables),
        q2=component_plan.q2,
        q3={},
    )
    _order, width = _min_fill_cubic_order(dummy_q)
    return int(width)


def _q3_free_component_plan_work_hint(component_plan: _Q3FreeConstraintComponentPlan) -> int:
    if component_plan.cutset_plan is not None:
        return int(component_plan.cutset_plan.estimated_total_work)
    if component_plan.backend == "constant":
        return 1
    if component_plan.backend == "forest":
        return max(1, len(component_plan.variables))
    if component_plan.backend == "treewidth" and component_plan.order is not None:
        dummy_q = _phase_function_from_parts(
            len(component_plan.variables),
            level=component_plan.level,
            q0=Fraction(0),
            q1=[0] * len(component_plan.variables),
            q2=component_plan.q2,
            q3={},
        )
        return max(1, _estimate_treewidth_dp_work(dummy_q, component_plan.order))
    dummy_q = _phase_function_from_parts(
        len(component_plan.variables),
        level=component_plan.level,
        q0=Fraction(0),
        q1=[0] * len(component_plan.variables),
        q2=component_plan.q2,
        q3={},
    )
    order, _width = _min_fill_cubic_order(dummy_q)
    return max(1, _estimate_treewidth_dp_work(dummy_q, order))


def _q3_free_cutset_plan_generic_penalty(
    plan: _Q3FreeCutsetConditioningPlan | None,
) -> int:
    if plan is None:
        return 1 << 20
    penalty = int(plan.remaining_backend == "generic")
    for component_plan in plan.remaining_components:
        penalty += _q3_free_component_plan_generic_penalty(component_plan)
    return penalty


def _q3_free_component_plan_generic_penalty(
    component_plan: _Q3FreeConstraintComponentPlan,
) -> int:
    penalty = int(component_plan.backend == "generic")
    if component_plan.cutset_plan is not None:
        penalty += _q3_free_cutset_plan_generic_penalty(component_plan.cutset_plan)
    return penalty


def _q3_free_tensor_slice_hint(q: PhaseFunction) -> tuple[int, ...]:
    """Return preferred cutset variables from a sliced tensor-contraction plan."""
    _cfg = _get_solver_config()
    if (
        q.n < _cfg.tensor_hint_min_vars
        or q.n > _cfg.tensor_hint_max_vars
        or not q.q2
        or not _kahypar_available()
    ):
        return ()

    cache_key = (
        _q_structure_key(q),
        int(_cfg.tensor_hint_target_width),
        int(_cfg.tensor_hint_max_repeats),
        float(_cfg.tensor_hint_max_time),
        bool(_kahypar_available()),
    )
    cached = _STRUCTURE_Q3_FREE_TENSOR_HINT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    qtn = _get_quimb_tensor_module()
    if qtn is None:
        return ()
    try:
        import cotengra as ctg
    except Exception:
        return ()

    _scalar, factors = _build_cubic_factors(q)
    if not factors:
        return ()

    del qtn
    ordered_scopes = [scope for scope, _table in sorted(factors.items())]
    inputs = [tuple(f"v{var}" for var in scope) for scope in ordered_scopes]
    size_dict = {f"v{var}": 2 for var in range(q.n)}

    optimizer = ctg.HyperOptimizer(
        methods=["kahypar"],
        minimize="flops",
        max_repeats=int(_cfg.tensor_hint_max_repeats),
        max_time=float(_cfg.tensor_hint_max_time),
        parallel=False,
        slicing_reconf_opts={
            "target_size": 1 << int(_cfg.tensor_hint_target_width),
        },
        reconf_opts={},
        progbar=False,
    )

    try:
        tree = ctg.array_contract_tree(
            inputs,
            output=(),
            size_dict=size_dict,
            optimize=optimizer,
            canonicalize=False,
        )
    except Exception:
        return ()

    sliced_inds = getattr(tree, "sliced_inds", None) or {}
    hint: list[int] = []
    for ind in sliced_inds:
        if isinstance(ind, str) and ind.startswith("v") and ind[1:].isdigit():
            hint.append(int(ind[1:]))
    result = tuple(sorted(set(hint)))
    _STRUCTURE_Q3_FREE_TENSOR_HINT_CACHE[cache_key] = result
    return result


def _candidate_q3_free_cutset_vertices(
    adjacency: Sequence[set[int]],
    *,
    preferred: set[int] | None = None,
    max_candidates: int = _Q3_FREE_CUTSET_CANDIDATE_POOL,
) -> tuple[int, ...]:
    preferred = set() if preferred is None else preferred
    scored: list[tuple[int, int, int, int, int]] = []
    for var, neighbors in enumerate(adjacency):
        triangle_count = 0
        ordered_neighbors = tuple(sorted(neighbors))
        for idx, left in enumerate(ordered_neighbors):
            left_neighbors = adjacency[left]
            for right in ordered_neighbors[idx + 1 :]:
                if right in left_neighbors:
                    triangle_count += 1
        scored.append(
            (
                int(var in preferred),
                len(neighbors),
                sum(len(adjacency[neighbor]) for neighbor in neighbors),
                triangle_count,
                var,
            )
        )
    scored.sort(reverse=True)
    return tuple(var for *_score, var in scored[: min(len(scored), int(max_candidates))])


def _merge_q3_free_cutset_candidate_orders(
    *candidate_orders: Sequence[int],
    max_candidates: int,
) -> tuple[int, ...]:
    merged: list[int] = []
    seen: set[int] = set()
    for order in candidate_orders:
        for var in order:
            if var in seen:
                continue
            merged.append(int(var))
            seen.add(int(var))
            if len(merged) >= int(max_candidates):
                return tuple(merged)
    return tuple(merged)


def _separator_ranked_q3_free_cutset_vertices(
    adjacency: Sequence[set[int]],
    *,
    preferred: set[int] | None = None,
    max_candidates: int = _Q3_FREE_ONE_SHOT_CUTSET_CANDIDATE_POOL,
) -> tuple[int, ...]:
    preferred = set() if preferred is None else {int(var) for var in preferred}
    all_vertices = tuple(range(len(adjacency)))
    if not all_vertices:
        return ()

    ranked: list[int] = []
    seen: set[int] = set()
    component_heap: list[tuple[int, tuple[int, ...]]] = [
        (-len(component), tuple(component))
        for component in _connected_components_on_vertices(adjacency, all_vertices)
        if len(component) >= _Q2_SEPARATOR_ORDER_MIN_VARS
    ]
    heapq.heapify(component_heap)

    while component_heap and len(ranked) < int(max_candidates):
        _neg_size, component = heapq.heappop(component_heap)
        separator_info = _choose_pair_graph_layer_separator(adjacency, component)
        if separator_info is None:
            continue
        separator, components = separator_info
        ordered_separator = sorted(
            separator,
            key=lambda var: (
                int(var in preferred),
                len(adjacency[var]),
                sum(len(adjacency[neighbor]) for neighbor in adjacency[var]),
                -int(var),
            ),
            reverse=True,
        )
        for var in ordered_separator:
            if var in seen:
                continue
            ranked.append(int(var))
            seen.add(int(var))
            if len(ranked) >= int(max_candidates):
                break
        for subcomponent in components:
            if len(subcomponent) >= _Q2_SEPARATOR_ORDER_MIN_VARS:
                heapq.heappush(component_heap, (-len(subcomponent), tuple(subcomponent)))

    return tuple(ranked)


def _build_q3_free_cutset_residue_data(
    q: PhaseFunction,
    cutset_vars: tuple[int, ...],
    remaining_vars: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q2_lift = q.mod_q1 // q.mod_q2 if q.mod_q2 else 0
    cutset_index = {var: idx for idx, var in enumerate(cutset_vars)}
    remaining_index = {var: idx for idx, var in enumerate(remaining_vars)}
    cutset_remaining = np.zeros((len(cutset_vars), len(remaining_vars)), dtype=np.int64)
    cutset_cutset_left: list[int] = []
    cutset_cutset_right: list[int] = []
    cutset_cutset_residue: list[int] = []

    for (left, right), coeff in q.q2.items():
        residue = (q2_lift * coeff) % q.mod_q1
        if not residue:
            continue
        if left in cutset_index and right in remaining_index:
            cutset_remaining[cutset_index[left], remaining_index[right]] = (
                cutset_remaining[cutset_index[left], remaining_index[right]] + residue
            ) % q.mod_q1
        elif right in cutset_index and left in remaining_index:
            cutset_remaining[cutset_index[right], remaining_index[left]] = (
                cutset_remaining[cutset_index[right], remaining_index[left]] + residue
            ) % q.mod_q1
        elif left in cutset_index and right in cutset_index:
            cutset_cutset_left.append(cutset_index[left])
            cutset_cutset_right.append(cutset_index[right])
            cutset_cutset_residue.append(residue)

    return (
        cutset_remaining,
        _as_int64_array(cutset_cutset_left),
        _as_int64_array(cutset_cutset_right),
        _as_int64_array(cutset_cutset_residue),
    )


def _evaluate_q3_free_cutset_candidate(
    q: PhaseFunction,
    cutset_vars: tuple[int, ...],
    *,
    remaining_universe: tuple[int, ...] | None = None,
    prioritize_width: bool = False,
    target_remaining_width: int | None = None,
    allow_generic_remaining: bool = False,
    prefer_one_shot_slicing: bool = False,
    ) -> _Q3FreeCutsetCandidateEvaluation | None:
    def make_score(
        viable_flag: int,
        *,
        target_miss: int,
        width: int,
        work: int,
        generic_penalty: int = 0,
    ) -> tuple[int, ...]:
        if prioritize_width:
            if prefer_one_shot_slicing:
                return (
                    viable_flag,
                    generic_penalty,
                    target_miss,
                    width,
                    work,
                    len(cutset_vars),
                )
            return (viable_flag, target_miss, width, work, len(cutset_vars))
        if prefer_one_shot_slicing:
            return (viable_flag, generic_penalty, work, width, len(cutset_vars))
        return (viable_flag, work, width, len(cutset_vars))

    if remaining_universe is None:
        remaining_universe = tuple(range(q.n))
    cutset_set = set(cutset_vars)
    remaining_vars = tuple(var for var in remaining_universe if var not in cutset_set)
    if not remaining_vars:
        return None

    cutset_remaining, cutset_cutset_left, cutset_cutset_right, cutset_cutset_residue = (
        _build_q3_free_cutset_residue_data(q, cutset_vars, remaining_vars)
    )
    branch_count = 1 << len(cutset_vars)
    remaining_q = _component_restriction(q, remaining_vars)

    def summarize_generic_remaining() -> tuple[tuple[int, ...], int, int, int]:
        component_sets = detect_factorization(remaining_q)
        covered = set().union(*component_sets) if component_sets else set()
        isolated_vars = tuple(sorted(set(range(remaining_q.n)) - covered))
        component_width = 0
        component_work = 0
        generic_penalty = 1
        for component in component_sets:
            component_q = _component_restriction(remaining_q, component)
            order, width = _min_fill_cubic_order(component_q)
            component_width = max(component_width, int(width))
            component_work += max(1, int(_estimate_treewidth_dp_work(component_q, order)))
            component_adjacency, component_edges = _q3_free_graph(component_q)
            component_depth, component_chords = _q3_free_spanning_data(component_adjacency, component_edges)
            if _q3_free_treewidth_order(
                component_q,
                len(_select_feedback_vertices(component_q.n, component_chords, component_depth)),
                order_hint=order,
                max_degree=max((len(neighbors) for neighbors in component_adjacency), default=0),
            ) is None:
                generic_penalty += 1
        return isolated_vars, component_width, max(1, component_work), generic_penalty

    if not remaining_q.q2:
        target_miss = 0
        plan = _Q3FreeCutsetConditioningPlan(
            level=q.level,
            cutset_vars=cutset_vars,
            remaining_vars=remaining_vars,
            remaining_backend="product",
            remaining_q2={},
            remaining_order=(),
            cutset_remaining_q2_residue=cutset_remaining,
            cutset_cutset_left=cutset_cutset_left,
            cutset_cutset_right=cutset_cutset_right,
            cutset_cutset_residue=cutset_cutset_residue,
            remaining_width=0,
            estimated_total_work=branch_count,
        )
        return _Q3FreeCutsetCandidateEvaluation(
            cutset_vars=cutset_vars,
            plan=plan,
            viable=True,
            score=make_score(0, target_miss=target_miss, width=0, work=branch_count),
        )

    candidate_order, width = _min_fill_cubic_order(remaining_q)
    work = _estimate_treewidth_dp_work(remaining_q, candidate_order)
    effective_work = branch_count * work
    reduced_adjacency, reduced_edges = _q3_free_graph(remaining_q)
    reduced_depth, reduced_chords = _q3_free_spanning_data(reduced_adjacency, reduced_edges)
    reduced_feedback = _select_feedback_vertices(remaining_q.n, reduced_chords, reduced_depth)
    viable_order = _q3_free_treewidth_order(
        remaining_q,
        len(reduced_feedback),
        order_hint=candidate_order,
        max_degree=max((len(neighbors) for neighbors in reduced_adjacency), default=0),
    )
    if viable_order is None:
        if allow_generic_remaining:
            isolated_vars, component_width, component_work, generic_penalty = summarize_generic_remaining()
            generic_work = branch_count * component_work
            plan = _Q3FreeCutsetConditioningPlan(
                level=q.level,
                cutset_vars=cutset_vars,
                remaining_vars=remaining_vars,
                remaining_backend="generic",
                remaining_q2=remaining_q.q2,
                remaining_order=(),
                cutset_remaining_q2_residue=cutset_remaining,
                cutset_cutset_left=cutset_cutset_left,
                cutset_cutset_right=cutset_cutset_right,
                cutset_cutset_residue=cutset_cutset_residue,
                remaining_isolated_vars=tuple(int(var) for var in isolated_vars),
                remaining_components=(),
                remaining_width=component_width,
                estimated_total_work=generic_work,
            )
            target_miss = int(
                target_remaining_width is not None
                and component_width > int(target_remaining_width)
            )
            return _Q3FreeCutsetCandidateEvaluation(
                cutset_vars=cutset_vars,
                plan=plan,
                viable=True,
                score=make_score(
                    0,
                    target_miss=target_miss,
                    width=component_width,
                    work=generic_work,
                    generic_penalty=generic_penalty,
                ),
            )
        miss_width = (
            max(width, int(target_remaining_width or 0))
            if prioritize_width
            else width
        )
        return _Q3FreeCutsetCandidateEvaluation(
            cutset_vars=cutset_vars,
            plan=None,
            viable=False,
            score=make_score(
                1,
                target_miss=1,
                width=miss_width,
                work=effective_work,
            ),
        )

    viable_width = _treewidth_order_width(remaining_q, viable_order)
    target_miss = int(
        target_remaining_width is not None
        and viable_width > int(target_remaining_width)
    )
    plan = _Q3FreeCutsetConditioningPlan(
        level=q.level,
        cutset_vars=cutset_vars,
        remaining_vars=remaining_vars,
        remaining_backend="treewidth",
        remaining_q2=remaining_q.q2,
        remaining_order=tuple(int(var) for var in viable_order),
        cutset_remaining_q2_residue=cutset_remaining,
        cutset_cutset_left=cutset_cutset_left,
        cutset_cutset_right=cutset_cutset_right,
        cutset_cutset_residue=cutset_cutset_residue,
        remaining_width=viable_width,
        estimated_total_work=effective_work,
    )
    return _Q3FreeCutsetCandidateEvaluation(
        cutset_vars=cutset_vars,
        plan=plan,
        viable=True,
        score=make_score(
            0,
            target_miss=target_miss,
            width=viable_width,
            work=effective_work,
        ),
    )


def _finalize_q3_free_cutset_conditioning_plan(
    plan: _Q3FreeCutsetConditioningPlan,
    *,
    prefer_one_shot_slicing: bool = False,
) -> _Q3FreeCutsetConditioningPlan:
    """Fill in generic remaining-component plans only for the chosen cutset."""
    if (
        plan.remaining_backend != "generic"
        or plan.remaining_components
        or (not plan.remaining_q2 and not plan.remaining_isolated_vars)
    ):
        return plan

    remaining_q = _phase_function_from_parts(
        len(plan.remaining_vars),
        level=plan.level,
        q0=Fraction(0),
        q1=[0] * len(plan.remaining_vars),
        q2=plan.remaining_q2,
        q3={},
    )
    isolated_vars, component_plans = _plan_q3_free_constraint_components(
        remaining_q,
        0,
        allow_tensor_contraction=True,
        prefer_reusable_decomposition=False,
        prefer_one_shot_slicing=prefer_one_shot_slicing,
    )
    component_width = max(
        (_q3_free_component_plan_width_hint(component_plan) for component_plan in component_plans),
        default=0,
    )
    component_work = max(
        1,
        sum(_q3_free_component_plan_work_hint(component_plan) for component_plan in component_plans),
    )
    branch_count = 1 << len(plan.cutset_vars)
    return _Q3FreeCutsetConditioningPlan(
        level=plan.level,
        cutset_vars=plan.cutset_vars,
        remaining_vars=plan.remaining_vars,
        remaining_backend=plan.remaining_backend,
        remaining_q2=plan.remaining_q2,
        remaining_order=plan.remaining_order,
        cutset_remaining_q2_residue=plan.cutset_remaining_q2_residue,
        cutset_cutset_left=plan.cutset_cutset_left,
        cutset_cutset_right=plan.cutset_cutset_right,
        cutset_cutset_residue=plan.cutset_cutset_residue,
        native_treewidth_plan=plan.native_treewidth_plan,
        remaining_isolated_vars=tuple(int(var) for var in isolated_vars),
        remaining_components=tuple(component_plans),
        remaining_width=max(plan.remaining_width, component_width),
        estimated_total_work=max(plan.estimated_total_work, branch_count * component_work),
    )


def _attach_q3_free_cutset_runtime_cache(
    plan: _Q3FreeCutsetConditioningPlan,
) -> _Q3FreeCutsetConditioningPlan:
    """Attach reusable branch-side runtime arrays to a cutset plan."""
    if plan.branch_bits is not None:
        return plan

    cutset_size = len(plan.cutset_vars)
    branch_count = 1 << cutset_size
    branch_masks = np.arange(branch_count, dtype=np.uint64)
    branch_bits = _branch_assignment_bits(branch_masks, cutset_size, np).astype(np.int64)

    if plan.cutset_cutset_residue.size:
        branch_pair_residue = np.zeros(branch_count, dtype=np.int64)
        for left, right, residue in zip(
            plan.cutset_cutset_left,
            plan.cutset_cutset_right,
            plan.cutset_cutset_residue,
        ):
            branch_pair_residue = (
                branch_pair_residue
                + int(residue) * branch_bits[:, int(left)] * branch_bits[:, int(right)]
            ) % (1 << int(plan.level))
    else:
        branch_pair_residue = np.zeros(branch_count, dtype=np.int64)

    if plan.cutset_remaining_q2_residue.size:
        branch_remaining_shift = (
            branch_bits @ np.asarray(plan.cutset_remaining_q2_residue, dtype=np.int64)
        ) % (1 << int(plan.level))
    else:
        branch_remaining_shift = np.zeros(
            (branch_count, len(plan.remaining_vars)),
            dtype=np.int64,
        )

    return _Q3FreeCutsetConditioningPlan(
        level=plan.level,
        cutset_vars=plan.cutset_vars,
        remaining_vars=plan.remaining_vars,
        remaining_backend=plan.remaining_backend,
        remaining_q2=plan.remaining_q2,
        remaining_order=plan.remaining_order,
        cutset_remaining_q2_residue=plan.cutset_remaining_q2_residue,
        cutset_cutset_left=plan.cutset_cutset_left,
        cutset_cutset_right=plan.cutset_cutset_right,
        cutset_cutset_residue=plan.cutset_cutset_residue,
        native_treewidth_plan=plan.native_treewidth_plan,
        remaining_isolated_vars=plan.remaining_isolated_vars,
        remaining_components=plan.remaining_components,
        remaining_width=plan.remaining_width,
        estimated_total_work=plan.estimated_total_work,
        branch_bits=branch_bits,
        branch_pair_residue=branch_pair_residue,
        branch_remaining_shift=branch_remaining_shift,
    )


def _build_q3_free_cutset_conditioning_plan_uncached(
    q: PhaseFunction,
    *,
    max_size: int = _Q3_FREE_CUTSET_MAX_SIZE,
    candidate_pool: int = _Q3_FREE_CUTSET_CANDIDATE_POOL,
    beam_width: int = _Q3_FREE_CUTSET_BEAM_WIDTH,
    branches_per_state: int = _Q3_FREE_CUTSET_BRANCHES_PER_STATE,
    prioritize_width: bool = False,
    target_remaining_width: int | None = None,
    candidate_override: Sequence[int] | None = None,
    allow_generic_remaining: bool = False,
    prefer_one_shot_slicing: bool = False,
) -> _Q3FreeCutsetConditioningPlan | None:
    if q.n <= 1 or not q.q2:
        return None

    adjacency, edges = _q3_free_graph(q)
    depth, chords = _q3_free_spanning_data(adjacency, edges)
    tensor_hint = _q3_free_tensor_slice_hint(q)
    preferred = set(_select_feedback_vertices(q.n, chords, depth))
    preferred.update(tensor_hint)
    if candidate_override is None:
        candidates = _candidate_q3_free_cutset_vertices(
            adjacency,
            preferred=preferred,
            max_candidates=int(candidate_pool),
        )
    else:
        candidates = _merge_q3_free_cutset_candidate_orders(
            candidate_override,
            max_candidates=int(candidate_pool),
        )
    if not candidates:
        return None

    best_evaluation: _Q3FreeCutsetCandidateEvaluation | None = None
    evaluation_cache: dict[tuple[int, ...], _Q3FreeCutsetCandidateEvaluation | None] = {}
    remaining_universe = tuple(range(q.n))
    frontier: list[tuple[int, ...]] = [()]
    max_size = min(int(max_size), len(candidates))

    def cached_evaluation(cutset_vars: tuple[int, ...]) -> _Q3FreeCutsetCandidateEvaluation | None:
        cached = evaluation_cache.get(cutset_vars)
        if cached is not None or cutset_vars in evaluation_cache:
            return cached
        evaluation = _evaluate_q3_free_cutset_candidate(
            q,
            cutset_vars,
            remaining_universe=remaining_universe,
            prioritize_width=prioritize_width,
            target_remaining_width=target_remaining_width,
            allow_generic_remaining=allow_generic_remaining,
            prefer_one_shot_slicing=prefer_one_shot_slicing,
        )
        evaluation_cache[cutset_vars] = evaluation
        return evaluation

    def meets_width_target(evaluation: _Q3FreeCutsetCandidateEvaluation | None) -> bool:
        return bool(
            prioritize_width
            and target_remaining_width is not None
            and evaluation is not None
            and evaluation.viable
            and evaluation.plan is not None
            and evaluation.plan.remaining_width <= int(target_remaining_width)
        )

    if tensor_hint:
        hinted_vars = [var for var in tensor_hint if var in candidates]
        for size in range(1, min(max_size, len(hinted_vars)) + 1):
            evaluation = cached_evaluation(tuple(sorted(hinted_vars[:size])))
            if evaluation is not None and evaluation.viable and (
                best_evaluation is None or evaluation.score < best_evaluation.score
            ):
                best_evaluation = evaluation

    selected: list[int] = []
    for _size in range(1, max_size + 1):
        best_choice: tuple[int, _Q3FreeCutsetCandidateEvaluation] | None = None
        for candidate in candidates:
            if candidate in selected:
                continue
            evaluation = cached_evaluation(tuple(sorted(selected + [candidate])))
            if evaluation is None:
                continue
            if best_choice is None or evaluation.score < best_choice[1].score:
                best_choice = (candidate, evaluation)
        if best_choice is None:
            break
        selected.append(best_choice[0])
        if best_choice[1].viable and (
            best_evaluation is None or best_choice[1].score < best_evaluation.score
        ):
            best_evaluation = best_choice[1]
            if not prioritize_width or meets_width_target(best_choice[1]):
                break

    if (
        best_evaluation is not None
        and q.n >= 64
        and len(best_evaluation.cutset_vars) <= 2
        and (not prioritize_width or meets_width_target(best_evaluation))
    ):
        plan = best_evaluation.plan
        assert plan is not None
        remaining_order = plan.remaining_order
        remaining_width = plan.remaining_width
        estimated_total_work = plan.estimated_total_work
        if plan.remaining_backend == "treewidth" and remaining_order:
            refined_q = _phase_function_from_parts(
                len(plan.remaining_vars),
                level=plan.level,
                q0=Fraction(0),
                q1=[0] * len(plan.remaining_vars),
                q2=plan.remaining_q2,
                q3={},
            )
            refined_order, refined_width = _finalize_q3_free_treewidth_order(
                refined_q,
                remaining_order,
            )
            remaining_order = tuple(int(var) for var in refined_order)
            remaining_width = int(refined_width)
            estimated_total_work = max(
                1,
                (1 << len(plan.cutset_vars)) * _estimate_treewidth_dp_work(refined_q, refined_order),
            )
        return _attach_q3_free_cutset_runtime_cache(_Q3FreeCutsetConditioningPlan(
            level=plan.level,
            cutset_vars=plan.cutset_vars,
            remaining_vars=plan.remaining_vars,
            remaining_backend=plan.remaining_backend,
            remaining_q2=plan.remaining_q2,
            remaining_order=remaining_order,
            cutset_remaining_q2_residue=plan.cutset_remaining_q2_residue,
            cutset_cutset_left=plan.cutset_cutset_left,
            cutset_cutset_right=plan.cutset_cutset_right,
            cutset_cutset_residue=plan.cutset_cutset_residue,
            native_treewidth_plan=(
                _build_native_q3_free_treewidth_plan(
                    n_vars=len(plan.remaining_vars),
                    level=plan.level,
                    q2=plan.remaining_q2,
                    order=remaining_order,
                )
                if plan.remaining_backend == "treewidth"
                else None
            ),
            remaining_isolated_vars=plan.remaining_isolated_vars,
            remaining_components=plan.remaining_components,
            remaining_width=remaining_width,
            estimated_total_work=estimated_total_work,
        ))

    for _size in range(1, max_size + 1):
        expansions: list[tuple[tuple[int, ...], tuple[int, ...], _Q3FreeCutsetCandidateEvaluation]] = []
        seen_cutsets: set[tuple[int, ...]] = set()
        for selected_indexes in frontier:
            next_idx = selected_indexes[-1] + 1 if selected_indexes else 0
            for candidate_idx in range(
                next_idx,
                min(len(candidates), next_idx + int(branches_per_state)),
            ):
                expanded_indexes = selected_indexes + (candidate_idx,)
                cutset_vars = tuple(sorted(candidates[idx] for idx in expanded_indexes))
                if cutset_vars in seen_cutsets:
                    continue
                seen_cutsets.add(cutset_vars)
                evaluation = cached_evaluation(cutset_vars)
                if evaluation is None:
                    continue
                expansions.append((evaluation.score, expanded_indexes, evaluation))

        if not expansions:
            break

        expansions.sort(key=lambda item: item[0])
        for _score, _expanded_indexes, evaluation in expansions:
            if evaluation.viable and (
                best_evaluation is None or evaluation.score < best_evaluation.score
            ):
                best_evaluation = evaluation

        frontier = [
            expanded_indexes
            for _score, expanded_indexes, _evaluation in expansions[: int(beam_width)]
        ]

    if best_evaluation is None:
        return None
    plan = best_evaluation.plan
    if plan is not None and plan.remaining_backend == "generic":
        plan = _finalize_q3_free_cutset_conditioning_plan(
            plan,
            prefer_one_shot_slicing=prefer_one_shot_slicing,
        )
    if (
        plan is not None
        and plan.remaining_backend == "treewidth"
        and plan.remaining_order
    ):
        refined_q = _phase_function_from_parts(
            len(plan.remaining_vars),
            level=plan.level,
            q0=Fraction(0),
            q1=[0] * len(plan.remaining_vars),
            q2=plan.remaining_q2,
            q3={},
        )
        refined_order, refined_width = _finalize_q3_free_treewidth_order(
            refined_q,
            plan.remaining_order,
        )
        plan = _Q3FreeCutsetConditioningPlan(
            level=plan.level,
            cutset_vars=plan.cutset_vars,
            remaining_vars=plan.remaining_vars,
            remaining_backend=plan.remaining_backend,
            remaining_q2=plan.remaining_q2,
            remaining_order=tuple(int(var) for var in refined_order),
            cutset_remaining_q2_residue=plan.cutset_remaining_q2_residue,
            cutset_cutset_left=plan.cutset_cutset_left,
            cutset_cutset_right=plan.cutset_cutset_right,
            cutset_cutset_residue=plan.cutset_cutset_residue,
            native_treewidth_plan=plan.native_treewidth_plan,
            remaining_isolated_vars=plan.remaining_isolated_vars,
            remaining_components=plan.remaining_components,
            remaining_width=int(refined_width),
            estimated_total_work=max(
                1,
                (1 << len(plan.cutset_vars)) * _estimate_treewidth_dp_work(refined_q, refined_order),
            ),
        )
    if (
        plan is not None
        and plan.remaining_backend == "treewidth"
        and plan.native_treewidth_plan is None
    ):
        return _attach_q3_free_cutset_runtime_cache(_Q3FreeCutsetConditioningPlan(
            level=plan.level,
            cutset_vars=plan.cutset_vars,
            remaining_vars=plan.remaining_vars,
            remaining_backend=plan.remaining_backend,
            remaining_q2=plan.remaining_q2,
            remaining_order=plan.remaining_order,
            cutset_remaining_q2_residue=plan.cutset_remaining_q2_residue,
            cutset_cutset_left=plan.cutset_cutset_left,
            cutset_cutset_right=plan.cutset_cutset_right,
            cutset_cutset_residue=plan.cutset_cutset_residue,
            native_treewidth_plan=_build_native_q3_free_treewidth_plan(
                n_vars=len(plan.remaining_vars),
                level=plan.level,
                q2=plan.remaining_q2,
                order=plan.remaining_order,
            ),
            remaining_isolated_vars=plan.remaining_isolated_vars,
            remaining_components=plan.remaining_components,
            remaining_width=plan.remaining_width,
            estimated_total_work=plan.estimated_total_work,
        ))
    return _attach_q3_free_cutset_runtime_cache(plan)


def _q3_free_cutset_conditioning_plan(
    q: PhaseFunction,
    *,
    max_size: int | None = None,
    candidate_pool: int | None = None,
    beam_width: int | None = None,
    branches_per_state: int | None = None,
    prioritize_width: bool = False,
    target_remaining_width: int | None = None,
    allow_generic_remaining: bool = False,
    prefer_one_shot_slicing: bool = False,
) -> _Q3FreeCutsetConditioningPlan | None:
    _cfg = _get_solver_config()
    max_size = _cfg.cutset_max_size if max_size is None else int(max_size)
    candidate_pool = _cfg.cutset_candidate_pool if candidate_pool is None else int(candidate_pool)
    beam_width = _cfg.cutset_beam_width if beam_width is None else int(beam_width)
    branches_per_state = _cfg.cutset_branches_per_state if branches_per_state is None else int(branches_per_state)
    cache_key = (
        _q_structure_key(q),
        int(max_size),
        int(candidate_pool),
        int(beam_width),
        int(branches_per_state),
        bool(prioritize_width),
        -1 if target_remaining_width is None else int(target_remaining_width),
        bool(allow_generic_remaining),
        bool(prefer_one_shot_slicing),
        bool(_quimb_import_enabled()),
        bool(_kahypar_available()),
    )
    cached = _STRUCTURE_Q3_FREE_CUTSET_PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    plan = _build_q3_free_cutset_conditioning_plan_uncached(
        q,
        max_size=max_size,
        candidate_pool=candidate_pool,
        beam_width=beam_width,
        branches_per_state=branches_per_state,
        prioritize_width=prioritize_width,
        target_remaining_width=target_remaining_width,
        allow_generic_remaining=allow_generic_remaining,
        prefer_one_shot_slicing=prefer_one_shot_slicing,
    )
    if plan is not None:
        _STRUCTURE_Q3_FREE_CUTSET_PLAN_CACHE[cache_key] = plan
    return plan


def _q3_free_one_shot_cutset_conditioning_plan(
    q: PhaseFunction,
) -> _Q3FreeCutsetConditioningPlan | None:
    """Try a stronger cutset search for giant one-shot q3-free components."""
    _cfg = _get_solver_config()
    plan = _q3_free_cutset_conditioning_plan(q, prefer_one_shot_slicing=True)
    if (
        plan is not None
        and plan.remaining_width <= _cfg.tensor_hint_target_width
        and _q3_free_cutset_plan_generic_penalty(plan) == 0
    ):
        return plan
    adjacency, edges = _q3_free_graph(q)
    depth, chords = _q3_free_spanning_data(adjacency, edges)
    preferred = set(_select_feedback_vertices(q.n, chords, depth))
    preferred.update(_q3_free_tensor_slice_hint(q))
    separator_candidates = _separator_ranked_q3_free_cutset_vertices(
        adjacency,
        preferred=preferred,
        max_candidates=_cfg.one_shot_cutset_candidate_pool,
    )
    generic_candidates = _candidate_q3_free_cutset_vertices(
        adjacency,
        preferred=preferred,
        max_candidates=_cfg.one_shot_cutset_candidate_pool,
    )
    merged_candidates = _merge_q3_free_cutset_candidate_orders(
        separator_candidates,
        generic_candidates,
        max_candidates=_cfg.one_shot_cutset_candidate_pool,
    )
    separator_plan = (
        _build_q3_free_cutset_conditioning_plan_uncached(
            q,
            max_size=_cfg.one_shot_cutset_max_size,
            candidate_pool=max(len(merged_candidates), 1),
            beam_width=_cfg.one_shot_cutset_beam_width,
            branches_per_state=_cfg.one_shot_cutset_branches_per_state,
            prioritize_width=True,
            target_remaining_width=_cfg.tensor_hint_target_width,
            candidate_override=merged_candidates,
            allow_generic_remaining=True,
            prefer_one_shot_slicing=True,
        )
        if merged_candidates
        else None
    )

    def plan_score(candidate: _Q3FreeCutsetConditioningPlan | None) -> tuple[int, int, int, int, int, int]:
        if candidate is None:
            return (1, 1 << 30, 1, 1 << 30, 1 << 30, 1 << 30)
        return (
            0,
            _q3_free_cutset_plan_generic_penalty(candidate),
            int(candidate.remaining_width > _cfg.tensor_hint_target_width),
            int(candidate.remaining_width),
            int(candidate.estimated_total_work),
            len(candidate.cutset_vars),
        )

    return min((plan, separator_plan), key=plan_score)


def _sum_q3_free_via_cutset_conditioning_scaled(q: PhaseFunction) -> ScaledComplex | None:
    plan = _q3_free_cutset_conditioning_plan(q)
    if plan is None:
        return None

    branch_count = 1 << len(plan.cutset_vars)
    branch_masks = np.arange(branch_count, dtype=np.uint64)
    branch_bits = _branch_assignment_bits(branch_masks, len(plan.cutset_vars), np)
    q0_eff = np.full(branch_count, _phase_fraction_to_residue(q.q0, q.mod_q1), dtype=np.int64)

    if plan.cutset_vars:
        cutset_q1 = np.asarray([q.q1[var] % q.mod_q1 for var in plan.cutset_vars], dtype=np.int64)
        q0_eff = (q0_eff + branch_bits @ cutset_q1) % q.mod_q1

    if plan.cutset_cutset_residue.size:
        for left, right, residue in zip(
            plan.cutset_cutset_left,
            plan.cutset_cutset_right,
            plan.cutset_cutset_residue,
        ):
            q0_eff = (
                q0_eff
                + int(residue) * branch_bits[:, int(left)] * branch_bits[:, int(right)]
            ) % q.mod_q1

    if plan.remaining_vars:
        base_q1 = np.asarray([q.q1[var] % q.mod_q1 for var in plan.remaining_vars], dtype=np.int64)
        q1_batch = np.broadcast_to(base_q1, (branch_count, len(plan.remaining_vars))).copy()
        if plan.cutset_remaining_q2_residue.size:
            q1_batch = (q1_batch + branch_bits @ plan.cutset_remaining_q2_residue) % q.mod_q1
    else:
        q1_batch = np.zeros((branch_count, 0), dtype=np.int64)

    if plan.remaining_backend == "product":
        branch_totals = [_product_q1_sum_scaled(row.tolist(), level=q.level) for row in q1_batch]
    elif plan.remaining_backend == "generic":
        row_map: dict[tuple[int, ...], int] = {}
        unique_rows: list[np.ndarray] = []
        inverse: list[int] = []
        for row in q1_batch:
            key = tuple(int(value) for value in row.tolist())
            existing = row_map.get(key)
            if existing is None:
                existing = len(unique_rows)
                row_map[key] = existing
                unique_rows.append(row.copy())
            inverse.append(existing)
        unique_batch = (
            np.vstack(unique_rows)
            if unique_rows
            else np.zeros((0, q1_batch.shape[1]), dtype=np.int64)
        )
        unique_totals: list[ScaledComplex] = [_ONE_SCALED] * len(unique_batch)
        if plan.remaining_isolated_vars:
            isolated_columns = unique_batch[:, plan.remaining_isolated_vars]
            unique_totals = [
                _product_q1_sum_scaled(row.tolist(), level=q.level)
                for row in isolated_columns
            ]
        for component_plan in plan.remaining_components:
            component_batch = unique_batch[:, component_plan.variables]
            component_totals = _evaluate_q3_free_component_plan_scaled_batch(
                component_plan,
                component_batch,
                level=q.level,
            )
            unique_totals = [
                _mul_scaled_complex(total, component_total)
                for total, component_total in zip(unique_totals, component_totals)
            ]
        branch_totals = [unique_totals[idx] for idx in inverse]
    else:
        if q1_batch.size == 0:
            branch_totals = [_ONE_SCALED] * branch_count
        else:
            row_map: dict[tuple[int, ...], int] = {}
            unique_rows: list[np.ndarray] = []
            inverse: list[int] = []
            for row in q1_batch:
                key = tuple(int(value) for value in row.tolist())
                existing = row_map.get(key)
                if existing is None:
                    existing = len(unique_rows)
                    row_map[key] = existing
                    unique_rows.append(row.copy())
                inverse.append(existing)
            unique_batch = np.vstack(unique_rows) if unique_rows else np.zeros((0, q1_batch.shape[1]), dtype=np.int64)
            unique_totals = _sum_q3_free_treewidth_dp_scaled_batch(
                n_vars=len(plan.remaining_vars),
                level=q.level,
                q1_batch=unique_batch,
                q2=plan.remaining_q2,
                order=plan.remaining_order,
                native_plan=plan.native_treewidth_plan,
            )
            branch_totals = [unique_totals[idx] for idx in inverse]

    omega_scaled = _omega_scaled_table(q.level)
    total = _ZERO_SCALED
    for q0_residue, branch_total in zip(q0_eff, branch_totals):
        total = _add_scaled_complex(
            total,
            _mul_scaled_complex(omega_scaled[int(q0_residue) % q.mod_q1], branch_total),
        )
    return total


def _evaluate_q3_free_cutset_conditioning_plan_scaled_batch(
    plan: _Q3FreeCutsetConditioningPlan,
    q1_batch: np.ndarray,
    *,
    level: int,
) -> list[ScaledComplex]:
    """Evaluate a reusable q3-free cutset plan for many q1 assignments."""
    plan = _attach_q3_free_cutset_runtime_cache(plan)
    q1_batch = np.asarray(q1_batch, dtype=np.int64)
    if q1_batch.ndim != 2:
        raise ValueError("Expected q1_batch to have shape (batch, n_vars).")

    n_vars = len(plan.cutset_vars) + len(plan.remaining_vars)
    if q1_batch.shape[1] != n_vars:
        raise ValueError(f"Expected q1 rows of length {n_vars}, received {q1_batch.shape[1]}.")

    batch_size = q1_batch.shape[0]
    if batch_size == 0:
        return []

    mod_q1 = 1 << int(level)
    branch_bits = np.asarray(plan.branch_bits, dtype=np.int64)
    branch_count = int(branch_bits.shape[0])

    q0_eff = np.zeros((batch_size, branch_count), dtype=np.int64)
    if plan.cutset_vars:
        cutset_q1 = np.asarray(q1_batch[:, plan.cutset_vars], dtype=np.int64) % mod_q1
        q0_eff = (cutset_q1 @ branch_bits.T) % mod_q1

    branch_pair_residue = np.asarray(plan.branch_pair_residue, dtype=np.int64)
    if branch_pair_residue.size:
        q0_eff = (q0_eff + branch_pair_residue[None, :]) % mod_q1

    if plan.remaining_vars:
        base_remaining_q1 = np.asarray(q1_batch[:, plan.remaining_vars], dtype=np.int64) % mod_q1
        branch_remaining_shift = np.asarray(plan.branch_remaining_shift, dtype=np.int64)
        if branch_remaining_shift.size:
            remaining_q1 = (
                base_remaining_q1[:, None, :] + branch_remaining_shift[None, :, :]
            ) % mod_q1
        else:
            remaining_q1 = np.broadcast_to(
                base_remaining_q1[:, None, :],
                (batch_size, branch_count, len(plan.remaining_vars)),
            ).copy()
    else:
        remaining_q1 = np.zeros((batch_size, branch_count, 0), dtype=np.int64)

    if plan.remaining_backend == "product":
        branch_totals = [
            _product_q1_sum_scaled(row.tolist(), level=level)
            for row in remaining_q1.reshape(batch_size * branch_count, -1)
        ]
    elif plan.remaining_backend == "generic":
        flat_remaining_q1 = remaining_q1.reshape(batch_size * branch_count, -1)
        row_map: dict[tuple[int, ...], int] = {}
        unique_rows: list[np.ndarray] = []
        inverse: list[int] = []
        for row in flat_remaining_q1:
            key = tuple(int(value) for value in row.tolist())
            existing = row_map.get(key)
            if existing is None:
                existing = len(unique_rows)
                row_map[key] = existing
                unique_rows.append(row.copy())
            inverse.append(existing)
        unique_batch = (
            np.vstack(unique_rows)
            if unique_rows
            else np.zeros((0, flat_remaining_q1.shape[1]), dtype=np.int64)
        )
        unique_totals: list[ScaledComplex] = [_ONE_SCALED] * len(unique_batch)
        if plan.remaining_isolated_vars:
            isolated_columns = unique_batch[:, plan.remaining_isolated_vars]
            unique_totals = [
                _product_q1_sum_scaled(row.tolist(), level=level)
                for row in isolated_columns
            ]
        for component_plan in plan.remaining_components:
            component_batch = unique_batch[:, component_plan.variables]
            component_totals = _evaluate_q3_free_component_plan_scaled_batch(
                component_plan,
                component_batch,
                level=level,
            )
            unique_totals = [
                _mul_scaled_complex(total, component_total)
                for total, component_total in zip(unique_totals, component_totals)
            ]
        branch_totals = [unique_totals[idx] for idx in inverse]
    else:
        if remaining_q1.size == 0:
            branch_totals = [_ONE_SCALED] * (batch_size * branch_count)
        else:
            flat_remaining_q1 = remaining_q1.reshape(batch_size * branch_count, -1)
            if flat_remaining_q1.shape[0] <= 2:
                branch_totals = _sum_q3_free_treewidth_dp_scaled_batch(
                    n_vars=len(plan.remaining_vars),
                    level=level,
                    q1_batch=np.ascontiguousarray(flat_remaining_q1, dtype=np.int64),
                    q2=plan.remaining_q2,
                    order=plan.remaining_order,
                    native_plan=plan.native_treewidth_plan,
                )
            else:
                row_map: dict[tuple[int, ...], int] = {}
                unique_rows: list[np.ndarray] = []
                inverse: list[int] = []
                for row in flat_remaining_q1:
                    key = tuple(int(value) for value in row.tolist())
                    existing = row_map.get(key)
                    if existing is None:
                        existing = len(unique_rows)
                        row_map[key] = existing
                        unique_rows.append(row.copy())
                    inverse.append(existing)
                unique_batch = (
                    np.vstack(unique_rows)
                    if unique_rows
                    else np.zeros((0, flat_remaining_q1.shape[1]), dtype=np.int64)
                )
                unique_totals = _sum_q3_free_treewidth_dp_scaled_batch(
                    n_vars=len(plan.remaining_vars),
                    level=level,
                    q1_batch=unique_batch,
                    q2=plan.remaining_q2,
                    order=plan.remaining_order,
                    native_plan=plan.native_treewidth_plan,
                )
                branch_totals = [unique_totals[idx] for idx in inverse]

    omega_scaled = _omega_scaled_table(level)
    totals: list[ScaledComplex] = []
    for row_idx in range(batch_size):
        row_total = _ZERO_SCALED
        base_idx = row_idx * branch_count
        for branch_idx in range(branch_count):
            row_total = _add_scaled_complex(
                row_total,
                _mul_scaled_complex(
                    omega_scaled[int(q0_eff[row_idx, branch_idx]) % mod_q1],
                    branch_totals[base_idx + branch_idx],
                ),
            )
        totals.append(row_total)
    return totals


def _evaluate_q3_free_cutset_conditioning_plan_scaled(
    plan: _Q3FreeCutsetConditioningPlan,
    q1_local: Sequence[int],
    *,
    level: int,
) -> ScaledComplex:
    """Evaluate one reusable q3-free cutset plan under a concrete q1 vector."""
    return _evaluate_q3_free_cutset_conditioning_plan_scaled_batch(
        plan,
        np.asarray([q1_local], dtype=np.int64),
        level=level,
    )[0]


# ==================================================================
# Phase-3 backend planning and execution
# ==================================================================

def _build_factor_scopes(q):
    """Return the unique non-scalar factor scopes induced by ``q``."""
    scopes = set()

    for var, coeff in enumerate(q.q1):
        if coeff % q.mod_q1:
            scopes.add((var,))

    for scope, coeff in q.q2.items():
        if coeff % q.mod_q2:
            scopes.add(scope)

    for scope, coeff in q.q3.items():
        if coeff % q.mod_q3:
            scopes.add(scope)

    return scopes


def _treewidth_order_width(q, order):
    """Return the maximum bucket scope induced by ``order`` on ``q``."""
    factors = _build_factor_scopes(q)
    max_scope = 0

    for var in order:
        bucket_scopes = [scope for scope in factors if var in scope]
        if not bucket_scopes:
            max_scope = max(max_scope, 1)
            continue

        for scope in bucket_scopes:
            factors.remove(scope)
        union_scope = tuple(sorted({vertex for scope in bucket_scopes for vertex in scope}))
        max_scope = max(max_scope, len(union_scope))
        new_scope = tuple(vertex for vertex in union_scope if vertex != var)
        if new_scope:
            factors.add(new_scope)

    return max_scope


def _treewidth_order_scope_trace(q, order):
    """Return the per-step bucket scope sizes induced by ``order`` on ``q``."""
    factors = _build_factor_scopes(q)
    scopes: list[int] = []

    for var in order:
        bucket_scopes = [scope for scope in factors if var in scope]
        if not bucket_scopes:
            scopes.append(1)
            continue

        for scope in bucket_scopes:
            factors.remove(scope)
        union_scope = tuple(sorted({vertex for scope in bucket_scopes for vertex in scope}))
        scopes.append(len(union_scope))
        new_scope = tuple(vertex for vertex in union_scope if vertex != var)
        if new_scope:
            factors.add(new_scope)

    return scopes


def _q3_free_treewidth_width_limit() -> int:
    """Return the q3-free treewidth width limit for the current exact backend."""
    if _schur_native is not None:
        return _Q3_FREE_SUM_TREEWIDTH_NATIVE_MAX_WIDTH
    return _Q3_FREE_SUM_TREEWIDTH_MAX_WIDTH


def _estimate_treewidth_dp_work(q, order):
    """
    Cheap proxy for the factor-elimination work along ``order``.

    The exact DP loops over all assignments of each elimination bucket's union
    scope and multiplies one local factor per bucket table. We therefore track
    only scopes and estimate the work as ``bucket_size * 2**|union_scope|`` at
    each elimination step.
    """
    if (
        not q.q3
        and _schur_native is not None
        and hasattr(_schur_native, "q3_free_treewidth_dp_work")
    ):
        try:
            return int(
                _schur_native.q3_free_treewidth_dp_work(
                    int(q.n),
                    int(q.level),
                    q.q1,
                    q.q2,
                    tuple(int(var) for var in order),
                )
            )
        except Exception:
            pass

    factors = _build_factor_scopes(q)
    work = 0

    for var in order:
        bucket_scopes = [scope for scope in factors if var in scope]
        if not bucket_scopes:
            work += 1
            continue

        for scope in bucket_scopes:
            factors.remove(scope)
        union_scope = tuple(sorted({vertex for scope in bucket_scopes for vertex in scope}))
        new_scope = tuple(vertex for vertex in union_scope if vertex != var)
        work += len(bucket_scopes) * (1 << len(union_scope))
        if new_scope:
            factors.add(new_scope)

    return work


def _move_order_entry(order: Sequence[int], src: int, dst: int) -> list[int]:
    moved = list(order)
    value = moved.pop(src)
    moved.insert(dst, value)
    return moved


def _refine_q3_free_treewidth_order_locally(q, order: Sequence[int], width: int):
    """
    Bounded local refinement on top of min-fill.

    This keeps the global heuristic but tries a small number of hotspot moves
    scored by the actual DP objective (width, then work).
    """
    if not order or q.n < 8 or width < 2:
        return list(order), int(width)

    best_order = list(order)
    best_width = int(width)
    best_work = int(_estimate_treewidth_dp_work(q, best_order))
    if best_width <= 1:
        return best_order, best_width

    max_passes = 2
    max_hotspots = 8
    move_radius = 2

    for _ in range(max_passes):
        scopes = _treewidth_order_scope_trace(q, best_order)
        hotspot_positions = [
            idx
            for idx, _scope in sorted(
                enumerate(scopes),
                key=lambda item: (item[1], -item[0]),
                reverse=True,
            )[:max_hotspots]
        ]
        improved = False
        seen: set[tuple[int, ...]] = set()
        for pos in hotspot_positions:
            for delta in range(-move_radius, move_radius + 1):
                if delta == 0:
                    continue
                dst = pos + delta
                if dst < 0 or dst >= len(best_order):
                    continue
                candidate = _move_order_entry(best_order, pos, dst)
                key = tuple(candidate)
                if key in seen:
                    continue
                seen.add(key)
                candidate_width = int(_treewidth_order_width(q, candidate))
                if candidate_width > best_width:
                    continue
                candidate_work = int(_estimate_treewidth_dp_work(q, candidate))
                candidate_score = (candidate_width, candidate_work)
                best_score = (best_width, best_work)
                if candidate_score < best_score:
                    best_order = candidate
                    best_width = candidate_width
                    best_work = candidate_work
                    improved = True
        if not improved:
            break

    return best_order, best_width


def _finalize_q3_free_treewidth_order(q, order: Sequence[int]):
    """
    Refine a chosen treewidth order once, after planning decides to use it.

    This is intentionally separate from `_q3_free_treewidth_order` so cutset
    candidate search can stay cheap.
    """
    base_order = tuple(int(var) for var in order)
    cache_key = (_q_structure_key(q), base_order)
    cached = _STRUCTURE_Q3_FREE_REFINED_ORDER_CACHE.get(cache_key)
    if cached is not None:
        refined_order, refined_width = cached
        return list(refined_order), int(refined_width)

    base_width = int(_treewidth_order_width(q, base_order))
    refined_order, refined_width = _refine_q3_free_treewidth_order_locally(q, base_order, base_width)
    cached = (tuple(int(var) for var in refined_order), int(refined_width))
    _STRUCTURE_Q3_FREE_REFINED_ORDER_CACHE[cache_key] = cached
    refined_order, refined_width = cached
    return list(refined_order), int(refined_width)


def _q3_free_treewidth_candidate_is_viable(q, order, width: int, feedback_size: int) -> bool:
    """Decide whether a q3-free treewidth candidate is worth accepting."""
    width_limit = _q3_free_treewidth_width_limit()
    if width > width_limit or width >= feedback_size:
        return False
    if (
        _schur_native is not None
        and width > _Q3_FREE_SUM_TREEWIDTH_MAX_WIDTH
        and _estimate_treewidth_dp_work(q, order) > _Q3_FREE_SUM_TREEWIDTH_NATIVE_MAX_WORK
    ):
        return False
    return True


def _q3_hypergraph_2core(q):
    """Return the live q3 2-core variables and the degree-1 peel order."""
    cache_key = _q_q3_support_key(q)
    cached = _STRUCTURE_Q3_2CORE_CACHE.get(cache_key)
    if cached is not None:
        core_vars, peel_order = cached
        return frozenset(core_vars), list(peel_order)

    active_q3_vars: set[int] = set()
    incident_edges: list[set[tuple[int, int, int]]] = [set() for _ in range(q.n)]
    live_edges: set[tuple[int, int, int]] = set()

    for edge_key, coeff in q.q3.items():
        if coeff % q.mod_q3 == 0:
            continue
        live_edges.add(edge_key)
        active_q3_vars.update(edge_key)
        for var in edge_key:
            incident_edges[var].add(edge_key)

    if not live_edges:
        return frozenset(), []

    peel_order: list[int] = []
    peeled: set[int] = set()
    pending = [var for var in sorted(active_q3_vars) if len(incident_edges[var]) <= 1]

    while pending:
        var = pending.pop()
        if var in peeled:
            continue
        live_incident = incident_edges[var] & live_edges
        if len(live_incident) > 1:
            continue
        peeled.add(var)
        peel_order.append(var)
        for edge_key in tuple(live_incident):
            if edge_key not in live_edges:
                continue
            live_edges.remove(edge_key)
            for neighbor in edge_key:
                if edge_key in incident_edges[neighbor]:
                    incident_edges[neighbor].remove(edge_key)
                    if neighbor in active_q3_vars and neighbor not in peeled and len(incident_edges[neighbor]) <= 1:
                        pending.append(neighbor)

    core_vars = frozenset(
        var
        for var in sorted(active_q3_vars)
        if incident_edges[var] & live_edges
    )
    cached = (tuple(sorted(core_vars)), tuple(peel_order))
    _STRUCTURE_Q3_2CORE_CACHE[cache_key] = cached
    core_vars, peel_order = cached
    return frozenset(core_vars), list(peel_order)


def _active_q3_variables(q) -> tuple[int, ...]:
    active: set[int] = set()
    for edge, coeff in q.q3.items():
        if coeff % q.mod_q3:
            active.update(edge)
    return tuple(sorted(active))


def _phase_function_q2_density_milli(q) -> int:
    if q.n <= 1:
        return 0
    return int(round(1000.0 * (2.0 * len(q.q2)) / (q.n * (q.n - 1))))


def _phase_function_structure_score(q) -> tuple[int, int, int, int, int, int, int, int, int]:
    active_q3 = _active_q3_variables(q)
    core_vars, _ = _q3_hypergraph_2core(q)
    components = detect_factorization(q)
    threshold = max(1, q.mod_q1 // 4)
    bad_q1 = sum(1 for coeff in q.q1 if int(coeff) % threshold)
    max_width = 0
    max_component_vars = 0
    max_density = 0
    total_width = 0
    for component in components:
        component_q = _component_restriction(q, component)
        _order, width = _min_fill_cubic_order(component_q)
        max_width = max(max_width, int(width))
        total_width += int(width)
        max_component_vars = max(max_component_vars, int(component_q.n))
        max_density = max(max_density, _phase_function_q2_density_milli(component_q))
    return (
        len(core_vars),
        len(q.q3),
        len(active_q3),
        max_width,
        max_component_vars,
        max_density,
        bad_q1,
        len(q.q2),
        -len(components),
        total_width,
    )


def _phase_structure_hotspot_centers(q) -> tuple[int, ...]:
    adjacency = _interaction_graph(q)
    threshold = max(1, q.mod_q1 // 4)
    active = {
        idx
        for idx, coeff in enumerate(q.q1)
        if int(coeff) % q.mod_q1
    }
    for (left, right), coeff in q.q2.items():
        if coeff % q.mod_q2:
            active.add(left)
            active.add(right)
    for edge, coeff in q.q3.items():
        if coeff % q.mod_q3:
            active.update(edge)

    ranked = sorted(
        active,
        key=lambda var: (
            len(adjacency[var]),
            int(q.q1[var] % threshold != 0),
            int(q.q1[var] % q.mod_q1 != 0),
            -var,
        ),
        reverse=True,
    )
    return tuple(ranked[:_PHASE_STRUCTURE_LOCAL_MAX_CENTERS])


def _phase_structure_local_region(
    adjacency: Sequence[set[int]],
    center: int,
    *,
    radius: int = _PHASE_STRUCTURE_LOCAL_REGION_RADIUS,
    max_vars: int = _PHASE_STRUCTURE_LOCAL_REGION_MAX_VARS,
) -> tuple[int, ...]:
    region: set[int] = {int(center)}
    frontier = {int(center)}
    for _ in range(radius):
        if len(region) >= max_vars or not frontier:
            break
        next_frontier: set[int] = set()
        for var in frontier:
            next_frontier.update(adjacency[var])
        next_frontier -= region
        if not next_frontier:
            break
        ranked = sorted(
            next_frontier,
            key=lambda var: (len(adjacency[var]), -var),
            reverse=True,
        )
        for var in ranked:
            region.add(int(var))
            if len(region) >= max_vars:
                break
        frontier = set(ranked)
    return tuple(sorted(region))


def _phase_structure_local_move_score(
    q,
    region: Sequence[int],
    target_local: int,
    sources_local: Sequence[int],
    *,
    context=None,
) -> tuple[int, ...] | None:
    region = tuple(int(var) for var in region)
    support = {int(region[int(target_local)])}
    support.update(int(region[int(src)]) for src in sources_local)
    adjacency = _interaction_graph(q)
    eval_region = set(support)
    for var in tuple(support):
        eval_region.update(adjacency[var])
    if len(eval_region) > _PHASE_STRUCTURE_LOCAL_REGION_MAX_VARS:
        ranked_boundary = sorted(
            (var for var in eval_region if var not in support),
            key=lambda var: (len(adjacency[var]), -var),
            reverse=True,
        )
        trimmed = set(support)
        for var in ranked_boundary:
            trimmed.add(int(var))
            if len(trimmed) >= _PHASE_STRUCTURE_LOCAL_REGION_MAX_VARS:
                break
        eval_region = trimmed
    eval_region_tuple = tuple(sorted(eval_region))
    local_q = _component_restriction(q, eval_region_tuple)
    before = _phase_function_structure_score(local_q)
    local_index = {var: idx for idx, var in enumerate(eval_region_tuple)}
    transformed_local = _basis_xor_transform(
        local_q,
        local_index[int(region[int(target_local)])],
        tuple(local_index[int(region[int(src)])] for src in sources_local),
        context=context,
    )
    after = _phase_function_structure_score(transformed_local)
    if after >= before:
        return None
    return after + before


def _optimize_phase_function_structure_locally(q, context=None):
    """Search exact XOR basis moves only inside dense/hard local neighborhoods."""
    current = q
    changed = False
    current_score = _phase_function_structure_score(current)

    for _ in range(_PHASE_STRUCTURE_LOCAL_MAX_PASSES):
        adjacency = _interaction_graph(current)
        regions: list[tuple[int, ...]] = []
        seen_regions: set[tuple[int, ...]] = set()
        for center in _phase_structure_hotspot_centers(current):
            region = _phase_structure_local_region(adjacency, center)
            if len(region) <= 1 or region in seen_regions:
                continue
            seen_regions.add(region)
            regions.append(region)

        candidate_moves: dict[tuple[int, tuple[int, ...]], tuple[int, ...]] = {}

        for region in regions:
            local_q = _component_restriction(current, region)
            for target_local, sources_local in _phase_function_basis_transform_candidates(local_q):
                score = _phase_structure_local_move_score(
                    current,
                    region,
                    target_local,
                    sources_local,
                    context=context,
                )
                if score is None:
                    continue
                global_move = (
                    int(region[target_local]),
                    tuple(int(region[src]) for src in sources_local),
                )
                existing = candidate_moves.get(global_move)
                if existing is None or score < existing:
                    candidate_moves[global_move] = score

        if not candidate_moves:
            break

        ranked_candidates = sorted(candidate_moves.items(), key=lambda item: item[1])[:_PHASE_STRUCTURE_LOCAL_CANDIDATE_POOL]
        best_global_q = None
        best_global_score = current_score
        for (target, sources), _local_score in ranked_candidates:
            candidate_q = _basis_xor_transform(current, target, sources, context=context)
            candidate_score = _phase_function_structure_score(candidate_q)
            if candidate_score < best_global_score:
                best_global_q = candidate_q
                best_global_score = candidate_score

        if best_global_q is None:
            break
        current = best_global_q
        current_score = best_global_score
        changed = True

    return current, changed


def _basis_xor_transform(q, target: int, sources: Sequence[int], context=None):
    gamma = [1 << idx for idx in range(q.n)]
    mask = 1 << target
    for source in sources:
        if source == target:
            raise ValueError("Basis XOR transforms require distinct source variables.")
        mask |= 1 << source
    gamma[target] = mask
    return _aff_compose_cached(q, 0, gamma, q.n, context=context)


def _phase_function_basis_candidate_variables(q) -> tuple[int, ...]:
    active: set[int]
    if q.q3:
        active = set(_active_q3_variables(q))
    else:
        active = {
            idx
            for idx, coeff in enumerate(q.q1)
            if int(coeff) % q.mod_q1
        }
        for (left, right), coeff in q.q2.items():
            if coeff % q.mod_q2:
                active.add(left)
                active.add(right)

    if len(active) <= 1:
        return ()

    q3_degree = {var: 0 for var in active}
    q2_degree = {var: 0 for var in active}
    for edge, coeff in q.q3.items():
        if coeff % q.mod_q3 == 0:
            continue
        for var in edge:
            if var in q3_degree:
                q3_degree[var] += 1
    adjacency = [set() for _ in range(q.n)]
    for (left, right), coeff in q.q2.items():
        if coeff % q.mod_q2 == 0:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
        if left in q2_degree:
            q2_degree[left] += 1
        if right in q2_degree:
            q2_degree[right] += 1

    threshold = max(1, q.mod_q1 // 4)
    ranked = sorted(
        active,
        key=lambda var: (
            q3_degree[var],
            q2_degree[var],
            int(q.q1[var] % threshold != 0),
            int(q.q1[var] % q.mod_q1 != 0),
            len(adjacency[var] & active),
            -var,
        ),
        reverse=True,
    )
    return tuple(ranked[:_PHASE_STRUCTURE_OPT_MAX_ACTIVE_VARS])


def _phase_function_basis_transform_candidates(q) -> list[tuple[int, tuple[int, ...]]]:
    candidate_vars = _phase_function_basis_candidate_variables(q)
    if len(candidate_vars) <= 1:
        return []

    q3_degree = {var: 0 for var in candidate_vars}
    q2_degree = {var: 0 for var in candidate_vars}
    adjacency = [set() for _ in range(q.n)]
    for edge, coeff in q.q3.items():
        if coeff % q.mod_q3 == 0:
            continue
        for var in edge:
            if var in q3_degree:
                q3_degree[var] += 1
    for (left, right), coeff in q.q2.items():
        if coeff % q.mod_q2 == 0:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
        if left in q2_degree:
            q2_degree[left] += 1
        if right in q2_degree:
            q2_degree[right] += 1

    moves: list[tuple[int, tuple[int, ...]]] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for target in candidate_vars:
        for source in candidate_vars:
            if source == target:
                continue
            move = (target, (source,))
            if move not in seen:
                seen.add(move)
                moves.append(move)

    if q.q3:
        for edge, coeff in q.q3.items():
            if coeff % q.mod_q3 == 0:
                continue
            if not all(var in candidate_vars for var in edge):
                continue
            a, b, c = edge
            for target, sources in (
                (a, (b, c)),
                (b, (a, c)),
                (c, (a, b)),
            ):
                move = (target, tuple(sorted(sources)))
                if move not in seen:
                    seen.add(move)
                    moves.append(move)
    return moves


def _optimize_phase_function_structure(q, context=None):
    """Beam-search exact XOR basis changes scored for TerKet's solver."""
    if q.n > _PHASE_STRUCTURE_OPT_MAX_VARS:
        return _optimize_phase_function_structure_locally(q, context=context)
    if not q.q2 and not q.q3:
        return q, False

    baseline_score = _phase_function_structure_score(q)
    best_q = q
    best_score = baseline_score
    changed = False
    beam: list[tuple[tuple[int, ...], PhaseFunction]] = [(baseline_score, q)]

    for _ in range(_PHASE_STRUCTURE_OPT_MAX_PASSES):
        pool: dict[tuple[Fraction, tuple[int, ...], tuple[tuple[int, int], int], tuple[tuple[int, int, int], int]], tuple[tuple[int, ...], PhaseFunction]] = {
            _q_key(best_q): (best_score, best_q)
        }
        for _score, candidate_q in beam:
            for target, sources in _phase_function_basis_transform_candidates(candidate_q):
                transformed = _basis_xor_transform(candidate_q, target, sources, context=context)
                key = _q_key(transformed)
                score = _phase_function_structure_score(transformed)
                existing = pool.get(key)
                if existing is None or score < existing[0]:
                    pool[key] = (score, transformed)
                if score < best_score:
                    best_q = transformed
                    best_score = score
                    changed = True

        ranked = sorted(pool.values(), key=lambda item: item[0])
        if not ranked or ranked[0][0] >= beam[0][0]:
            break
        beam = ranked[:_PHASE_STRUCTURE_OPT_BEAM_WIDTH]

    return best_q, changed


def _simplify_q3_basis(q, context=None):
    """Backward-compatible wrapper around the general structural optimizer."""
    if not q.q3:
        return q, False
    return _optimize_phase_function_structure(q, context=context)


def _projected_components_after_fixing(q, separator: Sequence[int]) -> list[set[int]]:
    """Connected components guaranteed to survive after fixing ``separator``."""
    removed = set(separator)
    adjacency = [set() for _ in range(q.n)]
    active_vars: set[int] = {
        idx for idx, coeff in enumerate(q.q1) if coeff % q.mod_q1 and idx not in removed
    }

    for (i, j), coeff in q.q2.items():
        if coeff % q.mod_q2 == 0 or i in removed or j in removed:
            continue
        adjacency[i].add(j)
        adjacency[j].add(i)
        active_vars.update((i, j))

    for edge, coeff in q.q3.items():
        if coeff % q.mod_q3 == 0:
            continue
        survivors = [var for var in edge if var not in removed]
        if not survivors:
            continue
        active_vars.update(survivors)
        if len(survivors) < 2:
            continue
        for left in range(len(survivors)):
            for right in range(left + 1, len(survivors)):
                a = survivors[left]
                b = survivors[right]
                adjacency[a].add(b)
                adjacency[b].add(a)

    components: list[set[int]] = []
    visited: set[int] = set()
    for start in sorted(active_vars):
        if start in visited:
            continue
        stack = [start]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            stack.extend(adjacency[node] - visited)
        components.append(component)
    return components


def _find_small_q3_separator(q) -> tuple[int, ...] | None:
    """Return a small separator whose fixing disconnects the residual kernel."""
    cache_key = _q_phase3_structure_key(q)
    cached = _STRUCTURE_Q3_SEPARATOR_CACHE.get(cache_key)
    if cached is not None:
        return None if cached == () else tuple(cached)

    active_q3 = _active_q3_variables(q)
    if len(active_q3) <= 2:
        _STRUCTURE_Q3_SEPARATOR_CACHE[cache_key] = ()
        return None

    q3_degree = {var: 0 for var in active_q3}
    for edge, coeff in q.q3.items():
        if coeff % q.mod_q3 == 0:
            continue
        for var in edge:
            q3_degree[var] += 1

    candidates = sorted(
        active_q3,
        key=lambda var: (q3_degree[var], -var),
        reverse=True,
    )[:_Q3_SEPARATOR_MAX_CANDIDATES]

    best_separator = None
    best_score = None
    max_size = min(_Q3_SEPARATOR_MAX_SIZE, len(candidates) - 1)
    for size in range(1, max_size + 1):
        for separator in combinations(candidates, size):
            components = _projected_components_after_fixing(q, separator)
            if len(components) < 2:
                continue
            largest = max(len(component) for component in components)
            score = (size, largest, -len(components))
            if best_score is None or score < best_score:
                best_separator = tuple(separator)
                best_score = score
        if best_separator is not None:
            break
    _STRUCTURE_Q3_SEPARATOR_CACHE[cache_key] = () if best_separator is None else tuple(best_separator)
    return best_separator


def _q3_core_cover_size(q, core_vars) -> int:
    """Return the exact q3-cover size on the surviving 2-core."""
    if not core_vars:
        return 0
    ordered_core = tuple(sorted(core_vars))
    remap = {var: idx for idx, var in enumerate(ordered_core)}
    core_q3 = {
        tuple(sorted(remap[var] for var in edge)): coeff
        for edge, coeff in q.q3.items()
        if coeff % q.mod_q3 and all(var in remap for var in edge)
    }
    if not core_q3:
        return 0
    core_phase = _phase_function_from_parts(
        len(ordered_core),
        level=q.level,
        q0=Fraction(0),
        q1=[0] * len(ordered_core),
        q2={},
        q3=core_q3,
    )
    return len(_minimum_q3_vertex_cover(core_phase))


def _estimate_q3_cover_work(q, cover_size):
    """
    Cheap proxy for q3-cover branching work.

    Cover branching pays the full ``2**cover_size`` leaf count, and each leaf
    still has to reduce the surviving variables after touching the local
    q1/q2 structure and the original q3 terms while applying the fixed values.
    """
    term_count = sum(1 for coeff in q.q1 if coeff % q.mod_q1) + len(q.q2) + len(q.q3)
    residual_vars = max(1, q.n - cover_size)
    per_leaf_work = residual_vars * max(1, q.n + term_count)
    return (1 << cover_size) * per_leaf_work


def _estimate_q3_separator_work(q, separator: Sequence[int]) -> int:
    """Cheap proxy for separator branching work."""
    if not separator:
        return max(1, q.n)
    components = _projected_components_after_fixing(q, separator)
    if len(components) < 2:
        return max(1, _estimate_q3_cover_work(q, len(separator)))
    term_count = sum(1 for coeff in q.q1 if coeff % q.mod_q1) + len(q.q2) + len(q.q3)
    branch_cost = 0
    for component in components:
        size = len(component)
        branch_cost += max(1, size * max(1, size + term_count))
    return (1 << len(separator)) * max(1, branch_cost)


def _prefer_treewidth_phase3(q, cover, order, width, *, fully_peeled: bool = False):
    """Decide whether Phase 3 should use treewidth DP on ``q``."""
    width_limit = (
        _Q3_TREEWIDTH_DP_PEELED_MAX_WIDTH
        if fully_peeled
        else _Q3_TREEWIDTH_DP_MAX_WIDTH
    )
    if width > width_limit:
        return False
    treewidth_work = max(1, int(_estimate_treewidth_dp_work(q, order)))
    if fully_peeled and treewidth_work <= _Q3_TREEWIDTH_DP_PEELED_MAX_WORK:
        return True
    if width <= max(1, len(cover)):
        return True
    cover_work = max(1, int(_estimate_q3_cover_work(q, len(cover))))
    return treewidth_work <= cover_work


_CUBIC_CONTRACTION_MAX_WIDTH = 12  # numpy bucket elim beats quimb up to this width
_Q3_COVER_GPU_MIN_COVER = 9
_Q3_COVER_GPU_MAX_COVER = 24
_Q3_COVER_GPU_MAX_REMAINING = 20
_Q3_COVER_GPU_BRANCH_CHUNK_MAX = 128
_Q3_COVER_GPU_ASSIGNMENT_CHUNK_LOG2 = 13

def _prefer_cubic_contraction_phase3(q, cover, order, width, *, fully_peeled: bool = False):
    """Decide whether the specialized cubic contraction should be used.

    Benchmarks show that the numpy bucket-elimination implementation
    outperforms quimb tensor contraction for residual treewidths up to
    roughly 12.  The C-native treewidth DP (_sum_via_treewidth_dp) is
    faster than both when it is the preferred strategy, so we defer to
    that path first.
    """
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


def _prefer_q3_cover_gpu(q, cover) -> bool:
    del q, cover
    return False


def _prefer_tensor_contraction_phase3(q, cover, order, width, allow_tensor_contraction=True):
    del q, cover, order, width, allow_tensor_contraction
    return False


def _prefer_hybrid_contraction_phase3(q, cover, order, width, allow_tensor_contraction=True):
    del q, cover, order, width, allow_tensor_contraction
    return False


def _select_direct_phase3_backend(
    q,
    cover,
    order,
    width,
    *,
    allow_tensor_contraction=True,
    fully_peeled: bool = False,
):
    """Return the direct Phase-3 backend worth preferring over Phase-2 branching."""
    if fully_peeled and _prefer_treewidth_phase3(
        q,
        cover,
        order,
        width,
        fully_peeled=True,
    ):
        return 'treewidth_dp_peeled'
    if _prefer_hybrid_contraction_phase3(
        q,
        cover,
        order,
        width,
        allow_tensor_contraction=allow_tensor_contraction,
    ):
        return 'hybrid_contraction'
    if _prefer_tensor_contraction_phase3(
        q,
        cover,
        order,
        width,
        allow_tensor_contraction=allow_tensor_contraction,
    ):
        return 'tensor_contraction'
    if _prefer_treewidth_phase3(q, cover, order, width, fully_peeled=fully_peeled):
        return 'treewidth_dp'
    if _prefer_cubic_contraction_phase3(
        q,
        cover,
        order,
        width,
        fully_peeled=fully_peeled,
    ):
        return 'cubic_contraction_cpu'
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
    """Return a runtime-oriented score for a concrete Phase-3 backend.

    Lower is better. Estimated backend work dominates; backend rank only breaks
    ties between similarly-sized plans.
    """
    if backend == "treewidth_dp_peeled":
        work = max(1, int(_estimate_treewidth_dp_work(q, order)))
        return (0, work, int(width), len(cover), int(structural_obstruction))
    if backend == "treewidth_dp":
        work = max(1, int(_estimate_treewidth_dp_work(q, order)))
        return (1, work, int(width), len(cover), int(structural_obstruction))
    if backend == "cubic_contraction_cpu":
        work = max(1, q.n * (1 << max(0, int(width))))
        return (2, work, int(width), len(cover), int(structural_obstruction))
    if backend == "q3_separator":
        separator_size = len(tuple(separator or ()))
        work = max(1, int(_estimate_q3_separator_work(q, tuple(separator or ()))))
        return (3, work, separator_size, len(cover), int(structural_obstruction))
    if backend == "q3_cover":
        work = max(1, int(_estimate_q3_cover_work(q, len(cover))))
        return (3, work, len(cover), len(cover), int(structural_obstruction))
    return (9, 1 << 62, 1 << 30, len(cover), int(structural_obstruction))


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
    candidates: list[tuple[tuple[int, int, int, int, int], str, tuple[int, ...] | None]] = []

    if fully_peeled and _prefer_treewidth_phase3(
        q,
        cover,
        order,
        width,
        fully_peeled=True,
    ):
        candidates.append((
            _phase3_backend_runtime_score(
                q,
                cover,
                order,
                width,
                structural_obstruction,
                "treewidth_dp_peeled",
                fully_peeled=True,
            ),
            "treewidth_dp_peeled",
            None,
        ))
    elif _prefer_treewidth_phase3(q, cover, order, width, fully_peeled=fully_peeled):
        candidates.append((
            _phase3_backend_runtime_score(
                q,
                cover,
                order,
                width,
                structural_obstruction,
                "treewidth_dp",
                fully_peeled=fully_peeled,
            ),
            "treewidth_dp",
            None,
        ))

    if _prefer_cubic_contraction_phase3(
        q,
        cover,
        order,
        width,
        fully_peeled=fully_peeled,
    ):
        candidates.append((
            _phase3_backend_runtime_score(
                q,
                cover,
                order,
                width,
                structural_obstruction,
                "cubic_contraction_cpu",
                fully_peeled=fully_peeled,
            ),
            "cubic_contraction_cpu",
            None,
        ))

    separator = None
    if _should_apply_extended_q3_reductions(q, extended_reductions):
        separator = _find_small_q3_separator(q)
    if separator is not None and len(separator) < len(cover):
        candidates.append((
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
            "q3_separator",
            tuple(separator),
        ))

    candidates.append((
        _phase3_backend_runtime_score(
            q,
            cover,
            order,
            width,
            structural_obstruction,
            "q3_cover",
            fully_peeled=fully_peeled,
        ),
        "q3_cover",
        None,
    ))

    best_score, best_backend, best_separator = min(candidates, key=lambda item: item[0])
    return best_backend, best_score, best_separator


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
    structural_obstruction = min(core_cover_size, width) if q.q3 else 0
    fully_peeled = bool(peel_order) and not core_vars
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


def _build_cubic_factors(q):
    """Convert q into local complex factors over sorted variable scopes."""
    factors = {}
    scalar = cmath.exp(2j * cmath.pi * float(q.q0))
    omega = _omega_table(q.level)

    for var, coeff in enumerate(q.q1):
        coeff %= q.mod_q1
        if coeff:
            scalar *= _combine_factor(factors, (var,), [1.0 + 0j, omega[coeff]])

    for (i, j), coeff in q.q2.items():
        coeff %= q.mod_q2
        if coeff:
            shift = ((q.mod_q1 // q.mod_q2) * coeff) % q.mod_q1
            scalar *= _combine_factor(
                factors,
                (i, j),
                [
                    1.0 + 0j,
                    1.0 + 0j,
                    1.0 + 0j,
                    omega[shift],
                ],
            )

    for (i, j, k), coeff in q.q3.items():
        coeff %= q.mod_q3
        if coeff:
            table = [1.0 + 0j] * 8
            table[7] = omega[((q.mod_q1 // q.mod_q3) * coeff) % q.mod_q1]
            scalar *= _combine_factor(factors, (i, j, k), table)

    return scalar, factors


def _build_cubic_factors_scaled(q):
    """Scaled-complex companion to ``_build_cubic_factors``."""
    factors = {}
    scalar = _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q.q0)))
    omega_scaled = _omega_scaled_table(q.level)

    for var, coeff in enumerate(q.q1):
        coeff %= q.mod_q1
        if coeff:
            scalar = _mul_scaled_complex(
                scalar,
                _combine_factor_scaled(
                    factors,
                    (var,),
                    [_ONE_SCALED, omega_scaled[coeff]],
                ),
            )

    for (i, j), coeff in q.q2.items():
        coeff %= q.mod_q2
        if coeff:
            shift = ((q.mod_q1 // q.mod_q2) * coeff) % q.mod_q1
            scalar = _mul_scaled_complex(
                scalar,
                _combine_factor_scaled(
                    factors,
                    (i, j),
                    [
                        _ONE_SCALED,
                        _ONE_SCALED,
                        _ONE_SCALED,
                        omega_scaled[shift],
                    ],
                ),
            )

    for (i, j, k), coeff in q.q3.items():
        coeff %= q.mod_q3
        if coeff:
            table = [_ONE_SCALED] * 8
            table[7] = omega_scaled[((q.mod_q1 // q.mod_q3) * coeff) % q.mod_q1]
            scalar = _mul_scaled_complex(
                scalar,
                _combine_factor_scaled(factors, (i, j, k), table),
            )

    return scalar, factors


def _freeze_complex_factor_tables(
    factors: dict[tuple[int, ...], Sequence[complex]],
) -> MappingProxyType:
    return MappingProxyType({
        tuple(scope): tuple(complex(entry) for entry in table)
        for scope, table in factors.items()
    })


def _build_cached_cubic_factors(q) -> tuple[complex, MappingProxyType]:
    cache_key = _q_key(q)
    cached = _STRUCTURE_PHASE3_FACTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    scalar, factors = _build_cubic_factors(q)
    cached = (complex(scalar), _freeze_complex_factor_tables(factors))
    _STRUCTURE_PHASE3_FACTOR_CACHE[cache_key] = cached
    return cached


def _freeze_scaled_factor_tables(
    factors: dict[tuple[int, ...], Sequence[ScaledComplex]],
) -> MappingProxyType:
    return MappingProxyType({
        tuple(scope): tuple(tuple(entry) for entry in table)
        for scope, table in factors.items()
    })


def _build_cached_phase3_treewidth_factor_plan_scaled(
    q,
) -> tuple[ScaledComplex, MappingProxyType]:
    cache_key = _q_key(q)
    cached = _STRUCTURE_PHASE3_TREEWIDTH_FACTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    scalar, factors = _build_cubic_factors_scaled(q)
    cached = (scalar, _freeze_scaled_factor_tables(factors))
    _STRUCTURE_PHASE3_TREEWIDTH_FACTOR_CACHE[cache_key] = cached
    return cached


def _build_native_phase3_treewidth_plan(
    *,
    q,
    order: Sequence[int],
) -> object | None:
    if (
        _schur_native is None
        or not hasattr(_schur_native, "build_scaled_factor_treewidth_plan")
    ):
        return None
    cache_key = (_q_key(q), tuple(int(var) for var in order))
    cached = _STRUCTURE_PHASE3_TREEWIDTH_NATIVE_PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    scalar, factors = _build_cached_phase3_treewidth_factor_plan_scaled(q)
    del scalar
    try:
        native_plan = _schur_native.build_scaled_factor_treewidth_plan(
            int(q.n),
            dict(factors),
            tuple(int(var) for var in order),
        )
    except Exception:
        return None
    _STRUCTURE_PHASE3_TREEWIDTH_NATIVE_PLAN_CACHE[cache_key] = native_plan
    return native_plan


def _build_native_level3_phase3_treewidth_plan(
    *,
    q,
    order: Sequence[int],
) -> object | None:
    if (
        _schur_native is None
        or not hasattr(_schur_native, "build_level3_treewidth_plan")
    ):
        return None
    cache_key = (_q_key(q), tuple(int(var) for var in order))
    cached = _STRUCTURE_PHASE3_LEVEL3_NATIVE_PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        native_plan = _schur_native.build_level3_treewidth_plan(
            int(q.n),
            tuple(int(coeff) for coeff in q.q1),
            q.q2,
            q.q3,
            tuple(int(var) for var in order),
        )
    except Exception:
        return None
    _STRUCTURE_PHASE3_LEVEL3_NATIVE_PLAN_CACHE[cache_key] = native_plan
    return native_plan


def _maybe_get_native_level3_phase3_treewidth_plan(
    *,
    q,
    order: Sequence[int],
) -> object | None:
    cache_key = (_q_key(q), tuple(int(var) for var in order))
    cached = _STRUCTURE_PHASE3_LEVEL3_NATIVE_PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if _STRUCTURE_PHASE3_LEVEL3_NATIVE_PLAN_SEEN_CACHE.get(cache_key) is None:
        _STRUCTURE_PHASE3_LEVEL3_NATIVE_PLAN_SEEN_CACHE[cache_key] = True
        return None
    return _build_native_level3_phase3_treewidth_plan(q=q, order=order)


def _factor_table_to_tensor_data(scope, table):
    """Reshape a factor table into a tensor with the same bit ordering."""
    if not scope:
        return np.asarray(table, dtype=np.complex128).reshape(())
    return np.asarray(table, dtype=np.complex128).reshape((2,) * len(scope), order="F")


def _sum_via_tensor_contraction(q, optimize=_Q3_TENSOR_CONTRACTION_OPTIMIZE):
    del q, optimize
    raise RuntimeError("Tensor-contraction Phase 3 has been removed from TerKet.")


def _build_reduced_tensor_network(q):
    del q
    raise RuntimeError("Hybrid tensor-contraction Phase 3 has been removed from TerKet.")


def _contract_reduced_network(
    tn,
    const,
    *,
    width_hint: int | None = None,
    max_repeats: int = 128,
):
    del tn, const, width_hint, max_repeats
    raise RuntimeError("Hybrid tensor-contraction Phase 3 has been removed from TerKet.")


def _sum_via_treewidth_dp(q, order):
    """
    Exact cubic sum by factor elimination along a low-width order.

    This is equivalent to dynamic programming over the tree decomposition
    induced by the elimination order.
    """
    if _native_level3_enabled(q):
        core_total, max_scope = _schur_native.sum_treewidth_dp_level3(
            q.n,
            q.q1,
            q.q2,
            q.q3,
            order,
        )
        return cmath.exp(2j * cmath.pi * float(q.q0)) * complex(core_total), max_scope

    scalar, factors = _build_cubic_factors(q)
    max_scope = 0

    for var in order:
        bucket_scopes = [scope for scope in factors if var in scope]
        if not bucket_scopes:
            scalar *= 2.0
            max_scope = max(max_scope, 1)
            continue

        bucket = [(scope, factors.pop(scope)) for scope in bucket_scopes]
        union_scope = tuple(sorted({vertex for scope, _ in bucket for vertex in scope}))
        max_scope = max(max_scope, len(union_scope))

        var_pos = union_scope.index(var)
        new_scope = tuple(vertex for vertex in union_scope if vertex != var)
        positions = [
            tuple(union_scope.index(vertex) for vertex in scope)
            for scope, _ in bucket
        ]

        new_table = [0j] * (1 << len(new_scope))
        for reduced_assignment in range(1 << len(new_scope)):
            total = 0j
            for fixed_value in [0, 1]:
                full_assignment = (
                    (reduced_assignment & ((1 << var_pos) - 1))
                    | (fixed_value << var_pos)
                    | ((reduced_assignment >> var_pos) << (var_pos + 1))
                )
                weight = 1.0 + 0j
                for (_, table), pos in zip(bucket, positions):
                    weight *= table[_project_assignment_bits(full_assignment, pos)]
                total += weight
            new_table[reduced_assignment] = total

        scalar *= _combine_factor(factors, new_scope, new_table)

    assert not factors, "All variables should be eliminated by the DP order."
    return scalar, max_scope


def _sum_via_treewidth_dp_scaled(q, order):
    """Scaled-complex companion to ``_sum_via_treewidth_dp``."""
    if _native_level3_enabled(q):
        core_total, max_scope = _schur_native.sum_treewidth_dp_level3(
            q.n,
            q.q1,
            q.q2,
            q.q3,
            order,
        )
        return _make_scaled_complex(
            cmath.exp(2j * cmath.pi * float(q.q0)) * complex(core_total),
        ), max_scope

    scalar, factors = _build_cubic_factors_scaled(q)
    return _sum_factor_tables_scaled(q.n, factors, order, scalar=scalar)


def _sum_via_treewidth_dp_peeled_scaled(q, order):
    """Cached/native exact DP for fully peeled cubic kernels."""
    if _native_level3_enabled(q):
        core_total, max_scope = _schur_native.sum_treewidth_dp_level3(
            q.n,
            q.q1,
            q.q2,
            q.q3,
            order,
        )
        return _make_scaled_complex(
            cmath.exp(2j * cmath.pi * float(q.q0)) * complex(core_total),
        ), max_scope
    scalar, factors = _build_cached_phase3_treewidth_factor_plan_scaled(q)
    native_plan = _build_native_phase3_treewidth_plan(q=q, order=order)
    if (
        native_plan is not None
        and _schur_native is not None
        and hasattr(_schur_native, "sum_scaled_factor_treewidth_preplanned")
    ):
        try:
            core_total, max_scope = _schur_native.sum_scaled_factor_treewidth_preplanned(native_plan)
            return _mul_scaled_complex(scalar, (complex(core_total[0]), int(core_total[1]))), int(max_scope)
        except Exception:
            pass
    return _sum_factor_tables_scaled(q.n, factors, order, scalar=scalar)


def _sum_via_treewidth_dp_peeled(q, order):
    total, max_scope = _sum_via_treewidth_dp_peeled_scaled(q, order)
    return _scaled_to_complex(total), max_scope


def _evaluate_half_phase_mediator_plan_scaled(
    mediator_plan: _HalfPhaseMediatorPlan,
    q1_local: Sequence[int],
) -> ScaledComplex:
    """Evaluate one exact mediator-eliminated component under a concrete q1."""
    if len(q1_local) != len(mediator_plan.core_vars) + len(mediator_plan.mediators):
        # The local q1 vector still uses the original component indexing.
        expected = max(
            (max(mediator_plan.core_vars, default=-1) + 1),
            (max((spec.mediator_var for spec in mediator_plan.mediators), default=-1) + 1),
        )
        if len(q1_local) < expected:
            raise ValueError(
                f"Expected q1_local to cover mediator-plan indices through {expected - 1}, "
                f"received length {len(q1_local)}."
            )

    core_q = _phase_function_from_parts(
        len(mediator_plan.core_vars),
        level=mediator_plan.level,
        q0=Fraction(0),
        q1=[q1_local[var] for var in mediator_plan.core_vars],
        q2=mediator_plan.core_q2,
        q3={},
    )
    scalar, factors = _build_cubic_factors_scaled(core_q)
    omega = _omega_table(mediator_plan.level)

    for spec in mediator_plan.mediators:
        residue = int(q1_local[spec.mediator_var]) % (1 << mediator_plan.level)
        even_value = _make_scaled_complex(1.0 + omega[residue])
        odd_value = _make_scaled_complex(1.0 - omega[residue])
        if len(spec.neighbor_vars) == 0:
            scalar = _mul_scaled_complex(scalar, even_value)
            continue
        if len(spec.neighbor_vars) == 1:
            scalar = _mul_scaled_complex(
                scalar,
                _combine_factor_scaled(
                    factors,
                    spec.neighbor_vars,
                    [even_value, odd_value],
                ),
            )
            continue
        scalar = _mul_scaled_complex(
            scalar,
            _combine_factor_scaled(
                factors,
                spec.neighbor_vars,
                [even_value, odd_value, odd_value, even_value],
            ),
        )

    total, _ = _sum_factor_tables_scaled(
        len(mediator_plan.core_vars),
        factors,
        mediator_plan.order,
        scalar=scalar,
    )
    return total


def _evaluate_generic_q2_mediator_plan_scaled(
    mediator_plan: _GenericQ2MediatorPlan,
    q1_local: Sequence[int],
) -> ScaledComplex:
    """Evaluate one exact arbitrary-q2 mediator-eliminated component."""
    if len(q1_local) != len(mediator_plan.core_vars) + len(mediator_plan.mediators):
        expected = max(
            (max(mediator_plan.core_vars, default=-1) + 1),
            (max((spec.mediator_var for spec in mediator_plan.mediators), default=-1) + 1),
        )
        if len(q1_local) < expected:
            raise ValueError(
                f"Expected q1_local to cover mediator-plan indices through {expected - 1}, "
                f"received length {len(q1_local)}."
            )

    core_q = _phase_function_from_parts(
        len(mediator_plan.core_vars),
        level=mediator_plan.level,
        q0=Fraction(0),
        q1=[q1_local[var] for var in mediator_plan.core_vars],
        q2=mediator_plan.core_q2,
        q3={},
    )
    scalar, factors = _build_cubic_factors_scaled(core_q)
    omega_scaled = _omega_scaled_table(mediator_plan.level)
    mod_q1 = 1 << mediator_plan.level
    mod_q2 = max(1, 1 << (mediator_plan.level - 1))
    q2_lift = mod_q1 // mod_q2 if mod_q2 else 0

    for spec in mediator_plan.mediators:
        base_residue = int(q1_local[spec.mediator_var]) % mod_q1
        table: list[ScaledComplex] = []
        for assignment in range(1 << len(spec.neighbor_vars)):
            residue = base_residue
            for neighbor_idx, coeff in enumerate(spec.neighbor_couplings):
                if (assignment >> neighbor_idx) & 1:
                    residue = (residue + (q2_lift * int(coeff))) % mod_q1
            table.append(_add_scaled_complex(_ONE_SCALED, omega_scaled[residue]))
        scalar = _mul_scaled_complex(
            scalar,
            _combine_factor_scaled(factors, spec.neighbor_vars, table),
        )

    total, _ = _sum_factor_tables_scaled(
        len(mediator_plan.core_vars),
        factors,
        mediator_plan.order,
        scalar=scalar,
    )
    return total


def _greedy_q3_vertex_cover(n_vars, edge_masks, edge_cover_masks, remaining_edges_mask=None):
    """Max-degree greedy cover used as an incumbent and large-instance fallback."""
    if remaining_edges_mask is None:
        remaining_edges_mask = (1 << len(edge_masks)) - 1

    chosen = []
    chosen_mask = 0
    while remaining_edges_mask:
        best_var = -1
        best_score = None
        for var in range(n_vars):
            if chosen_mask & (1 << var):
                continue
            covered = edge_cover_masks[var] & remaining_edges_mask
            if not covered:
                continue
            score = (covered.bit_count(), -var)
            if best_score is None or score > best_score:
                best_var = var
                best_score = score
        if best_var < 0:
            raise RuntimeError("Failed to build a q3 vertex cover.")
        chosen.append(best_var)
        chosen_mask |= 1 << best_var
        remaining_edges_mask &= ~edge_cover_masks[best_var]
    return sorted(chosen)


def _approximate_q3_vertex_cover(n_vars, edge_masks, edge_cover_masks):
    """
    Return a small q3 hypergraph cover.

    We keep the better of:
      * max-degree greedy, which is usually tighter in practice, and
      * the trivial 3-approximation that takes all vertices of uncovered edges.
    """
    greedy = _greedy_q3_vertex_cover(n_vars, edge_masks, edge_cover_masks)

    remaining_edges_mask = (1 << len(edge_masks)) - 1
    chosen_mask = 0
    while remaining_edges_mask:
        edge_bit = remaining_edges_mask & -remaining_edges_mask
        edge_idx = edge_bit.bit_length() - 1
        edge_mask = edge_masks[edge_idx]
        chosen_mask |= edge_mask

        covered_edges = 0
        vertices = edge_mask
        while vertices:
            vertex_bit = vertices & -vertices
            var = vertex_bit.bit_length() - 1
            covered_edges |= edge_cover_masks[var]
            vertices ^= vertex_bit
        remaining_edges_mask &= ~covered_edges

    three_approx = [var for var in range(n_vars) if chosen_mask & (1 << var)]
    return greedy if len(greedy) <= len(three_approx) else three_approx


def _q3_packing_lower_bound(remaining_edges_mask, edge_conflicts):
    """Lower-bound the cover size via a greedy packing of disjoint hyperedges."""
    packing = 0
    remaining = remaining_edges_mask
    while remaining:
        best_edge = -1
        best_conflicts = None
        probe = remaining
        while probe:
            edge_bit = probe & -probe
            edge_idx = edge_bit.bit_length() - 1
            conflicts = (edge_conflicts[edge_idx] & remaining).bit_count()
            if best_conflicts is None or conflicts < best_conflicts:
                best_edge = edge_idx
                best_conflicts = conflicts
            probe ^= edge_bit
        remaining &= ~edge_conflicts[best_edge]
        packing += 1
    return packing


def _pick_q3_branch_edge(remaining_edges_mask, edge_masks, edge_cover_masks):
    """Choose an uncovered hyperedge whose endpoints cover many remaining edges."""
    best_edge = -1
    best_score = None
    probe = remaining_edges_mask
    while probe:
        edge_bit = probe & -probe
        edge_idx = edge_bit.bit_length() - 1
        counts = []
        vertices = edge_masks[edge_idx]
        while vertices:
            vertex_bit = vertices & -vertices
            var = vertex_bit.bit_length() - 1
            counts.append((edge_cover_masks[var] & remaining_edges_mask).bit_count())
            vertices ^= vertex_bit
        counts.sort(reverse=True)
        score = tuple(counts)
        if best_score is None or score > best_score:
            best_edge = edge_idx
            best_score = score
        probe ^= edge_bit
    return best_edge


def _minimum_q3_vertex_cover_uncached(q):
    """Exact minimum q3-hypergraph cover on small cores, heuristic otherwise."""
    if not q.q3:
        return []

    return _minimum_vertex_cover_from_edge_masks(
        q.n,
        [((1 << i) | (1 << j) | (1 << k)) for i, j, k in q.q3],
        exact_size_cutoff=_Q3_VERTEX_COVER_EXACT_SIZE_CUTOFF,
        exact_edge_cutoff=_Q3_VERTEX_COVER_EXACT_EDGE_CUTOFF,
    )


def _minimum_vertex_cover_from_edge_masks(
    n_vars: int,
    edge_masks: Sequence[int],
    *,
    exact_size_cutoff: int,
    exact_edge_cutoff: int,
) -> list[int]:
    """Exact minimum vertex cover on small hypergraphs, heuristic otherwise."""
    if not edge_masks:
        return []

    edge_cover_masks = [0] * n_vars
    for edge_idx, edge_mask in enumerate(edge_masks):
        vertices = edge_mask
        while vertices:
            vertex_bit = vertices & -vertices
            var = vertex_bit.bit_length() - 1
            edge_cover_masks[var] |= 1 << edge_idx
            vertices ^= vertex_bit

    greedy_cover = _approximate_q3_vertex_cover(n_vars, edge_masks, edge_cover_masks)
    if (
        len(greedy_cover) > exact_size_cutoff
        or len(edge_masks) > exact_edge_cutoff
    ):
        return greedy_cover

    edge_conflicts = [0] * len(edge_masks)
    for edge_idx, edge_mask in enumerate(edge_masks):
        edge_conflicts[edge_idx] |= 1 << edge_idx
        for other_idx in range(edge_idx):
            if edge_mask & edge_masks[other_idx]:
                edge_conflicts[edge_idx] |= 1 << other_idx
                edge_conflicts[other_idx] |= 1 << edge_idx

    all_edges_mask = (1 << len(edge_masks)) - 1
    lower_bound = _q3_packing_lower_bound(all_edges_mask, edge_conflicts)
    if lower_bound == len(greedy_cover):
        return greedy_cover

    failed_states = set()

    def search(remaining_edges_mask, budget):
        if not remaining_edges_mask:
            return ()
        if budget == 0:
            return None
        state = (remaining_edges_mask, budget)
        if state in failed_states:
            return None
        if _q3_packing_lower_bound(remaining_edges_mask, edge_conflicts) > budget:
            failed_states.add(state)
            return None

        edge_idx = _pick_q3_branch_edge(remaining_edges_mask, edge_masks, edge_cover_masks)
        vertices = []
        vertex_mask = edge_masks[edge_idx]
        while vertex_mask:
            vertex_bit = vertex_mask & -vertex_mask
            var = vertex_bit.bit_length() - 1
            vertices.append(var)
            vertex_mask ^= vertex_bit
        vertices.sort(
            key=lambda var: ((edge_cover_masks[var] & remaining_edges_mask).bit_count(), -var),
            reverse=True,
        )

        for var in vertices:
            result = search(remaining_edges_mask & ~edge_cover_masks[var], budget - 1)
            if result is not None:
                return (var,) + result

        failed_states.add(state)
        return None

    for budget in range(lower_bound, len(greedy_cover)):
        result = search(all_edges_mask, budget)
        if result is not None:
            return sorted(result)
    return greedy_cover


def _minimum_q3_vertex_cover(q):
    cache_key = _q_structure_key(q)
    cached = _STRUCTURE_Q3_COVER_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)
    cover = tuple(_minimum_q3_vertex_cover_uncached(q))
    _STRUCTURE_Q3_COVER_CACHE[cache_key] = cover
    return list(cover)


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


@dataclass(frozen=True)
class Q3FreeBranchTemplate:
    """Shared residue updates for exact q3-cover branch batching."""

    cover_vars: tuple[int, ...]
    remaining_vars: tuple[int, ...]
    n_cover: int
    n_remaining: int
    mod_q1: int
    level: int
    base_q0_residue: int
    base_q1_residue: np.ndarray
    pair_left: np.ndarray
    pair_right: np.ndarray
    base_q2_residue: np.ndarray
    cover_q1_residue: np.ndarray
    cover_remaining_q2_residue: np.ndarray
    cover_cover_left: np.ndarray
    cover_cover_right: np.ndarray
    cover_cover_residue: np.ndarray
    cubic_pair_cover: np.ndarray
    cubic_pair_index: np.ndarray
    cubic_pair_residue: np.ndarray
    cubic_linear_cover_left: np.ndarray
    cubic_linear_cover_right: np.ndarray
    cubic_linear_var: np.ndarray
    cubic_linear_residue: np.ndarray
    cubic_constant_left: np.ndarray
    cubic_constant_middle: np.ndarray
    cubic_constant_right: np.ndarray
    cubic_constant_residue: np.ndarray


def _as_int64_array(values) -> np.ndarray:
    if not values:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(values, dtype=np.int64)


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

    pair_left = _as_int64_array([left for left, _ in pair_keys])
    pair_right = _as_int64_array([right for _, right in pair_keys])
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
        base_q1_residue=base_q1_residue,
        pair_left=pair_left,
        pair_right=pair_right,
        base_q2_residue=base_q2_residue,
        cover_q1_residue=cover_q1_residue,
        cover_remaining_q2_residue=cover_remaining_q2_residue,
        cover_cover_left=_as_int64_array(cover_cover_left),
        cover_cover_right=_as_int64_array(cover_cover_right),
        cover_cover_residue=_as_int64_array(cover_cover_residue),
        cubic_pair_cover=_as_int64_array(cubic_pair_cover),
        cubic_pair_index=_as_int64_array(cubic_pair_index),
        cubic_pair_residue=_as_int64_array(cubic_pair_residue),
        cubic_linear_cover_left=_as_int64_array(cubic_linear_cover_left),
        cubic_linear_cover_right=_as_int64_array(cubic_linear_cover_right),
        cubic_linear_var=_as_int64_array(cubic_linear_var),
        cubic_linear_residue=_as_int64_array(cubic_linear_residue),
        cubic_constant_left=_as_int64_array(cubic_constant_left),
        cubic_constant_middle=_as_int64_array(cubic_constant_middle),
        cubic_constant_right=_as_int64_array(cubic_constant_right),
        cubic_constant_residue=_as_int64_array(cubic_constant_residue),
    )


def _branch_assignment_bits(branch_masks: np.ndarray, n_cover: int, xp):
    if n_cover == 0:
        return xp.zeros((len(branch_masks), 0), dtype=xp.int64)
    masks = xp.asarray(branch_masks, dtype=xp.uint64).reshape(-1, 1)
    shifts = xp.arange(n_cover, dtype=xp.uint64).reshape(1, -1)
    return ((masks >> shifts) & 1).astype(xp.int64)


def _q3_cover_branch_chunk_size(template: Q3FreeBranchTemplate, budget_bytes: int | None) -> tuple[int, int]:
    assignment_chunk = 1 << min(template.n_remaining, _Q3_COVER_GPU_ASSIGNMENT_CHUNK_LOG2)
    branch_chunk = min(_Q3_COVER_GPU_BRANCH_CHUNK_MAX, 1 << template.n_cover)
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
    xp=np,
    assignment_chunk_size: int | None = None,
) -> np.ndarray:
    if branch_masks.size == 0:
        return np.zeros(0, dtype=np.complex128)

    if assignment_chunk_size is None:
        assignment_chunk_size = 1 << min(template.n_remaining, _Q3_COVER_GPU_ASSIGNMENT_CHUNK_LOG2)

    branch_bits = _branch_assignment_bits(branch_masks, template.n_cover, xp)
    mod_q1 = int(template.mod_q1)
    omega = xp.asarray(np.asarray(_omega_table(template.level), dtype=np.complex128))

    q0_eff = xp.full(branch_bits.shape[0], int(template.base_q0_residue), dtype=xp.int64)
    q1_eff = xp.broadcast_to(
        xp.asarray(template.base_q1_residue, dtype=xp.int64),
        (branch_bits.shape[0], template.n_remaining),
    ).copy()
    q2_eff = xp.broadcast_to(
        xp.asarray(template.base_q2_residue, dtype=xp.int64),
        (branch_bits.shape[0], template.pair_left.size),
    ).copy()

    if template.cover_q1_residue.size:
        q0_eff = (q0_eff + branch_bits @ xp.asarray(template.cover_q1_residue, dtype=xp.int64)) % mod_q1
    if template.cover_remaining_q2_residue.size:
        q1_eff = (q1_eff + branch_bits @ xp.asarray(template.cover_remaining_q2_residue, dtype=xp.int64)) % mod_q1
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

    totals = xp.zeros(branch_bits.shape[0], dtype=xp.complex128)
    n_assignments = 1 << template.n_remaining

    for start in range(0, n_assignments, assignment_chunk_size):
        stop = min(n_assignments, start + assignment_chunk_size)
        assignments = xp.arange(start, stop, dtype=xp.uint64).reshape(-1, 1)
        if template.n_remaining:
            shifts = xp.arange(template.n_remaining, dtype=xp.uint64).reshape(1, -1)
            x = ((assignments >> shifts) & 1).astype(xp.int64)
            residues = x @ q1_eff.T
            if template.pair_left.size:
                pair_terms = x[:, xp.asarray(template.pair_left, dtype=xp.int64)] * x[
                    :, xp.asarray(template.pair_right, dtype=xp.int64)
                ]
                residues = residues + pair_terms @ q2_eff.T
        else:
            residues = xp.zeros((stop - start, branch_bits.shape[0]), dtype=xp.int64)
        residues = (residues + q0_eff.reshape(1, -1)) % mod_q1
        totals = totals + omega[residues].sum(axis=0)

    if xp is np:
        return np.asarray(totals, dtype=np.complex128)
    return np.asarray(xp.asnumpy(totals), dtype=np.complex128)


def _sum_via_q3_cover_gpu(q, cover, context=None, *, structural_obstruction=None):
    """GPU-batched exact q3-cover evaluation for moderate cover sizes."""
    del context
    cp = _get_cupy_module()
    if cp is None:
        raise RuntimeError("GPU q3-cover evaluation requires CuPy.")

    template = _build_q3_free_branch_template(q, cover)
    budget = _gpu_memory_budget_bytes(cp)
    branch_chunk_size, assignment_chunk_size = _q3_cover_branch_chunk_size(template, budget)
    n_branches = 1 << template.n_cover
    total = 0.0 + 0.0j
    for start in range(0, n_branches, branch_chunk_size):
        stop = min(n_branches, start + branch_chunk_size)
        branch_masks = np.arange(start, stop, dtype=np.uint64)
        chunk = _evaluate_q3_free_branch_template_batch(
            template,
            branch_masks,
            xp=cp,
            assignment_chunk_size=assignment_chunk_size,
        )
        total += complex(chunk.sum())

    cubic_obstruction = len(cover) if structural_obstruction is None else structural_obstruction
    return _make_scaled_complex(total), {
        'quad': 0,
        'constraint': 0,
        'branched': len(cover),
        'remaining': len(cover),
        'structural_obstruction': cubic_obstruction,
        'gauss_obstruction': _gauss_obstruction(q, cubic_obstruction),
        'cost_r': len(cover),
        'phase_states': 0,
        'phase_splits': 0,
        'phase3_backend': 'q3_cover_gpu',
    }


def _sum_via_q3_cover(q, context=None, *, structural_obstruction=None):
    """
    Phase-3 fallback: branch on a q3-hypergraph cover, then use q3-free sums.

    The reported obstruction is supplied by the caller so the public metric can
    stay distinct from the operational branch exponent paid by this fallback.
    """
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
            xp=np,
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
    """
    Phase-3 solver for a genuinely cubic residual kernel.

    Prefer a low-treewidth elimination DP when the min-fill width estimate is
    within the hard cutoff and either already beats the q3-cover exponent or
    the cost model says it should win against multi-variable cover branching.
    Otherwise, if the residual kernel is small enough, contract its factor
    graph directly with quimb. Fall back to q3 cover branching for the rest.
    """
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
        backend_name = 'cubic_contraction_cpu'
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
            if context is not None and not context.preserve_scale:
                total_complex, actual_width = _sum_via_treewidth_dp_peeled(q, order)
                total = _make_scaled_complex(total_complex)
            else:
                total, actual_width = _sum_via_treewidth_dp_peeled_scaled(q, order)
        else:
            if context is not None and not context.preserve_scale:
                total_complex, actual_width = _sum_via_treewidth_dp(q, order)
                total = _make_scaled_complex(total_complex)
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
    if backend in {"cubic_contraction_cpu", "cubic_contraction_gpu"}:
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

    if selected_backend == "cubic_contraction_cpu":
        return run_cubic_contraction()

    if selected_backend == "q3_separator" and selected_separator is not None:
        return _sum_via_q3_separator(
            q,
            selected_separator,
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


# ==================================================================
# Variable classification [BL26 Prop. 12]
# ==================================================================

def _build_classification_data(q):
    cached_on_q = getattr(q, "_schur_q_classification_data", None)
    if cached_on_q is not None:
        return cached_on_q

    # Mutable build-time kernels churn quickly and almost never hit the
    # structure caches. Skip the cache-key / normalization overhead on that
    # path and use the native classifier output directly when available.
    if getattr(q, "_schur_mutable", True):
        if _schur_native is not None and q.level == 3:
            native_result = _schur_native.build_classification_data(q.n, q.q2, q.q3)
            if (
                len(native_result) == 3
                and len(native_result[0]) == q.n
                and len(native_result[1]) == q.n
                and len(native_result[2]) == q.n
            ):
                return native_result

        mod_q2 = max(1, 1 << (q.level - 1))
        mod_q3 = max(1, 1 << (q.level - 2))
        odd_bilinear = [False] * q.n
        parity_partners = [[] for _ in range(q.n)]
        cubic_incidence = [False] * q.n
        parity_residue = mod_q2 // 2 if mod_q2 > 1 else 0

        for (i, j), coeff in q.q2.items():
            residue = coeff % mod_q2
            if residue % 2:
                odd_bilinear[i] = True
                odd_bilinear[j] = True
            if parity_residue and residue == parity_residue:
                parity_partners[i].append(j)
                parity_partners[j].append(i)

        for (i, j, l), coeff in q.q3.items():
            if coeff % mod_q3:
                cubic_incidence[i] = True
                cubic_incidence[j] = True
                cubic_incidence[l] = True

        for partners in parity_partners:
            partners.sort()

        return (cubic_incidence, odd_bilinear, parity_partners)

    cache_key = _q_classification_structure_key(q)
    cached = _STRUCTURE_CLASSIFICATION_DATA_CACHE.get(cache_key)
    if cached is not None:
        if not getattr(q, "_schur_mutable", True):
            q._schur_q_classification_data = cached
        return cached

    if _schur_native is not None and q.level == 3:
        cubic_incidence, odd_bilinear, parity_partners = _schur_native.build_classification_data(q.n, q.q2, q.q3)
        if not (
            len(cubic_incidence) == q.n
            and len(odd_bilinear) == q.n
            and len(parity_partners) == q.n
        ):
            cubic_incidence = odd_bilinear = parity_partners = None
    else:
        cubic_incidence = odd_bilinear = parity_partners = None

    if cubic_incidence is None:
        mod_q2 = max(1, 1 << (q.level - 1))
        mod_q3 = max(1, 1 << (q.level - 2))
        odd_bilinear = [False] * q.n
        parity_partners = [[] for _ in range(q.n)]
        cubic_incidence = [False] * q.n
        parity_residue = mod_q2 // 2 if mod_q2 > 1 else 0

        for (i, j), coeff in q.q2.items():
            residue = coeff % mod_q2
            if residue % 2:
                odd_bilinear[i] = True
                odd_bilinear[j] = True
            if parity_residue and residue == parity_residue:
                parity_partners[i].append(j)
                parity_partners[j].append(i)

        for (i, j, l), coeff in q.q3.items():
            if coeff % mod_q3:
                cubic_incidence[i] = True
                cubic_incidence[j] = True
                cubic_incidence[l] = True

        for partners in parity_partners:
            partners.sort()

    result = (
        tuple(bool(value) for value in cubic_incidence),
        tuple(bool(value) for value in odd_bilinear),
        tuple(tuple(int(partner) for partner in partners) for partners in parity_partners),
    )
    _STRUCTURE_CLASSIFICATION_DATA_CACHE[cache_key] = result
    if not getattr(q, "_schur_mutable", True):
        q._schur_q_classification_data = result
    return result


def _classification_lookup(q):
    cached_on_q = getattr(q, "_schur_q_classification_lookup", None)
    if cached_on_q is not None:
        return cached_on_q

    cache_key = _q_classification_structure_key(q)
    cached = _STRUCTURE_CLASSIFICATION_LOOKUP_CACHE.get(cache_key)
    if cached is not None:
        if not getattr(q, "_schur_mutable", True):
            q._schur_q_classification_lookup = cached
        return cached

    mod_q1 = 1 << q.level
    threshold = max(1, mod_q1 // 4)
    cubic_incidence, odd_bilinear, parity_partners = _build_classification_data(q)

    if _schur_native is not None and q.level == 3:
        native_result = _schur_native.build_classification_lookup(
            q.n,
            q.level,
            cubic_incidence,
            odd_bilinear,
            parity_partners,
        )
        if len(native_result) == q.n:
            result = tuple(native_result)
            _STRUCTURE_CLASSIFICATION_LOOKUP_CACHE[cache_key] = result
            if not getattr(q, "_schur_mutable", True):
                q._schur_q_classification_lookup = result
            return result

    lookup = []
    for var in range(q.n):
        partners = parity_partners[var]
        var_entries = []
        for coeff in range(mod_q1):
            if coeff % threshold != 0 or cubic_incidence[var]:
                var_entries.append((_CLASS_CUBIC,))
                continue
            reduced = (coeff // threshold) % 4
            if reduced in (1, 3):
                var_entries.append((_CLASS_QUADRATIC, coeff, bool(odd_bilinear[var])))
                continue
            if odd_bilinear[var]:
                var_entries.append((_CLASS_CUBIC,))
                continue
            if reduced == 0 and not partners:
                var_entries.append((_CLASS_CONSTRAINT_DECOUPLED,))
            elif reduced == 2 and not partners:
                var_entries.append((_CLASS_CONSTRAINT_ZERO,))
            else:
                var_entries.append((_CLASS_CONSTRAINT_PARITY, partners, coeff))
        lookup.append(tuple(var_entries))
    result = tuple(lookup)
    _STRUCTURE_CLASSIFICATION_LOOKUP_CACHE[cache_key] = result
    if not getattr(q, "_schur_mutable", True):
        q._schur_q_classification_lookup = result
    return result


def _classification_entry(
    q,
    k: int,
    *,
    classification_data=None,
    threshold: int | None = None,
):
    if threshold is None:
        threshold = max(1, q.mod_q1 // 4)
    c = q.q1[k] % q.mod_q1
    if classification_data is not None:
        cubic_incidence, odd_bilinear, parity_partners = classification_data
        has_genuine_cubic = cubic_incidence[k]
    else:
        has_genuine_cubic = any(v and k in (i, j, l) for (i, j, l), v in q.q3.items())
        odd_bilinear = parity_partners = None
    if c % threshold != 0 or has_genuine_cubic:
        return (_CLASS_CUBIC,)

    reduced = (c // threshold) % 4
    if reduced in (1, 3):
        odd_flag = bool(odd_bilinear[k]) if classification_data is not None else _has_odd_bilinear_coupling(q, k)
        return (_CLASS_QUADRATIC, c, odd_flag)
    if classification_data is not None:
        if odd_bilinear[k]:
            return (_CLASS_CUBIC,)
        partners = parity_partners[k]
    else:
        if _has_odd_bilinear_coupling(q, k):
            return (_CLASS_CUBIC,)
        partners = tuple(
            j
            for j in range(q.n)
            if j != k
            and q.q2.get((min(k, j), max(k, j)), 0) % q.mod_q2 == (q.mod_q2 // 2 if q.mod_q2 > 1 else 0)
        )
    if reduced == 0 and not partners:
        return (_CLASS_CONSTRAINT_DECOUPLED,)
    if reduced == 2 and not partners:
        return (_CLASS_CONSTRAINT_ZERO,)
    return (_CLASS_CONSTRAINT_PARITY, partners, c)


def _has_odd_bilinear_coupling(q, k, classification_data=None):
    if classification_data is not None:
        _, odd_bilinear, _ = classification_data
        return odd_bilinear[k]
    return any(
        j != k and q.q2.get((min(k,j),max(k,j)),0) % 2 != 0
        for j in range(q.n)
    )


def _classify(q, k, classification_data=None):
    """
    Classify variable k for single-variable exact elimination.

    The ``'quadratic'`` label is intentionally narrow: it means ``k`` matches
    the Prop. 9 one-variable Gauss-sum rule. It does not mean odd ``q1`` or odd
    ``q2`` make the whole kernel non-Gaussian. Any residual q3-free kernel is
    still summed exactly by ``_gauss_sum_q3_free`` over binary variables.
    """
    entry = _classification_entry(q, k, classification_data=classification_data)
    tag = entry[0]
    if tag == _CLASS_CUBIC:
        return ('cubic', {})
    if tag == _CLASS_QUADRATIC:
        return ('quadratic', {'q1': entry[1]})
    if tag == _CLASS_CONSTRAINT_DECOUPLED:
        return ('constraint', {'type':'decoupled'})
    if tag == _CLASS_CONSTRAINT_ZERO:
        return ('constraint', {'type':'zero'})
    return ('constraint', {'type':'parity','partners':list(entry[1]),'q1':entry[2]})


# ==================================================================
# Quadratic elimination [BL26 Prop. 9]
# ==================================================================

def _incident_quadratic_couplings(q, k: int):
    """Yield ``(neighbor, residue)`` q2 couplings incident on ``k``."""
    for (left, right), coeff in q.q2.items():
        if left == k:
            residue = int(coeff) % q.mod_q2
            if residue:
                yield right, residue
        elif right == k:
            residue = int(coeff) % q.mod_q2
            if residue:
                yield left, residue


def _elim_quadratic(q, k, *, classification_data=None):
    """Gaussian sum over variable k at the current dyadic precision level."""
    nf = q.n
    threshold = _quadratic_residue_threshold(q)
    c = (q.q1[k] // threshold) % 4
    const_phase = Fraction(1,8) if c%4==1 else Fraction(7,8)
    sign = -1 if c%4==1 else +1
    if classification_data is not None and int(q.level) == 3:
        _cubic_incidence, odd_bilinear, parity_partners = classification_data
        coupled = (
            [(int(neighbor), q.mod_q2 // 2) for neighbor in parity_partners[k]]
            if not odd_bilinear[k]
            else list(_incident_quadratic_couplings(q, k))
        )
    else:
        coupled = list(_incident_quadratic_couplings(q, k))

    remap = [-1] * nf
    new_q1 = []
    next_idx = 0
    for var, coeff in enumerate(q.q1):
        if var == k:
            continue
        remap[var] = next_idx
        new_q1.append(coeff)
        next_idx += 1

    nn = nf - 1
    new_q2 = {(remap[i], remap[j]): value for (i, j), value in q.q2.items() if k not in (i, j)}
    new_q3 = {
        (remap[i], remap[j], remap[l]): value
        for (i, j, l), value in q.q3.items()
        if k not in (i, j, l)
    }
    for var, coupling in coupled:                              # linear corr
        new_q1[remap[var]] = (new_q1[remap[var]] + sign * coupling) % q.mod_q1
    for left_pos in range(len(coupled)):                       # [BL26 Eq.194]
        left_var, left_coeff = coupled[left_pos]
        for right_pos in range(left_pos + 1, len(coupled)):
            right_var, right_coeff = coupled[right_pos]
            new_left = remap[left_var]
            new_right = remap[right_var]
            if new_left > new_right:
                new_left, new_right = new_right, new_left
            corr = _quadratic_pair_correction(q, left_coeff, right_coeff)
            if corr:
                edge = (new_left, new_right)
                new_q2[edge] = (new_q2.get(edge, 0) + corr) % q.mod_q2
                if new_q2[edge] == 0:
                    del new_q2[edge]
    return _phase_function_from_parts(
        nn,
        level=q.level,
        q0=(q.q0 + const_phase) % 1,
        q1=new_q1,
        q2=new_q2,
        q3=new_q3,
    ), 1


def _elim_quadratic_via_split(q, k, context=None):
    """Exact elimination fallback for q1_k in {2,6} with odd bilinear couplings."""
    total = _make_scaled_complex(0j)
    nq = 1
    nc = 0
    max_branched = 0
    max_remaining = 0
    max_structural = 0
    max_cost_r = 0
    phase_states = phase_splits = 0
    for val in [0, 1]:
        q_branch = _fix_variable(q, k, val, context=context)
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
        max_cost_r = max(max_cost_r, branch_info.get('cost_r', branch_info['remaining']))
        phase_states += branch_info.get('phase_states', 0)
        phase_splits += branch_info.get('phase_splits', 0)
    return total, {
        'quad': nq,
        'constraint': nc,
        'branched': 1 + max_branched,
        'remaining': max_remaining,
        'structural_obstruction': max_structural,
        'cost_r': max_cost_r,
        'phase_states': phase_states,
        'phase_splits': phase_splits,
    }


# ==================================================================
# Constraint elimination [BL26 Prop. 11]
# ==================================================================

def _elim_constraint(q, k, info, context=None):
    """Character sum: sum exp(chi) = |R|*delta_{chi=0} [BL26 Eq.185]."""
    nf=q.n
    if info.get('type')=='zero': return None
    if info.get('type')=='decoupled':
        remap={}; idx=0
        for j in range(nf):
            if j!=k: remap[j]=idx; idx+=1
        nn=nf-1
        return (
            _phase_function_from_parts(
                nn,
                level=q.level,
                q0=q.q0,
                q1=[q.q1[j] for j in range(nf) if j != k],
                q2={(remap[i], remap[j]): v for (i, j), v in q.q2.items() if k not in (i, j)},
                q3={
                    (remap[i], remap[j], remap[l]): v
                    for (i, j, l), v in q.q3.items()
                    if k not in (i, j, l)
                },
            ),
            2,
        )
    if info.get('type')=='parity':
        partners=info['partners']; c=info['q1']
        p=partners[0]; target=1 if c == (q.mod_q1 // 2) else 0
        if len(partners) == 1:
            return _elim_single_partner_constraint(q, k, p, target)
        nn=nf-2
        gamma=[0] * nf
        idx = 0
        for j in range(nf):
            if j in(k,p):
                continue
            gamma[j] = 1 << idx
            idx += 1
        partner_mask = 0
        for j in partners[1:]:
            if j != k:
                partner_mask ^= gamma[j]
        gamma[p] = partner_mask
        shift_mask = (1 << p) if target else 0
        return _aff_compose_cached(q, shift_mask, gamma, nn, context=context), 2
    return (
        _phase_function_from_parts(
            nf - 1,
            level=q.level,
            q0=q.q0,
            q1=[q.q1[j] for j in range(nf) if j != k],
            q2={},
            q3={},
        ),
        0,
    )


def _elim_single_partner_constraint_python(q, k, p, target):
    """
    Fast path for parity constraints with a single residue-2 partner.

    For ``q1_k in {0, 4}`` and the sole parity partner ``p``, summing over
    ``k`` contributes a factor of ``2`` and fixes ``p = target``. The generic
    affine-compose path is correct but expensive on large structured families
    like Grover because it rebuilds the whole kernel through the full
    substitution machinery at every step.
    """
    nf = q.n
    removed = {k, p}
    remap = {}
    idx = 0
    for j in range(nf):
        if j in removed:
            continue
        remap[j] = idx
        idx += 1

    new_q0 = q.q0
    new_q1 = [q.q1[j] for j in range(nf) if j not in removed]
    new_q2 = {}
    new_q3 = {}

    if target and q.q1[p]:
        new_q0 = (new_q0 + Fraction(q.q1[p], q.mod_q1)) % 1

    for (i, j), coeff in q.q2.items():
        if k in (i, j):
            continue
        if p in (i, j):
            if not target:
                continue
            other = j if i == p else i
            new_q1[remap[other]] = (new_q1[remap[other]] + (q.mod_q1 // q.mod_q2) * coeff) % q.mod_q1
            continue
        key = (remap[i], remap[j])
        new_q2[key] = coeff

    for (i, j, l), coeff in q.q3.items():
        if k in (i, j, l):
            continue
        if p in (i, j, l):
            if not target:
                continue
            others = [var for var in (i, j, l) if var != p]
            a = remap[others[0]]
            b = remap[others[1]]
            if a > b:
                a, b = b, a
            key = (a, b)
            value = (new_q2.get(key, 0) + (q.mod_q2 // q.mod_q3) * coeff) % q.mod_q2
            if value:
                new_q2[key] = value
            elif key in new_q2:
                del new_q2[key]
            continue
        key = (remap[i], remap[j], remap[l])
        new_q3[key] = coeff

    return _phase_function_from_parts(
        nf - 2,
        level=q.level,
        q0=new_q0,
        q1=new_q1,
        q2=new_q2,
        q3=new_q3,
    ), 2


def _elim_single_partner_constraint(q, k, p, target):
    if _native_level3_enabled(q):
        q0_residue = (q.q0.numerator * (q.mod_q1 // q.q0.denominator)) % q.mod_q1
        new_q0_residue, new_q1, new_q2, new_q3 = _schur_native.elim_single_partner_constraint_terms(
            q0_residue,
            q.q1,
            q.q2,
            q.q3,
            k,
            p,
            int(target),
        )
        return _phase_function_from_parts(
            q.n - 2,
            level=q.level,
            q0=_fraction_from_residue(q.level, new_q0_residue),
            q1=new_q1,
            q2=new_q2,
            q3=new_q3,
        ), 2
    return _elim_single_partner_constraint_python(q, k, p, target)


def _elim_two_partner_constraint_q3_free(q, k: int, keep: int, remove: int, target: int):
    """Eliminate ``k`` and substitute ``remove = keep xor target`` in q3-free q."""
    if q.q3:
        return None
    removed = {int(k), int(remove)}
    if keep in removed:
        return None

    remap = {}
    idx = 0
    for var in range(q.n):
        if var in removed:
            continue
        remap[var] = idx
        idx += 1
    if keep not in remap:
        return None

    new_q0 = q.q0
    new_q1 = [q.q1[var] for var in range(q.n) if var not in removed]
    new_q2: dict[tuple[int, int], int] = {}
    keep_new = remap[keep]
    lift = q.mod_q1 // q.mod_q2 if q.mod_q2 else 0
    target = int(target) & 1

    remove_unary = q.q1[remove] % q.mod_q1
    if remove_unary:
        if target:
            new_q0 = (new_q0 + Fraction(remove_unary, q.mod_q1)) % 1
            new_q1[keep_new] = (new_q1[keep_new] - remove_unary) % q.mod_q1
        else:
            new_q1[keep_new] = (new_q1[keep_new] + remove_unary) % q.mod_q1

    for (left, right), coeff in q.q2.items():
        coeff %= q.mod_q2
        if not coeff or k in (left, right):
            continue
        if remove not in (left, right):
            if left not in remap or right not in remap:
                continue
            new_left, new_right = remap[left], remap[right]
            if new_left > new_right:
                new_left, new_right = new_right, new_left
            updated = (new_q2.get((new_left, new_right), 0) + coeff) % q.mod_q2
            if updated:
                new_q2[(new_left, new_right)] = updated
            elif (new_left, new_right) in new_q2:
                del new_q2[(new_left, new_right)]
            continue

        other = right if left == remove else left
        if other == keep:
            if not target:
                new_q1[keep_new] = (new_q1[keep_new] + lift * coeff) % q.mod_q1
            continue
        if other == k or other not in remap:
            continue

        other_new = remap[other]
        if target:
            new_q1[other_new] = (new_q1[other_new] + lift * coeff) % q.mod_q1
            pair_coeff = (-coeff) % q.mod_q2
        else:
            pair_coeff = coeff
        new_left, new_right = keep_new, other_new
        if new_left > new_right:
            new_left, new_right = new_right, new_left
        updated = (new_q2.get((new_left, new_right), 0) + pair_coeff) % q.mod_q2
        if updated:
            new_q2[(new_left, new_right)] = updated
        elif (new_left, new_right) in new_q2:
            del new_q2[(new_left, new_right)]

    return _phase_function_from_parts(
        q.n - 2,
        level=q.level,
        q0=new_q0,
        q1=new_q1,
        q2=new_q2,
        q3={},
    ), 2


# ==================================================================
# Affine composition [generalized BL26 Prop. 4]
# ==================================================================

def _aff_compose_python(q, shift_mask, row_masks, k, q0):
    """Pure Python affine composition fallback."""
    composed = _phase_function_from_parts(k, level=q.level, q0=q0, q1=[0] * k, q2={}, q3={})

    for idx, coeff in enumerate(q.q1):
        if coeff:
            _apply_affine_bit_in_place(composed, row_masks[idx], _mask_bit(shift_mask, idx), coeff)

    for (a, b), coeff in q.q2.items():
        sa = _mask_bit(shift_mask, a)
        sb = _mask_bit(shift_mask, b)
        _apply_affine_bit_in_place(composed, row_masks[a], sa, coeff)
        _apply_affine_bit_in_place(composed, row_masks[b], sb, coeff)
        _apply_affine_bit_in_place(composed, row_masks[a] ^ row_masks[b], sa ^ sb, (-coeff) % q.mod_q1)

    for (a, b, c), coeff in q.q3.items():
        sa = _mask_bit(shift_mask, a)
        sb = _mask_bit(shift_mask, b)
        sc = _mask_bit(shift_mask, c)
        _apply_affine_bit_in_place(composed, row_masks[a], sa, coeff)
        _apply_affine_bit_in_place(composed, row_masks[b], sb, coeff)
        _apply_affine_bit_in_place(composed, row_masks[c], sc, coeff)
        _apply_affine_bit_in_place(composed, row_masks[a] ^ row_masks[b], sa ^ sb, (-coeff) % q.mod_q1)
        _apply_affine_bit_in_place(composed, row_masks[a] ^ row_masks[c], sa ^ sc, (-coeff) % q.mod_q1)
        _apply_affine_bit_in_place(composed, row_masks[b] ^ row_masks[c], sb ^ sc, (-coeff) % q.mod_q1)
        _apply_affine_bit_in_place(
            composed,
            row_masks[a] ^ row_masks[b] ^ row_masks[c],
            sa ^ sb ^ sc,
            coeff,
        )

    return composed


def _aff_compose(q, shift, gamma, k):
    """q(shift + gamma*f) as CubicFunction on k variables.

    Compose algebraically rather than by sampling evaluations. For affine bits
    u, v, w over Z2 we use:

        2uv = u + v - (u xor v)
        4uvw = (u xor v xor w) + u + v + w
               - (u xor v) - (u xor w) - (v xor w)

    so every q2/q3 term reduces to a signed sum of affine parity bits. Each
    affine bit is then scattered directly into q1/q2/q3 coefficients.
    """
    m = q.n
    shift_mask = shift if isinstance(shift, int) else _mask_from_vector(shift)
    row_masks = _row_masks_from_gamma(gamma)
    assert len(row_masks) == m

    if k == 0:
        return _phase_function_from_parts(
            0,
            level=q.level,
            q0=_evaluate_q_from_mask(q, shift_mask),
            q1=[],
            q2={},
            q3={},
        )

    q0 = _evaluate_q_from_mask(q, shift_mask)
    if _native_aff_compose_enabled() and k < _NATIVE_AFF_COMPOSE_Q3_INDEX_LIMIT:
        try:
            new_q1, new_q2, new_q3 = _schur_native.aff_compose_terms(
                q.q1,
                q.q2,
                q.q3,
                shift_mask,
                row_masks,
                k,
                q.mod_q1,
                q.mod_q2,
                q.mod_q3,
            )
        except TypeError:
            if _native_level3_enabled(q):
                new_q1, new_q2, new_q3 = _schur_native.aff_compose_terms(
                    q.q1,
                    q.q2,
                    q.q3,
                    shift_mask,
                    row_masks,
                    k,
                )
            else:
                return _aff_compose_python(q, shift_mask, row_masks, k, q0)
        return _phase_function_from_parts(k, level=q.level, q0=q0, q1=new_q1, q2=new_q2, q3=new_q3)

    return _aff_compose_python(q, shift_mask, row_masks, k, q0)


def _info(
    init: int,
    nq: int,
    nc: int,
    nb: int,
    rem: int,
    structural_obstruction: int | None = None,
    gauss_obstruction: int | None = None,
    phase_states: int = 0,
    phase_splits: int = 0,
    zero: bool = False,
    cost_model_r: int | None = None,
    phase3_backend: str | None = None,
) -> ReductionInfo:
    if cost_model_r is None:
        cost_model_r = rem
    if structural_obstruction is None:
        structural_obstruction = rem
    if gauss_obstruction is None:
        gauss_obstruction = structural_obstruction
    return {'initial_free':init, 'quad_eliminated':nq,
            'constraint_eliminated':nc, 'branched':nb,
            'remaining_free':rem, 'branches':2**rem,
            'cost_model_r':cost_model_r,
            'cubic_obstruction':structural_obstruction,
            'has_cubic_obstruction':structural_obstruction > 0,
            'gauss_obstruction':gauss_obstruction,
            'has_gauss_obstruction':gauss_obstruction > 0,
            'phase_states':phase_states,
            'phase_splits':phase_splits,
            'phase3_backend':phase3_backend,
            'is_zero':zero}


# ==================================================================
# Public API
# ==================================================================

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
    """
    Public reduction routine for cubic phase functions over Z2.

    Set ``allow_tensor_contraction=False`` to keep Phase 3 on the exact
    non-quimb backends (`treewidth_dp` or `q3_cover`). ``extended_reductions``
    accepts ``"auto"``, ``"always"``, or ``"never"`` for the new gate-rewrite
    and q3-specific reduction passes.
    """
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
    state = SchurState(n)
    if input_bits is not None:
        state.eps0 = list(input_bits)
    if global_phase_radians:
        state.scalar *= cmath.exp(1j * float(global_phase_radians))
    if _should_apply_extended_gate_rewrite(extended_reductions, gates):
        gate_sequence = _rewrite_gate_sequence(gates)
    else:
        gate_sequence = tuple(gates)
    for gate in gate_sequence:
        getattr(state, gate[0])(*gate[1:])
    state._flush_pending_dead_variables()
    return state


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
                preserve_scale=preserve_scale,
                allow_tensor_contraction=allow_tensor_contraction,
                extended_reductions=extended_reductions,
            )
            results.append(info if analyze_only else (amplitude, info))
        return results

    if not state.q.q3 and state.m:
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
        reduced_total, elim_info = _reduce_and_sum_scaled(q_free, context=context)
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
        if analyze_only:
            results.append(info)
            continue

        scaled_amp = _normalize_scaled_complex(
            complex(state.scalar) * reduced_total[0],
            reduced_total[1] + state.scalar_half_pow2,
        )
        amp = ScaledAmplitude.from_tuple(scaled_amp) if preserve_scale else _scaled_to_complex(scaled_amp)
        results.append((amp, info))

    return results


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
    """
    Compute an amplitude from a gate list, QASM string, or Qiskit circuit.

    By default this returns ``ScaledAmplitude`` so tiny nonzero amplitudes do
    not collapse to ``0j``. Pass ``as_complex=True`` to request the legacy
    complex-valued return. Set ``allow_tensor_contraction=False`` to disable
    the optional quimb Phase-3 backend and stay on exact non-quimb reducers.
    Pass a ``SolverConfig`` to tune cutset and tensor-hint preferences.
    """
    from .circuits import _circuit_global_phase_radians, normalize_circuit

    spec = normalize_circuit(circuit)
    state = build_state(
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
    """
    Compute amplitudes for multiple output strings while reusing Schur-state work.

    The echelon solve over the output constraint matrix is prepared once and a
    single reduction context is shared across outputs so repeated affine
    restrictions can benefit from memoized reductions.
    Pass a ``SolverConfig`` to tune cutset and tensor-hint preferences.
    """
    from .circuits import _circuit_global_phase_radians, normalize_circuit

    spec = normalize_circuit(circuit)
    state = build_state(
        spec.n_qubits,
        spec.gates,
        input_bits,
        global_phase_radians=_circuit_global_phase_radians(spec),
        extended_reductions=extended_reductions,
    )
    _token = _SOLVER_CONFIG_VAR.set(solver_config) if solver_config is not None else None
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
        if _token is not None:
            _SOLVER_CONFIG_VAR.reset(_token)


def analyze_amplitudes(
    circuit: CircuitInput,
    input_bits: BitSequence,
    output_list: Sequence[BitSequence],
    *,
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
) -> list[ReductionInfo]:
    """Return reduction metadata for multiple outputs without materializing amplitudes."""
    from .circuits import _circuit_global_phase_radians, normalize_circuit

    spec = normalize_circuit(circuit)
    state = build_state(
        spec.n_qubits,
        spec.gates,
        input_bits,
        global_phase_radians=_circuit_global_phase_radians(spec),
        extended_reductions=extended_reductions,
    )
    return _batch_query_state(
        state,
        output_list,
        preserve_scale=False,
        allow_tensor_contraction=allow_tensor_contraction,
        extended_reductions=extended_reductions,
        analyze_only=True,
    )


def analyze_circuit(
    circuit: CircuitInput,
    input_bits: BitSequence,
    output_bits: BitSequence,
    *,
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
) -> ReductionInfo:
    """
    Return reduction statistics for a single amplitude query.

    The ``cubic_obstruction`` entry reports only the residual genuine q3
    obstruction after exact reductions. The broader ``gauss_obstruction``
    entry records the remaining obstruction to a BL26-style quadratic-tensor
    contraction, including any surviving qubit q1/q2 coefficients that lie
    outside BL26's quadratic coefficient groups. The separate ``cost_model_r``
    entry records the runtime exponent paid by the active Phase-3 backend: the
    low-treewidth DP width when that solver is chosen, the residual core size
    on the optional quimb contraction backend, or the q3-cover size on the
    cover fallback.
    The ``phase3_backend`` field records which backend supplied that
    ``cost_model_r`` value.
    The ``branched`` entry is the maximum number of explicit branch variables
    encountered along any recursive reduction path for the query.
    The ``phase_states`` and ``phase_splits`` fields are retained for metadata
    compatibility and are reported as zero by the direct q3-free reducer.
    """
    return analyze_amplitudes(
        circuit,
        input_bits,
        [output_bits],
        allow_tensor_contraction=allow_tensor_contraction,
        extended_reductions=extended_reductions,
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
    ``allow_tensor_contraction=False`` to disable the optional quimb Phase-3
    backend and stay on exact non-quimb reducers.
    """
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
