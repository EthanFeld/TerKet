
"""Reduction context, keys, and structural support helpers."""

from __future__ import annotations

import importlib

from ._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals

_LOCAL_NAMES = {
    '_ReductionContext',
    '_build_early_elim_batch_size',
    '_project_quadratic_elimination_q2_nnz_delta',
    '_should_defer_build_quadratic_elimination',
    '_fraction_from_residue',
    '_q_key_digest',
    '_q_key',
    '_cache_phase_structure_key',
    '_q_structure_key',
    '_q_phase3_structure_key',
    '_q_classification_structure_key',
    '_q_q3_support_key',
    '_q_cubic_treewidth_batch_support_key',
    '_build_cubic_treewidth_batch_support',
    '_phase3_support_plan',
    '_phase3_batch_support_plan_fast',
    '_phase_function_from_parts',
    '_phase_function_from_parts_mutable',

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


_INITIAL_ENGINE = importlib.import_module("terket._engine_impl")
_MISSING, _ENGINE_LOCAL_BASELINES = _bootstrap_extracted_globals(
    globals(),
    local_names=_LOCAL_NAMES,
    local_impls=_LOCAL_IMPLS,
    respect_mock_wraps=True,
)
del _INITIAL_ENGINE

class _ReductionContext:
    """Per-query memo tables for affine substitutions and reduced branch states."""

    __slots__ = (
        "affine_compose_cache",
        "fix_variables_cache",
        "reduce_cache",
        "q3_free_constraint_plan_cache",
        "preserve_scale",
        "allow_tensor_contraction",
        "extended_reductions",
    )

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

def _fraction_from_residue(level: int, residue: int) -> Fraction:
    modulus = 1 << level
    return Fraction(residue % modulus, modulus)

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

def _q_cubic_treewidth_batch_support_key(q):
    hasher = hashlib.blake2b(digest_size=20)
    hasher.update(_PACK_QSTRUCT_HEADER.pack(q.n, q.level))
    for (i, j), coeff in q.q2.items():
        if coeff % q.mod_q2:
            hasher.update(_PACK_QKEY_Q2.pack(i, j, 1))
    for (i, j, k), coeff in q.q3.items():
        if coeff % q.mod_q3:
            hasher.update(_PACK_QKEY_Q3.pack(i, j, k, 1))
    return (q.n, q.level, hasher.digest())

def _build_cubic_treewidth_batch_support(q):
    cache_key = _q_cubic_treewidth_batch_support_key(q)
    cached = _STRUCTURE_PHASE3_TREEWIDTH_BATCH_SUPPORT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    q2_support = tuple(
        sorted(
            (int(i), int(j))
            for (i, j), coeff in q.q2.items()
            if coeff % q.mod_q2
        )
    )
    q3_support = tuple(
        sorted(
            (int(i), int(j), int(k))
            for (i, j, k), coeff in q.q3.items()
            if coeff % q.mod_q3
        )
    )
    cached = (q2_support, q3_support)
    _STRUCTURE_PHASE3_TREEWIDTH_BATCH_SUPPORT_CACHE[cache_key] = cached
    return cached

def _cached_phase3_support_plan(
    q,
    *,
    cache,
    order_builder,
    finalize_order: bool,
):
    cache_key = _q_cubic_treewidth_batch_support_key(q)
    cached = cache.get(cache_key)
    if cached is None:
        cover = tuple(_minimum_q3_vertex_cover_uncached(q))
        order, width = order_builder(q)
        core_vars, peel_order = _q3_hypergraph_2core(q)
        core_cover_size = _q3_core_cover_size(q, core_vars) if q.q3 else 0
        if peel_order:
            peel_set = set(peel_order)
            order = peel_order + [var for var in order if var not in peel_set]
            width = _treewidth_order_width(q, order)
        if finalize_order and q.q3:
            order, width = _finalize_phase3_treewidth_order(q, order)
        structural_obstruction = min(core_cover_size, width) if q.q3 else 0
        cached = (
            tuple(int(var) for var in cover),
            tuple(int(var) for var in order),
            int(width),
            int(structural_obstruction),
            bool(peel_order) and not core_vars,
        )
        cache[cache_key] = cached
    cover, order, width, structural_obstruction, fully_peeled = cached
    return list(cover), list(order), int(width), int(structural_obstruction), bool(fully_peeled)

def _phase3_support_plan(q):
    """Return a support-only Phase-3 plan reusable across coefficient changes."""
    return _cached_phase3_support_plan(
        q,
        cache=_STRUCTURE_PHASE3_SUPPORT_PLAN_CACHE,
        order_builder=_min_fill_cubic_order_uncached,
        finalize_order=True,
    )

def _phase3_batch_support_plan_fast(q):
    """Return a cheap support-only plan for repeated native batch evaluation."""
    return _cached_phase3_support_plan(
        q,
        cache=_STRUCTURE_PHASE3_BATCH_FAST_PLAN_CACHE,
        order_builder=_min_degree_cubic_order_uncached,
        finalize_order=False,
    )

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

def _phase_function_from_parts_mutable(n, *, level, q0, q1, q2, q3):
    phase = _phase_function_from_parts(
        n,
        level=level,
        q0=q0,
        q1=q1,
        q2=q2,
        q3=q3,
    )
    phase._schur_mutable = True
    return phase


_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
