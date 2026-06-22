
"""Exact reduction classification helpers."""

from __future__ import annotations

import importlib

from ._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals

_LOCAL_NAMES = {
    '_build_classification_data',
    '_classification_lookup',
    '_classification_entry',
    '_has_odd_bilinear_coupling',
    '_classify',
    '_incident_quadratic_couplings',
    '_elim_sparse_dead_quadratics_batch',

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


def _classification_data_python(q):
    mod_q2 = max(1, 1 << (q.level - 1))
    mod_q3 = 1 << max(0, q.level - 2)
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

    return cubic_incidence, odd_bilinear, parity_partners


def _classification_lookup_python(
    q,
    cubic_incidence,
    odd_bilinear,
    parity_partners,
):
    mod_q1 = 1 << q.level
    threshold = max(1, mod_q1 // 4)
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
    return tuple(lookup)

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

        return _classification_data_python(q)

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
        cubic_incidence, odd_bilinear, parity_partners = _classification_data_python(q)

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

    if getattr(q, "_schur_mutable", True):
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
                return tuple(native_result)

        return _classification_lookup_python(
            q,
            cubic_incidence,
            odd_bilinear,
            parity_partners,
        )

    cache_key = _q_classification_structure_key(q)
    cached = _STRUCTURE_CLASSIFICATION_LOOKUP_CACHE.get(cache_key)
    if cached is not None:
        if not getattr(q, "_schur_mutable", True):
            q._schur_q_classification_lookup = cached
        return cached

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

    result = _classification_lookup_python(
        q,
        cubic_incidence,
        odd_bilinear,
        parity_partners,
    )
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

def _elim_sparse_quadratics_batch_native(q, candidates):
    native_batch = _native_symbol("elim_sparse_quadratics_batch_terms")
    if native_batch is None or not _native_level3_enabled(q):
        return None
    q0_residue = (q.q0.numerator * (q.mod_q1 // q.q0.denominator)) % q.mod_q1
    new_q0_residue, new_q1, new_q2, removed = native_batch(
        q0_residue,
        q.q1,
        q.q2,
        tuple(candidates),
    )
    return (
        _phase_function_from_parts_mutable(
            len(new_q1),
            level=q.level,
            q0=_fraction_from_residue(q.level, new_q0_residue),
            q1=new_q1,
            q2=new_q2,
            q3={},
        ),
        len(removed),
        tuple(int(var) for var in removed),
    )

def _elim_sparse_dead_quadratics_batch(q, candidates, *, classification_data=None):
    """Batch-eliminate sparse dead quadratic pivots with one q2 compaction.

    This targets build-time PauliExpBox states where thousands of pending dead
    variables are degree-2 quadratic pivots. The old path rebuilt and remapped
    q2 once per pivot; this mutates an adjacency map in old coordinates and
    compacts only once.
    """
    if int(q.level) != 3 or q.q3:
        return q, 0, ()
    native_result = _elim_sparse_quadratics_batch_native(q, candidates)
    if native_result is not None:
        return native_result
    if classification_data is None:
        classification_data = _build_classification_data(q)
    _cubic_incidence, odd_bilinear, _parity_partners = classification_data

    candidate_set = {int(var) for var in candidates if 0 <= int(var) < q.n}
    adjacency: dict[int, dict[int, int]] = {var: {} for var in range(q.n)}
    for (left, right), coeff in q.q2.items():
        residue = int(coeff) % q.mod_q2
        if not residue:
            continue
        adjacency[left][right] = residue
        adjacency[right][left] = residue

    q1 = list(q.q1)
    q0 = q.q0
    removed: list[int] = []
    removed_set: set[int] = set()
    threshold = _quadratic_residue_threshold(q)

    def remove_edge(left: int, right: int) -> None:
        adjacency[left].pop(right, None)
        adjacency[right].pop(left, None)

    def add_edge(left: int, right: int, value: int) -> None:
        if left == right:
            return
        if left > right:
            left, right = right, left
        value %= q.mod_q2
        if value:
            adjacency[left][right] = value
            adjacency[right][left] = value
        else:
            remove_edge(left, right)

    for var in sorted(candidate_set):
        if var in removed_set or odd_bilinear[var]:
            continue
        coupled = [(neighbor, coeff) for neighbor, coeff in adjacency[var].items() if neighbor not in removed_set]
        if len(coupled) > 2:
            continue
        c = (q1[var] // threshold) % 4
        # An earlier pivot in this batch may change an adjacent candidate's
        # linear residue. Only apply the Gauss rule if it is still quadratic
        # in the incrementally updated kernel.
        if c not in (1, 3):
            continue
        const_phase = Fraction(1, 8) if c % 4 == 1 else Fraction(7, 8)
        sign = -1 if c % 4 == 1 else +1
        q0 = (q0 + const_phase) % 1
        for neighbor, coupling in coupled:
            q1[neighbor] = (q1[neighbor] + sign * coupling) % q.mod_q1
        for left_pos in range(len(coupled)):
            left_var, left_coeff = coupled[left_pos]
            for right_pos in range(left_pos + 1, len(coupled)):
                right_var, right_coeff = coupled[right_pos]
                correction = _quadratic_pair_correction(q, left_coeff, right_coeff)
                if correction:
                    left = min(left_var, right_var)
                    right = max(left_var, right_var)
                    add_edge(left, right, adjacency[left].get(right, 0) + correction)
        for neighbor, _coupling in coupled:
            remove_edge(var, neighbor)
        removed.append(var)
        removed_set.add(var)

    if not removed:
        return q, 0, ()

    remap: dict[int, int] = {}
    new_q1: list[int] = []
    for idx, coeff in enumerate(q1):
        if idx in removed_set:
            continue
        remap[idx] = len(new_q1)
        new_q1.append(coeff)

    new_q2: dict[tuple[int, int], int] = {}
    for left, neighbors in adjacency.items():
        if left in removed_set:
            continue
        new_left = remap[left]
        for right, coeff in neighbors.items():
            if right in removed_set or left >= right:
                continue
            new_right = remap[right]
            edge = (new_left, new_right) if new_left < new_right else (new_right, new_left)
            residue = int(coeff) % q.mod_q2
            if residue:
                new_q2[edge] = residue

    return (
        _phase_function_from_parts_mutable(
            len(new_q1),
            level=q.level,
            q0=q0,
            q1=new_q1,
            q2=new_q2,
            q3={},
        ),
        len(removed),
        tuple(removed),
    )


_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
