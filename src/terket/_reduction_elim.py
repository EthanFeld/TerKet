
"""Exact elimination and affine-compose helpers."""

from __future__ import annotations

import importlib

from ._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals

_LOCAL_NAMES = {
    '_elim_quadratic',
    '_elim_quadratic_via_split',
    '_elim_constraint',
    '_elim_single_partner_constraint_python',
    '_elim_single_partner_constraint',
    '_elim_two_partner_constraint_python',
    '_elim_two_partner_constraint',
    '_elim_two_partner_constraint_q3_free',
    '_aff_compose_python',
    '_aff_compose',
    '_info',

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

    def remap_var(var: int) -> int:
        return int(var) - (1 if int(var) > k else 0)

    nn = nf - 1
    new_q1 = list(q.q1[:k])
    new_q1.extend(q.q1[k + 1 :])
    new_q2 = {
        (remap_var(i), remap_var(j)): value
        for (i, j), value in q.q2.items()
        if i != k and j != k
    }
    if q.q3:
        new_q3 = {
            (remap_var(i), remap_var(j), remap_var(l)): value
            for (i, j, l), value in q.q3.items()
            if i != k and j != k and l != k
        }
    else:
        new_q3 = {}
    for var, coupling in coupled:                              # linear corr
        mapped_var = remap_var(var)
        new_q1[mapped_var] = (new_q1[mapped_var] + sign * coupling) % q.mod_q1
    for left_pos in range(len(coupled)):                       # [BL26 Eq.194]
        left_var, left_coeff = coupled[left_pos]
        for right_pos in range(left_pos + 1, len(coupled)):
            right_var, right_coeff = coupled[right_pos]
            new_left = remap_var(left_var)
            new_right = remap_var(right_var)
            if new_left > new_right:
                new_left, new_right = new_right, new_left
            corr = _quadratic_pair_correction(q, left_coeff, right_coeff)
            if corr:
                edge = (new_left, new_right)
                new_q2[edge] = (new_q2.get(edge, 0) + corr) % q.mod_q2
                if new_q2[edge] == 0:
                    del new_q2[edge]
    return _phase_function_from_parts_mutable(
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
            _phase_function_from_parts_mutable(
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
        if len(partners) == 2:
            return _elim_two_partner_constraint(q, k, partners[1], p, target)
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
        composed = _aff_compose_cached(q, shift_mask, gamma, nn, context=context)
        composed._schur_mutable = True
        return composed, 2
    return (
        _phase_function_from_parts_mutable(
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

    return _phase_function_from_parts_mutable(
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
        return _phase_function_from_parts_mutable(
            q.n - 2,
            level=q.level,
            q0=_fraction_from_residue(q.level, new_q0_residue),
            q1=new_q1,
            q2=new_q2,
            q3=new_q3,
        ), 2
    return _elim_single_partner_constraint_python(q, k, p, target)

def _elim_two_partner_constraint_python(q, k: int, keep: int, remove: int, target: int):
    """
    Fast path for parity constraints with two residue-2 partners.

    The character sum over ``k`` enforces ``remove = keep xor target`` and
    contributes a factor of ``2``. Applying that substitution directly is much
    cheaper than routing through generic affine composition, especially on large
    cubic kernels that repeatedly hit the same low-arity parity pattern.
    """
    nf = q.n
    removed = {int(k), int(remove)}
    keep = int(keep)
    target = int(target) & 1
    if keep in removed:
        return None

    remap = {}
    idx = 0
    for var in range(nf):
        if var in removed:
            continue
        remap[var] = idx
        idx += 1
    if keep not in remap:
        return None

    keep_new = remap[keep]
    new_q0 = q.q0
    new_q1 = [q.q1[var] for var in range(nf) if var not in removed]
    new_q2: dict[tuple[int, int], int] = {}
    new_q3: dict[tuple[int, int, int], int] = {}
    lift_q2 = q.mod_q1 // q.mod_q2 if q.mod_q2 else 0
    lift_q3 = q.mod_q2 // q.mod_q3 if q.mod_q3 else 0

    def add_q2(left: int, right: int, coeff: int) -> None:
        coeff %= q.mod_q2
        if not coeff or left == right:
            if left == right and coeff:
                new_q1[left] = (new_q1[left] + lift_q2 * coeff) % q.mod_q1
            return
        if left > right:
            left, right = right, left
        updated = (new_q2.get((left, right), 0) + coeff) % q.mod_q2
        if updated:
            new_q2[(left, right)] = updated
        elif (left, right) in new_q2:
            del new_q2[(left, right)]

    def add_q3(a: int, b: int, c: int, coeff: int) -> None:
        coeff %= q.mod_q3
        if not coeff:
            return
        if a == b or a == c or b == c:
            vars_sorted = sorted((a, b, c))
            if vars_sorted[0] == vars_sorted[1] == vars_sorted[2]:
                new_q1[vars_sorted[0]] = (new_q1[vars_sorted[0]] + (lift_q2 * lift_q3) * coeff) % q.mod_q1
                return
            if vars_sorted[0] == vars_sorted[1]:
                add_q2(vars_sorted[0], vars_sorted[2], lift_q3 * coeff)
                return
            add_q2(vars_sorted[0], vars_sorted[1], lift_q3 * coeff)
            return
        key = tuple(sorted((a, b, c)))
        updated = (new_q3.get(key, 0) + coeff) % q.mod_q3
        if updated:
            new_q3[key] = updated
        elif key in new_q3:
            del new_q3[key]

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
            add_q2(remap[left], remap[right], coeff)
            continue

        other = right if left == remove else left
        if other == keep:
            if not target:
                new_q1[keep_new] = (new_q1[keep_new] + lift_q2 * coeff) % q.mod_q1
            continue
        if other == k or other not in remap:
            continue

        other_new = remap[other]
        if target:
            new_q1[other_new] = (new_q1[other_new] + lift_q2 * coeff) % q.mod_q1
            pair_coeff = (-coeff) % q.mod_q2
        else:
            pair_coeff = coeff
        add_q2(keep_new, other_new, pair_coeff)

    for (a, b, c), coeff in q.q3.items():
        coeff %= q.mod_q3
        if not coeff or k in (a, b, c):
            continue
        if remove not in (a, b, c):
            if a not in remap or b not in remap or c not in remap:
                continue
            add_q3(remap[a], remap[b], remap[c], coeff)
            continue

        others = [var for var in (a, b, c) if var != remove]
        if keep in others:
            other = others[0] if others[1] == keep else others[1]
            if target or other not in remap:
                continue
            add_q2(keep_new, remap[other], lift_q3 * coeff)
            continue

        left, right = others
        if left not in remap or right not in remap:
            continue
        left_new = remap[left]
        right_new = remap[right]
        if target:
            add_q2(left_new, right_new, lift_q3 * coeff)
            coeff = (-coeff) % q.mod_q3
        add_q3(keep_new, left_new, right_new, coeff)

    return _phase_function_from_parts_mutable(
        q.n - 2,
        level=q.level,
        q0=new_q0,
        q1=new_q1,
        q2=new_q2,
        q3=new_q3,
    ), 2

def _elim_two_partner_constraint(q, k: int, keep: int, remove: int, target: int):
    if not q.q3:
        return _elim_two_partner_constraint_q3_free(q, k, keep, remove, target)
    native_two_partner = _native_symbol("elim_two_partner_constraint_terms")
    if native_two_partner is not None and _native_level3_enabled(q):
        q0_residue = (q.q0.numerator * (q.mod_q1 // q.q0.denominator)) % q.mod_q1
        new_q0_residue, new_q1, new_q2, new_q3 = native_two_partner(
            q0_residue,
            q.q1,
            q.q2,
            q.q3,
            k,
            keep,
            remove,
            int(target),
        )
        return _phase_function_from_parts_mutable(
            q.n - 2,
            level=q.level,
            q0=_fraction_from_residue(q.level, new_q0_residue),
            q1=new_q1,
            q2=new_q2,
            q3=new_q3,
        ), 2
    if _native_aff_compose_enabled():
        nn = q.n - 2
        gamma = [0] * q.n
        idx = 0
        for var in range(q.n):
            if var in (k, remove):
                continue
            gamma[var] = 1 << idx
            idx += 1
        gamma[remove] = gamma[keep]
        composed = _aff_compose(q, (1 << remove) if target else 0, gamma, nn)
        composed._schur_mutable = True
        return composed, 2
    if q.n <= _DIRECT_TWO_PARTNER_CONSTRAINT_MAX_VARS:
        return _elim_two_partner_constraint_python(q, k, keep, remove, target)
    nn = q.n - 2
    gamma = [0] * q.n
    idx = 0
    for var in range(q.n):
        if var in (k, remove):
            continue
        gamma[var] = 1 << idx
        idx += 1
    gamma[remove] = gamma[keep]
    composed = _aff_compose(q, (1 << remove) if target else 0, gamma, nn)
    composed._schur_mutable = True
    return composed, 2

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


_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
