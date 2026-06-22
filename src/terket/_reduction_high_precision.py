"""Representation-safe exact reductions above Clifford+T precision."""

from __future__ import annotations

from fractions import Fraction

from ._reduction_support import _phase_function_from_parts_mutable


def _elim_decoupled_constraints_batch(q, variables):
    if not variables:
        return q, 0
    removed = set(variables)
    remap = {}
    for var in range(q.n):
        if var not in removed:
            remap[var] = len(remap)
    reduced = _phase_function_from_parts_mutable(
        q.n - len(removed),
        level=q.level,
        q0=q.q0,
        q1=[q.q1[var] for var in range(q.n) if var not in removed],
        q2={
            (remap[left], remap[right]): value
            for (left, right), value in q.q2.items()
            if left not in removed and right not in removed
        },
        q3={
            (remap[a], remap[b], remap[c]): value
            for (a, b, c), value in q.q3.items()
            if a not in removed and b not in removed and c not in removed
        },
    )
    return reduced, 2 * len(removed)


def _high_precision_constraint_actions(q):
    """Find representation-safe exact constraints above level 3."""
    parity_residue = q.mod_q2 // 2
    partners = [[] for _ in range(q.n)]
    blocked = [False] * q.n
    for (left, right), coeff in q.q2.items():
        residue = int(coeff) % q.mod_q2
        if residue == parity_residue:
            partners[left].append(right)
            partners[right].append(left)
        elif residue:
            blocked[left] = blocked[right] = True
    for scope, coeff in q.q3.items():
        if int(coeff) % q.mod_q3:
            for var in scope:
                blocked[var] = True

    decoupled = []
    candidates = []
    half_turn = q.mod_q1 // 2
    for var, coeff in enumerate(q.q1):
        residue = int(coeff) % q.mod_q1
        if blocked[var]:
            continue
        if not partners[var]:
            if residue == half_turn:
                return True, (), ()
            if residue == 0:
                decoupled.append(var)
            continue
        if residue in (0, half_turn) and len(partners[var]) <= 2:
            candidates.append((var, tuple(sorted(partners[var])), int(residue == half_turn)))

    selected = []
    occupied = set()
    for action in candidates:
        scope = {action[0], *action[1]}
        if scope.isdisjoint(occupied):
            selected.append(action)
            occupied.update(scope)
    return False, tuple(decoupled), tuple(selected)


def _elim_high_precision_constraints_batch(q, actions):
    removed = {var for constraint, partners, _target in actions for var in (constraint, partners[0])}
    remap = [-1] * q.n
    out_idx = 0
    for var in range(q.n):
        if var not in removed:
            remap[var] = out_idx
            out_idx += 1
    offsets = [0] * q.n
    for constraint, partners, target in actions:
        remove = partners[0]
        remap[constraint] = -1
        if len(partners) == 2:
            remap[remove] = remap[partners[1]]
        offsets[remove] = target
    return _compose_unit_affine_substitutions(q, remap, offsets, out_idx), 2 * len(actions)


def _compose_unit_affine_substitutions(q, remap, offsets, out_n):
    """Compose substitutions where each old bit is constant or one literal."""
    new_q0 = q.q0
    new_q1 = [0] * out_n
    new_q2 = {}
    new_q3 = {}

    def add_monomial(alpha, old_vars):
        nonlocal new_q0
        expanded = {(): int(alpha)}
        for old_var in old_vars:
            mapped = remap[old_var]
            offset = offsets[old_var]
            if mapped < 0:
                if not offset:
                    return
                continue
            updated = {}
            for scope, coeff in expanded.items():
                mapped_scope = tuple(sorted((*scope, mapped))) if mapped not in scope else scope
                if offset:
                    updated[scope] = updated.get(scope, 0) + coeff
                    updated[mapped_scope] = updated.get(mapped_scope, 0) - coeff
                else:
                    updated[mapped_scope] = updated.get(mapped_scope, 0) + coeff
            expanded = {scope: coeff for scope, coeff in updated.items() if coeff}
        for scope, coeff in expanded.items():
            degree = len(scope)
            if degree == 0:
                new_q0 = (new_q0 + Fraction(coeff, q.mod_q1)) % 1
            elif degree == 1:
                var = scope[0]
                new_q1[var] = (new_q1[var] + coeff) % q.mod_q1
            elif degree == 2:
                _update_term(new_q2, scope, coeff // 2, q.mod_q2)
            else:
                _update_term(new_q3, scope, coeff // 4, q.mod_q3)

    for var, coeff in enumerate(q.q1):
        if coeff:
            add_monomial(coeff, (var,))
    for scope, coeff in q.q2.items():
        if coeff:
            add_monomial(2 * coeff, scope)
    for scope, coeff in q.q3.items():
        if coeff:
            add_monomial(4 * coeff, scope)
    return _phase_function_from_parts_mutable(
        out_n,
        level=q.level,
        q0=new_q0,
        q1=new_q1,
        q2=new_q2,
        q3=new_q3,
    )


def _update_term(terms, scope, delta, modulus):
    value = (terms.get(scope, 0) + delta) % modulus
    if value:
        terms[scope] = value
    else:
        terms.pop(scope, None)


def _apply_safe_high_precision_eliminations(q):
    """Apply only substitutions that preserve cubic form at higher precision."""
    scale_half_pow2 = 0
    constraints = 0
    while q.n:
        zero, decoupled, actions = _high_precision_constraint_actions(q)
        if zero:
            return None, 0, {'quad': 0, 'constraint': constraints}
        if decoupled:
            q, half_pow2 = _elim_decoupled_constraints_batch(q, decoupled)
            scale_half_pow2 += half_pow2
            constraints += len(decoupled)
            continue
        if actions:
            q, half_pow2 = _elim_high_precision_constraints_batch(q, actions)
            scale_half_pow2 += half_pow2
            constraints += len(actions)
            continue
        break
    if q is not None:
        q._schur_mutable = False
    return q, scale_half_pow2, {'quad': 0, 'constraint': constraints}


__all__ = [
    "_apply_safe_high_precision_eliminations",
    "_elim_decoupled_constraints_batch",
]
