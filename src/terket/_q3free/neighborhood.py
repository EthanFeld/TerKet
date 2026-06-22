"""Exact q3-free sums on bounded-neighborhood-diversity pair graphs."""

from __future__ import annotations

import cmath
from collections.abc import Sequence

from .._engine_runtime_core import (
    _Q3_FREE_NEIGHBORHOOD_MAX_CLASSES,
    _Q3_FREE_NEIGHBORHOOD_MAX_STATES,
    _Q3_FREE_NEIGHBORHOOD_MAX_WORK_PER_VAR,
    _Q3_FREE_NEIGHBORHOOD_TREEWIDTH_MAX_WIDTH,
    _Q3_FREE_NEIGHBORHOOD_TREEWIDTH_MAX_WORK,
    _Q3_FREE_NEIGHBORHOOD_TREEWIDTH_MAX_CLASSES,
)
from .._factor_tables import _combine_factor_scaled, _sum_factor_tables_scaled
from ..scaling import (
    ScaledComplex,
    _ONE_SCALED,
    _ZERO_SCALED,
    _add_scaled_complex,
    _make_scaled_complex,
    _mul_scaled_complex,
    _omega_scaled_table,
)
from .models import _Q3FreeNeighborhoodPlan, _Q3FreeNeighborhoodTreewidthPlan


def _q3_free_twin_classes(q):
    if q.q3 or not q.q2:
        return None

    half_q2 = q.mod_q2 // 2 if q.mod_q2 else 0
    adjacency = [0] * q.n
    for (left, right), coeff in q.q2.items():
        if int(coeff) % q.mod_q2 != half_q2:
            return None
        adjacency[left] |= 1 << right
        adjacency[right] |= 1 << left

    parent = list(range(q.n))

    def find(var: int) -> int:
        while parent[var] != var:
            parent[var] = parent[parent[var]]
            var = parent[var]
        return var

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for signatures in (
        adjacency,
        [neighbors | (1 << var) for var, neighbors in enumerate(adjacency)],
    ):
        representative_by_signature: dict[int, int] = {}
        for var, signature in enumerate(signatures):
            representative = representative_by_signature.setdefault(signature, var)
            union(representative, var)

    grouped: dict[int, list[int]] = {}
    for var in range(q.n):
        grouped.setdefault(find(var), []).append(var)
    classes = tuple(tuple(group) for group in sorted(grouped.values(), key=lambda group: group[0]))
    class_count = len(classes)
    class_by_var = [0] * q.n
    clique_classes = [False] * class_count
    cross_masks = [0] * class_count
    for class_idx, variables in enumerate(classes):
        for var in variables:
            class_by_var[var] = class_idx
        representative = variables[0]
        clique_classes[class_idx] = len(variables) > 1 and bool(
            adjacency[representative] & (1 << variables[1])
        )
    for left, right in q.q2:
        left_class = class_by_var[left]
        right_class = class_by_var[right]
        if left_class != right_class:
            cross_masks[left_class] |= 1 << right_class
            cross_masks[right_class] |= 1 << left_class
    return classes, tuple(clique_classes), tuple(cross_masks)


def _build_q3_free_neighborhood_plan(q) -> _Q3FreeNeighborhoodPlan | None:
    """Build a full twin-class parity-enumeration plan."""
    twin_data = _q3_free_twin_classes(q)
    if twin_data is None:
        return None
    classes, clique_classes, cross_masks = twin_data
    class_count = len(classes)
    if (
        class_count > _Q3_FREE_NEIGHBORHOOD_MAX_CLASSES
        or (1 << class_count) > _Q3_FREE_NEIGHBORHOOD_MAX_STATES
    ):
        return None

    estimated_work = max(1, q.n + class_count * (1 << class_count))
    if estimated_work > _Q3_FREE_NEIGHBORHOOD_MAX_WORK_PER_VAR * max(1, q.n):
        return None
    return _Q3FreeNeighborhoodPlan(
        classes=classes,
        clique_classes=clique_classes,
        cross_masks=cross_masks,
        estimated_work=estimated_work,
    )


def _min_fill_quotient_order(cross_masks: Sequence[int]) -> tuple[tuple[int, ...], int, int]:
    adjacency = [int(mask) for mask in cross_masks]
    remaining = (1 << len(adjacency)) - 1
    order: list[int] = []
    width = 0
    work = 0
    while remaining:
        best = None
        for var in range(len(adjacency)):
            if not (remaining & (1 << var)):
                continue
            neighbors = adjacency[var] & remaining
            neighbor_vars = [idx for idx in range(len(adjacency)) if neighbors & (1 << idx)]
            missing = 0
            for pos, left in enumerate(neighbor_vars):
                for right in neighbor_vars[pos + 1 :]:
                    missing += not bool(adjacency[left] & (1 << right))
            score = (missing, len(neighbor_vars), var)
            if best is None or score < best[0]:
                best = score, var, neighbor_vars
        assert best is not None
        _, var, neighbor_vars = best
        scope_size = len(neighbor_vars) + 1
        width = max(width, scope_size)
        work += 1 << scope_size
        for left in neighbor_vars:
            for right in neighbor_vars:
                if left != right:
                    adjacency[left] |= 1 << right
        remaining &= ~(1 << var)
        order.append(var)
    return tuple(order), width, work


def _build_q3_free_neighborhood_treewidth_plan(q) -> _Q3FreeNeighborhoodTreewidthPlan | None:
    """Compress nontrivial twin classes, then plan treewidth DP on quotient."""
    twin_data = _q3_free_twin_classes(q)
    if twin_data is None:
        return None
    classes, clique_classes, cross_masks = twin_data
    if (
        len(classes) >= q.n
        or len(classes) < 2
        or len(classes) > _Q3_FREE_NEIGHBORHOOD_TREEWIDTH_MAX_CLASSES
    ):
        return None
    order, width, work = _min_fill_quotient_order(cross_masks)
    estimated_work = q.n + work
    if (
        width > _Q3_FREE_NEIGHBORHOOD_TREEWIDTH_MAX_WIDTH
        or estimated_work > _Q3_FREE_NEIGHBORHOOD_TREEWIDTH_MAX_WORK
    ):
        return None
    return _Q3FreeNeighborhoodTreewidthPlan(
        classes=classes,
        clique_classes=clique_classes,
        cross_masks=cross_masks,
        order=order,
        width=width,
        estimated_work=estimated_work,
    )


def _class_even_odd_weights(
    classes: Sequence[Sequence[int]],
    clique_classes: Sequence[bool],
    q1: Sequence[int],
    *,
    level: int,
) -> list[tuple[ScaledComplex, ScaledComplex]]:
    omega = _omega_scaled_table(level)
    modulus = 1 << level
    class_even_odd: list[tuple[ScaledComplex, ScaledComplex]] = []

    for variables, clique in zip(classes, clique_classes):
        residues = [_ZERO_SCALED, _ZERO_SCALED, _ZERO_SCALED, _ZERO_SCALED]
        residues[0] = _ONE_SCALED
        for var in variables:
            phase = omega[int(q1[var]) % modulus]
            previous = residues
            residues = [
                _add_scaled_complex(previous[residue], _mul_scaled_complex(phase, previous[(residue - 1) & 3]))
                for residue in range(4)
            ]
        if clique:
            residues[2] = _mul_scaled_complex(_make_scaled_complex(-1.0), residues[2])
            residues[3] = _mul_scaled_complex(_make_scaled_complex(-1.0), residues[3])
        class_even_odd.append(
            (
                _add_scaled_complex(residues[0], residues[2]),
                _add_scaled_complex(residues[1], residues[3]),
            )
        )
    return class_even_odd


def _evaluate_q3_free_neighborhood_plan_scaled(
    plan: _Q3FreeNeighborhoodPlan,
    q1: Sequence[int],
    *,
    level: int,
) -> ScaledComplex:
    """Evaluate one twin-class plan under arbitrary root-of-unity unary phases."""
    class_even_odd = _class_even_odd_weights(
        plan.classes,
        plan.clique_classes,
        q1,
        level=level,
    )

    total = _ZERO_SCALED
    for mask in range(1 << len(plan.classes)):
        term = _ONE_SCALED
        negative = False
        for class_idx, even_odd in enumerate(class_even_odd):
            odd = bool(mask & (1 << class_idx))
            term = _mul_scaled_complex(term, even_odd[int(odd)])
            if odd and ((plan.cross_masks[class_idx] & mask & ((1 << class_idx) - 1)).bit_count() & 1):
                negative = not negative
        if negative:
            term = _mul_scaled_complex(_make_scaled_complex(-1.0), term)
        total = _add_scaled_complex(total, term)
    return total


def _evaluate_q3_free_neighborhood_treewidth_plan_scaled(
    plan: _Q3FreeNeighborhoodTreewidthPlan,
    q1: Sequence[int],
    *,
    level: int,
) -> ScaledComplex:
    """Evaluate compressed twin classes by exact quotient treewidth DP."""
    class_even_odd = _class_even_odd_weights(
        plan.classes,
        plan.clique_classes,
        q1,
        level=level,
    )
    factors = {}
    scalar = _ONE_SCALED
    negative = _make_scaled_complex(-1.0)
    for class_idx, table in enumerate(class_even_odd):
        scalar = _mul_scaled_complex(
            scalar,
            _combine_factor_scaled(factors, (class_idx,), list(table)),
        )
    for left, cross_mask in enumerate(plan.cross_masks):
        for right in range(left + 1, len(plan.classes)):
            if cross_mask & (1 << right):
                scalar = _mul_scaled_complex(
                    scalar,
                    _combine_factor_scaled(
                        factors,
                        (left, right),
                        [_ONE_SCALED, _ONE_SCALED, _ONE_SCALED, negative],
                    ),
                )
    total, _max_scope = _sum_factor_tables_scaled(
        len(plan.classes),
        factors,
        plan.order,
        scalar=scalar,
    )
    return total


def _evaluate_q3_free_neighborhood_plan_scaled_batch(
    plan: _Q3FreeNeighborhoodPlan,
    q1_batch,
    *,
    level: int,
) -> list[ScaledComplex]:
    return [
        _evaluate_q3_free_neighborhood_plan_scaled(plan, row, level=level)
        for row in q1_batch
    ]


def _evaluate_q3_free_neighborhood_treewidth_plan_scaled_batch(
    plan: _Q3FreeNeighborhoodTreewidthPlan,
    q1_batch,
    *,
    level: int,
) -> list[ScaledComplex]:
    return [
        _evaluate_q3_free_neighborhood_treewidth_plan_scaled(plan, row, level=level)
        for row in q1_batch
    ]


def _sum_q3_free_via_neighborhood_scaled(q) -> ScaledComplex | None:
    plan = _build_q3_free_neighborhood_plan(q)
    if plan is None:
        return None
    total = _evaluate_q3_free_neighborhood_plan_scaled(plan, q.q1, level=q.level)
    if q.q0:
        total = _mul_scaled_complex(
            _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q.q0))),
            total,
        )
    return total


def _sum_q3_free_via_neighborhood_treewidth_scaled(q) -> ScaledComplex | None:
    plan = _build_q3_free_neighborhood_treewidth_plan(q)
    if plan is None:
        return None
    total = _evaluate_q3_free_neighborhood_treewidth_plan_scaled(plan, q.q1, level=q.level)
    if q.q0:
        total = _mul_scaled_complex(
            _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q.q0))),
            total,
        )
    return total


def _sum_q3_free_via_neighborhood_composed_scaled(q) -> ScaledComplex | None:
    """Choose full twin parity enumeration or treewidth DP on its quotient."""
    neighborhood_plan = _build_q3_free_neighborhood_plan(q)
    treewidth_plan = _build_q3_free_neighborhood_treewidth_plan(q)
    if neighborhood_plan is None and treewidth_plan is None:
        return None
    if (
        treewidth_plan is not None
        and (
            neighborhood_plan is None
            or treewidth_plan.estimated_work < neighborhood_plan.estimated_work
        )
    ):
        total = _evaluate_q3_free_neighborhood_treewidth_plan_scaled(
            treewidth_plan,
            q.q1,
            level=q.level,
        )
    else:
        assert neighborhood_plan is not None
        total = _evaluate_q3_free_neighborhood_plan_scaled(
            neighborhood_plan,
            q.q1,
            level=q.level,
        )
    if q.q0:
        total = _mul_scaled_complex(
            _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q.q0))),
            total,
        )
    return total
