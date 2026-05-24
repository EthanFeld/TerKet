"""Extracted arbitrary-angle factor helpers."""

from __future__ import annotations

import cmath
from fractions import Fraction
import math
from typing import Sequence

import numpy as np

from ._engine_runtime_core import _configure_extracted_module
from .cubic_arithmetic import PhaseFunction
from .scaling import ScaledComplex

_LOCAL_NAMES = {
    '_coalesce_arbitrary_phase_terms',
    '_arbitrary_phase_terms_are_unary',
    '_build_unary_arbitrary_factor_tables',
    '_restrict_unary_arbitrary_factor_tables',
    '_sum_q3_free_with_unary_factor_tables_for_order_scaled',
    '_evaluate_q3_free_remaining_with_unary_factor_tables_scaled',
    '_evaluate_q3_free_cutset_conditioning_plan_with_unary_factor_tables_scaled',
    '_sum_q3_free_with_unary_factor_tables_scaled',
    '_sum_q3_free_with_unary_arbitrary_phases_scaled',
    '_arbitrary_phase_factor_table',
    '_add_arbitrary_phase_factors_scaled',
    '_restrict_scaled_factor_table',
    '_sum_factor_tables_with_cutset_scaled'
}
_LOCAL_IMPLS = {}
_configure_extracted_module(globals(), local_names=_LOCAL_NAMES, local_impls=_LOCAL_IMPLS)


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


def _arbitrary_phase_terms_are_unary(
    terms: Sequence[_ArbitraryPhaseTerm],
) -> bool:
    """Return whether every deferred arbitrary phase depends on one variable."""
    for term in terms:
        row_mask = int(term.row_mask)
        if row_mask == 0 or (row_mask & (row_mask - 1)) != 0:
            return False
    return True


def _build_unary_arbitrary_factor_tables(
    n_vars: int,
    terms: Sequence[_ArbitraryPhaseTerm],
) -> tuple[tuple[ScaledComplex, ScaledComplex], ...]:
    """Return per-variable [x=0, x=1] factors for unary arbitrary phases."""
    tables = [[_ONE_SCALED, _ONE_SCALED] for _ in range(int(n_vars))]

    for term in terms:
        row_mask = int(term.row_mask)
        if row_mask == 0 or (row_mask & (row_mask - 1)) != 0:
            raise ValueError("Expected only unary arbitrary-phase terms.")
        var = row_mask.bit_length() - 1
        phase = _make_scaled_complex(cmath.exp(1j * float(term.angle)))
        table = [phase, _ONE_SCALED] if (int(term.offset) & 1) else [_ONE_SCALED, phase]
        tables[var][0] = _mul_scaled_complex(tables[var][0], table[0])
        tables[var][1] = _mul_scaled_complex(tables[var][1], table[1])

    return tuple((table[0], table[1]) for table in tables)


def _restrict_unary_arbitrary_factor_tables(
    unary_tables: Sequence[tuple[ScaledComplex, ScaledComplex]],
    variables: Sequence[int],
) -> tuple[tuple[ScaledComplex, ScaledComplex], ...]:
    return tuple(unary_tables[int(var)] for var in variables)


def _sum_q3_free_with_unary_factor_tables_for_order_scaled(
    q: PhaseFunction,
    unary_tables: Sequence[tuple[ScaledComplex, ScaledComplex]],
    order: Sequence[int],
) -> ScaledComplex:
    """Sum a q3-free kernel by exact factor elimination along ``order``."""
    scalar, factors = _build_cubic_factors_scaled(q)
    for var, table in enumerate(unary_tables):
        scalar = _mul_scaled_complex(
            scalar,
            _combine_factor_scaled(factors, (var,), [table[0], table[1]]),
        )
    total, _max_scope = _sum_factor_tables_scaled(
        q.n,
        factors,
        order,
        scalar=scalar,
    )
    return total


def _evaluate_q3_free_remaining_with_unary_factor_tables_scaled(
    *,
    level: int,
    q1: Sequence[int],
    unary_tables: Sequence[tuple[ScaledComplex, ScaledComplex]],
    backend: str,
    q2: dict[tuple[int, int], int],
    order: Sequence[int],
    isolated_vars: Sequence[int] = (),
    components: Sequence["_Q3FreeConstraintComponentPlan"] = (),
    native_treewidth_plan: object | None = None,
) -> ScaledComplex:
    """Evaluate one cutset-conditioned remainder with unary arbitrary factors."""
    n_vars = len(q1)
    if n_vars == 0:
        return _ONE_SCALED

    if backend == "product":
        total = _ONE_SCALED
        omega_scaled = _omega_scaled_table(level)
        modulus = 1 << int(level)
        for var in range(n_vars):
            unary_zero, unary_one = unary_tables[var]
            total = _mul_scaled_complex(
                total,
                _add_scaled_complex(
                    unary_zero,
                    _mul_scaled_complex(omega_scaled[int(q1[var]) % modulus], unary_one),
                ),
            )
        return total

    if backend == "treewidth":
        q_local = _phase_function_from_parts(
            n_vars,
            level=level,
            q0=Fraction(0),
            q1=list(q1),
            q2=q2,
            q3={},
        )
        if native_treewidth_plan is not None and _schur_native is not None:
            scalar, factors = _build_cubic_factors_scaled(q_local)
            for var, table in enumerate(unary_tables):
                scalar = _mul_scaled_complex(
                    scalar,
                    _combine_factor_scaled(factors, (var,), [table[0], table[1]]),
                )
            try:
                core_total, _max_scope = _schur_native.sum_scaled_factor_treewidth_preplanned(
                    native_treewidth_plan,
                    scalar,
                    dict(factors),
                )
                return tuple(core_total)
            except Exception:
                pass
        return _sum_q3_free_with_unary_factor_tables_for_order_scaled(q_local, unary_tables, order)

    total = _ONE_SCALED
    if len(isolated_vars):
        iso_q1 = [q1[int(var)] for var in isolated_vars]
        iso_tables = _restrict_unary_arbitrary_factor_tables(unary_tables, isolated_vars)
        total = _mul_scaled_complex(
            total,
            _evaluate_q3_free_remaining_with_unary_factor_tables_scaled(
                level=level,
                q1=iso_q1,
                unary_tables=iso_tables,
                backend="product",
                q2={},
                order=(),
            ),
        )

    for component_plan in components:
        q1_local = [q1[int(var)] for var in component_plan.variables]
        unary_local = _restrict_unary_arbitrary_factor_tables(unary_tables, component_plan.variables)
        component_q = _phase_function_from_parts(
            len(component_plan.variables),
            level=level,
            q0=Fraction(0),
            q1=q1_local,
            q2=component_plan.q2,
            q3={},
        )
        total = _mul_scaled_complex(
            total,
            _sum_q3_free_with_unary_factor_tables_scaled(component_q, unary_local),
        )

    return total


def _evaluate_q3_free_cutset_conditioning_plan_with_unary_factor_tables_scaled(
    q: PhaseFunction,
    plan: _Q3FreeCutsetConditioningPlan,
    unary_tables: Sequence[tuple[ScaledComplex, ScaledComplex]],
) -> ScaledComplex:
    """Evaluate a q3-free cutset plan with extra unary arbitrary factors."""
    plan = _attach_q3_free_cutset_runtime_cache(plan)
    branch_bits = np.asarray(plan.branch_bits, dtype=np.int64)
    branch_pair_residue = np.asarray(plan.branch_pair_residue, dtype=np.int64)
    branch_remaining_shift = np.asarray(plan.branch_remaining_shift, dtype=np.int64)
    cutset_size = len(plan.cutset_vars)
    branch_count = 1 << cutset_size
    omega_scaled = _omega_scaled_table(q.level)
    mod_q1 = 1 << int(q.level)

    q0_eff = np.full(branch_count, _phase_fraction_to_residue(q.q0, q.mod_q1), dtype=np.int64)
    if len(plan.cutset_vars):
        cutset_q1 = np.asarray([q.q1[var] % q.mod_q1 for var in plan.cutset_vars], dtype=np.int64)
        q0_eff = (q0_eff + branch_bits @ cutset_q1) % q.mod_q1
    if branch_pair_residue.size:
        q0_eff = (q0_eff + branch_pair_residue) % q.mod_q1

    remaining_unary = _restrict_unary_arbitrary_factor_tables(unary_tables, plan.remaining_vars)
    base_remaining_q1 = np.asarray([q.q1[var] % q.mod_q1 for var in plan.remaining_vars], dtype=np.int64)
    total = _ZERO_SCALED

    for branch_idx in range(branch_count):
        branch_total = omega_scaled[int(q0_eff[branch_idx]) % mod_q1]
        for local_idx, var in enumerate(plan.cutset_vars):
            branch_total = _mul_scaled_complex(
                branch_total,
                unary_tables[int(var)][int(branch_bits[branch_idx, local_idx])],
            )

        if len(plan.remaining_vars):
            if branch_remaining_shift.size:
                remaining_q1 = (
                    base_remaining_q1 + branch_remaining_shift[branch_idx]
                ) % mod_q1
            else:
                remaining_q1 = base_remaining_q1
            branch_total = _mul_scaled_complex(
                branch_total,
                _evaluate_q3_free_remaining_with_unary_factor_tables_scaled(
                    level=q.level,
                    q1=remaining_q1.tolist(),
                    unary_tables=remaining_unary,
                    backend=plan.remaining_backend,
                    q2=plan.remaining_q2,
                    order=plan.remaining_order,
                    isolated_vars=plan.remaining_isolated_vars,
                    components=plan.remaining_components,
                    native_treewidth_plan=plan.native_treewidth_plan,
                ),
            )

        total = _add_scaled_complex(total, branch_total)

    return total


def _sum_q3_free_with_unary_factor_tables_scaled(
    q: PhaseFunction,
    unary_tables: Sequence[tuple[ScaledComplex, ScaledComplex]],
) -> ScaledComplex:
    """Exactly sum a q3-free kernel with additional unary factors."""
    assert not q.q3, "Unary arbitrary-phase summation requires a q3-free kernel."
    if q.n == 0:
        return _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q.q0)))

    if not q.q2:
        return _mul_scaled_complex(
            _make_scaled_complex(cmath.exp(2j * cmath.pi * float(q.q0))),
            _evaluate_q3_free_remaining_with_unary_factor_tables_scaled(
                level=q.level,
                q1=q.q1,
                unary_tables=unary_tables,
                backend="product",
                q2={},
                order=(),
            ),
        )

    adjacency, edges = _q3_free_graph(q)
    max_degree = max((len(neighbors) for neighbors in adjacency), default=0)
    depth, chords = _q3_free_spanning_data(adjacency, edges)
    feedback_vars = _select_feedback_vertices(q.n, chords, depth)
    order = _q3_free_treewidth_order(
        q,
        len(feedback_vars),
        max_degree=max_degree,
    )
    if order is not None:
        return _sum_q3_free_with_unary_factor_tables_for_order_scaled(q, unary_tables, order)

    plan = None
    try:
        plan = _q3_free_one_shot_cutset_conditioning_plan(q)
    except MemoryError:
        plan = None
    if plan is not None:
        return _evaluate_q3_free_cutset_conditioning_plan_with_unary_factor_tables_scaled(
            q,
            plan,
            unary_tables,
        )

    fallback_order, _width = _factor_scope_order(q.n, tuple(_build_cubic_factors_scaled(q)[1]))
    return _sum_q3_free_with_unary_factor_tables_for_order_scaled(q, unary_tables, fallback_order)


def _sum_q3_free_with_unary_arbitrary_phases_scaled(
    q: PhaseFunction,
    terms: Sequence[_ArbitraryPhaseTerm],
) -> ScaledComplex:
    """Exactly sum a q3-free kernel with additional unary arbitrary-angle factors."""
    unary_tables = _build_unary_arbitrary_factor_tables(q.n, terms)
    return _sum_q3_free_with_unary_factor_tables_scaled(q, unary_tables)


def _arbitrary_phase_factor_table(term: _ArbitraryPhaseTerm) -> tuple[tuple[int, ...], list[ScaledComplex], ScaledComplex]:
    """Return a dense factor for one arbitrary phase of an affine parity."""
    row_mask = int(term.row_mask)
    phase = _make_scaled_complex(cmath.exp(1j * float(term.angle)))
    if row_mask == 0:
        return (), [], phase if (int(term.offset) & 1) else _ONE_SCALED

    scope = _support_from_mask(row_mask)
    if len(scope) > _MAX_ARBITRARY_PHASE_FACTOR_SCOPE:
        raise RuntimeError(
            f"Cannot compute amplitude directly: arbitrary-angle factor has scope {len(scope)}, "
            f"above limit {_MAX_ARBITRARY_PHASE_FACTOR_SCOPE}."
        )

    table = [_ONE_SCALED] * (1 << len(scope))
    offset = int(term.offset) & 1
    for assignment in range(len(table)):
        if (assignment.bit_count() & 1) ^ offset:
            table[assignment] = phase
    return scope, table, _ONE_SCALED


def _add_arbitrary_phase_factors_scaled(
    factors: dict[tuple[int, ...], Sequence[ScaledComplex]],
    terms: Sequence[_ArbitraryPhaseTerm],
) -> ScaledComplex:
    """Attach arbitrary affine phase factors without branching on their rank."""
    scalar = _ONE_SCALED
    for term in terms:
        scope, table, term_scalar = _arbitrary_phase_factor_table(term)
        scalar = _mul_scaled_complex(scalar, term_scalar)
        if scope:
            scalar = _mul_scaled_complex(
                scalar,
                _combine_factor_scaled(factors, scope, table),
            )
    return scalar


def _restrict_scaled_factor_table(
    scope: tuple[int, ...],
    table: Sequence[ScaledComplex],
    fixed: dict[int, int],
    remap: dict[int, int],
) -> tuple[tuple[int, ...], list[ScaledComplex], ScaledComplex]:
    fixed_positions = [
        (idx, int(fixed[var]) & 1)
        for idx, var in enumerate(scope)
        if var in fixed
    ]
    if not fixed_positions:
        return tuple(remap[var] for var in scope), list(table), _ONE_SCALED

    remaining = tuple(var for var in scope if var not in fixed)
    if not remaining:
        full_assignment = 0
        for pos, bit in fixed_positions:
            full_assignment |= bit << pos
        return (), [], table[full_assignment]

    residual_scope = tuple(remap[var] for var in remaining)
    remaining_positions = tuple(scope.index(var) for var in remaining)
    restricted = [_ZERO_SCALED] * (1 << len(remaining))
    for residual_assignment in range(len(restricted)):
        full_assignment = 0
        for residual_pos, full_pos in enumerate(remaining_positions):
            full_assignment |= ((residual_assignment >> residual_pos) & 1) << full_pos
        for full_pos, bit in fixed_positions:
            full_assignment |= bit << full_pos
        restricted[residual_assignment] = table[full_assignment]
    return residual_scope, restricted, _ONE_SCALED


def _sum_factor_tables_with_cutset_scaled(
    n_vars: int,
    factors: Mapping[tuple[int, ...], Sequence[ScaledComplex]],
    plan: _ArbitraryFactorCutsetPlan,
    *,
    scalar: ScaledComplex,
    require_native: bool,
) -> tuple[ScaledComplex, int]:
    cutset = tuple(int(var) for var in plan.cutset)
    cutset_set = set(cutset)
    remaining_original = tuple(var for var in range(n_vars) if var not in cutset_set)
    remap = {var: idx for idx, var in enumerate(remaining_original)}
    total = _ZERO_SCALED
    max_scope = len(cutset)

    for mask in range(1 << len(cutset)):
        fixed = {var: (mask >> idx) & 1 for idx, var in enumerate(cutset)}
        branch_scalar = scalar
        branch_factors: dict[tuple[int, ...], Sequence[ScaledComplex]] = {}
        for scope, table in factors.items():
            residual_scope, residual_table, residual_scalar = _restrict_scaled_factor_table(
                tuple(scope),
                table,
                fixed,
                remap,
            )
            branch_scalar = _mul_scaled_complex(branch_scalar, residual_scalar)
            if residual_scope:
                branch_scalar = _mul_scaled_complex(
                    branch_scalar,
                    _combine_factor_scaled(branch_factors, residual_scope, residual_table),
                )
        branch_total, branch_scope = _sum_factor_tables_scaled(
            len(remaining_original),
            branch_factors,
            plan.residual_order,
            scalar=branch_scalar,
            require_native=require_native,
        )
        total = _add_scaled_complex(total, branch_total)
        max_scope = max(max_scope, len(cutset) + int(branch_scope))

    return total, max_scope

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
