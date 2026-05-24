
"""Shared exact factor-table algebra helpers."""

from __future__ import annotations

from ._engine_runtime_core import _configure_extracted_module

_LOCAL_NAMES = {
    '_factor_table_multiply',
    '_factor_table_multiply_scaled',
    '_project_assignment_bits',
    '_combine_factor',
    '_combine_factor_scaled',
    '_sum_acyclic_factor_tables_scaled',
    '_sum_acyclic_factor_tables_scaled_batch',
    '_factor_table_multiply_scaled_batch',
    '_combine_factor_scaled_batch',
    '_sum_factor_tables_scaled_batch',
    '_sum_factor_tables_scaled',
}
_LOCAL_IMPLS = {}
_configure_extracted_module(
    globals(),
    local_names=_LOCAL_NAMES,
    local_impls=_LOCAL_IMPLS,
    respect_mock_wraps=True,
)

def _factor_table_multiply(left, right):
    return [left_value * right_value for left_value, right_value in zip(left, right)]

def _factor_table_multiply_scaled(left, right):
    return [_mul_scaled_complex(left_value, right_value) for left_value, right_value in zip(left, right)]

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

def _sum_acyclic_factor_tables_scaled(
    n_vars: int,
    factors: dict[tuple[int, ...], list[ScaledComplex]],
    *,
    scalar: ScaledComplex = _ONE_SCALED,
) -> ScaledComplex | None:
    """Exact sum-product on an acyclic variable/factor graph."""
    factor_items = [(tuple(scope), table) for scope, table in factors.items() if scope]
    if not factor_items:
        return _scale_scaled_complex(scalar, 2 * n_vars)

    used_vars = sorted({int(var) for scope, _table in factor_items for var in scope})
    if not used_vars:
        return scalar

    factor_neighbors = [tuple(int(var) for var in scope) for scope, _table in factor_items]
    var_to_factors: dict[int, list[int]] = {var: [] for var in used_vars}
    for factor_idx, scope in enumerate(factor_neighbors):
        for var in scope:
            var_to_factors[var].append(factor_idx)

    edge_count = sum(len(scope) for scope in factor_neighbors)
    node_count = len(used_vars) + len(factor_items)
    visited_vars: set[int] = set()
    visited_factors: set[int] = set()
    component_roots: list[tuple[str, int]] = []

    for start_var in used_vars:
        if start_var in visited_vars:
            continue
        component_roots.append(("var", start_var))
        queue: deque[tuple[str, int]] = deque([("var", start_var)])
        while queue:
            kind, idx = queue.popleft()
            if kind == "var":
                if idx in visited_vars:
                    continue
                visited_vars.add(idx)
                for factor_idx in var_to_factors.get(idx, ()):
                    if factor_idx not in visited_factors:
                        queue.append(("factor", factor_idx))
            else:
                if idx in visited_factors:
                    continue
                visited_factors.add(idx)
                for var in factor_neighbors[idx]:
                    if var not in visited_vars:
                        queue.append(("var", var))

    component_count = len(component_roots)
    if edge_count != node_count - component_count:
        return None

    factor_to_var_cache: dict[tuple[int, int], tuple[ScaledComplex, ScaledComplex]] = {}
    var_to_factor_cache: dict[tuple[int, int], tuple[ScaledComplex, ScaledComplex]] = {}

    def msg_var_to_factor(var: int, parent_factor: int) -> tuple[ScaledComplex, ScaledComplex]:
        cache_key = (var, parent_factor)
        cached = var_to_factor_cache.get(cache_key)
        if cached is not None:
            return cached
        values = [_ONE_SCALED, _ONE_SCALED]
        for factor_idx in var_to_factors.get(var, ()):
            if factor_idx == parent_factor:
                continue
            incoming = msg_factor_to_var(factor_idx, var)
            values[0] = _mul_scaled_complex(values[0], incoming[0])
            values[1] = _mul_scaled_complex(values[1], incoming[1])
        result = (values[0], values[1])
        var_to_factor_cache[cache_key] = result
        return result

    def msg_factor_to_var(factor_idx: int, parent_var: int) -> tuple[ScaledComplex, ScaledComplex]:
        cache_key = (factor_idx, parent_var)
        cached = factor_to_var_cache.get(cache_key)
        if cached is not None:
            return cached
        scope, table = factor_items[factor_idx]
        parent_pos = scope.index(parent_var)
        other_positions = [idx for idx in range(len(scope)) if idx != parent_pos]
        other_messages = [
            msg_var_to_factor(scope[pos], factor_idx)
            for pos in other_positions
        ]
        outputs = [_ZERO_SCALED, _ZERO_SCALED]
        for parent_bit in (0, 1):
            total = _ZERO_SCALED
            for assignment in range(1 << len(other_positions)):
                full_assignment = 0
                full_assignment |= parent_bit << parent_pos
                for offset, pos in enumerate(other_positions):
                    bit = (assignment >> offset) & 1
                    full_assignment |= bit << pos
                weight = table[full_assignment]
                for offset, pos in enumerate(other_positions):
                    bit = (assignment >> offset) & 1
                    weight = _mul_scaled_complex(weight, other_messages[offset][bit])
                total = _add_scaled_complex(total, weight)
            outputs[parent_bit] = total
        result = (outputs[0], outputs[1])
        factor_to_var_cache[cache_key] = result
        return result

    def component_total_from_var(root_var: int) -> ScaledComplex:
        incoming = msg_var_to_factor(root_var, -1)
        return _add_scaled_complex(incoming[0], incoming[1])

    total = scalar
    used_var_set = set(used_vars)
    isolated_count = max(0, n_vars - len(used_var_set))
    if isolated_count:
        total = _scale_scaled_complex(total, 2 * isolated_count)
    for kind, idx in component_roots:
        if kind != "var":
            continue
        total = _mul_scaled_complex(total, component_total_from_var(idx))
    return total

def _sum_acyclic_factor_tables_scaled_batch(
    n_vars: int,
    factors: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]],
    *,
    scalar: tuple[np.ndarray, np.ndarray],
) -> list[ScaledComplex] | None:
    """Batched companion to ``_sum_acyclic_factor_tables_scaled``."""
    scalar_values, scalar_exponents = scalar
    batch_size = len(scalar_values)
    if batch_size == 0:
        return []
    row_totals: list[ScaledComplex] = []
    for row_idx in range(batch_size):
        row_factors = {
            tuple(scope): [
                (complex(values[row_idx, assignment]), int(exponents[row_idx, assignment]))
                for assignment in range(values.shape[1])
            ]
            for scope, (values, exponents) in factors.items()
        }
        total = _sum_acyclic_factor_tables_scaled(
            n_vars,
            row_factors,
            scalar=(complex(scalar_values[row_idx]), int(scalar_exponents[row_idx])),
        )
        if total is None:
            return None
        row_totals.append(total)
    return row_totals

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
    require_native: bool = False,
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
        except Exception as exc:
            if require_native:
                raise RuntimeError("Native factor-table path-sum backend failed.") from exc
    elif require_native:
        raise RuntimeError("Native factor-table path-sum backend is unavailable.")

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


_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
