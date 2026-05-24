
"""Exact hard-support cluster and mediator factor helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ._engine_runtime_core import _configure_extracted_module

_LOCAL_NAMES = {
    '_ArbitraryFactorCutsetPlan',
    '_HalfPhaseMediatorSpec',
    '_HalfPhaseMediatorPlan',
    '_GenericQ2MediatorSpec',
    '_GenericQ2MediatorPlan',
    '_HalfPhaseClusterSpec',
    '_HalfPhaseClusterPlan',
    '_build_generic_q1_cluster_plan',
    '_build_q1_cluster_plan',
    '_fold_phase_shifted_q1_batch',
    '_evaluate_half_phase_cluster_plan_scaled',
    '_build_core_factor_batch',
    '_evaluate_half_phase_mediator_plan_scaled_batch',
    '_evaluate_generic_q2_mediator_plan_scaled_batch',
    '_evaluate_half_phase_cluster_plan_scaled_batch',

}
_LOCAL_IMPLS = {}
_configure_extracted_module(
    globals(),
    local_names=_LOCAL_NAMES,
    local_impls=_LOCAL_IMPLS,
    respect_mock_wraps=True,
)

@dataclass(frozen=True, slots=True)
class _ArbitraryFactorCutsetPlan:
    cutset: tuple[int, ...]
    residual_order: tuple[int, ...]
    residual_width: int
    residual_work: int
    residual_table_entries: int

@dataclass(frozen=True, slots=True)
class _HalfPhaseMediatorSpec:
    mediator_var: int
    neighbor_vars: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class _HalfPhaseMediatorPlan:
    level: int
    core_vars: tuple[int, ...]
    core_q2: dict[tuple[int, int], int]
    order: tuple[int, ...]
    width: int
    mediators: tuple[_HalfPhaseMediatorSpec, ...]

@dataclass(frozen=True, slots=True)
class _GenericQ2MediatorSpec:
    mediator_var: int
    neighbor_vars: tuple[int, ...]
    neighbor_couplings: tuple[int, ...]
    assignment_residue_shifts: tuple[int, ...] = ()

@dataclass(frozen=True, slots=True)
class _GenericQ2MediatorPlan:
    level: int
    core_vars: tuple[int, ...]
    core_q2: dict[tuple[int, int], int]
    order: tuple[int, ...]
    width: int
    mediators: tuple[_GenericQ2MediatorSpec, ...]

@dataclass(frozen=True, slots=True)
class _HalfPhaseClusterSpec:
    cluster_vars: tuple[int, ...]
    boundary_vars: tuple[int, ...]
    internal_q2: dict[tuple[int, int], int]
    boundary_couplings: tuple[tuple[int, int, int], ...]
    boundary_shift_table: np.ndarray | None = None
    cluster_order: tuple[int, ...] = ()
    native_treewidth_plan: object | None = None

@dataclass(frozen=True, slots=True)
class _HalfPhaseClusterPlan:
    level: int
    core_vars: tuple[int, ...]
    core_q2: dict[tuple[int, int], int]
    order: tuple[int, ...]
    width: int
    clusters: tuple[_HalfPhaseClusterSpec, ...]

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
    cluster_plan = _build_generic_q1_cluster_plan(q)
    if cluster_plan is not None:
        return cluster_plan
    cluster_plan = _build_block_cut_tree_region_plan(q)
    if cluster_plan is not None:
        return cluster_plan
    return _build_small_boundary_region_plan(q)

def _fold_phase_shifted_q1_batch(
    q1_batch: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Deduplicate identical phase-shifted q1 rows while preserving encounter order."""
    batch = np.ascontiguousarray(np.asarray(q1_batch, dtype=np.int64))
    if batch.ndim != 2:
        raise ValueError("Expected q1_batch to have shape (batch, n_vars).")
    if len(batch) == 0:
        return np.zeros((0, batch.shape[1]), dtype=np.int64), []

    row_map: dict[bytes, int] = {}
    unique_rows: list[np.ndarray] = []
    inverse: list[int] = []
    for row in batch:
        key = row.tobytes()
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

    total = _sum_acyclic_factor_tables_scaled(
        len(cluster_plan.core_vars),
        factors,
        scalar=scalar,
    )
    if total is None:
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


def _prepare_core_factor_plan_batch(
    q1_local_batch: np.ndarray,
    *,
    level: int,
    core_vars: Sequence[int],
    core_q2: dict[tuple[int, int], int],
    max_index: int,
    label: str,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]]] | None:
    batch = np.ascontiguousarray(np.asarray(q1_local_batch, dtype=np.int64))
    if len(batch) == 0:
        return None
    if batch.shape[1] <= max_index:
        raise ValueError(
            f"Expected q1_local_batch to cover {label} indices through {max_index}, "
            f"received width {batch.shape[1]}."
        )
    scalar, factors = _build_core_factor_batch(
        level=level,
        core_vars=core_vars,
        core_q2=core_q2,
        q1_local_batch=batch,
    )
    return batch, scalar, factors

def _evaluate_half_phase_mediator_plan_scaled_batch(
    mediator_plan: _HalfPhaseMediatorPlan,
    q1_local_batch: np.ndarray,
) -> list[ScaledComplex]:
    """Batch exact evaluation of one half-phase mediator plan."""
    prepared = _prepare_core_factor_plan_batch(
        q1_local_batch,
        level=mediator_plan.level,
        core_vars=mediator_plan.core_vars,
        core_q2=mediator_plan.core_q2,
        max_index=max(
            max(mediator_plan.core_vars, default=-1),
            max((spec.mediator_var for spec in mediator_plan.mediators), default=-1),
        ),
        label="mediator-plan",
    )
    if prepared is None:
        return []
    batch, (scalar_values, scalar_exponents), factors = prepared
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
    prepared = _prepare_core_factor_plan_batch(
        q1_local_batch,
        level=mediator_plan.level,
        core_vars=mediator_plan.core_vars,
        core_q2=mediator_plan.core_q2,
        max_index=max(
            max(mediator_plan.core_vars, default=-1),
            max((spec.mediator_var for spec in mediator_plan.mediators), default=-1),
        ),
        label="generic-mediator",
    )
    if prepared is None:
        return []
    batch, (scalar_values, scalar_exponents), factors = prepared
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
    prepared = _prepare_core_factor_plan_batch(
        q1_local_batch,
        level=cluster_plan.level,
        core_vars=cluster_plan.core_vars,
        core_q2=cluster_plan.core_q2,
        max_index=max(
            max(cluster_plan.core_vars, default=-1),
            max((var for spec in cluster_plan.clusters for var in spec.cluster_vars), default=-1),
        ),
        label="cluster-plan",
    )
    if prepared is None:
        return []
    batch, (scalar_values, scalar_exponents), factors = prepared
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

    totals = _sum_acyclic_factor_tables_scaled_batch(
        len(cluster_plan.core_vars),
        factors,
        scalar=(scalar_values, scalar_exponents),
    )
    if totals is None:
        totals, _ = _sum_factor_tables_scaled_batch(
            len(cluster_plan.core_vars),
            factors,
            cluster_plan.order,
            scalar=(scalar_values, scalar_exponents),
        )
    return totals


_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
