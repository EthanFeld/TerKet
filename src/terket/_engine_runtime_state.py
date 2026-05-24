from __future__ import annotations

from ._engine_runtime_core import *


_bind_extracted_forwarders(
    "_state_runtime",
    "_apply_affine_bit_in_place",
    "_apply_diag_phase_in_place",
    "_apply_bilinear_phase_in_place",
    "_solve_output_from_echelon",
)
_bind_extracted_forwarders(
    "_arbitrary_factors",
    "_coalesce_arbitrary_phase_terms",
    "_arbitrary_phase_terms_are_unary",
    "_build_unary_arbitrary_factor_tables",
    "_restrict_unary_arbitrary_factor_tables",
    "_sum_q3_free_with_unary_factor_tables_for_order_scaled",
    "_evaluate_q3_free_remaining_with_unary_factor_tables_scaled",
    "_evaluate_q3_free_cutset_conditioning_plan_with_unary_factor_tables_scaled",
    "_sum_q3_free_with_unary_factor_tables_scaled",
    "_sum_q3_free_with_unary_arbitrary_phases_scaled",
    "_arbitrary_phase_factor_table",
    "_add_arbitrary_phase_factors_scaled",
    "_restrict_scaled_factor_table",
    "_sum_factor_tables_with_cutset_scaled",
)
_bind_extracted_forwarders(
    "_arbitrary_bp",
    "_arbitrary_bp_backend",
    "_arbitrary_exact_metadata",
    "_arbitrary_approx_metadata",
    "_mark_invalid_arbitrary_bp_info",
    "_raise_if_invalid_arbitrary_bp_amplitude",
    "_sum_pairwise_factor_graph_bethe_scaled",
    "_sum_factor_graph_bethe_scaled",
    "_sum_factor_graph_with_sparse_parity_bethe_scaled",
)
_bind_extracted_forwarders(
    "_arbitrary_runtime",
    "_factor_graph_is_forest",
    "_scaled_log2_abs",
    "_scaled_phase",
    "_phase_distance",
    "_arbitrary_bp_heuristic_candidate",
    "_sum_arbitrary_bp_heuristic_ensemble_scaled",
    "_arbitrary_factor_graph_for_state_output",
    "solve_arbitrary_exact",
    "solve_arbitrary_approx",
    "_sum_with_arbitrary_phases_scaled",
)
_bind_extracted_forwarders(
    "_pauli_support",
    "_normalize_pauli_expbox_terms",
    "apply_pauli_expbox_to_state",
)

_sum_with_arbitrary_phases_exact_scaled = solve_arbitrary_exact
_sum_with_arbitrary_phases_approx_scaled = solve_arbitrary_approx

# ==================================================================
# Schur-state construction and output constraint solving
# ==================================================================

# ==================================================================
# Core: recursive reduction and summation
# ==================================================================

_bind_extracted_forwarders(
    "_reduction_runtime",
    "_pre_exact_phase3_treewidth_escape",
    "_reduce_and_sum_scaled",
    "_reduce_and_sum",
    "_reduce_and_sum_scaled_batch",
    "_invert_native_gate",
    "_invert_native_gates",
    "_fork_state_for_extension",
    "_pauli_string_gates",
    "_validate_pauli_observables",
    "_elim_decoupled_constraints_batch",
    "_apply_exact_eliminations",
    "_product_q1_sum",
    "_product_q1_sum_scaled",
)

# ==================================================================
# q3-free exact summation and scaled-number helpers
# ==================================================================

_bind_extracted_forwarders(
    "_q3free.primitives",
    "_q3_free_graph",
    "_is_binary_phase_quadratic",
    "_is_half_phase_q2",
    "_is_binary_phase_q1_vector",
    "_nonbinary_unary_support_size",
    "_is_qubit_quadratic_tensor_q1_vector",
    "_is_qubit_quadratic_tensor",
    "_q3_free_phase3_backend_name",
    "_component_fixed_nonbinary_unary_support_size",
    "_build_binary_phase_quadratic_plan",
    "_evaluate_binary_phase_quadratic_plan_scaled_batch",
    "_sum_binary_phase_quadratic_scaled",
    "_sum_half_phase_q2_unary_expansion_with_plan_scaled",
    "_sum_half_phase_q2_unary_expansion_with_plan_scaled_batch",
    "_sum_half_phase_q2_unary_expansion_scaled",
    "_apply_safe_q3_free_parity_substitutions",
    "_half_phase_parity_component_reduction",
    "_sum_half_phase_parity_component_reduction_scaled",
    "_build_half_phase_mediator_plan",
    "_build_generic_q2_mediator_plan",
)
_bind_extracted_forwarders(
    "_q3free.factor_plans",
    "_factor_scope_order",
    "_estimate_factor_table_dp_cost",
    "_factor_order_scope_sets",
    "_factor_cutset_residual_scopes",
    "_factor_cutset_candidates",
    "_find_arbitrary_factor_cutset_plan",
    "_factor_scope_degeneracy",
)
_bind_extracted_forwarders(
    "_q3free.clusters",
    "_build_cluster_boundary_shift_table",
    "_build_q2_adjacency",
    "_build_selected_boundary_region_plan",
    "_small_boundary_region_candidates",
    "_articulation_boundary_region_candidates",
    "_q2_block_cut_decomposition",
    "_block_cut_boundary_region_candidates",
    "_build_small_boundary_region_plan",
    "_build_block_cut_tree_region_plan",
    "_build_half_phase_cluster_plan",
)
_bind_extracted_forwarders(
    "_arbitrary_clusters",
    "_build_generic_q1_cluster_plan",
    "_build_q1_cluster_plan",
    "_fold_phase_shifted_q1_batch",
    "_evaluate_half_phase_cluster_plan_scaled",
    "_build_core_factor_batch",
    "_evaluate_half_phase_mediator_plan_scaled_batch",
    "_evaluate_generic_q2_mediator_plan_scaled_batch",
    "_evaluate_half_phase_cluster_plan_scaled_batch",
)

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

def _engine_cache(name: str, max_entries: int) -> _BoundedMemoCache:
    return make_bounded_cache(f"engine.{name}", max_entries)

_STRUCTURE_CLASSIFICATION_DATA_CACHE = _engine_cache("structure.classification_data", _STRUCTURE_CLASSIFICATION_CACHE_MAX)
_STRUCTURE_CLASSIFICATION_LOOKUP_CACHE = _engine_cache("structure.classification_lookup", _STRUCTURE_CLASSIFICATION_CACHE_MAX)
_STRUCTURE_MIN_FILL_CACHE = _engine_cache("structure.min_fill", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_Q3_COVER_CACHE = _engine_cache("structure.q3_cover", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_BAD_Q2_COVER_CACHE = _engine_cache("structure.bad_q2_cover", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_PLAN_CACHE = _engine_cache("structure.phase3_plan", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_TREEWIDTH_FACTOR_CACHE = _engine_cache("structure.phase3_treewidth_factor", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_TREEWIDTH_NATIVE_PLAN_CACHE = _engine_cache("structure.phase3_treewidth_native_plan", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_TREEWIDTH_BATCH_SUPPORT_CACHE = _engine_cache("structure.phase3_treewidth_batch_support", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_SUPPORT_PLAN_CACHE = _engine_cache("structure.phase3_support_plan", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_LEVEL3_NATIVE_PLAN_CACHE = _engine_cache("structure.phase3_level3_native_plan", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_LEVEL3_NATIVE_PLAN_SEEN_CACHE = _engine_cache("structure.phase3_level3_native_plan_seen", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_LEVEL3_BATCH_NATIVE_PLAN_CACHE = _engine_cache("structure.phase3_level3_batch_native_plan", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_GENERIC_BATCH_NATIVE_PLAN_CACHE = _engine_cache("structure.phase3_generic_batch_native_plan", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_BATCH_FAST_PLAN_CACHE = _engine_cache("structure.phase3_batch_fast_plan", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_FACTOR_CACHE = _engine_cache("structure.phase3_factor", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_Q3_SEPARATOR_CACHE = _engine_cache("structure.q3_separator", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_Q3_COVER_TEMPLATE_CACHE = _engine_cache("structure.q3_cover_template", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_FIX_VARIABLE_TEMPLATE_CACHE = _engine_cache("structure.fix_variable_template", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_Q3_2CORE_CACHE = _engine_cache("structure.q3_2core", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_Q3_FREE_CUTSET_PLAN_CACHE = _engine_cache("structure.q3_free_cutset_plan", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_Q3_FREE_TENSOR_HINT_CACHE = _engine_cache("structure.q3_free_tensor_hint", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_Q3_FREE_EXECUTION_PLAN_CACHE = _engine_cache("structure.q3_free_execution_plan", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_Q3_FREE_REUSABLE_EXECUTION_PLAN_CACHE = _engine_cache("structure.q3_free_reusable_execution_plan", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_Q3_FREE_BLOCK_CUT_PLAN_CACHE = _engine_cache("structure.q3_free_block_cut_plan", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_Q3_FREE_REFINED_ORDER_CACHE = _engine_cache("structure.q3_free_refined_order", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_OPT_CACHE = _engine_cache("structure.phase3_opt", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_REFINED_ORDER_CACHE = _engine_cache("structure.phase3_refined_order", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_INTERACTION_GRAPH_CACHE = _engine_cache("structure.interaction_graph", _STRUCTURE_PHASE3_CACHE_MAX)
_STRUCTURE_PHASE3_TREEWIDTH_CUTSET_CACHE = _engine_cache("structure.phase3_treewidth_cutset", _STRUCTURE_PHASE3_CACHE_MAX)

_bind_extracted_forwarders(
    "_reduction_support",
    "_build_early_elim_batch_size",
    "_project_quadratic_elimination_q2_nnz_delta",
    "_should_defer_build_quadratic_elimination",
)

@lru_cache(maxsize=64)
def _fraction_from_residue(*args, **kwargs):
    return _call_extracted("_reduction_support", "_fraction_from_residue", *args, **kwargs)

register_lru_cache("engine.fraction_from_residue", _fraction_from_residue)

_PACK_QKEY_HEADER = struct.Struct("<iiqq")
_PACK_QKEY_Q1 = struct.Struct("<q")
_PACK_QKEY_Q2 = struct.Struct("<iiq")
_PACK_QKEY_Q3 = struct.Struct("<iiiq")
_PACK_QSTRUCT_HEADER = struct.Struct("<ii")

_bind_extracted_forwarders(
    "_reduction_support",
    "_q_key_digest",
    "_q_key",
    "_cache_phase_structure_key",
    "_q_structure_key",
    "_q_phase3_structure_key",
    "_q_classification_structure_key",
    "_q_q3_support_key",
    "_q_cubic_treewidth_batch_support_key",
    "_build_cubic_treewidth_batch_support",
    "_phase3_support_plan",
    "_phase3_batch_support_plan_fast",
    "_phase_function_from_parts",
    "_phase_function_from_parts_mutable",
)

@lru_cache(maxsize=1 << 15)
def _direct_affine_mask_pattern(*args, **kwargs):
    return _call_extracted("_state_direct", "_direct_affine_mask_pattern", *args, **kwargs)

register_lru_cache("engine.direct_affine_mask_pattern", _direct_affine_mask_pattern)

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

def _copy_cubic_function_extended(q, final_n: int):
    target_n = max(int(final_n), int(q.n))
    q1 = list(q.q1)
    if target_n > int(q.n):
        q1.extend([0] * (target_n - int(q.n)))
    phase = _phase_function_from_parts(
        target_n,
        level=q.level,
        q0=q.q0,
        q1=q1,
        q2=dict(q.q2),
        q3=dict(q.q3),
    )
    phase._schur_mutable = True
    return phase

def _lift_direct_linear_coeff(level: int, coeff: int, precision_level: int) -> int:
    modulus = 1 << int(level)
    return (int(coeff) * (1 << (int(level) - int(precision_level)))) % modulus

def _lift_direct_quadratic_coeff(level: int, coeff: int, precision_level: int) -> int:
    modulus = max(1, 1 << (int(level) - 1))
    return (int(coeff) * (1 << (int(level) - int(precision_level)))) % modulus

_bind_extracted_forwarders(
    "_state_direct",
    "_build_post_replay_state",
    "_build_direct_post_replay_validation_observable",
    "_build_direct_post_replay_template",
    "_construct_direct_post_replay_payload",
    "_direct_post_replay_payload_matches_state",
)

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
    if context is None or getattr(q, "_schur_mutable", True):
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

__all__ = [name for name in globals() if not name.startswith("__")]
