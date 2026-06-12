"""Engine-runtime splice module for Phase-3 planning and execution exports."""

from __future__ import annotations

from ._engine_runtime_core import *


_bind_extracted_forwarders(
    "_phase3.order",
    "_q3_free_treewidth_candidate_is_viable",
    "_q3_hypergraph_2core",
    "_active_q3_variables",
    "_phase_function_q2_density_milli",
    "_phase_function_structure_score",
    "_phase_structure_opt_max_vars",
    "_phase_structure_opt_active_vars",
    "_phase_structure_opt_beam_width",
    "_phase_structure_opt_max_passes",
    "_phase_structure_local_region_max_vars",
    "_phase_structure_local_max_centers",
    "_phase_structure_local_max_passes",
    "_phase_structure_local_candidate_pool",
    "_phase_structure_hotspot_centers",
    "_phase_structure_local_region",
    "_phase_structure_local_move_score",
)
_bind_synced_local_impl_forwarders(
    "_phase3.structure",
    "_optimize_phase_function_structure_locally",
    "_optimize_phase_function_structure",
    "_estimate_q3_cover_work",
    "_prefer_treewidth_phase3",
)
_bind_extracted_forwarders(
    "_phase3.structure",
    "_basis_xor_transform",
    "_phase_function_basis_candidate_variables",
    "_phase_function_basis_transform_candidates",
    "_simplify_q3_basis",
    "_projected_components_after_fixing",
    "_find_small_q3_separator",
    "_q3_core_cover_size",
    "_estimate_q3_separator_work",
    "_phase3_treewidth_cutset_width_limit",
    "_phase3_residual_after_cutset",
    "_phase3_cutset_worst_residual",
    "_phase3_treewidth_cutset_candidates",
    "_find_q3_treewidth_cutset",
    "_estimate_q3_treewidth_cutset_work",
    call_local=True,
)

_CUBIC_CONTRACTION_MAX_WIDTH = 12  # numpy bucket elim beats quimb up to this width
_Q3_COVER_BRANCH_CHUNK_MAX = 128
_Q3_COVER_ASSIGNMENT_CHUNK_LOG2 = 13

_bind_extracted_forwarders(
    "_phase3.select",
    "_prefer_cubic_contraction_phase3",
    "_select_direct_phase3_backend",
    "_phase3_backend_runtime_score",
    "_phase3_treewidth_candidate",
    "_phase3_cubic_contraction_candidate",
    "_phase3_separator_candidate",
    "_phase3_treewidth_cutset_candidate",
    "_phase3_cover_candidate",
    "_choose_phase3_backend",
    "_phase3_plan",
)

_PHASE3_BACKEND_CANDIDATE_BUILDERS = (
    _phase3_treewidth_candidate,
    _phase3_cubic_contraction_candidate,
    _phase3_separator_candidate,
    _phase3_treewidth_cutset_candidate,
    _phase3_cover_candidate,
)

_bind_extracted_forwarders(
    "_phase3.factors",
    "_build_cubic_factors",
    "_build_cubic_factors_scaled",
    "_freeze_complex_factor_tables",
    "_build_cached_cubic_factors",
    "_freeze_scaled_factor_tables",
    "_build_cached_phase3_treewidth_factor_plan_scaled",
    "_build_native_phase3_treewidth_plan",
    "_build_native_level3_phase3_treewidth_plan",
    "_build_native_level3_phase3_treewidth_batch_support_plan",
    "_sum_native_level3_phase3_treewidth_batch_shared_support",
    "_build_native_phase_function_treewidth_batch_support_plan",
    "_sum_native_phase_function_treewidth_batch_shared_support",
    "_maybe_get_native_level3_phase3_treewidth_plan",
    "_sum_native_level3_phase3_treewidth_preplanned",
    "_factor_table_to_tensor_data",
    "_sum_via_treewidth_dp",
    "_sum_via_treewidth_dp_scaled",
    "_sum_via_treewidth_dp_scaled_batch_shared_support",
    "_sum_via_treewidth_dp_peeled_scaled",
    "_sum_via_treewidth_dp_peeled",
)
_bind_extracted_forwarders(
    "_phase3.cover",
    "_evaluate_half_phase_mediator_plan_scaled",
    "_evaluate_generic_q2_mediator_plan_scaled",
    "_greedy_q3_vertex_cover",
    "_approximate_q3_vertex_cover",
    "_q3_packing_lower_bound",
    "_pick_q3_branch_edge",
    "_minimum_q3_vertex_cover_uncached",
    "_minimum_vertex_cover_from_edge_masks",
    "_minimum_q3_vertex_cover",
)
_bind_extracted_forwarders(
    "_q3free.batch",
    "_as_int64_array",
    "_compact_unsigned_storage_dtype",
    "_compact_index_storage_array",
    "_compact_residue_storage_array",
    "_phase_fraction_to_residue",
    "_build_q3_free_branch_template",
    "_branch_assignment_bits",
    "_q3_cover_branch_chunk_size",
    "_evaluate_q3_free_branch_template_batch",
)
_bind_extracted_forwarders(
    "_phase3.exec",
    "_sum_via_q3_separator",
    "_sum_via_q3_treewidth_cutset",
    "_sum_via_q3_cover",
    "_sum_irreducible_cubic_core",
)


__all__ = [name for name in globals() if not name.startswith("__")]
