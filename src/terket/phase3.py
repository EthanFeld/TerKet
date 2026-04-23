"""Grouped Phase-3 planning and residual-cubic execution helpers."""

from __future__ import annotations

from .engine import (
    _approximate_q3_vertex_cover,
    _estimate_q3_cover_work,
    _greedy_q3_vertex_cover,
    _minimum_bad_q2_vertex_cover,
    _minimum_q3_vertex_cover,
    _phase3_plan,
    _pick_q3_branch_edge,
    _prefer_cubic_contraction_phase3,
    _prefer_hybrid_contraction_phase3,
    _prefer_tensor_contraction_phase3,
    _prefer_treewidth_phase3,
    _q3_core_cover_size,
    _select_direct_phase3_backend,
    _sum_via_q3_cover,
    _sum_via_tensor_contraction,
    _sum_via_treewidth_dp,
    _sum_via_treewidth_dp_scaled,
)

__all__ = [
    "_approximate_q3_vertex_cover",
    "_estimate_q3_cover_work",
    "_greedy_q3_vertex_cover",
    "_minimum_bad_q2_vertex_cover",
    "_minimum_q3_vertex_cover",
    "_phase3_plan",
    "_pick_q3_branch_edge",
    "_prefer_cubic_contraction_phase3",
    "_prefer_hybrid_contraction_phase3",
    "_prefer_tensor_contraction_phase3",
    "_prefer_treewidth_phase3",
    "_q3_core_cover_size",
    "_select_direct_phase3_backend",
    "_sum_via_q3_cover",
    "_sum_via_tensor_contraction",
    "_sum_via_treewidth_dp",
    "_sum_via_treewidth_dp_scaled",
]
