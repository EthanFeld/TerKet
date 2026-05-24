"""Grouped Phase-3 planning and residual-cubic execution helpers."""

from __future__ import annotations

from ._phase3.cover import _approximate_q3_vertex_cover, _greedy_q3_vertex_cover, _minimum_q3_vertex_cover, _pick_q3_branch_edge
from ._phase3.exec import _sum_via_q3_cover
from ._phase3.factors import _sum_via_treewidth_dp, _sum_via_treewidth_dp_scaled
from ._phase3.select import _phase3_plan, _prefer_cubic_contraction_phase3, _select_direct_phase3_backend
from ._phase3.structure import _estimate_q3_cover_work, _prefer_treewidth_phase3, _q3_core_cover_size
from ._q3free.fallbacks import _minimum_bad_q2_vertex_cover

__all__ = [
    "_approximate_q3_vertex_cover",
    "_estimate_q3_cover_work",
    "_greedy_q3_vertex_cover",
    "_minimum_bad_q2_vertex_cover",
    "_minimum_q3_vertex_cover",
    "_phase3_plan",
    "_pick_q3_branch_edge",
    "_prefer_cubic_contraction_phase3",
    "_prefer_treewidth_phase3",
    "_q3_core_cover_size",
    "_select_direct_phase3_backend",
    "_sum_via_q3_cover",
    "_sum_via_treewidth_dp",
    "_sum_via_treewidth_dp_scaled",
]
