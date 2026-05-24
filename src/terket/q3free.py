"""Grouped q3-free planning and exact summation entry points."""

from __future__ import annotations

from ._phase3.exec import _build_q3_free_branch_template, _evaluate_q3_free_branch_template_batch
from ._q3free.batch import Q3FreeBranchTemplate
from ._q3free.cutset import _q3_free_cutset_conditioning_plan, _q3_free_one_shot_cutset_conditioning_plan
from ._q3free.cutset_exec import _sum_q3_free_via_cutset_conditioning_scaled
from ._q3free.cutset_residue import _build_q3_free_cutset_residue_data
from ._q3free.exact import (
    _gauss_sum_q3_free,
    _gauss_sum_q3_free_scaled,
    _sum_q3_free_via_gauss_reduction,
    _sum_q3_free_via_gauss_reduction_scaled,
)
from ._q3free.fallbacks import (
    _sum_q3_free_via_bad_q2_cover_scaled,
    _sum_q3_free_via_nonquadratic_support,
    _sum_q3_free_via_nonquadratic_support_scaled,
)
from ._q3free.models import (
    _Q3FreeConstraintComponentPlan,
    _Q3FreeConstraintPlan,
    _Q3FreeCutsetCandidateEvaluation,
    _Q3FreeCutsetConditioningPlan,
    _Q3FreeRawConstraintPlan,
    _Q3FreeRawConstraintRestrictedPlan,
)
from ._q3free.native import _build_native_q3_free_treewidth_plan, _sum_q3_free_treewidth_dp_scaled_batch
from ._q3free.plans import (
    _build_q3_free_constraint_plan,
    _q3_free_edge_density,
    _q3_free_prefers_locality_preserving_cutset,
    _q3_free_prefers_one_shot_cutset,
    _q3_free_prefers_reusable_cutset,
)
from ._q3free.primitives import _q3_free_graph, _q3_free_phase3_backend_name
from ._q3free.raw_constraints import _build_q3_free_raw_constraint_plan
from ._q3free.treewidth import _q3_free_treewidth_order, _sum_q3_free_component, _sum_q3_free_component_scaled
from ._reduction_classify import _classify
from ._reduction_elim import _elim_two_partner_constraint_q3_free

__all__ = [
    "Q3FreeBranchTemplate",
    "_Q3FreeConstraintComponentPlan",
    "_Q3FreeConstraintPlan",
    "_Q3FreeCutsetCandidateEvaluation",
    "_Q3FreeCutsetConditioningPlan",
    "_Q3FreeRawConstraintPlan",
    "_Q3FreeRawConstraintRestrictedPlan",
    "_build_native_q3_free_treewidth_plan",
    "_build_q3_free_branch_template",
    "_build_q3_free_constraint_plan",
    "_build_q3_free_cutset_residue_data",
    "_build_q3_free_raw_constraint_plan",
    "_classify",
    "_elim_two_partner_constraint_q3_free",
    "_evaluate_q3_free_branch_template_batch",
    "_gauss_sum_q3_free",
    "_gauss_sum_q3_free_scaled",
    "_q3_free_cutset_conditioning_plan",
    "_q3_free_edge_density",
    "_q3_free_graph",
    "_q3_free_one_shot_cutset_conditioning_plan",
    "_q3_free_phase3_backend_name",
    "_q3_free_prefers_locality_preserving_cutset",
    "_q3_free_prefers_one_shot_cutset",
    "_q3_free_prefers_reusable_cutset",
    "_q3_free_treewidth_order",
    "_sum_q3_free_component",
    "_sum_q3_free_component_scaled",
    "_sum_q3_free_treewidth_dp_scaled_batch",
    "_sum_q3_free_via_bad_q2_cover_scaled",
    "_sum_q3_free_via_cutset_conditioning_scaled",
    "_sum_q3_free_via_gauss_reduction",
    "_sum_q3_free_via_gauss_reduction_scaled",
    "_sum_q3_free_via_nonquadratic_support",
    "_sum_q3_free_via_nonquadratic_support_scaled",
]
