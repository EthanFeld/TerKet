"""Owned q3-free cutset runtime helpers.

Owns:
- generic-remaining finalization for chosen cutset plans
- reusable runtime cache attachment for branch-conditioned evaluation
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .batch import _branch_assignment_bits
from .._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals

_LOCAL_NAMES = {
    "_finalize_q3_free_cutset_conditioning_plan",
    "_attach_q3_free_cutset_runtime_cache",
}


_MISSING = object()
_ENGINE_LOCAL_BASELINES = {}
_LOCAL_IMPLS = {}


def _sync_from_engine(engine) -> None:
    _sync_extracted_globals(
        globals(),
        engine,
        local_names=_LOCAL_NAMES,
        local_impls=_LOCAL_IMPLS,
        baselines=_ENGINE_LOCAL_BASELINES,
        missing=_MISSING,
        respect_mock_wraps=True,
    )


_MISSING, _ENGINE_LOCAL_BASELINES = _bootstrap_extracted_globals(
    globals(),
    local_names=_LOCAL_NAMES,
    local_impls=_LOCAL_IMPLS,
    respect_mock_wraps=True,
)


def _finalize_q3_free_cutset_conditioning_plan(
    plan: _Q3FreeCutsetConditioningPlan,
    *,
    prefer_one_shot_slicing: bool = False,
) -> _Q3FreeCutsetConditioningPlan:
    """Fill generic remaining plans only for the chosen cutset."""
    if (
        plan.remaining_backend != "generic"
        or plan.remaining_components
        or (not plan.remaining_q2 and not plan.remaining_isolated_vars)
    ):
        return plan

    remaining_q = _phase_function_from_parts(
        len(plan.remaining_vars),
        level=plan.level,
        q0=Fraction(0),
        q1=[0] * len(plan.remaining_vars),
        q2=plan.remaining_q2,
        q3={},
    )
    isolated_vars, component_plans = _plan_q3_free_constraint_components(
        remaining_q,
        0,
        allow_tensor_contraction=True,
        prefer_reusable_decomposition=False,
        prefer_one_shot_slicing=prefer_one_shot_slicing,
    )
    component_width = max(
        (_q3_free_component_plan_width_hint(component_plan) for component_plan in component_plans),
        default=0,
    )
    component_work = max(
        1,
        sum(_q3_free_component_plan_work_hint(component_plan) for component_plan in component_plans),
    )
    branch_count = 1 << len(plan.cutset_vars)
    return _Q3FreeCutsetConditioningPlan(
        level=plan.level,
        cutset_vars=plan.cutset_vars,
        remaining_vars=plan.remaining_vars,
        remaining_backend=plan.remaining_backend,
        remaining_q2=plan.remaining_q2,
        remaining_order=plan.remaining_order,
        cutset_remaining_q2_residue=plan.cutset_remaining_q2_residue,
        cutset_cutset_left=plan.cutset_cutset_left,
        cutset_cutset_right=plan.cutset_cutset_right,
        cutset_cutset_residue=plan.cutset_cutset_residue,
        native_treewidth_plan=plan.native_treewidth_plan,
        remaining_isolated_vars=tuple(int(var) for var in isolated_vars),
        remaining_components=tuple(component_plans),
        remaining_width=max(plan.remaining_width, component_width),
        estimated_total_work=max(plan.estimated_total_work, branch_count * component_work),
    )


def _attach_q3_free_cutset_runtime_cache(
    plan: _Q3FreeCutsetConditioningPlan,
) -> _Q3FreeCutsetConditioningPlan:
    """Attach reusable branch-side residue arrays to a cutset plan."""
    if plan.branch_bits is not None:
        return plan

    cutset_size = len(plan.cutset_vars)
    branch_count = 1 << cutset_size
    branch_masks = np.arange(branch_count, dtype=np.uint64)
    branch_bits = _branch_assignment_bits(branch_masks, cutset_size).astype(np.int64)

    if plan.cutset_cutset_residue.size:
        branch_pair_residue = np.zeros(branch_count, dtype=np.int64)
        for left, right, residue in zip(
            plan.cutset_cutset_left,
            plan.cutset_cutset_right,
            plan.cutset_cutset_residue,
        ):
            branch_pair_residue = (
                branch_pair_residue
                + int(residue) * branch_bits[:, int(left)] * branch_bits[:, int(right)]
            ) % (1 << int(plan.level))
    else:
        branch_pair_residue = np.zeros(branch_count, dtype=np.int64)

    if plan.cutset_remaining_q2_residue.size:
        branch_remaining_shift = (
            branch_bits @ np.asarray(plan.cutset_remaining_q2_residue, dtype=np.int64)
        ) % (1 << int(plan.level))
    else:
        branch_remaining_shift = np.zeros(
            (branch_count, len(plan.remaining_vars)),
            dtype=np.int64,
        )

    return _Q3FreeCutsetConditioningPlan(
        level=plan.level,
        cutset_vars=plan.cutset_vars,
        remaining_vars=plan.remaining_vars,
        remaining_backend=plan.remaining_backend,
        remaining_q2=plan.remaining_q2,
        remaining_order=plan.remaining_order,
        cutset_remaining_q2_residue=plan.cutset_remaining_q2_residue,
        cutset_cutset_left=plan.cutset_cutset_left,
        cutset_cutset_right=plan.cutset_cutset_right,
        cutset_cutset_residue=plan.cutset_cutset_residue,
        native_treewidth_plan=plan.native_treewidth_plan,
        remaining_isolated_vars=plan.remaining_isolated_vars,
        remaining_components=plan.remaining_components,
        remaining_width=plan.remaining_width,
        estimated_total_work=plan.estimated_total_work,
        branch_bits=branch_bits,
        branch_pair_residue=branch_pair_residue,
        branch_remaining_shift=branch_remaining_shift,
    )


_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
