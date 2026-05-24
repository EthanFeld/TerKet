"""Exact reduction and affine-composition facade."""

from __future__ import annotations

import importlib
from ._reduction_classify import (
    _build_classification_data,
    _classification_entry,
    _classification_lookup,
    _classify,
    _elim_sparse_dead_quadratics_batch,
)
from ._reduction_elim import (
    _aff_compose,
    _aff_compose_python,
    _elim_constraint,
    _elim_quadratic,
    _elim_quadratic_via_split,
    _elim_single_partner_constraint,
    _elim_single_partner_constraint_python,
    _elim_two_partner_constraint,
    _elim_two_partner_constraint_python,
    _info,
)
from ._reduction_runtime import (
    _apply_exact_eliminations,
    _elim_decoupled_constraints_batch,
    _product_q1_sum,
    _product_q1_sum_scaled,
    _reduce_and_sum,
    _reduce_and_sum_scaled,
    _reduce_and_sum_scaled_batch,
)
from ._reduction_support import _ReductionContext
from .state import ReducerInfo, ReductionInfo


def affine_compose(*args, **kwargs):
    module = importlib.import_module("terket._amplitude_api")
    return module.affine_compose(*args, **kwargs)


def reduce_and_sum(*args, **kwargs):
    module = importlib.import_module("terket._amplitude_api")
    return module.reduce_and_sum(*args, **kwargs)

__all__ = [
    "ReducerInfo",
    "ReductionInfo",
    "_ReductionContext",
    "_aff_compose",
    "_aff_compose_python",
    "_apply_exact_eliminations",
    "_build_classification_data",
    "_classify",
    "_classification_entry",
    "_classification_lookup",
    "_elim_constraint",
    "_elim_decoupled_constraints_batch",
    "_elim_quadratic",
    "_elim_quadratic_via_split",
    "_elim_single_partner_constraint",
    "_elim_single_partner_constraint_python",
    "_elim_sparse_dead_quadratics_batch",
    "_elim_two_partner_constraint",
    "_elim_two_partner_constraint_python",
    "_info",
    "_product_q1_sum",
    "_product_q1_sum_scaled",
    "_reduce_and_sum",
    "_reduce_and_sum_scaled",
    "_reduce_and_sum_scaled_batch",
    "affine_compose",
    "reduce_and_sum",
]
