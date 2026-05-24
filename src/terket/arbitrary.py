"""Arbitrary-angle exact facade."""

from __future__ import annotations

from ._arbitrary_clusters import _ArbitraryFactorCutsetPlan
from ._arbitrary_factors import (
    _add_arbitrary_phase_factors_scaled,
    _arbitrary_phase_factor_table,
    _arbitrary_phase_terms_are_unary,
    _build_unary_arbitrary_factor_tables,
    _coalesce_arbitrary_phase_terms,
    _restrict_scaled_factor_table,
    _restrict_unary_arbitrary_factor_tables,
    _sum_factor_tables_with_cutset_scaled,
    _sum_q3_free_with_unary_arbitrary_phases_scaled,
    _sum_q3_free_with_unary_factor_tables_for_order_scaled,
    _sum_q3_free_with_unary_factor_tables_scaled,
)
from ._arbitrary_runtime import (
    _arbitrary_factor_graph_for_state_output,
    _sum_with_arbitrary_phases_exact_scaled,
    _sum_with_arbitrary_phases_scaled,
    solve_arbitrary_exact,
)
from ._state_runtime import _ArbitraryPhaseTerm
from .scaling import _complex_logsum

__all__ = [
    "_ArbitraryFactorCutsetPlan",
    "_ArbitraryPhaseTerm",
    "_add_arbitrary_phase_factors_scaled",
    "_arbitrary_factor_graph_for_state_output",
    "_arbitrary_phase_factor_table",
    "_arbitrary_phase_terms_are_unary",
    "_build_unary_arbitrary_factor_tables",
    "_coalesce_arbitrary_phase_terms",
    "_complex_logsum",
    "_restrict_scaled_factor_table",
    "_restrict_unary_arbitrary_factor_tables",
    "_sum_factor_tables_with_cutset_scaled",
    "_sum_q3_free_with_unary_arbitrary_phases_scaled",
    "_sum_q3_free_with_unary_factor_tables_for_order_scaled",
    "_sum_q3_free_with_unary_factor_tables_scaled",
    "_sum_with_arbitrary_phases_exact_scaled",
    "_sum_with_arbitrary_phases_scaled",
    "solve_arbitrary_exact",
]
