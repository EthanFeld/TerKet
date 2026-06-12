"""Engine-runtime splice module for reduction/export compatibility helpers."""

from __future__ import annotations

from ._engine_runtime_core import *

_bind_extracted_forwarders(
    "_reduction_classify",
    "_build_classification_data",
    "_classification_lookup",
    "_classification_entry",
    "_has_odd_bilinear_coupling",
    "_classify",
)

# ==================================================================
# Quadratic elimination [BL26 Prop. 9]
# ==================================================================

_bind_extracted_forwarders(
    "_reduction_classify",
    "_incident_quadratic_couplings",
    "_elim_sparse_dead_quadratics_batch",
)
_bind_extracted_forwarders(
    "_reduction_elim",
    "_elim_quadratic",
    "_elim_quadratic_via_split",
)

# ==================================================================
# Constraint elimination [BL26 Prop. 11]
# ==================================================================

_bind_extracted_forwarders(
    "_reduction_elim",
    "_elim_constraint",
    "_elim_single_partner_constraint_python",
    "_elim_single_partner_constraint",
    "_elim_two_partner_constraint_python",
    "_elim_two_partner_constraint",
    "_elim_two_partner_constraint_q3_free",
)

# ==================================================================
# Affine composition [generalized BL26 Prop. 4]
# ==================================================================

_bind_extracted_forwarders("_reduction_elim", "_aff_compose_python", "_aff_compose", "_info")

# ==================================================================
# Public API
# ==================================================================

__all__ = [name for name in globals() if not name.startswith("__")]
