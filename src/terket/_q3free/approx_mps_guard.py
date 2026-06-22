"""Convergence and reliability guard for boundary-MPS q3-free contraction."""

from __future__ import annotations

import math
from typing import Any

from ..scaling import ScaledComplex, _add_scaled_complex, _scaled_log2_abs
from ..state import SolverConfig
from .approx_mps import _sum_q3_free_boundary_mps_scaled


def _relative_change(high: ScaledComplex, low: ScaledComplex) -> tuple[float, float]:
    difference_log2 = _scaled_log2_abs(_add_scaled_complex(high, (-low[0], low[1])))
    high_log2 = _scaled_log2_abs(high)
    if difference_log2 == -math.inf:
        relative = 0.0
    elif high_log2 == -math.inf or difference_log2 - high_log2 > 1024.0:
        relative = math.inf
    else:
        relative = 2.0 ** (difference_log2 - high_log2)
    return relative, difference_log2


def _sum_q3_free_boundary_mps_checked_scaled(
    q,
    *,
    config: SolverConfig,
) -> ScaledComplex | None:
    high_bond = max(
        2,
        min(int(config.approx_tensor_max_bond), int(config.approx_tensor_mps_max_bond)),
    )
    low_bond = max(1, high_bond // 2)
    low_result = _sum_q3_free_boundary_mps_scaled(
        q, max_bond=low_bond, cutoff=float(config.approx_tensor_cutoff)
    )
    high_result = _sum_q3_free_boundary_mps_scaled(
        q, max_bond=high_bond, cutoff=float(config.approx_tensor_cutoff)
    )
    if low_result is None or high_result is None:
        return None
    low, _low_info = low_result
    high, high_info = high_result
    relative, error_log2 = _relative_change(high, low)
    log2_abs = _scaled_log2_abs(high)
    bound_violation = max(0.0, log2_abs - float(q.n))
    max_discarded = float(high_info["max_discarded"])
    reliable = (
        bound_violation <= 1e-9
        and relative <= float(config.approx_tensor_mps_max_rel_change)
        and max_discarded <= float(config.approx_tensor_mps_max_discarded)
    )
    reason = _rejection_reason(
        bound_violation=bound_violation,
        relative=relative,
        max_discarded=max_discarded,
        config=config,
    )
    diagnostics: dict[str, Any] = {
        "approx_q3_free_method": "boundary_mps",
        "approx_q3_free_reliable": bool(reliable),
        "approx_q3_free_log2_abs": float(log2_abs),
        "approx_q3_free_error_log2_abs": float(error_log2),
        "approx_q3_free_rel_stderr": float(relative),
        "approx_q3_free_mps_bond": int(high_bond),
        **{f"approx_q3_free_mps_{key}": value for key, value in high_info.items()},
    }
    if bound_violation:
        diagnostics["approx_q3_free_bound_violation_log2"] = float(bound_violation)
    if reason:
        diagnostics["approx_q3_free_rejection_reason"] = reason
    from .approx_guard import _set_q3_free_approx_diagnostics

    _set_q3_free_approx_diagnostics(diagnostics)
    if not reliable and bool(config.approx_tensor_reliability_reject):
        if bool(config.approx_tensor_raise_on_unreliable):
            raise RuntimeError(
                "Unreliable approximate q3-free boundary MPS estimate: "
                f"{reason}; relative_change={relative:.6g}, "
                f"max_discarded={max_discarded:.6g}."
            )
        return None
    return high


def _rejection_reason(
    *, bound_violation: float, relative: float, max_discarded: float, config: SolverConfig
) -> str:
    if bound_violation > 1e-9:
        return "count_bound"
    if max_discarded > float(config.approx_tensor_mps_max_discarded):
        return "mps_truncation"
    if relative > float(config.approx_tensor_mps_max_rel_change):
        return "mps_bond_convergence"
    return ""
