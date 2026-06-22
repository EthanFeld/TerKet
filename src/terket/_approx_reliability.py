"""Reliability metadata helpers for approximate summation backends."""

from __future__ import annotations

import math
from typing import Any, Mapping, MutableMapping

from .scaling import ScaledComplex, _scaled_log2_abs
from .state import SolverConfig, _get_solver_config

APPROX_REDUCER_KEYS = (
    "approx_q3_free_method",
    "approx_q3_free_reliable",
    "approx_q3_free_rejection_reason",
    "approx_q3_free_repeats",
    "approx_q3_free_level",
    "approx_q3_free_samples",
    "approx_q3_free_log2_abs",
    "approx_q3_free_error_log2_abs",
    "approx_q3_free_rel_stderr",
    "approx_q3_free_log2_spread",
    "approx_q3_free_bound_violation_log2",
    "approx_q3_free_mps_bond",
    "approx_q3_free_mps_order",
    "approx_q3_free_mps_route_swaps",
    "approx_q3_free_mps_width",
    "approx_q3_free_mps_peak_active",
    "approx_q3_free_mps_peak_bond",
    "approx_q3_free_mps_discarded_rss",
    "approx_q3_free_mps_max_discarded",
)


def _copy_approx_reducer_info(target: MutableMapping[str, Any], source: Mapping[str, Any]) -> None:
    for key in APPROX_REDUCER_KEYS:
        if key in source:
            target[key] = source[key]


def _scaled_amplitude_bound_violation_log2(
    scaled_amp: ScaledComplex,
    *,
    config: SolverConfig | None = None,
) -> float:
    cfg = _get_solver_config() if config is None else config
    log2_abs = _scaled_log2_abs(scaled_amp)
    if log2_abs == -math.inf:
        return 0.0
    slack = max(0.0, float(cfg.approx_tensor_amplitude_bound_slack_log2))
    return max(0.0, float(log2_abs) - slack)


def _validate_approx_amplitude_reliability(
    scaled_amp: ScaledComplex,
    info: MutableMapping[str, Any],
    *,
    config: SolverConfig | None = None,
) -> None:
    if "approx_q3_free_method" not in info:
        return
    cfg = _get_solver_config() if config is None else config
    violation = _scaled_amplitude_bound_violation_log2(scaled_amp, config=cfg)
    if violation <= 0.0:
        return
    info["approx_q3_free_reliable"] = False
    info["approx_q3_free_rejection_reason"] = "amplitude_bound"
    info["approx_q3_free_bound_violation_log2"] = float(violation)
    if bool(cfg.approx_tensor_raise_on_unreliable):
        raise RuntimeError(
            "Unreliable approximate q3-free amplitude: "
            f"|amplitude| exceeds 1 by {violation:.6g} log2 units."
        )
