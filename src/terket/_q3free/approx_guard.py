"""Reliability guard for approximate q3-free residue estimates."""

from __future__ import annotations

import contextvars
from dataclasses import replace
import math
from typing import Any

from ..scaling import (
    ScaledComplex,
    _add_scaled_complex,
    _make_scaled_complex,
    _normalize_scaled_complex,
    _scale_complex_by_half_pow2,
    _scaled_log2_abs,
)
from ..state import SolverConfig
from .approx_residue import _sum_q3_free_residue_forest_scaled

_Q3_FREE_APPROX_DIAGNOSTICS: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "_terket_q3_free_approx_diagnostics",
    default=None,
)


def _clear_q3_free_approx_diagnostics() -> None:
    _Q3_FREE_APPROX_DIAGNOSTICS.set(None)


def _set_q3_free_approx_diagnostics(info: dict[str, Any]) -> None:
    _Q3_FREE_APPROX_DIAGNOSTICS.set(dict(info))


def _get_q3_free_approx_diagnostics() -> dict[str, Any] | None:
    info = _Q3_FREE_APPROX_DIAGNOSTICS.get()
    return None if info is None else dict(info)


def _mean_scaled_complex(values: list[ScaledComplex]) -> ScaledComplex:
    if not values:
        return _make_scaled_complex(0j)
    total = values[0]
    for value in values[1:]:
        total = _add_scaled_complex(total, value)
    return _normalize_scaled_complex(total[0] / float(len(values)), total[1])


def _scaled_standard_error_log2(values: list[ScaledComplex], mean: ScaledComplex) -> float:
    if len(values) <= 1:
        return -math.inf
    ref_exp = max([mean[1], *(value[1] for value in values)])
    mean_at_ref = _scale_complex_by_half_pow2(mean[0], int(mean[1]) - ref_exp)
    diffs = [
        _scale_complex_by_half_pow2(value[0], int(value[1]) - ref_exp) - mean_at_ref
        for value in values
    ]
    variance = sum(abs(diff) ** 2 for diff in diffs) / float(len(values) * (len(values) - 1))
    stderr = math.sqrt(max(0.0, variance))
    if stderr == 0.0:
        return -math.inf
    return math.log2(stderr) + 0.5 * float(ref_exp)


def _log2_spread(log2_values: list[float]) -> float:
    finite = [value for value in log2_values if math.isfinite(value)]
    if not finite:
        return 0.0
    if len(finite) != len(log2_values):
        return math.inf
    return max(finite) - min(finite)


def _relative_stderr(error_log2_abs: float, mean_log2_abs: float) -> float:
    if error_log2_abs == -math.inf or mean_log2_abs == -math.inf:
        return 0.0 if error_log2_abs == -math.inf else math.inf
    delta = error_log2_abs - mean_log2_abs
    return math.inf if delta > 1024.0 else 2.0**delta


def _residue_forest_reliability_reason(
    q,
    config: SolverConfig,
    *,
    mean_log2_abs: float,
    spread: float,
    rel_stderr: float,
) -> tuple[bool, str, float]:
    count_bound_violation = max(0.0, mean_log2_abs - float(q.n))
    if count_bound_violation > 1e-9:
        return False, "count_bound", count_bound_violation
    if spread > float(config.approx_tensor_reliability_max_log2_spread):
        return False, "log2_spread", count_bound_violation
    if (
        mean_log2_abs >= float(config.approx_tensor_reliability_min_log2_abs_for_rel)
        and rel_stderr > float(config.approx_tensor_reliability_max_rel_stderr)
    ):
        return False, "relative_stderr", count_bound_violation
    return True, "", count_bound_violation


def _residue_channel_modes(config: SolverConfig, repeats: int) -> tuple[str, ...]:
    mode = str(config.approx_tensor_residue_sample_mode).strip().lower()
    if repeats >= 2 and mode in {"unified", "path", "path_variations", "nondegenerate"}:
        return "unified", "unified_random"
    return (mode,)


def _collect_residue_channel(
    q,
    config: SolverConfig,
    *,
    mode: str,
    repeats: int,
    seed_stride: int,
    channel_idx: int,
) -> list[ScaledComplex] | None:
    estimates: list[ScaledComplex] = []
    channel_offset = channel_idx * 32452843
    for repeat_idx in range(repeats):
        run_config = replace(
            config,
            approx_tensor_reliability_repeats=1,
            approx_tensor_residue_sample_mode=mode,
            approx_tensor_residue_seed=(
                int(config.approx_tensor_residue_seed)
                + channel_offset
                + repeat_idx * seed_stride
            ),
        )
        estimate = _sum_q3_free_residue_forest_scaled(q, config=run_config)
        if estimate is None:
            return None
        estimates.append(estimate)
    return estimates


def _combine_residue_channels(
    channels: list[list[ScaledComplex]],
) -> tuple[ScaledComplex, float, list[ScaledComplex]]:
    if len(channels) == 1:
        mean = _mean_scaled_complex(channels[0])
        return mean, _scaled_standard_error_log2(channels[0], mean), channels[0]
    means = [_mean_scaled_complex(estimates) for estimates in channels]
    errors = [
        _scaled_standard_error_log2(estimates, mean)
        for estimates, mean in zip(channels, means)
    ]
    exact = [idx for idx, error in enumerate(errors) if error == -math.inf]
    if exact:
        selected = [means[idx] for idx in exact]
        return _mean_scaled_complex(selected), -math.inf, [value for rows in channels for value in rows]
    finite = [idx for idx, error in enumerate(errors) if math.isfinite(error)]
    if not finite:
        merged = [value for rows in channels for value in rows]
        mean = _mean_scaled_complex(merged)
        return mean, _scaled_standard_error_log2(merged, mean), merged
    reference = min(errors[idx] for idx in finite)
    precision = [2.0 ** (-2.0 * (errors[idx] - reference)) for idx in finite]
    precision_sum = sum(precision)
    weighted = [
        _normalize_scaled_complex(means[idx][0] * (weight / precision_sum), means[idx][1])
        for idx, weight in zip(finite, precision)
    ]
    mean = weighted[0]
    for value in weighted[1:]:
        mean = _add_scaled_complex(mean, value)
    error_log2_abs = reference - 0.5 * math.log2(precision_sum)
    return mean, error_log2_abs, [value for rows in channels for value in rows]


def _sum_q3_free_residue_forest_checked_scaled(q, *, config: SolverConfig) -> ScaledComplex | None:
    repeats = max(1, int(config.approx_tensor_reliability_repeats))
    seed_stride = max(1, int(config.approx_tensor_reliability_seed_stride))
    modes = _residue_channel_modes(config, repeats)
    channels: list[list[ScaledComplex]] = []
    for channel_idx, mode in enumerate(modes):
        estimates = _collect_residue_channel(
            q,
            config,
            mode=mode,
            repeats=repeats,
            seed_stride=seed_stride,
            channel_idx=channel_idx,
        )
        if estimates is None:
            _set_q3_free_approx_diagnostics(
                {
                    "approx_q3_free_method": "residue_forest",
                    "approx_q3_free_reliable": False,
                    "approx_q3_free_rejection_reason": "backend_unavailable",
                    "approx_q3_free_repeats": 0,
                    "approx_q3_free_level": int(config.approx_tensor_residue_level),
                    "approx_q3_free_samples": int(config.approx_tensor_residue_forest_samples),
                }
            )
            return None
        channels.append(estimates)

    mean, error_log2_abs, estimates = _combine_residue_channels(channels)
    log2_values = [_scaled_log2_abs(value) for value in estimates]
    mean_log2_abs = _scaled_log2_abs(mean)
    rel_stderr = _relative_stderr(error_log2_abs, mean_log2_abs)
    spread = _log2_spread(log2_values)
    reliable, reason, count_bound_violation = _residue_forest_reliability_reason(
        q,
        config,
        mean_log2_abs=mean_log2_abs,
        spread=spread,
        rel_stderr=rel_stderr,
    )
    diagnostics: dict[str, Any] = {
        "approx_q3_free_method": "residue_forest",
        "approx_q3_free_reliable": bool(reliable),
        "approx_q3_free_repeats": int(repeats),
        "approx_q3_free_level": int(config.approx_tensor_residue_level),
        "approx_q3_free_samples": int(config.approx_tensor_residue_forest_samples) * len(modes),
        "approx_q3_free_log2_abs": float(mean_log2_abs),
        "approx_q3_free_error_log2_abs": float(error_log2_abs),
        "approx_q3_free_rel_stderr": float(rel_stderr),
        "approx_q3_free_log2_spread": float(spread),
    }
    if count_bound_violation > 0.0:
        diagnostics["approx_q3_free_bound_violation_log2"] = float(count_bound_violation)
    if not reliable:
        diagnostics["approx_q3_free_rejection_reason"] = reason
    if not reliable and bool(config.approx_tensor_mps_fallback):
        from .approx_mps_guard import _sum_q3_free_boundary_mps_checked_scaled

        mps_total = _sum_q3_free_boundary_mps_checked_scaled(q, config=config)
        if mps_total is not None:
            return mps_total
        mps_diagnostics = _get_q3_free_approx_diagnostics()
        if mps_diagnostics and mps_diagnostics.get("approx_q3_free_method") == "boundary_mps":
            return None
    _set_q3_free_approx_diagnostics(diagnostics)
    if not reliable and bool(config.approx_tensor_reliability_reject):
        if bool(config.approx_tensor_raise_on_unreliable):
            raise RuntimeError(
                "Unreliable approximate q3-free residue_forest estimate: "
                f"{reason}; log2_spread={spread:.6g}, rel_stderr={rel_stderr:.6g}."
            )
        return None
    return mean
