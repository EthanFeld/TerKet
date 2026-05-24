"""Scaled-complex arithmetic and result wrappers."""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Sequence

import numpy as np

from .cache import register_lru_cache


ScaledComplex = tuple[complex, int]
_SQRT2 = math.sqrt(2.0)
_INV_SQRT2 = 1.0 / _SQRT2
_SCALED_RENORMALIZE_MIN = math.ldexp(1.0, -256)
_SCALED_RENORMALIZE_MAX = math.ldexp(1.0, 256)


def _normalize_scaled_complex(value: complex, half_pow2_exp: int) -> ScaledComplex:
    if value == 0j:
        return 0j, 0

    magnitude = max(abs(value.real), abs(value.imag))
    _, shift = math.frexp(magnitude)
    if shift:
        value = complex(
            math.ldexp(value.real, -shift),
            math.ldexp(value.imag, -shift),
        )
        half_pow2_exp += 2 * shift
    return value, half_pow2_exp


def _make_scaled_complex(value: complex) -> ScaledComplex:
    return _normalize_scaled_complex(complex(value), 0)


def _renormalize_scaled_complex_if_needed(value: complex, half_pow2_exp: int) -> ScaledComplex:
    if value == 0j:
        return 0j, 0

    magnitude = max(abs(value.real), abs(value.imag))
    if _SCALED_RENORMALIZE_MIN <= magnitude < _SCALED_RENORMALIZE_MAX:
        return value, half_pow2_exp
    return _normalize_scaled_complex(value, half_pow2_exp)


def _scale_scaled_complex(scaled: ScaledComplex, half_pow2_exp: int) -> ScaledComplex:
    value, base_half_pow2_exp = scaled
    return _normalize_scaled_complex(value, base_half_pow2_exp + half_pow2_exp)


def _scale_complex_by_half_pow2(value: complex, half_pow2_exp: int) -> complex:
    """Scale a complex value by 2 ** (half_pow2_exp / 2) without huge floats."""
    if value == 0j or half_pow2_exp == 0:
        return complex(value)

    scaled = complex(value)
    if half_pow2_exp > 0 and half_pow2_exp % 2:
        scaled *= _SQRT2
        half_pow2_exp -= 1
    elif half_pow2_exp < 0 and half_pow2_exp % 2:
        scaled *= _INV_SQRT2
        half_pow2_exp += 1

    shift = half_pow2_exp // 2
    return complex(
        math.ldexp(scaled.real, shift),
        math.ldexp(scaled.imag, shift),
    )


def _scale_complex_array_by_half_pow2(values: np.ndarray, half_pow2_exp: np.ndarray) -> np.ndarray:
    """Vectorized companion to ``_scale_complex_by_half_pow2``."""
    scaled = np.asarray(values, dtype=np.complex128).copy()
    exponents = np.asarray(half_pow2_exp, dtype=np.int64).copy()
    if scaled.size == 0:
        return scaled

    odd_mask = (exponents & 1) != 0
    positive_odd = odd_mask & (exponents > 0)
    negative_odd = odd_mask & (exponents < 0)
    if np.any(positive_odd):
        scaled[positive_odd] *= _SQRT2
        exponents[positive_odd] -= 1
    if np.any(negative_odd):
        scaled[negative_odd] *= _INV_SQRT2
        exponents[negative_odd] += 1

    shift = exponents // 2
    return np.ldexp(scaled.real, shift) + 1j * np.ldexp(scaled.imag, shift)


def _normalize_scaled_complex_arrays(
    values: np.ndarray,
    half_pow2_exp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize arrays of scaled-complex values elementwise."""
    values_array = np.asarray(values, dtype=np.complex128).copy()
    exponents = np.asarray(half_pow2_exp, dtype=np.int64).copy()
    if values_array.size == 0:
        return values_array, exponents

    zero_mask = values_array == 0j
    if np.any(~zero_mask):
        magnitudes = np.maximum(np.abs(values_array.real), np.abs(values_array.imag))
        _, shifts = np.frexp(magnitudes[~zero_mask])
        shift_array = np.zeros_like(exponents)
        shift_array[~zero_mask] = shifts.astype(np.int64, copy=False)
        nonzero_shift = (~zero_mask) & (shift_array != 0)
        if np.any(nonzero_shift):
            values_array.real[nonzero_shift] = np.ldexp(
                values_array.real[nonzero_shift],
                -shift_array[nonzero_shift],
            )
            values_array.imag[nonzero_shift] = np.ldexp(
                values_array.imag[nonzero_shift],
                -shift_array[nonzero_shift],
            )
            exponents[nonzero_shift] += 2 * shift_array[nonzero_shift]
    exponents[zero_mask] = 0
    return values_array, exponents


def _scaled_arrays_from_constant(
    scaled: ScaledComplex,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast one scaled value into array form."""
    value, half_pow2_exp = scaled
    return (
        np.full(shape, complex(value), dtype=np.complex128),
        np.full(shape, int(half_pow2_exp), dtype=np.int64),
    )


def _scaled_table_to_arrays(table: Sequence[ScaledComplex]) -> tuple[np.ndarray, np.ndarray]:
    """Pack a scaled table into parallel value/exponent arrays."""
    return (
        np.asarray([complex(value) for value, _exp in table], dtype=np.complex128),
        np.asarray([int(exp) for _value, exp in table], dtype=np.int64),
    )


def _scaled_list_to_arrays(
    scaled_list: Sequence[ScaledComplex],
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape a list of scaled values into parallel arrays."""
    values = np.asarray([complex(value) for value, _exp in scaled_list], dtype=np.complex128)
    exponents = np.asarray([int(exp) for _value, exp in scaled_list], dtype=np.int64)
    return values.reshape(shape), exponents.reshape(shape)


def _mul_scaled_complex_arrays(
    left_values: np.ndarray,
    left_half_pow2_exp: np.ndarray,
    right_values: np.ndarray,
    right_half_pow2_exp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise batch multiplication of scaled-complex arrays."""
    return _normalize_scaled_complex_arrays(
        np.asarray(left_values, dtype=np.complex128) * np.asarray(right_values, dtype=np.complex128),
        np.asarray(left_half_pow2_exp, dtype=np.int64) + np.asarray(right_half_pow2_exp, dtype=np.int64),
    )


def _add_scaled_complex_arrays(
    left_values: np.ndarray,
    left_half_pow2_exp: np.ndarray,
    right_values: np.ndarray,
    right_half_pow2_exp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise batch addition of scaled-complex arrays."""
    left_values = np.asarray(left_values, dtype=np.complex128)
    left_exponents = np.asarray(left_half_pow2_exp, dtype=np.int64)
    right_values = np.asarray(right_values, dtype=np.complex128)
    right_exponents = np.asarray(right_half_pow2_exp, dtype=np.int64)

    result_values = left_values.copy()
    result_exponents = left_exponents.copy()

    left_zero = left_values == 0j
    right_zero = right_values == 0j
    take_right = left_zero & ~right_zero
    if np.any(take_right):
        result_values[take_right] = right_values[take_right]
        result_exponents[take_right] = right_exponents[take_right]

    active = ~(left_zero | right_zero)
    if not np.any(active):
        result_exponents[result_values == 0j] = 0
        return result_values, result_exponents

    base_values = left_values[active].copy()
    base_exponents = left_exponents[active].copy()
    other_values = right_values[active].copy()
    other_exponents = right_exponents[active].copy()

    swap_mask = base_exponents < other_exponents
    if np.any(swap_mask):
        base_values_swap = base_values[swap_mask].copy()
        base_exponents_swap = base_exponents[swap_mask].copy()
        base_values[swap_mask] = other_values[swap_mask]
        base_exponents[swap_mask] = other_exponents[swap_mask]
        other_values[swap_mask] = base_values_swap
        other_exponents[swap_mask] = base_exponents_swap

    aligned_other = _scale_complex_array_by_half_pow2(
        other_values,
        other_exponents - base_exponents,
    )
    summed_values, summed_exponents = _normalize_scaled_complex_arrays(
        base_values + aligned_other,
        base_exponents,
    )
    result_values[active] = summed_values
    result_exponents[active] = summed_exponents
    result_exponents[result_values == 0j] = 0
    return result_values, result_exponents


def _mul_scaled_complex(left: ScaledComplex, right: ScaledComplex) -> ScaledComplex:
    left_value, left_half_pow2_exp = left
    right_value, right_half_pow2_exp = right
    return _renormalize_scaled_complex_if_needed(
        left_value * right_value,
        left_half_pow2_exp + right_half_pow2_exp,
    )


def _add_scaled_complex(left: ScaledComplex, right: ScaledComplex) -> ScaledComplex:
    left_value, left_half_pow2_exp = left
    right_value, right_half_pow2_exp = right

    if left_value == 0j:
        return right
    if right_value == 0j:
        return left
    if left_half_pow2_exp < right_half_pow2_exp:
        left_value, right_value = right_value, left_value
        left_half_pow2_exp, right_half_pow2_exp = right_half_pow2_exp, left_half_pow2_exp

    aligned_right = _scale_complex_by_half_pow2(
        right_value,
        right_half_pow2_exp - left_half_pow2_exp,
    )
    return _renormalize_scaled_complex_if_needed(left_value + aligned_right, left_half_pow2_exp)


def _scaled_to_complex(
    scaled: ScaledComplex,
    extra_scalar: complex = 1.0 + 0j,
    extra_half_pow2: int = 0,
) -> complex:
    value, half_pow2_exp = scaled
    return _scale_complex_by_half_pow2(
        complex(extra_scalar) * value,
        half_pow2_exp + extra_half_pow2,
    )


def _scaled_to_plain_complex(value: ScaledComplex) -> complex:
    return complex(value[0]) * (2.0 ** (int(value[1]) / 2.0))


def _complex_logsum(values: Sequence[complex]) -> complex:
    if not values:
        return complex(float("-inf"), 0.0)
    pivot = max(values, key=abs)
    if pivot == 0j:
        return complex(float("-inf"), 0.0)
    return cmath.log(pivot) + cmath.log(sum(value / pivot for value in values))


def _scaled_from_complex_log(log_value: complex) -> ScaledComplex:
    if not math.isfinite(log_value.real):
        return _ZERO_SCALED
    half_pow2 = int(round((2.0 * log_value.real) / math.log(2.0)))
    value = cmath.exp(log_value - (0.5 * half_pow2 * math.log(2.0)))
    return _normalize_scaled_complex(value, half_pow2)


def _scaled_complex_log(value: ScaledComplex) -> complex | None:
    scalar, half_pow2 = value
    if scalar == 0j:
        return None
    return cmath.log(scalar) + 0.5 * int(half_pow2) * math.log(2.0)


def _scaled_probability_log2(value: ScaledComplex) -> float:
    scalar, half_pow2 = value
    if scalar == 0j:
        return -math.inf
    return 2.0 * math.log2(abs(scalar)) + float(half_pow2)


def _scaled_log2_abs(value: ScaledComplex) -> float:
    scalar, half_pow2 = value
    if scalar == 0j:
        return -math.inf
    return math.log2(abs(scalar)) + 0.5 * float(half_pow2)


def _scaled_phase(value: ScaledComplex) -> float | None:
    scalar, _half_pow2 = value
    if scalar == 0j:
        return None
    return math.atan2(scalar.imag, scalar.real)


def _scaled_complex_ratio_to_plain(
    numerator: ScaledComplex,
    denominator: ScaledComplex,
) -> complex | None:
    num_value, num_half_pow2 = numerator
    den_value, den_half_pow2 = denominator
    if num_value == 0j:
        return 0j
    if den_value == 0j:
        return None
    log_ratio = (
        cmath.log(num_value)
        - cmath.log(den_value)
        + 0.5 * (int(num_half_pow2) - int(den_half_pow2)) * math.log(2.0)
    )
    if not (math.isfinite(log_ratio.real) and math.isfinite(log_ratio.imag)):
        return None
    if log_ratio.real > 700.0:
        return None
    if log_ratio.real < -745.0:
        return 0j
    return cmath.exp(log_ratio)


@dataclass(frozen=True, slots=True)
class ScaledAmplitude:
    """Amplitude represented as ``mantissa * 2 ** (half_pow2_exp / 2)``."""

    mantissa: complex
    half_pow2_exp: int = 0

    def __post_init__(self) -> None:
        value, half_pow2_exp = _normalize_scaled_complex(self.mantissa, self.half_pow2_exp)
        object.__setattr__(self, "mantissa", value)
        object.__setattr__(self, "half_pow2_exp", half_pow2_exp)

    @classmethod
    def from_tuple(cls, scaled: ScaledComplex) -> ScaledAmplitude:
        value, half_pow2_exp = scaled
        return cls(value, half_pow2_exp)

    def as_tuple(self) -> ScaledComplex:
        return self.mantissa, self.half_pow2_exp

    def to_complex(self) -> complex:
        return _scaled_to_complex(self.as_tuple())

    def log2_abs(self) -> float:
        if self.mantissa == 0j:
            return -math.inf
        return math.log2(abs(self.mantissa)) + self.half_pow2_exp / 2.0


_ONE_SCALED = _make_scaled_complex(1.0 + 0j)
_ZERO_SCALED = _make_scaled_complex(0j)


@lru_cache(maxsize=16)
def _omega_table(level: int) -> tuple[complex, ...]:
    modulus = 1 << level
    return tuple(cmath.exp(2j * cmath.pi * residue / modulus) for residue in range(modulus))


@lru_cache(maxsize=16)
def _omega_scaled_table(level: int) -> tuple[ScaledComplex, ...]:
    return tuple(_make_scaled_complex(value) for value in _omega_table(level))


@lru_cache(maxsize=16)
def _omega_plus_one_scaled_table(level: int) -> tuple[ScaledComplex, ...]:
    return tuple(_make_scaled_complex(1 + value) for value in _omega_table(level))


@lru_cache(maxsize=16)
def _omega_scaled_arrays(level: int) -> tuple[np.ndarray, np.ndarray]:
    return _scaled_table_to_arrays(_omega_scaled_table(level))


register_lru_cache("engine.omega_table", _omega_table)
register_lru_cache("engine.omega_scaled_table", _omega_scaled_table)
register_lru_cache("engine.omega_plus_one_scaled_table", _omega_plus_one_scaled_table)
register_lru_cache("engine.omega_scaled_arrays", _omega_scaled_arrays)


__all__ = [
    "ScaledAmplitude",
    "ScaledComplex",
    "_ONE_SCALED",
    "_ZERO_SCALED",
    "_add_scaled_complex",
    "_add_scaled_complex_arrays",
    "_complex_logsum",
    "_make_scaled_complex",
    "_mul_scaled_complex",
    "_mul_scaled_complex_arrays",
    "_normalize_scaled_complex",
    "_normalize_scaled_complex_arrays",
    "_omega_plus_one_scaled_table",
    "_omega_scaled_arrays",
    "_omega_scaled_table",
    "_omega_table",
    "_renormalize_scaled_complex_if_needed",
    "_scale_complex_array_by_half_pow2",
    "_scale_complex_by_half_pow2",
    "_scale_scaled_complex",
    "_scaled_arrays_from_constant",
    "_scaled_complex_log",
    "_scaled_complex_ratio_to_plain",
    "_scaled_from_complex_log",
    "_scaled_list_to_arrays",
    "_scaled_log2_abs",
    "_scaled_phase",
    "_scaled_probability_log2",
    "_scaled_table_to_arrays",
    "_scaled_to_complex",
    "_scaled_to_plain_complex",
]
