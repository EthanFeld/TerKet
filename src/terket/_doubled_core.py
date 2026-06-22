"""Core difference-sector construction and reduction."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from operator import index
from typing import Iterator

from ._engine_runtime_state import _aff_compose_cached
from ._reduction_runtime import _reduce_and_sum_scaled_batch
from ._reduction_support import _ReductionContext, _phase_function_from_parts
from .cubic_arithmetic import PhaseFunction
from .scaling import ScaledAmplitude, ScaledComplex, _add_scaled_complex, _normalize_scaled_complex
from .state import ExtendedReductionMode


@dataclass(frozen=True, slots=True)
class DoubledSumResult:
    """Truncated doubled-sum estimate and execution metadata."""

    estimate: ScaledAmplitude
    path_variables: int
    max_difference_weight: int
    sectors_evaluated: int
    sectors_total: int
    exact: bool
    max_reducer_remaining: int
    phase3_backends: tuple[str, ...]
    omitted_magnitude_bound: ScaledAmplitude | None = None

    def to_complex(self) -> complex:
        """Return the estimate as a normal complex number."""
        return self.estimate.to_complex()

    def to_float(self, *, imaginary_tolerance: float = 1e-9) -> float:
        """Return the real probability estimate, rejecting large numeric drift."""
        value = self.to_complex()
        if abs(value.imag) > imaginary_tolerance * max(1.0, abs(value.real)):
            raise ArithmeticError(
                f"Doubled-sum probability has imaginary residual {value.imag!r}."
            )
        return float(value.real)


def _add_term(terms: dict[tuple[int, ...], int], key: tuple[int, ...], value: int, modulus: int) -> None:
    value = (terms.get(key, 0) + value) % modulus
    if value:
        terms[key] = value
    else:
        terms.pop(key, None)


def _difference_phase(q: PhaseFunction, difference_mask: int) -> PhaseFunction:
    """Build ``q(x) - q(x xor d)`` without materializing a doubled polynomial."""
    q1 = [0] * q.n
    q2: dict[tuple[int, int], int] = {}
    q3: dict[tuple[int, int, int], int] = {}
    constant_residue = 0

    for idx, coeff in enumerate(q.q1):
        if coeff and ((difference_mask >> idx) & 1):
            constant_residue -= coeff
            q1[idx] = (q1[idx] + 2 * coeff) % q.mod_q1

    for (left, right), coeff in q.q2.items():
        left_flipped = (difference_mask >> left) & 1
        right_flipped = (difference_mask >> right) & 1
        weight = left_flipped + right_flipped
        if weight == 1:
            fixed = right if left_flipped else left
            q1[fixed] = (q1[fixed] - 2 * coeff) % q.mod_q1
            _add_term(q2, (left, right), 2 * coeff, q.mod_q2)
        elif weight == 2:
            constant_residue -= 2 * coeff
            q1[left] = (q1[left] + 2 * coeff) % q.mod_q1
            q1[right] = (q1[right] + 2 * coeff) % q.mod_q1

    for (a, b, c), coeff in q.q3.items():
        flipped = tuple(var for var in (a, b, c) if (difference_mask >> var) & 1)
        weight = len(flipped)
        if weight == 1:
            fixed = tuple(var for var in (a, b, c) if var not in flipped)
            _add_term(q2, fixed, -2 * coeff, q.mod_q2)
            _add_term(q3, (a, b, c), 2 * coeff, q.mod_q3)
        elif weight == 2:
            fixed = next(var for var in (a, b, c) if var not in flipped)
            q1[fixed] = (q1[fixed] - 4 * coeff) % q.mod_q1
            for var in flipped:
                _add_term(q2, tuple(sorted((var, fixed))), 2 * coeff, q.mod_q2)
        elif weight == 3:
            constant_residue -= 4 * coeff
            for var in (a, b, c):
                q1[var] = (q1[var] + 4 * coeff) % q.mod_q1
            for pair in ((a, b), (a, c), (b, c)):
                _add_term(q2, pair, -2 * coeff, q.mod_q2)
            _add_term(q3, (a, b, c), 2 * coeff, q.mod_q3)

    return _phase_function_from_parts(
        q.n,
        level=q.level,
        q0=Fraction(constant_residue, q.mod_q1) % 1,
        q1=q1,
        q2=q2,
        q3=q3,
    )


def _chunked_masks(
    n: int,
    max_weight: int,
    chunk_size: int,
    *,
    include_zero: bool = False,
) -> Iterator[list[int]]:
    chunk: list[int] = []
    for weight in range(0 if include_zero else 1, max_weight + 1):
        for support in combinations(range(n), weight):
            mask = sum(1 << var for var in support)
            chunk.append(mask)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def _sum_sector_rows(rows, total, max_remaining, backends):
    for sector_total, info in rows:
        total = _add_scaled_complex(total, sector_total)
        max_remaining = max(max_remaining, int(info["remaining"]))
        backend = info.get("phase3_backend")
        if backend:
            backends.add(str(backend))
    return total, max_remaining


def _clear_sector_caches(context: _ReductionContext) -> None:
    """Release sector-specific states while retaining reusable structural plans."""
    context.affine_compose_cache.clear()
    context.fix_variables_cache.clear()
    context.reduce_cache.clear()


def _result(total, path_variables, max_weight, sectors_evaluated, max_remaining, backends):
    return DoubledSumResult(
        estimate=ScaledAmplitude.from_tuple(total),
        path_variables=path_variables,
        max_difference_weight=max_weight,
        sectors_evaluated=sectors_evaluated,
        sectors_total=1 << path_variables,
        exact=max_weight == path_variables,
        max_reducer_remaining=max_remaining,
        phase3_backends=tuple(sorted(backends)),
    )


def sum_doubled_phase(
    q: PhaseFunction,
    *,
    max_difference_weight: int,
    sector_batch_size: int = 128,
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
) -> DoubledSumResult:
    """Approximate ``sum[x,y] exp(2 pi i (q(x)-q(y)))`` by low-weight ``x xor y``."""
    max_weight = _validate_options(q.n, max_difference_weight, sector_batch_size)
    context = _ReductionContext(
        preserve_scale=True,
        allow_tensor_contraction=allow_tensor_contraction,
        extended_reductions=extended_reductions,
    )
    if max_weight == q.n:
        rows = _reduce_and_sum_scaled_batch([q], context=context)
        amplitude, info = rows[0]
        total = _normalize_scaled_complex(abs(amplitude[0]) ** 2, 2 * amplitude[1])
        backend = info.get("phase3_backend")
        return _result(
            total,
            q.n,
            max_weight,
            1,
            int(info["remaining"]),
            {str(backend)} if backend else set(),
        )

    total: ScaledComplex = (1.0 + 0j, 2 * q.n)
    max_remaining = 0
    backends: set[str] = set()
    sectors_evaluated = 1
    for masks in _chunked_masks(q.n, max_weight, int(sector_batch_size)):
        rows = _reduce_and_sum_scaled_batch(
            [_difference_phase(q, mask) for mask in masks],
            context=context,
        )
        sectors_evaluated += len(rows)
        total, max_remaining = _sum_sector_rows(rows, total, max_remaining, backends)
        _clear_sector_caches(context)
    return _result(total, q.n, max_weight, sectors_evaluated, max_remaining, backends)


def _validate_options(n: int, max_difference_weight: int, sector_batch_size: int) -> int:
    try:
        max_weight = index(max_difference_weight)
        batch_size = index(sector_batch_size)
    except TypeError as exc:
        raise TypeError("Difference weight and sector batch size must be integers.") from exc
    if max_weight < 0:
        raise ValueError("max_difference_weight must be nonnegative.")
    if batch_size <= 0:
        raise ValueError("sector_batch_size must be positive.")
    return min(max_weight, n)


def sum_coupled_doubled_phase(
    q_xy: PhaseFunction,
    *,
    contour_variables: int,
    max_difference_weight: int,
    sector_batch_size: int = 128,
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
) -> DoubledSumResult:
    """Truncate coupled ``Q(x,y)`` after substituting ``y=x xor d``.

    Variables are ordered as all ``x`` variables followed by all ``y``
    variables. Mixed terms can encode operators, traces, or averaged noise.
    """
    try:
        k = index(contour_variables)
    except TypeError as exc:
        raise TypeError("contour_variables must be an integer.") from exc
    if k < 0 or q_xy.n != 2 * k:
        raise ValueError("q_xy must contain exactly 2 * contour_variables variables.")
    max_weight = _validate_options(k, max_difference_weight, sector_batch_size)
    context = _ReductionContext(
        preserve_scale=True,
        allow_tensor_contraction=allow_tensor_contraction,
        extended_reductions=extended_reductions,
    )
    if max_weight == k:
        rows = _reduce_and_sum_scaled_batch([q_xy], context=context)
        total, info = rows[0]
        backend = info.get("phase3_backend")
        return _result(
            total,
            k,
            max_weight,
            1,
            int(info["remaining"]),
            {str(backend)} if backend else set(),
        )

    row_masks = tuple(1 << idx for idx in range(k)) * 2
    total: ScaledComplex = (0j, 0)
    max_remaining = 0
    backends: set[str] = set()
    sectors_evaluated = 0
    for masks in _chunked_masks(k, max_weight, int(sector_batch_size), include_zero=True):
        rows = _reduce_and_sum_scaled_batch(
            [
                _aff_compose_cached(q_xy, mask << k, row_masks, k, context=context)
                for mask in masks
            ],
            context=context,
        )
        sectors_evaluated += len(rows)
        total, max_remaining = _sum_sector_rows(rows, total, max_remaining, backends)
        _clear_sector_caches(context)
    return _result(total, k, max_weight, sectors_evaluated, max_remaining, backends)


__all__ = ["DoubledSumResult", "sum_coupled_doubled_phase", "sum_doubled_phase"]
