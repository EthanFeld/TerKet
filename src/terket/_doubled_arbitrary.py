"""Compact doubled sectors for arbitrary affine-parity phases."""

from __future__ import annotations

import cmath
from itertools import islice
from operator import index
from typing import Literal, Sequence

from ._arbitrary_runtime import solve_arbitrary_exact
from ._doubled_core import DoubledSumResult, _chunked_masks, _difference_phase, _validate_options
from ._engine_runtime_state import _aff_compose_cached
from ._reduction_support import _ReductionContext
from ._state_runtime import _ArbitraryPhaseTerm
from .scaling import (
    ScaledAmplitude,
    ScaledComplex,
    _add_scaled_complex,
    _make_scaled_complex,
    _mul_scaled_complex,
    _normalize_scaled_complex,
    _scale_scaled_complex,
)


def _difference_arbitrary_terms(
    terms: Sequence[_ArbitraryPhaseTerm],
    difference_mask: int,
) -> tuple[complex, tuple[_ArbitraryPhaseTerm, ...]]:
    scalar = 1.0 + 0j
    active: list[_ArbitraryPhaseTerm] = []
    for term in terms:
        if (int(term.row_mask) & difference_mask).bit_count() & 1:
            angle = float(term.angle)
            scalar *= cmath.exp(-1j * angle)
            active.append(_ArbitraryPhaseTerm(int(term.row_mask), int(term.offset) & 1, 2.0 * angle))
    return scalar, tuple(active)


def _flat_omitted_bound(path_variables: int, sectors_evaluated: int) -> ScaledAmplitude:
    omitted = (1 << path_variables) - sectors_evaluated
    if omitted <= 0:
        return ScaledAmplitude(0j)
    shift = max(0, omitted.bit_length() - 53)
    rounded = (omitted + (1 << shift) - 1) >> shift
    return ScaledAmplitude.from_tuple(
        _scale_scaled_complex(
            _make_scaled_complex(float(rounded)),
            2 * (path_variables + shift),
        )
    )


def _low_incidence_coordinate_basis(q, terms: Sequence[_ArbitraryPhaseTerm], dimension: int) -> tuple[int, ...]:
    incidence = [0] * q.n
    for term in terms:
        row_mask = int(term.row_mask)
        for var in range(q.n):
            incidence[var] += (row_mask >> var) & 1
    for scope in (*q.q2, *q.q3):
        for var in scope:
            incidence[var] += 1
    order = sorted(range(q.n), key=lambda var: (incidence[var], var))
    return tuple(1 << var for var in order[:dimension])


def _sum_arbitrary_subspace(q, terms: Sequence[_ArbitraryPhaseTerm], basis: Sequence[int]):
    n = q.n
    dimension = len(basis)
    target_n = n + dimension
    x_rows = tuple(1 << var for var in range(n))
    y_rows = tuple(
        (1 << var)
        | sum(
            (((int(direction) >> var) & 1) << (n + basis_idx))
            for basis_idx, direction in enumerate(basis)
        )
        for var in range(n)
    )
    context = _ReductionContext(preserve_scale=True)
    grouped_phase = (
        _aff_compose_cached(q, 0, x_rows, target_n, context=context)
        - _aff_compose_cached(q, 0, y_rows, target_n, context=context)
    )
    grouped_terms: list[_ArbitraryPhaseTerm] = []
    for term in terms:
        row_mask = int(term.row_mask)
        grouped_terms.append(term)
        shift_mask = sum(
            (((row_mask & int(direction)).bit_count() & 1) << (n + basis_idx))
            for basis_idx, direction in enumerate(basis)
        )
        grouped_terms.append(
            _ArbitraryPhaseTerm(
                row_mask ^ shift_mask,
                int(term.offset) & 1,
                -float(term.angle),
            )
        )
    return solve_arbitrary_exact(grouped_phase, grouped_terms)


def _subspace_result(q, terms: Sequence[_ArbitraryPhaseTerm], sector_limit: int) -> DoubledSumResult:
    dimension = sector_limit.bit_length() - 1
    if dimension > q.n:
        raise ValueError("subspace max_sectors exceeds full difference space.")
    if dimension == q.n:
        amplitude, remaining, backend, _metadata = solve_arbitrary_exact(q, terms)
        total = _normalize_scaled_complex(abs(amplitude[0]) ** 2, 2 * amplitude[1])
        return DoubledSumResult(
            estimate=ScaledAmplitude.from_tuple(total),
            path_variables=q.n,
            max_difference_weight=q.n,
            sectors_evaluated=1 << q.n,
            sectors_total=1 << q.n,
            exact=True,
            max_reducer_remaining=int(remaining),
            phase3_backends=(str(backend),),
            omitted_magnitude_bound=ScaledAmplitude(0j),
        )
    basis = _low_incidence_coordinate_basis(q, terms, dimension)
    total, remaining, backend, _metadata = _sum_arbitrary_subspace(q, terms, basis)
    return DoubledSumResult(
        estimate=ScaledAmplitude.from_tuple(total),
        path_variables=q.n,
        max_difference_weight=max((mask.bit_count() for mask in basis), default=0),
        sectors_evaluated=sector_limit,
        sectors_total=1 << q.n,
        exact=False,
        max_reducer_remaining=int(remaining),
        phase3_backends=(str(backend),),
        omitted_magnitude_bound=_flat_omitted_bound(q.n, sector_limit),
    )


def sum_doubled_arbitrary_phase(
    q,
    terms: Sequence[_ArbitraryPhaseTerm],
    *,
    max_difference_weight: int,
    sector_batch_size: int = 128,
    max_sectors: int | None = None,
    difference_strategy: Literal["hamming", "general_bound", "subspace"] = "hamming",
) -> DoubledSumResult:
    """Sum retained doubled sectors without dense ket/bra arbitrary-factor tables."""
    max_weight = _validate_options(q.n, max_difference_weight, sector_batch_size)
    if difference_strategy not in {"hamming", "general_bound", "subspace"}:
        raise ValueError("Compact arbitrary doubled sums support hamming, general_bound, or subspace.")
    if max_sectors is not None:
        try:
            sector_limit = index(max_sectors)
        except TypeError as exc:
            raise TypeError("max_sectors must be an integer.") from exc
        if sector_limit <= 0:
            raise ValueError("max_sectors must be positive.")
    else:
        sector_limit = None
    if difference_strategy == "general_bound" and (max_weight != q.n or sector_limit is None):
        raise ValueError("general_bound requires full difference range and max_sectors.")
    if difference_strategy == "subspace":
        if sector_limit is None or sector_limit & (sector_limit - 1):
            raise ValueError("subspace requires max_sectors to be a power of two.")
        return _subspace_result(q, terms, sector_limit)
    if sector_limit is None and max_weight > 0 and q.n > 512:
        raise RuntimeError(
            "Compact arbitrary doubled sum has flat magnitude bounds and an unbounded "
            f"Hamming shell over {q.n} path variables. Set max_sectors explicitly."
        )
    if max_weight == q.n and sector_limit is None:
        amplitude, remaining, backend, _metadata = solve_arbitrary_exact(q, terms)
        total = _normalize_scaled_complex(abs(amplitude[0]) ** 2, 2 * amplitude[1])
        return DoubledSumResult(
            estimate=ScaledAmplitude.from_tuple(total),
            path_variables=q.n,
            max_difference_weight=max_weight,
            sectors_evaluated=1,
            sectors_total=1 << q.n,
            exact=True,
            max_reducer_remaining=int(remaining),
            phase3_backends=(str(backend),),
        )

    masks = (
        mask
        for chunk in _chunked_masks(q.n, max_weight, index(sector_batch_size), include_zero=True)
        for mask in chunk
    )
    if sector_limit is not None:
        masks = islice(masks, sector_limit)

    total: ScaledComplex = (0j, 0)
    max_remaining = 0
    backends: set[str] = set()
    sectors_evaluated = 0
    for mask in masks:
        if mask == 0:
            sector_total = (1.0 + 0j, 2 * q.n)
        else:
            scalar, active_terms = _difference_arbitrary_terms(terms, mask)
            sector_total, remaining, backend, _metadata = solve_arbitrary_exact(
                _difference_phase(q, mask),
                active_terms,
            )
            sector_total = _mul_scaled_complex(sector_total, _make_scaled_complex(scalar))
            max_remaining = max(max_remaining, int(remaining))
            backends.add(str(backend))
        total = _add_scaled_complex(total, sector_total)
        sectors_evaluated += 1

    exact = max_weight == q.n and (sector_limit is None or sector_limit >= 1 << q.n)
    omitted_bound = (
        _flat_omitted_bound(q.n, sectors_evaluated)
        if difference_strategy == "general_bound"
        else None
    )
    return DoubledSumResult(
        estimate=ScaledAmplitude.from_tuple(total),
        path_variables=q.n,
        max_difference_weight=max_weight,
        sectors_evaluated=sectors_evaluated,
        sectors_total=1 << q.n,
        exact=exact,
        max_reducer_remaining=max_remaining,
        phase3_backends=tuple(sorted(backends)),
        omitted_magnitude_bound=omitted_bound,
    )


__all__ = ["sum_doubled_arbitrary_phase"]
