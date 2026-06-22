"""Tests for doubled-sum sector selection and execution strategy."""

from __future__ import annotations

import pytest

import terket
from terket.doubled import DoubledFactorProblem, sum_coupled_doubled_phase, sum_doubled_factor_problem


def test_factor_bound_strategy_prioritizes_strong_couplings() -> None:
    damping = [0.01, 0.01, 0.9, 0.8]
    problem = DoubledFactorProblem(
        contour_variables=4,
        factors={
            (idx, 4 + idx): [1, strength, strength, 1]
            for idx, strength in enumerate(damping)
        },
    )
    exact = sum_doubled_factor_problem(problem, max_difference_weight=4)
    hamming = sum_doubled_factor_problem(problem, max_difference_weight=4, max_sectors=3)
    ranked = sum_doubled_factor_problem(
        problem,
        max_difference_weight=4,
        max_sectors=3,
        difference_strategy="factor_bound",
    )
    assert abs(ranked.to_complex() - exact.to_complex()) < abs(
        hamming.to_complex() - exact.to_complex()
    )


def test_factor_bound_strategy_certifies_pair_hard_constraints() -> None:
    problem = DoubledFactorProblem(
        contour_variables=8,
        factors={(idx, 8 + idx): [1, 0, 0, 1] for idx in range(8)},
    )
    result = sum_doubled_factor_problem(
        problem,
        max_difference_weight=8,
        max_sectors=1,
        difference_strategy="factor_bound",
    )
    assert result.exact
    assert result.sectors_evaluated == 1
    assert result.to_complex() == pytest.approx(2**8)


def test_all_zero_restricted_factor_skips_elimination() -> None:
    problem = DoubledFactorProblem(contour_variables=2, factors={(0,): [0, 0]})
    result = sum_doubled_factor_problem(problem, max_difference_weight=2)
    assert result.exact
    assert result.sectors_evaluated == 0
    assert result.to_complex() == 0j


def test_factor_bound_requires_budget_when_bounds_are_informative() -> None:
    problem = DoubledFactorProblem(contour_variables=1, factors={(0, 1): [1, 0.2, 0.2, 1]})
    with pytest.raises(ValueError, match="require max_sectors"):
        sum_doubled_factor_problem(
            problem,
            max_difference_weight=1,
            difference_strategy="factor_bound",
        )


def test_factor_bound_rejects_hamming_cap_and_uninformative_factors() -> None:
    informative = DoubledFactorProblem(
        contour_variables=2,
        factors={(0, 2): [1, 0.2, 0.2, 1]},
    )
    with pytest.raises(ValueError, match="full difference-weight range"):
        sum_doubled_factor_problem(
            informative,
            max_difference_weight=1,
            max_sectors=2,
            difference_strategy="factor_bound",
        )


def test_general_bound_ranks_correlated_difference_factor() -> None:
    # This factor rewards d0 == d1, which independent pair bounds cannot see.
    table = []
    for assignment in range(16):
        x0, x1, y0, y1 = [(assignment >> idx) & 1 for idx in range(4)]
        table.append(1 if (x0 ^ y0) == (x1 ^ y1) else 0.01)
    problem = DoubledFactorProblem(contour_variables=2, factors={(0, 1, 2, 3): table})
    result = sum_doubled_factor_problem(
        problem,
        max_difference_weight=2,
        max_sectors=2,
        difference_strategy="general_bound",
    )
    exact = sum_doubled_factor_problem(problem, max_difference_weight=2)
    assert result.to_complex() == pytest.approx(8)
    assert exact.to_complex() == pytest.approx(8.08)
    assert result.omitted_magnitude_bound is not None
    assert result.omitted_magnitude_bound.to_complex().real >= abs(
        exact.to_complex() - result.to_complex()
    )


def test_general_bound_handles_flat_bounds_without_breadth_first_explosion() -> None:
    problem = DoubledFactorProblem(contour_variables=40)
    result = sum_doubled_factor_problem(
        problem,
        max_difference_weight=40,
        max_sectors=2,
        difference_strategy="general_bound",
    )
    assert result.sectors_evaluated == 2


def test_general_bound_collapses_constant_scoped_bounds() -> None:
    from terket._doubled_bound_graph import compile_difference_bound_graph

    problem = DoubledFactorProblem(
        contour_variables=2,
        factors={(0, 1): [2, -2, 2j, -2j]},
    )
    scalar, factors = compile_difference_bound_graph(problem)
    assert scalar == 2
    assert factors == ()


def test_general_bound_can_stop_at_rigorous_omitted_tolerance() -> None:
    problem = DoubledFactorProblem(
        contour_variables=4,
        factors={(idx, 4 + idx): [1, 0.001, 0.001, 1] for idx in range(4)},
    )
    result = sum_doubled_factor_problem(
        problem,
        max_difference_weight=4,
        max_sectors=16,
        difference_strategy="general_bound",
        omitted_magnitude_tolerance=0.2,
    )
    assert result.sectors_evaluated < 16
    assert not result.exact
    assert result.omitted_magnitude_bound is not None
    assert result.omitted_magnitude_bound.to_complex().real <= 0.2
    with pytest.raises(ValueError, match="informative"):
        sum_doubled_factor_problem(
            DoubledFactorProblem(contour_variables=2),
            max_difference_weight=2,
            max_sectors=2,
            difference_strategy="factor_bound",
        )


def test_doubled_options_require_integer_counts() -> None:
    q = terket.PhaseFunction(2)
    with pytest.raises(TypeError, match="integers"):
        terket.sum_doubled_phase(q, max_difference_weight=1.5)
    with pytest.raises(TypeError, match="integer"):
        sum_coupled_doubled_phase(q, contour_variables=1.5, max_difference_weight=1)


def test_exact_general_factor_rejects_impossible_sector_fallback(monkeypatch) -> None:
    import terket._doubled_direct as doubled_direct

    monkeypatch.setattr(doubled_direct, "_DIRECT_EXACT_MAX_WORK", 0)
    problem = DoubledFactorProblem(contour_variables=21, factors={(0, 21): [1, 1, 1, 1]})
    with pytest.raises(RuntimeError, match="sector enumeration"):
        sum_doubled_factor_problem(problem, max_difference_weight=21)


def test_general_bound_rejects_oversized_partition_plan(monkeypatch) -> None:
    import terket._doubled_bound_graph as bound_graph

    monkeypatch.setattr(bound_graph, "_MAX_BOUND_PARTITION_WORK", 0)
    problem = DoubledFactorProblem(
        contour_variables=2,
        factors={(0, 1, 2, 3): [1] * 16},
    )
    with pytest.raises(RuntimeError, match="partition sum exceeds"):
        sum_doubled_factor_problem(
            problem,
            max_difference_weight=2,
            max_sectors=1,
            difference_strategy="general_bound",
        )


def test_compact_arbitrary_flat_bound_handles_huge_path_count() -> None:
    from terket._doubled_arbitrary import _flat_omitted_bound

    bound = _flat_omitted_bound(20_000, 1)
    assert bound.half_pow2_exp >= 40_000


def test_compact_arbitrary_requires_budget_for_huge_shell() -> None:
    from terket._doubled_arbitrary import sum_doubled_arbitrary_phase

    q = terket.PhaseFunction(513)
    with pytest.raises(RuntimeError, match="Set max_sectors explicitly"):
        sum_doubled_arbitrary_phase(q, (), max_difference_weight=1)
    result = sum_doubled_arbitrary_phase(q, (), max_difference_weight=1, max_sectors=2)
    assert result.sectors_evaluated == 2


def test_compact_arbitrary_full_cutoff_uses_direct_exact_route() -> None:
    from terket._doubled_arbitrary import sum_doubled_arbitrary_phase
    from terket._state_runtime import _ArbitraryPhaseTerm

    q = terket.PhaseFunction(8, q1=[1] * 8)
    result = sum_doubled_arbitrary_phase(
        q,
        (_ArbitraryPhaseTerm((1 << 8) - 1, 0, 0.37),),
        max_difference_weight=8,
    )
    assert result.exact
    assert result.sectors_evaluated == 1


def test_compact_arbitrary_subspace_matches_explicit_sector_sum() -> None:
    from terket._arbitrary_runtime import solve_arbitrary_exact
    from terket._doubled_arbitrary import (
        _difference_arbitrary_terms,
        _low_incidence_coordinate_basis,
        _sum_arbitrary_subspace,
    )
    from terket._doubled_core import _difference_phase
    from terket._state_runtime import _ArbitraryPhaseTerm
    from terket.scaling import _add_scaled_complex, _make_scaled_complex, _mul_scaled_complex

    q = terket.PhaseFunction(4, q1=[1, 2, 3, 4], q2={(0, 2): 1})
    terms = (_ArbitraryPhaseTerm(0b1011, 0, 0.37),)
    basis = _low_incidence_coordinate_basis(q, terms, 2)
    grouped, _remaining, _backend, _metadata = _sum_arbitrary_subspace(q, terms, basis)
    explicit = (0j, 0)
    for coefficients in range(4):
        mask = 0
        for idx, direction in enumerate(basis):
            if (coefficients >> idx) & 1:
                mask ^= direction
        scalar, active_terms = _difference_arbitrary_terms(terms, mask)
        sector, _remaining, _backend, _metadata = solve_arbitrary_exact(
            _difference_phase(q, mask),
            active_terms,
        )
        explicit = _add_scaled_complex(
            explicit,
            _mul_scaled_complex(sector, _make_scaled_complex(scalar)),
        )
    assert terket.ScaledAmplitude.from_tuple(grouped).to_complex() == pytest.approx(
        terket.ScaledAmplitude.from_tuple(explicit).to_complex()
    )
