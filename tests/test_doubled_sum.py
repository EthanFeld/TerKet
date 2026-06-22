"""Tests for the low-difference-weight doubled-sum backend."""

from __future__ import annotations

import cmath
from fractions import Fraction
import itertools
import math

import pytest

import terket
import terket._doubled_core as doubled_core
from terket._doubled_core import _difference_phase
from terket.doubled import DoubledFactorProblem, sum_coupled_doubled_phase, sum_doubled_factor_problem


def _bits(mask: int, n: int) -> list[int]:
    return [(mask >> idx) & 1 for idx in range(n)]


def test_difference_phase_matches_direct_shift() -> None:
    q = terket.PhaseFunction(
        4,
        level=4,
        q0=Fraction(3, 16),
        q1=[1, 5, 9, 14],
        q2={(0, 1): 3, (1, 3): 6},
        q3={(0, 2, 3): 2, (1, 2, 3): 3},
    )
    for difference_mask in range(1 << q.n):
        difference = _difference_phase(q, difference_mask)
        for x_mask in range(1 << q.n):
            expected = (
                q.evaluate(_bits(x_mask, q.n))
                - q.evaluate(_bits(x_mask ^ difference_mask, q.n))
            ) % 1
            assert difference.evaluate(_bits(x_mask, q.n)) == expected


def test_low_level_full_cutoff_matches_direct_doubled_sum() -> None:
    q = terket.PhaseFunction(
        3,
        q1=[1, 3, 6],
        q2={(0, 1): 1, (1, 2): 2},
        q3={(0, 1, 2): 1},
    )
    direct = sum(
        cmath.exp(2j * math.pi * float(q.evaluate(x) - q.evaluate(y)))
        for x, y in itertools.product(itertools.product((0, 1), repeat=q.n), repeat=2)
    )
    result = terket.sum_doubled_phase(q, max_difference_weight=q.n, sector_batch_size=2)
    assert result.exact
    assert result.sectors_evaluated == 1
    assert abs(result.to_complex() - direct) < 1e-9


def test_level_one_phase_is_supported() -> None:
    q = terket.PhaseFunction(2, level=1, q1=[1, 0])
    result = terket.sum_doubled_phase(q, max_difference_weight=2)
    exact, _ = terket.reduce_and_sum(q)
    assert result.to_complex() == pytest.approx(abs(exact) ** 2)


def test_high_precision_difference_sectors_use_generic_exact_reducer() -> None:
    q = terket.PhaseFunction(
        4,
        level=4,
        q1=[0, 13, 2, 8],
        q2={(0, 2): 7, (0, 3): 3, (1, 2): 1, (1, 3): 3, (2, 3): 7},
        q3={(0, 1, 2): 1, (0, 1, 3): 2, (0, 2, 3): 3, (1, 2, 3): 1},
    )
    direct = sum(
        cmath.exp(2j * math.pi * float(q.evaluate(x) - q.evaluate(y)))
        for x, y in itertools.product(itertools.product((0, 1), repeat=q.n), repeat=2)
        if sum(left ^ right for left, right in zip(x, y)) <= 2
    )
    result = terket.sum_doubled_phase(q, max_difference_weight=2)
    assert result.to_complex() == pytest.approx(direct)


def test_zero_weight_is_diagonal_incoherent_sector() -> None:
    q = terket.PhaseFunction(5, q1=[1, 2, 3, 4, 5])
    result = terket.sum_doubled_phase(q, max_difference_weight=0)
    assert result.to_complex() == 2**q.n
    assert result.sectors_evaluated == 1
    assert not result.exact


def test_sector_specific_caches_are_released_per_chunk(monkeypatch) -> None:
    q = terket.PhaseFunction(4, q1=[1, 2, 3, 4])
    calls = 0
    original = doubled_core._clear_sector_caches

    def counted(context) -> None:
        nonlocal calls
        calls += 1
        original(context)

    monkeypatch.setattr(doubled_core, "_clear_sector_caches", counted)
    terket.sum_doubled_phase(q, max_difference_weight=2, sector_batch_size=3)
    assert calls == 4


def test_coupled_doubled_phase_supports_mixed_contour_terms() -> None:
    q_xy = terket.PhaseFunction(
        4,
        q1=[1, 2, 3, 4],
        q2={(0, 2): 1, (1, 3): 2},
        q3={(0, 1, 3): 1},
    )
    direct = sum(
        cmath.exp(2j * math.pi * float(q_xy.evaluate(x + y)))
        for x, y in itertools.product(itertools.product((0, 1), repeat=2), repeat=2)
    )
    result = sum_coupled_doubled_phase(
        q_xy,
        contour_variables=2,
        max_difference_weight=2,
        sector_batch_size=1,
    )
    assert result.exact
    assert abs(result.to_complex() - direct) < 1e-9
    assert result.sectors_evaluated == 1


def test_general_factors_can_suppress_all_off_diagonal_sectors() -> None:
    k = 4
    equality = [1, 0, 0, 1]
    problem = DoubledFactorProblem(
        contour_variables=k,
        factors={(idx, k + idx): equality for idx in range(k)},
    )
    truncated = sum_doubled_factor_problem(problem, max_difference_weight=0)
    exact = sum_doubled_factor_problem(problem, max_difference_weight=k)
    assert truncated.to_complex() == pytest.approx(2**k)
    assert exact.to_complex() == pytest.approx(truncated.to_complex())
    assert exact.sectors_evaluated == 1


def test_auxiliary_average_can_project_onto_diagonal() -> None:
    # Average_r (-1) ** (r * (x xor y)) = delta[x=y].
    problem = DoubledFactorProblem(
        contour_variables=1,
        auxiliary_variables=1,
        factors={(0, 1, 2): [0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, 0.5]},
    )
    truncated = sum_doubled_factor_problem(problem, max_difference_weight=0)
    exact = sum_doubled_factor_problem(problem, max_difference_weight=1)
    assert truncated.to_complex() == pytest.approx(2)
    assert exact.to_complex() == pytest.approx(2)


def test_general_factor_problem_with_phase_matches_brute_force() -> None:
    phase = terket.PhaseFunction(
        5,
        q1=[1, 2, 3, 4, 5],
        q2={(0, 2): 1, (1, 3): 2, (2, 4): 3},
        q3={(0, 1, 4): 1},
    )
    factor = [1, 0.25j, -0.5, 0.75]
    problem = DoubledFactorProblem(
        contour_variables=2,
        auxiliary_variables=1,
        phase=phase,
        factors={(0, 2): factor, (1, 4): [1, 2, 3, 4]},
        scalar=0.25,
    )
    direct = 0j
    for assignment in itertools.product((0, 1), repeat=5):
        x0, x1, y0, _y1, aux = assignment
        factor_a = factor[x0 | (y0 << 1)]
        factor_b = [1, 2, 3, 4][x1 | (aux << 1)]
        direct += 0.25 * factor_a * factor_b * cmath.exp(
            2j * math.pi * float(phase.evaluate(assignment))
        )
    result = sum_doubled_factor_problem(problem, max_difference_weight=2)
    assert result.exact
    assert result.to_complex() == pytest.approx(direct)


def test_general_factor_problem_validates_factor_shape() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        DoubledFactorProblem(contour_variables=1, factors={(1, 0): [1, 1, 1, 1]})
    with pytest.raises(ValueError, match="length"):
        DoubledFactorProblem(contour_variables=1, factors={(0, 1): [1, 1]})
    with pytest.raises(TypeError, match="integers"):
        DoubledFactorProblem(contour_variables=1.5)
    with pytest.raises(ValueError, match="finite"):
        DoubledFactorProblem(contour_variables=1, factors={(0,): [1, math.inf]})
    with pytest.raises(TypeError, match="max_sectors"):
        sum_doubled_factor_problem(
            DoubledFactorProblem(contour_variables=1),
            max_difference_weight=1,
            max_sectors=1.5,
        )


def test_circuit_full_cutoff_matches_exact_probability() -> None:
    circuit = terket.make_circuit(
        3,
        [("h", 0), ("h", 1), ("t", 0), ("cz", 0, 1), ("h", 0), ("cnot", 1, 2)],
    )
    for output in itertools.product((0, 1), repeat=3):
        amplitude, _ = terket.compute_circuit_amplitude(circuit, [0, 0, 0], output, as_complex=True)
        result = terket.compute_circuit_probability_doubled(
            circuit,
            [0, 0, 0],
            output,
            max_difference_weight=64,
            sector_batch_size=1,
        )
        assert result.exact
        assert abs(result.to_float() - abs(amplitude) ** 2) < 1e-9


def test_circuit_supports_nondyadic_arbitrary_phases() -> None:
    circuit = terket.make_circuit(1, [("h", 0), ("rz_arbitrary", 0, 0.37), ("h", 0)])
    amplitude, _ = terket.compute_circuit_amplitude(circuit, [0], [0], as_complex=True)
    result = terket.compute_circuit_probability_doubled(
        circuit,
        [0],
        [0],
        max_difference_weight=1,
    )
    assert result.exact
    assert result.to_float() == pytest.approx(abs(amplitude) ** 2)


def test_circuit_general_bound_preserves_scaled_certificate() -> None:
    circuit = terket.make_circuit(1, [("h", 0), ("rz_arbitrary", 0, 0.37), ("h", 0)])
    amplitude, _ = terket.compute_circuit_amplitude(circuit, [0], [0], as_complex=True)
    result = terket.compute_circuit_probability_doubled(
        circuit,
        [0],
        [0],
        max_difference_weight=1,
        max_sectors=1,
        difference_strategy="general_bound",
    )
    assert result.omitted_magnitude_bound is not None
    assert result.omitted_magnitude_bound.to_complex().real >= abs(
        result.to_float() - abs(amplitude) ** 2
    )


def test_observable_aware_doubled_probability_matches_exact_expectation_square() -> None:
    circuit = terket.make_circuit(
        2,
        [("h", 0), ("rz_arbitrary", 0, 0.37), ("cnot", 0, 1), ("h", 1)],
    )
    exact = terket.compute_circuit_pauli_expectations(
        circuit,
        ["XI", "ZZ"],
        input_bits=[0, 0],
        as_complex=True,
    )
    doubled = terket.compute_circuit_pauli_expectation_probabilities_doubled(
        circuit,
        ["XI", "ZZ"],
        input_bits=[0, 0],
        max_difference_weight=64,
    )
    for (value, _info), result in zip(exact, doubled):
        assert result.exact
        assert result.to_float() == pytest.approx(abs(value) ** 2)


def test_circuit_accepts_arbitrary_phase_when_no_path_variable_remains() -> None:
    circuit = terket.make_circuit(1, [("rz_arbitrary", 0, 0.37)])
    result = terket.compute_circuit_probability_doubled(
        circuit,
        [0],
        [0],
        max_difference_weight=0,
    )
    assert result.exact
    assert result.to_float() == pytest.approx(1.0)
