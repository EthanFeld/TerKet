"""Tests for approximate q3-free tensor-network backends."""

from fractions import Fraction

import pytest

from terket.cubic_arithmetic import PhaseFunction
from terket.native import _get_quimb_tensor_module
from terket.scaling import _scaled_to_complex
from terket.state import SolverConfig, _reset_solver_config, _set_solver_config
from terket._q3free import treewidth
from terket.q3free import _sum_q3_free_approx_tensor_scaled


pytestmark = pytest.mark.skipif(_get_quimb_tensor_module() is None, reason="quimb not available")


def test_q3_free_approx_tensor_matches_bruteforce_on_small_quadratic_sum():
    q = PhaseFunction(
        5,
        level=4,
        q0=Fraction(3, 16),
        q1=[1, 3, 7, 2, 5],
        q2={(0, 1): 1, (1, 2): 3, (2, 3): 5, (0, 4): 2, (3, 4): 7},
        q3={},
    )
    config = SolverConfig(
        approx_q3_free_tensor=True,
        approx_tensor_method="compressed",
        approx_tensor_max_bond=64,
        approx_tensor_cutoff=1e-12,
    )

    actual = _scaled_to_complex(_sum_q3_free_approx_tensor_scaled(q, config=config))
    expected = treewidth._bruteforce_q3_free_sum(q)

    assert actual == pytest.approx(expected, abs=1e-9, rel=1e-9)


def test_q3_free_bethe_bp_is_exact_on_tree_quadratic_sum():
    q = PhaseFunction(
        6,
        level=3,
        q0=Fraction(1, 8),
        q1=[1, 2, 3, 4, 5, 6],
        q2={(0, 1): 1, (1, 2): 2, (1, 3): 3, (3, 4): 1, (3, 5): 2},
        q3={},
    )
    config = SolverConfig(
        approx_q3_free_tensor=True,
        approx_tensor_method="bp",
        approx_tensor_bp_max_iters=50,
        approx_tensor_bp_tol=1e-12,
    )

    actual = _scaled_to_complex(_sum_q3_free_approx_tensor_scaled(q, config=config))
    expected = treewidth._bruteforce_q3_free_sum(q)

    assert actual == pytest.approx(expected, abs=1e-9, rel=1e-9)


def test_q3_free_residue_sampler_is_exact_for_zero_phase_sum():
    q = PhaseFunction(10, level=5, q1=[0] * 10, q2={}, q3={})
    config = SolverConfig(
        approx_q3_free_tensor=True,
        approx_tensor_method="residue_sample",
        approx_tensor_residue_samples=17,
        approx_tensor_residue_batch=5,
    )

    actual = _scaled_to_complex(_sum_q3_free_approx_tensor_scaled(q, config=config))

    assert actual == pytest.approx(2**10)


def test_q3_free_residue_sampler_stays_within_count_bound():
    q = PhaseFunction(
        8,
        level=5,
        q0=Fraction(1, 32),
        q1=[1, 3, 5, 7, 9, 11, 13, 15],
        q2={(0, 1): 1, (1, 2): 3, (2, 3): 5, (3, 4): 7, (4, 5): 9, (5, 6): 11, (6, 7): 13},
        q3={},
    )
    config = SolverConfig(
        approx_q3_free_tensor=True,
        approx_tensor_method="residue_sample",
        approx_tensor_residue_samples=257,
        approx_tensor_residue_batch=64,
        approx_tensor_residue_seed=123,
    )

    actual = _scaled_to_complex(_sum_q3_free_approx_tensor_scaled(q, config=config))

    assert abs(actual) <= 2**8 + 1e-9


def test_q3_free_residue_forest_is_exact_on_tree_at_full_level():
    q = PhaseFunction(
        7,
        level=4,
        q0=Fraction(1, 16),
        q1=[1, 2, 3, 4, 5, 6, 7],
        q2={(0, 1): 1, (1, 2): 2, (1, 3): 3, (3, 4): 4, (4, 5): 5, (4, 6): 6},
        q3={},
    )
    config = SolverConfig(
        approx_q3_free_tensor=True,
        approx_tensor_method="residue_forest",
        approx_tensor_residue_level=4,
        approx_tensor_reliability_repeats=1,
    )

    actual = _scaled_to_complex(_sum_q3_free_approx_tensor_scaled(q, config=config))
    expected = treewidth._bruteforce_q3_free_sum(q)

    assert actual == pytest.approx(expected, abs=1e-9, rel=1e-9)


def test_q3_free_residue_forest_stays_within_count_bound_on_loopy_graph():
    q = PhaseFunction(
        8,
        level=5,
        q0=Fraction(1, 32),
        q1=[1, 3, 5, 7, 9, 11, 13, 15],
        q2={
            (0, 1): 1,
            (1, 2): 3,
            (2, 3): 5,
            (3, 4): 7,
            (4, 5): 9,
            (5, 6): 11,
            (6, 7): 13,
            (0, 7): 15,
            (1, 6): 2,
        },
        q3={},
    )
    config = SolverConfig(
        approx_q3_free_tensor=True,
        approx_tensor_method="residue_forest",
        approx_tensor_residue_level=5,
        approx_tensor_residue_forest_samples=32,
        approx_tensor_residue_seed=123,
        approx_tensor_reliability_repeats=1,
    )

    actual = _scaled_to_complex(_sum_q3_free_approx_tensor_scaled(q, config=config))

    assert abs(actual) <= 2**8 + 1e-9


@pytest.mark.parametrize("sample_mode", ["antithetic", "balanced", "stratified"])
def test_q3_free_residue_forest_sample_modes_stay_within_count_bound(sample_mode):
    q = PhaseFunction(
        8,
        level=5,
        q0=Fraction(1, 32),
        q1=[1, 3, 5, 7, 9, 11, 13, 15],
        q2={(0, 1): 1, (1, 2): 3, (2, 3): 5, (3, 4): 7, (0, 7): 15},
        q3={},
    )
    config = SolverConfig(
        approx_q3_free_tensor=True,
        approx_tensor_method="residue_forest",
        approx_tensor_residue_level=5,
        approx_tensor_residue_forest_samples=32,
        approx_tensor_residue_seed=123,
        approx_tensor_residue_sample_mode=sample_mode,
        approx_tensor_residue_stratified_vars=2,
        approx_tensor_reliability_repeats=1,
    )

    actual = _scaled_to_complex(_sum_q3_free_approx_tensor_scaled(q, config=config))

    assert abs(actual) <= 2**8 + 1e-9


def test_q3_free_residue_forest_reliability_guard_rejects_unstable_estimate(monkeypatch):
    q = PhaseFunction(
        8,
        level=5,
        q1=[1, 3, 5, 7, 9, 11, 13, 15],
        q2={(0, 1): 1, (2, 3): 1},
        q3={},
    )
    values = iter([(1.0 + 0j, 0), (1.0 + 0j, 40), (1.0 + 0j, -40)])

    monkeypatch.setattr(
        "terket._q3free.approx_guard._sum_q3_free_residue_forest_scaled",
        lambda _q, *, config=None: next(values),
    )
    config = SolverConfig(
        approx_q3_free_tensor=True,
        approx_tensor_method="residue_forest",
        approx_tensor_residue_sample_mode="balanced",
        approx_tensor_reliability_repeats=3,
        approx_tensor_reliability_max_log2_spread=2.0,
        approx_tensor_reliability_reject=True,
        approx_tensor_raise_on_unreliable=False,
    )

    assert _sum_q3_free_approx_tensor_scaled(q, config=config) is None


def test_q3_free_residue_forest_reliability_guard_can_return_mean_when_non_strict(monkeypatch):
    q = PhaseFunction(4, level=3, q1=[0] * 4, q2={}, q3={})
    values = iter([(1.0 + 0j, 0), (1.0 + 0j, 2), (1.0 + 0j, 4)])

    monkeypatch.setattr(
        "terket._q3free.approx_guard._sum_q3_free_residue_forest_scaled",
        lambda _q, *, config=None: next(values),
    )
    config = SolverConfig(
        approx_q3_free_tensor=True,
        approx_tensor_method="residue_forest",
        approx_tensor_residue_sample_mode="balanced",
        approx_tensor_reliability_repeats=3,
        approx_tensor_reliability_max_log2_spread=0.0,
        approx_tensor_reliability_reject=False,
    )

    actual = _scaled_to_complex(_sum_q3_free_approx_tensor_scaled(q, config=config))

    assert actual == pytest.approx((1 + 2 + 4) / 3)


def test_q3_free_component_uses_approx_tensor_before_feedback_guard(monkeypatch):
    n = 30
    q = PhaseFunction(
        n,
        level=3,
        q1=[idx % 8 for idx in range(n)],
        q2={(left, right): 1 for left in range(n) for right in range(left + 1, n)},
        q3={},
    )
    sentinel = (3.0 + 4.0j, 0)

    monkeypatch.setattr(treewidth, "_sum_q3_free_via_neighborhood_composed_scaled", lambda _q: None)
    monkeypatch.setattr(treewidth, "_sum_q3_free_via_gauss_reduction_scaled", lambda _q: None)
    monkeypatch.setattr(treewidth, "_build_block_cut_tree_region_plan", lambda _q: None)
    monkeypatch.setattr(treewidth, "_sum_binary_phase_quadratic_scaled", lambda _q: None)
    monkeypatch.setattr(treewidth, "_q3_free_treewidth_order", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(treewidth, "_q3_free_prefers_locality_preserving_cutset", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(treewidth, "_sum_q3_free_via_cutset_conditioning_scaled", lambda _q: None)
    monkeypatch.setattr(treewidth, "_supports_exact_dense_schur", lambda _q: False)
    monkeypatch.setattr(treewidth, "_sum_q3_free_approx_tensor_scaled", lambda _q: sentinel)

    token = _set_solver_config(SolverConfig(approx_q3_free_tensor=True))
    try:
        assert treewidth._sum_q3_free_component_scaled(q) == sentinel
    finally:
        _reset_solver_config(token)
