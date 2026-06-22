"""Tests for nondegenerate q3-free feedback batches."""

import cmath
import itertools

import numpy as np
import pytest

from terket.cubic_arithmetic import PhaseFunction
from terket.native import _schur_native
from terket.scaling import _scaled_to_complex
from terket.state import SolverConfig
from terket._q3free.approx_residue import _sum_q3_free_residue_forest_scaled
from terket._q3free.approx_sampling import _feedback_bond_order, _feedback_sample_rows
from terket._q3free.approx_residue_native import (
    _forest_parent_data,
    _stable_unique_q1_batch,
    _sum_q3_free_residue_forest_native_batch_scaled,
)
from terket._q3free.exact import _bruteforce_q3_free_sum
from terket._q3free.approx_guard import (
    _combine_residue_channels,
    _get_q3_free_approx_diagnostics,
    _residue_channel_modes,
    _sum_q3_free_residue_forest_checked_scaled,
)
from terket._q3free.approx_mps import _sum_q3_free_boundary_mps_scaled
from terket._q3free.approx_mps_guard import _sum_q3_free_boundary_mps_checked_scaled


def test_unified_path_samples_are_distinct_and_reproducible():
    first = _feedback_sample_rows(
        14,
        8192,
        np.random.default_rng(123),
        mode="unified",
        stratified_vars=0,
    )
    second = _feedback_sample_rows(
        14,
        8192,
        np.random.default_rng(123),
        mode="unified",
        stratified_vars=0,
    )

    assert np.array_equal(first, second)
    assert len(np.unique(first, axis=0)) == 8192
    assert np.all(first.min(axis=0) == 0)
    assert np.all(first.max(axis=0) == 1)


def test_unified_path_exactly_retains_priority_bonds():
    priority = np.asarray([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int64)

    rows = _feedback_sample_rows(
        8,
        8,
        np.random.default_rng(9),
        mode="unified",
        stratified_vars=0,
        priority_columns=priority,
    )

    assert len(np.unique(rows[:, priority[:3]], axis=0)) == 8


def test_feedback_bond_order_retains_strong_tensor_bonds_first():
    order = _feedback_bond_order(
        3,
        16,
        [(0, 1), (1, 8), (2, 2)],
        [(2, 0, 8)],
        [],
    )

    assert order.tolist() == [2, 1, 0]


def test_multichannel_combine_prefers_lower_variance_channel():
    mean, error_log2_abs, _estimates = _combine_residue_channels(
        [
            [(1.0 + 0j, 0), (3.0 + 0j, 0)],
            [(0.0 + 0j, 0), (4.0 + 0j, 0)],
        ]
    )

    assert _scaled_to_complex(mean) == pytest.approx(2.0 + 0j)
    assert 2**error_log2_abs == pytest.approx(1.0 / np.sqrt(1.25))


def test_default_unified_guard_uses_ranked_and_random_channels():
    assert _residue_channel_modes(SolverConfig(), 3) == ("unified", "unified_random")
    assert _residue_channel_modes(SolverConfig(), 1) == ("unified",)


def test_unified_path_sampling_is_default():
    assert SolverConfig().approx_tensor_residue_sample_mode == "unified"


@pytest.mark.parametrize(
    "q2",
    [
        {(0, 1): 1, (1, 2): 2, (2, 3): 3, (3, 4): 4},
        {(0, 1): 1, (1, 2): 2, (2, 3): 3, (3, 4): 4, (0, 4): 5},
        {(0, 3): 1, (0, 4): 2, (1, 3): 3, (1, 4): 4, (2, 3): 5, (2, 4): 6},
    ],
)
def test_boundary_mps_matches_exact_without_truncation(q2):
    q = PhaseFunction(5, level=4, q1=[1, 3, 5, 7, 9], q2=q2, q3={})
    result = _sum_q3_free_boundary_mps_scaled(q, max_bond=64, cutoff=0.0)
    assert result is not None
    actual, diagnostics = result
    assert _scaled_to_complex(actual) == pytest.approx(
        _bruteforce_q3_free_sum(q), rel=1e-9, abs=1e-9
    )
    assert diagnostics["max_discarded"] == 0.0


def test_boundary_mps_guard_accepts_bond_convergence_on_tree():
    q = PhaseFunction(
        5,
        level=4,
        q1=[1, 3, 5, 7, 9],
        q2={(0, 1): 1, (1, 2): 2, (2, 3): 3, (3, 4): 4},
        q3={},
    )
    config = SolverConfig(
        approx_tensor_max_bond=8,
        approx_tensor_mps_max_bond=8,
        approx_tensor_raise_on_unreliable=False,
    )
    actual = _sum_q3_free_boundary_mps_checked_scaled(q, config=config)
    assert actual is not None
    assert _scaled_to_complex(actual) == pytest.approx(_bruteforce_q3_free_sum(q))


def test_residue_guard_falls_back_to_boundary_mps():
    q = PhaseFunction(
        6,
        level=4,
        q1=[1, 3, 5, 7, 9, 11],
        q2={(left, right): left + right + 1 for left in range(6) for right in range(left + 1, 6)},
        q3={},
    )
    config = SolverConfig(
        approx_tensor_residue_forest_samples=1,
        approx_tensor_reliability_repeats=2,
        approx_tensor_reliability_max_log2_spread=0.0,
        approx_tensor_reliability_max_rel_stderr=0.0,
        approx_tensor_reliability_reject=False,
        approx_tensor_raise_on_unreliable=False,
        approx_tensor_mps_fallback=True,
        approx_tensor_mps_max_bond=4,
    )
    assert _sum_q3_free_residue_forest_checked_scaled(q, config=config) is not None
    diagnostics = _get_q3_free_approx_diagnostics()
    assert diagnostics is not None
    assert diagnostics["approx_q3_free_method"] == "boundary_mps"


def test_residue_forest_native_bridge_matches_exact_enumeration():
    q = PhaseFunction(
        6,
        level=4,
        q1=[1, 3, 5, 7, 9, 11],
        q2={(0, 1): 1, (1, 2): 2, (2, 3): 3, (3, 4): 4, (4, 5): 5, (0, 5): 6},
        q3={},
    )
    config = SolverConfig(
        approx_q3_free_tensor=True,
        approx_tensor_method="residue_forest",
        approx_tensor_residue_level=4,
        approx_tensor_residue_forest_samples=8,
        approx_tensor_reliability_repeats=1,
    )

    actual = _scaled_to_complex(_sum_q3_free_residue_forest_scaled(q, config=config))

    assert actual == pytest.approx(_bruteforce_q3_free_sum(q))


def test_native_arbitrary_residue_forest_matches_bruteforce():
    adjacency = [{1: 1}, {0: 1, 2: 3}, {1: 3, 3: 5}, {2: 5}]
    q1_batch = np.asarray([[0, 1, 2, 3], [3, 2, 1, 0], [7, 5, 3, 1]], dtype=np.int64)
    parent, parent_phase, postorder = _forest_parent_data(adjacency)

    rows = _schur_native.sum_residue_forest_batch_scaled_array(
        3,
        q1_batch,
        parent,
        parent_phase,
        postorder,
    )

    for q1, row in zip(q1_batch, rows):
        expected = 0j
        for bits in itertools.product((0, 1), repeat=4):
            residue = sum(int(q1[idx]) * bits[idx] for idx in range(4))
            residue += bits[0] * bits[1] + 3 * bits[1] * bits[2] + 5 * bits[2] * bits[3]
            expected += cmath.exp(2j * cmath.pi * residue / 8)
        actual = _scaled_to_complex((complex(row[0]), int(row[1])))
        assert actual == pytest.approx(expected)


def test_native_residue_batch_evaluates_each_induced_q1_once(monkeypatch):
    captured = {}

    def fake_batch(q1_batch, _adjacency, *, level):
        captured["q1_batch"] = q1_batch.copy()
        return [(1.0 + 0j, 0)] * len(q1_batch)

    monkeypatch.setattr(
        "terket._q3free.approx_residue_native._sum_arbitrary_residue_forest_native_batch",
        fake_batch,
    )
    q = PhaseFunction(2, level=3, q1=[0, 0], q2={}, q3={})
    total = _sum_q3_free_residue_forest_native_batch_scaled(
        q,
        target_level=3,
        feedback_count=1,
        fixed_bit_rows=np.asarray([[0], [0], [0], [1]], dtype=np.uint8),
        base_q1=[0],
        free_adjacency=[{}],
        fixed_linear=[(0, 4)],
        fixed_to_free=[],
        fixed_to_fixed=[],
    )

    assert total is not None
    assert captured["q1_batch"].tolist() == [[0]]
    assert _scaled_to_complex(total) == pytest.approx(1.0 + 0j)


def test_q1_dedup_preserves_first_path_order():
    rows = np.asarray([[3, 1], [2, 1], [3, 1], [1, 1], [2, 1]], dtype=np.int64)

    unique, inverse = _stable_unique_q1_batch(rows, level=8)

    assert unique.tolist() == [[3, 1], [2, 1], [1, 1]]
    assert inverse is not None
    assert unique[inverse].tolist() == rows.tolist()
