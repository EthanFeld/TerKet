"""Tests for explicit arbitrary-angle snapping to exact dyadic phases."""

from __future__ import annotations

import math

import pytest

from terket import (
    build_state,
    compute_circuit_amplitude,
    make_circuit,
    snap_arbitrary_angles,
)


def test_snap_arbitrary_angles_converts_supported_gate_forms_and_reports_error():
    spec = make_circuit(
        2,
        [
            ("rz_arbitrary", 0, math.pi / 7.0),
            ("pauli_expbox", ("Z", "Z"), (0, 1), 0.71),
        ],
    )

    snapped = snap_arbitrary_angles(spec, max_level=3)

    assert snapped.gates[0][0] == "rz_dyadic"
    assert snapped.gates[1][0] == "pauli_expbox"
    assert snapped.metadata["approximation_mode"] == "snap_dyadic"
    assert snapped.metadata["approximation_phase_count"] == 2
    assert snapped.metadata["approximation_max_angle_error"] > 0.0
    assert snapped.metadata["snap_dyadic_max_level"] == 3


def test_snap_arbitrary_angles_matches_manual_nearest_dyadic_circuit():
    source = make_circuit(
        2,
        [
            ("h", 0),
            ("h", 1),
            ("rz_arbitrary", 0, 0.42),
            ("pauli_expbox", ("Z", "Z"), (0, 1), 0.71),
        ],
    )
    snapped = snap_arbitrary_angles(source, max_level=3)
    manual = make_circuit(
        2,
        [
            ("h", 0),
            ("h", 1),
            ("rz_dyadic", 0, 1, 3),
            ("pauli_expbox", ("Z", "Z"), (0, 1), math.pi / 4.0),
        ],
    )

    actual, _ = compute_circuit_amplitude(snapped, [0, 0], [0, 0], as_complex=True)
    expected, _ = compute_circuit_amplitude(manual, [0, 0], [0, 0], as_complex=True)

    assert abs(actual - expected) < 1e-12


def test_snap_arbitrary_angles_can_reject_large_error():
    spec = make_circuit(1, [("rz_arbitrary", 0, math.pi / 7.0)])

    with pytest.raises(ValueError, match="exceeds max_error"):
        snap_arbitrary_angles(spec, max_level=3, max_error=1e-3)


def test_snap_arbitrary_angles_can_cap_total_error():
    spec = make_circuit(
        2,
        [
            ("rz_arbitrary", 0, math.pi / 7.0),
            ("rz_arbitrary", 1, math.pi / 7.0),
        ],
    )

    with pytest.raises(ValueError, match="Total dyadic snap error"):
        snap_arbitrary_angles(spec, max_level=3, max_total_error=0.1)


def test_exact_dyadic_pauli_expbox_uses_phase_polynomial_automatically():
    spec = make_circuit(
        2,
        [
            ("h", 0),
            ("h", 1),
            ("pauli_expbox", ("Z", "Z"), (0, 1), math.pi / 4.0),
        ],
    )

    state = build_state(spec.n_qubits, spec.gates, [0, 0])

    assert not state._arbitrary_phases
    assert state.q.level >= 3
