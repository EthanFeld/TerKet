"""Tests for public API and compatibility facade import surfaces."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import terket
import terket.phase3 as phase3
import terket.schur_engine as schur_engine


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def test_public_import_surface_stays_small() -> None:
    expected = {
        "CircuitSpec",
        "CubicFunction",
        "PhaseFunction",
        "ScaledAmplitude",
        "SchurState",
        "SolverConfig",
        "analyze_amplitudes",
        "analyze_circuit",
        "bits_to_big_endian_string",
        "bits_to_index",
        "bits_to_little_endian_string",
        "build_state",
        "cache_stats",
        "clear_caches",
        "compute_amplitude",
        "compute_amplitudes",
        "compute_amplitude_scaled",
        "compute_circuit_amplitude",
        "compute_circuit_amplitude_scaled",
        "compute_circuit_pauli_expectations",
        "compute_circuit_pauli_expectations_approx",
        "from_qiskit",
        "lift_exact_dyadic_precision",
        "make_circuit",
        "normalize_circuit",
        "reduce_and_sum",
    }
    assert set(terket.__all__) == expected


def test_top_level_import_does_not_load_benchmarking_or_qiskit() -> None:
    code = (
        "import sys, terket; "
        "print('qiskit' in sys.modules); "
        "print('terket.benchmarking' in sys.modules); "
        "print('terket.interop' in sys.modules)"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.stdout.splitlines() == ["False", "False", "False"]


def test_schur_engine_facade_keeps_core_public_functions() -> None:
    assert schur_engine.compute_circuit_amplitude is terket.compute_circuit_amplitude
    assert schur_engine.compute_circuit_amplitude_scaled is terket.compute_circuit_amplitude_scaled
    assert schur_engine.build_state is terket.build_state


def test_phase3_facade_excludes_removed_tensor_stubs() -> None:
    assert "_sum_via_tensor_contraction" not in phase3.__all__
    assert not hasattr(phase3, "_sum_via_tensor_contraction")
    assert not hasattr(phase3, "_build_reduced_tensor_network")
    assert not hasattr(phase3, "_contract_reduced_network")


def test_internal_backend_packages_import() -> None:
    import terket._phase3.select as phase3_select
    import terket._q3free.plans as q3free_plans

    assert callable(phase3_select._phase3_plan)
    assert callable(q3free_plans._build_q3_free_constraint_plan)
