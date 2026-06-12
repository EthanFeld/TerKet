"""Repo guard tests for bloat, readability, and private import surfaces."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "report_file_sizes.py"
READABILITY_TOOL_PATH = REPO_ROOT / "tools" / "report_readability.py"

ALLOWED_ENGINE_IMPL_IMPORTERS = {
    "src/terket/__init__.py",
    "src/terket/engine.py",
    "src/terket/schur_engine.py",
}

ALLOWED_CIRCUIT_SPEC_IMPORTERS = {
    "src/terket/circuit_io.py",
    "src/terket/interop/angles.py",
    "src/terket/interop/qasm2.py",
    "src/terket/interop/qasm3.py",
    "src/terket/interop/qiskit_export.py",
    "src/terket/interop/qiskit_import.py",
    "src/terket/interop/rewrite.py",
    "src/terket/interop/ross_selinger.py",
    "src/terket/spec.py",
    "tools/quantinuum_public_qec_probe.py",
    "tools/try_cirq_to_qasm.py",
}

ALLOWED_ENGINE_IMPL_DYNAMIC_IMPORTERS = {
    "src/terket/_amplitude_api.py",
    "src/terket/_arbitrary_runtime.py",
    "src/terket/_pauli_api.py",
    "src/terket/_pauli_approx_runtime.py",
    "src/terket/_phase3/order.py",
    "src/terket/_phase3/structure.py",
    "src/terket/_q3free/execution.py",
    "src/terket/_q3free/treewidth.py",
    "src/terket/_reduction_classify.py",
    "src/terket/_reduction_elim.py",
    "src/terket/_reduction_runtime.py",
    "src/terket/_reduction_support.py",
    "src/terket/_state_direct.py",
    "src/terket/_state_runtime.py",
}

ALLOWED_OVER_REPORT_PATHS = {
    "benchmarks/curated_benchmark.py",
    "benchmarks/targeted/mqt/mqt_full_sweep.py",
    "benchmarks/targeted/mqt/mqt_head_to_head_core.py",
    "benchmarks/targeted/rcs/amplitude_post_elimination_tensor_rcs.py",
    "benchmarks/targeted/rcs/rcs_import_strategy_probe.py",
    "src/terket/_amplitude_api.py",
    "src/terket/_arbitrary_bp.py",
    "src/terket/_arbitrary_clusters.py",
    "src/terket/_arbitrary_factors.py",
    "src/terket/_arbitrary_runtime.py",
    "src/terket/_engine_runtime_core.py",
    "src/terket/_engine_runtime_state.py",
    "src/terket/_factor_tables.py",
    "src/terket/_pauli_api.py",
    "src/terket/_pauli_approx_runtime.py",
    "src/terket/_pauli_support.py",
    "src/terket/_phase3/cover.py",
    "src/terket/_phase3/exec.py",
    "src/terket/_phase3/factors.py",
    "src/terket/_phase3/order.py",
    "src/terket/_phase3/select.py",
    "src/terket/_phase3/structure.py",
    "src/terket/_q3free/clusters.py",
    "src/terket/_q3free/batch.py",
    "src/terket/_q3free/components.py",
    "src/terket/_q3free/cutset_search.py",
    "src/terket/_q3free/cutset_search_core.py",
    "src/terket/_q3free/cutset_residue.py",
    "src/terket/_q3free/cutset_support.py",
    "src/terket/_q3free/exact.py",
    "src/terket/_q3free/execution.py",
    "src/terket/_q3free/fallbacks.py",
    "src/terket/_q3free/plans.py",
    "src/terket/_q3free/primitives.py",
    "src/terket/_q3free/raw_constraints.py",
    "src/terket/_q3free/treewidth.py",
    "src/terket/_reduction_classify.py",
    "src/terket/_reduction_elim.py",
    "src/terket/_reduction_runtime.py",
    "src/terket/_reduction_support.py",
    "src/terket/_schur_native_graph.c",
    "src/terket/_schur_native_support.c",
    "src/terket/_state_direct.py",
    "src/terket/_state_runtime.py",
    "src/terket/benchmarking/head_to_head_cases.py",
    "src/terket/cubic_arithmetic.py",
    "src/terket/interop/angles.py",
    "src/terket/interop/qasm2.py",
    "src/terket/interop/qiskit_import.py",
    "src/terket/interop/rewrite.py",
    "src/terket/interop/ross_selinger.py",
    "src/terket/native_classification.c",
    "src/terket/native_constraint_elim.c",
    "src/terket/scaling.py",
    "tests/test_phase3_treewidth_batch.py",
    "tests/test_phase_structure_optimizer_phase3.py",
    "tests/test_q3free_one_shot_slicing_heuristics.py",
    "tests/test_q3free_one_shot_slicing_regions.py",
    "tests/test_q3free_one_shot_slicing_runtime.py",
    "tests/test_q3free_one_shot_slicing_search.py",
    "tests/test_q3free_treewidth_native_runtime.py",
    "tests/test_rz_native_arbitrary.py",
    "tests/test_rz_native_pauli.py",
    "tools/report_readability.py",
    "tools/quantinuum_challenge_terket_graphs.py",
}

ENGINE_IMPL_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+(?:\.+_engine_impl|terket\._engine_impl)\s+import\b|"
    r"from\s+\.\s+import\s+_engine_impl\b|"
    r"import\s+terket\._engine_impl\b)"
)

CIRCUIT_SPEC_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+(?:\.+circuit_spec|terket\.circuit_spec)\s+import\b|"
    r"import\s+terket\.circuit_spec\b)"
)

DYNAMIC_ENGINE_IMPL_IMPORT_RE = re.compile(
    r"importlib\.import_module\(\s*[\"']terket\._engine_impl[\"']\s*\)"
)


def _load_report_module():
    spec = importlib.util.spec_from_file_location("terket_report_file_sizes", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load tool module from {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_readability_report_module():
    spec = importlib.util.spec_from_file_location("terket_report_readability", READABILITY_TOOL_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load tool module from {READABILITY_TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _scan_importers(pattern: re.Pattern[str]) -> set[str]:
    importers: set[str] = set()
    for root_name in ("src", "tests", "benchmarks", "tools", "scripts"):
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            text = path.read_text(encoding="utf-8")
            if any(pattern.match(line) for line in text.splitlines()):
                importers.add(rel)
    return importers


def _scan_text_matches(pattern: re.Pattern[str]) -> set[str]:
    importers: set[str] = set()
    for root_name in ("src", "tests", "benchmarks", "tools", "scripts"):
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            text = path.read_text(encoding="utf-8")
            if pattern.search(text):
                importers.add(rel)
    return importers


def _assert_frozen_set(actual: set[str], expected: set[str]) -> None:
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    assert not extra and not missing, f"extra={extra} missing={missing}"


def test_file_size_allowlist_matches_current_repo_state() -> None:
    report = _load_report_module()
    policy = report.load_policy()
    summary = report.summarize_scan(report.scan_repo(REPO_ROOT, policy), policy)
    assert not summary.unexpected_over_fail
    assert not summary.stale_allowlist


def test_readability_allowlist_matches_current_repo_state() -> None:
    report = _load_readability_report_module()
    policy = report.load_policy()
    summary = report.summarize_scan(report.scan_repo(REPO_ROOT, policy), policy)
    assert not summary.unexpected_over_fail_files
    assert not summary.stale_allow_over_file
    assert not summary.unexpected_over_fail_functions
    assert not summary.stale_allow_over_function
    assert not summary.unexpected_missing_module_docstrings
    assert not summary.stale_allow_missing_module_docstring
    assert not summary.unexpected_module_replace_shims
    assert not summary.stale_allow_module_replace_shim


def test_engine_impl_importers_are_frozen() -> None:
    _assert_frozen_set(_scan_importers(ENGINE_IMPL_IMPORT_RE), ALLOWED_ENGINE_IMPL_IMPORTERS)


def test_circuit_spec_importers_are_frozen() -> None:
    _assert_frozen_set(_scan_importers(CIRCUIT_SPEC_IMPORT_RE), ALLOWED_CIRCUIT_SPEC_IMPORTERS)


def test_engine_impl_dynamic_importers_are_frozen() -> None:
    _assert_frozen_set(
        _scan_text_matches(DYNAMIC_ENGINE_IMPL_IMPORT_RE),
        ALLOWED_ENGINE_IMPL_DYNAMIC_IMPORTERS,
    )


def test_over_report_paths_are_frozen() -> None:
    report = _load_report_module()
    policy = report.load_policy()
    summary = report.summarize_scan(report.scan_repo(REPO_ROOT, policy), policy)
    _assert_frozen_set({item.path for item in summary.over_report}, ALLOWED_OVER_REPORT_PATHS)
