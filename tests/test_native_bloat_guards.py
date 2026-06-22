"""Guard tests for native-source layout and build-matrix expectations."""

from __future__ import annotations

import os
from pathlib import Path
import re
import unittest
from unittest import mock

import terket
from terket import engine
from terket import _reduction_classify as reduction_classify
from terket.cubic_arithmetic import PhaseFunction


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_DIR = REPO_ROOT / "src" / "terket"

EXPECTED_NATIVE_SOURCES = (
    "src/terket/_schur_native.c",
    "src/terket/_schur_native_support.c",
    "src/terket/_schur_native_graph.c",
    "src/terket/native_phase_eval.c",
    "src/terket/native_constraint_elim.c",
    "src/terket/native_affine_compose.c",
    "src/terket/native_classification.c",
    "src/terket/native_output_solve.c",
    "src/terket/native_level3_dp.c",
    "src/terket/native_phase_function_dp.c",
    "src/terket/native_q3_free_dp.c",
)

EXPECTED_NATIVE_METHODS = {
    "aff_compose_terms",
    "build_classification_data",
    "build_classification_lookup",
    "build_level3_treewidth_plan",
    "build_phase_function_treewidth_support_plan",
    "build_q3_free_treewidth_plan",
    "build_scaled_factor_treewidth_plan",
    "classification_structure_key",
    "clear_support_cache",
    "cubic_order_width",
    "elim_single_partner_constraint_terms",
    "elim_sparse_quadratics_batch_terms",
    "elim_two_partner_constraint_terms",
    "evaluate_q_mask_terms",
    "min_degree_cubic_order",
    "min_fill_cubic_order",
    "q3_free_treewidth_dp_work",
    "rank_q3_free_cutset_extensions",
    "solve_output_shift_mask_u64",
    "solve_output_shift_masks_u64",
    "sum_factor_tables_scaled",
    "sum_level3_treewidth_preplanned",
    "sum_level3_treewidth_preplanned_batch_array",
    "sum_phase_function_treewidth_preplanned_batch_scaled_array",
    "sum_q3_free_treewidth_batch_scaled",
    "sum_q3_free_treewidth_preplanned_batch_scaled",
    "sum_q3_free_treewidth_preplanned_batch_scaled_array",
    "sum_residue_forest_batch_scaled_array",
    "sum_scaled_factor_treewidth_preplanned",
    "sum_treewidth_dp_level3",
    "support_from_mask",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_setup_native_source_list_is_stable() -> None:
    setup_text = _read(REPO_ROOT / "setup.py")
    actual_sources = tuple(re.findall(r'"(src/terket/[^"]+\.c)"', setup_text))
    assert actual_sources == EXPECTED_NATIVE_SOURCES
    assert '"src/terket/_schur_native_algebra.c"' not in setup_text
    assert '"src/terket/_schur_native_dp.c"' not in setup_text


def test_native_module_method_set_is_stable() -> None:
    module_text = _read(NATIVE_DIR / "_schur_native.c")
    methods = set(re.findall(r'NATIVE_METHOD\(\s*"([^"]+)"', module_text))
    methods.update(re.findall(r'^\s*\{\s*\n\s*"([^"]+)"', module_text, flags=re.MULTILINE))
    assert methods == EXPECTED_NATIVE_METHODS


@unittest.skipIf(
    engine._schur_native is None
    or not hasattr(engine._schur_native, "elim_sparse_quadratics_batch_terms"),
    "native sparse quadratic batch unavailable",
)
def test_native_sparse_quadratic_batch_matches_python_fallback() -> None:
    q = PhaseFunction(
        9,
        level=3,
        q1=[2, 2, 6, 2, 6, 2, 2, 6, 2],
        q2={(idx, idx + 1): 2 for idx in range(8)},
        q3={},
    )
    candidates = tuple(range(q.n))

    native_q, native_scale, native_removed = reduction_classify._elim_sparse_dead_quadratics_batch(
        q,
        candidates,
    )
    with mock.patch.object(reduction_classify, "_native_symbol", return_value=None):
        python_q, python_scale, python_removed = reduction_classify._elim_sparse_dead_quadratics_batch(
            q,
            candidates,
        )

    assert native_scale == python_scale
    assert native_removed == python_removed
    assert native_q.q0 == python_q.q0
    assert native_q.q1 == python_q.q1
    assert native_q.q2 == python_q.q2


@unittest.skipIf(engine._schur_native is None, "native accelerator unavailable")
def test_native_min_degree_heap_preserves_fill_updated_order() -> None:
    q2 = {
        (0, 1): 1,
        (0, 2): 1,
        (1, 3): 1,
        (2, 3): 1,
        (3, 4): 1,
        (4, 5): 1,
        (4, 6): 1,
    }
    q3 = {(1, 2, 5): 1}

    order, width = engine._schur_native.min_degree_cubic_order(7, q2, q3)

    assert order == [6, 0, 4, 1, 2, 3, 5]
    assert width == 4


def test_native_ownership_map_lists_current_c_files() -> None:
    ownership = _read(REPO_ROOT / "docs" / "native_c_ownership_map.md")
    for source in EXPECTED_NATIVE_SOURCES:
        assert os.path.basename(source) in ownership
    assert "_schur_native_internal.h" in ownership


@unittest.skipIf(engine._schur_native is None, "native accelerator unavailable")
def test_native_and_python_treewidth_dp_match_before_c_split() -> None:
    q = engine._phase_function_from_parts(
        5,
        level=3,
        q0=0,
        q1=[1, 2, 3, 4, 5],
        q2={(0, 1): 1, (1, 2): 2, (2, 3): 3},
        q3={(0, 2, 4): 1, (1, 3, 4): 1},
    )
    order = [0, 1, 2, 3, 4]

    native_total, native_width = engine._sum_via_treewidth_dp_scaled(q, order)
    native_module = engine._schur_native
    terket.clear_caches()
    with mock.patch.object(engine, "_schur_native", None):
        python_total, python_width = engine._sum_via_treewidth_dp_scaled(q, order)
    terket.clear_caches()
    assert engine._schur_native is native_module

    assert native_width == python_width
    assert abs(engine._scaled_to_complex(native_total) - engine._scaled_to_complex(python_total)) <= 1e-12


@unittest.skipIf(engine._schur_native is None, "native accelerator unavailable")
def test_native_and_python_public_amplitudes_match_before_c_split() -> None:
    circuit = terket.make_circuit(
        4,
        [
            ("h", 0),
            ("t", 1),
            ("cnot", 0, 2),
            ("rz_dyadic", 2, 1, 3),
            ("h", 3),
            ("cnot", 3, 1),
            ("tdg", 2),
            ("s", 0),
        ],
    )
    input_bits = [0, 1, 0, 1]
    output_bits = [1, 0, 1, 0]

    native_value, native_info = terket.compute_circuit_amplitude(
        circuit,
        input_bits,
        output_bits,
        as_complex=True,
    )
    native_module = engine._schur_native
    terket.clear_caches()
    with mock.patch.object(engine, "_schur_native", None):
        python_value, python_info = terket.compute_circuit_amplitude(
            circuit,
            input_bits,
            output_bits,
            as_complex=True,
        )
    terket.clear_caches()
    assert engine._schur_native is native_module

    assert abs(native_value - python_value) <= 1e-12
    assert native_info["phase3_backend"] == python_info["phase3_backend"]
