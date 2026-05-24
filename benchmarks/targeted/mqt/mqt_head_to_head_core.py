"""Sequential MQT Bench sample benchmark for TerKet versus quimb."""

from __future__ import annotations

import argparse
import cProfile
import math
from dataclasses import dataclass
from pathlib import Path
import pstats
import sys
import time
from typing import Sequence

import networkx as nx
from mqt.bench import get_benchmark_alg, get_benchmark_indep
import psutil
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
RESULTS_ROOT = REPO_ROOT / "results"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket import analyze_circuit, compute_circuit_amplitude, normalize_circuit
from terket.benchmarking import (
    measure_callable,
    quimb_amplitude,
    quimb_circuit_from_circuit,
    runtime_versions,
    write_rows,
)
from terket.benchmarking.mqt import bind_deterministic_parameters, hash_bits
from terket.circuits import _circuit_global_phase_radians
from terket.cubic_arithmetic import detect_factorization
from terket.engine import _aff_compose_cached, _min_fill_cubic_order, build_state

MB = 1024 * 1024
AFFINE_RESTRICTION_GUARD_MIN_FREE_VARS = 512
AFFINE_RESTRICTION_GUARD_MIN_Q3_TERMS = 50_000


DEFAULT_SAMPLE: tuple[tuple[str, int], ...] = (
    ("ghz", 24),
    ("graphstate", 24),
    ("bv", 24),
    ("dj", 24),
    ("qft", 16),
    ("qftentangled", 16),
    ("qaoa", 16),
    ("grover", 12),
    ("draper_qft_adder", 8),
    ("cdkm_ripple_carry_adder", 8),
    ("qpeexact", 8),
    ("vqe_two_local", 16),
    ("randomcircuit", 12),
)


@dataclass(frozen=True)
class MqtBenchRow:
    benchmark: str
    circuit_size: int
    n_qubits: int | None
    gate_count: int | None
    depth: int | None
    two_qubit_gate_count: int | None
    input_bits: str | None
    output_bits: str | None
    output_hamming_weight: int | None
    restricted_free_vars: int | None
    restricted_q1_support: int | None
    restricted_q2_terms: int | None
    restricted_q3_terms: int | None
    restricted_q3_support: int | None
    restricted_component_count: int | None
    restricted_largest_component_vars: int | None
    restricted_largest_component_q2_terms: int | None
    restricted_largest_component_q3_terms: int | None
    restricted_min_fill_width: int | None
    interaction_edge_count: int | None
    interaction_min_fill_width: int | None
    quimb_optimize: str
    quimb_rehearsal_wall_time_s: float | None
    quimb_contraction_width: float | None
    quimb_log2_max_tensor_size: float | None
    quimb_log10_total_flops: float | None
    terket_phase3_backend: str | None
    terket_cubic_obstruction: int | None
    terket_gauss_obstruction: int | None
    terket_cost_model_r: int | None
    terket_wall_time_s: float | None
    terket_peak_rss_mb: float | None
    quimb_wall_time_s: float | None
    quimb_peak_rss_mb: float | None
    abs_error: float | None
    relative_error: float | None
    quimb_over_terket_time_ratio: float | None
    quimb_over_terket_peak_rss_ratio: float | None
    unexpected_children_before: int
    unexpected_children_after: int
    python_version: str
    numpy_version: str
    qiskit_version: str
    quimb_version: str
    cotengra_version: str
    status: str
    error_type: str | None
    error_message: str | None


def _log2_int(value: int) -> float:
    if value <= 0:
        return float("-inf")
    bits = int(value).bit_length()
    if bits <= 53:
        return math.log2(float(value))
    shift = bits - 53
    head = int(value) >> shift
    return math.log2(float(head)) + shift


def _log10_int(value: int) -> float:
    if value <= 0:
        return float("-inf")
    return _log2_int(value) * math.log10(2.0)


def _bits_to_string(bits: Sequence[int]) -> str:
    return "".join(str(int(bit)) for bit in bits)


def _interaction_graph_metrics(circuit: QuantumCircuit) -> tuple[int, int]:
    graph = nx.Graph()
    graph.add_nodes_from(range(circuit.num_qubits))
    dag = circuit_to_dag(circuit)
    for node in dag.op_nodes():
        qargs = tuple(circuit.find_bit(qubit).index for qubit in node.qargs)
        if len(qargs) == 2:
            graph.add_edge(min(qargs), max(qargs))

    if graph.number_of_nodes() == 0:
        return 0, 0

    width, _decomposition = nx.approximation.treewidth_min_fill_in(graph)
    return int(graph.number_of_edges()), int(width)


def _available_memory_mb() -> float:
    return float(psutil.virtual_memory().available / MB)


def _cleanup_child_processes() -> int:
    parent = psutil.Process()
    children = parent.children(recursive=True)
    if not children:
        return 0

    for child in children:
        try:
            child.terminate()
        except psutil.Error:
            pass

    _gone, alive = psutil.wait_procs(children, timeout=5.0)
    for child in alive:
        try:
            child.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=5.0)
    return len(children)


def _restricted_phase_metrics(spec, input_bits: Sequence[int], output_bits: Sequence[int]) -> dict[str, int]:
    state = build_state(
        spec.n_qubits,
        spec.gates,
        input_bits,
        global_phase_radians=_circuit_global_phase_radians(spec),
    )
    cache = state._prepare_echelon()
    solved = state._solve_for_output(cache, output_bits)
    if solved is None:
        return {
            "restricted_free_vars": 0,
            "restricted_q1_support": 0,
            "restricted_q2_terms": 0,
            "restricted_q3_terms": 0,
            "restricted_q3_support": 0,
            "restricted_component_count": 0,
            "restricted_largest_component_vars": 0,
            "restricted_largest_component_q2_terms": 0,
            "restricted_largest_component_q3_terms": 0,
            "restricted_min_fill_width": 0,
        }

    shift_mask, _, gamma, k = solved
    q_free = _aff_compose_cached(state.q, shift_mask, gamma, k)
    components = detect_factorization(q_free)
    largest_component = max(components, key=len) if components else ()
    largest_component_set = set(largest_component)
    q3_support = set()
    for i, j, k in q_free.q3:
        q3_support.add(i)
        q3_support.add(j)
        q3_support.add(k)
    largest_component_q2_terms = sum(1 for i, j in q_free.q2 if i in largest_component_set and j in largest_component_set)
    largest_component_q3_terms = sum(
        1 for i, j, k in q_free.q3 if i in largest_component_set and j in largest_component_set and k in largest_component_set
    )
    _order, width = _min_fill_cubic_order(q_free)
    return {
        "restricted_free_vars": int(q_free.n),
        "restricted_q1_support": int(sum(1 for coeff in q_free.q1 if coeff)),
        "restricted_q2_terms": int(len(q_free.q2)),
        "restricted_q3_terms": int(len(q_free.q3)),
        "restricted_q3_support": int(len(q3_support)),
        "restricted_component_count": int(len(components)),
        "restricted_largest_component_vars": int(len(largest_component)),
        "restricted_largest_component_q2_terms": int(largest_component_q2_terms),
        "restricted_largest_component_q3_terms": int(largest_component_q3_terms),
        "restricted_min_fill_width": int(width),
    }


def _affine_restriction_guard(
    spec,
    input_bits: Sequence[int],
    output_bits: Sequence[int],
) -> tuple[dict[str, int | None], RuntimeError] | None:
    state = build_state(
        spec.n_qubits,
        spec.gates,
        input_bits,
        global_phase_radians=_circuit_global_phase_radians(spec),
    )
    cache = state._prepare_echelon()
    solved = state._solve_for_output(cache, output_bits)
    if solved is None:
        return None

    _shift_mask, _free_vars, _gamma, free_count = solved
    q3_terms = len(state.q.q3)
    if free_count < AFFINE_RESTRICTION_GUARD_MIN_FREE_VARS or q3_terms < AFFINE_RESTRICTION_GUARD_MIN_Q3_TERMS:
        return None

    q3_support = set()
    for i, j, k in state.q.q3:
        q3_support.add(i)
        q3_support.add(j)
        q3_support.add(k)

    metrics: dict[str, int | None] = {
        "restricted_free_vars": int(free_count),
        "restricted_q1_support": int(sum(1 for coeff in state.q.q1 if coeff)),
        "restricted_q2_terms": int(len(state.q.q2)),
        "restricted_q3_terms": int(q3_terms),
        "restricted_q3_support": int(len(q3_support)),
        "restricted_component_count": None,
        "restricted_largest_component_vars": None,
        "restricted_largest_component_q2_terms": None,
        "restricted_largest_component_q3_terms": None,
        "restricted_min_fill_width": None,
    }
    error = RuntimeError(
        "affine restriction too large to materialize safely: "
        f"free_vars={free_count}, pre_restriction_q3_terms={q3_terms}"
    )
    return metrics, error


def _rehearsal_metrics(spec, input_bits: Sequence[int], output_bits: Sequence[int], optimize: str) -> tuple[float, float, float, float]:
    quimb_circuit = quimb_circuit_from_circuit(spec, input_bits)
    output_label = "".join(str(int(bit)) for bit in reversed(tuple(output_bits)))
    start = time.perf_counter()
    # Use rank-only simplification so the width proxy still reflects circuit
    # structure instead of collapsing to trivial scalarized networks.
    rehearsal = quimb_circuit.amplitude_rehearse(output_label, optimize=optimize, simplify_sequence="R")
    elapsed = time.perf_counter() - start
    tree = rehearsal["tree"]
    return (
        float(elapsed),
        float(tree.contraction_width()),
        float(_log2_int(int(tree.max_size()))),
        float(_log10_int(int(tree.total_flops()))),
    )


def _build_row(
    benchmark: str,
    circuit_size: int,
    *,
    quimb_optimize: str,
    versions: dict[str, str],
    status: str,
    unexpected_children_before: int,
    unexpected_children_after: int,
    error: Exception | None = None,
    n_qubits: int | None = None,
    gate_count: int | None = None,
    depth: int | None = None,
    two_qubit_gate_count: int | None = None,
    input_bits: str | None = None,
    output_bits: str | None = None,
    output_hamming_weight: int | None = None,
    restricted_free_vars: int | None = None,
    restricted_q1_support: int | None = None,
    restricted_q2_terms: int | None = None,
    restricted_q3_terms: int | None = None,
    restricted_q3_support: int | None = None,
    restricted_component_count: int | None = None,
    restricted_largest_component_vars: int | None = None,
    restricted_largest_component_q2_terms: int | None = None,
    restricted_largest_component_q3_terms: int | None = None,
    restricted_min_fill_width: int | None = None,
    interaction_edge_count: int | None = None,
    interaction_min_fill_width: int | None = None,
    quimb_rehearsal_wall_time_s: float | None = None,
    quimb_contraction_width: float | None = None,
    quimb_log2_max_tensor_size: float | None = None,
    quimb_log10_total_flops: float | None = None,
    terket_phase3_backend: str | None = None,
    terket_cubic_obstruction: int | None = None,
    terket_gauss_obstruction: int | None = None,
    terket_cost_model_r: int | None = None,
    terket_wall_time_s: float | None = None,
    terket_peak_rss_mb: float | None = None,
    quimb_wall_time_s: float | None = None,
    quimb_peak_rss_mb: float | None = None,
    abs_error: float | None = None,
    relative_error: float | None = None,
) -> MqtBenchRow:
    time_ratio = None
    if quimb_wall_time_s is not None and terket_wall_time_s is not None:
        time_ratio = float(quimb_wall_time_s / max(terket_wall_time_s, 1e-12))

    rss_ratio = None
    if quimb_peak_rss_mb is not None and terket_peak_rss_mb is not None:
        rss_ratio = float(quimb_peak_rss_mb / max(terket_peak_rss_mb, 1e-12))

    return MqtBenchRow(
        benchmark=benchmark,
        circuit_size=circuit_size,
        n_qubits=n_qubits,
        gate_count=gate_count,
        depth=depth,
        two_qubit_gate_count=two_qubit_gate_count,
        input_bits=input_bits,
        output_bits=output_bits,
        output_hamming_weight=output_hamming_weight,
        restricted_free_vars=restricted_free_vars,
        restricted_q1_support=restricted_q1_support,
        restricted_q2_terms=restricted_q2_terms,
        restricted_q3_terms=restricted_q3_terms,
        restricted_q3_support=restricted_q3_support,
        restricted_component_count=restricted_component_count,
        restricted_largest_component_vars=restricted_largest_component_vars,
        restricted_largest_component_q2_terms=restricted_largest_component_q2_terms,
        restricted_largest_component_q3_terms=restricted_largest_component_q3_terms,
        restricted_min_fill_width=restricted_min_fill_width,
        interaction_edge_count=interaction_edge_count,
        interaction_min_fill_width=interaction_min_fill_width,
        quimb_optimize=quimb_optimize,
        quimb_rehearsal_wall_time_s=quimb_rehearsal_wall_time_s,
        quimb_contraction_width=quimb_contraction_width,
        quimb_log2_max_tensor_size=quimb_log2_max_tensor_size,
        quimb_log10_total_flops=quimb_log10_total_flops,
        terket_phase3_backend=terket_phase3_backend,
        terket_cubic_obstruction=terket_cubic_obstruction,
        terket_gauss_obstruction=terket_gauss_obstruction,
        terket_cost_model_r=terket_cost_model_r,
        terket_wall_time_s=terket_wall_time_s,
        terket_peak_rss_mb=terket_peak_rss_mb,
        quimb_wall_time_s=quimb_wall_time_s,
        quimb_peak_rss_mb=quimb_peak_rss_mb,
        abs_error=abs_error,
        relative_error=relative_error,
        quimb_over_terket_time_ratio=time_ratio,
        quimb_over_terket_peak_rss_ratio=rss_ratio,
        unexpected_children_before=unexpected_children_before,
        unexpected_children_after=unexpected_children_after,
        python_version=versions["python_version"],
        numpy_version=versions["numpy_version"],
        qiskit_version=versions["qiskit_version"],
        quimb_version=versions["quimb_version"],
        cotengra_version=versions["cotengra_version"],
        status=status,
        error_type=None if error is None else type(error).__name__,
        error_message=None if error is None else str(error),
    )


def _run_profiled(func, profile_path: Path | None):
    if profile_path is None:
        return func()
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    result = profiler.runcall(func)
    profiler.dump_stats(str(profile_path))
    text_path = profile_path.with_suffix(profile_path.suffix + ".txt")
    with text_path.open("w", encoding="utf-8") as handle:
        stats = pstats.Stats(profiler, stream=handle)
        stats.sort_stats("cumulative")
        stats.print_stats(80)
    return result


def run_case(
    benchmark: str,
    circuit_size: int,
    *,
    quimb_optimize: str,
    repeats: int,
    mqt_level: str,
    opt_level: int,
    min_available_memory_mb: float,
    max_interaction_width: int,
    max_quimb_width: float,
    max_quimb_log2_tensor_size: float,
    terket_only: bool,
    profile_dir: Path | None,
) -> MqtBenchRow:
    versions = runtime_versions()
    unexpected_children_before = _cleanup_child_processes()

    try:
        if mqt_level == "alg":
            circuit = get_benchmark_alg(
                benchmark,
                circuit_size=circuit_size,
                random_parameters=False,
            )
        else:
            circuit = get_benchmark_indep(
                benchmark,
                circuit_size=circuit_size,
                opt_level=opt_level,
                random_parameters=False,
            )
        circuit = bind_deterministic_parameters(circuit, benchmark, circuit_size)
        spec = normalize_circuit(circuit)
        input_bits = (0,) * spec.n_qubits
        output_bits = hash_bits(f"{benchmark}:{circuit_size}:output", spec.n_qubits)
        two_qubit_gate_count = sum(1 for instruction in circuit.data if len(instruction.qubits) == 2)
        if len(spec.gates) >= 30_000:
            print(f"stage=restriction_guard benchmark={benchmark} size={circuit_size}", flush=True)
            restriction_guard = _affine_restriction_guard(spec, input_bits, output_bits)
            if restriction_guard is not None:
                restricted_metrics, guard_error = restriction_guard
                unexpected_children_after = _cleanup_child_processes()
                return _build_row(
                    benchmark,
                    circuit_size,
                    quimb_optimize=quimb_optimize,
                    versions=versions,
                    status="guard_skip_restriction_complexity",
                    unexpected_children_before=unexpected_children_before,
                    unexpected_children_after=unexpected_children_after,
                    n_qubits=spec.n_qubits,
                    gate_count=len(spec.gates),
                    depth=int(circuit.depth()),
                    two_qubit_gate_count=two_qubit_gate_count,
                    input_bits=_bits_to_string(input_bits),
                    output_bits=_bits_to_string(output_bits),
                    output_hamming_weight=int(sum(output_bits)),
                    **restricted_metrics,
                    error=guard_error,
                )
        print(f"stage=analyze benchmark={benchmark} size={circuit_size}", flush=True)
        analysis = analyze_circuit(spec, input_bits, output_bits)
        print(f"stage=restricted_metrics benchmark={benchmark} size={circuit_size}", flush=True)
        restricted_metrics = _restricted_phase_metrics(spec, input_bits, output_bits)
        interaction_edge_count, interaction_min_fill_width = _interaction_graph_metrics(circuit)
        available_before_mb = _available_memory_mb()
        if available_before_mb < min_available_memory_mb:
            unexpected_children_after = _cleanup_child_processes()
            return _build_row(
                benchmark,
                circuit_size,
                quimb_optimize=quimb_optimize,
                versions=versions,
                status="guard_skip_low_memory",
                unexpected_children_before=unexpected_children_before,
                unexpected_children_after=unexpected_children_after,
                n_qubits=spec.n_qubits,
                gate_count=len(spec.gates),
                depth=int(circuit.depth()),
                two_qubit_gate_count=two_qubit_gate_count,
                input_bits=_bits_to_string(input_bits),
                output_bits=_bits_to_string(output_bits),
                output_hamming_weight=int(sum(output_bits)),
                **restricted_metrics,
                interaction_edge_count=interaction_edge_count,
                interaction_min_fill_width=interaction_min_fill_width,
                terket_phase3_backend=str(analysis.get("phase3_backend") or ""),
                terket_cubic_obstruction=int(analysis["cubic_obstruction"]),
                terket_gauss_obstruction=int(analysis["gauss_obstruction"]),
                terket_cost_model_r=int(analysis["cost_model_r"]),
                error=RuntimeError(
                    f"available_memory_mb={available_before_mb:.1f} below guard threshold {min_available_memory_mb}"
                ),
            )

        if interaction_min_fill_width > max_interaction_width:
            unexpected_children_after = _cleanup_child_processes()
            return _build_row(
                benchmark,
                circuit_size,
                quimb_optimize=quimb_optimize,
                versions=versions,
                status="guard_skip_interaction_width",
                unexpected_children_before=unexpected_children_before,
                unexpected_children_after=unexpected_children_after,
                n_qubits=spec.n_qubits,
                gate_count=len(spec.gates),
                depth=int(circuit.depth()),
                two_qubit_gate_count=two_qubit_gate_count,
                input_bits=_bits_to_string(input_bits),
                output_bits=_bits_to_string(output_bits),
                output_hamming_weight=int(sum(output_bits)),
                interaction_edge_count=interaction_edge_count,
                interaction_min_fill_width=interaction_min_fill_width,
                terket_phase3_backend=str(analysis.get("phase3_backend") or ""),
                terket_cubic_obstruction=int(analysis["cubic_obstruction"]),
                terket_gauss_obstruction=int(analysis["gauss_obstruction"]),
                terket_cost_model_r=int(analysis["cost_model_r"]),
                error=RuntimeError(
                    f"interaction_min_fill_width={interaction_min_fill_width} above guard threshold {max_interaction_width}"
                ),
            )

        quimb_rehearsal_wall_time_s = None
        quimb_contraction_width = None
        quimb_log2_max_tensor_size = None
        quimb_log10_total_flops = None
        if not terket_only:
            print(f"stage=quimb_rehearsal benchmark={benchmark} size={circuit_size}", flush=True)
            try:
                (
                    quimb_rehearsal_wall_time_s,
                    quimb_contraction_width,
                    quimb_log2_max_tensor_size,
                    quimb_log10_total_flops,
                ) = _rehearsal_metrics(
                    spec,
                    input_bits,
                    output_bits,
                    quimb_optimize,
                )
            except Exception:
                pass

            if (
                quimb_contraction_width is not None
                and quimb_log2_max_tensor_size is not None
                and (quimb_contraction_width > max_quimb_width or quimb_log2_max_tensor_size > max_quimb_log2_tensor_size)
            ):
                unexpected_children_after = _cleanup_child_processes()
                return _build_row(
                    benchmark,
                    circuit_size,
                    quimb_optimize=quimb_optimize,
                    versions=versions,
                    status="guard_skip_quimb_width",
                    unexpected_children_before=unexpected_children_before,
                    unexpected_children_after=unexpected_children_after,
                    n_qubits=spec.n_qubits,
                    gate_count=len(spec.gates),
                    depth=int(circuit.depth()),
                    two_qubit_gate_count=two_qubit_gate_count,
                    input_bits=_bits_to_string(input_bits),
                    output_bits=_bits_to_string(output_bits),
                    output_hamming_weight=int(sum(output_bits)),
                    **restricted_metrics,
                    interaction_edge_count=interaction_edge_count,
                    interaction_min_fill_width=interaction_min_fill_width,
                    quimb_rehearsal_wall_time_s=quimb_rehearsal_wall_time_s,
                    quimb_contraction_width=quimb_contraction_width,
                    quimb_log2_max_tensor_size=quimb_log2_max_tensor_size,
                    quimb_log10_total_flops=quimb_log10_total_flops,
                    terket_phase3_backend=str(analysis.get("phase3_backend") or ""),
                    terket_cubic_obstruction=int(analysis["cubic_obstruction"]),
                    terket_gauss_obstruction=int(analysis["gauss_obstruction"]),
                    terket_cost_model_r=int(analysis["cost_model_r"]),
                    error=RuntimeError(
                        f"quimb_width={quimb_contraction_width}, quimb_log2_max_tensor_size={quimb_log2_max_tensor_size}"
                    ),
                )

        terket_profile_path = None if profile_dir is None else profile_dir / f"{benchmark}_{circuit_size}.prof"
        print(f"stage=terket benchmark={benchmark} size={circuit_size}", flush=True)
        terket_wall_time_s, terket_peak_rss_mb, terket_result = measure_callable(
            lambda: _run_profiled(
                lambda: compute_circuit_amplitude(spec, input_bits, output_bits, as_complex=True),
                terket_profile_path,
            ),
            repeats,
        )
        terket_amp, _ = terket_result
        quimb_wall_time_s = None
        quimb_peak_rss_mb = None
        quimb_amp = None
        if not terket_only:
            print(f"stage=quimb benchmark={benchmark} size={circuit_size}", flush=True)
            quimb_wall_time_s, quimb_peak_rss_mb, quimb_amp = measure_callable(
                lambda: quimb_amplitude(spec, input_bits, output_bits, optimize=quimb_optimize),
                repeats,
            )

        terket_amp = complex(terket_amp)
        abs_error = None
        relative_error = None
        if quimb_amp is not None:
            quimb_amp = complex(quimb_amp)
            abs_error = abs(terket_amp - quimb_amp)
            denom = max(abs(terket_amp), abs(quimb_amp))
            relative_error = 0.0 if denom == 0.0 else abs_error / denom
        unexpected_children_after = _cleanup_child_processes()

        return _build_row(
            benchmark,
            circuit_size,
            quimb_optimize=quimb_optimize,
            versions=versions,
            status="ok",
            unexpected_children_before=unexpected_children_before,
            unexpected_children_after=unexpected_children_after,
            n_qubits=spec.n_qubits,
            gate_count=len(spec.gates),
            depth=int(circuit.depth()),
            two_qubit_gate_count=two_qubit_gate_count,
            input_bits=_bits_to_string(input_bits),
            output_bits=_bits_to_string(output_bits),
            output_hamming_weight=int(sum(output_bits)),
            **restricted_metrics,
            interaction_edge_count=interaction_edge_count,
            interaction_min_fill_width=interaction_min_fill_width,
            quimb_rehearsal_wall_time_s=quimb_rehearsal_wall_time_s,
            quimb_contraction_width=quimb_contraction_width,
            quimb_log2_max_tensor_size=quimb_log2_max_tensor_size,
            quimb_log10_total_flops=quimb_log10_total_flops,
            terket_phase3_backend=str(analysis.get("phase3_backend") or ""),
            terket_cubic_obstruction=int(analysis["cubic_obstruction"]),
            terket_gauss_obstruction=int(analysis["gauss_obstruction"]),
            terket_cost_model_r=int(analysis["cost_model_r"]),
            terket_wall_time_s=float(terket_wall_time_s),
            terket_peak_rss_mb=float(terket_peak_rss_mb),
            quimb_wall_time_s=None if quimb_wall_time_s is None else float(quimb_wall_time_s),
            quimb_peak_rss_mb=None if quimb_peak_rss_mb is None else float(quimb_peak_rss_mb),
            abs_error=None if abs_error is None else float(abs_error),
            relative_error=None if relative_error is None else float(relative_error),
        )
    except Exception as exc:
        unexpected_children_after = _cleanup_child_processes()
        return _build_row(
            benchmark,
            circuit_size,
            quimb_optimize=quimb_optimize,
            versions=versions,
            status="error",
            unexpected_children_before=unexpected_children_before,
            unexpected_children_after=unexpected_children_after,
            error=exc,
        )


def parse_sample_args(sample_args: Sequence[str]) -> list[tuple[str, int]]:
    if not sample_args:
        return list(DEFAULT_SAMPLE)

    sample: list[tuple[str, int]] = []
    for item in sample_args:
        if ":" not in item:
            raise ValueError(f"Expected benchmark item in name:size form, received {item!r}.")
        name, raw_size = item.split(":", 1)
        sample.append((name, int(raw_size)))
    return sample
