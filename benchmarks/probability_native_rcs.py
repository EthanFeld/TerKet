"""Benchmark the standalone probability-native path on Google-style RCS cases."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
import sys
import time

import networkx as nx
import rustworkx as rx
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import iSwapGate


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
RESULTS_ROOT = REPO_ROOT / "results"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket import compute_circuit_amplitude_scaled, normalize_circuit
from terket.probability_native import compute_circuit_probability_scaled


SQRT_ISWAP = iSwapGate().power(0.5)


@dataclass(frozen=True)
class GridSpec:
    name: str
    num_qubits: int
    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class BenchmarkRow:
    case: str
    n_qubits: int
    n_edges: int
    cycles: int
    seed: int
    gate_count: int
    mode: str
    amplitude_wall_time_s: float | None
    amplitude_phase3_backend: str | None
    amplitude_probability: float | None
    amplitude_probability_log2: float | None
    probability_wall_time_s: float | None
    probability_phase3_backend: str | None
    probability_value: float | None
    probability_log2: float | None
    abs_error: float | None
    raw_var_count: int | None
    transformed_var_count: int | None
    transformed_q2_terms: int | None
    transformed_q3_terms: int | None
    effective_factor_count: int | None
    effective_order_width: int | None
    effective_max_scope: int | None
    cubic_obstruction: int | None
    cost_model_r: int | None
    status: str


def edge_layers(num_qubits: int, edges: list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], ...]:
    graph = rx.PyGraph(multigraph=False)
    graph.add_nodes_from(range(num_qubits))
    for left, right in edges:
        if left == right or graph.has_edge(left, right):
            continue
        graph.add_edge(left, right, None)
    color_map = rx.graph_greedy_edge_color(graph)
    by_color: dict[int, list[tuple[int, int]]] = {}
    for edge_index, color in color_map.items():
        left, right = graph.get_edge_endpoints_by_index(edge_index)
        by_color.setdefault(int(color), []).append((int(left), int(right)))
    return tuple(tuple(by_color[color]) for color in sorted(by_color))


def grid_spec(rows: int, cols: int, *, drop_last: bool = False, name: str | None = None) -> GridSpec:
    graph = nx.grid_2d_graph(rows, cols)
    nodes = sorted(graph.nodes())
    if drop_last:
        graph.remove_node(nodes[-1])
        nodes = sorted(graph.nodes())
    mapping = {node: index for index, node in enumerate(nodes)}
    relabeled = nx.relabel_nodes(graph, mapping, copy=True)
    edges = tuple(sorted((min(int(left), int(right)), max(int(left), int(right))) for left, right in relabeled.edges()))
    return GridSpec(
        name=name or f"grid_{rows}x{cols}" + ("_minus1" if drop_last else ""),
        num_qubits=relabeled.number_of_nodes(),
        edges=edges,
    )


def append_random_rcs_single_qubit_layer(qc: QuantumCircuit, n_qubits: int, rng: random.Random) -> None:
    for qubit in range(n_qubits):
        if rng.randrange(2) == 0:
            qc.rx(math.pi / 2.0, qubit)
        else:
            qc.ry(math.pi / 2.0, qubit)


def build_rcs_qiskit(spec: GridSpec, cycles: int, *, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    layers = edge_layers(spec.num_qubits, list(spec.edges))
    qc = QuantumCircuit(spec.num_qubits, name=f"{spec.name}_rcs_{cycles}")
    for cycle in range(cycles):
        append_random_rcs_single_qubit_layer(qc, spec.num_qubits, rng)
        for left, right in layers[cycle % len(layers)]:
            qc.append(SQRT_ISWAP, [left, right])
    return qc


def normalize_transpiled_circuit(circuit: QuantumCircuit):
    transpiled = transpile(
        circuit,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=1,
    )
    return normalize_circuit(transpiled, rz_compile_mode="dyadic")


def probability_from_amplitude_scaled(amplitude) -> tuple[float, float]:
    amplitude_complex = amplitude.to_complex()
    probability = float(abs(amplitude_complex) ** 2)
    return probability, float(2.0 * amplitude.log2_abs())


def run_case(case: str, cycles: int, *, seed: int, mode: str) -> BenchmarkRow:
    if case == "smoke":
        spec = grid_spec(3, 3, name="rcs_exact_3x3")
    elif case == "sycamore53":
        spec = grid_spec(6, 9, drop_last=True, name="sycamore_like_53")
    else:
        raise ValueError(f"Unknown case {case!r}.")

    circuit = normalize_transpiled_circuit(build_rcs_qiskit(spec, cycles, seed=seed))
    input_bits = [0] * circuit.n_qubits
    output_bits = [0] * circuit.n_qubits

    amplitude_wall_time_s = None
    amplitude_phase3_backend = None
    amplitude_probability = None
    amplitude_probability_log2 = None
    probability_value_from_amplitude = None

    if mode == "both":
        start = time.perf_counter()
        amplitude_scaled, amplitude_info = compute_circuit_amplitude_scaled(
            circuit,
            input_bits,
            output_bits,
            allow_tensor_contraction=False,
        )
        amplitude_wall_time_s = time.perf_counter() - start
        amplitude_phase3_backend = str(amplitude_info.get("phase3_backend"))
        amplitude_probability, amplitude_probability_log2 = probability_from_amplitude_scaled(amplitude_scaled)
        probability_value_from_amplitude = amplitude_probability

    start = time.perf_counter()
    probability_scaled, probability_info = compute_circuit_probability_scaled(
        circuit,
        input_bits,
        output_bits,
        allow_tensor_contraction=False,
    )
    probability_wall_time_s = time.perf_counter() - start
    probability_value = probability_scaled.to_float()

    abs_error = None
    if probability_value_from_amplitude is not None:
        abs_error = abs(probability_value - probability_value_from_amplitude)

    return BenchmarkRow(
        case=case,
        n_qubits=spec.num_qubits,
        n_edges=len(spec.edges),
        cycles=cycles,
        seed=seed,
        gate_count=len(circuit.gates),
        mode=mode,
        amplitude_wall_time_s=amplitude_wall_time_s,
        amplitude_phase3_backend=amplitude_phase3_backend,
        amplitude_probability=amplitude_probability,
        amplitude_probability_log2=amplitude_probability_log2,
        probability_wall_time_s=probability_wall_time_s,
        probability_phase3_backend=probability_info.get("phase3_backend"),
        probability_value=probability_value,
        probability_log2=float(probability_scaled.log2()),
        abs_error=abs_error,
        raw_var_count=probability_info.get("raw_var_count"),
        transformed_var_count=probability_info.get("transformed_var_count"),
        transformed_q2_terms=probability_info.get("transformed_q2_terms"),
        transformed_q3_terms=probability_info.get("transformed_q3_terms"),
        effective_factor_count=probability_info.get("effective_factor_count"),
        effective_order_width=probability_info.get("effective_order_width"),
        effective_max_scope=probability_info.get("effective_max_scope"),
        cubic_obstruction=probability_info.get("cubic_obstruction"),
        cost_model_r=probability_info.get("cost_model_r"),
        status="ok",
    )


def write_rows(rows: list[BenchmarkRow], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["smoke", "sycamore53"], default="smoke")
    parser.add_argument("--cycles", type=int, default=None, help="Override the default cycle count for the case.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=["probability", "both"], default="both")
    parser.add_argument(
        "--csv",
        type=Path,
        default=RESULTS_ROOT / "probability_native_rcs.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    default_cycles = 2 if args.case == "smoke" else 6
    cycles = args.cycles if args.cycles is not None else default_cycles
    row = run_case(args.case, cycles, seed=args.seed, mode=args.mode)
    print(
        f"{row.case}: probability={row.probability_wall_time_s:.6f}s "
        f"(backend={row.probability_phase3_backend or 'direct'}, "
        f"log2={row.probability_log2:.6f}, "
        f"width={row.effective_order_width if row.effective_order_width is not None else row.cost_model_r}, "
        f"max_scope={row.effective_max_scope if row.effective_max_scope is not None else 'na'}, "
        f"q3={row.transformed_q3_terms})"
    )
    if row.amplitude_wall_time_s is not None:
        print(
            f"{row.case}: amplitude={row.amplitude_wall_time_s:.6f}s "
            f"(backend={row.amplitude_phase3_backend or 'q3_free'}, abs_error={row.abs_error:.3e})"
        )
    write_rows([row], args.csv)
    print(f"Wrote 1 row to {args.csv}")


if __name__ == "__main__":
    main()
