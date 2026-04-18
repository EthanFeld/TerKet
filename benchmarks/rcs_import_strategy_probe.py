"""Bounded structural probe for RCS import strategies."""

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
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.circuit.library import iSwapGate
import rustworkx as rx


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
RESULTS_ROOT = REPO_ROOT / "results"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket import normalize_circuit
from terket.circuits import _circuit_global_phase_radians
from terket.engine import (
    _build_q3_free_raw_constraint_plan,
    _min_fill_cubic_order,
    _phase_function_from_parts,
    _restrict_q3_free_raw_constraint_plan,
    build_state,
)


SQRT_ISWAP = iSwapGate().power(0.5)


@dataclass(frozen=True)
class GridSpec:
    name: str
    num_qubits: int
    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ProbeRow:
    case: str
    cycles: int
    seed: int
    strategy: str
    n_qubits: int
    gate_count: int
    rz_count: int
    sx_count: int
    cx_count: int
    cz_count: int
    x_count: int
    rzz_count: int
    import_wall_time_s: float
    built_state_vars: int
    built_state_q2_edges: int
    built_state_q2_density: float
    built_state_q3_terms: int
    largest_component_backend: str | None
    largest_component_vars: int | None
    largest_component_q2_edges: int | None
    largest_component_q2_density: float | None
    largest_component_minfill_width: int | None
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


def _wrap_qasm_with_qelib(qasm_text: str) -> str:
    body = [
        line
        for line in qasm_text.splitlines()
        if not line.startswith("OPENQASM") and "include" not in line
    ]
    return 'OPENQASM 2.0;\ninclude "qelib1.inc";\n' + "\n".join(body)


def _zx_basic_optimize_qiskit(circuit: QuantumCircuit) -> QuantumCircuit:
    try:
        import pyzx as zx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pyzx is required for the zx_basic strategy.") from exc

    zxc = zx.Circuit.from_qasm(qasm2.dumps(circuit))
    optimized = zx.optimize.basic_optimization(zxc.to_basic_gates())
    return qasm2.loads(_wrap_qasm_with_qelib(optimized.to_qasm()))


def _graph_density(num_vars: int, edge_count: int) -> float:
    if num_vars <= 1:
        return 0.0
    return float(edge_count) / float(num_vars * (num_vars - 1) // 2)


def _transpile_strategy(circuit: QuantumCircuit, strategy: str) -> QuantumCircuit:
    if strategy == "cx_opt0":
        return transpile(circuit, basis_gates=["rz", "sx", "x", "cx"], optimization_level=0)
    if strategy == "cx_opt1":
        return transpile(circuit, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
    if strategy == "cx_opt2":
        return transpile(circuit, basis_gates=["rz", "sx", "x", "cx"], optimization_level=2)
    if strategy == "cz_opt0":
        return transpile(circuit, basis_gates=["rz", "sx", "x", "cz"], optimization_level=0)
    if strategy == "cz_opt1":
        return transpile(circuit, basis_gates=["rz", "sx", "x", "cz"], optimization_level=1)
    if strategy == "rzz_opt1":
        return transpile(circuit, basis_gates=["rz", "sx", "x", "cz", "rzz"], optimization_level=1)
    if strategy == "zx_basic_cx_opt1":
        base = transpile(circuit, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
        optimized = _zx_basic_optimize_qiskit(base)
        return transpile(optimized, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
    raise ValueError(f"Unknown strategy {strategy!r}.")


def _largest_component_summary(spec) -> tuple[str | None, int | None, int | None, float | None, int | None]:
    state = build_state(
        spec.n_qubits,
        spec.gates,
        [0] * spec.n_qubits,
        global_phase_radians=_circuit_global_phase_radians(spec),
    )
    if state.q.q3:
        return None, None, None, None, None

    plan = _build_q3_free_raw_constraint_plan(
        state,
        allow_tensor_contraction=False,
        prefer_reusable_decomposition=False,
        prefer_one_shot_slicing=True,
    )
    restricted = _restrict_q3_free_raw_constraint_plan(plan, state.n)
    if not restricted.components:
        return "none", 0, 0, 0.0, 0
    component_plan = max(restricted.components, key=lambda component: len(component.variables))
    component_q = _phase_function_from_parts(
        len(component_plan.variables),
        level=component_plan.level,
        q0=0,
        q1=[plan.base_q1[var] for var in component_plan.variables],
        q2=component_plan.q2,
        q3={},
    )
    return (
        component_plan.backend,
        len(component_plan.variables),
        len(component_plan.q2),
        _graph_density(len(component_plan.variables), len(component_plan.q2)),
        int(_min_fill_cubic_order(component_q)[1]),
    )


def run_probe(case: str, cycles: int, *, seed: int, strategy: str) -> ProbeRow:
    if case == "smoke":
        grid = grid_spec(3, 3, name="rcs_exact_3x3")
    elif case == "sycamore53":
        grid = grid_spec(6, 9, drop_last=True, name="sycamore_like_53")
    else:
        raise ValueError(f"Unknown case {case!r}.")

    qiskit_circuit = build_rcs_qiskit(grid, cycles, seed=seed)
    start = time.perf_counter()
    try:
        transpiled = _transpile_strategy(qiskit_circuit, strategy)
        spec = normalize_circuit(transpiled, rz_compile_mode="dyadic")
    except Exception:
        import_wall_time_s = time.perf_counter() - start
        return ProbeRow(
            case=case,
            cycles=cycles,
            seed=seed,
            strategy=strategy,
            n_qubits=grid.num_qubits,
            gate_count=0,
            rz_count=0,
            sx_count=0,
            cx_count=0,
            cz_count=0,
            x_count=0,
            rzz_count=0,
            import_wall_time_s=float(import_wall_time_s),
            built_state_vars=0,
            built_state_q2_edges=0,
            built_state_q2_density=0.0,
            built_state_q3_terms=0,
            largest_component_backend=None,
            largest_component_vars=None,
            largest_component_q2_edges=None,
            largest_component_q2_density=None,
            largest_component_minfill_width=None,
            status="import_error",
        )
    import_wall_time_s = time.perf_counter() - start

    state = build_state(
        spec.n_qubits,
        spec.gates,
        [0] * spec.n_qubits,
        global_phase_radians=_circuit_global_phase_radians(spec),
    )
    backend, vars_count, edge_count, component_density, width = _largest_component_summary(spec)

    counts = transpiled.count_ops()
    status = "q3_present" if state.q.q3 else "ok"
    return ProbeRow(
        case=case,
        cycles=cycles,
        seed=seed,
        strategy=strategy,
        n_qubits=spec.n_qubits,
        gate_count=len(spec.gates),
        rz_count=int(counts.get("rz", 0)),
        sx_count=int(counts.get("sx", 0)),
        cx_count=int(counts.get("cx", 0)),
        cz_count=int(counts.get("cz", 0)),
        x_count=int(counts.get("x", 0)),
        rzz_count=int(counts.get("rzz", 0)),
        import_wall_time_s=float(import_wall_time_s),
        built_state_vars=int(state.q.n),
        built_state_q2_edges=int(len(state.q.q2)),
        built_state_q2_density=float(_graph_density(int(state.q.n), int(len(state.q.q2)))),
        built_state_q3_terms=int(len(state.q.q3)),
        largest_component_backend=backend,
        largest_component_vars=vars_count,
        largest_component_q2_edges=edge_count,
        largest_component_q2_density=component_density,
        largest_component_minfill_width=width,
        status=status,
    )


def write_rows(rows: list[ProbeRow], csv_path: Path) -> None:
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
    parser.add_argument("--cycles", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--strategy",
        nargs="+",
        choices=["cx_opt0", "cx_opt1", "cx_opt2", "cz_opt0", "cz_opt1", "rzz_opt1", "zx_basic_cx_opt1"],
        default=["cx_opt0", "cx_opt1", "cz_opt0", "cz_opt1", "rzz_opt1"],
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=RESULTS_ROOT / "rcs_import_strategy_probe.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows: list[ProbeRow] = []
    for cycles in args.cycles:
        for strategy in args.strategy:
            row = run_probe(args.case, cycles, seed=args.seed, strategy=strategy)
            rows.append(row)
            print(
                f"{row.case} cycles={row.cycles} {row.strategy}: "
                f"vars={row.built_state_vars} q2={row.built_state_q2_edges} q3={row.built_state_q3_terms} "
                f"comp={row.largest_component_backend} width={row.largest_component_minfill_width} status={row.status}"
            )
    write_rows(rows, args.csv)
    print(f"Wrote {len(rows)} row(s) to {args.csv}")


if __name__ == "__main__":
    main()
