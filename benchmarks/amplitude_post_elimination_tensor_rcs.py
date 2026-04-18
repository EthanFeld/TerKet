"""Probe tensor-network viability on post-elimination q3-free amplitude residuals."""

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
from terket.circuits import _circuit_global_phase_radians
from terket.engine import (
    _build_cubic_factors,
    _build_q3_free_raw_constraint_plan,
    _dense_q2_matrix,
    _min_fill_cubic_order,
    _phase_function_from_parts,
    _restrict_q3_free_raw_constraint_plan,
    _scaled_to_complex,
    _schur_complement_q3_free_sum_scaled_dense,
    _sum_q3_free_component_scaled,
    _sum_via_tensor_contraction,
    _treewidth_order_width,
    build_state,
)


SQRT_ISWAP = iSwapGate().power(0.5)


@dataclass(frozen=True)
class GridSpec:
    name: str
    num_qubits: int
    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class BenchmarkRow:
    case: str
    cycles: int
    seed: int
    n_qubits: int
    n_edges: int
    gate_count: int
    full_amplitude_wall_time_s: float | None
    full_amplitude_log2_abs: float | None
    full_amplitude_phase3_backend: str | None
    component_index: int
    component_backend: str
    component_prefer_cutset: bool
    component_var_count: int
    component_q2_terms: int
    component_treewidth_width: int | None
    component_cutset_size: int | None
    schur_scale_half_pow2: int
    residual_var_count: int
    residual_q2_terms: int
    residual_min_fill_width: int
    residual_factor_count: int
    greedy_plan_wall_time_s: float | None
    greedy_plan_width: float | None
    greedy_plan_nslices: int | None
    greedy_plan_log2_max_tensor_size: float | None
    greedy_plan_log10_flops: float | None
    exact_residual_wall_time_s: float | None
    exact_residual_abs_value: float | None
    tensor_wall_time_s: float | None
    tensor_abs_value: float | None
    tensor_abs_error: float | None
    tensor_relative_error: float | None
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


def case_spec(case: str) -> tuple[GridSpec, int]:
    if case == "smoke":
        return grid_spec(3, 3, name="rcs_exact_3x3"), 2
    if case == "sycamore53":
        return grid_spec(6, 9, drop_last=True, name="sycamore_like_53"), 6
    raise ValueError(f"Unknown case {case!r}.")


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


def _target_q1(plan, output_bits: list[int]) -> list[int]:
    q1 = list(plan.base_q1)
    for idx, bit in enumerate(output_bits):
        if (int(bit) ^ int(plan.eps0[idx])) & 1:
            q1[plan.lambda_offset + idx] = plan.rhs_linear_coeff
    return q1


def _component_q(component_plan, q1: list[int]):
    q1_local = [q1[var] for var in component_plan.variables]
    return _phase_function_from_parts(
        len(component_plan.variables),
        level=component_plan.level,
        q0=0,
        q1=q1_local,
        q2=component_plan.q2,
        q3={},
    )


def _extract_residual(component_q):
    schur_result = _schur_complement_q3_free_sum_scaled_dense(
        component_q.level,
        component_q.q1,
        _dense_q2_matrix(component_q),
        q0=component_q.q0,
        allow_recursive_fallback=False,
        return_residual_on_fallback=True,
    )
    if isinstance(schur_result[0], complex):
        return None, int(schur_result[1])
    residual_q, scale_half_pow2 = schur_result
    return residual_q, int(scale_half_pow2)


def _select_target_component(restricted_plan, q1: list[int]):
    best = None
    for index, component_plan in enumerate(restricted_plan.components):
        component_q = _component_q(component_plan, q1)
        residual_q, schur_scale = _extract_residual(component_q)
        residual_vars = 0 if residual_q is None else int(residual_q.n)
        residual_edges = 0 if residual_q is None else int(len(residual_q.q2))
        residual_factors = 0 if residual_q is None else int(len(_build_cubic_factors(residual_q)[1]))
        residual_width = 0 if residual_q is None else int(_min_fill_cubic_order(residual_q)[1])
        score = (
            residual_vars,
            residual_edges,
            len(component_plan.variables),
            len(component_plan.q2),
            -index,
        )
        if best is None or score > best["score"]:
            component_width = None
            if component_plan.order:
                component_width = int(_treewidth_order_width(component_q, component_plan.order))
            cutset_size = None
            if component_plan.cutset_plan is not None:
                cutset_size = len(component_plan.cutset_plan.cutset_vars)
            best = {
                "score": score,
                "index": index,
                "component_plan": component_plan,
                "component_q": component_q,
                "component_width": component_width,
                "cutset_size": cutset_size,
                "residual_q": residual_q,
                "schur_scale": schur_scale,
                "residual_factors": residual_factors,
                "residual_width": residual_width,
            }
    assert best is not None
    return best


def _greedy_tensor_plan(residual_q):
    import cotengra as ctg

    _scalar, factors = _build_cubic_factors(residual_q)
    inputs = [tuple(f"v{var}" for var in scope) for scope, _table in sorted(factors.items())]
    size_dict = {f"v{var}": 2 for var in range(residual_q.n)}

    start = time.perf_counter()
    tree = ctg.array_contract_tree(
        inputs,
        output=(),
        size_dict=size_dict,
        optimize="greedy",
        canonicalize=False,
    )
    elapsed = time.perf_counter() - start
    max_size = int(tree.max_size())
    total_flops = int(tree.total_flops())
    return {
        "wall_time_s": float(elapsed),
        "width": float(tree.contraction_width()),
        "nslices": int(tree.nslices),
        "log2_max_tensor_size": float(_log2_int(max_size)),
        "log10_flops": float(_log10_int(total_flops)),
    }


def run_case(
    case: str,
    cycles: int,
    *,
    seed: int,
    run_contraction_up_to_vars: int,
    include_full_amplitude: bool,
) -> BenchmarkRow:
    spec, _default_cycles = case_spec(case)
    circuit = normalize_transpiled_circuit(build_rcs_qiskit(spec, cycles, seed=seed))
    input_bits = [0] * circuit.n_qubits
    output_bits = [0] * circuit.n_qubits

    full_amplitude_wall_time_s = None
    full_amplitude_log2_abs = None
    full_amplitude_phase3_backend = None
    if include_full_amplitude:
        start = time.perf_counter()
        amplitude, info = compute_circuit_amplitude_scaled(
            circuit,
            input_bits,
            output_bits,
            allow_tensor_contraction=False,
        )
        full_amplitude_wall_time_s = time.perf_counter() - start
        full_amplitude_log2_abs = float(amplitude.log2_abs())
        full_amplitude_phase3_backend = str(info.get("phase3_backend"))

    state = build_state(
        circuit.n_qubits,
        circuit.gates,
        input_bits,
        global_phase_radians=_circuit_global_phase_radians(circuit),
    )
    if state.q.q3:
        raise RuntimeError("RCS post-elimination tensor benchmark expects a q3-free amplitude kernel.")

    plan = _build_q3_free_raw_constraint_plan(
        state,
        allow_tensor_contraction=False,
        prefer_reusable_decomposition=False,
        prefer_one_shot_slicing=True,
    )
    restricted_plan = _restrict_q3_free_raw_constraint_plan(plan, state.n)
    q1 = _target_q1(plan, output_bits)
    target = _select_target_component(restricted_plan, q1)

    component_plan = target["component_plan"]
    residual_q = target["residual_q"]
    if residual_q is None:
        return BenchmarkRow(
            case=case,
            cycles=cycles,
            seed=seed,
            n_qubits=spec.num_qubits,
            n_edges=len(spec.edges),
            gate_count=len(circuit.gates),
            full_amplitude_wall_time_s=full_amplitude_wall_time_s,
            full_amplitude_log2_abs=full_amplitude_log2_abs,
            full_amplitude_phase3_backend=full_amplitude_phase3_backend,
            component_index=target["index"],
            component_backend=component_plan.backend,
            component_prefer_cutset=bool(component_plan.prefer_cutset_backend),
            component_var_count=len(component_plan.variables),
            component_q2_terms=len(component_plan.q2),
            component_treewidth_width=target["component_width"],
            component_cutset_size=target["cutset_size"],
            schur_scale_half_pow2=target["schur_scale"],
            residual_var_count=0,
            residual_q2_terms=0,
            residual_min_fill_width=0,
            residual_factor_count=0,
            greedy_plan_wall_time_s=None,
            greedy_plan_width=None,
            greedy_plan_nslices=None,
            greedy_plan_log2_max_tensor_size=None,
            greedy_plan_log10_flops=None,
            exact_residual_wall_time_s=None,
            exact_residual_abs_value=None,
            tensor_wall_time_s=None,
            tensor_abs_value=None,
            tensor_abs_error=None,
            tensor_relative_error=None,
            status="schur_solved",
        )

    plan_info = _greedy_tensor_plan(residual_q)

    exact_residual_wall_time_s = None
    exact_residual_abs_value = None
    tensor_wall_time_s = None
    tensor_abs_value = None
    tensor_abs_error = None
    tensor_relative_error = None
    status = "planned"

    if residual_q.n <= int(run_contraction_up_to_vars):
        start = time.perf_counter()
        exact_residual = _scaled_to_complex(
            _sum_q3_free_component_scaled(
                residual_q,
                allow_schur_complement=False,
                allow_tensor_contraction=False,
            )
        )
        exact_residual_wall_time_s = time.perf_counter() - start
        exact_residual_abs_value = float(abs(exact_residual))

        start = time.perf_counter()
        tensor_total = _sum_via_tensor_contraction(residual_q)
        tensor_wall_time_s = time.perf_counter() - start
        tensor_abs_value = float(abs(tensor_total))
        tensor_abs_error = float(abs(tensor_total - exact_residual))
        tensor_relative_error = float(
            tensor_abs_error / max(abs(exact_residual), abs(tensor_total), 1e-300)
        )
        status = "contracted"

    return BenchmarkRow(
        case=case,
        cycles=cycles,
        seed=seed,
        n_qubits=spec.num_qubits,
        n_edges=len(spec.edges),
        gate_count=len(circuit.gates),
        full_amplitude_wall_time_s=full_amplitude_wall_time_s,
        full_amplitude_log2_abs=full_amplitude_log2_abs,
        full_amplitude_phase3_backend=full_amplitude_phase3_backend,
        component_index=target["index"],
        component_backend=component_plan.backend,
        component_prefer_cutset=bool(component_plan.prefer_cutset_backend),
        component_var_count=len(component_plan.variables),
        component_q2_terms=len(component_plan.q2),
        component_treewidth_width=target["component_width"],
        component_cutset_size=target["cutset_size"],
        schur_scale_half_pow2=target["schur_scale"],
        residual_var_count=int(residual_q.n),
        residual_q2_terms=int(len(residual_q.q2)),
        residual_min_fill_width=int(target["residual_width"]),
        residual_factor_count=int(target["residual_factors"]),
        greedy_plan_wall_time_s=plan_info["wall_time_s"],
        greedy_plan_width=plan_info["width"],
        greedy_plan_nslices=plan_info["nslices"],
        greedy_plan_log2_max_tensor_size=plan_info["log2_max_tensor_size"],
        greedy_plan_log10_flops=plan_info["log10_flops"],
        exact_residual_wall_time_s=exact_residual_wall_time_s,
        exact_residual_abs_value=exact_residual_abs_value,
        tensor_wall_time_s=tensor_wall_time_s,
        tensor_abs_value=tensor_abs_value,
        tensor_abs_error=tensor_abs_error,
        tensor_relative_error=tensor_relative_error,
        status=status,
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
    parser.add_argument(
        "--cycles",
        type=int,
        nargs="+",
        default=None,
        help="One or more RCS cycle counts. Defaults to the case headline.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--run-contraction-up-to-vars",
        type=int,
        default=64,
        help="Run the actual tensor contraction only when the Schur residual has at most this many vars.",
    )
    parser.add_argument(
        "--include-full-amplitude",
        action="store_true",
        help="Also time the full non-quimb amplitude baseline for each row.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=RESULTS_ROOT / "amplitude_post_elimination_tensor_rcs.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _spec, default_cycles = case_spec(args.case)
    cycles_list = args.cycles or [default_cycles]
    rows = [
        run_case(
            args.case,
            cycles,
            seed=args.seed,
            run_contraction_up_to_vars=args.run_contraction_up_to_vars,
            include_full_amplitude=args.include_full_amplitude,
        )
        for cycles in cycles_list
    ]
    for row in rows:
        print(
            f"{row.case} cycles={row.cycles}: component={row.component_backend}"
            f" vars={row.component_var_count}, residual={row.residual_var_count} vars/{row.residual_q2_terms} q2,"
            f" greedy_width={row.greedy_plan_width}, log2_max={row.greedy_plan_log2_max_tensor_size:.1f},"
            f" log10_flops={row.greedy_plan_log10_flops:.1f}, status={row.status}"
        )
        if row.tensor_wall_time_s is not None:
            print(
                f"  tensor={row.tensor_wall_time_s:.6f}s vs exact={row.exact_residual_wall_time_s:.6f}s,"
                f" rel_error={row.tensor_relative_error:.3e}"
            )
    write_rows(rows, args.csv)
    print(f"Wrote {len(rows)} row(s) to {args.csv}")


if __name__ == "__main__":
    main()
