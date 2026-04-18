import argparse
import json
import math
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import rustworkx as rx
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import iSwapGate

sys.path.insert(0, "src")

from terket import compute_circuit_amplitude_scaled, normalize_circuit  # noqa: E402
import terket.engine as engine  # noqa: E402


SQRT_ISWAP = iSwapGate().power(0.5)


def grid_spec_53():
    graph = nx.grid_2d_graph(6, 9)
    nodes = sorted(graph.nodes())
    graph.remove_node(nodes[-1])
    nodes = sorted(graph.nodes())
    mapping = {node: index for index, node in enumerate(nodes)}
    relabeled = nx.relabel_nodes(graph, mapping, copy=True)
    edges = tuple(
        sorted((min(int(l), int(r)), max(int(l), int(r))) for l, r in relabeled.edges())
    )
    return relabeled.number_of_nodes(), edges


def edge_layers(num_qubits, edges):
    graph = rx.PyGraph(multigraph=False)
    graph.add_nodes_from(range(num_qubits))
    for left, right in edges:
        if left != right and not graph.has_edge(left, right):
            graph.add_edge(left, right, None)
    color_map = rx.graph_greedy_edge_color(graph)
    by_color = {}
    for edge_index, color in color_map.items():
        left, right = graph.get_edge_endpoints_by_index(edge_index)
        by_color.setdefault(int(color), []).append((int(left), int(right)))
    return tuple(tuple(by_color[color]) for color in sorted(by_color))


def build_sycamore53_rcs(cycles: int = 8, seed: int = 7):
    num_qubits, edges = grid_spec_53()
    layers = edge_layers(num_qubits, list(edges))
    rng = random.Random(seed)
    qc = QuantumCircuit(num_qubits)
    for cycle in range(cycles):
        for qubit in range(num_qubits):
            if rng.randrange(2) == 0:
                qc.rx(math.pi / 2.0, qubit)
            else:
                qc.ry(math.pi / 2.0, qubit)
        for left, right in layers[cycle % len(layers)]:
            qc.append(SQRT_ISWAP, [left, right])
    transpiled = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
    return normalize_circuit(transpiled, rz_compile_mode="dyadic")


class CoarseProfiler:
    def __init__(self, module, names):
        self.module = module
        self.names = tuple(names)
        self.originals = {}
        self.elapsed = defaultdict(float)
        self.calls = Counter()

    def __enter__(self):
        for name in self.names:
            original = getattr(self.module, name)
            self.originals[name] = original

            def make_wrapper(fn_name, fn):
                def wrapper(*args, **kwargs):
                    start = time.perf_counter()
                    try:
                        return fn(*args, **kwargs)
                    finally:
                        self.calls[fn_name] += 1
                        self.elapsed[fn_name] += time.perf_counter() - start

                return wrapper

            setattr(self.module, name, make_wrapper(name, original))
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, original in self.originals.items():
            setattr(self.module, name, original)
        return False

    def summary(self):
        rows = []
        total = sum(self.elapsed.values())
        for name in sorted(self.elapsed, key=self.elapsed.get, reverse=True):
            elapsed = self.elapsed[name]
            calls = self.calls[name]
            rows.append(
                {
                    "function": name,
                    "seconds": elapsed,
                    "calls": calls,
                    "avg_ms": (elapsed * 1000.0 / calls) if calls else 0.0,
                    "share": (elapsed / total) if total else 0.0,
                }
            )
        return rows


def install_old_way_monkeypatch():
    original = engine._optimize_phase_function_structure

    def no_op(q, context=None):
        return q, False

    engine._optimize_phase_function_structure = no_op
    return original


def install_timeout(seconds: int | None):
    if not seconds:
        return None
    timed_out = {"value": False}

    def _fire_timeout():
        timed_out["value"] = True
        import _thread

        _thread.interrupt_main()

    timer = threading.Timer(seconds, _fire_timeout)
    timer.daemon = True
    timer.start()
    return timer, timed_out, seconds


def clear_timeout(previous):
    if previous is None:
        return
    timer, _timed_out, _seconds = previous
    timer.cancel()


def run_case(mode: str, timeout_s: int | None, out_path: Path):
    os.environ["TERKET_DISABLE_QUIMB"] = "1"
    circuit = build_sycamore53_rcs(cycles=8, seed=7)
    gate_counts = Counter(gate[0] for gate in circuit.gates)

    monkeypatch_original = None
    if mode == "old":
        monkeypatch_original = install_old_way_monkeypatch()

    profile_names = (
        "_optimize_phase_function_structure",
        "_build_q3_free_execution_plan",
        "_evaluate_q3_free_execution_plan_scaled",
        "_plan_q3_free_constraint_components",
        "_q3_free_one_shot_cutset_conditioning_plan",
        "_q3_free_cutset_conditioning_plan",
        "_evaluate_q3_free_component_plan_scaled",
        "_evaluate_q3_free_cutset_conditioning_plan_scaled",
        "_evaluate_q3_free_constraint_plan_scaled",
        "_evaluate_q3_free_raw_constraint_plan_scaled",
    )

    timeout_token = None
    started = time.perf_counter()
    status = "ok"
    result = None
    info = None
    error = None

    try:
        with CoarseProfiler(engine, profile_names) as profiler:
            timeout_token = install_timeout(timeout_s)
            try:
                result, info = compute_circuit_amplitude_scaled(
                    circuit,
                    [0] * 53,
                    [0] * 53,
                    allow_tensor_contraction=False,
                )
            except KeyboardInterrupt as exc:
                if timeout_token is not None and timeout_token[1]["value"]:
                    raise TimeoutError(f"timed out after {timeout_token[2]}s") from exc
                raise
            finally:
                clear_timeout(timeout_token)
        profile_rows = profiler.summary()
    except TimeoutError as exc:
        status = "timeout"
        error = str(exc)
        profile_rows = profiler.summary()
    except Exception as exc:  # pragma: no cover - diagnostic path
        status = "error"
        error = repr(exc)
        profile_rows = profiler.summary() if "profiler" in locals() else []
    finally:
        if monkeypatch_original is not None:
            engine._optimize_phase_function_structure = monkeypatch_original

    elapsed = time.perf_counter() - started
    payload = {
        "mode": mode,
        "status": status,
        "elapsed_s": elapsed,
        "timeout_s": timeout_s,
        "error": error,
        "gate_counts": dict(gate_counts),
        "total_gates": len(circuit.gates),
        "profile": profile_rows,
    }
    if result is not None:
        payload["log2_abs"] = result.log2_abs()
    if info is not None:
        payload["info"] = info

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("new", "old"), required=True)
    parser.add_argument("--timeout-s", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_case(args.mode, args.timeout_s, args.out)


if __name__ == "__main__":
    main()
