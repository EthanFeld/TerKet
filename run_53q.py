"""Wrapper to run the 53-qubit amplitude benchmark directly."""
import sys
sys.path.insert(0, 'src')

# Replicate what the benchmark does
import math, random, time
from pathlib import Path
import networkx as nx
import rustworkx as rx
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import iSwapGate

from terket import compute_circuit_amplitude_scaled, normalize_circuit

SQRT_ISWAP = iSwapGate().power(0.5)

def grid_spec_53():
    graph = nx.grid_2d_graph(6, 9)
    nodes = sorted(graph.nodes())
    graph.remove_node(nodes[-1])
    nodes = sorted(graph.nodes())
    mapping = {node: index for index, node in enumerate(nodes)}
    relabeled = nx.relabel_nodes(graph, mapping, copy=True)
    edges = tuple(sorted((min(int(l), int(r)), max(int(l), int(r))) for l, r in relabeled.edges()))
    return relabeled.number_of_nodes(), edges

def edge_layers(num_qubits, edges):
    graph = rx.PyGraph(multigraph=False)
    graph.add_nodes_from(range(num_qubits))
    for l, r in edges:
        if l != r and not graph.has_edge(l, r):
            graph.add_edge(l, r, None)
    color_map = rx.graph_greedy_edge_color(graph)
    by_color = {}
    for ei, color in color_map.items():
        l, r = graph.get_edge_endpoints_by_index(ei)
        by_color.setdefault(int(color), []).append((int(l), int(r)))
    return tuple(tuple(by_color[c]) for c in sorted(by_color))

n_qubits, edges = grid_spec_53()
print(f"53-qubit grid: {n_qubits} qubits, {len(edges)} edges")

layers = edge_layers(n_qubits, list(edges))
rng = random.Random(7)
qc = QuantumCircuit(n_qubits)
for cycle in range(8):
    for qubit in range(n_qubits):
        if rng.randrange(2) == 0:
            qc.rx(math.pi / 2.0, qubit)
        else:
            qc.ry(math.pi / 2.0, qubit)
    for l, r in layers[cycle % len(layers)]:
        qc.append(SQRT_ISWAP, [l, r])

print("Transpiling...")
transpiled = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
circuit = normalize_circuit(transpiled, rz_compile_mode="dyadic")
import collections
counts = collections.Counter(g[0] for g in circuit.gates)
print(f"Gate counts: {dict(counts)}")
print(f"Total gates: {len(circuit.gates)}")

print("\nRunning amplitude (allow_tensor_contraction=True)...")
t0 = time.perf_counter()
try:
    amp, info = compute_circuit_amplitude_scaled(
        circuit,
        [0] * n_qubits,
        [0] * n_qubits,
        allow_tensor_contraction=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"Amplitude done in {elapsed:.3f}s")
    print(f"log2|amp|: {amp.log2_abs():.4f}")
    print(f"Info: phase3_backend={info.get('phase3_backend')}, cost_r={info.get('cost_model_r')}")
except Exception as e:
    elapsed = time.perf_counter() - t0
    print(f"Error after {elapsed:.3f}s: {e}")
    import traceback; traceback.print_exc()
