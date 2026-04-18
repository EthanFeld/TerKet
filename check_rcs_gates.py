"""Check gate compositions in the RCS circuit to understand optimization opportunities."""
import sys
sys.path.insert(0, 'src')
import math, random, collections
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import iSwapGate
import networkx as nx
import rustworkx as rx

SQRT_ISWAP = iSwapGate().power(0.5)

def build_grid_spec(rows, cols, drop_last=False):
    graph = nx.grid_2d_graph(rows, cols)
    nodes = sorted(graph.nodes())
    if drop_last:
        graph.remove_node(nodes[-1])
        nodes = sorted(graph.nodes())
    mapping = {node: index for index, node in enumerate(nodes)}
    relabeled = nx.relabel_nodes(graph, mapping, copy=True)
    edges = tuple(sorted((min(int(l), int(r)), max(int(l), int(r))) for l, r in relabeled.edges()))
    return relabeled.number_of_nodes(), edges

def build_edge_layers(n_qubits, edges):
    graph = rx.PyGraph(multigraph=False)
    graph.add_nodes_from(range(n_qubits))
    for left, right in edges:
        if left == right or graph.has_edge(left, right): continue
        graph.add_edge(left, right, None)
    color_map = rx.graph_greedy_edge_color(graph)
    by_color = {}
    for edge_index, color in color_map.items():
        left, right = graph.get_edge_endpoints_by_index(edge_index)
        by_color.setdefault(int(color), []).append((int(left), int(right)))
    return tuple(tuple(by_color[c]) for c in sorted(by_color))

def build_rcs(n_qubits, edges, cycles, seed):
    rng = random.Random(seed)
    layers = build_edge_layers(n_qubits, list(edges))
    qc = QuantumCircuit(n_qubits)
    for cycle in range(cycles):
        for qubit in range(n_qubits):
            if rng.randrange(2) == 0:
                qc.rx(math.pi / 2.0, qubit)
            else:
                qc.ry(math.pi / 2.0, qubit)
        for left, right in layers[cycle % len(layers)]:
            qc.append(SQRT_ISWAP, [left, right])
    return qc

n_qubits, edges = build_grid_spec(3, 3)
for cycles in [2, 4, 8]:
    qc = build_rcs(n_qubits, edges, cycles, seed=7)
    transpiled = transpile(qc, basis_gates=['rz', 'sx', 'x', 'cx'], optimization_level=1)
    qiskit_counts = collections.Counter(op.operation.name for op in transpiled.data)

    from terket.circuit_spec import normalize_circuit
    spec = normalize_circuit(transpiled, rz_compile_mode='dyadic')
    terket_counts = collections.Counter(gate[0] for gate in spec.gates)

    print(f"\n=== 3x3 grid, {cycles} cycles ===")
    print(f"After Qiskit transpile: {dict(qiskit_counts)}")
    print(f"After TerKet normalize: {dict(terket_counts)}")
    print(f"H gates: {terket_counts.get('h', 0)}, SX gates: {terket_counts.get('sx', 0)}")
    print(f"Total gates: {len(spec.gates)}, H-var gates: {terket_counts.get('h', 0) + terket_counts.get('sx', 0)}")
