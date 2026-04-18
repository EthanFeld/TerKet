import sys
sys.path.insert(0, 'src')
import math, random, collections
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import iSwapGate
import networkx as nx
import rustworkx as rx

SQRT_ISWAP = iSwapGate().power(0.5)

graph = nx.grid_2d_graph(3, 3)
nodes = sorted(graph.nodes())
mapping = {node: index for index, node in enumerate(nodes)}
relabeled = nx.relabel_nodes(graph, mapping, copy=True)
edges = tuple(sorted((min(int(l), int(r)), max(int(l), int(r))) for l, r in relabeled.edges()))
n_qubits = 9

graph2 = rx.PyGraph(multigraph=False)
graph2.add_nodes_from(range(n_qubits))
for left, right in edges:
    if left == right or graph2.has_edge(left, right): continue
    graph2.add_edge(left, right, None)
color_map = rx.graph_greedy_edge_color(graph2)
by_color = {}
for edge_index, color in color_map.items():
    left, right = graph2.get_edge_endpoints_by_index(edge_index)
    by_color.setdefault(int(color), []).append((int(left), int(right)))
layers = tuple(tuple(by_color[c]) for c in sorted(by_color))

rng = random.Random(7)
qc = QuantumCircuit(n_qubits)
for cycle in range(2):
    for qubit in range(n_qubits):
        if rng.randrange(2) == 0:
            qc.rx(math.pi / 2.0, qubit)
        else:
            qc.ry(math.pi / 2.0, qubit)
    for left, right in layers[cycle % len(layers)]:
        qc.append(SQRT_ISWAP, [left, right])

transpiled = transpile(qc, basis_gates=['rz', 'sx', 'x', 'cx'], optimization_level=1)
counts = collections.Counter(op.operation.name for op in transpiled.data)
print('Gate counts (3x3, 2 cycles, basis=[rz,sx,x,cx]):', dict(counts))

# Also check what terket circuit_spec produces
from terket.circuit_spec import normalize_circuit
spec = normalize_circuit(transpiled, rz_compile_mode='dyadic')
spec_counts = collections.Counter(gate[0] for gate in spec.gates)
print('TerKet gate counts:', dict(spec_counts))
print('Total TerKet gates:', len(spec.gates))

# Check how many H gates appear
print('H gates in TerKet spec:', spec_counts.get('h', 0))
