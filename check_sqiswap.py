"""Analyze the sqrt_iSWAP decomposition and pattern of gates around SX."""
import sys
sys.path.insert(0, 'src')
import math, collections
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import iSwapGate

SQRT_ISWAP = iSwapGate().power(0.5)

# Look at a single sqrt_iSWAP gate
qc = QuantumCircuit(2)
qc.append(SQRT_ISWAP, [0, 1])
transpiled = transpile(qc, basis_gates=['rz', 'sx', 'x', 'cx'], optimization_level=0)
print("sqrt_iSWAP decomposition (no opt):")
for gate in transpiled.data:
    op = gate.operation
    qubits = [transpiled.find_bit(q).index for q in gate.qubits]
    if op.params:
        print(f"  {op.name}({[float(p) for p in op.params]}) q{qubits}")
    else:
        print(f"  {op.name} q{qubits}")

print()

# Also check with optimization level 1
transpiled_opt = transpile(qc, basis_gates=['rz', 'sx', 'x', 'cx'], optimization_level=1)
print("sqrt_iSWAP decomposition (opt level 1):")
for gate in transpiled_opt.data:
    op = gate.operation
    qubits = [transpiled_opt.find_bit(q).index for q in gate.qubits]
    if op.params:
        print(f"  {op.name}({[round(float(p)/math.pi, 4) for p in op.params]}*pi) q{qubits}")
    else:
        print(f"  {op.name} q{qubits}")

# Now look at what TerKet does with this decomposition
print()
from terket.circuit_spec import normalize_circuit
spec = normalize_circuit(transpiled_opt, rz_compile_mode='dyadic')
print("After TerKet normalize:")
for gate in spec.gates:
    print(f"  {gate}")
print()
# Look at gate counts and how many fresh vars are introduced
sx_count = sum(1 for g in spec.gates if g[0] == 'sx')
h_count = sum(1 for g in spec.gates if g[0] == 'h')
print(f"SX gates: {sx_count}, H gates: {h_count}")
print(f"Fresh variables needed: {sx_count + h_count}")
