"""Verify sxdg native implementation: correctness and simplification rules."""
import sys
sys.path.insert(0, 'src')
import math
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from terket import compute_circuit_amplitude
from terket.circuit_spec import from_qiskit

# Circuit with sxdg gate
qc = QuantumCircuit(2)
qc.h(0)
qc.sxdg(0)
qc.cx(0, 1)
qc.sxdg(1)

spec = from_qiskit(qc)
statevector = Statevector.from_instruction(qc).data

ok = True
for bits in ((0, 0), (1, 0), (0, 1), (1, 1)):
    idx = sum((b & 1) << i for i, b in enumerate(bits))
    amp, _ = compute_circuit_amplitude(spec, [0, 0], bits, as_complex=True)
    exp = complex(statevector[idx])
    if abs(amp - exp) > 1e-10:
        print(f'MISMATCH bits={bits}: got {amp}, expected {exp}')
        ok = False

print('sxdg amplitude correctness:', 'OK' if ok else 'FAIL')

# Check H*S*H -> sx simplification
qc2 = QuantumCircuit(1)
qc2.h(0)
qc2.s(0)
qc2.h(0)
spec2 = from_qiskit(qc2)
print('H*S*H gates:', [g[0] for g in spec2.gates], '(expected: [sx])')

# Check H*Sdg*H -> sxdg simplification
qc3 = QuantumCircuit(1)
qc3.h(0)
qc3.sdg(0)
qc3.h(0)
spec3 = from_qiskit(qc3)
print('H*Sdg*H gates:', [g[0] for g in spec3.gates], '(expected: [sxdg])')

# Check sxdg*sxdg -> x
qc4 = QuantumCircuit(1)
qc4.sxdg(0)
qc4.sxdg(0)
spec4 = from_qiskit(qc4)
print('sxdg*sxdg gates:', [g[0] for g in spec4.gates], '(expected: [x])')

# Check sx*sxdg -> I
qc5 = QuantumCircuit(1)
qc5.sx(0)
qc5.sxdg(0)
spec5 = from_qiskit(qc5)
print('sx*sxdg gates:', [g[0] for g in spec5.gates], '(expected: [])')

# Check sxdg*sx -> I
qc6 = QuantumCircuit(1)
qc6.sxdg(0)
qc6.sx(0)
spec6 = from_qiskit(qc6)
print('sxdg*sx gates:', [g[0] for g in spec6.gates], '(expected: [])')
