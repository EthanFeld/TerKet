from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import cirq
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket import compute_circuit_amplitude, compute_circuit_probability
from terket.circuit_spec import from_qiskit


def _line_pairs(qubits, parity: int):
    return [(qubits[idx], qubits[idx + 1]) for idx in range(parity, len(qubits) - 1, 2)]


def _grid_qubits(rows: int, cols: int):
    return [cirq.GridQubit(r, c) for r in range(rows) for c in range(cols)]


def _grid_pairs(rows: int, cols: int, parity: int):
    pairs = []
    for r in range(rows):
        for c in range((r + parity) % 2, cols - 1, 2):
            pairs.append((cirq.GridQubit(r, c), cirq.GridQubit(r, c + 1)))
    for c in range(cols):
        for r in range((c + parity) % 2, rows - 1, 2):
            pairs.append((cirq.GridQubit(r, c), cirq.GridQubit(r + 1, c)))
    return pairs


def build_echo_circuit(
    *,
    n_qubits: int,
    depth: int,
    seed: int,
    layout: str,
    rows: int,
    cols: int,
) -> cirq.Circuit:
    if layout == "line":
        qubits = cirq.LineQubit.range(n_qubits)
    elif layout == "grid":
        if rows * cols != n_qubits:
            raise ValueError("For grid layout, rows * cols must equal qubits.")
        qubits = _grid_qubits(rows, cols)
    else:
        raise ValueError(f"Unsupported layout {layout!r}.")

    rng = __import__("random").Random(seed)
    forward = cirq.Circuit()
    one_qubit_ops = (
        lambda q: cirq.H(q),
        lambda q: cirq.S(q),
        lambda q: cirq.T(q),
        lambda q: cirq.X(q),
        lambda q: cirq.Y(q),
        lambda q: cirq.Z(q),
        lambda q: cirq.rx(rng.choice([0.125, 0.25, 0.5, -0.25]) * math.pi).on(q),
        lambda q: cirq.ry(rng.choice([0.125, 0.25, 0.5, -0.25]) * math.pi).on(q),
        lambda q: cirq.rz(rng.choice([0.125, 0.25, 0.5, -0.25]) * math.pi).on(q),
    )

    for layer in range(depth):
        for q in qubits:
            forward.append(rng.choice(one_qubit_ops)(q))
        if layout == "line":
            for qa, qb in _line_pairs(qubits, layer % 2):
                forward.append(cirq.CZ(qa, qb))
        else:
            for qa, qb in _grid_pairs(rows, cols, layer % 2):
                forward.append(cirq.CZ(qa, qb))

    return forward + cirq.inverse(forward)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Temporary bridge: build a demo Cirq echo circuit, export QASM, and try TerKet.",
    )
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", choices=("line", "grid"), default="line")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=("amplitude", "amplitude_and_probability"),
        default="amplitude",
    )
    parser.add_argument(
        "--qasm-out",
        type=Path,
        default=REPO_ROOT / "results" / "cirq_echo_demo.qasm",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    circuit = build_echo_circuit(
        n_qubits=args.qubits,
        depth=args.depth,
        seed=args.seed,
        layout=args.layout,
        rows=args.rows,
        cols=args.cols,
    )
    t1 = time.perf_counter()
    qasm_text = cirq.qasm(circuit)
    args.qasm_out.parent.mkdir(parents=True, exist_ok=True)
    args.qasm_out.write_text(qasm_text, encoding="utf-8")
    t2 = time.perf_counter()
    qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_text)
    spec = from_qiskit(qiskit_circuit)
    t3 = time.perf_counter()
    zero_bits = [0] * spec.n_qubits
    amplitude, amp_info = compute_circuit_amplitude(spec, zero_bits, zero_bits, as_complex=True)
    t4 = time.perf_counter()
    probability = None
    prob_info = None
    t5 = t4
    if args.mode == "amplitude_and_probability":
        probability, prob_info = compute_circuit_probability(spec, zero_bits, zero_bits, as_float=True)
        t5 = time.perf_counter()

    print(f"wrote_qasm={args.qasm_out}")
    print(f"layout={args.layout}")
    print(f"build_s={t1 - t0:.3f}")
    print(f"qasm_s={t2 - t1:.3f}")
    print(f"import_s={t3 - t2:.3f}")
    print(f"amplitude_s={t4 - t3:.3f}")
    if args.mode == "amplitude_and_probability":
        print(f"probability_s={t5 - t4:.3f}")
    print(f"cirq_ops={len(list(circuit.all_operations()))}")
    print(f"qiskit_ops={len(qiskit_circuit.data)}")
    print(f"terket_qubits={spec.n_qubits}")
    print(f"terket_gates={len(spec.gates)}")
    print(f"amplitude_0_to_0={amplitude!r}")
    print(f"amp_backend={amp_info.get('phase3_backend')!r}")
    if args.mode == "amplitude_and_probability":
        print(f"probability_0_to_0={probability!r}")
        print(f"prob_backend={prob_info.get('phase3_backend')!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
