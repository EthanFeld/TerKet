"""Depth-scaling head-to-head benchmark for TerKet versus quimb."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

from qiskit import QuantumCircuit


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
RESULTS_ROOT = REPO_ROOT / "results"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket import analyze_circuit, compute_circuit_amplitude, normalize_circuit
from terket.benchmarking import (
    count_t_gates,
    measure_callable,
    quimb_amplitude,
    runtime_versions,
    warm_up_quimb,
    warm_up_terket,
    write_rows,
)
from terket.head_to_head_cases import _append_multi_controlled_z_ladder, _grover_logical_qubits, transpile_to_supported_basis


@dataclass
class DepthScalingRow:
    case: str
    family: str
    n_qubits: int
    depth_parameter: int
    gate_count: int
    t_count: int
    cubic_obstruction: int
    gauss_obstruction: int
    cost_model_r: int
    terket_phase3_backend: str
    terket_wall_time_s: float
    terket_peak_rss_mb: float
    quimb_wall_time_s: float
    quimb_peak_rss_mb: float
    abs_error: float
    relative_error: float
    quimb_over_terket_time_ratio: float
    python_version: str
    numpy_version: str
    qiskit_version: str
    quimb_version: str
    cotengra_version: str


def build_toffoli_depth_case(n_qubits: int, rounds: int):
    qc = QuantumCircuit(n_qubits, name=f"toffoli_depth_{n_qubits}_{rounds}")
    for _ in range(rounds):
        for idx in range(n_qubits - 2):
            qc.ccx(idx, idx + 1, idx + 2)
    circuit = transpile_to_supported_basis(qc, optimization_level=0)
    zero_bits = (0,) * n_qubits
    return circuit, zero_bits, zero_bits


def build_magic_rounds_case_supported(distance: int, rounds: int):
    data_qubits = distance
    ancillas = max(1, distance - 1)
    full = QuantumCircuit(data_qubits + ancillas, name=f"magic_rounds_{distance}_{rounds}")
    for _ in range(rounds):
        for qubit in range(data_qubits):
            full.h(qubit)
        for ancilla in range(ancillas):
            ancilla_idx = data_qubits + ancilla
            full.cx(ancilla, ancilla_idx)
            full.cx((ancilla + 1) % data_qubits, ancilla_idx)
            full.t(ancilla_idx)
            full.cx(ancilla, ancilla_idx)
            full.cx((ancilla + 1) % data_qubits, ancilla_idx)
    circuit = transpile_to_supported_basis(full, optimization_level=0)
    zero_bits = (0,) * circuit.n_qubits
    return circuit, zero_bits, zero_bits


def build_grover_depth_case(n_qubits: int, iterations: int):
    logical_qubits = _grover_logical_qubits(n_qubits)
    search_qubits = list(range(logical_qubits))
    ancillas = list(range(logical_qubits, n_qubits))
    qc = QuantumCircuit(n_qubits, name=f"grover_depth_{n_qubits}_{iterations}")
    for _ in range(iterations):
        _append_multi_controlled_z_ladder(qc, search_qubits, ancillas)
        for qubit in search_qubits:
            qc.h(qubit)
            qc.x(qubit)
        _append_multi_controlled_z_ladder(qc, search_qubits, ancillas)
        for qubit in search_qubits:
            qc.x(qubit)
            qc.h(qubit)
    circuit = transpile_to_supported_basis(qc, optimization_level=0)
    input_bits = tuple((idx % 2) for idx in range(n_qubits))
    output_bits = input_bits
    return circuit, input_bits, output_bits


def run_case(name: str, family: str, depth_parameter: int, circuit, input_bits, output_bits, *, quimb_optimize: str):
    spec = normalize_circuit(circuit)
    analysis = analyze_circuit(spec, input_bits, output_bits)
    t_count = count_t_gates(spec)

    terket_wall_time_s, terket_peak_rss_mb, terket_result = measure_callable(
        lambda: compute_circuit_amplitude(spec, input_bits, output_bits, as_complex=True),
        1,
    )
    terket_amp, _ = terket_result
    quimb_wall_time_s, quimb_peak_rss_mb, quimb_amp = measure_callable(
        lambda: quimb_amplitude(spec, input_bits, output_bits, optimize=quimb_optimize),
        1,
    )
    terket_amp = complex(terket_amp)
    quimb_amp = complex(quimb_amp)
    abs_error = abs(terket_amp - quimb_amp)
    relative_error = abs_error / max(abs(terket_amp), abs(quimb_amp), 1e-300)
    versions = runtime_versions()

    return DepthScalingRow(
        case=name,
        family=family,
        n_qubits=spec.n_qubits,
        depth_parameter=depth_parameter,
        gate_count=len(spec.gates),
        t_count=t_count,
        cubic_obstruction=int(analysis["cubic_obstruction"]),
        gauss_obstruction=int(analysis["gauss_obstruction"]),
        cost_model_r=int(analysis["cost_model_r"]),
        terket_phase3_backend=str(analysis.get("phase3_backend") or ""),
        terket_wall_time_s=float(terket_wall_time_s),
        terket_peak_rss_mb=float(terket_peak_rss_mb),
        quimb_wall_time_s=float(quimb_wall_time_s),
        quimb_peak_rss_mb=float(quimb_peak_rss_mb),
        abs_error=float(abs_error),
        relative_error=float(relative_error),
        quimb_over_terket_time_ratio=float(quimb_wall_time_s / max(terket_wall_time_s, 1e-12)),
        python_version=versions["python_version"],
        numpy_version=versions["numpy_version"],
        qiskit_version=versions["qiskit_version"],
        quimb_version=versions["quimb_version"],
        cotengra_version=versions["cotengra_version"],
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quimb-optimize", default="auto-hq")
    parser.add_argument(
        "--csv",
        type=Path,
        default=RESULTS_ROOT / "depth_scaling_head_to_head.csv",
    )
    parser.add_argument("--toffoli-qubits", type=int, default=32)
    parser.add_argument("--magic-distance", type=int, default=32)
    parser.add_argument("--grover-qubits", type=int, default=24)
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 4, 8])
    return parser.parse_args()


def main():
    args = parse_args()
    warm_up_terket()
    warm_up_quimb(args.quimb_optimize)

    rows: list[DepthScalingRow] = []
    for depth in args.depths:
        circuit, input_bits, output_bits = build_toffoli_depth_case(args.toffoli_qubits, depth)
        rows.append(
            run_case(
                f"toffoli_depth_{args.toffoli_qubits}_{depth}",
                "toffoli_depth",
                depth,
                circuit,
                input_bits,
                output_bits,
                quimb_optimize=args.quimb_optimize,
            )
        )

        circuit, input_bits, output_bits = build_magic_rounds_case_supported(args.magic_distance, depth)
        rows.append(
            run_case(
                f"magic_rounds_{args.magic_distance}_{depth}",
                "magic_rounds",
                depth,
                circuit,
                input_bits,
                output_bits,
                quimb_optimize=args.quimb_optimize,
            )
        )

        circuit, input_bits, output_bits = build_grover_depth_case(args.grover_qubits, depth)
        rows.append(
            run_case(
                f"grover_depth_{args.grover_qubits}_{depth}",
                "grover_depth",
                depth,
                circuit,
                input_bits,
                output_bits,
                quimb_optimize=args.quimb_optimize,
            )
        )

    for row in rows:
        print(
            f"{row.case}: TerKet={row.terket_wall_time_s:.6f}s, "
            f"quimb={row.quimb_wall_time_s:.6f}s, "
            f"ratio={row.quimb_over_terket_time_ratio:.3f}x, "
            f"backend={row.terket_phase3_backend or 'q3_free'}"
        )

    write_rows(rows, args.csv)
    print(f"Wrote {len(rows)} row(s) to {args.csv}")


if __name__ == "__main__":
    main()
