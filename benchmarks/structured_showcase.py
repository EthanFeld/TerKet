"""Structured showcase benchmark for TerKet hidden-shift circuits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
RESULTS_ROOT = REPO_ROOT / "results"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket import analyze_circuit, compute_circuit_amplitude, compute_circuit_amplitude_scaled, normalize_circuit
from terket.benchmarking import (
    StructuredShowcaseRow,
    count_t_gates,
    runtime_versions,
    scaled_amplitude_fields,
    warm_up_terket,
    write_rows,
)
from terket.circuits import bits_to_big_endian_string
from terket.benchmarking.structured_cases import CASES, SUITES, resolve_cases


def run_case(case) -> StructuredShowcaseRow:
    query = case.build_query()
    spec = normalize_circuit(query.circuit)
    analysis = analyze_circuit(spec, query.input_bits, query.output_bits)
    start = time.perf_counter()
    scaled_amp, _ = compute_circuit_amplitude_scaled(spec, query.input_bits, query.output_bits)
    wall_time_s = time.perf_counter() - start
    wrong_amp, _ = compute_circuit_amplitude(spec, query.input_bits, query.wrong_output_bits, as_complex=True)
    (
        target_real,
        target_imag,
        target_log2_abs,
        mantissa_real,
        mantissa_imag,
        half_pow2_exp,
    ) = scaled_amplitude_fields(scaled_amp)
    versions = runtime_versions()

    return StructuredShowcaseRow(
        case=case.name,
        family=case.family,
        size=case.size,
        n_qubits=spec.n_qubits,
        gate_count=len(spec.gates),
        t_count=count_t_gates(spec),
        cubic_obstruction=int(analysis["cubic_obstruction"]),
        cost_model_r=int(analysis["cost_model_r"]),
        terket_phase3_backend=str(analysis.get("phase3_backend") or ""),
        wall_time_s=float(wall_time_s),
        target_bits=bits_to_big_endian_string(query.output_bits),
        target_amplitude_real=target_real,
        target_amplitude_imag=target_imag,
        target_log2_abs=target_log2_abs,
        target_scaled_mantissa_real=mantissa_real,
        target_scaled_mantissa_imag=mantissa_imag,
        target_scaled_half_pow2_exp=half_pow2_exp,
        wrong_bits=bits_to_big_endian_string(query.wrong_output_bits),
        wrong_abs=float(abs(complex(wrong_amp))),
        python_version=versions["python_version"],
        numpy_version=versions["numpy_version"],
        qiskit_version=versions["qiskit_version"],
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        default="large",
        choices=sorted(SUITES),
        help="Named structured showcase suite. Ignored when --case is provided.",
    )
    parser.add_argument(
        "--case",
        nargs="+",
        default=None,
        help=f"Explicit structured showcase case list. Known cases: {', '.join(sorted(CASES))}.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=RESULTS_ROOT / "structured_showcase.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cases = resolve_cases(args.case, suite=args.suite)
    warm_up_terket()

    rows = [run_case(case) for case in cases]
    for row in rows:
        print(
            f"{row.case}: TerKet={row.wall_time_s:.6f}s, "
            f"log2|amp|={row.target_log2_abs:.6f}, "
            f"wrong_abs={row.wrong_abs:.3e}, "
            f"backend={row.terket_phase3_backend or 'q3_free'}"
        )

    write_rows(rows, args.csv)
    print(f"Wrote {len(rows)} row(s) to {args.csv}")


if __name__ == "__main__":
    main()
