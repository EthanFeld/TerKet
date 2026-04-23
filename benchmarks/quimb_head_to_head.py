"""Head-to-head benchmark for TerKet versus quimb."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
RESULTS_ROOT = REPO_ROOT / "results"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket import analyze_circuit, compute_circuit_amplitude, normalize_circuit
from terket.benchmarking import (
    HeadToHeadRow,
    count_t_gates,
    measure_callable,
    quimb_amplitude,
    runtime_versions,
    warm_up_quimb,
    warm_up_terket,
    write_rows,
)
from terket.benchmarking.head_to_head_cases import CASES, SUITES, resolve_cases


def run_case(case, *, repeats: int, quimb_optimize: str) -> HeadToHeadRow:
    query = case.build_query()
    spec = normalize_circuit(query.circuit)
    analysis = analyze_circuit(spec, query.input_bits, query.output_bits)
    t_count = count_t_gates(spec)
    obstruction_fraction = 0.0 if t_count == 0 else analysis["cubic_obstruction"] / t_count

    terket_wall_time_s, terket_peak_rss_mb, terket_result = measure_callable(
        lambda: compute_circuit_amplitude(spec, query.input_bits, query.output_bits, as_complex=True),
        repeats,
    )
    terket_amp, _ = terket_result
    quimb_wall_time_s, quimb_peak_rss_mb, quimb_amp = measure_callable(
        lambda: quimb_amplitude(spec, query.input_bits, query.output_bits, optimize=quimb_optimize),
        repeats,
    )
    terket_amp = complex(terket_amp)
    quimb_amp = complex(quimb_amp)
    abs_error = abs(terket_amp - quimb_amp)
    relative_error = abs_error / max(abs(terket_amp), abs(quimb_amp), 1e-300)
    versions = runtime_versions()

    return HeadToHeadRow(
        case=case.name,
        family=case.family,
        size=case.size,
        n_qubits=spec.n_qubits,
        gate_count=len(spec.gates),
        t_count=t_count,
        cubic_obstruction=int(analysis["cubic_obstruction"]),
        gauss_obstruction=int(analysis["gauss_obstruction"]),
        cost_model_r=int(analysis["cost_model_r"]),
        terket_phase3_backend=str(analysis.get("phase3_backend") or ""),
        obstruction_fraction=float(obstruction_fraction),
        repeats=int(repeats),
        quimb_optimize=quimb_optimize,
        terket_wall_time_s=float(terket_wall_time_s),
        terket_peak_rss_mb=float(terket_peak_rss_mb),
        quimb_wall_time_s=float(quimb_wall_time_s),
        quimb_peak_rss_mb=float(quimb_peak_rss_mb),
        terket_real=float(terket_amp.real),
        terket_imag=float(terket_amp.imag),
        quimb_real=float(quimb_amp.real),
        quimb_imag=float(quimb_amp.imag),
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
    parser.add_argument(
        "--suite",
        default="hero",
        choices=sorted(SUITES),
        help="Named case suite. Ignored when --case is provided.",
    )
    parser.add_argument(
        "--case",
        nargs="+",
        default=None,
        help=f"Explicit case list. Known cases: {', '.join(sorted(CASES))}.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Timing repeats per backend.")
    parser.add_argument("--quimb-optimize", default="auto-hq", help="Optimizer passed to quimb.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=RESULTS_ROOT / "quimb_head_to_head.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cases = resolve_cases(args.case, suite=args.suite)
    warm_up_terket()
    warm_up_quimb(args.quimb_optimize)

    rows = [run_case(case, repeats=args.repeats, quimb_optimize=args.quimb_optimize) for case in cases]
    for row in rows:
        print(
            f"{row.case}: TerKet={row.terket_wall_time_s:.6f}s, "
            f"quimb={row.quimb_wall_time_s:.6f}s, "
            f"ratio={row.quimb_over_terket_time_ratio:.3f}x, "
            f"abs_error={row.abs_error:.3e}, "
            f"backend={row.terket_phase3_backend or 'q3_free'}"
        )

    write_rows(rows, args.csv)
    print(f"Wrote {len(rows)} row(s) to {args.csv}")


if __name__ == "__main__":
    main()
