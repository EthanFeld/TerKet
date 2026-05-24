"""Sequential MQT Bench sample benchmark for TerKet versus quimb."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.targeted.mqt.mqt_head_to_head_core import (
    DEFAULT_SAMPLE,
    MqtBenchRow,
    RESULTS_ROOT,
    parse_sample_args,
    run_case,
)
from terket.benchmarking import write_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        nargs="*",
        default=None,
        help="Benchmark sample in name:size form. Defaults to a mixed MQT Bench sample.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Timing repeats per backend.")
    parser.add_argument("--mqt-level", choices=["alg", "indep"], default="alg", help="MQT Bench circuit level.")
    parser.add_argument("--opt-level", type=int, default=0, help="MQT Bench target-independent optimization level.")
    parser.add_argument("--quimb-optimize", default="auto-hq", help="Optimizer passed to quimb.")
    parser.add_argument("--min-available-memory-mb", type=float, default=2048.0, help="Skip case below free-memory floor.")
    parser.add_argument("--max-interaction-width", type=int, default=18, help="Skip case above interaction min-fill width.")
    parser.add_argument("--max-quimb-width", type=float, default=20.0, help="Skip case above rank-simplified quimb width.")
    parser.add_argument(
        "--max-quimb-log2-tensor-size",
        type=float,
        default=24.0,
        help="Skip case above rank-simplified quimb log2 max tensor size.",
    )
    parser.add_argument("--terket-only", action="store_true", help="Skip quimb rehearsal and quimb amplitude.")
    parser.add_argument("--profile-dir", type=Path, default=None, help="Optional directory for cProfile dumps.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=RESULTS_ROOT / "mqt_bench_head_to_head.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = parse_sample_args(args.sample or [])
    rows: list[MqtBenchRow] = []

    for benchmark, circuit_size in sample:
        row = run_case(
            benchmark,
            circuit_size,
            quimb_optimize=args.quimb_optimize,
            repeats=args.repeats,
            mqt_level=args.mqt_level,
            opt_level=args.opt_level,
            min_available_memory_mb=args.min_available_memory_mb,
            max_interaction_width=args.max_interaction_width,
            max_quimb_width=args.max_quimb_width,
            max_quimb_log2_tensor_size=args.max_quimb_log2_tensor_size,
            terket_only=bool(args.terket_only),
            profile_dir=args.profile_dir,
        )
        rows.append(row)
        write_rows(rows, args.csv)
        print(
            f"{benchmark}:{circuit_size} status={row.status} "
            f"n={row.n_qubits} terket_backend={row.terket_phase3_backend or 'n/a'} "
            f"terket_r={row.terket_cost_model_r} quimb_width={row.quimb_contraction_width} "
            f"time_ratio={row.quimb_over_terket_time_ratio} rss_ratio={row.quimb_over_terket_peak_rss_ratio}"
        )
        if row.status != "ok":
            print(f"  error={row.error_type}: {row.error_message}")

    print(f"Wrote {len(rows)} row(s) to {args.csv}")


if __name__ == "__main__":
    main()
