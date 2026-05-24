"""Guarded one-case-at-a-time MQT Bench runner."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
RESULTS_ROOT = REPO_ROOT / "results"
CASE_SCRIPT = REPO_ROOT / "benchmarks" / "targeted" / "mqt" / "mqt_bench_head_to_head.py"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket.benchmarking import default_guarded_rss_limit_mb, run_guarded_subprocess_csv, write_rows

DEFAULT_SAMPLE: tuple[tuple[str, int], ...] = (
    ("ghz", 24),
    ("graphstate", 24),
    ("bv", 24),
    ("dj", 24),
    ("qft", 16),
    ("qftentangled", 16),
    ("grover", 12),
    ("draper_qft_adder", 8),
    ("cdkm_ripple_carry_adder", 8),
    ("qpeexact", 8),
    ("qaoa", 16),
    ("randomcircuit", 12),
)

def parse_sample_args(sample_args: list[str] | None) -> list[tuple[str, int]]:
    if not sample_args:
        return list(DEFAULT_SAMPLE)

    sample: list[tuple[str, int]] = []
    for item in sample_args:
        if ":" not in item:
            raise ValueError(f"Expected name:size, got {item!r}.")
        name, raw_size = item.split(":", 1)
        sample.append((name, int(raw_size)))
    return sample

def run_case(
    benchmark: str,
    circuit_size: int,
    *,
    rss_limit_mb: float,
    timeout_s: float,
    case_timeout_s: float,
    min_available_memory_mb: float,
    max_interaction_width: int,
    max_quimb_width: float,
    max_quimb_log2_tensor_size: float,
    quimb_optimize: str,
    terket_only: bool,
    profile_dir: Path | None,
    temp_dir: Path,
) -> dict[str, str]:
    safe_name = f"{benchmark}_{circuit_size}"
    temp_csv = temp_dir / f"{safe_name}.csv"
    temp_log = temp_dir / f"{safe_name}.log"
    cmd = [
        sys.executable,
        str(CASE_SCRIPT),
        "--sample",
        f"{benchmark}:{circuit_size}",
        "--repeats",
        "1",
        "--min-available-memory-mb",
        str(min_available_memory_mb),
        "--max-interaction-width",
        str(max_interaction_width),
        "--max-quimb-width",
        str(max_quimb_width),
        "--max-quimb-log2-tensor-size",
        str(max_quimb_log2_tensor_size),
        "--quimb-optimize",
        quimb_optimize,
        "--csv",
        str(temp_csv),
    ]
    if terket_only:
        cmd.append("--terket-only")
    if profile_dir is not None:
        cmd.extend(["--profile-dir", str(profile_dir)])

    result = run_guarded_subprocess_csv(
        cmd,
        cwd=REPO_ROOT,
        temp_csv=temp_csv,
        missing_row={
            "benchmark": benchmark,
            "circuit_size": str(circuit_size),
            "status": "runner_no_row",
            "error_type": "",
            "error_message": "",
        },
        rss_limit_mb=rss_limit_mb,
        timeout_s=min(timeout_s, case_timeout_s),
        stdout_path=temp_log,
    )
    row = result.row
    row["runner_status"] = result.runner_status
    row["runner_peak_rss_mb"] = f"{result.peak_rss_mb:.6f}"
    row["runner_wall_time_s"] = f"{result.wall_time_s:.6f}"
    row["runner_log_path"] = str(temp_log)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", nargs="*", default=None, help="Sample in name:size form.")
    parser.add_argument(
        "--rss-limit-mb",
        type=float,
        default=default_guarded_rss_limit_mb(),
        help="Kill case above combined child RSS. Default: min(3072, max(1536, 20%% of RAM)).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=1200.0,
        help="Kill case above wall time. Default: 1200 seconds (20 minutes).",
    )
    parser.add_argument(
        "--case-timeout-s",
        type=float,
        default=1200.0,
        help="Per-case timeout. Kept separate for future tuning.",
    )
    parser.add_argument("--min-available-memory-mb", type=float, default=2048.0)
    parser.add_argument("--max-interaction-width", type=int, default=18)
    parser.add_argument("--max-quimb-width", type=float, default=20.0)
    parser.add_argument("--max-quimb-log2-tensor-size", type=float, default=24.0)
    parser.add_argument("--quimb-optimize", default="auto-hq")
    parser.add_argument("--terket-only", action="store_true", help="Skip all quimb work.")
    parser.add_argument("--profile-dir", type=Path, default=None, help="Optional cProfile output directory.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=RESULTS_ROOT / "mqt_bench_guarded_runner.csv",
        help="Aggregate CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = parse_sample_args(args.sample)
    temp_dir = RESULTS_ROOT / "mqt_guarded_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    if args.profile_dir is not None:
        args.profile_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for benchmark, circuit_size in sample:
        row = run_case(
            benchmark,
            circuit_size,
            rss_limit_mb=args.rss_limit_mb,
            timeout_s=args.timeout_s,
            case_timeout_s=args.case_timeout_s,
            min_available_memory_mb=args.min_available_memory_mb,
            max_interaction_width=args.max_interaction_width,
            max_quimb_width=args.max_quimb_width,
            max_quimb_log2_tensor_size=args.max_quimb_log2_tensor_size,
            quimb_optimize=args.quimb_optimize,
            terket_only=bool(args.terket_only),
            profile_dir=args.profile_dir,
            temp_dir=temp_dir,
        )
        rows.append(row)
        write_rows(rows, args.csv)
        print(
            f"{benchmark}:{circuit_size} case_status={row.get('status')} "
            f"runner_status={row.get('runner_status')} "
            f"peak_rss_mb={row.get('runner_peak_rss_mb')}"
        )

    print(f"Wrote {len(rows)} row(s) to {args.csv}")


if __name__ == "__main__":
    main()
