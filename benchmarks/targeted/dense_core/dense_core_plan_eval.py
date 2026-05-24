"""Evaluate one custom cutset plan exactly for a dense q2 core."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket.benchmarking import measure_callable, write_rows
from terket.engine import _scaled_to_complex

from benchmarks.targeted.dense_core.dense_core_common import (
    builtin_cutset_row,
    exact_cutset_total,
    exact_full_total,
    extract_qaoa_case,
    heuristic_specs,
    scan_heuristics,
)


@dataclass
class ExactEvalRow:
    size: int
    heuristic: str
    category: str
    budget: int
    remaining_backend: str
    remaining_width: int
    estimated_total_work: int
    estimated_total_work_log2: float
    wall_time_s: float | None
    peak_rss_mb: float | None
    abs_value: float | None
    exact_rel_error_vs_full: float | None
    status: str
    error_type: str | None
    error_message: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--heuristic", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    case = extract_qaoa_case(args.size)

    if args.heuristic == "baseline_builtin_cutset":
        candidate = builtin_cutset_row(case)
    else:
        specs = heuristic_specs()
        spec = next((item for item in specs if item.name == args.heuristic), None)
        if spec is None:
            raise ValueError(f"Unknown heuristic {args.heuristic!r}.")
        rows = scan_heuristics(case, [spec], [args.budget])
        candidate = rows[0] if rows else None

    if candidate is None:
        row = ExactEvalRow(
            size=args.size,
            heuristic=args.heuristic,
            category="baseline" if args.heuristic == "baseline_builtin_cutset" else "unknown",
            budget=args.budget,
            remaining_backend="none",
            remaining_width=-1,
            estimated_total_work=-1,
            estimated_total_work_log2=float("nan"),
            wall_time_s=None,
            peak_rss_mb=None,
            abs_value=None,
            exact_rel_error_vs_full=None,
            status="no_candidate",
            error_type="RuntimeError",
            error_message="No candidate row.",
        )
        write_rows([row], args.csv)
        return 0

    try:
        wall_time_s, peak_rss_mb, value = measure_callable(
            lambda: exact_cutset_total(case, candidate.cutset_vars),
            1,
        )
        complex_value = _scaled_to_complex(value)
        full_rel_error = None
        if args.size <= 24:
            full_value = _scaled_to_complex(exact_full_total(case))
            full_rel_error = abs(complex_value - full_value) / max(abs(full_value), 1e-300)
        row = ExactEvalRow(
            size=args.size,
            heuristic=args.heuristic,
            category=candidate.category,
            budget=candidate.budget,
            remaining_backend=candidate.remaining_backend,
            remaining_width=candidate.remaining_width,
            estimated_total_work=candidate.estimated_total_work,
            estimated_total_work_log2=candidate.estimated_total_work_log2,
            wall_time_s=float(wall_time_s),
            peak_rss_mb=float(peak_rss_mb),
            abs_value=float(abs(complex_value)),
            exact_rel_error_vs_full=None if full_rel_error is None else float(full_rel_error),
            status="ok",
            error_type=None,
            error_message=None,
        )
    except Exception as exc:
        row = ExactEvalRow(
            size=args.size,
            heuristic=args.heuristic,
            category=candidate.category,
            budget=candidate.budget,
            remaining_backend=candidate.remaining_backend,
            remaining_width=candidate.remaining_width,
            estimated_total_work=candidate.estimated_total_work,
            estimated_total_work_log2=candidate.estimated_total_work_log2,
            wall_time_s=None,
            peak_rss_mb=None,
            abs_value=None,
            exact_rel_error_vs_full=None,
            status="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    write_rows([row], args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
