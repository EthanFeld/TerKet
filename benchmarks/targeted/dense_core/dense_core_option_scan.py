"""Scan dense-core heuristics on restricted MQT QAOA kernels."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
RESULTS_ROOT = REPO_ROOT / "results"
PLAN_EVAL_SCRIPT = REPO_ROOT / "benchmarks" / "targeted" / "dense_core" / "dense_core_plan_eval.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket.benchmarking import default_guarded_rss_limit_mb, run_guarded_subprocess_csv, write_rows

from benchmarks.targeted.dense_core.dense_core_common import (
    DEEP_BUDGETS,
    DEFAULT_BUDGETS,
    DEFAULT_SIZES,
    TARGET_REMAINING_WIDTH,
    builtin_cutset_row,
    extract_qaoa_case,
    heuristic_specs,
    row_to_dict,
    scan_heuristics,
)


@dataclass
class GuardedExactRow:
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
    runner_status: str
    runner_peak_rss_mb: float
    runner_wall_time_s: float

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="*", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--budgets", nargs="*", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--deep-budgets", nargs="*", type=int, default=list(DEEP_BUDGETS))
    parser.add_argument("--shortlist-k", type=int, default=5)
    parser.add_argument("--rss-limit-mb", type=float, default=default_guarded_rss_limit_mb())
    parser.add_argument("--timeout-s", type=float, default=1200.0)
    parser.add_argument("--scan-csv", type=Path, default=RESULTS_ROOT / "qaoa_dense_core_option_scan.csv")
    parser.add_argument("--deep-csv", type=Path, default=RESULTS_ROOT / "qaoa_dense_core_option_deep_scan.csv")
    parser.add_argument("--exact-csv", type=Path, default=RESULTS_ROOT / "qaoa_dense_core_exact_guarded.csv")
    return parser.parse_args()

def shortlist_rows(scan_rows: list[dict[str, object]], shortlist_k: int) -> list[dict[str, object]]:
    by_heuristic: dict[str, list[dict[str, object]]] = {}
    for row in scan_rows:
        if int(row["size"]) != 25:
            continue
        if str(row["category"]) == "baseline":
            continue
        by_heuristic.setdefault(str(row["heuristic"]), []).append(row)

    winners: list[dict[str, object]] = []
    for heuristic, rows in by_heuristic.items():
        best = min(
            rows,
            key=lambda row: (
                int(row["remaining_width"]),
                int(row["generic_penalty"]),
                float(row["estimated_total_work_log2"]),
                int(row["budget"]),
                heuristic,
            ),
        )
        winners.append(best)

    winners.sort(
        key=lambda row: (
            int(row["remaining_width"]),
            int(row["generic_penalty"]),
            float(row["estimated_total_work_log2"]),
            int(row["budget"]),
            str(row["heuristic"]),
        )
    )
    return winners[: int(shortlist_k)]


def run_guarded_exact(
    *,
    size: int,
    heuristic: str,
    budget: int,
    rss_limit_mb: float,
    timeout_s: float,
) -> GuardedExactRow:
    temp_csv = RESULTS_ROOT / "dense_core_tmp" / f"{heuristic}_{size}_{budget}.csv"
    temp_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PLAN_EVAL_SCRIPT),
        "--size",
        str(size),
        "--heuristic",
        heuristic,
        "--budget",
        str(budget),
        "--csv",
        str(temp_csv),
    ]
    result = run_guarded_subprocess_csv(
        cmd,
        cwd=REPO_ROOT,
        temp_csv=temp_csv,
        missing_row={
            "size": str(size),
            "heuristic": heuristic,
            "category": "unknown",
            "budget": str(budget),
            "remaining_backend": "none",
            "remaining_width": "-1",
            "estimated_total_work": "-1",
            "estimated_total_work_log2": "nan",
            "wall_time_s": "",
            "peak_rss_mb": "",
            "abs_value": "",
            "exact_rel_error_vs_full": "",
            "status": "runner_no_row",
            "error_type": "",
            "error_message": "",
        },
        rss_limit_mb=rss_limit_mb,
        timeout_s=timeout_s,
        suppress_output=True,
    )
    row = result.row

    return GuardedExactRow(
        size=int(row["size"]),
        heuristic=str(row["heuristic"]),
        category=str(row["category"]),
        budget=int(row["budget"]),
        remaining_backend=str(row["remaining_backend"]),
        remaining_width=int(row["remaining_width"]),
        estimated_total_work=int(row["estimated_total_work"]),
        estimated_total_work_log2=float(row["estimated_total_work_log2"]),
        wall_time_s=float(row["wall_time_s"]) if row["wall_time_s"] else None,
        peak_rss_mb=float(row["peak_rss_mb"]) if row["peak_rss_mb"] else None,
        abs_value=float(row["abs_value"]) if row["abs_value"] else None,
        exact_rel_error_vs_full=float(row["exact_rel_error_vs_full"]) if row["exact_rel_error_vs_full"] else None,
        status=str(row["status"]),
        error_type=str(row["error_type"]) if row["error_type"] else None,
        error_message=str(row["error_message"]) if row["error_message"] else None,
        runner_status=result.runner_status,
        runner_peak_rss_mb=float(result.peak_rss_mb),
        runner_wall_time_s=float(result.wall_time_s),
    )


def main() -> int:
    args = parse_args()
    specs = heuristic_specs()
    cases = {int(size): extract_qaoa_case(int(size)) for size in args.sizes}

    scan_rows: list[dict[str, object]] = []
    print(f"Scan target width <= {TARGET_REMAINING_WIDTH}.")
    for size, case in cases.items():
        print(f"Scan qaoa:{size} free={case.free_var_count} q2={len(case.q.q2)} width={case.min_fill_width}")
        rows = scan_heuristics(case, specs, args.budgets)
        builtin = builtin_cutset_row(case)
        if builtin is not None:
            rows.append(builtin)
        rows.sort(
            key=lambda row: (
                row.category,
                row.heuristic,
                row.budget,
            )
        )
        for row in rows:
            scan_rows.append(row_to_dict(row))
    write_rows(scan_rows, args.scan_csv)
    print(f"Wrote scan rows to {args.scan_csv}.")

    shortlist = shortlist_rows(scan_rows, args.shortlist_k)
    print("Shortlist from qaoa:25:")
    for row in shortlist:
        print(
            f"  {row['heuristic']} budget={row['budget']} width={row['remaining_width']} "
            f"log2work={float(row['estimated_total_work_log2']):.2f} backend={row['remaining_backend']}"
        )

    deep_rows: list[dict[str, object]] = []
    for row in shortlist:
        heuristic = str(row["heuristic"])
        spec = next(item for item in specs if item.name == heuristic)
        case = cases[25]
        rows = scan_heuristics(case, [spec], args.deep_budgets)
        rows.sort(key=lambda item: item.budget)
        for deep_row in rows:
            deep_rows.append(row_to_dict(deep_row))
    write_rows(deep_rows, args.deep_csv)
    print(f"Wrote deep rows to {args.deep_csv}.")

    exact_jobs: list[tuple[int, str, int]] = []
    for row in shortlist:
        heuristic = str(row["heuristic"])
        best_budget = int(row["budget"])
        exact_jobs.append((25, heuristic, best_budget))
        exact_jobs.append((24, heuristic, best_budget))
    exact_jobs.append((25, "baseline_builtin_cutset", 0))
    exact_jobs.append((24, "baseline_builtin_cutset", 0))

    exact_rows: list[dict[str, object]] = []
    for size, heuristic, budget in exact_jobs:
        print(f"Exact {heuristic} qaoa:{size} budget={budget}")
        result = run_guarded_exact(
            size=size,
            heuristic=heuristic,
            budget=budget,
            rss_limit_mb=args.rss_limit_mb,
            timeout_s=args.timeout_s,
        )
        exact_rows.append(asdict(result))
        status_bits = [result.status, result.runner_status]
        if result.wall_time_s is not None:
            status_bits.append(f"wall={result.wall_time_s:.3f}s")
        if result.peak_rss_mb is not None:
            status_bits.append(f"rss={result.peak_rss_mb:.3f}MB")
        print("  " + " ".join(status_bits))
    write_rows(exact_rows, args.exact_csv)
    print(f"Wrote exact rows to {args.exact_csv}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
