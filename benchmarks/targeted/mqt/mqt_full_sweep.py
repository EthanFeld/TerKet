"""Sequential TerKet-only MQT Bench frontier sweep with hard guards."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import statistics
import sys
from typing import Any

from mqt.bench import get_benchmark_alg
from mqt.bench.benchmarks._registry import benchmark_catalog, benchmark_names


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.targeted.mqt.mqt_bench_guarded_runner import run_case, write_rows


CANDIDATE_SIZES: tuple[int, ...] = (
    2, 3, 4, 5, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23, 24,
    25, 26, 27, 28, 32, 34, 40, 42, 48, 52, 58, 64, 74, 80, 96, 112, 128,
)
HARD_FAIL_RUNNER = {"killed_rss_guard", "killed_timeout_guard"}
HARD_FAIL_STATUS = {"error", "guard_skip_low_memory"}
NEAR_CAP_TIME_S = 900.0
NEAR_CAP_RSS_MB = 3800.0
CSV_FIELDS = (
    "benchmark", "description", "circuit_size", "attempt_kind", "status", "runner_status",
    "runner_wall_time_s", "runner_peak_rss_mb", "n_qubits", "gate_count", "depth",
    "two_qubit_gate_count", "restricted_free_vars", "restricted_q2_terms", "restricted_q3_terms",
    "restricted_min_fill_width", "terket_phase3_backend", "terket_cubic_obstruction",
    "terket_gauss_obstruction", "terket_wall_time_s", "terket_peak_rss_mb", "frontier_note",
    "error_type", "error_message", "runner_log_path",
)
_VALID_SIZE_CACHE: dict[str, tuple[int, ...]] = {}


@dataclass(frozen=True)
class FamilyFrontier:
    benchmark: str
    description: str
    valid_candidate_sizes: tuple[int, ...]
    run_sizes: tuple[int, ...]
    success_sizes: tuple[int, ...]
    inferred_simulable_sizes: tuple[int, ...]
    predicted_skip_sizes: tuple[int, ...]
    max_success_size: int | None
    frontier_size: int | None
    frontier_status: str
    frontier_note: str
    max_success_qubits: int | None
    max_success_gate_count: int | None
    max_success_wall_time_s: float | None
    max_success_peak_rss_mb: float | None


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(rows: list[dict[str, Any]], csv_path: Path) -> None:
    ordered_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized = {field: row.get(field, "") for field in CSV_FIELDS}
        for key, value in row.items():
            if key not in normalized:
                normalized[key] = value
        ordered_rows.append(normalized)
    write_rows(ordered_rows, csv_path)


def _generation_ok(benchmark: str, circuit_size: int) -> bool:
    try:
        get_benchmark_alg(
            benchmark,
            circuit_size=circuit_size,
            random_parameters=False,
        )
        return True
    except Exception:
        return False


def _valid_candidate_sizes(benchmark: str) -> tuple[int, ...]:
    cached = _VALID_SIZE_CACHE.get(benchmark)
    if cached is not None:
        return cached
    valid = tuple(size for size in CANDIDATE_SIZES if _generation_ok(benchmark, size))
    _VALID_SIZE_CACHE[benchmark] = valid
    return valid


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _int_or_none(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _step_after_success(row: dict[str, Any]) -> int:
    wall = _float_or_none(str(row.get("terket_wall_time_s", "")))
    peak = _float_or_none(str(row.get("runner_peak_rss_mb", "")))
    if wall is None or peak is None:
        return 1
    if wall < 0.25 and peak < 256.0:
        return 4
    if wall < 2.0 and peak < 512.0:
        return 3
    if wall < 30.0 and peak < 1024.0:
        return 2
    return 1


def _synthetic_skip_row(
    benchmark: str,
    description: str,
    circuit_size: int,
    *,
    frontier_note: str,
) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "description": description,
        "circuit_size": circuit_size,
        "attempt_kind": "predicted_skip",
        "status": "predicted_skip_after_frontier",
        "runner_status": "",
        "frontier_note": frontier_note,
    }


def _append_future_synthetic_skips(
    benchmark: str,
    description: str,
    valid_sizes: tuple[int, ...],
    pending_idx: int,
    existing: dict[int, dict[str, Any]],
    case_rows: list[dict[str, Any]],
    *,
    frontier_note: str,
) -> list[dict[str, Any]]:
    for later_size in valid_sizes[pending_idx + 1 :]:
        if later_size in existing:
            continue
        synthetic = _synthetic_skip_row(
            benchmark,
            description,
            later_size,
            frontier_note=frontier_note,
        )
        case_rows.append(synthetic)
        existing[later_size] = synthetic
    return case_rows


def _hard_fail(row: dict[str, Any]) -> bool:
    return (
        str(row.get("runner_status", "")) in HARD_FAIL_RUNNER
        or str(row.get("status", "")) in HARD_FAIL_STATUS
    )


def _near_cap_success(row: dict[str, Any]) -> bool:
    if str(row.get("status", "")) != "ok":
        return False
    wall = _float_or_none(str(row.get("terket_wall_time_s", "")))
    peak = _float_or_none(str(row.get("runner_peak_rss_mb", "")))
    return (
        (wall is not None and wall >= NEAR_CAP_TIME_S)
        or (peak is not None and peak >= NEAR_CAP_RSS_MB)
    )


def _run_family(
    benchmark: str,
    description: str,
    *,
    case_rows: list[dict[str, Any]],
    temp_dir: Path,
    rss_limit_mb: float,
    timeout_s: float,
    case_timeout_s: float,
    min_available_memory_mb: float,
) -> list[dict[str, Any]]:
    existing = {
        int(str(row["circuit_size"])): row
        for row in case_rows
        if row.get("benchmark") == benchmark
    }
    valid_sizes = _valid_candidate_sizes(benchmark)
    if not valid_sizes:
        row = {
            "benchmark": benchmark,
            "description": description,
            "circuit_size": "",
            "attempt_kind": "meta",
            "status": "no_valid_sizes_in_candidate_universe",
            "runner_status": "",
            "frontier_note": "No valid circuit_size found in configured candidate set.",
        }
        if "" not in {str(row.get("circuit_size", "")) for row in case_rows if row.get("benchmark") == benchmark}:
            case_rows.append(row)
        return case_rows

    pending_idx = 0
    lower_success_idx: int | None = None
    while pending_idx < len(valid_sizes):
        size = valid_sizes[pending_idx]
        if size in existing:
            row = existing[size]
        else:
            row = run_case(
                benchmark,
                size,
                rss_limit_mb=rss_limit_mb,
                timeout_s=timeout_s,
                case_timeout_s=case_timeout_s,
                min_available_memory_mb=min_available_memory_mb,
                max_interaction_width=1_000_000,
                max_quimb_width=1_000_000.0,
                max_quimb_log2_tensor_size=1_000_000.0,
                quimb_optimize="auto-hq",
                terket_only=True,
                profile_dir=None,
                temp_dir=temp_dir,
            )
            row = dict(row)
            row["description"] = description
            row["attempt_kind"] = "run"
            row["frontier_note"] = ""
            case_rows.append(row)
            existing[size] = row

        if _hard_fail(row):
            if lower_success_idx is not None and pending_idx - lower_success_idx > 1:
                pending_idx = (lower_success_idx + pending_idx) // 2
                continue
            return _append_future_synthetic_skips(
                benchmark,
                description,
                valid_sizes,
                pending_idx,
                existing,
                case_rows,
                frontier_note=f"Hard fail at size {size}; larger valid sizes skipped.",
            )

        if _near_cap_success(row):
            lower_success_idx = pending_idx
            return _append_future_synthetic_skips(
                benchmark,
                description,
                valid_sizes,
                pending_idx,
                existing,
                case_rows,
                frontier_note=f"Near cap at size {size}; larger valid sizes skipped.",
            )

        if str(row.get("status", "")) != "ok":
            if lower_success_idx is not None and pending_idx - lower_success_idx > 1:
                pending_idx = (lower_success_idx + pending_idx) // 2
                continue
            pending_idx += 1
            continue

        lower_success_idx = pending_idx
        step = _step_after_success(row)
        next_idx = min(len(valid_sizes) - 1, pending_idx + step)
        if next_idx <= pending_idx:
            break
        pending_idx = next_idx

    return case_rows


def _family_frontier(
    benchmark: str,
    description: str,
    rows: list[dict[str, Any]],
) -> FamilyFrontier:
    family_rows = [
        row for row in rows
        if row.get("benchmark") == benchmark and row.get("circuit_size", "") != ""
    ]
    valid_sizes = _valid_candidate_sizes(benchmark)
    run_rows = [row for row in family_rows if row.get("attempt_kind") == "run"]
    run_sizes = tuple(sorted(int(str(row["circuit_size"])) for row in run_rows))
    success_rows = [row for row in run_rows if row.get("status") == "ok"]
    success_sizes = tuple(sorted(int(str(row["circuit_size"])) for row in success_rows))
    predicted_skip_sizes = tuple(
        sorted(
            int(str(row["circuit_size"]))
            for row in family_rows
            if row.get("attempt_kind") == "predicted_skip"
        )
    )
    max_success_row = None
    if success_rows:
        max_success_row = max(success_rows, key=lambda row: int(str(row["circuit_size"])))
    inferred_simulable_sizes = tuple(
        size
        for size in valid_sizes
        if max_success_row is not None and size <= int(str(max_success_row["circuit_size"]))
    )

    frontier_status = "unknown"
    frontier_note = ""
    frontier_size = None
    for row in sorted(run_rows, key=lambda item: int(str(item["circuit_size"]))):
        if str(row.get("status")) != "ok":
            frontier_status = str(row.get("status"))
            frontier_note = str(row.get("frontier_note", "")) or str(row.get("runner_status", ""))
            frontier_size = int(str(row["circuit_size"]))
            break
        if _hard_fail(row):
            frontier_status = str(row.get("runner_status"))
            frontier_note = str(row.get("frontier_note", "")) or str(row.get("runner_status", ""))
            frontier_size = int(str(row["circuit_size"]))
            break
    if frontier_status == "unknown":
        if predicted_skip_sizes:
            frontier_status = "predicted_skip_after_frontier"
            frontier_note = next(
                (
                    str(row.get("frontier_note", ""))
                    for row in family_rows
                    if row.get("attempt_kind") == "predicted_skip"
                ),
                "",
            )
            frontier_size = predicted_skip_sizes[0]
        elif success_rows:
            frontier_status = "solved_all_candidates"
            frontier_note = "Solved every valid size in candidate universe."
        else:
            frontier_status = "no_success"
            frontier_note = "No successful run."

    return FamilyFrontier(
        benchmark=benchmark,
        description=description,
        valid_candidate_sizes=valid_sizes,
        run_sizes=run_sizes,
        success_sizes=success_sizes,
        inferred_simulable_sizes=inferred_simulable_sizes,
        predicted_skip_sizes=predicted_skip_sizes,
        max_success_size=None if max_success_row is None else int(str(max_success_row["circuit_size"])),
        frontier_size=frontier_size,
        frontier_status=frontier_status,
        frontier_note=frontier_note,
        max_success_qubits=None if max_success_row is None else _int_or_none(str(max_success_row.get("n_qubits", ""))),
        max_success_gate_count=None if max_success_row is None else _int_or_none(str(max_success_row.get("gate_count", ""))),
        max_success_wall_time_s=None if max_success_row is None else _float_or_none(str(max_success_row.get("terket_wall_time_s", ""))),
        max_success_peak_rss_mb=None if max_success_row is None else _float_or_none(str(max_success_row.get("runner_peak_rss_mb", ""))),
    )


def _frontier_rows(frontiers: list[FamilyFrontier]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frontier in frontiers:
        rows.append(
            {
                "benchmark": frontier.benchmark,
                "description": frontier.description,
                "valid_candidate_sizes": " ".join(str(size) for size in frontier.valid_candidate_sizes),
                "run_sizes": " ".join(str(size) for size in frontier.run_sizes),
                "success_sizes": " ".join(str(size) for size in frontier.success_sizes),
                "inferred_simulable_sizes": " ".join(str(size) for size in frontier.inferred_simulable_sizes),
                "predicted_skip_sizes": " ".join(str(size) for size in frontier.predicted_skip_sizes),
                "max_success_size": frontier.max_success_size,
                "frontier_size": frontier.frontier_size,
                "frontier_status": frontier.frontier_status,
                "frontier_note": frontier.frontier_note,
                "max_success_qubits": frontier.max_success_qubits,
                "max_success_gate_count": frontier.max_success_gate_count,
                "max_success_wall_time_s": frontier.max_success_wall_time_s,
                "max_success_peak_rss_mb": frontier.max_success_peak_rss_mb,
            }
        )
    return rows


def _report_text(case_rows: list[dict[str, Any]], frontiers: list[FamilyFrontier]) -> str:
    valid_rows = [row for row in case_rows if row.get("circuit_size", "") != ""]
    solved_rows = [row for row in valid_rows if row.get("status") == "ok"]
    run_rows = [row for row in valid_rows if row.get("attempt_kind") == "run"]
    predicted_rows = [row for row in valid_rows if row.get("attempt_kind") == "predicted_skip"]
    total_valid_candidates = sum(len(frontier.valid_candidate_sizes) for frontier in frontiers)
    total_simulable_candidates = sum(len(frontier.inferred_simulable_sizes) for frontier in frontiers)
    attempted_success_rate = (
        total_simulable_candidates / total_valid_candidates
        if total_valid_candidates
        else math.nan
    )
    family_success_rate = (
        sum(1 for frontier in frontiers if frontier.max_success_size is not None) / max(1, len(frontiers))
    )

    impressive = sorted(
        solved_rows,
        key=lambda row: (
            _int_or_none(str(row.get("n_qubits", ""))) or -1,
            _int_or_none(str(row.get("gate_count", ""))) or -1,
            _float_or_none(str(row.get("terket_wall_time_s", ""))) or -1.0,
        ),
        reverse=True,
    )[:10]
    future_targets = []
    for frontier in frontiers:
        if frontier.max_success_size is None:
            continue
        if frontier.frontier_status not in {"predicted_skip_after_frontier", "runner_no_row", "error"} and not frontier.predicted_skip_sizes:
            continue
        future_targets.append(frontier)
    future_targets.sort(
        key=lambda item: (
            item.max_success_size or -1,
            -(item.max_success_wall_time_s or 0.0),
        ),
        reverse=True,
    )
    future_targets = future_targets[:10]

    lines = [
        "# MQT Bench TerKet Full Sweep",
        "",
        f"- Families in registry: {len(frontiers)}",
        f"- Valid candidate instances in configured universe: {total_valid_candidates}",
        f"- Actual runs: {len(run_rows)}",
        f"- Predicted skips after family frontier: {len(predicted_rows)}",
        f"- Actual solved runs: {len(solved_rows)}",
        f"- Inferred simulable instances: {total_simulable_candidates}",
        f"- Solved-instance percentage in configured universe: {attempted_success_rate:.2%}" if not math.isnan(attempted_success_rate) else "- Solved-instance percentage in configured universe: n/a",
        f"- Families with at least one success: {family_success_rate:.2%}",
        "",
        "## Impressive Cases",
        "",
    ]
    for row in impressive:
        lines.append(
            f"- {row['benchmark']}:{row['circuit_size']} | qubits={row.get('n_qubits','')} "
            f"| gates={row.get('gate_count','')} | wall={row.get('terket_wall_time_s','')} s "
            f"| rss={row.get('runner_peak_rss_mb','')} MB | backend={row.get('terket_phase3_backend','')}"
        )

    lines.extend(["", "## Future Targets", ""])
    for frontier in future_targets:
        lines.append(
            f"- {frontier.benchmark} | max_success={frontier.max_success_size} "
            f"| next_frontier={frontier.frontier_size} | status={frontier.frontier_status} "
            f"| note={frontier.frontier_note or 'n/a'}"
        )

    width_rows = [
        row for row in solved_rows
        if _int_or_none(str(row.get("restricted_min_fill_width", ""))) is not None
    ]
    if width_rows:
        median_width = statistics.median(
            _int_or_none(str(row.get("restricted_min_fill_width", ""))) for row in width_rows
        )
        lines.extend(
            [
                "",
                "## Hardness Snapshot",
                "",
                f"- Median solved restricted min-fill width: {median_width}",
                f"- Max solved restricted min-fill width: {max(_int_or_none(str(row.get('restricted_min_fill_width', ''))) for row in width_rows)}",
                f"- Max solved restricted q3 terms: {max(_int_or_none(str(row.get('restricted_q3_terms', ''))) or 0 for row in solved_rows)}",
            ]
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Candidate universe finite, not all natural numbers. Sizes tested from configured ladder up to 128 plus special MQT fixed sizes.",
            "- Larger sizes after a hard family frontier are marked predicted skips, not actual runs.",
            "- Every actual run used one child process at a time with 20 min / 4 GiB guards.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="*", default=None, help="Optional subset of benchmark names.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing case CSV rows.")
    parser.add_argument("--rss-limit-mb", type=float, default=4096.0)
    parser.add_argument("--timeout-s", type=float, default=1200.0)
    parser.add_argument("--case-timeout-s", type=float, default=1200.0)
    parser.add_argument("--min-available-memory-mb", type=float, default=1024.0)
    parser.add_argument("--cases-csv", type=Path, default=RESULTS_ROOT / "mqt_full_sweep_cases.csv")
    parser.add_argument("--summary-csv", type=Path, default=RESULTS_ROOT / "mqt_full_sweep_summary.csv")
    parser.add_argument("--report-md", type=Path, default=RESULTS_ROOT / "mqt_full_sweep_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    families = sorted(args.families or benchmark_names())
    catalog = benchmark_catalog()
    temp_dir = RESULTS_ROOT / "mqt_guarded_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    case_rows: list[dict[str, Any]] = _read_rows(args.cases_csv) if args.resume else []
    if not args.resume:
        case_rows = [row for row in case_rows if row.get("benchmark") not in families]

    for benchmark in families:
        description = catalog.get(benchmark, "")
        print(f"family_start benchmark={benchmark}", flush=True)
        case_rows = _run_family(
            benchmark,
            description,
            case_rows=case_rows,
            temp_dir=temp_dir,
            rss_limit_mb=args.rss_limit_mb,
            timeout_s=args.timeout_s,
            case_timeout_s=args.case_timeout_s,
            min_available_memory_mb=args.min_available_memory_mb,
        )
        _write_rows(case_rows, args.cases_csv)
        frontier = _family_frontier(benchmark, description, case_rows)
        print(
            f"family_done benchmark={benchmark} max_success={frontier.max_success_size} "
            f"frontier_status={frontier.frontier_status}",
            flush=True,
        )

    frontiers = [_family_frontier(benchmark, catalog.get(benchmark, ""), case_rows) for benchmark in families]
    write_rows(_frontier_rows(frontiers), args.summary_csv)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(_report_text(case_rows, frontiers), encoding="utf-8")
    print(f"Wrote {args.cases_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.report_md}")


if __name__ == "__main__":
    main()
