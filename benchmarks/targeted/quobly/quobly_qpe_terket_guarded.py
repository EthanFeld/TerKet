"""Guarded sweep for Quobly-style Heisenberg Trotter-QPE in TerKet."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
RESULTS_ROOT = REPO_ROOT / "results"
CASE_SCRIPT = Path(__file__).resolve().with_name("quobly_qpe_terket_case.py")

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket.benchmarking import default_guarded_rss_limit_mb, run_guarded_subprocess_csv, write_rows


DEFAULT_CASES: tuple[tuple[int, int], ...] = (
    (2, 3),
    (4, 3),
    (6, 3),
    (8, 3),
    (4, 5),
    (6, 5),
    (8, 5),
    (10, 5),
    (4, 7),
    (6, 7),
    (8, 7),
)

def parse_cases(case_args: list[str] | None) -> list[tuple[int, int]]:
    if not case_args:
        return list(DEFAULT_CASES)
    out: list[tuple[int, int]] = []
    for item in case_args:
        if ":" not in item:
            raise ValueError(f"Expected n_data:n_phase, got {item!r}.")
        left, right = item.split(":", 1)
        out.append((int(left), int(right)))
    return out

def run_case(
    n_data: int,
    n_phase: int,
    *,
    n_steps0: int,
    trotter_order: int,
    rss_limit_mb: float,
    timeout_s: float,
    temp_dir: Path,
) -> dict[str, str]:
    temp_csv = temp_dir / f"quobly_qpe_{n_data}_{n_phase}.csv"
    cmd = [
        sys.executable,
        str(CASE_SCRIPT),
        "--n-data",
        str(n_data),
        "--n-phase",
        str(n_phase),
        "--n-steps0",
        str(n_steps0),
        "--trotter-order",
        str(trotter_order),
        "--csv",
        str(temp_csv),
    ]
    result = run_guarded_subprocess_csv(
        cmd,
        cwd=REPO_ROOT,
        temp_csv=temp_csv,
        missing_row={
            "n_data": str(n_data),
            "n_phase": str(n_phase),
            "status": "runner_no_row",
        },
        rss_limit_mb=rss_limit_mb,
        timeout_s=timeout_s,
    )
    row = result.row
    row["runner_status"] = result.runner_status
    row["runner_peak_rss_mb"] = f"{result.peak_rss_mb:.6f}"
    row["runner_wall_time_s"] = f"{result.wall_time_s:.6f}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", default=None, help="Cases in n_data:n_phase form.")
    parser.add_argument("--n-steps0", type=int, default=4)
    parser.add_argument("--trotter-order", type=int, default=2)
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
        "--csv",
        type=Path,
        default=RESULTS_ROOT / "quobly_qpe_terket_guarded.csv",
    )
    args = parser.parse_args()

    temp_dir = RESULTS_ROOT / "quobly_qpe_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for n_data, n_phase in parse_cases(args.cases):
        row = run_case(
            n_data,
            n_phase,
            n_steps0=args.n_steps0,
            trotter_order=args.trotter_order,
            rss_limit_mb=args.rss_limit_mb,
            timeout_s=args.timeout_s,
            temp_dir=temp_dir,
        )
        rows.append(row)
        write_rows(rows, args.csv)
        print(
            f"n_data={n_data} n_phase={n_phase} case_status={row.get('status')} "
            f"runner_status={row.get('runner_status')} peak_rss_mb={row.get('runner_peak_rss_mb')}"
        )

    print(f"Wrote {len(rows)} row(s) to {args.csv}")


if __name__ == "__main__":
    main()
