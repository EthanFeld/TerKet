"""Compare TerKet bloat-refactor benchmark CSVs against a baseline.

Exit code is nonzero when a configured performance gate fails.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import statistics
import sys


TIME_FIELDS = ("terket_wall_time_s", "wall_time_s")
RSS_FIELDS = ("terket_peak_rss_mb", "peak_rss_mb")
BACKEND_FIELDS = ("terket_phase3_backend", "phase3_backend")
STRICT_FIELDS = (
    "cost_model_r",
    "cubic_obstruction",
    "gauss_obstruction",
    "target_log2_abs",
    "wrong_abs",
    "abs_error",
    "relative_error",
)


@dataclass(frozen=True)
class RowPair:
    key: str
    baseline: dict[str, str]
    candidate: dict[str, str]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_key(row: dict[str, str], index: int) -> str:
    for field in ("case", "name", "benchmark"):
        value = row.get(field)
        if value:
            return str(value)
    return f"row:{index}"


def _index_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for idx, row in enumerate(rows):
        key = _row_key(row, idx)
        if key in indexed:
            key = f"{key}#{idx}"
        indexed[key] = row
    return indexed


def _pick_float(row: dict[str, str], fields: tuple[str, ...]) -> float | None:
    for field in fields:
        value = row.get(field)
        if value in (None, ""):
            continue
        try:
            parsed = float(value)
        except ValueError:
            continue
        if math.isfinite(parsed):
            return parsed
    return None


def _pick_text(row: dict[str, str], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    return ""


def _float_equal(left: str | None, right: str | None, *, abs_tol: float) -> bool:
    if left in (None, "") and right in (None, ""):
        return True
    if left in (None, "") or right in (None, ""):
        return False
    try:
        lval = float(left)
        rval = float(right)
    except ValueError:
        return left == right
    return math.isclose(lval, rval, rel_tol=0.0, abs_tol=abs_tol)


def _pair_rows(baseline: Path, candidate: Path) -> tuple[list[RowPair], list[str]]:
    base_index = _index_rows(_read_rows(baseline))
    cand_index = _index_rows(_read_rows(candidate))
    messages: list[str] = []
    for key in sorted(set(base_index) - set(cand_index)):
        messages.append(f"missing candidate row: {key}")
    for key in sorted(set(cand_index) - set(base_index)):
        messages.append(f"extra candidate row: {key}")
    pairs = [
        RowPair(key, base_index[key], cand_index[key])
        for key in sorted(set(base_index) & set(cand_index))
    ]
    return pairs, messages


def compare(
    baseline: Path,
    candidate: Path,
    *,
    median_time_gate: float,
    worst_time_gate: float,
    rss_gate: float,
    strict_abs_tol: float,
    allow_backend_change: bool,
) -> tuple[bool, list[str]]:
    pairs, messages = _pair_rows(baseline, candidate)
    ok = not messages
    time_ratios: list[float] = []
    rss_ratios: list[float] = []

    for pair in pairs:
        base_time = _pick_float(pair.baseline, TIME_FIELDS)
        cand_time = _pick_float(pair.candidate, TIME_FIELDS)
        if base_time is not None and cand_time is not None and base_time > 0.0:
            ratio = cand_time / base_time
            time_ratios.append(ratio)
            if ratio > worst_time_gate:
                ok = False
                messages.append(
                    f"{pair.key}: time ratio {ratio:.4f} > worst gate {worst_time_gate:.4f}"
                )

        base_rss = _pick_float(pair.baseline, RSS_FIELDS)
        cand_rss = _pick_float(pair.candidate, RSS_FIELDS)
        if base_rss is not None and cand_rss is not None and base_rss > 0.0:
            ratio = cand_rss / base_rss
            rss_ratios.append(ratio)
            if ratio > rss_gate:
                ok = False
                messages.append(f"{pair.key}: RSS ratio {ratio:.4f} > gate {rss_gate:.4f}")

        if not allow_backend_change:
            base_backend = _pick_text(pair.baseline, BACKEND_FIELDS)
            cand_backend = _pick_text(pair.candidate, BACKEND_FIELDS)
            if base_backend != cand_backend:
                ok = False
                messages.append(
                    f"{pair.key}: backend changed {base_backend!r} -> {cand_backend!r}"
                )

        for field in STRICT_FIELDS:
            if field in pair.baseline or field in pair.candidate:
                if not _float_equal(
                    pair.baseline.get(field),
                    pair.candidate.get(field),
                    abs_tol=strict_abs_tol,
                ):
                    ok = False
                    messages.append(
                        f"{pair.key}: field {field} changed "
                        f"{pair.baseline.get(field)!r} -> {pair.candidate.get(field)!r}"
                    )

    if time_ratios:
        median_ratio = statistics.median(time_ratios)
        messages.append(f"median_time_ratio={median_ratio:.4f}")
        messages.append(f"max_time_ratio={max(time_ratios):.4f}")
        if median_ratio > median_time_gate:
            ok = False
            messages.append(
                f"median time ratio {median_ratio:.4f} > gate {median_time_gate:.4f}"
            )
    else:
        messages.append("median_time_ratio=NA")

    if rss_ratios:
        messages.append(f"max_rss_ratio={max(rss_ratios):.4f}")
    else:
        messages.append("max_rss_ratio=NA")

    messages.append(f"matched_rows={len(pairs)}")
    return ok, messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_csv", type=Path)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--median-time-gate", type=float, default=1.05)
    parser.add_argument("--worst-time-gate", type=float, default=1.15)
    parser.add_argument("--rss-gate", type=float, default=1.10)
    parser.add_argument("--strict-abs-tol", type=float, default=1e-9)
    parser.add_argument("--allow-backend-change", action="store_true")
    args = parser.parse_args(argv)

    ok, messages = compare(
        args.baseline_csv,
        args.candidate_csv,
        median_time_gate=args.median_time_gate,
        worst_time_gate=args.worst_time_gate,
        rss_gate=args.rss_gate,
        strict_abs_tol=args.strict_abs_tol,
        allow_backend_change=args.allow_backend_change,
    )
    print("PASS" if ok else "FAIL")
    for message in messages:
        print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
